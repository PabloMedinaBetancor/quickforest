"""Backends `codegen`, `codegen-block` and `codegen-pgo`: per-tree code generation. Every tree of a FIXED model becomes a Rust function of nested `if`/`else` with the
thresholds and leaf values as constants (the feature index is a constant offset, so there is no node load at all); rustc/LLVM turn
them into machine code and the result is one `.so` that holds the model. The sums, the transform and the averaging are the same
float32 operations in the same order as xgboost's (rust/src/finish.rs is copied into the generated crate).
    codegen        row at a time: for each row, all trees
    codegen-block  row blocking: for each tree, a block of rows (the tree's code stays hot in the instruction cache)
    codegen-pgo    codegen-block built with profile-guided optimisation (needs `options["pgo_rows"]`: an .npy of representative rows;
                   rustc's own -Cprofile-generate/-Cprofile-use and llvm-profdata from the llvm-tools component)
Needs rustc at COMPILE time (RUST_DIR=<folder with rustup/> or rustc on PATH); running the `.so` needs numpy only. The model is split in
chunks of ~`chunk_nodes` nodes that are compiled as separate objects in parallel (bounded memory), then linked (`unit = "chunks"`, QF 1.0's way), or,
with `unit = "single"` (QF 1.1), the chunks become MODULES of ONE crate with the driver and rustc builds, optimises (fat LTO included) and links all of it as one
unit, the way cargo builds a crate; that is one big rustc process, so the RAM plan falls back to chunks (one line) when it does not fit.
The default build (QF 1.1: `DEFAULT_COMPILE_PRESET`, the preset "1.1") is opt-level 3, fat link-time optimisation, 16 codegen units (all threads) and `-C llvm-args=-inline-threshold=3000`,
with the profile-guided pass (when `best` has rows) built in exactly the same shape. The settings are `options["lto"]` (off | fat), `options["codegen_units"]`,
`options["inline_threshold"]` (None = rustc's own) and `options["pgo"]` (on | off | None = on when rows are given); `options["compile_preset"] = "1.0"` is
exactly QF 1.0's build (the "fast" shape: no LTO, 16 codegen units a chunk, rustc's inlining), and `options["shape"]` ("fast", the same build) names it. The instrumented build and the final one get the same flags, only `profile-generate` against `profile-use` differs. Before it builds, the peak
memory of the rustc jobs is estimated against the RAM available (higher under fat LTO and a high inline threshold) and the jobs (then the units, then the chunk size) are lowered to fit: the build never swaps. The code is built for a PORTABLE CPU by default (x86-64-v2; `generic` on arm: the library runs
wherever the architecture's baseline does) and, as an opt-in, for the machine it is built on (`options["target_cpu"] = "native"`, which then
runs only on that CPU type); `options["target_cpu"]` may name any other CPU rustc knows. The CPU features of the build are written INTO the `.so` and
`CodegenForest` refuses (one line) to load a library whose features the current CPU lacks, before the first instruction of it can run."""

from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
from evaluator import NotCopyable, argmax_outputs, float32_rows, margin_array, output_dtype, resolve_threads, split_outputs
from qf_common import AVERAGE, FEATURES, OBJECTIVE, Forest, Tree, check_vector, replace_file

HERE = Path(__file__).resolve().parent
FINISH_RS = HERE.parent / "rust" / "src" / "finish.rs"
ZERO = "1e-35f32"  # LightGBM kZeroThreshold as a float32
SHAPES = {
    "fast": 16,
}  # QF 1.0's build by name: opt 3, no LTO, 16 codegen units. One codegen unit is never a default anywhere: only `--codegen-units 1` given by hand asks it
REMOVED_SHAPES = {"top": "it was one codegen unit and no LTO"}  # QF 1.0's second shape, dropped (31/10): an old caller gets a clear error

# ---- the compile settings (QF 1.1) -------------------------------------------------------------------------------------------------------------------
# One build = lto + codegen_units + inline_threshold (+ pgo, which is on whenever the backend profiles). A PRESET is a whole build; `shape` names one of the
# two QF 1.0 builds; the single options override what the preset (or shape) says. The default preset is the next line: set it to "1.0" and the library builds
# exactly as QF 1.0 did. The default may change after further measurement.
DEFAULT_COMPILE_PRESET = "1.1"
# How the generated forest is put together (QF 1.1 "one unit"): "chunks" = every chunk its own crate, joined as objects (QF 1.0; link-time optimisation cannot
# cross chunks) or "single" = the whole forest and the driver ONE crate (the chunks are modules), so fat LTO and the inliner see all of it. The measurement that
# decides the default has not been made yet; the next line is the only one to change to make preset 1.1 the one-crate build.
# Preset 1.0 and the two `shape` builds are chunks (they are QF 1.0's builds) unless `unit` single is given: `--unit single` overrides the preset's unit like any other setting.
DEFAULT_COMPILE_UNIT = "chunks"
UNIT_MODES = ("single", "chunks")
LTO_MODES = ("off", "fat")
COMPILE_PRESETS = {
    "1.1": {
        "lto": "fat",
        "codegen_units": SHAPES["fast"],
        "inline_threshold": 3000,
        "pgo": "auto",
        "cspgo": "auto",
        "unit": DEFAULT_COMPILE_UNIT,
    },  # fat LTO + 16 units + inline 3000 (+ PGO of the same shape); chunks or one crate as DEFAULT_COMPILE_UNIT says
    "1.0": {
        "lto": "off",
        "codegen_units": SHAPES["fast"],
        "inline_threshold": None,
        "pgo": "auto",
        "cspgo": "auto",
        "unit": "chunks",
    },  # QF 1.0, bit for bit the same flags
}
SHAPE_BUILDS = {
    "fast": {"lto": "off", "codegen_units": SHAPES["fast"], "inline_threshold": None, "unit": "chunks"},
}
DEFAULT_COMPILE = COMPILE_PRESETS[DEFAULT_COMPILE_PRESET]
COMPILE_KEYS = (
    "compile_preset",
    "lto",
    "codegen_units",
    "inline_threshold",
    "pgo",
    "cspgo",
    "shape",
    "unit",
)  # the options that choose how generated code is built
PGO_VALUES = ("on", "off")
CSPGO_VALUES = (
    "on",
    "off",
)  # CS-PGO (context-sensitive PGO, the linker route) is a second profile pass on top of PGO: on by default where it can be built
# Vector trees: the table of leaf rows is a `static` of generated code, which rustc holds in memory while it builds; above this many values the
# build is refused and the flat format (which maps the table from the file) is named instead.
CODEGEN_MAX_VECTOR_VALUES = 50_000_000
# a vector tree weighs `nodes + leaves * groups / VECTOR_LEAVES_PER_NODE` node-equivalents in the chunk and RAM plans (factor measured on rustc, see LIMITS)
VECTOR_LEAVES_PER_NODE = 8


class ToolchainMissing(FileNotFoundError):
    """rustc (or llvm-profdata) is not installed: a refusal with the way out in the message, not a bug (`quickforest compile` exits 3)."""


class BuildFailed(RuntimeError, ValueError):
    """rustc, the linker or llvm-profdata failed on generated code. Still a RuntimeError (it is not the user's input), but also a ValueError so
    that every command reports it as one line and exit 3 like any other refusal."""


class ProfileUnreliable(BuildFailed):
    """The rows given for the profile are too few, or two halves of them give different profiles: a build guided by it would be tuned for
    something other than the workload. `best` builds `codegen-block` instead (compile_model); `--backend codegen-pgo` fails with this line."""


class NotEnoughRam(BuildFailed):
    """Not even one rustc job on a small chunk fits in the RAM that is available: the build is refused instead of swapping."""


class TargetCpuMismatch(ValueError):
    """A library built for CPU features this machine lacks: loading it would end in an illegal instruction (SIGILL) on plausible input."""


# --- the CPU the code is built for ------------------------------------------------------------------------------------------------------
ARCH = platform.machine().lower()
CPUINFO = Path("/proc/cpuinfo")  # where this machine's CPU features are read (a test points it at a made-up CPU)
TARGET_MARK = "QFTARGET1"  # the record the build writes into the library: QFTARGET1|<cpu>|<arch>|<feature,feature,...>|
# Rust target feature -> the name Linux prints in /proc/cpuinfo, for the features whose absence can be told from there. A feature that is
# not listed (a new one whose name an older kernel may not print, an architecture-version marker) is not checked: refusing a library on the
# very machine that built it is worse than not checking a feature the generated code (compares and adds) cannot use.
CPUINFO_FLAGS = {
    "x86_64": {
        **{
            f: f
            for f in "sse sse2 ssse3 fxsr popcnt aes avx avx2 bmi1 bmi2 f16c fma movbe pclmulqdq adx rdrand rdseed xsave xsaveopt xsavec xsaves".split()
        },
        **{f: f for f in "avx512f avx512bw avx512cd avx512dq avx512vl avx512ifma avx512vbmi vaes vpclmulqdq gfni clflushopt clwb".split()},
        "sse3": "pni",
        "sse4.1": "sse4_1",
        "sse4.2": "sse4_2",
        "cmpxchg16b": "cx16",
        "lzcnt": "abm",
        "sha": "sha_ni",
        "avx512vnni": "avx512_vnni",
        "avx512vbmi2": "avx512_vbmi2",
        "avx512vpopcntdq": "avx512_vpopcntdq",
        "avx512bitalg": "avx512_bitalg",
        "ermsb": "erms",
        "prfchw": "3dnowprefetch",
    },
    "aarch64": {
        **{f: f for f in "fp aes sha2 sha3 sve sve2 i8mm bf16 fcma flagm".split()},
        "neon": "asimd",
        "crc": "crc32",
        "lse": "atomics",
        "fp16": "fphp",
        "rdm": "asimdrdm",
        "dotprod": "asimddp",
        "rcpc": "lrcpc",
        "dpb": "dcpop",
        "jsconv": "jscvt",
        "fhm": "asimdfhm",
    },
}
CPUINFO_FLAGS["amd64"] = CPUINFO_FLAGS["x86_64"]
CPUINFO_FLAGS["arm64"] = CPUINFO_FLAGS["aarch64"]
TARGET_CPU_HINT = "x86-64-v2 (the default), x86-64-v3, x86-64 or native (`rustc --print target-cpus` lists them)"


def default_cpu() -> str:
    """The CPU generated code is built for unless one is asked for: x86-64-v2 (SSE4.2 and POPCNT: Intel Nehalem, AMD Bulldozer and later), `generic` on arm.
    AVX2 and later are never compiled in by default; the flat library detects AVX2 at run time on its own."""
    return "x86-64-v2" if ARCH in ("x86_64", "amd64") else "generic"


def cpu_option(native: bool, target_cpu: str | None) -> str | None:
    """`--native` is `--target-cpu native`: the one value the two command-line spellings give, or a refusal when they disagree."""
    if native and target_cpu not in (None, "native"):
        raise ValueError(f"--native and --target-cpu {target_cpu} are alternatives: give one")
    return "native" if native else target_cpu


def resolve_target_cpu(options: dict) -> str:
    """The `-C target-cpu` value: `options["target_cpu"]`, default the portable `default_cpu()`; `native` is the opt-in for this machine only."""
    cpu = options.get("target_cpu")
    if cpu is not None and (not isinstance(cpu, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,63}", cpu)):
        raise ValueError(f"--target-cpu {cpu!r}: give a CPU name, one of {TARGET_CPU_HINT}")
    return cpu or default_cpu()


def target_features(rustc: str, cpu: str) -> list[str]:
    """The CPU features rustc enables for `-C target-cpu=<cpu>` (the `target_feature=` lines of `rustc --print cfg`). An unknown CPU name is
    refused: rustc alone would only warn and build for the baseline."""
    try:
        p = subprocess.run([rustc, "--print", "cfg", "-C", f"target-cpu={cpu}"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        raise ValueError(f"cannot ask rustc for the features of target CPU {cpu!r}: {type(e).__name__}") from None
    if p.returncode or "not a recognized processor" in p.stderr:
        raise ValueError(f"target CPU {cpu!r}: rustc does not know it; use one of {TARGET_CPU_HINT}")
    return sorted(set(re.findall(r'^target_feature="([^"]+)"', p.stdout, re.M)))


def cpu_flags() -> set[str] | None:
    """The features EVERY logical CPU of this machine reports (a process may run on any of them), None when they cannot be read."""
    try:
        text = CPUINFO.read_text(errors="replace")
    except OSError:
        return None
    sets = [set(v.split()) for k, _, v in (ln.partition(":") for ln in text.splitlines()) if k.strip().lower() in ("flags", "features")]
    return set.intersection(*sets) if sets else None


def read_target(path) -> tuple[str, str, list[str]] | None:
    """(cpu, architecture, features) a library was built for, from the record inside it; None for a library built before the record existed."""
    try:
        with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            i = mm.find(TARGET_MARK.encode() + b"|")
            j = mm.find(b"\0", i) if i >= 0 else -1
            text = mm[i:j].decode("ascii", "replace") if j > i else ""
    except (OSError, ValueError):  # unreadable, or empty (mmap refuses a zero-length file): the loader reports it in its own words
        return None
    parts = text.split("|")
    return (parts[1], parts[2], [f for f in parts[3].split(",") if f]) if len(parts) >= 4 and parts[0] == TARGET_MARK else None


def missing_features(feats) -> list[str]:
    """The features among `feats` that this CPU does not report (none when its features cannot be read or a name is not one the check knows)."""
    have = cpu_flags()
    names = CPUINFO_FLAGS.get(ARCH, {})
    return [f for f in feats if f in names and have is not None and names[f] not in have]


def check_target(path) -> tuple[str, str, list[str]] | None:
    """Refuses (TargetCpuMismatch, one line) a library built with features this CPU lacks. Done in Python BEFORE the library is loaded: a check
    compiled into a native library would itself be built for that CPU (and could be folded to true)."""
    t = read_target(path)
    if t is None or t[1] != ARCH:  # old library, or another architecture (the loader says so itself)
        return t
    cpu, _, feats = t
    if missing := missing_features(feats):
        more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
        raise TargetCpuMismatch(
            f"{Path(path).name} was built for the {cpu!r} CPU and needs {', '.join(missing[:5])}{more}, which this CPU lacks "
            f"(running it would crash with an illegal instruction): rebuild it on this machine, or build without --native / --target-cpu "
            f"(the default x86-64-v2 build runs on Intel Nehalem, AMD Bulldozer and the mainstream x86 cores after them; --target-cpu x86-64 is the widest)"
        )
    return t


# Generated code writes one tree as ONE nested if/else expression, and rustc's parser overflows its stack on nesting deeper than about
# 1,300 levels (measured on rustc 1.99: a chain of 1,300 splits builds, 1,400 on the right spine and 1,500 on either crash with a backtrace).
# A deeper tree is refused up front; the flat format holds trees as deep as its 65,535-node child distance allows.
MAX_TREE_DEPTH = 1000


def tree_depth(t: Tree) -> int:
    """Splits on the longest root-to-leaf path of one tree (a level at a time, so a deep chain costs one numpy step per level)."""
    frontier, depth = np.zeros(1, np.int64), 0
    while True:
        inner = frontier[t.left[frontier] != -1]
        if not len(inner):
            return depth
        frontier, depth = np.concatenate([t.left[inner], t.right[inner]]), depth + 1


def too_deep(forest: Forest) -> int | None:
    """The depth of the first tree deeper than MAX_TREE_DEPTH, else None. A tree of 2 * MAX_TREE_DEPTH nodes or fewer cannot be that deep
    (every level adds a leaf) and is not walked."""
    for m in forest.models:
        for t in m.trees:
            if len(t.left) > 2 * MAX_TREE_DEPTH and (d := tree_depth(t)) > MAX_TREE_DEPTH:
                return d
    return None


def toolchain_bin() -> Path:
    rd = os.environ.get("RUST_DIR")
    if rd:
        tcs = sorted(Path(rd, "rustup", "toolchains").glob("*/"))
        if tcs:
            return tcs[0] / "bin"
    w = shutil.which("rustc")
    if w:
        return Path(w).parent
    raise ToolchainMissing(
        "rustc not found (needed to compile generated code): install it with `apt install rustc` or rustup (https://rustup.rs), "
        "then put it on PATH or set RUST_DIR=<folder with rustup/>"
    )


LLVM_ROOT = Path("/usr/lib")  # where a distro's `apt install llvm` puts llvm-<version>/bin/ (a test points it elsewhere)
PROFDATA_HINT = (
    "get it with `rustup component add llvm-tools` (a rustup toolchain) or `apt install llvm` (a distro rustc); "
    "it must come from the same LLVM major version as rustc (`rustc -vV` prints it)"
)


class ProfdataMissing(ToolchainMissing):
    """No llvm-profdata of rustc's LLVM major version. `why` is the one-line reason without the way out (a notice adds its own ending)."""

    def __init__(self, why: str):
        super().__init__(f"{why}: {PROFDATA_HINT}")
        self.why = why


def rustc_llvm() -> tuple[Path | None, str | None, int | None]:
    """(sysroot, host triple, LLVM major version) of the rustc that builds the generated code, from `rustc --print sysroot` and `rustc -vV`
    (the `host:` and `LLVM version:` lines); None for what cannot be read. With rustc on PATH the rustc found is usually rustup's proxy in
    the cargo home, whose folder says nothing about the toolchain it runs: only rustc itself knows where its toolchain is."""
    rustc = str(toolchain_bin() / "rustc")

    def ask(*args: str) -> str:
        try:
            return subprocess.run([rustc, *args], capture_output=True, text=True, timeout=60).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    verbose = ask("-vV")
    host, llvm = re.search(r"^host:\s*(\S+)", verbose, re.M), re.search(r"^LLVM version:\s*(\d+)", verbose, re.M)
    sysroot = ask("--print", "sysroot").strip()
    return (Path(sysroot) if sysroot else None, host.group(1) if host else None, int(llvm.group(1)) if llvm else None)


def tool_llvm_major(tool: Path) -> int | None:
    """The LLVM major version a tool reports (`llvm-profdata --version` prints `LLVM version 18.1.3`, with a vendor's suffix or prefix on some
    builds), None when it cannot be read."""
    try:
        out = subprocess.run([str(tool), "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"LLVM version\s+(\d+)", out)
    return int(m.group(1)) if m else None


def llvm_profdata() -> Path:
    """The llvm-profdata of the LLVM that rustc was built on (a profile written by one LLVM is unusable by another's profdata), never another:
    first the rustup `llvm-tools` copy of rustc's own toolchain (`<rustc --print sysroot>/lib/rustlib/<host>/bin`), then, only when it reports
    rustc's LLVM major version (`rustc -vV`, `--version` of each), `llvm-profdata` on PATH and the `/usr/lib/llvm-*/bin` ones of a distro
    (its rustc has no llvm-tools component). With none, ProfdataMissing names rustc's version and the versions found."""
    sysroot, host, want = rustc_llvm()
    if sysroot:
        rustlib = sysroot / "lib" / "rustlib"  # a toolchain keeps llvm-tools for its host only
        for p in [rustlib / host / "bin" / "llvm-profdata"] if host else sorted(rustlib.glob("*/bin/llvm-profdata")):
            if p.is_file():
                return p
    if want is None:
        raise ProfdataMissing(
            "rustc's LLVM version could not be read (`rustc -vV` prints no `LLVM version:` line), so no llvm-profdata can be matched to it"
        )
    on_path = shutil.which("llvm-profdata")
    others = ([Path(on_path)] if on_path else []) + sorted(
        (q for q in LLVM_ROOT.glob("llvm-*/bin/llvm-profdata")), key=lambda q: -int(re.sub(r"\D", "", q.parent.parent.name) or 0)
    )
    seen: list[str] = []
    done = set()
    for p in others:
        real = p.resolve()
        if real in done:
            continue
        done.add(real)
        got = tool_llvm_major(p)
        if got == want:
            return p
        seen.append(f"LLVM {got} at {p}" if got else f"an unreadable version at {p}")
    found = f" (found only {'; '.join(seen)})" if seen else ""
    raise ProfdataMissing(f"no llvm-profdata of rustc's LLVM {want}{found}, and the profiles of another LLVM version are rejected")


def merge_profiles(merged: Path, raw_files) -> None:
    code, msg = run([str(llvm_profdata()), "merge", "-o", str(merged), *map(str, raw_files)])
    if code:
        raise BuildFailed(f"llvm-profdata failed: {msg}\n{PROFDATA_HINT}")


def lit(v) -> str:
    f = np.float32(v)
    if np.isnan(f):
        raise ValueError("NaN threshold or leaf")
    if np.isinf(f):  # LightGBM writes `inf` for the threshold of a NaN-versus-the-rest split: always left for a number
        return "f32::INFINITY" if f > 0 else "f32::NEG_INFINITY"
    return str(f) + "f32"


def lit64(v) -> str:
    """A finite double as a Rust f64 literal (Python's repr is the shortest text that reads back as the same double)."""
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("a non-finite linear constant or coefficient")
    return f"{v!r}f64"


def linear_leaf(rec: tuple) -> str:
    """A linear leaf as an f64 expression: the fallback when any of its features is NaN, else the constant plus coefficient * x terms added
    left to right (the order of LightGBM's loop; Rust does not fuse the multiply-add, so the sums are the library's own)."""
    const, fallback, feats, coefs = rec
    if not len(feats):
        return lit64(const)
    nan = " || ".join(f"x[{int(f)}].is_nan()" for f in feats)
    terms = " + ".join(f"{lit64(c)} * (x[{int(f)}] as f64)" for f, c in zip(feats, coefs, strict=True))
    return f"(if {nan} {{ {lit64(fallback)} }} else {{ {lit64(const)} + {terms} }})"


def f64_tree(t: Tree, exact: bool) -> bool:
    """Does this tree's expression evaluate to an f64? (linear leaves; or the f64 leaves, whose double values the tree then carries.)"""
    return bool(t.linear) or (exact and t.leaf64 is not None)


def tree_expr(t: Tree, exact: bool = False, leaf=None) -> str:
    """The tree as one nested if/else expression. Semantics: `x < thr` goes left, NaN goes the node's default way, which is why a
    default-left node tests `!(x >= thr)` (true for NaN) and a default-right node tests `x < thr` (false for NaN). Zero-as-missing nodes
    also send |x| <= 1e-35 the default way; a categorical node tests `trunc(x)` against its set (right for NaN, for values below 0 after
    truncation toward zero, and for absent categories). `leaf(v)` (vector trees) gives the text of the leaf at node `v`, called in emission order."""
    left, right, feat, dl = t.left.tolist(), t.right.tolist(), t.feature.tolist(), t.default_left.tolist()
    zm = t.zero_missing.tolist() if t.zero_missing is not None else None
    cats = {int(k): np.asarray(v, dtype=np.int64).tolist() for k, v in (t.cat or {}).items()}
    lits = [lit(v) for v in t.value]
    # f64 leaves: every leaf is a double literal (a non-finite one keeps its f32 form)
    wide = exact and not t.linear and t.leaf64 is not None
    out, stack = [], [(0, 0)]
    while stack:
        kind, v = stack.pop()
        if kind == 1:
            out.append(v)
        elif left[v] == -1:
            if leaf is not None:
                out.append(leaf(v))
            elif wide and np.isfinite(t.leaf64[v]):
                out.append(lit64(t.leaf64[v]))
            else:
                out.append(
                    linear_leaf(t.linear[v]) if t.linear and v in t.linear else f"({lits[v]} as f64)" if t.linear or wide else lits[v]
                )
        else:
            f = f"x[{feat[v]}]"
            if v in cats:  # `as i64` truncates toward zero (so -0.5 is category 0, as in LightGBM) and turns NaN into 0, hence the guard
                members = " | ".join(map(str, cats[v]))
                cond = f"({f} == {f} && matches!({f} as i64, {members}))" if members else "false"  # an empty set sends every row right
            elif zm and zm[v]:
                cond = f"(!({f} >= {lits[v]}) || {f}.abs() <= {ZERO})" if dl[v] else f"({f} < {lits[v]} && {f}.abs() > {ZERO})"
            else:
                cond = f"!({f} >= {lits[v]})" if dl[v] else f"{f} < {lits[v]}"
            out.append(f"if {cond}{{")
            stack += [(1, "}"), (0, right[v]), (1, "}else{"), (0, left[v])]
    return "".join(out)


def vector_table_values(forest: Forest) -> int:
    """How many numbers the leaf tables of the vector trees hold (leaves times groups)."""
    return sum(n_vector_leaves(t) * t.vector.shape[1] for m in forest.models for t in m.trees if t.vector is not None)


def vector_type(t: Tree, exact: bool) -> str:
    """The element type of a vector tree's leaf table: double when the tree carries double rows and the forest sums in double (`--f64-leaves`)."""
    return "f64" if exact and t.vector64 is not None else "f32"


def n_vector_leaves(t: Tree) -> int:
    return int((t.left == -1).sum())


def tree_weight(t: Tree) -> int:
    """What a tree costs the chunk and RAM plans, in node-equivalents: its nodes, and for a vector tree also its leaf table."""
    if t.vector is None:
        return len(t.left)
    return len(t.left) + n_vector_leaves(t) * t.vector.shape[1] // VECTOR_LEAVES_PER_NODE


def vector_expr(t: Tree, exact: bool, k: int) -> tuple[str, str]:
    """(expression, table) of vector tree number `k`: the expression is `tree_expr`'s nested if/else whose leaves are `&V{k}[j]`, j the leaf's
    ordinal in emission order (depth first, left first); the table is `static V{k}: [[T; G]; L]` with the leaf rows in that same order."""
    order: list[int] = []

    def leaf(v: int) -> str:
        order.append(v)
        return f"&V{k}[{len(order) - 1}]"

    expr = tree_expr(t, exact, leaf)
    ty = vector_type(t, exact)
    rows = (t.vector64 if ty == "f64" else t.vector)[order]
    fmt = lit64 if ty == "f64" else lit
    body = ",".join("[" + ",".join(fmt(v) for v in row) + "]" for row in rows)
    return expr, f"static V{k}: [[{ty}; G]; {len(order)}] = [{body}];\n"


def module_of_tree(trees: list[Tree], modules: int) -> list[int]:
    """The module (0..modules-1) each tree of a chunk goes to: consecutive trees, balanced by weight (node count; a vector tree also counts its leaf table)."""
    total, done, j, out = sum(tree_weight(t) for t in trees), 0, 0, []
    for t in trees:
        if j < modules - 1 and done >= total * (j + 1) / modules:
            j += 1
        out.append(j)
        done += tree_weight(t)
    return out


def chunk_source(
    trees: list[Tree], first: int, ci: int, n_features: int, groups: int, mode: str, acc: str = "f32", exact: bool = False, units: int = 1
) -> str:
    """The Rust source of one chunk. With `units` > 1 the trees are spread over that many modules (`g0`, `g1`, ...): rustc makes one codegen unit
    per module, so `-C codegen-units=N` only has something to split when the source has modules. With 1 unit the source is a single module."""
    vector = any(t.vector is not None for t in trees)
    head = (
        f"#![allow(clippy::all, unused_unsafe, dead_code, unused_imports)]\nconst NF: usize = {n_features};\nconst G: usize = {groups};\n"
    )
    modules = min(units, len(trees))
    home = module_of_tree(trees, modules) if modules > 1 else [0] * len(trees)
    vis, where = ("pub(super) ", "g{}::") if modules > 1 else ("", "")
    body, calls = [[] for _ in range(max(modules, 1))], []
    for k, t in enumerate(trees, first):
        if t.vector is not None:
            e, table = vector_expr(t, exact, k)
            ty = vector_type(t, exact)
            if ty == "f64" and acc != "f64":
                raise ValueError("f64 leaves need double sums")
            cast = " as f64" if acc == "f64" and ty == "f32" else ""
            if mode == "row":
                body[home[k - first]].append(f"{table}#[inline(never)] {vis}fn t{k}(x: &[f32; NF]) -> &'static [{ty}; G] {{ {e} }}\n")
                calls.append(f"{{ let v = {where.format(home[k - first])}t{k}(x); for g in 0..G {{ a[g] += v[g]{cast}; }} }}")
            else:
                body[home[k - first]].append(
                    f"{table}#[inline(never)] {vis}unsafe fn b{k}(xs: *const f32, rows: usize, acc: *mut {acc}, stride: usize) {{\n"
                    f"  for r in 0..rows {{ let x = &*(xs.add(r * NF) as *const [f32; NF]); let v = {e};\n"
                    f"    for g in 0..G {{ *acc.add(g * stride + r) += v[g]{cast}; }} }}\n}}\n"
                )
                calls.append(f"{where.format(home[k - first])}b{k}(xs, rows, m, stride);")
            continue
        e = tree_expr(t, exact)
        if f64_tree(t, exact) and acc != "f64":
            raise ValueError("linear leaves and f64 leaves need double sums")
        ty = "f64" if f64_tree(t, exact) else "f32"  # a linear tree's leaves are doubles, and so are the f64 leaves
        cast = " as f64" if acc == "f64" and ty == "f32" else ""
        if mode == "row":
            body[home[k - first]].append(f"#[inline(never)] {vis}fn t{k}(x: &[f32; NF]) -> {ty} {{ {e} }}\n")
            calls.append(f"a[{t.group}] += {where.format(home[k - first])}t{k}(x){cast};")
        else:
            body[home[k - first]].append(
                f"#[inline(never)] {vis}unsafe fn b{k}(xs: *const f32, rows: usize, acc: *mut {acc}) {{\n"
                f"  for r in 0..rows {{ let x = &*(xs.add(r * NF) as *const [f32; NF]); *acc.add(r) += ({e}){cast}; }}\n}}\n"
            )
            calls.append(f"{where.format(home[k - first])}b{k}(xs, rows, m.add({t.group} * stride));")
    sig = f'#[no_mangle] pub unsafe extern "C" fn c{ci}(xs: *const f32, rows: usize, m: *mut {acc}, stride: usize)'
    if mode == "row":
        drv = (
            f"{sig} {{\n for r in 0..rows {{\n  let x = &*(xs.add(r * NF) as *const [f32; NF]);\n  let mut a = [0{acc}; G];\n"
            f"  for g in 0..G {{ a[g] = *m.add(g * stride + r); }}\n  {' '.join(calls)}\n"
            f"  for g in 0..G {{ *m.add(g * stride + r) = a[g]; }}\n }}\n}}\n"
        )
    else:
        drv = f"{sig} {{\n {' '.join(calls)}\n}}\n"
    if modules > 1:
        uses = "use super::{NF, G};" if vector else "use super::NF;"
        return head + "".join(f"mod g{j} {{\n{uses}\n{''.join(fns)}}}\n" for j, fns in enumerate(body)) + drv
    return head + "".join(body[0]) + drv


def driver_source(
    forest: Forest, chunks_per_fold: list[list[int]], block: int, acc: str = "f32", target: str | None = None, single: bool = False
) -> str:
    """The driver crate's source. With `single` the chunks are not external symbols but modules of this very crate (`mod c0;` finds c0.rs next to lib.rs and the
    table holds `c0::c0`): one crate, so fat LTO and the inliner see the whole forest. Everything else is the same text in both modes."""
    g, nfold = forest.groups, len(forest.models)
    base = ",".join(lit(v) for m in forest.models for v in m.base)
    if single:
        decl = "\n".join(f"mod c{c};" for cs in chunks_per_fold for c in cs)
        folds = ",".join("&[" + ",".join(f"c{c}::c{c}" for c in cs) + "]" for cs in chunks_per_fold)
        extern = decl
    else:
        decl = "\n".join(f"    fn c{c}(xs: *const f32, rows: usize, m: *mut {acc}, stride: usize);" for cs in chunks_per_fold for c in cs)
        folds = ",".join("&[" + ",".join(f"c{c}" for c in cs) + "]" for cs in chunks_per_fold)
        extern = f'extern "C" {{\n{decl}\n}}'
    scale = "f64::NAN" if forest.margin_scale != forest.margin_scale else f"{float(forest.margin_scale)!r}f64"
    segments = vector_driver_source(forest) if any(t.vector is not None for m in forest.models for t in m.trees) else ""
    # the CPU features the library was built for, as bytes the loader can find in the file without running anything
    mark = f'#[used]\n#[no_mangle]\npub static TFL_CG_TARGET: [u8; {len(target) + 1}] = *b"{target}\\0";\n' if target else ""
    return f"""#![allow(clippy::all, dead_code)]
mod finish;
use finish::{{predict_blocks, Meta}};
const NF: usize = {forest.n_features};
const G: usize = {g};
const BLOCK: usize = {block};
const BASE: [f32; {nfold * g}] = [{base}];
type Chunk = unsafe extern "C" fn(*const f32, usize, *mut {acc}, usize);
{extern}
static FOLDS: [&[Chunk]; {nfold}] = [{folds}];
{mark}
/// [n_features, n_groups, n_folds, n_trees, n_nodes, objective, average]
#[no_mangle]
pub unsafe extern "C" fn tfl_cg_info(out: *mut u32) {{
    for (i, v) in [{forest.n_features}u32, {g}, {nfold}, {forest.n_trees}, {forest.n_nodes}, {OBJECTIVE[forest.kind]}, {AVERAGE[forest.average]}].iter().enumerate() {{
        *out.add(i) = *v;
    }}
}}

/// The factor a base margin is multiplied by (LightGBM `sigmoid:s` is folded into the leaves); NaN: the models disagree.
#[no_mangle]
pub extern "C" fn tfl_cg_margin_scale() -> f64 {{
    {scale}
}}

{segments}#[no_mangle]
pub unsafe extern "C" fn tfl_cg_predict(x: *const f32, n_rows: usize, out: *mut f32, threads: u32) -> i32 {{
    run::<f32>(x, n_rows, out, threads, std::ptr::null())
}}

/// As `tfl_cg_predict`, plus a base margin (`n_rows * G` f64, row major, or null) added to every fold's raw score before the transform.
#[no_mangle]
pub unsafe extern "C" fn tfl_cg_predict_margin(x: *const f32, n_rows: usize, out: *mut f32, threads: u32, base_margin: *const f64) -> i32 {{
    run::<f32>(x, n_rows, out, threads, base_margin)
}}

/// As `tfl_cg_predict_margin` with f64 answers (`out` holds `n_rows * G` f64): the sum without the final rounding to float32 (a float32
/// forest's result widened).
#[no_mangle]
pub unsafe extern "C" fn tfl_cg_predict_f64(x: *const f32, n_rows: usize, out: *mut f64, threads: u32, base_margin: *const f64) -> i32 {{
    run::<f64>(x, n_rows, out, threads, base_margin)
}}

unsafe fn run<O: finish::Out>(x: *const f32, n_rows: usize, out: *mut O, threads: u32, base_margin: *const f64) -> i32 {{
    if NF == 0 {{ return 1; }}
    let x = std::slice::from_raw_parts(x, n_rows * NF);
    let out = std::slice::from_raw_parts_mut(out, n_rows * G);
    let bm = if base_margin.is_null() {{ None }} else {{ Some(std::slice::from_raw_parts(base_margin, n_rows * G)) }};
    let meta = Meta {{ n_features: NF, n_groups: G, n_folds: {nfold}, objective: {OBJECTIVE[forest.kind]}, average: {AVERAGE[forest.average]}, base: BASE.to_vec(), late: {str(bool(forest.late_margin and not forest.sum64)).lower()} }};
    let fold_block = |_: &mut (), f: usize, xb: &[f32], rows: usize, margin: &mut [{acc}]| {{
        for c in FOLDS[f] {{
            c(xb.as_ptr(), rows, margin.as_mut_ptr(), BLOCK);
        }}
    }};
    predict_blocks(&meta, BLOCK, x, out, bm, threads as usize, &|| (), &fold_block);
    0
}}
"""


def file_segments(forest: Forest) -> list[int]:
    """The output segments a vector forest exports (DISENO A1, rule R-SEG; the library has no record of the output kind, so the table carries K):
    `Forest.segments` if set, else one column per output for K-output regressors (identity, regression) and xgboost multilabel (sigmoid), else none
    (a YDF multi-class forest or a softmax is ONE output)."""
    if forest.segments:
        return list(forest.segments)
    task = forest.task if forest.task != "auto" else ("regression" if forest.kind == "identity" else "classification")  # Spec's rule
    if forest.groups >= 2 and ((forest.kind == "identity" and task == "regression") or forest.kind == "sigmoid"):
        return list(range(forest.groups + 1))
    return []


def vector_driver_source(forest: Forest) -> str:
    """The export a vector forest adds, `tfl_segments`' contract: returns the number of segments (0: none) and writes at most `cap` offsets."""
    seg = file_segments(forest)
    return f"""static SEGMENTS: [u32; {len(seg)}] = [{",".join(str(v) for v in seg)}];

/// Output segments: the number of segments (0: none); `out` receives min(cap, segments + 1) offsets.
#[no_mangle]
pub unsafe extern "C" fn tfl_cg_segments(out: *mut u32, cap: usize) -> u32 {{
    if !out.is_null() {{
        for (i, v) in SEGMENTS.iter().take(cap).enumerate() {{
            *out.add(i) = *v;
        }}
    }}
    SEGMENTS.len().saturating_sub(1) as u32
}}

"""


def host_lib_dir(rustc: str) -> Path | None:
    """`<sysroot>/lib/rustlib/<host triple>` of this rustc (a sysroot can hold several targets), None when rustc does not say."""
    code, root = run([rustc, "--print", "sysroot"])
    code2, vv = run([rustc, "-vV"])
    host = next((ln.split(":", 1)[1].strip() for ln in vv.splitlines() if ln.startswith("host:")), None) if not code2 else None
    return Path(root.strip()) / "lib" / "rustlib" / host if not code and host else None


def cspgo_unavailable(settings: dict, unit: str, rustc: str) -> str | None:
    """Why CS-PGO cannot be built for these settings on this toolchain (None = it can). CS-PGO is the linker route (rust-lld instruments and
    re-optimises after inlining): chunk crates joined by the linker with fat LTO, Linux x86-64 (the only place it was measured), rustc's rust-lld
    and its profiler runtime present."""
    if unit != "chunks":
        return "it works on the chunks build (the one-crate build is linked by rustc itself, with no linker plug-in)"
    if settings["lto"] != "fat":
        return "it needs the fat-LTO shape (--compile-preset 1.1)"
    if platform.system() != "Linux" or ARCH not in ("x86_64", "amd64"):
        return f"it was built and measured on Linux x86-64 only, not {platform.system()} {ARCH}"
    host_dir = host_lib_dir(rustc)
    if host_dir is None:
        return "rustc does not say where its sysroot or host is"
    if not list((host_dir / "lib").glob("libprofiler_builtins-*.rlib")):
        return "this rustc has no profiler runtime (libprofiler_builtins)"
    if not (host_dir / "bin" / "rust-lld").exists():
        return "this rustc has no rust-lld"
    return None


def raw_profile_version_object(rustc: str, src: Path, raw_dir: Path) -> Path:
    """A tiny object defining `__llvm_profile_raw_version` with the context-sensitive flag added: the linker route needs the symbol and the profiler
    runtime does not define it. The version is read from a raw profile the level-1 run wrote (bytes 8..16, little endian), the CS bit is 1 << 57."""
    raw = next(iter(sorted(raw_dir.glob("*.profraw"))), None)
    if raw is None:
        raise BuildFailed("CS-PGO: no raw profile to read the profile version from")
    magic, version = struct.unpack("<QQ", raw.read_bytes()[:16])
    if magic != 0xFF6C70726F667281:
        raise BuildFailed(f"CS-PGO: {raw.name} is not a 64-bit little-endian raw profile")
    rs = src / "rawver.rs"
    rs.write_text(f"#![no_std]\n#[no_mangle]\npub static __llvm_profile_raw_version: u64 = {version | (1 << 57):#x};\n")
    obj = src / "rawver.o"
    code, msg = run(
        [
            rustc,
            "--edition",
            "2021",
            "--crate-type",
            "lib",
            "--emit",
            "obj",
            "-C",
            "panic=abort",
            "-C",
            "relocation-model=pic",
            "-o",
            str(obj),
            str(rs),
        ]
    )
    if code:
        raise BuildFailed(f"CS-PGO: the profile version object failed to build: {msg}")
    return obj


def cs_profile_functions(path: Path) -> int:
    """Functions in the context-sensitive part of a merged profile (`llvm-profdata show --showcs`); 0 when the linker route did not instrument."""
    p = subprocess.run([str(llvm_profdata()), "show", "--showcs", str(path)], capture_output=True, text=True)
    m = re.search(r"Total functions:\s*(\d+)", p.stdout)
    return int(m.group(1)) if m else 0


def merge_profiles_with(merged: Path, inputs) -> None:
    code, msg = run([str(llvm_profdata()), "merge", "-o", str(merged), *map(str, inputs)])
    if code:
        raise BuildFailed(f"llvm-profdata merge failed: {msg}")


def cspgo_stages(build, rustc: str, src: Path, merged: Path, parts) -> tuple[Path, Watch, dict]:
    """CS-PGO on top of the merged level-1 profile: a CS-instrumented build (the chunks as bitcode, rust-lld instruments after inlining), the
    profile runs on the same rows, the merge with the level-1 profile, the final build with both. Returns (library, its Watch, stats)."""
    t0 = time.perf_counter()
    thin = f"-Wl,--thinlto-jobs={os.cpu_count() or 2}"
    warn = ["-C", "llvm-args=-pgo-warn-missing-function=true"]
    prof_rlib = sorted((host_lib_dir(rustc) / "lib").glob("libprofiler_builtins-*.rlib"))[0]
    rawver = raw_profile_version_object(rustc, src, src / "raw-a")
    rawcs = src / "rawcs"
    rawcs.mkdir()
    link_gen = [
        "-C",
        f"link-arg={thin}",
        "-C",
        "link-arg=-Wl,--lto-cs-profile-generate",
        "-C",
        f"link-arg=-Wl,--lto-cs-profile-file={rawcs}/cs.profraw",
        "-C",
        "link-arg=-Wl,-u,__llvm_profile_runtime",
        "-C",
        f"link-arg={rawver}",
        "-C",
        f"link-arg={prof_rlib}",
    ]
    inst = build(["-C", f"profile-use={merged}", *warn], "csgen", None, lpl=True, link_extra=link_gen, strip=False)
    t_inst = time.perf_counter()
    for name, _ in parts:
        profile_run(inst, src / f"rows-{name}.npy", src / f"rawcs-{name}", "cover" if name == "c" else "main")
    t_run = time.perf_counter()
    raws = [f for name, _ in parts for f in (src / f"rawcs-{name}").glob("*.profraw")]
    both = src / "cs-merged.profdata"
    merge_profiles_with(both, [merged, *raws])
    n_cs = cs_profile_functions(both)
    if n_cs <= 0:
        raise BuildFailed("the CS profile is empty: the linker route did not instrument the code")
    watch = Watch()
    so = build(
        ["-C", f"profile-use={both}", *warn],
        "csfinal",
        watch,
        lpl=True,
        link_extra=["-C", f"link-arg={thin}", "-C", f"link-arg=-Wl,--lto-cs-profile-file={both}"],
        strip=True,
    )
    t_end = time.perf_counter()
    return (
        so,
        watch,
        {
            "cs_profile_functions": n_cs,
            "cs_seconds": {
                "instrumented_build": round(t_inst - t0, 1),
                "profile_runs": round(t_run - t_inst, 1),
                "final_build": round(t_end - t_run, 1),
            },
        },
    )


class Watch:
    """Keeps the lines of a build's output that say a profile was not applied (several rustc jobs share one, hence the lock)."""

    # LLVM's own warnings: a function the profile has no entry for, and one whose control flow differs from the profiled one
    pattern = re.compile(r"function control flow change detected|no profile data available for function")
    named = re.compile(r"(?:hash mismatch\)|available for function)\s+(\S+)")  # the function a warning is about

    def __init__(self):
        self.lines, self.cold, self._lock = [], 0, threading.Lock()

    def add(self, line: str) -> None:
        with self._lock:
            self.lines.append(line[:240])

    @property
    def n(self) -> int:
        return len(self.lines)


def run_measured(cmd, env=None, watch: Watch | None = None) -> tuple[int, str, float]:
    """(exit code, first lines of the output, peak memory of THIS child in MB). `wait4` gives the rusage of one child: `RUSAGE_CHILDREN` is
    the maximum over every child the process ever waited for, so the second build of a process would report the first one's peak.
    `watch` counts the output lines that say a profile was not applied."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    lines = []
    for ln in p.stdout:
        if watch is not None and watch.pattern.search(ln):
            watch.add(ln.strip())
        if len(lines) < 12 and ln.strip():
            lines.append(ln.strip()[:240])
    p.stdout.close()
    _, status, ru = os.wait4(p.pid, 0)
    p.returncode = os.waitstatus_to_exitcode(status)
    return p.returncode, "\n".join(lines), ru.ru_maxrss / 1e3


def run(cmd, env=None) -> tuple[int, str]:
    code, msg, _ = run_measured(cmd, env)
    return code, msg


# ---- the build plan: shape and RAM -----------------------------------------------------------------------------------------------------------------
# Peak memory of ONE rustc job on a chunk of `n` nodes at `u` codegen units: RUSTC_BASE_MB + RUSTC_MB_PER_KNODE * n/1000 * (1 + RUSTC_UNIT_EXTRA * (u - 1));
# the final link of the driver (which runs alone, after the chunks) needs RUSTC_LINK_MB. The constants lie ABOVE every peak read so far (`wait4` ru_maxrss
# of the rustc child, MB), because a build killed for memory is worse than one with a job fewer:
#   test fixture, 880 nodes ........................ 148 (LIMITS.md)        driver link, opt 3 ............ 227-234 (measured 2026-10-23, below)
#   measured 2026-10-23, xgboost forest of depth-10 trees, one chunk, opt 3, codegen-block:
#     32.1k nodes: 227 (1 unit) / 260 (16)      68.7k nodes: 346 (1 unit) / 386 (16)       codegen-pgo, 68.7k nodes: 384 (1 unit) / 439 (16)
#   documented earlier (opt 2, 1 unit): 404 / 444 (block / pgo) at 89k-node chunks; 330 for the 208,649-node benchmark forest; ~850 per process on the
#   150k-node chunks of a 4.5M-node forest (read before a measurement fix): the per-node figure is set to cover that one too.
# Calibrated on chunks of up to 69k nodes; a 150k-node chunk at 16 units (the default chunk size) has not been measured.
# With 16 units the same chunk is built 1.6x (block) to 2.9x (pgo) faster at 10 to 15 % more memory.
RUSTC_BASE_MB = 150.0
RUSTC_MB_PER_KNODE = 4.8
RUSTC_UNIT_EXTRA = 0.025
RUSTC_LINK_MB = 250.0
RAM_SHARE = 0.85  # of the RAM available the jobs may plan to use: the rest is the parent process, the linker and estimate error
MIN_CHUNK_NODES = 4_000  # chunks are halved down to this when even one job does not fit; below it the build is refused


def shape_removed_message(shape: str) -> str:
    return (
        f"--shape {shape}: removed ({REMOVED_SHAPES[shape]}); every build uses {SHAPES['fast']} codegen units, on all the threads. "
        f"Use --shape fast (QF 1.0's build) or no --shape (the default)"
    )


def shape_argument(value: str) -> str:
    """argparse `type` of --shape: fast, or a clear refusal (a removed shape says what replaced it)."""
    if value in REMOVED_SHAPES:
        raise argparse.ArgumentTypeError(shape_removed_message(value))
    if value not in SHAPES:
        raise argparse.ArgumentTypeError(f"invalid choice {value!r} (choose from {', '.join(SHAPES)})")
    return value


def resolve_shape(options: dict) -> str | None:
    """The named QF 1.0 build of `options["shape"]` (fast), None when none is named; anything else is refused in one line."""
    shape = options.get("shape")
    if shape is None or shape == "":
        return None
    if shape in REMOVED_SHAPES:
        raise ValueError(shape_removed_message(shape))
    if shape not in SHAPES:
        raise ValueError(f"--shape {shape!r}: give fast (QF 1.0's build: opt 3, no LTO, {SHAPES['fast']} codegen units)")
    return shape


def resolve_pgo(value) -> str:
    """`options["pgo"]` as "on", "off" or "auto" (not given: the backend profiles whenever it has rows). Booleans are taken as on / off."""
    if value is None or value == "":
        return "auto"
    if value is True:
        return "on"
    if value is False:
        return "off"
    if value not in PGO_VALUES:
        raise ValueError(f"--pgo {value!r}: give on (a profile-guided build; needs --data) or off")
    return value


def resolve_cspgo(value) -> str:
    """`options["cspgo"]` as "on", "off" or "auto" (not given: CS-PGO is added to every profile-guided build that can have it). Booleans are taken as on / off."""
    if value is None or value == "":
        return "auto"
    if value is True:
        return "on"
    if value is False:
        return "off"
    if value not in CSPGO_VALUES:
        raise ValueError(
            f"--cspgo {value!r}: give on (CS-PGO on top of the profile-guided build; refused where it cannot be built) or off (plain PGO)"
        )
    return value


def resolve_inline_threshold(value) -> int | None:
    """`options["inline_threshold"]`: a whole number >= 0 for `-C llvm-args=-inline-threshold=N`; None or "default" = rustc's own."""
    if value is None or (isinstance(value, str) and value.strip().lower() in ("", "default", "none")):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = -1
    if isinstance(value, bool) or n < 0 or n > MAX_INLINE_THRESHOLD or (isinstance(value, float) and value != n):
        raise ValueError(
            f"--inline-threshold {value!r}: give a whole number from 0 to {MAX_INLINE_THRESHOLD:,} (LLVM's inline cost threshold) or default (rustc's own)"
        )
    return n


def resolve_codegen_units(value) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 0
    if isinstance(value, bool) or n < 1 or n > MAX_CODEGEN_UNITS or (isinstance(value, float) and value != n):
        raise ValueError(f"--codegen-units {value!r}: give a whole number from 1 to {MAX_CODEGEN_UNITS}")
    return n


MAX_INLINE_THRESHOLD = 100_000
MAX_CODEGEN_UNITS = 256


def resolve_unit(value) -> str:
    """`options["unit"]`: "single" (the whole forest and the driver as one crate) or "chunks" (every chunk its own crate, QF 1.0's way)."""
    if value not in UNIT_MODES:
        raise ValueError(
            f"--unit {value!r}: give single (the whole forest and the driver as ONE crate, so fat LTO and inlining cross the chunks) "
            "or chunks (every chunk its own crate, QF 1.0's way)"
        )
    return value


def resolve_compile(options: dict) -> dict:
    """How generated code is built, from the options: {"lto", "codegen_units", "inline_threshold", "unit", "pgo", "preset", "shape"}.
    The preset (`compile_preset`, default DEFAULT_COMPILE_PRESET) is the start; a named `shape` replaces its lto / units / inline with that QF 1.0
    build; then each of `lto`, `codegen_units`, `inline_threshold`, `unit`, `pgo` that was given overrides. "preset" is the name of the preset whose lto / units / inline the result equals ("custom" when none); "shape" the same for "fast" and "top"."""
    name = options.get("compile_preset") or DEFAULT_COMPILE_PRESET
    if name not in COMPILE_PRESETS:
        raise ValueError(f"--compile-preset {name!r}: give one of {', '.join(COMPILE_PRESETS)} (1.0 = exactly QF 1.0's build)")
    s = dict(COMPILE_PRESETS[name])
    shape = resolve_shape(options)
    if shape:
        s.update(SHAPE_BUILDS[shape])
    if options.get("lto") is not None:
        lto = options["lto"]
        if isinstance(
            lto, bool
        ):  # `--opt lto=off` / `lto=on` reach here as False / True (qf_common.parse_options reads off / on as booleans)
            lto = "fat" if lto else "off"
        if lto not in LTO_MODES:
            raise ValueError(f"--lto {options['lto']!r}: give off or fat")
        s["lto"] = lto
    if options.get("codegen_units") is not None:
        s["codegen_units"] = resolve_codegen_units(options["codegen_units"])
    if "inline_threshold" in options and options["inline_threshold"] is not None:
        s["inline_threshold"] = resolve_inline_threshold(options["inline_threshold"])
    if options.get("unit") is not None:
        s["unit"] = resolve_unit(options["unit"])
    if resolve_pgo(options.get("pgo")) != "auto":
        s["pgo"] = resolve_pgo(options["pgo"])
    s["cspgo"] = resolve_cspgo(options.get("cspgo")) if resolve_cspgo(options.get("cspgo")) != "auto" else s.get("cspgo", "auto")
    return name_build(s)


def name_build(s: dict) -> dict:
    """Sets `preset` / `shape` of a settings dict: the preset or QF 1.0 shape whose lto, codegen units, inline threshold and unit the build equals, else "custom"."""
    build = ("lto", "codegen_units", "inline_threshold", "unit")
    s["preset"] = next((n for n, p in COMPILE_PRESETS.items() if all(s[k] == p[k] for k in build)), "custom")
    s["shape"] = next((n for n, p in SHAPE_BUILDS.items() if all(s[k] == p[k] for k in build)), "custom")
    return s


def add_compile_arguments(ap) -> None:
    """The command-line flags of the compile settings (compile.py and optimize.py share them); `compile_options(args)` reads them back."""
    ap.add_argument(
        "--compile-preset",
        choices=tuple(COMPILE_PRESETS),
        help=f"a whole build: 1.1 (the default, {describe_compile({**COMPILE_PRESETS['1.1']})}, profile-guided of the same shape with --data) "
        f"or 1.0 (exactly QF 1.0's build: {describe_compile({**COMPILE_PRESETS['1.0']})}); the flags below override what the preset says",
    )
    ap.add_argument(
        "--lto", choices=LTO_MODES, help="link-time optimisation of generated code: fat or off (the default follows the preset)"
    )
    ap.add_argument(
        "--codegen-units",
        type=int,
        metavar="N",
        help="rustc codegen units per chunk (default 16, all threads; 1 = one unit everywhere, driver included, only when asked here; the RAM plan may lower N)",
    )
    ap.add_argument(
        "--unit",
        choices=UNIT_MODES,
        help=f"how the forest is put together: single = the whole forest and the driver as ONE crate (the chunks become modules), so fat LTO and inlining "
        f"cross the chunks (one big rustc process; falls back to chunks, with a notice, when the RAM does not allow it); chunks = every chunk its own crate, "
        f"joined as objects (QF 1.0's way). The default is {DEFAULT_COMPILE_UNIT} (--compile-preset 1.0 and --shape are chunks unless --unit single is given)",
    )
    ap.add_argument(
        "--inline-threshold",
        metavar="N|default",
        help="LLVM inline threshold, -C llvm-args=-inline-threshold=N (3000 in the 1.1 default; `default` = rustc's own)",
    )
    ap.add_argument(
        "--pgo",
        choices=PGO_VALUES,
        help="profile-guided build, instrumented and final in the same shape: on needs --data; off builds codegen-block; "
        "the default is on whenever --data gives rows",
    )
    ap.add_argument(
        "--cspgo",
        choices=CSPGO_VALUES,
        help="CS-PGO, a second profile pass on the linker route on top of the profile-guided build (QF standard, docs/BUILD-STANDARD.md rule 4): "
        "the default is on in every profile-guided fat-LTO chunks build the toolchain can do it for (else plain PGO, with a notice); "
        "off is plain PGO; on refuses where it cannot be built. Compile time is 2 to 4 times longer",
    )
    ap.add_argument(
        "--shape",
        type=shape_argument,
        metavar="fast",
        help="QF 1.0's build by name (fast = --compile-preset 1.0: opt 3, no LTO, 16 codegen units); "
        "it replaces the preset's lto / units / inline",
    )


def compile_options(a) -> dict:
    """The compile settings given on the command line, as the options the API takes (only what was given)."""
    got = {
        "compile_preset": getattr(a, "compile_preset", None),
        "lto": getattr(a, "lto", None),
        "codegen_units": getattr(a, "codegen_units", None),
        "inline_threshold": getattr(a, "inline_threshold", None),
        "unit": getattr(a, "unit", None),
        "pgo": getattr(a, "pgo", None),
        "cspgo": getattr(a, "cspgo", None),
        "shape": getattr(a, "shape", None),
    }
    return {k: v for k, v in got.items() if v is not None}


def describe_compile(s: dict) -> str:
    """'fat LTO, 16 codegen units, inline 3000' -- how the build is written in the notes and the reports."""
    u = s["codegen_units"]
    return (
        f"{'fat LTO' if s['lto'] == 'fat' else 'no LTO'}, {u} codegen {'unit' if u == 1 else 'units'}, "
        + (f"inline {s['inline_threshold']}" if s["inline_threshold"] is not None else "rustc's inlining")
        + (", one crate" if s.get("unit") == "single" else "")
    )


def inline_flags(s: dict) -> list[str]:
    return ["-C", f"llvm-args=-inline-threshold={s['inline_threshold']}"] if s["inline_threshold"] is not None else []


def chunk_flags(opt: int, s: dict, units: int) -> list[str]:
    """The flags of every chunk build, instrumented and final alike (a profile only matches code built the same way). `units` is the codegen units
    the RAM plan left (the wanted ones when the memory allows). `-C lto=fat` is emitted only for the one-crate build (unit single), where it acts: on
    a chunk crate (`--crate-type lib --emit obj`) it is a no-op (objects byte-identical, checked on rustc 1.99), so a chunks build asks fat LTO only of
    the driver's link (driver_flags). `lto=off` stays on a chunk of a no-LTO build: QF 1.0's flags, exactly."""
    lto = [] if s["lto"] == "fat" and s.get("unit") != "single" else ["-C", f"lto={s['lto']}"]
    return ["-C", f"opt-level={opt}", *lto, "-C", f"codegen-units={units}", *inline_flags(s), *check_flags(opt)]


def check_flags(opt: int) -> list[str]:
    """rustc turns overflow checks and debug assertions on at opt-level 0 and off from opt-level 1. A chunk built at opt-level 0 with the checks on calls
    core's `panic_const_*` functions, which a fat-LTO link of the driver drops, so the library would not load (`undefined symbol`). Opt-level 0 is
    therefore built with the checks off, the state every other level is built in: the same arithmetic, the answers bit for bit those of opt-level 3."""
    return ["-C", "overflow-checks=off", "-C", "debug-assertions=off"] if opt == 0 else []


def driver_flags(s: dict) -> list[str]:
    """The flags of the driver's link (the hot loop over row blocks), instrumented and final alike. QF 1.0's `fast` build adds none (rustc's own
    defaults, the state it was measured in); a codegen-unit count the user gave goes to the driver's link too; fat LTO is asked of the link too, where the
    standard library and the driver meet."""
    return [
        *(
            ["-C", f"codegen-units={s['codegen_units']}"] if s["codegen_units"] != SHAPES["fast"] else []
        ),  # the user's own number; 16 is rustc's default
        *(["-C", "lto=fat"] if s["lto"] == "fat" else []),
        *inline_flags(s),
    ]


# Peak memory of a rustc job under the settings, as a multiple of the QF 1.0 estimate (the per-node formula below was measured on that build).
# These factors are estimates, not a per-chunk measurement, and may change once the peak is measured per chunk. They lie above the growth seen on a
# whole small pipeline built by cargo on purpose: a build that takes a job fewer is better than one the kernel kills.
LTO_PEAK_FACTOR = 2.0  # fat LTO
INLINE_PEAK_FACTOR_PER_1000 = 0.15  # + this per 1,000 of inline threshold above rustc's own (3000 -> +0.45)


def link_peak_factor(s: dict) -> float:
    """How much more memory a rustc job that runs fat LTO may need than the same job of QF 1.0's build: the one-crate build and the driver's link."""
    f = LTO_PEAK_FACTOR if s["lto"] == "fat" else 1.0
    return f * (1 + INLINE_PEAK_FACTOR_PER_1000 * s["inline_threshold"] / 1000) if s["inline_threshold"] else f


def peak_factor(s: dict) -> float:
    """How much more memory one rustc job of this build may need than the same job of QF 1.0's build. A chunk crate gets no LTO (fat LTO is a no-op
    on it, see chunk_flags), so in chunks mode only the inline threshold is charged; the one-crate build pays for both (and the driver's link of a
    chunks build, which does run fat LTO, is planned with link_peak_factor)."""
    if s.get("unit") == "single":
        return link_peak_factor(s)
    return link_peak_factor({**s, "lto": "off"})


def rustc_peak_estimate_mb(nodes: int, units: int, factor: float = 1.0) -> float:
    return factor * (RUSTC_BASE_MB + RUSTC_MB_PER_KNODE * nodes / 1000 * (1 + RUSTC_UNIT_EXTRA * (units - 1)))


def meminfo_available_mb() -> float | None:
    try:
        for ln in Path("/proc/meminfo").read_text().splitlines():
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1e3  # kB -> MB
    except (OSError, ValueError, IndexError):
        pass
    return None


def cgroup_headroom_mb(proc_cgroup: str = "/proc/self/cgroup", root: str = "/sys/fs/cgroup") -> float | None:
    """What the tightest memory limit of this process's cgroup (v2: `memory.max` of it or of any parent) still leaves; None where there is no limit."""
    top = Path(root)
    try:
        here = next(ln.split(":", 2)[2].strip() for ln in Path(proc_cgroup).read_text().splitlines() if ln.startswith("0::"))
    except (OSError, StopIteration, IndexError):
        return None
    room, p = None, top / here.lstrip("/")
    while True:
        try:
            limit = (p / "memory.max").read_text().strip()
            if limit != "max":
                left = (int(limit) - int((p / "memory.current").read_text())) / 1e6
                room = left if room is None else min(room, left)
        except (OSError, ValueError):
            pass
        if p == top or p == p.parent:
            return room
        p = p.parent


def available_ram_mb(options: dict) -> float | None:
    """The RAM a build may use, in MB: `options["ram_mb"]` (or the environment variable QF_RAM_MB) when given, else the smaller of MemAvailable and the
    room left under a cgroup memory limit; None where neither can be read."""
    given = options.get("ram_mb", os.environ.get("QF_RAM_MB"))
    if given not in (None, ""):
        try:
            mb = float(given)
        except (TypeError, ValueError):
            mb = -1.0
        if not mb > 0:
            raise ValueError(f"ram_mb / QF_RAM_MB {given!r}: give the MB of RAM the build may use, a positive number")
        return mb
    found = [v for v in (meminfo_available_mb(), cgroup_headroom_mb()) if v is not None]
    return min(found) if found else None


def plan_build(
    chunk_nodes: list[int], jobs: int, units: int, available_mb: float | None, factor: float = 1.0, link_factor: float | None = None
) -> tuple[int, int] | None:
    """(jobs, units) to build with, so that the estimated peak of the rustc jobs running together (the biggest chunks at once) fits in RAM_SHARE of
    `available_mb`: the wanted ones if they fit, else fewer jobs first, then fewer units (one job); None if one job at one unit does not fit either.
    `factor` (peak_factor) scales the estimate of a job for a high inline threshold (and fat LTO in a one-crate build); `link_factor` (link_peak_factor,
    default `factor`) the final link, which runs fat LTO even when the chunks do not.
    Never swaps: the plan only ever uses memory that is free."""
    jobs = max(1, min(jobs, len(chunk_nodes)))
    if available_mb is None:
        return jobs, units
    budget = available_mb * RAM_SHARE
    big = sorted(chunk_nodes, reverse=True)

    def need(j: int, u: int) -> float:
        return max(
            RUSTC_LINK_MB * (factor if link_factor is None else link_factor), sum(rustc_peak_estimate_mb(n, u, factor) for n in big[:j])
        )

    for j in range(jobs, 0, -1):
        if need(j, units) <= budget:
            return j, units
    u = units
    while u > 1:
        u //= 2
        if need(1, u) <= budget:
            return 1, u
    return None


# One-crate build (unit = single): ONE rustc process holds the whole forest, so its peak is the per-node estimate of ALL the nodes at once (not of the biggest
# chunk): `plan_build([all nodes], 1, units, ...)` with the same factor as the chunk builds. This is an estimate, not a measurement: the per-node figure was measured on single chunks of up to 69k nodes, and under fat LTO the module is merged into one before code generation. The
# estimate errs high on purpose; when it says the build does not fit, `_compile_in` builds chunks instead and says so in one line.
def plan_single(total_nodes: int, units: int, available_mb: float | None, factor: float = 1.0) -> int | None:
    """The codegen units a one-crate build of `total_nodes` can use inside RAM_SHARE of `available_mb` (the wanted ones, else fewer), None when even one unit does
    not fit (then the build is made in chunks)."""
    fit = plan_build([total_nodes], 1, units, available_mb, factor)
    return None if fit is None else fit[1]


def split_chunks(forest: Forest, chunk_nodes: int) -> tuple[list, list[list[int]]]:
    """[(chunk id, first tree number, trees)] and the chunk ids of each fold: chunks never straddle two models (folds), the margins are re-initialised between them."""
    plan, chunks_per_fold, ci, k = [], [], 0, 0
    for m in forest.models:
        ids, cur, nodes = [], [], 0
        for t in m.trees:
            if cur and nodes + tree_weight(t) > chunk_nodes:
                plan.append((ci, k - len(cur), cur))
                ids.append(ci)
                ci, cur, nodes = ci + 1, [], 0
            cur.append(t)
            nodes += tree_weight(t)
            k += 1
        plan.append((ci, k - len(cur), cur))
        ids.append(ci)
        ci += 1
        chunks_per_fold.append(ids)
    return plan, chunks_per_fold


def object_files(folder: Path) -> list[Path]:
    """The objects rustc left for a chunk: one `<crate>.<unit>.rcgu.o` per codegen unit, or `<crate>.o` when it made a single one."""
    return sorted(folder.glob("*.rcgu.o")) or sorted(folder.glob("*.o"))


def mangled_names(name: str) -> list[str]:
    """The identifiers of a Rust symbol written as length-prefixed words (`_RNvCs<id>_2c03b25` v0, `_ZN2c03b2517h<hash>E` legacy):
    ['c0', 'b25']. A plain C name (`c0`) is its own single word."""
    name = name.rsplit(";", 1)[-1]  # a function of its own object is listed as `<object name>;<symbol>`
    m = re.search(r"_RNv(?:Nt)*Cs[0-9A-Za-z]*_(\d.*)$", name) or re.search(r"_ZN(\d.*)$", name)
    if not m:
        return [name]
    rest, words = m.group(1), []
    while (d := re.match(r"\d+", rest)) and (n := int(d.group())) and len(rest) >= d.end() + n:
        words.append(rest[d.end() : d.end() + n])
        rest = rest[d.end() + n :]
    if len(words) > 2 and re.fullmatch(r"h[0-9a-f]{16}", words[-1]):  # the hash a legacy symbol ends with
        words.pop()
    return words


def profile_table(merged: Path) -> dict[str, tuple[str, int]]:
    """{function name: (hash, weight)} of the merged profile, from `llvm-profdata show --all-functions --counts`. The weight of a function is the sum
    of its block counters (the entry count alone misses loops); a function the profile run never executed weighs 0."""
    p = subprocess.run([str(llvm_profdata()), "show", "--all-functions", "--counts", str(merged)], capture_output=True, text=True)
    if p.returncode:
        raise BuildFailed(f"llvm-profdata failed: {' '.join(p.stderr.split())[:240]}\n{PROFDATA_HINT}")
    table: dict[str, tuple[str, int]] = {}
    name = hash_ = None
    blocks: list[int] = []
    for ln in p.stdout.splitlines():
        head = re.match(r"^  (\S.*):$", ln)
        if head:
            name, hash_, blocks = head.group(1), "", []
            table[name] = (hash_, 0)
        elif name is not None:
            text = ln.strip()
            if text.startswith("Hash:"):
                hash_ = text.split(":", 1)[1].strip()
            elif text.startswith("Function count:") and not blocks:
                blocks = [int(text.split(":", 1)[1])]
            elif text.startswith("Block counts:"):
                blocks = [int(x) for x in re.findall(r"\d+", text.split(":", 1)[1])]
            table[name] = (hash_, sum(blocks))
    return table


HOT_SHARE = 0.9  # the hot functions are the fewest that together hold this share of the profile's counts
# A profile is the workload the code is tuned for, so it must be one: enough rows, and the same profile from two disjoint halves of them.
# Measured on aarch64, 2026-10-23, counts not timing, instrumented build at opt-level 3 like the real one (LIMITS.md has the table): on three
# xgboost models (1.4k, 3.2k and 35.5k nodes) a profile of 2,048 rows reaches 98.1, 97.3 and 87.9 % edge overlap with a 20,000-row profile and
# its two halves agree on 96.7, 94.7 and 79.7 % (20,000 rows: 98.8, 98.5 and 92.4 %). The floor is where the two moderate forests flatten out
# (+1 point from there to 20,000 rows). The threshold is set by the halves figure alone, between the profiles that must be refused and the ones
# that must be accepted: 10 rows reach at most 79.4 % (any model, three draws), 5,000 rows at least 85.7 % (the deepest forest, the lowest of
# them); 82 % is the middle, 3.2 points from each side. 100 rows are told apart by the floor.
PROFILE_MIN_ROWS = 2048  # fewer rows than this are refused: a profile of a handful of rows tunes the code for those rows
PROFILE_MIN_OVERLAP = 82.0  # percent: the edge-level overlap `llvm-profdata overlap` finds between the profiles of two halves of the rows
PROFILE_COVER_FACTOR = 2.5  # the coverage pass runs over this many blocks of rows (more than 2 so that two threads both get one)


def too_few_rows(n_rows: int) -> ProfileUnreliable:
    return ProfileUnreliable(
        f"{n_rows:,} profile rows are too few: a profile-guided build needs at least {PROFILE_MIN_ROWS:,} representative rows "
        "(the profile is the workload the code is tuned for); give --data with more rows"
    )


def hot_functions(table: dict[str, tuple[str, int]], share: float = HOT_SHARE) -> list[str]:
    """The names of the fewest functions that together hold `share` of all the counts of the profile, hottest first."""
    items = sorted(table.items(), key=lambda kv: -kv[1][1])
    total, hot, held = sum(w for _, (_, w) in items), [], 0
    for name, (_, w) in items:
        if held >= share * total:
            break
        hot.append(name)
        held += w
    return hot


def function_key(name: str) -> tuple[str, ...]:
    """What identifies a function whatever its object: (chunk crate, function) for a Rust symbol (a module path, a hash and the `<object>;` that
    private functions carry in the profile are dropped), the name itself for a plain C one."""
    words = mangled_names(name)
    if len(words) < 2:
        return (name.rsplit(";", 1)[-1],)
    # one crate per chunk: the crate is the first word (`c0`); one crate for the whole forest (unit = single): the chunk is the module `c0` inside it
    chunk = next((w for w in words[:-1] if re.fullmatch(r"c\d+", w)), words[0])
    return (chunk, words[-1])


def profile_row_count(path) -> int | None:
    """The number of rows of an .npy, None when it cannot be read (the build then fails on it with its own message)."""
    try:
        return int(np.load(path, mmap_mode="r").shape[0])
    except (OSError, ValueError):
        return None


def edge_overlap(base: Path, test: Path) -> float:
    """The edge-level program overlap in percent of two merged profiles of the same code (`llvm-profdata overlap`: 100 = every counter holds
    the same share of the counts in both). A line that cannot be read fails the build: an overlap nobody can read is no check."""
    p = subprocess.run([str(llvm_profdata()), "overlap", str(base), str(test)], capture_output=True, text=True)
    m = re.search(r"^\s*Edge profile overlap:\s*([0-9.]+)%", p.stdout, re.M)
    if p.returncode or not m:
        raise BuildFailed(
            f"llvm-profdata overlap gave no readable `Edge profile overlap` line: {' '.join((p.stderr or p.stdout).split())[:200]}"
        )
    return float(m.group(1))


def function_overlap(a: dict, b: dict) -> float:
    """The overlap in percent of the share of all the counts each function holds in two profile tables (`profile_table`): the sum over the
    functions of the smaller share. 100 = the same functions are as hot in both."""
    wa, wb = sum(w for _, w in a.values()), sum(w for _, w in b.values())
    if wa <= 0 or wb <= 0:
        return 0.0
    return 100.0 * sum(min(w / wa, b[n][1] / wb) for n, (_, w) in a.items() if n in b)


PROFILE_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
sys.path.extend(p for p in json.loads(sys.argv[5] if len(sys.argv) > 5 else "[]") if p not in sys.path)
import numpy as np
from backend_codegen import CodegenForest
f = CodegenForest(sys.argv[2])
x = np.load(sys.argv[3])
if sys.argv[4] == "main":
    f.predict(x, 1)
else:
    for call in (
        lambda: f.predict(x, 2),
        lambda: f.predict(x, 1, dtype=np.float64),
        lambda: f.predict(x, 1, base_margin=np.zeros((len(x), f.n_groups))),
    ):
        try:
            call()
        except ValueError:
            pass
"""


def parent_site_paths() -> list[str]:
    """The folders this process imports numpy from: its site-packages (and user site) directories that are on its `sys.path`, and the folder numpy itself
    was found in. `python -I` (isolated: no PYTHON* variables, no user site, no current folder) cannot see a numpy that lives in the user site, so the profile
    run is told these folders EXPLICITLY, appended after the standard library: isolation from the environment and the working folder stays, and the
    run imports exactly the packages the parent imports."""
    import site

    # site.getsitepackages does not exist in every virtualenv (older virtualenv builds ship a site.py without it)
    try:
        system_sites = list(site.getsitepackages())
    except AttributeError:
        system_sites = []
    try:
        user_site = site.getusersitepackages()
    except AttributeError:
        user_site = None
    sites = {*system_sites, *([user_site] if user_site else [])}
    found = [p for p in sys.path if p in sites and Path(p).is_dir()]
    numpy_home = str(Path(np.__file__).resolve().parent.parent)
    return [*dict.fromkeys([*found, numpy_home])]


def profile_run(inst: Path, rows: Path, raw: Path, kind: str) -> None:
    """Runs the instrumented library over the rows of an .npy in a process of its own: its counters become one raw profile in `raw`.
    `main` = what the library does with a batch (one thread, so that no counter is lost to a race). `cover` = the other ways it is called in
    production, on a few blocks of the rows: two threads, float64 answers, a base margin. They run the same tree functions, but the driver around
    them exists once per answer type and the thread dispatch only on this path, and a function the profile never ran would be tuned for size."""
    raw.mkdir()
    env = {**os.environ, "LLVM_PROFILE_FILE": str(raw / "%p.profraw"), "PYTHONPATH": str(HERE)}
    paths = json.dumps(parent_site_paths())
    code, msg = run([sys.executable, "-I", "-c", PROFILE_SCRIPT, str(HERE), str(inst), str(rows), kind, paths], env)
    if code or not list(raw.glob("*.profraw")):
        raise BuildFailed(f"profile run failed: {msg}")


def check_profile(watch: Watch, merged: Path, plan: list, mode: str) -> dict:
    """The profile-use build must have used the profile for every HOT function: the fewest functions that together hold 90 % of the counts of the
    profile (the generated trees, whose code runs once per row block, in practice). rustc applies a profile that does not match its code silently
    (exit 0, a function whose control flow changed just loses its counts), so the build fails in one line when
      * the profile has no counts at all (the profile run did not execute the code),
      * a hot generated function (`c<chunk>::b<tree>` / `t<tree>`) is not in this code: the profile belongs to another model,
      * LLVM said, while building, that a hot function lost its profile (a hash mismatch, or no data under a name the profile knows the function
        by, as when a codegen unit is numbered differently), or said something this check cannot read.
    LLVM's warnings about any other function, one the profile does not hold or weighs nothing (code that never ran in the profile run), are cold:
    counted in the result, never fatal. Returns {"generated": generated functions with profile data, "hot": hot functions, "cold": cold warnings}."""
    want = {(f"c{c}", ("b" if mode == "block" else "t") + str(k)) for c, first, trees in plan for k in range(first, first + len(trees))}
    table = profile_table(merged)
    if not table or sum(w for _, w in table.values()) <= 0:
        raise BuildFailed(
            "the profile has no counts: the profile run did not execute the generated code; the profile-guided build was not kept"
        )
    hot = hot_functions(table)
    hot_keys = {function_key(n) for n in hot}
    stale = sorted(k for k in hot_keys if len(k) == 2 and re.fullmatch(r"c\d+", k[0]) and re.fullmatch(r"[bt]\d+", k[1]) and k not in want)
    if stale:
        raise BuildFailed(
            f"the profile does not belong to this code: {len(stale):,} of its {len(hot):,} hottest functions are not in it "
            f"(first: {stale[0][0]}::{stale[0][1]}); the profile-guided build was not kept"
        )

    def lost(line: str) -> bool:
        m = Watch.named.search(line)
        return m is None or function_key(m.group(1)) in hot_keys  # a line this check cannot read counts against the build

    bad = [ln for ln in watch.lines if lost(ln)]
    watch.cold = watch.n - len(bad)
    if bad:
        raise BuildFailed(
            f"the profile does not match this code: rustc dropped the profile of {len(bad):,} of its {len(hot):,} hottest functions "
            f"({bad[0][:200]}); the profile-guided build was not kept"
        )
    have = {function_key(n) for n in table}
    generated = len(want & have) + sum(1 for c, _, _ in plan if (f"c{c}",) in have) + any("predict_blocks" in n for n in table)
    return {"generated": generated, "hot": len(hot), "cold": watch.cold}


_DLOPENED: dict[tuple[int, int], int] = {}  # (device, inode) of a library file -> its dlopen handle
_DLOPEN_NAMES: set[str] = set()  # the real paths dlopen has been given
_DLOPEN_LOCK = threading.Lock()


def dlopen_current(path: str) -> ctypes.CDLL:
    """The library at `path` as the file is NOW. dlopen matches a name against the libraries already mapped BEFORE it looks at the file, so a
    model compiled again to the same path (`replace_file` writes a new inode) and loaded in the same process would answer with the model it
    replaced, for good. The first library of a path is opened by its name; a file with another identity is opened through a throw-away hard
    link (a copy on another file system) whose new name sends dlopen to the file, and every library is remembered by the identity of its
    file, so the same file is mapped once. The libraries already loaded stay mapped for the forests that hold them."""
    real = os.path.realpath(path)
    st = os.stat(real)
    ident = (st.st_dev, st.st_ino)
    with _DLOPEN_LOCK:
        if ident in _DLOPENED:
            return ctypes.CDLL(real, handle=_DLOPENED[ident])
        if real not in _DLOPEN_NAMES:
            lib = ctypes.CDLL(real)  # not the bare name: dlopen would not look in the current folder
        else:
            folder = tempfile.mkdtemp(prefix="qf-load-")
            link = os.path.join(folder, f"{os.path.basename(real)}.{os.getpid()}.{len(_DLOPENED)}.so")
            try:
                try:
                    os.link(real, link)
                except OSError:
                    shutil.copyfile(real, link)
                lib = ctypes.CDLL(link)
            finally:
                shutil.rmtree(folder, ignore_errors=True)
        _DLOPEN_NAMES.add(real)
        _DLOPENED[ident] = lib._handle
        return lib


class CodegenForest(NotCopyable):
    """Predictor for a generated `.so`."""

    def __init__(self, path):
        self.path = Path(path)
        target = check_target(self.path)  # before dlopen: a library for a CPU this one is not must not get to run
        self.target_cpu = target[0] if target else None
        self._lib = dlopen_current(os.path.abspath(self.path))
        lib = self._lib
        lib.tfl_cg_info.argtypes = [ctypes.c_void_p]
        plain = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint32]
        for name, args in (
            ("tfl_cg_predict", plain),
            ("tfl_cg_predict_margin", [*plain, ctypes.c_void_p]),
            ("tfl_cg_predict_f64", [*plain, ctypes.c_void_p]),
        ):
            getattr(lib, name).restype, getattr(lib, name).argtypes = ctypes.c_int, args
        lib.tfl_cg_margin_scale.restype, lib.tfl_cg_margin_scale.argtypes = ctypes.c_double, []
        self.margin_scale = lib.tfl_cg_margin_scale()
        info = (ctypes.c_uint32 * 7)()
        self._lib.tfl_cg_info(info)
        self.n_features, self.n_groups, self.n_folds, self.n_trees, self.n_nodes, self.objective, self.average = info
        self.engine_used = None
        self.segments, self.output_kind, self.n_outputs = self._read_outputs()

    def _read_outputs(self) -> tuple[list[int] | None, str, int]:
        """(segments, output kind, number of outputs), read as the flat format reads them (DISENO A1, R-SEG): a library with no segment table (built
        before 1.2, or from scalar trees) is one output, a regression when it is one identity group and a classification otherwise."""
        try:
            seg_fn = self._lib.tfl_cg_segments
        except AttributeError:
            n_seg = 0
        else:
            seg_fn.restype, seg_fn.argtypes = ctypes.c_uint32, [ctypes.c_void_p, ctypes.c_size_t]
            n_seg = seg_fn(None, 0)
        if not n_seg:
            return None, "regression" if self.objective == OBJECTIVE["identity"] and self.n_groups == 1 else "classification", 1
        buf = (ctypes.c_uint32 * (n_seg + 1))()
        if seg_fn(buf, n_seg + 1) != n_seg:
            raise ValueError(f"{self.path}: the output segments changed while they were read")
        segments = [int(v) for v in buf]
        regression = self.objective == OBJECTIVE["identity"] and all(b - a == 1 for a, b in pairwise(segments))
        return segments, "regression" if regression else "classification", n_seg

    def predict(self, x, threads: int | None = None, base_margin=None, dtype=np.float32) -> np.ndarray:
        """`threads`: None (default) = every usable core from `evaluator.PARALLEL_MIN_ROWS` rows on, one below; an integer forces that many.
        `base_margin`: (rows, groups) or (rows,) numbers added to every model's raw score before the objective's transform (float64).
        `dtype=np.float64`: the answer without the final rounding to float32 (a float32 forest's result widened)."""
        x = float32_rows(x)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"expected rows of {self.n_features} features")
        threads = resolve_threads(threads, len(x))
        dtype = output_dtype(dtype)
        out = np.empty((len(x), self.n_groups), dtype=dtype)
        if dtype == np.float64:
            bm = None if base_margin is None else margin_array(base_margin, len(x), self.n_groups, self.margin_scale)
            bad = self._lib.tfl_cg_predict_f64(x.ctypes.data, len(x), out.ctypes.data, threads, None if bm is None else bm.ctypes.data)
        elif base_margin is not None:
            bm = margin_array(base_margin, len(x), self.n_groups, self.margin_scale)
            bad = self._lib.tfl_cg_predict_margin(x.ctypes.data, len(x), out.ctypes.data, threads, bm.ctypes.data)
        else:
            bad = self._lib.tfl_cg_predict(x.ctypes.data, len(x), out.ctypes.data, threads)
        if bad:
            raise RuntimeError("tfl_cg_predict failed")
        return out[:, 0] if self.n_groups == 1 else out

    def split_outputs(self, p):
        """`predict` answer -> one array per output (a classifier: sklearn's `predict_proba` list; a regressor: one column each)."""
        return split_outputs(p, self.n_groups, getattr(self, "segments", None), self._output_kind())

    def argmax_outputs(self, p):
        """`predict` answer of a classifier -> (rows, K) integer decisions (argmax inside each output's segment, p >= 0.5 for a width-1 one)."""
        return argmax_outputs(p, self.n_groups, getattr(self, "segments", None), self._output_kind())

    def _output_kind(self) -> str:
        return getattr(self, "output_kind", "classification" if self.n_groups > 1 else "regression")


class CodegenBackend:
    name = "codegen"
    suffix = ".so"
    supports = FEATURES
    mode = "row"
    pgo = False
    default_block = 64
    opt_level = 3

    def compile(self, forest: Forest, out: Path, options: dict) -> dict:
        """The sources are built in a folder this call creates in the system temp folder (`qf-codegen-XXXX`) and removes when it ends: a
        clean path whatever the output path is called (rustc and the linker read a leading `-` and a `%` in a path as an option and an
        escape) and a folder of the user's is never touched. `keep_src` copies it next to the output as `<output stem>-src-XXXX` (the
        result names it as `src_dir`); only the finished library is copied to the output."""
        out = Path(out)
        check_vector(forest)  # a negative group index must never reach the generator (numpy would write the last group)
        if (values := vector_table_values(forest)) > CODEGEN_MAX_VECTOR_VALUES:
            raise ValueError(
                f"the leaf vectors hold {values:,} values and generated code holds at most {CODEGEN_MAX_VECTOR_VALUES:,} (rustc keeps the "
                f"table in memory while it builds); use the flat format (--backend flat), which maps the table from the file"
            )
        settings = resolve_compile(options)  # a mistyped or contradictory setting is refused before anything else
        if settings["pgo"] == "on" and not self.pgo:
            raise ValueError(
                f"--pgo on asks for a profile-guided build and {self.name} is not one: use --backend codegen-pgo (or best, with --data)"
            )
        if settings["pgo"] == "off" and self.pgo:
            raise ValueError(f"--pgo off and --backend {self.name} contradict each other: use --backend codegen-block (or best)")
        if settings["cspgo"] == "on" and (not self.pgo or settings["pgo"] == "off"):
            raise ValueError(
                "--cspgo on is a second profile pass on top of a profile-guided build: use --pgo on (or --backend codegen-pgo) with --data"
            )
        if settings["cspgo"] == "on" and settings["unit"] == "single":
            raise ValueError("--cspgo on needs the chunks build (--unit chunks): the one-crate build has no linker plug-in")
        if (depth := too_deep(forest)) is not None:  # before anything else: no toolchain is needed to know rustc cannot parse it
            raise ValueError(
                f"a tree is {depth:,} levels deep: generated code holds at most {MAX_TREE_DEPTH:,} (rustc's parser overflows its stack "
                f"on deeper nesting); use the flat format (--backend flat)"
            )
        rustc = str(toolchain_bin() / "rustc")  # first: with no rustc this refuses before anything is written
        cpu = resolve_target_cpu(options)
        feats = target_features(rustc, cpu)  # an unknown CPU is refused here, before a folder is made
        if self.pgo and (missing := missing_features(feats)):  # the profile pass RUNS the code it builds: refused before the first build
            more = f" and {len(missing) - 5} more" if len(missing) > 5 else ""
            raise BuildFailed(
                f"codegen-pgo profiles the code by running it, and this CPU lacks {', '.join(missing[:5])}{more}, which the {cpu!r} target "
                "needs: use --backend codegen-block (it is not run while it builds), or build where the profile can run"
            )
        if self.pgo:
            llvm_profdata()  # a profile of another LLVM version is rejected after a whole instrumented build: refused before the first one
            if options.get("pgo_rows") is not None and (n := profile_row_count(options["pgo_rows"])) is not None and n < PROFILE_MIN_ROWS:
                raise too_few_rows(n)
        out.parent.mkdir(parents=True, exist_ok=True)
        src = Path(tempfile.mkdtemp(prefix="qf-codegen-"))
        keep = bool(options.get("keep_src"))
        try:
            if "%" in str(src):  # the linker's driver reads it as a spec escape in an output name
                raise ValueError(f"the temporary folder {src} has a % in its path: set TMPDIR to another folder to build generated code")
            info = self._compile_in(forest, out, options, rustc, src, cpu, feats)
        except BaseException:
            if keep:
                self._keep_sources(src, out)
            shutil.rmtree(src, ignore_errors=True)
            raise
        if keep:
            info["src_dir"] = self._keep_sources(src, out)
        shutil.rmtree(src, ignore_errors=True)
        return info

    @staticmethod
    def _keep_sources(src: Path, out: Path) -> str:
        dest = Path(tempfile.mkdtemp(prefix=f"{out.stem}-src-", dir=out.parent))
        shutil.copytree(src, dest, dirs_exist_ok=True)
        return str(dest)

    def _compile_in(self, forest: Forest, out: Path, options: dict, rustc: str, src: Path, cpu: str, feats: list[str]) -> dict:
        block = int(options.get("block", self.default_block))
        chunk_nodes = int(options.get("chunk_nodes", 150_000))
        want_jobs = int(options.get("jobs", 3))
        opt = int(options.get("opt_level", self.opt_level))
        settings = resolve_compile(options)  # lto, codegen units, inline threshold: the instrumented and the final build get exactly these
        want_units = settings["codegen_units"]
        factor, link_factor = peak_factor(settings), link_peak_factor(settings)
        acc = "f64" if forest.sum64 else "f32"
        exact = bool(options.get("f64_leaves")) and forest.sum64  # the leaves a float32 would round are generated as double literals
        shutil.copyfile(FINISH_RS, src / "finish.rs")
        # the RAM plan: fewer jobs, then fewer units, then smaller chunks, until the estimated peak fits in the RAM that is free
        avail, first_chunk_nodes = available_ram_mb(options), chunk_nodes
        unit, unit_note = settings["unit"], None
        # ONE rustc process for the whole forest: if it does not fit, the chunks are built (decided once, before the instrumented build)
        if unit == "single":
            total = sum(tree_weight(t) for m in forest.models for t in m.trees)
            got = plan_single(total, want_units, avail, factor)
            if got is None:
                unit_note = (
                    f"a one-crate build ({describe_compile(settings)}) of {total:,} nodes needs about {rustc_peak_estimate_mb(total, 1, factor):,.0f} MB of RAM "
                    f"in one rustc process and {avail:,.0f} MB are free (the build keeps to {RAM_SHARE:.0%} of that): building in chunks instead (--unit chunks)"
                )
                print(f"RAM: {unit_note}", file=sys.stderr)
                unit = "chunks"
                # from here on the build is what it really is: the instrumented one gets it too
                settings = name_build({**settings, "unit": "chunks"})
                factor = peak_factor(
                    settings
                )  # a chunk crate runs no LTO: its jobs are lighter than the one crate's (the link keeps link_factor)
        if unit == "single":
            plan, chunks_per_fold = split_chunks(forest, chunk_nodes)  # the chunks are only the modules of the one crate
            sizes = [sum(tree_weight(t) for t in trees) for _, _, trees in plan]
            fit = (1, got)
        else:
            while True:
                plan, chunks_per_fold = split_chunks(forest, chunk_nodes)
                sizes = [sum(tree_weight(t) for t in trees) for _, _, trees in plan]
                fit = plan_build(sizes, want_jobs, want_units, avail, factor, link_factor)
                if fit or chunk_nodes // 2 < MIN_CHUNK_NODES:
                    break
                chunk_nodes //= 2
        if fit is None:
            raise NotEnoughRam(
                f"generated code ({describe_compile(settings)}) needs about {rustc_peak_estimate_mb(max(sizes), 1, factor):,.0f} MB of RAM for one rustc job "
                f"and {avail:,.0f} MB are free (the build keeps to {RAM_SHARE:.0%} of that and never swaps): free memory, "
                + (
                    "build lighter (--compile-preset 1.0 is QF 1.0's build, --lto off drops the biggest part), "
                    if max(factor, link_factor) > 1
                    else ""
                )
                + "or use the flat format (--backend flat)"
            )
        jobs, units = fit
        single = unit == "single"
        cs, cs_note = False, None  # CS-PGO: on whenever the build is profile-guided and the toolchain can do it, unless switched off
        if self.pgo and settings["cspgo"] != "off":
            if (why := cspgo_unavailable(settings, unit, rustc)) is None:
                cs = True
            elif settings["cspgo"] == "on":
                raise ValueError(f"--cspgo on: {why}; use --cspgo off for plain PGO")
            else:
                cs_note = f"CS-PGO skipped: {why}"
        line = f"RAM: building {'ONE crate ' if single else ''}with {jobs} {'job' if jobs == 1 else 'jobs'} / {units} {'unit' if units == 1 else 'units'}"
        if avail is None:
            line += " (available RAM could not be read)"
        elif single and units != settings["codegen_units"]:
            line += f" (wanted {settings['codegen_units']} units; {avail:,.0f} MB free)"
        elif not single and (jobs, units, chunk_nodes) != (min(want_jobs, len(plan)), want_units, first_chunk_nodes):
            line += (
                f" (wanted {want_jobs} / {want_units}; {avail:,.0f} MB free"
                + (f", chunks of {chunk_nodes:,} nodes" if chunk_nodes != first_chunk_nodes else "")
                + ")"
            )
        print(line, file=sys.stderr)
        # one crate: the codegen units are made of modules (rustc makes one per module, then merges them down to the wanted number), so each chunk is spread over
        # about units / chunks modules; separate crates: over `units` modules each
        modules = max(1, -(-units // len(plan))) if single else units
        for c, first, trees in plan:
            (src / f"c{c}.rs").write_text(chunk_source(trees, first, c, forest.n_features, forest.groups, self.mode, acc, exact, modules))
        (src / "lib.rs").write_text(
            driver_source(forest, chunks_per_fold, block, acc, f"{TARGET_MARK}|{cpu}|{ARCH}|{','.join(feats)}|", single=single)
        )
        # The shape of the chunk builds: the instrumented and the final build get exactly these flags (a profile only matches code built the same way)
        shape_flags = chunk_flags(opt, settings, units)
        # The driver (the hot loop over row blocks): one codegen unit is one everywhere, so its link gets one too; QF 1.0's `fast` build leaves rustc's
        # own defaults (the state it was measured in); fat LTO and the inline threshold go to the link as well. The same in the instrumented and in the final build.
        link_shape = driver_flags(settings)
        link_shape_plugin = [
            x for i, x in enumerate(link_shape) if x != "lto=fat" and not (x == "-C" and link_shape[i + 1] == "lto=fat")
        ]  # the linker does the LTO
        base_flags = [
            "--edition",
            "2021",
            "-C",
            "panic=abort",
            "-C",
            "debuginfo=0",
            "-C",
            "relocation-model=pic",
            "-C",
            f"target-cpu={cpu}",
        ]
        pgo_dir = src / "pgo"
        flags_gen = ["-C", f"profile-generate={pgo_dir}"] if self.pgo else []
        peaks: list[float] = []  # MB, one per rustc of this build

        def no_fallback(target):
            """The instrumented build has the final's shape or there is no profile-guided build: it never quietly takes another shape."""
            if target != "inst":
                # the final build: a fat-LTO link that fails or runs out of memory has a lighter way out
                return (
                    " (try --compile-preset 1.0, QF 1.0's build without fat LTO, or --lto off)"
                    if settings["lto"] == "fat" or settings["inline_threshold"]
                    else ""
                )
            return (
                f" (the instrumented build has exactly the final build's flags, {describe_compile(settings)}, as a profile only matches code "
                "built the same way; it does not fall back to another shape: build with --pgo off, or with --compile-preset 1.0)"
            )

        def build_one_crate(extra, target, watch=None):
            """unit = single: ONE rustc builds, optimises and links the driver with the chunks as its modules (fat LTO, the inliner and the units all inside
            rustc itself, no linker plug-in: nothing for GNU ld to choke on). The instrumented and the final build get exactly the same flags."""
            watching = {} if watch is None else {"watch": watch}
            so = src / f"model.{target}.so"
            cmd = [rustc, *base_flags, "--crate-type", "cdylib", *shape_flags, *extra]
            if not self.pgo or target == "final":
                cmd += ["-C", "strip=symbols"]
            code, msg, mb = run_measured([*cmd, "-o", str(so), str(src / "lib.rs")], **watching)
            peaks.append(mb)
            if code:
                raise BuildFailed(f"rustc failed on the one-crate build: {msg}{no_fallback(target)}")
            return so

        def build(extra, target, watch=None, lpl=False, link_extra=(), strip=None):
            lpl_flags = (
                ["-C", "linker-plugin-lto"] if lpl else []
            )  # CS-PGO: the chunks stay bitcode and the linker (rust-lld) optimises them
            if single:
                return build_one_crate(extra, target, watch)
            watching = {} if watch is None else {"watch": watch}  # a build without a profile counts nothing
            objects: dict[int, list[Path]] = {}

            def one(c):
                odir = (
                    src / f"o-{target}-c{c}"
                )  # `-o` cannot name the several objects of several codegen units: rustc writes them into a folder
                odir.mkdir()
                cmd = [rustc, *base_flags, "--crate-type", "lib", "--emit", "obj", *shape_flags, *lpl_flags, *extra]
                code, msg, mb = run_measured([*cmd, "--out-dir", str(odir), str(src / f"c{c}.rs")], **watching)
                peaks.append(mb)
                if code:
                    raise BuildFailed(f"rustc failed on chunk {c}: {msg}{no_fallback(target)}")
                objects[c] = object_files(odir)

            with ThreadPoolExecutor(jobs) as ex:
                list(ex.map(one, [c for c, _, _ in plan]))
            link = [
                rustc,
                *base_flags,
                "--crate-type",
                "cdylib",
                "-C",
                "opt-level=3",
                *(link_shape_plugin if lpl else link_shape),
                *lpl_flags,
                *extra,
                *link_extra,
            ]
            if strip if strip is not None else (not self.pgo or target == "final"):
                link += ["-C", "strip=symbols"]
            for c, _, _ in plan:
                for o in objects[c]:
                    link += ["-C", f"link-arg={o}"]
            so = src / f"model.{target}.so"
            code, msg, mb = run_measured([*link, "-o", str(so), str(src / "lib.rs")], **watching)
            peaks.append(mb)
            if code:
                raise BuildFailed(f"rustc failed linking: {msg}{no_fallback(target)}")
            return so

        if self.pgo:
            rows = options.get("pgo_rows")
            if rows is None:
                raise ValueError("codegen-pgo needs options['pgo_rows'] (an .npy of representative rows)")
            n_rows = int(np.load(rows, mmap_mode="r").shape[0])
            if n_rows < PROFILE_MIN_ROWS:
                raise too_few_rows(n_rows)  # compile() refuses it before the first build; this guards a caller that skipped that
            inst = build(flags_gen, "inst")
            # two disjoint halves (a seeded shuffle: the rows may come ordered) are profiled apart and then merged, so that the profile's
            # agreement with itself is measured for the price of the merge; a third small pass covers the other ways the library is called
            order = np.random.default_rng(0).permutation(n_rows)
            all_rows = np.load(rows, mmap_mode="r")
            parts = (("a", order[: n_rows // 2]), ("b", order[n_rows // 2 :]), ("c", order[: int(PROFILE_COVER_FACTOR * block)]))
            profdata = []
            for name, idx in parts:
                part_rows = src / f"rows-{name}.npy"
                np.save(part_rows, np.ascontiguousarray(all_rows[np.sort(idx)]))
                profile_run(inst, part_rows, src / f"raw-{name}", "cover" if name == "c" else "main")
                profdata.append(src / f"{name}.profdata")
                merge_profiles(profdata[-1], (src / f"raw-{name}").glob("*.profraw"))
            merged = src / "merged.profdata"
            merge_profiles(merged, profdata)
            overlap = edge_overlap(profdata[0], profdata[1])
            func_overlap = function_overlap(
                profile_table(profdata[0]), profile_table(profdata[1])
            )  # reported, not gated: 94.9 % or more on every forest measured
            if overlap < PROFILE_MIN_OVERLAP:
                raise ProfileUnreliable(
                    f"two halves of the {n_rows:,} profile rows give profiles that agree on {overlap:.1f} % of the edge counts "
                    f"(at least {PROFILE_MIN_OVERLAP:g} % is required): the rows are too few or too unlike one another to be the workload; "
                    "give --data with more, representative rows"
                )
            # warn about every function without profile data: the output of a build that worked is not kept, but the hot ones are checked
            cs_info = {}
            if cs:
                try:
                    so, watch, cs_info = cspgo_stages(build, rustc, src, merged, parts)
                except BuildFailed as e:
                    if settings["cspgo"] == "on":
                        raise
                    cs, cs_note = False, f"CS-PGO failed, built plain PGO: {e}"
                    print(f"RAM/CS: {cs_note}", file=sys.stderr)
            if not cs:
                watch = Watch()
                so = build(["-C", f"profile-use={merged}", "-C", "llvm-args=-pgo-warn-missing-function=true"], "final", watch)
            gate = check_profile(watch, merged, plan, self.mode)
            pgo_functions, hot_functions_n, cold_functions = gate["generated"], gate["hot"], gate["cold"]
        else:
            so = build([], "final")
            pgo_functions, hot_functions_n, cold_functions = None, 0, 0
            cs_info = {}
        replace_file(out, src=so)
        src_mb = round(sum(f.stat().st_size for f in src.glob("*.rs")) / 1e6, 1)
        return {
            "artifact": str(out),
            "bytes": out.stat().st_size,
            "nodes": forest.n_nodes,
            "trees": forest.n_trees,
            "rustc_peak_mb": round(max(peaks)),
            "chunks": len(plan),
            "src_mb": src_mb,
            "target_cpu": cpu,
            "target_features": len(feats),
            "shape": settings["shape"],
            "compile_preset": settings["preset"],
            "build": describe_compile(settings),
            "lto": settings["lto"],
            "unit": unit,
            **({"unit_fallback": unit_note} if unit_note else {}),
            "codegen_units": units,
            "inline_threshold": settings["inline_threshold"],
            "pgo": bool(self.pgo),
            "cspgo": cs,
            **cs_info,
            **({"cspgo_note": cs_note} if cs_note else {}),
            "peak_factor": round(factor, 2),
            "jobs": jobs,
            **({"profiled_functions": pgo_functions} if pgo_functions else {}),
            **({"profile_hot_functions": hot_functions_n} if pgo_functions else {}),
            **({"profile_cold_unmatched": cold_functions} if cold_functions else {}),
            **(
                {"profile_rows": n_rows, "profile_overlap": round(overlap, 2), "profile_function_overlap": round(func_overlap, 2)}
                if self.pgo
                else {}
            ),
            "note": f"build {describe_compile(settings)}{' (preset ' + settings['preset'] + ')' if settings['preset'] != 'custom' else ''}"
            f"{', shape ' + settings['shape'] if settings['shape'] != 'custom' else ''}, block {block}, opt {opt}"
            f"{' (overflow checks and debug assertions off, as at every other level)' if opt == 0 else ''}, target {cpu}, "
            f"{len(plan)} chunks{' as modules of one crate' if single else ''}, {units} codegen units, jobs {jobs}"
            + (" (one-crate build did not fit the RAM: chunks)" if unit_note else "")
            + (", CS-PGO" if cs else "")
            + (f", profile on {pgo_functions:,} functions" if pgo_functions else "")
            + (f" ({cold_functions:,} cold functions without profile data)" if cold_functions else ""),
        }

    def load(self, artifact: Path, **options):
        return CodegenForest(artifact)


class CodegenBlockBackend(CodegenBackend):
    name = "codegen-block"
    mode = "block"
    default_block = 512


class CodegenPgoBackend(CodegenBlockBackend):
    name = "codegen-pgo"
    pgo = True
