"""The hostile-input matrix: every command, fed every kind of input that cannot be used, must refuse it the way docs/REFERENCE.md says.

For each entry point (prune, optimize, check, compile, choose, compare) and each damaged input (a model, a data array, an artifact, an
output path, a numeric option) the command must end with the documented exit code (1 for an input, 3 for compile and choose, 2 for the
option parser), ONE line on stderr, no traceback, no native backtrace, no signal and no hang. A damage that is meant to be accepted
(`clean`) may succeed or refuse, but only in that same clean way.

The cells are generated from two small lists, ENTRY_POINTS and DAMAGES; a cell is an entry point that has the damaged slot, times each
library the damage is tried on. Commands run in one worker process (this file started with --worker) that is restarted when a cell kills
it, so that a signal or a hang is reported as a failing cell instead of ending the run.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
FIX = HERE / "fixtures"
CELL_SECONDS = 180
HAS_YDF = importlib.util.find_spec("ydf") is not None


def has_rustc() -> bool:
    """Whether the generated-code backends can build here (the same lookup they make)."""
    sys.path.insert(0, str(SCRIPTS))
    import backend_codegen

    try:
        backend_codegen.toolchain_bin()
    except FileNotFoundError:
        return False
    return True


HAS_RUSTC = has_rustc()
N_FEATURES, N_ROWS = 5, 400


# ---------------------------------------------------------------------------------------------------- the worker
def worker() -> None:
    """Reads one JSON cell per line from stdin ({"command", "argv"}), runs that command's `main()` in this process with fd 2 redirected to a
    file (so that native writes are seen too) and answers one JSON line {"code", "stderr"}. An uncaught exception is reported as Python
    would: a traceback on stderr and exit 1."""
    import importlib
    import traceback

    sys.path.insert(0, str(SCRIPTS))
    sys.dont_write_bytecode = True
    reply = os.fdopen(os.dup(1), "w")
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    sys.stdout = open(os.devnull, "w")
    for line in sys.stdin:
        cell = json.loads(line)
        saved = os.dup(2)
        with tempfile.TemporaryFile() as capture:
            os.dup2(capture.fileno(), 2)
            code = 0
            try:
                sys.argv = [cell["command"], *cell["argv"]]
                importlib.import_module(cell["command"]).main()
            except SystemExit as e:
                if e.code is None:
                    code = 0
                elif isinstance(e.code, int):
                    code = e.code
                else:
                    print(e.code, file=sys.stderr)
                    code = 1
            except BaseException:
                traceback.print_exc()
                code = 1
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
            capture.seek(0)
            err = capture.read().decode("utf-8", "replace")
        reply.write(json.dumps({"code": code, "stderr": err}) + "\n")
        reply.flush()


class Worker:
    """The worker process: `run(cell)` -> (code, stderr); code is "hang" when the cell took too long and a negative number for a signal."""

    def __init__(self):
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            [sys.executable, "-s", str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(SCRIPTS), os.environ.get("PYTHONPATH")]))},
        )

    def run(self, command: str, argv: list[str]):
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        self.proc.stdin.write(json.dumps({"command": command, "argv": argv}) + "\n")
        self.proc.stdin.flush()
        ready, _, _ = select.select([self.proc.stdout], [], [], CELL_SECONDS)
        if not ready:
            self.proc.kill()
            self.proc.wait()
            self.close_pipes()
            return "hang", ""
        reply = self.proc.stdout.readline()
        if not reply:
            code = self.proc.wait()
            self.close_pipes()
            return code, ""
        out = json.loads(reply)
        return out["code"], out["stderr"]

    def close_pipes(self):
        for pipe in (self.proc.stdin, self.proc.stdout):
            with contextlib.suppress(OSError):
                pipe.close()
        self.proc = None

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.stdout.close()
            self.proc.wait(timeout=60)


# --------------------------------------------------------------------------------------------- the two small lists
@dataclass
class Slots:
    """What a command line is made of. A damage replaces ONE of them (or adds options); the others stay valid."""

    model: str
    data: str
    artifact: str
    out: str
    lib: str = "xgb"
    opts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Entry:
    name: str
    command: str  # the script (module) whose main() runs
    codes: tuple[int, ...]  # the exit code of a refused input, as the exit-code table of docs/REFERENCE.md gives it
    argv: object  # Slots -> list[str]
    slots: frozenset  # the slots this command reads: "model", "data", "artifact", "out", "opts"


def regression_flag(s: Slots) -> list[str]:
    """The regression models are judged by the regression bar, which is the only place the truth `y` is read."""
    return ["--regression", "raw"] if s.lib == "xgbr" else []


def _npy_huge_header() -> bytes:
    """A .npy of 64 bytes of data whose header claims a (10^6, 10^6) float32 array: numpy would try to allocate 4 TB."""
    import struct

    h = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1000000, 1000000), }"
    h += b" " * (64 - (10 + len(h) + 1) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(h)) + h + b"\0" * 64


NPY_HUGE_HEADER = _npy_huge_header()


def _npz_flipped(c) -> str:
    """A compressed archive (np.savez_compressed) with one byte of its member flipped: zlib.error when read."""
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, X=c.x)
    raw = bytearray(buf.getvalue())
    raw[100] ^= 0xFF
    return c.write("flipped.npz", bytes(raw))


def _npz_unsupported_method(c) -> str:
    """An archive whose member claims zip compression method 99: zipfile raises NotImplementedError when it is read."""
    import io
    import struct
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("X.npy", b"")
    raw = bytearray(buf.getvalue())
    for sig, off in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        i = raw.find(sig)
        raw[i + off : i + off + 2] = struct.pack("<H", 99)
    return c.write("method99.npz", bytes(raw))


def _npy_unbalanced_header() -> bytes:
    """A .npy whose header dictionary is never closed: numpy's header filter raises tokenize.TokenError."""
    import struct

    h = b"{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4, }"
    h += b" " * (64 - (10 + len(h) + 1) % 64) + b"\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(h)) + h + b"\0" * 48


def _artifact_with_choice(c, name: str, body: bytes) -> str:
    """A valid compiled artifact copied under a new name with a damaged `.choice.json` next to it."""
    path = c.write(name, c.artifacts["xgb"].read_bytes())
    c.write(name + ".choice.json", body)
    return path


ENTRY_POINTS = [
    Entry(
        "prune",
        "prune",
        (1,),
        lambda s: [s.model, s.data, s.out, "--keep", "50%", *regression_flag(s), *s.opts],
        frozenset("model data out opts".split()),
    ),
    Entry(
        "optimize",
        "optimize",
        (1, 3),
        lambda s: [s.model, "--data", s.data, "--keep", "50%", "--min-agreement", "1", "--out", s.out, *regression_flag(s), *s.opts],
        frozenset("model data out opts".split()),
    ),
    Entry(
        "optimize keeping all",
        "optimize",
        (1, 3),
        lambda s: [s.model, "--data", s.data, "--keep", "100%", "--out", s.out, *regression_flag(s), *s.opts],
        frozenset("model data out opts".split()),
    ),
    Entry(
        "optimize compile verify",
        "optimize",
        (1, 3),
        lambda s: [
            s.model,
            "--data",
            s.data,
            "--keep",
            "50%",
            "--min-agreement",
            "1",
            "--compile",
            "--verify",
            "--backend",
            "flat",
            "--out",
            s.out,
            *s.opts,
        ],
        frozenset("model data out opts".split()),
    ),
    Entry(
        "check",
        "check",
        (1, 3),
        lambda s: [s.model, s.artifact, "--data", s.data, "--random", "200", *s.opts],
        frozenset("model data artifact opts".split()),
    ),
    Entry("compile", "compile", (3,), lambda s: [s.model, s.out, "--backend", "flat", *s.opts], frozenset("model out opts".split())),
    *(
        [
            Entry(
                "compile codegen",
                "compile",
                (3,),
                lambda s: [s.model, s.out, "--backend", "codegen-block", *s.opts],
                frozenset("model out opts".split()),
            )
        ]
        if HAS_RUSTC
        else []
    ),
    Entry("structure", "structure", (1,), lambda s: [s.model, s.out, *s.opts], frozenset("model out opts".split())),
    Entry(
        "choose",
        "choose",
        (3,),
        lambda s: [s.model, s.out, "--data", s.data, "--only", "flat", *s.opts],
        frozenset("model data out opts".split()),
    ),
    Entry(
        "compare",
        "compare",
        (1,),
        lambda s: [s.model, "--data", s.data, "--work", s.out, "--backends", "flat", "--random", "200", "--eq-rows", "300", *s.opts],
        frozenset("model data out opts".split()),
    ),
]


@dataclass(frozen=True)
class Damage:
    """`slot`: which input is damaged. `make(ctx, lib)` -> the damaged value (a path, or the list of options to add). `kind`: "refuse" (the
    command must refuse it) or "clean" (it may accept it, but not misbehave). `libs`: the model libraries it is tried on (the damages that
    do not depend on the library use xgboost alone). `only`: restrict it to some entry points (the heavy ones)."""

    name: str
    slot: str
    make: object
    kind: str = "refuse"
    libs: tuple[str, ...] = ("xgb",)
    only: tuple[str, ...] | None = None
    not_root: bool = False  # a file mode that stops a user does not stop root: the cell only runs for a user


# ------------------------------------------------------------------------------------------------ fixtures + makers
class Ctx:
    """The valid pieces every cell is built from (small models of each library, their compiled forests, a data file) and the folder the
    damaged inputs are written to."""

    def __init__(self, root: Path):
        import lightgbm as lgb
        import xgboost as xgb

        sys.path.insert(0, str(SCRIPTS))
        import compile as qf_compile

        self.root = root
        (root / "out").mkdir()  # the output folder every command writes into, whichever cells ran before
        rng = np.random.default_rng(7)
        self.x = rng.normal(size=(N_ROWS, N_FEATURES)).astype(np.float32)
        self.x[rng.random(self.x.shape) < 0.05] = np.nan
        z = np.nan_to_num(self.x)
        self.y_class = (z[:, 0] + z[:, 4] > 0).astype(float)
        self.y_reg = (z[:, 0] * 2 + z[:, 3]).astype(np.float64)
        self.models: dict[str, Path] = {}
        self.artifacts: dict[str, Path] = {}
        mdir = root / "models"
        mdir.mkdir()
        xp = {"max_depth": 3, "nthread": 1, "verbosity": 0, "seed": 1}
        xgb.train({**xp, "objective": "binary:logistic"}, xgb.DMatrix(self.x, label=self.y_class), 8).save_model(str(mdir / "xgb.ubj"))
        xgb.train({**xp, "objective": "reg:squarederror"}, xgb.DMatrix(self.x, label=self.y_reg), 8).save_model(str(mdir / "xgbr.ubj"))
        for k in range(2):  # two folds of one k-fold ensemble
            (mdir / "kfold").mkdir(exist_ok=True)
            xgb.train({**xp, "objective": "binary:logistic", "seed": k}, xgb.DMatrix(self.x, label=self.y_class), 5).save_model(
                str(mdir / "kfold" / f"fold{k}.ubj")
            )
        lgb.train(
            dict(objective="binary", num_leaves=7, verbose=-1, num_threads=1, seed=1), lgb.Dataset(self.x, self.y_class), 8
        ).save_model(str(mdir / "lgb.txt"))
        # the usual way to train: a DataFrame, so the model stores feature names (the data file is a bare matrix by definition)
        xgb.train(
            {**xp, "objective": "binary:logistic"},
            xgb.DMatrix(self.x, label=self.y_class, feature_names=[f"col_{i}" for i in range(N_FEATURES)]),
            8,
        ).save_model(str(mdir / "xgbn.ubj"))
        self.models = {
            "xgb": mdir / "xgb.ubj",
            "xgbr": mdir / "xgbr.ubj",
            "xgbn": mdir / "xgbn.ubj",
            "kfold": mdir / "kfold",
            "lgb": mdir / "lgb.txt",
        }
        if HAS_YDF:
            import ydf

            cols = {f"f{i}": np.nan_to_num(self.x[:, i]) for i in range(N_FEATURES)}
            ydf.GradientBoostedTreesLearner(label="y", task=ydf.Task.CLASSIFICATION, num_trees=8, max_depth=3).train(
                {**cols, "y": self.y_class.astype(int)}, verbose=0
            ).save(str(mdir / "ydfm"))
            self.models["ydf"] = mdir / "ydfm"
        for lib, m in self.models.items():
            adir = root / "artifacts" / lib
            adir.mkdir(parents=True)
            qf_compile.compile_model(m, adir / "m.flat", backend="flat")
            self.artifacts[lib] = adir / "m.flat"
        self.data = self.save_npz("good.npz", X=self.x, y=self.y_reg)
        self.data_class = self.save_npz("good-class.npz", X=self.x)
        self.n = 0

    def save_npz(self, name: str, **arrays) -> str:
        p = self.root / "data" / name
        p.parent.mkdir(exist_ok=True)
        np.savez(p, **arrays)
        return str(p)

    def fresh(self, name: str) -> Path:
        """A path that does not exist yet, for a command to write to."""
        self.n += 1
        return self.root / "out" / f"{self.n}-{name}"

    def write(self, name: str, data: bytes) -> str:
        p = self.root / "files" / name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(data)
        return str(p)


def data_with(**changes):
    """A data file made from the good one: each keyword replaces (or adds) one array; a callable value gets the good X."""

    def make(ctx, lib):
        arrays = {"X": ctx.x, "y": ctx.y_reg}
        for k, v in changes.items():
            arrays[k] = v(ctx) if callable(v) else v
        return ctx.save_npz(ctx.fresh("data").name + ".npz", **arrays)

    return make


def per_row(name: str, **cases):
    """The damages of one per-row array: each case is a function of the good X returning the damaged array."""
    return [Damage(f"{name} {what}", "data", data_with(**{name: fn}), kind, ("xgbr",)) for what, (fn, kind) in cases.items()]


def raw_file(name: str, content):
    """A damaged file of the given bytes, or built from the model's bytes by `content(bytes) -> bytes`."""

    def make(ctx, lib):
        src = Path(ctx.models[lib])
        blob = src.read_bytes() if callable(content) else content
        return ctx.write(name + src.suffix, content(blob) if callable(content) else blob)

    return make


def lgb_text(change):
    """The LightGBM model of the fixtures with its text changed."""

    def make(ctx, lib):
        return ctx.write(f"lgb{ctx.n}.txt", change(ctx.models["lgb"].read_text()).encode())

    return make


def special(kind: str):
    """A path that is not a regular file: "fifo", "dir" (empty), "missing", "loop" (a link to itself), "denied" (mode 000)."""

    def make(ctx, lib):
        p = ctx.root / "files" / f"{kind}{ctx.fresh(kind).name}.ubj"
        p.parent.mkdir(exist_ok=True)
        if kind == "fifo":
            os.mkfifo(p)
        elif kind == "dir":
            p.mkdir()
        elif kind == "loop":
            p.symlink_to(p.name)
        elif kind == "denied":
            p.write_bytes(b"x")
            p.chmod(0)
        return str(p)

    return make


def non_utf8(make_path):
    """The same file under a name that is not valid UTF-8 (a Latin-1 byte)."""

    def make(ctx, lib):
        src = make_path(ctx, lib)
        name = os.fsdecode(os.fsencode(ctx.root / "files") + b"/caf\xe9" + os.fsencode(ctx.fresh("-").name) + os.fsencode(Path(src).name))
        os.makedirs(os.path.dirname(name), exist_ok=True)
        if Path(src).is_dir():
            shutil.copytree(src, name)
        else:
            shutil.copy(src, name)
        return name

    return make


def good_model(ctx, lib):
    return str(ctx.models[lib])


def good_data(ctx, lib):
    return ctx.data


def deep_path(ctx, lib):
    """A path of 700 bytes (valid: every name is short) that a command is asked to write to."""
    p = ctx.fresh("deep")
    for i in range(7):
        p = p / (f"d{i}" + "x" * 96)
    p.mkdir(parents=True)
    return str(p / "m.flat")


WIDTHS = {"one column too few": N_FEATURES - 1, "one column too many": N_FEATURES + 1, "no columns": 0}
ALL_LIBS = ("xgb", "lgb", "ydf") if HAS_YDF else ("xgb", "lgb")
PRUNING = ("prune", "optimize", "optimize compile verify")  # the entry points that select rows
FILE_OUTPUTS = (
    "compile",
    "compile codegen",
    "choose",
    "structure",
)  # write one artifact file (an old one is replaced), the others write a folder
FOLDER_OUTPUTS = ("prune", "optimize", "optimize keeping all", "optimize compile verify", "compare")
CREATES_FOLDERS = ("prune", "optimize", "optimize keeping all", "optimize compile verify", "choose", "compare")
NOT_SCRATCH = (
    "prune",
    "optimize",
    "optimize keeping all",
    "optimize compile verify",
    "compile",
    "compile codegen",
    "choose",
    "structure",
)  # compare's work folder is scratch
WRITES_FLAT = (
    "optimize compile verify",
    "compile",
    "compile codegen",
    "choose",
    "compare",
)  # the entry points whose output the evaluator has to open again


def width_data(w: int):
    def make(ctx, lib):
        rng = np.random.default_rng(w)
        return ctx.save_npz(f"w{w}-{ctx.fresh('w').name}.npz", X=rng.normal(size=(50, w)).astype(np.float32))

    return make


def column(fn):
    return lambda ctx: fn(ctx)


# ---- the decision rule files: a rule is tried once when it is built, and every answer is checked ---------------------
RULE_ENTRIES = ("prune", "optimize", "optimize keeping all", "check", "choose", "compile", "compare")
RULES = {
    "raises": ("def decide(p):\n    raise KeyError('boom')\n", "refuse"),
    "returns a scalar": ("def decide(p):\n    return 1\n", "refuse"),
    "returns None": ("def decide(p):\n    return None\n", "refuse"),
    "returns a list of the wrong length": ("def decide(p):\n    return [0, 1]\n", "refuse"),
    "returns text": ("def decide(p):\n    return ['a'] * len(p)\n", "refuse"),
    "abstains with NaN": (
        "import numpy as np\ndef decide(p):\n    return np.where(p[..., 0] > 0.9, 1.0, np.where(p[..., 0] < 0.1, 0.0, np.nan))\n",
        "refuse",
    ),
    "returns floats": ("def decide(p):\n    return (p[..., 0] > 0.5) * 1.0\n", "refuse"),
    "written for 2-D input": ("def decide(p):\n    return (p[:, 0] >= 0.5).astype(int)\n", "refuse"),
    "written for 3-D input": ("def decide(p):\n    return (p[:, :, 0] >= 0.5).astype(int)\n", "refuse"),
    "returns a column": ("def decide(p):\n    return (p[..., :1] >= 0.5).astype(int)\n", "refuse"),
    "returns bools": ("def decide(p):\n    return p[..., 0] >= 0.5\n", "clean"),
    "abstains with -1": (
        "import numpy as np\ndef decide(p):\n    return np.where(p[..., 0] > 0.9, 1, np.where(p[..., 0] < 0.1, 0, -1))\n",
        "clean",
    ),
}


def rule_damages():
    for what, (source, kind) in RULES.items():
        yield what, source, kind


DAMAGES: list[Damage] = [
    # ---- the data array X ----
    *[Damage(f"X with {what}", "data", width_data(w), "refuse", ALL_LIBS) for what, w in WIDTHS.items()],
    Damage("npy header claims 10^12 values", "data", lambda c, lib: c.write("huge.npy", NPY_HUGE_HEADER)),
    Damage("npz compressed with a flipped byte", "data", lambda c, lib: _npz_flipped(c)),
    Damage("npz with an unsupported zip method", "data", lambda c, lib: _npz_unsupported_method(c)),
    Damage("npy header never closed", "data", lambda c, lib: c.write("unclosed.npy", _npy_unbalanced_header())),
    Damage("X 1-D", "data", data_with(X=lambda c: c.x[:, 0])),
    Damage("X 3-D", "data", data_with(X=lambda c: c.x[None])),
    Damage("X 0-d", "data", data_with(X=lambda c: np.float32(1.0))),
    Damage("X without rows", "data", data_with(X=lambda c: c.x[:0])),
    Damage("X of text", "data", data_with(X=lambda c: np.array([["a"] * N_FEATURES] * 4))),
    Damage("X of objects", "data", data_with(X=lambda c: np.array([[None] * N_FEATURES] * 4, dtype=object))),
    Damage("X complex", "data", data_with(X=lambda c: c.x.astype(np.complex64))),
    Damage("X with infinity", "data", data_with(X=lambda c: np.where(c.x > 1, np.inf, c.x))),
    Damage("X all NaN", "data", data_with(X=lambda c: np.full_like(c.x, np.nan)), "clean"),
    Damage("X beyond the float32 range", "data", data_with(X=lambda c: c.x.astype(np.float64) * 1e300)),
    Damage("X float64", "data", data_with(X=lambda c: c.x.astype(np.float64)), "clean"),
    # ---- the per-row arrays ----
    *per_row(
        "y",
        **{
            "one NaN": (lambda c: np.r_[np.nan, c.y_reg[1:]], "refuse"),
            "one infinity": (lambda c: np.r_[np.inf, c.y_reg[1:]], "refuse"),
            "of text": (lambda c: np.array(["a"] * len(c.x)), "refuse"),
            "one row short": (lambda c: c.y_reg[:-1], "refuse"),
            "two columns": (lambda c: np.stack([c.y_reg, c.y_reg], 1), "refuse"),
            "3-D": (lambda c: c.y_reg[:, None, None], "refuse"),
            "0-d": (lambda c: np.float64(1), "refuse"),
        },
    ),
    *[
        d
        for name in ("fold", "group", "slice", "eligible")
        for d in per_row(
            name,
            **{
                "one row short": (lambda c: np.zeros(len(c.x) - 1, np.int64), "refuse"),
                "two columns": (lambda c: np.zeros((len(c.x), 2), np.int64), "refuse"),
                "3-D": (lambda c: np.zeros((len(c.x), 1, 1), np.int64), "refuse"),
                "0-d": (lambda c: np.int64(0), "refuse"),
                "of text": (lambda c: np.array(["a"] * len(c.x)), "refuse"),
                "of NaN": (lambda c: np.full(len(c.x), np.nan), "refuse"),
                "of infinity": (lambda c: np.full(len(c.x), np.inf), "refuse"),
            },
        )
    ],
    Damage("slice with values 2 and 9", "data", data_with(slice=lambda c: np.arange(len(c.x)) % 2 * 7 + 2), "refuse", ("xgbr",), PRUNING),
    Damage("slice with a 2 among 0 and 1", "data", data_with(slice=lambda c: np.arange(len(c.x)) % 3), "refuse", ("xgbr",), PRUNING),
    Damage(
        "slice with a 0.7", "data", data_with(slice=lambda c: np.where(np.arange(len(c.x)) % 2, 0.7, 0.0)), "refuse", ("xgbr",), PRUNING
    ),
    Damage("slice with a -2", "data", data_with(slice=lambda c: np.where(np.arange(len(c.x)) % 2, -2, 0)), "refuse", ("xgbr",), PRUNING),
    Damage("slice of floats 0.0 and 1.0", "data", data_with(slice=lambda c: np.arange(len(c.x)) % 2 * 1.0), "clean", ("xgbr",), PRUNING),
    Damage("slice of booleans", "data", data_with(slice=lambda c: np.arange(len(c.x)) % 2 == 1), "clean", ("xgbr",), PRUNING),
    Damage("fold with a 0.5", "data", data_with(fold=lambda c: np.where(np.arange(len(c.x)) % 2, 0.5, 0.0)), "refuse", ("xgbr",), PRUNING),
    Damage("group with a 0.5", "data", data_with(group=lambda c: np.arange(len(c.x)) + 0.5), "refuse", ("xgbr",), PRUNING),
    Damage("group beyond the int64 range", "data", data_with(group=lambda c: np.full(len(c.x), 1e30)), "refuse", ("xgbr",), PRUNING),
    Damage("eligible all false", "data", data_with(eligible=lambda c: np.zeros(len(c.x), bool)), "refuse", ("xgbr",), PRUNING),
    Damage("group of negative ids", "data", data_with(group=lambda c: -np.arange(len(c.x))), "clean", ("xgbr",)),
    Damage("base_margin one row short", "data", data_with(base_margin=lambda c: np.zeros(len(c.x) - 1)), "refuse", ("xgbr",)),
    Damage("base_margin 3-D", "data", data_with(base_margin=lambda c: np.zeros((len(c.x), 1, 1))), "refuse", ("xgbr",)),
    # ---- the data path itself ----
    Damage("data path missing", "data", lambda c, lib: str(c.root / "nosuch.npz")),
    Damage("data path a folder", "data", lambda c, lib: str(c.root)),
    Damage("data path a pipe", "data", special("fifo")),
    Damage("data file of random bytes", "data", lambda c, lib: c.write("junk.npz", os.urandom(300))),
    Damage("data file empty", "data", lambda c, lib: c.write("empty.npz", b"")),
    Damage("data file truncated", "data", lambda c, lib: c.write("cut.npz", Path(c.data).read_bytes()[:200])),
    Damage("data file .npy of random bytes", "data", lambda c, lib: c.write("junk.npy", os.urandom(300))),
    Damage("data file .npy with pickled objects", "data", lambda c, lib: _pickled_npy(c)),
    # the loader goes by what np.load returns, not by the file name
    Damage("data .npy bytes under the name .dat", "data", lambda c, lib: _npy_named(c, "x.dat"), "clean"),
    Damage("data .npy bytes under the name .NPY", "data", lambda c, lib: _npy_named(c, "X.NPY"), "clean"),
    Damage("data .npz archive under the name .npy", "data", lambda c, lib: c.write("arch.npy", Path(c.data_class).read_bytes()), "clean"),
    Damage("data .npz archive under the name .dat", "data", lambda c, lib: c.write("arch.dat", Path(c.data_class).read_bytes()), "clean"),
    Damage("data archive without X", "data", lambda c, lib: c.save_npz("noX.npz", Z=c.x)),
    Damage("data file a text file", "data", lambda c, lib: c.write("data.csv", b"1,2,3\n4,5,6\n")),
    Damage("data path not UTF-8", "data", non_utf8(good_data), "clean"),
    # ---- the model ----
    Damage("model missing", "model", lambda c, lib: str(c.root / "nosuch.ubj")),
    Damage("model a pipe", "model", special("fifo")),
    Damage("model a link to itself", "model", special("loop")),
    Damage("model unreadable (mode 000)", "model", special("denied"), not_root=True),
    Damage("model folder empty", "model", special("dir")),
    Damage(
        "model json nested 100000 deep",
        "model",
        lambda c, lib: c.write("nest.json", b'{"learner": ' + b"[" * 100000 + b"]" * 100000 + b"}"),
    ),
    Damage("model file empty (.ubj)", "model", lambda c, lib: c.write("empty.ubj", b"")),
    Damage("model file empty (.json)", "model", lambda c, lib: c.write("empty.json", b"")),
    Damage("model file empty (.txt)", "model", lambda c, lib: c.write("empty.txt", b"")),
    Damage("model of random bytes (.ubj)", "model", lambda c, lib: c.write("junk.ubj", os.urandom(500))),
    Damage("model of random bytes (.json)", "model", lambda c, lib: c.write("junk.json", os.urandom(500))),
    Damage("xgboost model cut in half", "model", raw_file("half", lambda b: b[: len(b) // 2])),
    Damage("xgboost model with a flipped byte", "model", raw_file("flip", lambda b: b[:200] + bytes([b[200] ^ 0xFF]) + b[201:]), "clean"),
    Damage("xgboost JSON without a learner", "model", lambda c, lib: c.write("nolearner.json", b'{"version": [1, 0, 0]}')),
    Damage("xgboost JSON of an empty learner", "model", lambda c, lib: c.write("emptylearner.json", b'{"learner": {}}')),
    Damage("xgboost JSON with no trees", "model", lambda c, lib: _xgb_without_trees(c)),
    Damage("LightGBM model cut in half", "model", lgb_text(lambda t: t[: len(t) // 2])),
    Damage("LightGBM model with trailing junk", "model", lgb_text(lambda t: t + "\x00\xff junk\n" * 5), "clean"),
    Damage(
        "LightGBM tree_sizes too large",
        "model",
        lgb_text(lambda t: re.sub(r"(?m)^tree_sizes=(\d+)", lambda m: f"tree_sizes={int(m.group(1)) + 7}", t, count=1)),
    ),
    Damage(
        "LightGBM tree_sizes too small",
        "model",
        lgb_text(lambda t: re.sub(r"(?m)^tree_sizes=(\d+)", lambda m: f"tree_sizes={int(m.group(1)) - 7}", t, count=1)),
    ),
    Damage("LightGBM tree_sizes not numbers", "model", lgb_text(lambda t: re.sub(r"(?m)^tree_sizes=.*", "tree_sizes=a b c", t, count=1))),
    Damage("LightGBM feature_infos one short", "model", lgb_text(lambda t: re.sub(r"(?m)^(feature_infos=.*) \S+$", r"\1", t, count=1))),
    Damage("LightGBM feature_names one short", "model", lgb_text(lambda t: re.sub(r"(?m)^(feature_names=.*) \S+$", r"\1", t, count=1))),
    Damage("LightGBM feature_names one more", "model", lgb_text(lambda t: re.sub(r"(?m)^(feature_names=.*)$", r"\1 extra", t, count=1))),
    Damage(
        "LightGBM max_feature_idx too large",
        "model",
        lgb_text(lambda t: re.sub(r"(?m)^max_feature_idx=\d+", "max_feature_idx=99", t, count=1)),
    ),
    *[
        Damage(
            f"LightGBM sigmoid:{v}",
            "model",
            lgb_text(lambda t, v=v: t.replace("objective=binary sigmoid:1\n", f"objective=binary sigmoid:{v}\n", 1)),
            kind,
        )
        for v, kind in (("0", "refuse"), ("-1", "refuse"), ("nan", "refuse"), ("inf", "refuse"), ("abc", "refuse"))
    ],
    Damage("xgboost model of 1,025 classes", "model", lambda c, lib: _many_classes(c), "refuse", ("xgb",), ("compile", "choose")),
    *(
        [
            Damage("YDF folder with a cut node shard", "model", lambda c, lib: _ydf_folder(c, _cut_shard), "refuse", ("ydf",)),
            Damage(
                "YDF folder without its node shard",
                "model",
                lambda c, lib: _ydf_folder(c, lambda d: _shard(d).unlink()),
                "refuse",
                ("ydf",),
            ),
            Damage(
                "YDF folder with an empty node shard",
                "model",
                lambda c, lib: _ydf_folder(c, lambda d: _shard(d).write_bytes(b"")),
                "refuse",
                ("ydf",),
            ),
            Damage(
                "YDF folder with a flipped byte in the node shard", "model", lambda c, lib: _ydf_folder(c, _flip_shard), "clean", ("ydf",)
            ),
            Damage(
                "YDF folder without header.pb",
                "model",
                lambda c, lib: _ydf_folder(c, lambda d: (d / "header.pb").unlink()),
                "refuse",
                ("ydf",),
            ),
            Damage(
                "YDF folder with a random data_spec.pb",
                "model",
                lambda c, lib: _ydf_folder(c, lambda d: (d / "data_spec.pb").write_bytes(os.urandom(200))),
                "refuse",
                ("ydf",),
            ),
            Damage(
                "YDF folder holding only header.pb",
                "model",
                lambda c, lib: _ydf_folder(c, lambda d: [q.unlink() for q in d.iterdir() if q.name != "header.pb"]),
                "refuse",
                ("ydf",),
            ),
        ]
        if HAS_YDF
        else []
    ),
    Damage("LightGBM categorical split with an empty category set", "model", lambda c, lib: _lgb_empty_category(c)),
    Damage("model path not UTF-8", "model", non_utf8(good_model), "clean", ALL_LIBS),
    # prune writes the pruned model and report.json (and heldout_probs.npz) into one folder: a model with such a name would be overwritten
    *[
        Damage(f"model file named {name}", "model", lambda c, lib, name=name: _model_named(c, lib, name), "refuse", libs, PRUNING)
        for name, libs in (
            ("report.json", ALL_LIBS),
            ("Report.JSON", ALL_LIBS),
            ("heldout_probs.npz", tuple(x for x in ALL_LIBS if x != "xgb")),
        )
    ],
    Damage("folder of two formats", "model", lambda c, lib: _mixed_folder(c)),
    Damage("folder holding a pipe named like a fold", "model", lambda c, lib: _folder_with_pipe(c), "clean"),
    Damage("folds that disagree on the feature count", "model", lambda c, lib: _folds_of_two_widths(c)),
    # ---- the compiled artifact (check) ----
    Damage("artifact missing", "artifact", lambda c, lib: str(c.root / "nosuch.flat")),
    Damage("artifact a folder", "artifact", lambda c, lib: str(c.root)),
    Damage("artifact a pipe", "artifact", special("fifo")),
    Damage("artifact empty", "artifact", lambda c, lib: c.write("empty.flat", b"")),
    Damage("artifact choice file nested 200000 deep", "artifact", lambda c, lib: _artifact_with_choice(c, "deepc.flat", b"[" * 200000)),
    Damage("artifact choice file not UTF-8", "artifact", lambda c, lib: _artifact_with_choice(c, "binc.flat", b"\xff\xfe\x00")),
    Damage("artifact of random bytes", "artifact", lambda c, lib: c.write("junk.flat", os.urandom(4000))),
    Damage("artifact cut", "artifact", lambda c, lib: c.write("cut.flat", c.artifacts["xgb"].read_bytes()[:300])),
    Damage("artifact with a flipped byte", "artifact", lambda c, lib: _flipped_flat(c)),
    Damage("artifact of another model's width", "artifact", lambda c, lib: _other_width_flat(c)),
    Damage("artifact is the model itself", "artifact", lambda c, lib: str(c.models["xgb"])),
    Damage("artifact path not UTF-8", "artifact", non_utf8(lambda c, lib: str(c.artifacts["xgb"])), "refuse"),
    # ---- the output path ----
    Damage(
        "output is an existing file (a folder expected)",
        "out",
        lambda c, lib: c.write(f"exists{c.fresh('f').name}.out", b"x"),
        "refuse",
        ("xgb",),
        FOLDER_OUTPUTS,
    ),
    Damage(
        "output is an existing file", "out", lambda c, lib: c.write(f"exists{c.fresh('f').name}.out", b"x"), "clean", ("xgb",), FILE_OUTPUTS
    ),
    Damage(  # these create the folders they need
        "output in a missing folder", "out", lambda c, lib: str(c.root / "no" / "such" / "out"), "clean", ("xgb",), CREATES_FOLDERS
    ),
    Damage(
        "output in a missing folder",
        "out",
        lambda c, lib: str(c.root / "no" / "such" / "out"),
        "refuse",
        ("xgb",),
        ("compile", "structure"),
    ),
    Damage("output in a read-only folder", "out", lambda c, lib: _readonly_out(c), not_root=True),
    Damage(
        "output is the model itself", "out", lambda c, lib: str(c.models["xgb"]), "refuse", ("xgb",), ("compile", "choose", "structure")
    ),
    Damage("output is a link to itself", "out", special("loop"), "clean"),
    Damage("output is a folder holding files", "out", lambda c, lib: str(c.root / "models"), "refuse", ("xgb",), NOT_SCRATCH),
    Damage("output is a folder holding files", "out", lambda c, lib: str(c.root / "models"), "clean", ("xgb",), ("compare",)),
    Damage(
        "output path not UTF-8", "out", lambda c, lib: os.fsdecode(os.fsencode(c.fresh("x")) + b"/caf\xe9"), "refuse", ("xgb",), WRITES_FLAT
    ),
    Damage(
        "output path not UTF-8",
        "out",
        lambda c, lib: os.fsdecode(os.fsencode(c.fresh("x")) + b"/caf\xe9"),
        "clean",
        ("xgb",),
        ("prune", "optimize", "optimize keeping all", "structure"),
    ),
    # the archive of `structure` must end in .npz: its output damages again, under a valid name
    *[
        Damage(f"npz {what}", "out", make, kind, ("xgb",), ("structure",), not_root)
        for what, make, kind, not_root in (
            ("output in a missing folder", lambda c, lib: str(c.fresh("missing") / "no" / "such" / "o.npz"), "refuse", False),
            ("output is a folder", lambda c, lib: _dir_named(c, ".npz"), "refuse", False),
            ("output is a link to itself", lambda c, lib: _loop_named(c, ".npz"), "clean", False),
            ("output is an existing file", lambda c, lib: c.write(f"exists{c.fresh('f').name}.npz", b"x"), "clean", False),
            ("output name of 300 bytes", lambda c, lib: str(c.fresh("n" * 295 + ".npz")), "refuse", False),
            ("output path of 700 bytes", lambda c, lib: deep_path(c, lib).replace("m.flat", "m.npz"), "clean", False),
            ("output in a read-only folder", lambda c, lib: _readonly_out(c) + ".npz", "refuse", True),
            ("output path not UTF-8", lambda c, lib: os.fsdecode(os.fsencode(c.fresh("x")) + b"-caf\xe9.npz"), "clean", False),
        )
    ],
    # rustc and the linker read a `%` in a path as an escape: the generated-code backends build in a clean folder
    Damage("output path with a percent sign", "out", lambda c, lib: _odd_out(c, "50%done", "m.flat"), "clean", ("xgb",), ("compile",)),
    Damage(
        "output path with a percent sign", "out", lambda c, lib: _odd_out(c, "50%done", "m.so"), "clean", ("xgb",), ("compile codegen",)
    ),
    Damage(
        "output path with spaces and quotes",
        "out",
        lambda c, lib: _odd_out(c, 'it\'s a "dir"', "m.so"),
        "clean",
        ("xgb",),
        ("compile codegen",),
    ),
    Damage("output path of 700 bytes", "out", deep_path, "clean"),
    Damage("output name of 300 bytes", "out", lambda c, lib: str(c.fresh("n" * 300))),
    *[
        Damage(
            f"decision rule file that {what}",
            "opts",
            lambda c, lib, src=source: ["--decision", c.write(f"rule{c.fresh('r').name}.py", src.encode())],
            kind,
            ("xgb",),
            RULE_ENTRIES,
        )
        for what, source, kind in rule_damages()
    ],
]


def _npy_named(ctx, name: str) -> str:
    """X alone in the .npy format, under a name that does not say so."""
    p = ctx.root / "files" / name
    p.parent.mkdir(exist_ok=True)
    with open(p, "wb") as fh:
        np.save(fh, ctx.x)
    return str(p)


def _ydf_folder(ctx, change=None) -> str:
    """A copy of the YDF model folder of the fixtures, `change(folder)` applied to it."""
    d = ctx.fresh("ydf-folder")
    shutil.copytree(ctx.models["ydf"], d)
    if change:
        change(d)
    return str(d)


def _shard(folder: Path) -> Path:
    return next(folder.glob("nodes-*"))


def _cut_shard(folder: Path) -> None:
    _shard(folder).write_bytes(_shard(folder).read_bytes()[:10])


def _flip_shard(folder: Path) -> None:
    blob = bytearray(_shard(folder).read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    _shard(folder).write_bytes(bytes(blob))


def _many_classes(ctx) -> str:
    """A multi:softprob model of 1,025 classes (one tree each): the compiled formats hold at most 1,024 output groups."""
    p = ctx.root / "models" / "classes1025.ubj"
    if not p.exists():
        import xgboost as xgb

        rng = np.random.default_rng(0)
        x = rng.normal(size=(2100, N_FEATURES)).astype(np.float32)
        y = np.arange(2100) % 1025
        xgb.train(
            {"objective": "multi:softprob", "num_class": 1025, "max_depth": 1, "nthread": 1, "verbosity": 0}, xgb.DMatrix(x, label=y), 1
        ).save_model(str(p))
    return str(p)


def _odd_out(ctx, folder: str, name: str) -> str:
    d = ctx.fresh("odd") / folder
    d.mkdir(parents=True)
    return str(d / name)


def _lgb_empty_category(ctx) -> str:
    """A LightGBM model with a categorical split whose category bitset is all zero (LightGBM loads it and sends every row right)."""
    import lightgbm as lgb

    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 3)).astype(np.float32)
    x[:, 2] = rng.integers(0, 6, 600)
    y = (x[:, 2] % 2 == 0).astype(float) + x[:, 0] * 0.1
    params = dict(objective="regression", num_leaves=4, verbose=-1, num_threads=1, min_data_per_group=3, cat_smooth=1, max_cat_to_onehot=2)
    text = lgb.train(params, lgb.Dataset(x, y, categorical_feature=[2]), 1).model_to_string()
    found = re.search(r"(?m)^cat_threshold=(.*)$", text)
    assert found, "the fixture has no categorical split"
    text = re.sub(r"(?m)^cat_threshold=.*$", "cat_threshold=" + " ".join("0" for _ in found.group(1).split()), text)
    text = re.sub(r"(?m)^tree_sizes=.*\n", "", text)
    return ctx.write("emptycat.txt", text.encode())


def _pickled_npy(ctx) -> str:
    p = ctx.root / "files" / "pickled.npy"
    p.parent.mkdir(exist_ok=True)
    np.save(p, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    return str(p)


def _xgb_without_trees(ctx) -> str:
    j = json.loads(Path(ctx.models["xgb"]).with_suffix(".ubj").read_bytes() and _xgb_json(ctx))
    m = j["learner"]["gradient_booster"]["model"]
    m["trees"], m["tree_info"], m["iteration_indptr"] = [], [], [0]
    m["gbtree_model_param"]["num_trees"] = "0"
    return ctx.write("notrees.json", json.dumps(j).encode())


def _xgb_json(ctx) -> bytes:
    import xgboost as xgb

    b = xgb.Booster()
    b.load_model(str(ctx.models["xgb"]))
    return bytes(b.save_raw(raw_format="json"))


def _mixed_folder(ctx) -> str:
    d = ctx.fresh("mixed")
    d.mkdir(parents=True)
    shutil.copy(ctx.models["xgb"], d / "fold0.ubj")
    shutil.copy(ctx.models["lgb"], d / "fold1.txt")
    return str(d)


def _model_named(ctx, lib: str, name: str) -> str:
    """A good model under a name that prune writes in its output folder (the model itself is copied there under its own name)."""
    d = ctx.fresh("named-" + name)
    d.mkdir(parents=True)
    if lib == "xgb":  # an xgboost model is a model file only as .ubj or as JSON: re-save it in the format the name asks for
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(ctx.models[lib]))
        booster.save_model(str(d / name))
    elif ctx.models[lib].is_dir():
        shutil.copytree(ctx.models[lib], d / name)
    else:
        shutil.copy(ctx.models[lib], d / name)
    return str(d / name)


def _folder_with_pipe(ctx) -> str:
    d = ctx.fresh("pipefolder")
    d.mkdir(parents=True)
    shutil.copy(ctx.models["xgb"], d / "fold0.ubj")
    os.mkfifo(d / "fold1.ubj")
    return str(d)


def _folds_of_two_widths(ctx) -> str:
    import xgboost as xgb

    d = ctx.fresh("twowidths")
    d.mkdir(parents=True)
    shutil.copy(ctx.models["xgb"], d / "fold0.ubj")
    narrow = ctx.x[:, :3]
    xgb.train({"objective": "binary:logistic", "max_depth": 2, "verbosity": 0}, xgb.DMatrix(narrow, label=ctx.y_class), 3).save_model(
        str(d / "fold1.ubj")
    )
    return str(d)


def _flipped_flat(ctx) -> str:
    blob = bytearray(ctx.artifacts["xgb"].read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    return ctx.write("flip.flat", bytes(blob))


def _other_width_flat(ctx) -> str:
    import xgboost as xgb

    sys.path.insert(0, str(SCRIPTS))
    import compile as qf_compile

    m = ctx.root / "models" / f"narrow{ctx.fresh('m').name}.ubj"
    xgb.train({"objective": "binary:logistic", "max_depth": 2, "verbosity": 0}, xgb.DMatrix(ctx.x[:, :3], label=ctx.y_class), 3).save_model(
        str(m)
    )
    out = ctx.fresh("narrow") / "m.flat"
    out.parent.mkdir(parents=True, exist_ok=True)
    qf_compile.compile_model(m, out, backend="flat")
    return str(out)


def _dir_named(ctx, suffix: str) -> str:
    d = ctx.fresh("dir").with_suffix(suffix)
    d.mkdir(parents=True)
    return str(d)


def _loop_named(ctx, suffix: str) -> str:
    p = ctx.fresh("loop").with_suffix(suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.symlink_to(p.name)
    return str(p)


def _readonly_out(ctx) -> str:
    d = ctx.fresh("readonly")
    d.mkdir(parents=True)
    d.chmod(0o555)
    return str(d / "out")


# ---- the numeric and text options: (entry points, flag, [(value, kind)]) -------------------------------------------
LEARN = ("prune", "optimize", "optimize keeping all", "optimize compile verify")
FINITE = [("nan", "refuse"), ("inf", "refuse"), ("-inf", "refuse"), ("abc", "refuse"), ("", "refuse")]
OPTIONS = [
    (
        LEARN,
        "--keep",
        [
            ("0", "refuse"),
            ("-3", "refuse"),
            ("nan", "refuse"),
            ("abc", "refuse"),
            ("", "refuse"),
            ("0%", "refuse"),
            ("-5%", "refuse"),
            ("nan%", "refuse"),
            ("%", "refuse"),
            ("150%", "clean"),
            ("inf", "clean"),
            ("inf%", "clean"),
            ("1e999", "clean"),
            ("auto", "clean"),
            ("AUTO", "clean"),
            ("auto%", "refuse"),
        ],
    ),
    (
        LEARN,
        "--keep auto --keep-ladder",  # a ladder only goes with --keep auto: the flag text carries both words
        [
            ("", "refuse"),
            (",", "refuse"),
            ("5%,", "refuse"),
            ("5%, ,10%", "refuse"),
            ("0", "refuse"),
            ("-3", "refuse"),
            ("0%", "refuse"),
            ("0.1", "refuse"),
            ("nan", "refuse"),
            ("abc", "refuse"),
            ("inf", "refuse"),
            ("1e999", "refuse"),
            ("100%", "refuse"),
            ("150%", "refuse"),
            ("5%,5%", "refuse"),
            ("5%,10%", "clean"),
            ("3,7", "clean"),
            ("10%,3", "clean"),
        ],
    ),
    (LEARN, "--sub", [("0", "refuse"), ("-1", "refuse"), ("2", "refuse"), *FINITE]),
    (("prune",), "--eval-frac", [("0", "refuse"), ("1", "refuse"), ("-0.5", "refuse"), ("2", "refuse"), *FINITE]),
    (
        ("prune",),
        "--max-sel-rows",
        [("0", "refuse"), ("-1", "refuse"), ("1.5", "refuse"), ("abc", "refuse"), ("100000000000000000000", "clean")],
    ),
    (
        ("prune",),
        "--memory-mb",
        [("0", "refuse"), ("-1", "refuse"), ("1.5", "refuse"), ("abc", "refuse"), ("", "refuse"), ("100000000000000000000", "clean")],
    ),
    (
        ("prune",),
        "--max-held-rows",
        [("0", "refuse"), ("-1", "refuse"), ("1.5", "refuse"), ("abc", "refuse"), ("", "refuse"), ("100000000000000000000", "clean")],
    ),
    (
        ("prune",),
        "--threads",
        [
            ("0", "clean"),
            ("-1", "clean"),
            ("all", "clean"),
            ("1.5", "refuse"),
            ("abc", "refuse"),
            ("", "refuse"),
            ("1000000000000", "clean"),
        ],
    ),
    (LEARN, "--seed", [("-1", "refuse"), ("1.5", "refuse"), ("abc", "refuse"), ("", "refuse"), ("100000000000000000000", "clean")]),
    (("prune",), "--tol", [("0", "refuse"), ("-1", "refuse"), *FINITE]),
    (("prune",), "--cases", [("-1", "refuse"), ("abc", "refuse"), ("1.5", "refuse"), ("0", "clean")]),
    (
        ("prune", "optimize", "optimize keeping all", "check", "choose", "compile", "compare"),
        "--decision",
        [
            ("threshold:nan", "refuse"),
            ("threshold:inf", "refuse"),
            ("threshold:-inf", "refuse"),
            ("threshold:", "refuse"),
            ("threshold:abc", "refuse"),
            ("threshold", "refuse"),
            ("", "refuse"),
            ("ARGMAX", "refuse"),
            ("nosuch.py", "refuse"),
            ("nosuch.py:fn", "refuse"),
            ("/", "refuse"),
            ("threshold:0.5", "clean"),
            ("threshold:1e-300", "clean"),
        ],
    ),
    (
        ("optimize", "optimize keeping all", "optimize compile verify"),
        "--min-agreement",
        [*FINITE, ("-5", "clean"), ("150", "clean"), ("0", "clean"), ("100", "clean")],
    ),
    (
        ("optimize", "optimize keeping all", "optimize compile verify", "check", "compare"),
        "--threads",
        [
            ("0", "clean"),
            ("-1", "clean"),
            ("1000000000000", "clean"),
            ("all", "clean"),
            ("ALL", "clean"),
            ("abc", "refuse"),
            ("none", "refuse"),
            ("1.5", "refuse"),
            ("", "refuse"),
        ],
    ),
    (
        ("check",),
        "--speed --threads",
        [("all", "clean"), ("1", "clean"), ("0", "clean"), ("abc", "refuse"), ("1.5", "refuse"), ("", "refuse")],
    ),
    (("check",), "--random", [("-1", "refuse"), ("0", "clean"), ("abc", "refuse"), ("1.5", "refuse"), ("1000000000", "refuse")]),
    (("check",), "--seed", [("-1", "refuse"), ("abc", "refuse"), ("1.5", "refuse")]),
    (("check", "compile", "choose", "compare", "prune", "optimize"), "--average", [("nosuch", "refuse"), ("", "refuse")]),
    (
        ("check", "optimize", "compile"),
        "--backend",
        [("nosuch", "refuse"), ("", "refuse"), ("nosuch.py", "refuse"), ("nosuch.py:Cls", "refuse"), ("wave", "clean")],
    ),
    (("check", "compare", "choose"), "--cpu", [("-1", "refuse"), ("abc", "refuse"), ("1000000", "refuse")]),
    (("compare",), "--random", [("-1", "refuse"), ("abc", "refuse"), ("1000000000", "refuse")]),
    (("compare",), "--eq-rows", [("0", "refuse"), ("-1", "refuse"), ("abc", "refuse"), ("1000000000", "clean")]),
    (("compare",), "--speed-rows", [("0", "refuse"), ("-1", "refuse"), ("abc", "refuse"), ("1000000000", "clean")]),
    (("compare",), "--backends", [("nosuch", "refuse"), ("", "refuse"), ("flat,nosuch", "refuse")]),
    (("choose",), "--only", [("nosuch", "refuse"), ("", "refuse")]),
    (
        ("choose", "compare", "compile"),
        "--opt",
        [("novalue", "refuse"), ("=", "refuse"), ("block=notanumber", "refuse"), ("chunk_nodes=-5", "refuse")],
    ),
    # a folder the user cannot write ({readonly}: skipped for root, who can) and a target that is not a regular file are refused BEFORE the run
    (
        ("check",),
        "--json",
        [("{missing}/x.json", "refuse"), ("{root}", "refuse"), ("{readonly}/x.json", "refuse"), ("/dev/null", "refuse")],
    ),
    (
        ("compare",),
        "--json",
        [("{missing}/x.json", "refuse"), ("{root}", "refuse"), ("{readonly}/x.json", "refuse"), ("/dev/null", "refuse")],
    ),
    (("compare",), "--md", [("{missing}/x.md", "refuse"), ("{root}", "refuse"), ("{readonly}/x.md", "refuse"), ("/dev/null", "refuse")]),
    (("optimize", "optimize compile verify"), "--regression", [("sqrt", "refuse"), ("", "refuse"), ("raw", "refuse"), ("log", "refuse")]),
    (("optimize", "optimize compile verify", "prune"), "--med-tol", [*FINITE]),
    (("optimize", "optimize compile verify", "prune"), "--max-tol", [*FINITE]),
    (("optimize", "optimize compile verify", "prune"), "--error-slack", [*FINITE]),
]


def option_damages():
    for entries, flag, values in OPTIONS:
        for value, kind in values:
            yield entries, flag, value, kind


# --------------------------------------------------------------------------------------------------- the judgement
def judge(entry: Entry, kind: str, code, err: str) -> str | None:
    """None when the cell behaved as the documentation says, else what was wrong."""
    lines = [ln for ln in err.splitlines() if ln.strip()]
    if code == "hang":
        return f"hang (no answer in {CELL_SECONDS} s)"
    if isinstance(code, int) and code < 0:
        return f"killed by signal {-code}"
    if "Traceback" in err or "[bt]" in err:
        return "traceback: " + (lines[-1] if lines else "")[:150]
    if code == 0:
        return "accepted" if kind == "refuse" else None
    if code == 2 and any(" error: " in ln for ln in lines[-1:]):
        return None  # the option parser's own refusal: usage and an "error:" line
    if code in entry.codes:
        if len(lines) != 1:
            return f"exit {code} with {len(lines)} lines on stderr: {' | '.join(lines)[:200]!r}"
        return None
    if code == 2 and not lines:
        return "a failed bar (exit 2) instead of a refusal" if kind == "refuse" else None
    return f"exit {code}: {' | '.join(lines)[:200]!r}"


# ----------------------------------------------------------------------------------------------------- the cells
def build_cells(ctx: Ctx):
    """Yields (label, entry, kind, command, argv-builder) for every cell; the builder makes the damaged value only when the cell is run
    (some damages create files or folders)."""
    for entry in ENTRY_POINTS:
        for d in DAMAGES:
            if d.slot not in entry.slots or (d.only and entry.name not in d.only) or (d.not_root and os.geteuid() == 0):
                continue
            for lib in d.libs:
                yield f"{entry.name} | {d.name} | {lib}", entry, d.kind, d.slot, d.make, lib
        for entries, flag, value, kind in option_damages():
            if "{readonly}" in value and os.geteuid() == 0:
                continue  # a mode that stops a user does not stop root
            if entry.name in entries:
                yield (
                    f"{entry.name} | {flag} {value!r}",
                    entry,
                    kind,
                    "opts",
                    (lambda c, lib, f=flag, v=value: [*f.split(), option_value(c, v)]),
                    "xgb",
                )


def option_value(ctx: Ctx, value: str) -> str:
    """An option value of OPTIONS with its folders filled in; `{readonly}` is a new read-only folder (made only when the value asks for it)."""
    readonly = str(Path(_readonly_out(ctx)).parent) if "{readonly}" in value else ""
    return value.format(root=ctx.root, missing=ctx.root / "nosuch", readonly=readonly)


def run_cell(ctx: Ctx, worker_: Worker, cell):
    label, entry, kind, slot, make, lib = cell
    s = good_slots(ctx, entry, lib)
    try:
        if slot == "opts":
            s.opts = make(ctx, lib)
        else:
            setattr(s, slot, make(ctx, lib))
        argv = entry.argv(s)
    except Exception as e:  # a damage the harness could not build is a failing cell, not a stopped run
        return label, f"the damaged input could not be built: {type(e).__name__}: {e}"
    code, err = worker_.run(entry.command, argv)
    return label, judge(entry, kind, code, err)


def good_slots(ctx: Ctx, entry: Entry, lib: str) -> Slots:
    """The valid command line of an entry point on one library's model (the regression model gets the data that holds its truth)."""
    return Slots(
        model=str(ctx.models[lib]),
        data=ctx.data if lib == "xgbr" else ctx.data_class,
        artifact=str(ctx.artifacts[lib]),
        out=str(ctx.fresh(entry.name.replace(" ", "-")))
        + (
            ".flat"
            if entry.name in ("compile", "choose")
            else ".so"
            if entry.name == "compile codegen"
            else ".npz"
            if entry.name == "structure"
            else ""
        ),
        lib=lib,
    )


def control_cells(ctx: Ctx):
    """Valid inputs through every entry point and library: the harness must see success where there is nothing wrong."""
    for entry in ENTRY_POINTS:
        for lib in (*ALL_LIBS, "xgbn"):
            yield entry, lib


class InputMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ctx = Ctx(Path(cls.tmp.name))
        cls.worker = Worker()
        cls.results = None

    @classmethod
    def tearDownClass(cls):
        cls.worker.close()
        for p in Path(cls.tmp.name).rglob("*"):
            with contextlib.suppress(OSError):
                if p.is_dir() and not p.is_symlink():
                    p.chmod(0o755)
        cls.tmp.cleanup()

    @classmethod
    def results_by_group(cls) -> dict[str, list]:
        """Runs every cell once (the first test that asks) and groups the verdicts by what was damaged."""
        if cls.results is None:
            cls.results = {}
            report = os.environ.get("INPUT_MATRIX_REPORT")  # a file that gets every cell's verdict as it is judged (for developing)
            only = os.environ.get("INPUT_MATRIX_ONLY", "")
            for cell in build_cells(cls.ctx):
                if only not in cell[0]:
                    continue
                label, verdict = run_cell(cls.ctx, cls.worker, cell)
                cls.results.setdefault(cell[3], []).append((label, verdict))
                if report:
                    with open(report, "a") as fh:
                        fh.write(f"{'FAIL' if verdict else 'ok  '} {label}{': ' + verdict if verdict else ''}\n")
        return cls.results

    def assert_slot_clean(self, slot: str):
        rows = self.results_by_group().get(slot, [])
        self.assertTrue(rows, f"no cell damages {slot}")
        bad = [f"{label}: {why}" for label, why in rows if why]
        self.assertFalse(bad, f"{len(bad)} of {len(rows)} cells misbehave:\n  " + "\n  ".join(bad))

    def test_every_valid_input_runs(self):
        worker_ = self.worker
        bad = []
        for entry, lib in control_cells(self.ctx):
            code, err = worker_.run(entry.command, entry.argv(good_slots(self.ctx, entry, lib)))
            if code != 0 or "Traceback" in err:
                bad.append(f"{entry.name} | {lib}: exit {code}, stderr {err[-200:]!r}")
        self.assertFalse(bad, "\n  ".join(bad))

    def test_damaged_model_files_and_folders(self):
        self.assert_slot_clean("model")

    def test_damaged_data_arrays_and_files(self):
        self.assert_slot_clean("data")

    def test_damaged_compiled_artifacts(self):
        self.assert_slot_clean("artifact")

    def test_output_paths_that_cannot_be_written(self):
        self.assert_slot_clean("out")

    def test_numeric_and_text_options(self):
        self.assert_slot_clean("opts")


# ------------------------------------------------------------------------------------------ the Python API
# Every damage a caller can hand to `predict` of a compiled forest, and every path `FlatForest` can be opened on: a ValueError that says
# what is wrong (never another exception type, a crash or a wrong answer). The compiled forest has FEATURES inputs and one output group.
FEATURES = N_FEATURES
PREDICT_DAMAGES = {
    "x 1-D": lambda x: dict(x=x[0]),
    "x 3-D": lambda x: dict(x=x[None]),
    "x 0-d": lambda x: dict(x=np.float32(1)),
    "x one column too few": lambda x: dict(x=x[:, :-1]),
    "x one column too many": lambda x: dict(x=np.hstack([x, x[:, :1]])),
    "x no columns": lambda x: dict(x=x[:, :0]),
    "x of text": lambda x: dict(x=np.array([["a"] * FEATURES] * 3)),
    "x complex": lambda x: dict(x=x.astype(np.complex64)),
    "x masked": lambda x: dict(x=np.ma.masked_array(x, mask=x > 0)),
    "x ragged lists": lambda x: dict(x=[[1.0] * FEATURES, [1.0]]),
    "threads text": lambda x: dict(x=x, threads="two"),
    "threads list": lambda x: dict(x=x, threads=[2]),
    "threads 1.5": lambda x: dict(x=x, threads=1.5),
    "dtype float16": lambda x: dict(x=x, dtype=np.float16),
    "dtype int32": lambda x: dict(x=x, dtype=np.int32),
    "dtype text": lambda x: dict(x=x, dtype="nonsense"),
    "base_margin one row short": lambda x: dict(x=x, base_margin=np.zeros(len(x) - 1)),
    "base_margin two columns": lambda x: dict(x=x, base_margin=np.zeros((len(x), 2))),
    "base_margin 3-D": lambda x: dict(x=x, base_margin=np.zeros((len(x), 1, 1))),
    "base_margin NaN": lambda x: dict(x=x, base_margin=np.full(len(x), np.nan)),
    "base_margin infinite": lambda x: dict(x=x, base_margin=np.full(len(x), np.inf)),
    "base_margin of text": lambda x: dict(x=x, base_margin=np.array(["a"] * len(x))),
    "base_margin complex": lambda x: dict(x=x, base_margin=np.zeros(len(x), complex)),
    "base_margin masked": lambda x: dict(x=x, base_margin=np.ma.masked_array(np.zeros(len(x)), mask=True)),
    "base_margin and float64": lambda x: dict(x=x, base_margin=np.zeros(len(x) + 1), dtype=np.float64),
}


class ApiInputMatrixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.ctx = Ctx(Path(cls.tmp.name))
        import backends
        import evaluator

        cls.evaluator = evaluator
        cls.predictors = {
            "flat": evaluator.FlatForest(cls.ctx.artifacts["xgb"]),
            "wave": evaluator.FlatForest(cls.ctx.artifacts["xgb"], engine=1),
        }
        cls.predictors["lightgbm flat"] = evaluator.FlatForest(cls.ctx.artifacts["lgb"])
        try:
            sys.path.insert(0, str(SCRIPTS))
            import qf_common as qf

            so = cls.ctx.root / "codegen" / "m.so"
            backends.get("codegen-block").compile(qf.load_forest([cls.ctx.models["xgb"]]), so, {})
            cls.predictors["codegen-block"] = backends.get("codegen-block").load(so)
        except Exception:  # no rustc here: the generated code is not part of this table
            pass

    @classmethod
    def tearDownClass(cls):
        for p in cls.predictors.values():
            if hasattr(p, "close"):
                p.close()
        cls.tmp.cleanup()

    def test_predict_refuses_a_damaged_argument_with_a_value_error(self):
        bad = []
        for name, predictor in self.predictors.items():
            x = self.ctx.x[:8].copy()
            for what, make in PREDICT_DAMAGES.items():
                if name == "wave" and what.startswith("base_margin"):
                    continue
                try:
                    predictor.predict(**make(x))
                except ValueError as e:
                    if not str(e):
                        bad.append(f"{name} | {what}: a ValueError without a message")
                except Exception as e:  # any other type is what this test is looking for
                    bad.append(f"{name} | {what}: {type(e).__name__}: {str(e)[:100]}")
                else:
                    bad.append(f"{name} | {what}: accepted")
        self.assertFalse(bad, "\n  " + "\n  ".join(bad))

    def test_predict_accepts_what_is_valid(self):
        for name, predictor in self.predictors.items():
            x = self.ctx.x[:8]
            self.assertEqual(len(predictor.predict(x)), 8, name)
            self.assertEqual(len(predictor.predict(x[:0])), 0, name)  # no rows: an empty answer
            self.assertEqual(len(predictor.predict(x.astype(np.float64), threads=3, dtype=np.float64)), 8, name)
            self.assertEqual(len(predictor.predict(x.tolist(), threads=-4)), 8, name)

    def test_a_forest_that_was_closed_refuses_to_predict(self):
        forest = self.evaluator.FlatForest(self.ctx.artifacts["xgb"])
        forest.close()
        forest.close()
        with self.assertRaises(ValueError):
            forest.predict(self.ctx.x[:2])

    def test_opening_a_path_that_is_not_a_forest_is_a_value_error(self):
        ctx = self.ctx
        fifo = ctx.root / "pipe.flat"
        os.mkfifo(fifo)
        blob = ctx.artifacts["xgb"].read_bytes()
        paths = {
            "missing": ctx.root / "nosuch.flat",
            "folder": ctx.root,
            "pipe": fifo,
            "empty": Path(ctx.write("empty.flat", b"")),
            "random bytes": Path(ctx.write("junk.flat", os.urandom(5000))),
            "cut": Path(ctx.write("cut.flat", blob[: len(blob) // 2])),
            "header only": Path(ctx.write("head.flat", blob[:64])),
            "one flipped byte": Path(ctx.write("flip.flat", blob[:100] + bytes([blob[100] ^ 0xFF]) + blob[101:])),
            "NUL in the name": str(ctx.artifacts["xgb"]) + "\0x",
        }
        bad = []
        for what, path in paths.items():
            try:
                self.evaluator.FlatForest(path).close()
            except ValueError as e:
                if not str(e):
                    bad.append(f"{what}: a ValueError without a message")
            except Exception as e:
                bad.append(f"{what}: {type(e).__name__}: {str(e)[:100]}")
            else:
                bad.append(f"{what}: opened")
        self.assertFalse(bad, "\n  " + "\n  ".join(bad))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker()
    else:
        unittest.main()
