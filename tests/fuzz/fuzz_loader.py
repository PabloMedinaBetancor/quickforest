#!/usr/bin/env python3
"""Fuzz harness for every file the evaluator and the compiler read: a corrupt file must end in a clean Python exception with a message,
never a segfault, a hang, an out-of-bounds read or a wrong answer.

    fuzz_loader.py --mode quick            # 1-2 minutes, for CI
    fuzz_loader.py --mode full             # ~100k cases
    fuzz_loader.py --replay flat:xgb_multi:child:1234     # write that mutated file to --out and load it (target:base:kind:seed)

Targets
  flat    the flat forest file (rust/src/lib.rs `from_storage`, the one file the Rust side reads): truncated at every kind of length,
          extended, bit-flipped (checksum left wrong, and fixed so the structure checks are reached), wrong magic / format / flags, huge
          and zero counts in every header field, child distances out of range, feature indices out of range, NaN thresholds, tree and fold
          records out of order, category tables and linear-leaf tables corrupted, single bits of the header flipped, 2-5 of these stacked.
          Every file is loaded through the real ctypes path (`evaluator.FlatForest`), and when it loads it is predicted with every engine
          (0, 1, 2), 1 and 3 threads, float32 and float64, with and without a base margin; the engines must agree bit for bit, and the
          answers must be bit-identical to the undamaged file's. The one exception is a mutation whose checksum was repaired (a changed
          file with a correct checksum is a different forest, expected, counted apart as "LOADED a different forest").
  lgb     a LightGBM text model (qf_lgb): truncated, fields removed, counts huge or negative, child indices out of range or cyclic,
          non-numeric numbers, NUL bytes.
  xgb     an xgboost JSON model (qf_common.parse_model): the same, on the JSON.
          For both, a text file that QF accepts is also loaded and asked for a row by the model's own library, in a process of its own: when
          that library aborts or segfaults on the damaged file the case is counted apart as "SOURCE LIBRARY CRASH" (library-crashes.txt), not as a failure.
  npz     the selection / data file (`qf_common.load_data`): truncated archive, missing X, wrong dtypes, pickled objects.
  choice  `ARTIFACT.choice.json` (`backends.load_artifact`): invalid JSON, unknown backend, wrong types.
  api     misuse of the Python API: wrong shapes and dtypes, closed forest, bad threads / engine / margin, odd paths (FIFO, directory,
          empty file, NUL in the path), concurrent predict from several Python threads.

Each target runs in a child process that handles a batch of cases; a child that dies by a signal or stops answering for TIMEOUT seconds is
recorded as CRASH / HANG on the case it was on and a new child continues with the next case. "Clean" means ValueError, RuntimeError
(EngineRefused included), OSError, EOFError, or a library's own error class (XGBoostError, LightGBMError, BadZipFile) WITH a non-empty
message. KeyError, IndexError, TypeError, AttributeError, OverflowError, SystemExit, MemoryError ... are UNCLEAN: they mean the parser
leaked an internal error. Output: one line per non-clean case (and a summary). Exit code 1 if there is any CRASH, HANG, UNCLEAN or DIVERGE."""

from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

CLEAN_NAMES = {"XGBoostError", "LightGBMError", "BadZipFile", "YdfError"}
TIMEOUT = 30.0

# The libraries under test keep memory across thousands of damaged files (a source-library probe worker reached 3.8 GB in a full run), so a
# long-lived worker is replaced between two requests, with the same verdicts: when its resident size passes PROBE_RSS_MB or after
# PROBE_REQUESTS requests; a case process when its resident size has grown CHILD_GROWTH_MB since its first case or after CHILD_CASES cases.
# A probe worker also gets a ceiling on its address space (PROBE_HEADROOM_MB above what it holds after importing the libraries): a library
# asked for more is refused with a MemoryError / bad_alloc, which is the source library's answer, not QuickForest's.
# The numbers can be lowered through the environment (QF_FUZZ_PROBE_RSS_MB and so on) to exercise the replacement in a short run.
PROBE_RSS_MB = int(os.environ.get("QF_FUZZ_PROBE_RSS_MB", 1024))
PROBE_REQUESTS = int(os.environ.get("QF_FUZZ_PROBE_REQUESTS", 2000))
PROBE_HEADROOM_MB = int(os.environ.get("QF_FUZZ_PROBE_HEADROOM_MB", 3072))
CHILD_GROWTH_MB = int(os.environ.get("QF_FUZZ_CHILD_GROWTH_MB", 700))
CHILD_CASES = int(os.environ.get("QF_FUZZ_CHILD_CASES", 20000))


def rss_mb(pid) -> float | None:
    """Resident size of a process in MB from /proc (pid, or "self"); None where /proc cannot say."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        pass
    return None


# ---------------------------------------------------------------------------------------------------------- the flat layout
HEADER = 64
OFF = {
    "format": 8,
    "n_features": 12,
    "n_groups": 16,
    "n_folds": 20,
    "n_trees": 24,
    "n_nodes": 28,
    "objective": 32,
    "average": 36,
    "checksum": 40,
    "flags": 48,
    "cat_words": 52,
    "lin_leaves": 56,
    "lin_coefs": 60,
}
SPECIAL32 = [0, 1, 2, 3, 7, 8, 255, 256, 0x7FFF, 0x8000, 0xFFFF, 0x10000, 0x7FFFFFFF, 0x80000000, 0x80000001, 0xFFFFFFFE, 0xFFFFFFFF]
SPECIAL64 = [
    0,
    0xFFFFFFFFFFFFFFFF,
    0x8000000000000000,
    0x7FFFFFFFFFFFFFFF,
    0x00000000FFFFFFFF,
    0xFFFFFFFF00000000,
    0x7FC00000FFFF0000,
    0x0001000000000000,
    0xFFFF000000000000,
    0x00007FFF00000000,
]
NAN32 = [0x7FC00000, 0xFFC00000, 0x7FA00000, 0x7F800001, 0xFFFFFFFF]
KINDS = [
    "trunc",
    "extend",
    "flip",
    "flip-fix",
    "word-fix",
    "hdr",
    "magic",
    "format",
    "flags",
    "child",
    "feature",
    "nan-thr",
    "tree-rec",
    "fold-rec",
    "tail",
    "lin",
    "multi",
    "hdr-flip",  # appended last: the seeds of the kinds above stay what they were
]


def u32(b, at):
    return struct.unpack_from("<I", b, at)[0]


def layout(b: bytes) -> dict:
    """Section offsets of a (valid) flat file, from its own header."""
    f = {k: u32(b, v) for k, v in OFF.items() if k != "checksum"}
    base_bytes = -(-(f["n_folds"] * f["n_groups"] * 4) // 8) * 8
    tree_at = HEADER + f["n_folds"] * 8 + base_bytes
    node_at = tree_at + f["n_trees"] * 8
    cats_at = node_at + f["n_nodes"] * 8
    lin_at = cats_at + -(-(f["cat_words"] * 4) // 8) * 8
    coef_at = lin_at + f["lin_leaves"] * 24
    return {**f, "tree_at": tree_at, "node_at": node_at, "cats_at": cats_at, "lin_at": lin_at, "coef_at": coef_at}


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def refix(b: bytearray) -> bytes:
    """Recompute the checksum (the evaluator's own implementation: the whole file, the checksum field read as zero), so the structural
    checks behind it are reached."""
    import evaluator

    return evaluator.seal(b)


def mutate(base: bytes, kind: str, rng, depth: int = 0) -> bytes:
    b = bytearray(base)
    L = layout(base)
    n = len(b)

    def put32(at, v):
        if at + 4 <= len(b):
            struct.pack_into("<I", b, at, v & 0xFFFFFFFF)

    def put64(at, v):
        if at + 8 <= len(b):
            struct.pack_into("<Q", b, at, v & 0xFFFFFFFFFFFFFFFF)

    def pick(seq):
        return seq[int(rng.integers(0, len(seq)))]

    def node_words():
        return [L["node_at"] + 8 * i for i in range(min(L["n_nodes"], n // 8))]  # a stacked mutation may have made the count huge

    def internal_nodes():
        out = []
        for i in range(min(L["n_nodes"], n // 8)):
            at = L["node_at"] + 8 * i
            if at + 8 <= len(b) and (struct.unpack_from("<Q", b, at)[0] >> 32) & 0xFFFF:
                out.append(at)
        return out

    if kind == "trunc":
        cuts = [0, 1, 7, 8, 63, 64, 65, HEADER + 8, L["tree_at"], L["node_at"], L["node_at"] + 4, L["cats_at"], n - 1, n - 4, n - 8, n - 9]
        cut = pick(cuts) if rng.random() < 0.5 else int(rng.integers(0, n))
        return bytes(b[: max(0, min(cut, n))])
    if kind == "extend":
        return bytes(b) + bytes(rng.integers(0, 256, int(rng.integers(1, 65)), dtype=np.uint8))
    if kind == "flip":
        for _ in range(int(rng.integers(1, 4))):
            at = int(rng.integers(0, n))
            b[at] ^= 1 << int(rng.integers(0, 8))
        return bytes(b)
    if kind == "flip-fix":
        for _ in range(int(rng.integers(1, 5))):
            at = int(rng.integers(HEADER, n))
            b[at] ^= 1 << int(rng.integers(0, 8))
        return refix(b)
    if kind == "word-fix":
        for _ in range(int(rng.integers(1, 4))):
            at = HEADER + 8 * int(rng.integers(0, max(1, (n - HEADER) // 8)))
            put64(at, pick(SPECIAL64) if rng.random() < 0.5 else int(rng.integers(0, 2**63)) * 2 + int(rng.integers(0, 2)))
        return refix(b)
    if kind == "hdr":
        field = pick([k for k in OFF if k not in ("checksum", "format")])
        put32(OFF[field], pick(SPECIAL32) if rng.random() < 0.7 else int(rng.integers(0, 2**32)))
        return bytes(b) if rng.random() < 0.5 else refix(b)
    if kind == "magic":
        if rng.random() < 0.3:
            b[0:8] = pick([b"\0" * 8, b"TREEFLT2", b"treeflt1", b"TREEFLT1"[::-1], b"XGBOOST!"])
        else:
            b[int(rng.integers(0, 8))] ^= int(rng.integers(1, 256))
        return bytes(b)
    if kind == "format":
        put32(OFF["format"], pick([0, 1, 2, 3, 4, 5, 6, 7, 8, 255, 0xFFFFFFFF]))
        return bytes(b)
    if kind == "hdr-flip":  # the header is under the checksum: a flipped bit of it, checksum left alone, must be refused
        for _ in range(int(rng.integers(1, 3))):
            b[int(rng.integers(0, HEADER))] ^= 1 << int(rng.integers(0, 8))
        return bytes(b)
    if kind == "flags":
        put32(OFF["flags"], int(rng.integers(0, 64)) if rng.random() < 0.7 else pick(SPECIAL32))
        return bytes(b) if rng.random() < 0.3 else refix(b)
    if kind == "child":
        nodes = internal_nodes()
        if nodes:
            at = pick(nodes)
            w = struct.unpack_from("<Q", b, at)[0]
            end = L["node_at"] + 8 * L["n_nodes"]
            dist = pick(
                [0, 1, 2, 3, 0xFFFF, 0x8000, (end - at) // 8, (end - at) // 8 - 1, (end - at) // 8 + 1, int(rng.integers(0, 0x10000))]
            )
            put64(at, (w & ~(0xFFFF << 32)) | ((dist & 0xFFFF) << 32))
        return refix(b)
    if kind == "feature":
        nodes = internal_nodes()
        if nodes:
            at = pick(nodes)
            w = struct.unpack_from("<Q", b, at)[0]
            feat = pick([L["n_features"], L["n_features"] + 1, 0x1FFF, 0x7FFF, 0x2000, 0x4000, int(rng.integers(0, 0x8000))])
            put64(at, (w & ~(0x7FFF << 48)) | ((feat & 0x7FFF) << 48))
        return refix(b)
    if kind == "nan-thr":
        nodes = internal_nodes() or node_words()
        for _ in range(int(rng.integers(1, 3))):
            put32(pick(nodes), pick(NAN32))
        return refix(b)
    if kind == "tree-rec":
        for _ in range(int(rng.integers(1, 3))):
            i = int(rng.integers(0, max(1, L["n_trees"])))
            at = L["tree_at"] + 8 * i
            put32(at, pick([0, 1, L["n_nodes"], L["n_nodes"] - 1, 0xFFFFFFFF, int(rng.integers(0, L["n_nodes"] + 3))]))
            put32(at + 4, pick([0, 1, L["n_groups"], L["n_groups"] + 1, 0xFFFFFFFF, int(rng.integers(0, 8))]))
        return refix(b)
    if kind == "fold-rec":
        i = int(rng.integers(0, max(1, L["n_folds"])))
        put32(HEADER + 8 * i, pick([*SPECIAL32, L["n_trees"]]))
        put32(HEADER + 8 * i + 4, pick([*SPECIAL32, L["n_trees"]]))
        return refix(b)
    if kind == "tail":  # category table / linear tables / anything after the nodes
        lo = L["cats_at"]
        if lo + 8 <= len(b):
            for _ in range(int(rng.integers(1, 4))):
                at = lo + 4 * int(rng.integers(0, max(1, (len(b) - lo) // 4)))
                put32(at, pick(SPECIAL32 + NAN32))
        return refix(b)
    if kind == "lin":
        if L["lin_leaves"]:
            k = int(rng.integers(0, L["lin_leaves"]))
            put64(L["lin_at"] + 24 * k + 16, pick(SPECIAL64))  # (first coefficient, count)
            put64(L["lin_at"] + 24 * k, pick([0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000]))  # NaN / inf constant
        if L["lin_coefs"]:
            k = int(rng.integers(0, L["lin_coefs"]))
            put64(L["coef_at"] + 16 * k, pick([L["n_features"], 0xFFFFFFFF, 0x1_0000_0000, 0x7FFFFFFF]))
            put64(L["coef_at"] + 16 * k + 8, pick([0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000]))
        return refix(b)
    if kind == "multi" and depth == 0:
        out = bytes(b)
        for _ in range(int(rng.integers(2, 6))):
            out = mutate(out if len(out) >= HEADER + 8 else base, pick([k for k in KINDS if k != "multi"]), rng, depth + 1)
        return out
    return bytes(b)


# ---------------------------------------------------------------------------------------------------------- text / json mutations
def mutate_text(data: bytes, rng, fmt: str) -> bytes:
    """Mutations of a LightGBM text model or an xgboost JSON model that a parser may trip on."""
    n = len(data)
    pick = lambda seq: seq[int(rng.integers(0, len(seq)))]  # noqa: E731
    mode = int(rng.integers(0, 10))
    if mode == 0:
        return data[: int(rng.integers(0, n))]
    if mode == 1:
        b = bytearray(data)
        for _ in range(int(rng.integers(1, 6))):
            b[int(rng.integers(0, n))] = int(rng.integers(0, 256))
        return bytes(b)
    if mode == 2:
        b = bytearray(data)
        at = int(rng.integers(0, n))
        b[at:at] = pick([b"\0", b"\xff\xfe", b"nan", b"-1", b"99999999999999999999", b"1e999", b"[", b"}", b'"'])
        return bytes(b)
    if mode == 3:  # drop a whole line / a key
        lines = data.split(b"\n")
        del lines[int(rng.integers(0, len(lines)))]
        return b"\n".join(lines)
    if mode == 4:  # duplicate a line
        lines = data.split(b"\n")
        i = int(rng.integers(0, len(lines)))
        lines.insert(i, lines[i])
        return b"\n".join(lines)
    key = (
        pick(
            [
                b"left_child",
                b"right_child",
                b"split_feature",
                b"threshold",
                b"decision_type",
                b"num_leaves",
                b"num_cat",
                b"leaf_value",
                b"cat_boundaries",
                b"cat_threshold",
                b"leaf_coeff",
                b"leaf_features",
                b"leaf_const",
                b"num_class",
                b"num_tree_per_iteration",
                b"max_feature_idx",
                b"tree_sizes",
                b"objective",
                b"feature_names",
            ]
        )
        if fmt == "lgb"
        else pick(
            [
                b"left_children",
                b"right_children",
                b"split_indices",
                b"split_conditions",
                b"default_left",
                b"base_weights",
                b"tree_info",
                b"num_feature",
                b"num_class",
                b"base_score",
                b"split_type",
                b"categories",
                b"objective",
                b"num_trees",
                b"size_leaf_vector",
            ]
        )
    )
    i = data.find(key)
    if i < 0:
        return data[: n // 2]
    if mode == 5:  # delete the key's name
        return data[:i] + b"x" + data[i + len(key) :]
    # replace the value after the key (up to the end of its line / the next , or ] ) with something hostile
    j = i + len(key)
    stop = min([k for k in (data.find(b"\n", j), data.find(b",", j), data.find(b"]", j)) if k >= 0] or [n])
    hostile = pick(
        [
            b"=-1",
            b"=0",
            b"=99999999999",
            b"=1e999",
            b"=nan",
            b"=abc",
            b"=",
            b"=1 2 3 -5 7 7 7 7 7",
            b":-1",
            b":0",
            b":99999999999",
            b":null",
            b":[]",
            b':"x"',
            b":[0,0,0,0]",
            b":[-1,-1]",
            b":[1,0,2]",
        ]
    )
    return data[:j] + hostile + data[stop:]


# ---------------------------------------------------------------------------------------------------------- children
import numpy as np


def classify(e: BaseException) -> str:
    name = type(e).__name__
    msg = str(e).strip()
    if isinstance(e, KeyboardInterrupt):
        raise e
    clean = isinstance(e, (ValueError, RuntimeError, OSError, EOFError)) or name in CLEAN_NAMES
    # ModelFileError is both: a ValueError for callers, a clean exit for the CLIs
    if isinstance(e, SystemExit) and not isinstance(e, ValueError):
        clean = False
    if isinstance(e, (OverflowError, MemoryError, RecursionError)):
        clean = False
    if clean and not msg:
        return f"UNCLEAN {name} without a message"
    return f"CLEAN {name}: {msg[:140]}" if clean else f"UNCLEAN {name}: {msg[:140]}"


def same_bits(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def loaded_verdict(data: bytes, original: bytes, fo, orig_fo, x, bm) -> str | None:
    """The oracle for a mutant that LOADS. None: it is clean (the same file, or one that answers bit for bit like the undamaged file).
    A changed file whose checksum bytes are the original's would be a damaged file accepted: DIVERGE. A changed file with another
    checksum (the mutation repaired it) that answers differently loaded a different forest: expected, counted apart."""
    if data == original:
        return None
    if data[40:48] == original[40:48]:
        return "DIVERGE loaded a changed file that still has the original checksum"
    orig_fo.engine = 0
    try:
        same = (fo.n_features, fo.n_groups) == (orig_fo.n_features, orig_fo.n_groups) and all(
            same_bits(
                fo.predict(x, 1, base_margin=bm if use_bm else None, dtype=dt),
                orig_fo.predict(x, 1, base_margin=bm if use_bm else None, dtype=dt),
            )
            for dt, use_bm in ((np.float32, False), (np.float64, True))
        )
    except ValueError as e:  # a repaired header that refuses the base margin ("the scales differ") while the original takes it
        if not classify(e).startswith("CLEAN"):
            raise
        same = False
    return None if same else "LOADED a different forest (checksum repaired)"


def child_flat(cases: list[str], work: Path, emit):
    import evaluator

    cache = {}
    undamaged = {}  # base -> the undamaged file loaded, to compare the answers of a mutant that loads
    for spec in cases:
        _tgt, base, kind, seed = spec.split(":")
        emit("START", spec)
        if base not in cache:
            cache[base] = (work / "bases" / f"{base}.flat").read_bytes()
            undamaged[base] = evaluator.FlatForest(work / "bases" / f"{base}.flat")
        rng = np.random.default_rng([int(seed), KINDS.index(kind)])
        data = mutate(cache[base], kind, rng)
        path = work / "tmp" / f"fz-{os.getpid()}.flat"
        path.write_bytes(data)
        try:
            fo = evaluator.FlatForest(path)
        except BaseException as e:
            emit("RESULT", spec, classify(e))
            continue
        try:
            x = (np.random.default_rng(int(seed)).normal(size=(37, fo.n_features)) * 3).astype(np.float32)
            x[np.random.default_rng(int(seed) + 1).random(x.shape) < 0.15] = np.nan
            bm = np.random.default_rng(int(seed) + 2).uniform(-1, 1, (len(x), fo.n_groups))
            ref = None
            note = "LOADED"
            for eng in (0, 1, 2):
                fo.engine = eng
                for th, dt, use_bm in ((1, np.float32, False), (3, np.float32, False), (1, np.float64, True), (3, np.float64, False)):
                    try:
                        out = fo.predict(x, th, base_margin=bm if use_bm else None, dtype=dt).copy()
                    except evaluator.EngineRefused:
                        note = "LOADED (wave engines refuse)" if eng else note
                        continue
                    except ValueError as e:  # the library refusing a base margin with a message is a clean answer
                        if not use_bm or not classify(e).startswith("CLEAN"):
                            raise
                        note = "LOADED (base margin refused)"
                        continue
                    if eng == 0 and th == 1 and dt is np.float32:
                        ref = out
                    elif eng == 0 and th == 3 and dt is np.float32 and ref is not None and not same_bits(ref, out):
                        emit("RESULT", spec, "DIVERGE threads 3 differ from 1")
                    elif eng and dt is np.float32 and th == 1 and ref is not None and not same_bits(ref, out):
                        emit("RESULT", spec, f"DIVERGE engine {eng} differs from the walker")
            fo.engine = 0
            verdict = loaded_verdict(data, cache[base], fo, undamaged[base], x, bm)
            fo.close()
            emit("RESULT", spec, verdict or note)
        except BaseException as e:
            emit("RESULT", spec, "UNCLEAN after load: " + classify(e))


PROBE_CODE = """
import os, sys, json
from pathlib import Path
sys.path[:0] = json.loads(sys.argv[1])
reply = os.fdopen(int(sys.argv[2]), "w")
import numpy as np
import fuzz_loader as fz
import qf_formats
import resource
for lib in ("xgboost", "lightgbm"):
    try:
        __import__(lib)
    except ImportError:
        pass
vm_kb = next((int(ln.split()[1]) for ln in open("/proc/self/status") if ln.startswith("VmSize:")), 0)
if vm_kb:
    ceiling = vm_kb * 1024 + int(sys.argv[3]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (ceiling, ceiling))
for line in sys.stdin:
    try:
        source = qf_formats.open_source(Path(line.rstrip("\\n")))
        source.predict(np.zeros((1, source.spec.n_features), np.float32))
        verdict = "OK"
    except MemoryError:
        verdict = "SOURCE LIBRARY CRASH (out of memory under the probe's address-space ceiling)"
    except BaseException as e:
        verdict = fz.classify(e)
    reply.write(verdict.replace("\\n", " ") + "\\n")
    reply.flush()
"""


class SourceProbe:
    """The model's own library (xgboost, LightGBM) loading a text file in a process of its own, one request per line. A damaged file that
    QF's parser accepts can still abort or segfault the library; that is the library's crash, not QF's, so it is answered here as
    "SOURCE LIBRARY CRASH (...)" (counted apart, not a failure) and the next request starts a new process. `check(path)` answers "OK",
    a `classify` line, "SOURCE LIBRARY CRASH ..." or "HANG ..." (no answer in `timeout` seconds). Replies travel on their own pipe, as the
    libraries print to stdout."""

    def __init__(self, code: str = PROBE_CODE, timeout: float = 20.0):
        self.code, self.timeout, self.proc, self.fd = code, timeout, None, -1
        self.requests = 0  # answered since the worker started

    def _start(self):
        r, w = os.pipe()
        self.proc = subprocess.Popen(
            [sys.executable, "-s", "-c", self.code, json.dumps([str(HERE), *sys.path]), str(w), str(PROBE_HEADROOM_MB)],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            pass_fds=(w,),
        )
        os.close(w)
        self.fd = r
        self.requests = 0

    def _stop(self):
        p, self.proc = self.proc, None
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        if p is not None:
            try:
                p.stdin.close()
            except OSError:
                pass
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.kill(p.pid, signal.SIGKILL)  # exactly this process
                p.wait()

    def check(self, path) -> str:
        if self.proc is None:
            self._start()
        try:
            self.proc.stdin.write(f"{path}\n")
            self.proc.stdin.flush()
        except OSError:
            pass  # the process is gone: the read below sees the end of the pipe
        got = b""
        while not got.endswith(b"\n"):
            ready, _, _ = select.select([self.fd], [], [], self.timeout)
            if not ready:
                self._stop()
                return f"HANG the source library did not answer in {self.timeout:.0f} s"
            chunk = os.read(self.fd, 4096)
            if not chunk:  # the process died on this file
                rc = self.proc.wait()
                self._stop()
                return f"SOURCE LIBRARY CRASH (exit {rc}{', signal ' + signal.Signals(-rc).name if rc < 0 else ''})"
            got += chunk
        self.requests += 1
        rss = rss_mb(self.proc.pid)
        if self.requests >= PROBE_REQUESTS or (rss is not None and rss > PROBE_RSS_MB):
            self._stop()  # between two requests: the next one starts a fresh worker
        return got.decode(errors="replace").rstrip("\n")

    def close(self):
        self._stop()


def child_text(cases: list[str], work: Path, emit):
    import qf_common as qf
    import qf_formats

    cache = {}
    probe = SourceProbe()
    try:
        for spec in cases:
            tgt, base, _kind, seed = spec.split(":")
            emit("START", spec)
            src = work / "bases" / base
            if base not in cache:
                cache[base] = src.read_bytes()
            rng = np.random.default_rng([int(seed), 99])
            data = mutate_text(cache[base], rng, "lgb" if tgt == "lgb" else "xgb")
            path = work / "tmp" / f"fz-{os.getpid()}{src.suffix}"
            path.write_bytes(data)
            try:
                files = qf_formats.model_files(path)
                forest = qf.load_forest(files)
                import backends

                out = work / "tmp" / f"fz-{os.getpid()}.flat"
                backends.get("flat").compile(forest, out, {})
                import evaluator

                fo = evaluator.FlatForest(out)
                fo.predict(np.zeros((3, fo.n_features), np.float32))
                fo.close()
                verdict = probe.check(path)  # the model's own library must load it and answer a row too
                emit("RESULT", spec, "LOADED" if verdict == "OK" else verdict)
            except BaseException as e:
                emit("RESULT", spec, classify(e))
    finally:
        probe.close()


def child_npz(cases: list[str], work: Path, emit):
    import qf_common as qf

    rng0 = np.random.default_rng(5)
    for spec in cases:
        _tgt, _b, _kind, seed = spec.split(":")
        emit("START", spec)
        rng = np.random.default_rng([int(seed), 7])
        path = work / "tmp" / f"fz-{os.getpid()}.npz"
        x = rng0.normal(size=(50, 4)).astype(np.float32)
        arrays = {"X": x, "fold": np.zeros(50, np.int64), "slice": np.zeros(50, np.int8), "y": rng0.normal(size=50)}
        mode = int(rng.integers(0, 9))
        if mode == 0:
            np.savez(path, **{k: v for k, v in arrays.items() if k != "X"})
        elif mode == 1:
            np.savez(path, X=np.array(["a", "b"]))
        elif mode == 2:
            np.savez(path, X=np.array([{"a": 1}], dtype=object), allow_pickle=True)
        elif mode == 3:
            np.savez(path, X=x.reshape(-1), **{k: v for k, v in arrays.items() if k != "X"})
        elif mode == 4:
            np.savez(path, **arrays)
            raw = path.read_bytes()
            path.write_bytes(raw[: int(rng.integers(0, len(raw)))])
        elif mode == 5:
            np.savez(path, **arrays)
            b = bytearray(path.read_bytes())
            for _ in range(5):
                b[int(rng.integers(0, len(b)))] ^= 0xFF
            path.write_bytes(bytes(b))
        elif mode == 6:
            np.savez(path, X=x, fold=np.arange(30), slice=np.arange(7))
        elif mode == 7:
            np.savez(path, X=np.zeros((0, 4), np.float32))
        else:
            path.write_bytes(b"")
        try:
            d = qf.load_data(path)
            qf.slices(d, 0.3, 1)
            emit("RESULT", spec, "LOADED")
        except BaseException as e:
            emit("RESULT", spec, classify(e))


def child_choice(cases: list[str], work: Path, emit):
    import backends

    bases = work / "bases"
    for spec in cases:
        _tgt, _b, _k, seed = spec.split(":")
        emit("START", spec)
        rng = np.random.default_rng([int(seed), 11])
        art = work / "tmp" / f"fz-{os.getpid()}.flat"
        shutil.copyfile(bases / "xgb_multi.flat", art)
        choice = Path(f"{art}.choice.json")
        hostile = [
            b"",
            b"{",
            b"[]",
            b"null",
            b'{"backend": 3}',
            b'{"backend": "nope"}',
            b'{"backend": "../../x.py:Evil"}',
            b"{}",
            b'{"backend": ""}',
            b'{"backend": "wave"}',
            b'{"backend": "codegen-block"}',
            b'{"backend": null}',
            b"\xff\xfe",
            json.dumps({"backend": "x" * 100000}).encode(),
        ]
        choice.write_bytes(hostile[int(rng.integers(0, len(hostile)))])
        try:
            p = backends.load_artifact(art)
            if hasattr(p, "close"):
                p.close()
            emit("RESULT", spec, "LOADED")
        except BaseException as e:
            emit("RESULT", spec, classify(e))
        finally:
            choice.unlink(missing_ok=True)


def child_api(cases: list[str], work: Path, emit):
    import threading

    import evaluator

    base = work / "bases" / "xgb_multi.flat"
    nf = None

    def fresh():
        nonlocal nf
        fo = evaluator.FlatForest(base)
        nf = fo.n_features
        return fo

    fifo = work / "tmp" / f"fz-fifo-{os.getpid()}"
    if not fifo.exists():
        os.mkfifo(fifo)
    empty = work / "tmp" / "fz-empty"
    empty.write_bytes(b"")
    fo0 = fresh()
    g = fo0.n_groups
    ok_x = np.zeros((4, nf), np.float32)
    closed = fresh()
    closed.close()
    odd = [
        ("missing file", lambda: evaluator.FlatForest(work / "tmp" / "nope.flat"), None),
        ("directory", lambda: evaluator.FlatForest(work / "tmp"), None),
        ("empty file", lambda: evaluator.FlatForest(empty), None),
        ("fifo", lambda: evaluator.FlatForest(fifo), None),
        ("dev null", lambda: evaluator.FlatForest("/dev/null"), None),
        ("dev zero", lambda: evaluator.FlatForest("/dev/zero"), None),
        ("proc self mem", lambda: evaluator.FlatForest("/proc/self/mem"), None),
        ("NUL in path", lambda: evaluator.FlatForest(str(base) + "\0x"), "must raise"),
        ("non-UTF-8 path", lambda: evaluator.FlatForest(os.fsdecode(b"/tmp/\xff\xfe")), None),
        ("bad engine", lambda: evaluator.FlatForest(base, engine=7), "must raise"),
        ("engine None", lambda: evaluator.FlatForest(base, engine=None), "must raise"),
        ("x wrong features", lambda: fo0.predict(np.zeros((4, nf + 1), np.float32)), "must raise"),
        ("x 1-D", lambda: fo0.predict(np.zeros(nf, np.float32)), "must raise"),
        ("x 3-D", lambda: fo0.predict(np.zeros((2, 2, nf), np.float32)), "must raise"),
        ("x zero rows", lambda: fo0.predict(np.zeros((0, nf), np.float32)), None),
        ("x strings", lambda: fo0.predict(np.array([["a"] * nf])), "must raise"),
        ("x object", lambda: fo0.predict(np.array([[None] * nf], dtype=object)), None),
        ("x float64 cast", lambda: fo0.predict(np.zeros((3, nf))), None),
        ("x int", lambda: fo0.predict(np.zeros((3, nf), np.int64)), None),
        ("x fortran order", lambda: fo0.predict(np.asfortranarray(np.zeros((9, nf), np.float32))), None),
        ("x strided view", lambda: fo0.predict(np.zeros((9, nf * 2), np.float32)[:, ::2]), None),
        ("x read-only", lambda: fo0.predict(np.frombuffer(np.zeros((3, nf), np.float32).tobytes(), np.float32).reshape(3, nf)), None),
        ("x inf", lambda: fo0.predict(np.full((3, nf), np.inf, np.float32)), None),
        ("x list of lists", lambda: fo0.predict([[0.0] * nf, [1.0] * nf]), None),
        ("threads 0", lambda: fo0.predict(ok_x, 0), None),
        ("threads -1", lambda: fo0.predict(ok_x, -1), None),
        ("threads 2**40", lambda: fo0.predict(ok_x, 2**40), "any"),
        ("threads 10**6", lambda: fo0.predict(ok_x, 10**6), None),
        ("threads float", lambda: fo0.predict(ok_x, 1.5), "any"),
        ("dtype float16", lambda: fo0.predict(ok_x, dtype=np.float16), "must raise"),
        ("dtype str", lambda: fo0.predict(ok_x, dtype="nonsense"), "must raise"),
        ("margin wrong shape", lambda: fo0.predict(ok_x, base_margin=np.zeros((3, g))), "must raise"),
        ("margin nan", lambda: fo0.predict(ok_x, base_margin=np.full((4, g), np.nan)), "must raise"),
        ("margin inf", lambda: fo0.predict(ok_x, base_margin=np.full((4, g), np.inf)), "must raise"),
        ("margin strings", lambda: fo0.predict(ok_x, base_margin=np.array([["a"] * g] * 4)), "must raise"),
        ("margin empty", lambda: fo0.predict(ok_x, base_margin=np.zeros((0, g))), "must raise"),
        ("margin scalar", lambda: fo0.predict(ok_x, base_margin=1.0), "must raise"),
        ("predict after close", lambda: closed.predict(ok_x), "must raise"),
        ("double close", lambda: (closed.close(), closed.close()), None),
        ("huge zero-row 2-D", lambda: fo0.predict(np.zeros((0, nf), np.float32), base_margin=np.zeros((0, g))), None),
        ("lib argument types", lambda: evaluator.FlatForest(base, lib=work / "tmp" / "nolib.so"), "must raise"),
        ("lib is not a library", lambda: evaluator.FlatForest(base, lib=base), "must raise"),
    ]
    byname = {n: (f, req) for n, f, req in odd}

    def concurrent():
        errs = []

        def run(i):
            try:
                x = np.random.default_rng(i).normal(size=(500, nf)).astype(np.float32)
                ref = fo0.predict(x)
                for _ in range(20):
                    if not same_bits(fo0.predict(x, 2), ref):
                        errs.append("differs")
            except BaseException as e:
                errs.append(repr(e))

        ts = [threading.Thread(target=run, args=(i,)) for i in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        if errs:
            raise RuntimeError("concurrent predict: " + errs[0])

    byname["concurrent predict"] = (concurrent, None)
    for spec in cases:
        name = spec.split(":", 3)[1].replace("_", " ")
        emit("START", spec)
        f, req = byname[name]
        try:
            f()
            emit("RESULT", spec, "UNCLEAN accepted what must be refused" if req == "must raise" else "LOADED")
        except BaseException as e:
            r = classify(e)
            emit("RESULT", spec, r if req != "any" else r.replace("UNCLEAN", "CLEAN"))
    return [n for n in byname]


CHILD = {"flat": child_flat, "lgb": child_text, "xgb": child_text, "npz": child_npz, "choice": child_choice, "api": child_api}
API_NAMES = None


def child_main(args) -> int:
    work = Path(args.work)
    cases = Path(args.cases_file).read_text().split("\n")
    cases = [c for c in cases if c]
    out = sys.stdout

    started = {"cases": 0, "rss0": 0.0}

    def emit(*parts):
        if parts[0] == "START":  # between two cases: a process that has grown, or run long, hands over to a new one (run_children resumes)
            n = started["cases"]
            started["cases"] = n + 1
            if n == 1:
                started["rss0"] = rss_mb("self") or 0.0  # after the first case: libraries imported, caches warm
            elif n > 1 and (n >= CHILD_CASES or (rss_mb("self") or 0.0) - started["rss0"] > CHILD_GROWTH_MB):
                out.write(f"RECYCLE\t{parts[1]}\n")
                out.flush()
                os._exit(0)  # a source-library probe worker sees its stdin close and ends itself
        out.write("\t".join(parts) + "\n")
        out.flush()

    CHILD[cases[0].split(":")[0]](cases, work, emit)
    return 0


# ---------------------------------------------------------------------------------------------------------- bases and the parent
def build_bases(work: Path) -> list[str]:
    """Valid flat files of every shape the reader knows, plus a LightGBM text and an xgboost JSON model for the parsers."""
    import backends
    import qf_common as qf
    import qf_formats
    import xgboost as xgb

    bases = work / "bases"
    bases.mkdir(parents=True, exist_ok=True)
    (work / "tmp").mkdir(exist_ok=True)
    rng = np.random.default_rng(3)
    names = []

    def xgbm(name, obj, nf, rounds, depth, folds=1, **extra):
        d = bases / f"{name}-models"
        d.mkdir(exist_ok=True)
        for k in range(folds):
            x = rng.normal(size=(300, nf)).astype(np.float32)
            x[rng.random(x.shape) < 0.1] = np.nan
            z = np.nan_to_num(x)
            y = z[:, 0] + (z[:, 1] > 0)
            if obj.startswith("multi"):
                y = np.digitize(y, np.quantile(y, [0.33, 0.66]))
            elif obj.startswith("binary"):
                y = (y > np.median(y)).astype(float)
            p = {"objective": obj, "max_depth": depth, "eta": 0.3, "nthread": 1, "verbosity": 0, **extra}
            if obj.startswith("multi"):
                p["num_class"] = 3
            xgb.train(p, xgb.DMatrix(x, label=y), rounds).save_model(str(d / f"fold{k}.ubj"))
        f = qf.load_forest(qf_formats.model_files(d), "output")
        backends.get("flat").compile(f, bases / f"{name}.flat", {})
        names.append(name)

    xgbm("xgb_multi", "multi:softprob", 6, 4, 3)
    xgbm("xgb_bin", "binary:logistic", 5, 6, 4)
    xgbm("xgb_folds", "reg:squarederror", 5, 4, 3, folds=3)
    xgbm("xgb_stump", "reg:squarederror", 3, 1, 1)
    xgbm("xgb_deep", "reg:squarederror", 6, 3, 10)
    xgbm("xgb_exact_gamma", "binary:logistic", 6, 8, 6, tree_method="exact", gamma=1.0)  # keeps its dropped splits in the model's arrays
    xgbm("xgb_approx_gamma", "reg:squarederror", 5, 6, 5, tree_method="approx", gamma=0.5)
    import lightgbm as lgb

    x = rng.normal(size=(400, 6)).astype(np.float32)
    x[:, 5] = rng.integers(0, 12, 400)
    x[rng.random(x.shape) < 0.1] = np.nan
    z = np.nan_to_num(x)
    y = z[:, 0] + (z[:, 5] % 3 == 0) + z[:, 1] * 0.5
    common = {"num_leaves": 7, "verbose": -1, "min_data_in_leaf": 5, "num_threads": 1, "learning_rate": 0.2}
    for name, p, cat in (
        ("lgb_cat", {"objective": "regression", "zero_as_missing": True, "max_cat_to_onehot": 2, "min_data_per_group": 3}, [5]),
        ("lgb_linear", {"objective": "regression", "linear_tree": True, "linear_lambda": 0.1}, []),
        ("lgb_multi", {"objective": "multiclass", "num_class": 3}, []),
    ):
        yy = np.digitize(y, np.quantile(y, [0.33, 0.66])) if p["objective"] == "multiclass" else y
        b = lgb.train({**common, **p}, lgb.Dataset(x, yy, categorical_feature=cat or "auto"), 6)
        txt = bases / f"{name}.txt"
        b.save_model(str(txt))
        backends.get("flat").compile(qf.load_forest([txt], "output"), bases / f"{name}.flat", {})
        backends.get("flat").compile(qf.load_forest([txt], "output"), bases / f"{name}_f64.flat", {"f64_leaves": True})
        names += [name, f"{name}_f64"]
    try:
        import ydf

        d = {f"f{i}": x[:, i] for i in range(6)}
        ydf.GradientBoostedTreesLearner(label="y", task=ydf.Task.REGRESSION, num_trees=8, max_depth=4, num_threads=1).train(
            {**d, "y": y + 12}, verbose=0
        ).save(str(bases / "ydf-model"))
        backends.get("flat").compile(qf.load_forest([bases / "ydf-model"], "output", True), bases / "ydf_f32.flat", {})
        names.append("ydf_f32")
    except ImportError:
        pass
    # text / json models for the parsers
    d = bases / "xgb_bin-models"
    b = xgb.Booster()
    b.load_model(str(d / "fold0.ubj"))
    (bases / "xgb_bin.json").write_bytes(bytes(b.save_raw(raw_format="json")))
    b = xgb.Booster()
    b.load_model(str(bases / "xgb_exact_gamma-models" / "fold0.ubj"))
    (bases / "xgb_exact_gamma.json").write_bytes(bytes(b.save_raw(raw_format="json")))
    return names


def run_children(target: str, cases: list[str], work: Path, out: Path, tag: str) -> dict:
    """Run the cases in child processes; return {status: count} and append the non-clean lines to out/findings.txt."""
    counts: dict[str, int] = {}
    findings = out / "findings.txt"
    todo = list(cases)
    batch_file = work / "tmp" / f"batch-{target}.txt"
    t_start = time.time()
    while todo:
        batch_file.write_text("\n".join(todo) + "\n")
        env = {**os.environ, "PYTHONPATH": os.environ.get("PYTHONPATH", ""), "TREEFLAT_FORCE_GATHER": "1"}
        p = subprocess.Popen(
            [sys.executable, "-s", str(Path(__file__).resolve()), "--child", "--work", str(work), "--cases-file", str(batch_file)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )
        cur = None
        done_here = 0
        status = None
        recycle_at = None
        while True:
            r, _, _ = select.select([p.stdout], [], [], TIMEOUT)
            if not r:
                status = "HANG"
                break
            line = p.stdout.readline()
            if not line:
                break
            parts = line.rstrip("\n").split("\t")
            if parts[0] == "START":
                cur = parts[1]
            elif parts[0] == "RECYCLE":
                recycle_at = parts[1]
            elif parts[0] == "RESULT":
                done_here += 1
                res = parts[2]
                key = res.split(":")[0].split(" (")[0]
                counts[key] = counts.get(key, 0) + 1
                if res.startswith("SOURCE LIBRARY CRASH"):  # the library's own crash on a damaged file: counted apart, not a failure
                    with (out / "library-crashes.txt").open("a") as f:
                        f.write(f"{tag} {parts[1]} -> {res}\n")
                elif not res.startswith(("CLEAN", "LOADED")):
                    with findings.open("a") as f:
                        f.write(f"{tag} {parts[1]} -> {res}\n")
        if status == "HANG":
            os.kill(p.pid, signal.SIGKILL)  # exactly this child
        rc = p.wait()
        err = p.stderr.read()
        if status is None and rc == 0 and recycle_at in todo:  # the process stopped between two cases to give its memory back
            todo = todo[todo.index(recycle_at) :]
            continue
        if status is None and rc == 0 and done_here == len(todo):
            break
        if status is None:
            status = f"CRASH (exit {rc}{', signal ' + signal.Signals(-rc).name if rc < 0 else ''})"
        culprit = cur if cur else (todo[done_here] if done_here < len(todo) else "?")
        counts[status.split(" ")[0]] = counts.get(status.split(" ")[0], 0) + 1
        with findings.open("a") as f:
            f.write(f"{tag} {culprit} -> {status}  {err.strip()[-300:].replace(chr(10), ' | ')}\n")
        # continue after the culprit
        idx = todo.index(culprit) if culprit in todo else done_here
        todo = todo[idx + 1 :]
        if time.time() - t_start > 6 * 3600:
            break
    return counts


def count_non_clean(counts: dict) -> int:
    """How many results of a target fail the run. A key is the head of a verdict line ("UNCLEAN KeyError", "DIVERGE threads 3 differ from 1",
    "CRASH"), so the test is on its first word: comparing the whole key with "UNCLEAN" never matched, and a run full of them exited 0."""
    return sum(v for k, v in counts.items() if k.split(" ")[0] in ("CRASH", "HANG", "UNCLEAN", "DIVERGE"))


def make_cases(mode: str, bases: list[str], seed0: int, api_names: list[str]) -> dict[str, list[str]]:
    per = 6 if mode == "quick" else 220  # seeds per (base, kind)
    cases: dict[str, list[str]] = {"flat": [], "lgb": [], "xgb": [], "npz": [], "choice": [], "api": []}
    for base in bases:
        for kind in KINDS:
            n = per * (3 if kind in ("multi", "word-fix", "child", "hdr", "trunc") else 1)
            cases["flat"] += [f"flat:{base}:{kind}:{seed0 + s}" for s in range(n)]
    n_text = 150 if mode == "quick" else 6000
    cases["lgb"] = [f"lgb:{b}:text:{seed0 + s}" for b in ("lgb_cat.txt", "lgb_linear.txt", "lgb_multi.txt") for s in range(n_text)]
    cases["xgb"] = [f"xgb:{b}:text:{seed0 + s}" for b, k in (("xgb_bin.json", 2), ("xgb_exact_gamma.json", 1)) for s in range(n_text * k)]
    cases["npz"] = [f"npz:-:data:{seed0 + s}" for s in range(36 if mode == "quick" else 500)]
    cases["choice"] = [f"choice:-:json:{seed0 + s}" for s in range(28 if mode == "quick" else 280)]
    cases["api"] = [f"api:{n.replace(' ', '_')}:x:0" for n in api_names]
    return cases


API_CASE_NAMES = [
    "missing file",
    "directory",
    "empty file",
    "fifo",
    "dev null",
    "dev zero",
    "proc self mem",
    "NUL in path",
    "non-UTF-8 path",
    "bad engine",
    "engine None",
    "x wrong features",
    "x 1-D",
    "x 3-D",
    "x zero rows",
    "x strings",
    "x object",
    "x float64 cast",
    "x int",
    "x fortran order",
    "x strided view",
    "x read-only",
    "x inf",
    "x list of lists",
    "threads 0",
    "threads -1",
    "threads 2**40",
    "threads 10**6",
    "threads float",
    "dtype float16",
    "dtype str",
    "margin wrong shape",
    "margin nan",
    "margin inf",
    "margin strings",
    "margin empty",
    "margin scalar",
    "predict after close",
    "double close",
    "huge zero-row 2-D",
    "lib argument types",
    "lib is not a library",
    "concurrent predict",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--only", default=None, help="comma list of targets: flat,lgb,xgb,npz,choice,api")
    ap.add_argument("--replay", default=None, help="target:base:kind:seed -> write the mutated file into --out and load it")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--work", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cases-file", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        return child_main(args)
    out = Path(args.out)
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "findings.txt").exists():
        (out / "findings.txt").unlink()
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    bases = build_bases(work)
    if args.replay:
        tgt, base, kind, seed = args.replay.split(":")
        if tgt == "flat":
            data = mutate((work / "bases" / f"{base}.flat").read_bytes(), kind, np.random.default_rng([int(seed), KINDS.index(kind)]))
            (out / "replay.flat").write_bytes(data)
            print(f"wrote {out / 'replay.flat'} ({len(data)} bytes)")
        batch = work / "tmp" / "replay.txt"
        batch.write_text(args.replay + "\n")
        return subprocess.call([sys.executable, "-s", __file__, "--child", "--work", str(work), "--cases-file", str(batch)])
    cases = make_cases(args.mode, bases, args.seed0, API_CASE_NAMES)
    targets = args.only.split(",") if args.only else list(cases)
    t0 = time.time()
    summary = {}
    for t in targets:
        t1 = time.time()
        summary[t] = run_children(t, cases[t], work, out, t)
        print(f"{t:7s} {len(cases[t]):6d} cases  {time.time() - t1:6.1f} s   {summary[t]}", flush=True)
    bad = sum(count_non_clean(c) for c in summary.values())
    f = out / "findings.txt"
    lines = f.read_text().splitlines() if f.exists() else []
    print(f"\ntotal {sum(len(v) for k, v in cases.items() if k in targets)} cases in {time.time() - t0:.0f} s; non-clean: {bad}")
    for ln in lines[:60]:
        print("  " + ln[:300])
    if len(lines) > 60:
        print(f"  ... {len(lines) - 60} more in {f}")
    shutil.rmtree(work / "tmp", ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
