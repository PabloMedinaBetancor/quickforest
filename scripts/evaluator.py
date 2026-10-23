"""Runtime side of the flat forest: loads the Rust evaluator through ctypes and predicts. Needs numpy only (no xgboost)."""

from __future__ import annotations

import ctypes
import operator
import os
import platform
import struct
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB_NAME = f"libtreeflat-{platform.machine()}.so"
ERROR_BYTES = 4096 + 256  # the reason the library gives for refusing a file: it repeats the path (at most 4,096 bytes on Linux) before it


def find_lib(near: Path | None = None) -> Path:
    """TREEFLAT_LIB, else next to the model file, else the prebuilt one in the repository's bin/."""
    name = LIB_NAME
    candidates = [os.environ.get("TREEFLAT_LIB"), near and near.parent / name, HERE.parent / "bin" / name]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    machine = platform.machine()
    hint = (
        f"no prebuilt evaluator ships for {machine} (only x86_64 does): build it from source on this machine with `QUICK=1 rust/build.sh` "
        "(needs cargo and rustc; the Rust code is portable and the AVX2 engine is simply absent), or set TREEFLAT_LIB"
        if machine != "x86_64"
        else "build it with rust/build.sh or set TREEFLAT_LIB"
    )
    raise FileNotFoundError(f"{name} not found: {hint}")


_LOADED: dict[str, ctypes.CDLL] = {}  # content digest -> the library loaded from the first path that had it


def lib_digest(path: str) -> str:
    """The sha256 of the library's bytes, read each call (about 1 ms for 2 MB): a cache keyed by size and modification time would take a
    same-size rebuild whose mtime was restored (rsync -a) for the old build."""
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_lib(path: Path) -> ctypes.CDLL:
    """The evaluator library at `path`. Copies of the same build (every compiled model folder carries its own) share ONE loaded instance:
    dlopen maps a copy at another path as a new library, and each such instance that spawns a thread takes a pthread key that is never
    given back (Rust's std), of the 1,024 a process has: after about a thousand copies the process aborts with `out of TLS keys`.
    A different build (other content) is a different library."""
    path = os.path.abspath(path)  # a bare `libtreeflat.so` is searched by dlopen in the system paths, never in the current folder
    digest = lib_digest(path)
    if digest in _LOADED:
        return _LOADED[digest]
    lib = ctypes.CDLL(path)
    p, sz, u32, i32 = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_int
    # (name, result type, argument types): the ABI of rust/src/lib.rs. A library without one of them is not a build of this evaluator.
    abi = (
        ("tfl_open", p, [ctypes.c_char_p, ctypes.c_char_p, sz]),
        ("tfl_info", None, [p, ctypes.POINTER(u32)]),
        ("tfl_margin_scale", ctypes.c_double, [p]),
        ("tfl_predict", i32, [p, p, sz, p, u32]),
        ("tfl_predict_ex", i32, [p, p, sz, p, u32, u32]),
        ("tfl_engine_ok", i32, [u32]),
        ("tfl_predict_margin", i32, [p, p, sz, p, u32, u32, p]),
        ("tfl_predict_f64", i32, [p, p, sz, p, u32, u32, p]),
        ("tfl_fnv1a", ctypes.c_uint64, [ctypes.c_char_p, sz]),
        ("tfl_close", None, [p]),
    )
    try:
        for name, restype, argtypes in abi:
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = restype, argtypes
    except AttributeError as e:
        raise ValueError(f"{shown(path)} is not a build of this repository's evaluator ({e}): build it with rust/build.sh") from None
    _LOADED[digest] = lib
    return lib


def fnv1a(data: bytes, lib_path: Path | None = None) -> int:
    """The file checksum, computed by the evaluator's own implementation so that both sides agree by construction."""
    return int(load_lib(lib_path or find_lib()).tfl_fnv1a(data, len(data)))


def seal(image, lib_path: Path | None = None) -> bytes:
    """`image` (the bytes of a flat file) with its checksum field (header bytes 40..48) set to the file checksum: the FNV-1a 64 of the whole
    file with those eight bytes read as zero (rust/src/lib.rs `file_checksum`), computed by the library's own implementation. For code that
    edits a file on purpose and needs the loader to get past the checksum to the checks behind it (the fuzz harness, the tests)."""
    b = bytearray(image)
    if len(b) >= 64:
        b[40:48] = bytes(8)
        b[40:48] = struct.pack("<Q", fnv1a(bytes(b), lib_path))
    return bytes(b)


def shown(text) -> str:
    """A path (or a message that holds one) for an error line. A file name that is not valid UTF-8 is legal on Linux; it is shown with a
    backslash escape (`caf\\xe9`) because the raw characters cannot be written to a UTF-8 stream."""
    try:
        return str(text).encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")
    except UnicodeEncodeError:
        return str(text).encode("utf-8", "backslashreplace").decode("utf-8")


MAX_THREADS = 64  # the most threads one predict call uses (the library clamps to 64 as well)

# `threads=None` (the default of every predict) uses all usable cores, but only from this many rows on: below it one thread is used, because
# starting and joining a thread costs about 0.1 to 0.2 ms and a tiny forest walks 4,096 rows in less than that. MEASURED 2026-10-22 (flat and
# codegen-block, forests of 3, 30 and 200 trees, rows 1 to 100,000, one thread against every core, on a quiet 2-core laptop (Celeron N4020) and, indicatively,
# on a shared 8-thread one (i7-8550U); table in docs/REFERENCE.md "Threads"): from 8,192 rows the all-cores call was faster in every case (1.2x to
# 5.4x), at 4,096 a 3-tree forest still lost on the 2-core laptop (0.69x), and the library already stays on one thread up to one block of rows.
# Conservative on purpose (a larger forest gains from about 2,000 rows); not tuned further.
PARALLEL_MIN_ROWS = 8192


# Where a container's CPU quota is written (cgroup v2, then v1): `docker run --cpus 2` leaves every core visible to the affinity call but
# lets the process use two cores' worth of time, and more threads than that only wait to be scheduled.
CGROUP_CPU_FILES = (("/sys/fs/cgroup/cpu.max", None), ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "/sys/fs/cgroup/cpu/cpu.cfs_period_us"))


def cgroup_cpu_limit() -> int | None:
    """The CPU quota of this process's cgroup in whole CPUs (rounded up, at least 1), read from the mount root that a container sees as its
    own; None when there is no quota, or the files are missing or unreadable."""
    for quota_file, period_file in CGROUP_CPU_FILES:
        try:
            if period_file is None:  # v2: "max 100000" (no limit) or "200000 100000" (quota and period, in microseconds)
                quota, period = Path(quota_file).read_text().split()[:2]
            else:  # v1: the quota is -1 for no limit
                quota, period = Path(quota_file).read_text().strip(), Path(period_file).read_text().strip()
            if quota in ("max", "-1"):
                return None
            q, p = int(quota), int(period)
            if q > 0 and p > 0:
                return max(1, -(-q // p))
        except (OSError, ValueError):
            continue
    return None


def usable_cpus() -> int:
    """The number of CPUs this process may run on (`os.sched_getaffinity` where it exists, else `os.cpu_count()`), lowered to the cgroup CPU
    quota when there is one (`cgroup_cpu_limit`), at least 1 and at most MAX_THREADS."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 1
    quota = cgroup_cpu_limit()
    if quota is not None:
        n = min(n, quota)
    return max(1, min(n, MAX_THREADS))


def clamp_threads(threads) -> int:
    """The thread count of a predict call as the libraries take it: a whole number, at least 1 and at most 64 (the library clamps to 64 as
    well, and a huge int must not wrap in the C unsigned). Anything that is not a whole number is a ValueError."""
    try:
        return max(1, min(operator.index(threads), MAX_THREADS))
    except TypeError as e:
        raise ValueError(f"threads must be an integer, not {threads!r}") from e


def resolve_threads(threads, n_rows: int) -> int:
    """The thread count of one predict call. `None` (the default): one thread below PARALLEL_MIN_ROWS rows, else every usable core
    (`usable_cpus`). A whole number is used as given (clamped to 1..64), whatever the row count. The answers do not depend on it: they are
    bit-identical for any thread count."""
    if threads is None:
        return usable_cpus() if n_rows >= PARALLEL_MIN_ROWS else 1
    return clamp_threads(threads)


def output_dtype(dtype) -> np.dtype:
    """The answer type a predict call asks for: float32 (the default) or float64."""
    try:
        dt = np.dtype(dtype)
    except TypeError as e:
        raise ValueError(f"dtype must be np.float32 or np.float64, not {dtype!r}") from e
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"dtype must be np.float32 or np.float64, not {dt}")
    return dt


def plain_numbers(a, what: str):
    """`a` itself when it holds real, unmasked numbers. A complex array would be cast to float32 with its imaginary part dropped, and a masked
    array would be read through its mask (the raw data under it): both would answer from numbers the caller did not mean, so both are refused."""
    if isinstance(a, np.ma.MaskedArray):
        raise ValueError(f"{what} is a masked array: fill the masked values first (np.ma.filled)")
    if np.iscomplexobj(a):
        raise ValueError(f"{what} is complex: pass real numbers")
    return a


def float32_rows(x) -> np.ndarray:
    """`x` as a contiguous, ALIGNED float32 array: numpy may hold a view of a byte buffer at an odd address, and the library builds slices
    from the pointer (a misaligned slice is undefined behaviour in Rust)."""
    x = np.ascontiguousarray(plain_numbers(x, "the rows"), dtype=np.float32)
    return x if x.flags.aligned else x.copy()


def margin_array(base_margin, rows: int, groups: int, scale: float = 1.0) -> np.ndarray:
    """A base margin as the contiguous float64 (rows, groups) the evaluators take; (rows,) is accepted for one group. `scale`: the forest's
    margin scale (LightGBM `sigmoid:s` is folded into the leaves, so the margin, which is in raw-score units, is multiplied by it)."""
    bm = np.ascontiguousarray(plain_numbers(base_margin, "base_margin"), dtype=np.float64)
    if not bm.flags.aligned:
        bm = bm.copy()
    if bm.ndim == 1 and groups == 1:
        bm = bm.reshape(-1, 1)
    if bm.shape != (rows, groups):
        raise ValueError(f"base_margin must have shape ({rows}, {groups}), not {bm.shape}")
    if not np.isfinite(bm).all():
        raise ValueError("base_margin must be finite")
    if scale != 1.0:
        if not (np.isfinite(scale) and scale > 0):
            raise ValueError(
                "this forest averages models with different sigmoid scales (or its scale is unreadable): a base margin is ambiguous"
            )
        bm = bm * scale
        if not np.isfinite(bm).all():
            raise ValueError("base_margin times the forest's sigmoid scale is not finite")
    return bm


# A finite double no real leaf fallback is. In a linear-format file (the header bytes 56..64 hold counts there) `backend_flat.pack` stores
# the margin scale as the constant of a record with this fallback and no coefficients, after the last leaf record; the library finds it.
_SCALE_MARKER_BITS = 0x5153_4341_4C45_5343


def margin_scale_marker() -> float:
    return struct.unpack("<d", struct.pack("<Q", _SCALE_MARKER_BITS))[0]


class EngineRefused(RuntimeError):
    """The wavefront engines (1, 2) were asked to run a file they cannot hold. The C ABI only returns a code (1); this says what it means.
    A RuntimeError."""


class NotCopyable:
    """A loaded forest owns a library handle: `copy.copy` would share the raw handle (two `tfl_close` calls free it twice, and a copy used
    after the original is closed reads freed memory), so copying, deep-copying and pickling are refused with one line. Load the file again."""

    def _refuse(self, *_):
        raise TypeError(f"{type(self).__name__} owns a library handle and cannot be copied or pickled: load the file again")

    __copy__ = __deepcopy__ = __reduce__ = _refuse


class FlatForest(NotCopyable):
    """predict(X) answers like xgboost's `predict`: (rows,) for one output, (rows, classes) otherwise.
    `engine`: 0 lock-step walker, 1 scalar wavefront, 2 AVX2 wavefront (the scalar one on a CPU without AVX2; `engine_used` says)."""

    def __init__(self, path, lib: Path | None = None, engine: int = 0):
        self._h = None  # __del__ runs even when the library below could not be opened
        self._busy = 0  # predictions in flight; close() waits for them (ctypes drops the GIL during the walk)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self.path = Path(path)
        if "\0" in str(path):
            raise ValueError(f"{str(path)!r}: a file name cannot hold a NUL character")
        if engine not in (0, 1, 2):
            raise ValueError(f"engine must be 0 (lock-step walker), 1 (scalar wavefront) or 2 (AVX2 wavefront), not {engine!r}")
        self._lib = load_lib(lib or find_lib(self.path))
        self.engine = engine
        self.engine_used = engine if engine == 0 or self._lib.tfl_engine_ok(engine) else 1
        err = ctypes.create_string_buffer(ERROR_BYTES)
        self._h = self._lib.tfl_open(os.fsencode(self.path), err, len(err))
        if not self._h:
            raise ValueError(f"{shown(self.path)}: {err.value.decode(errors='replace')}")
        info = (ctypes.c_uint32 * 7)()
        self._lib.tfl_info(self._h, info)
        self.n_features, self.n_groups, self.n_folds, self.n_trees, self.n_nodes, self.objective, self.average = info
        self.margin_scale = self._lib.tfl_margin_scale(self._h)  # read by the library from the image it verified (rust/src/lib.rs)

    def predict(self, x, threads: int | None = None, base_margin=None, dtype=np.float32) -> np.ndarray:
        """`threads`: None (default) = every usable core from PARALLEL_MIN_ROWS rows on, one below; an integer forces that many (1..64).
        `base_margin`: (rows, groups) or (rows,) numbers ADDED to every model's raw score before the objective's transform (the base
        model of a boosted residual, an offset); float64 end to end. `dtype=np.float64` returns the answer without the final rounding to
        float32: for a double-sum forest (LightGBM, YDF) the sum itself, for an xgboost forest (float32 arithmetic) its float32 result
        widened."""
        with self._lock:
            h = self._h
            if h:
                self._busy += 1
        if not h:
            raise ValueError("this forest was closed")  # a null handle would crash the process in the library
        try:
            x = float32_rows(x)
            if x.ndim != 2 or x.shape[1] != self.n_features:
                raise ValueError(f"expected rows of {self.n_features} features")
            threads = resolve_threads(threads, len(x))
            dtype = output_dtype(dtype)
            out = np.empty((len(x), self.n_groups), dtype=dtype)
            if dtype == np.float64:
                bm = None if base_margin is None else margin_array(base_margin, len(x), self.n_groups, self.margin_scale)
                bad = self._lib.tfl_predict_f64(
                    h, x.ctypes.data, len(x), out.ctypes.data, threads, self.engine, None if bm is None else bm.ctypes.data
                )
            elif base_margin is not None:
                bm = margin_array(base_margin, len(x), self.n_groups, self.margin_scale)
                bad = self._lib.tfl_predict_margin(h, x.ctypes.data, len(x), out.ctypes.data, threads, self.engine, bm.ctypes.data)
            elif self.engine == 0:
                bad = self._lib.tfl_predict(h, x.ctypes.data, len(x), out.ctypes.data, threads)
            else:
                bad = self._lib.tfl_predict_ex(h, x.ctypes.data, len(x), out.ctypes.data, threads, self.engine)
            if bad == 1 and self.engine:  # the shapes were checked above, so the library can only be refusing the file for the wave engines
                raise EngineRefused(
                    f"{shown(self.path.name)}: engine {self.engine} (the wavefront engines) cannot evaluate this file, which holds "
                    f"double-precision sums "
                    "(a LightGBM forest, or a YDF one not compiled with --float32-sums), extended nodes (categorical or zero-as-missing "
                    "splits) "
                    "or linear leaves; use engine=0 (the lock-step walker: backend flat) or the generated code (codegen-block)"
                )
            if bad:
                raise RuntimeError(f"tfl_predict failed ({bad})")
            return out[:, 0] if self.n_groups == 1 else out
        finally:
            with self._lock:
                self._busy -= 1
                if not self._busy and self._h is None:
                    self._idle.notify_all()  # a close() is waiting for the last prediction

    def close(self):
        """Frees the forest. New predictions are refused from now on; one already running in another thread finishes first (close waits
        for it), so a service can close the forest it just replaced while requests are still in flight."""
        with self._idle:
            h, self._h = self._h, None
            while self._busy:
                self._idle.wait()
        if h:
            self._lib.tfl_close(h)

    def __del__(self):
        self.close()
