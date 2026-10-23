"""Shared by the quickforest commands: reading xgboost models, the decision rule, the selection data, and the neutral
forest description (`Forest`) that every compile backend consumes, so that a backend never has to parse model files.
LightGBM and YDF models are read by the adapters in qf_lgb.py and qf_ydf.py through the `Source` interface (qf_formats.py)."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import re
import secrets
import shutil
import struct
import sys
import tokenize
import zipfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The numbers the flat file's header and the generated crate's metadata give the output transform and the fold averaging.
OBJECTIVE = {"identity": 0, "sigmoid": 1, "softmax": 2, "exp": 3}
AVERAGE = {"output": 0, "margin": 1}
# Everything `Forest.requires` can list: what a backend that holds every forest declares as its `supports`.
FEATURES = frozenset({"sum64", "objective:exp", "objective:identity-with-several-groups", "zero_missing", "categorical", "linear_leaves"})

KINDS = {"multi:softprob": "softmax", "binary:logistic": "sigmoid", "reg:logistic": "sigmoid", "reg:squarederror": "identity"}


def odd_names_are_printable() -> None:
    """A file or folder name that is not valid UTF-8 (a Latin-1 `caf\\xe9` from an old archive) is legal on Linux: print it with a backslash
    escape instead of failing on the way out, after the work was done (stderr already does)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def library_path(path) -> str:
    """`str(path)` for a model library (xgboost, LightGBM, YDF) that reads and writes UTF-8 paths only: a name that is not valid UTF-8 ends
    in one line that says so, not in the library's own UnicodeEncodeError."""
    text = str(path)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(
            f"{text.encode('utf-8', 'backslashreplace').decode()}: path is not UTF-8 (the model libraries take only UTF-8 paths)"
        ) from None
    return text


def require_utf8_path(path, what: str = "the output") -> None:
    """The evaluator library opens UTF-8 paths only. A compiled forest written under a name that is not valid UTF-8 could be written but
    never opened (`check` and `optimize --verify` would refuse it), so such an output path is refused before anything is written."""
    raw = os.fsencode(path)
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(
            f"{raw.decode('utf-8', 'backslashreplace')}: {what} path is not valid UTF-8 (the evaluator opens only UTF-8 paths)"
        ) from None


def utf8(text: str) -> bytes:
    """`text.encode()` for text that holds a path: a name that is not valid UTF-8 is written with its backslash escape, never a crash."""
    return text.encode("utf-8", "backslashreplace")


def replace_file(dest, data: bytes | None = None, src=None) -> None:
    """Writes `dest` as a NEW file (a temp name, then a rename over the old one), never in place: the flat file is memory-mapped and the
    generated `.so` is dlopen-ed, so a process that has the old one loaded (a service, or this very process) must keep seeing the old
    bytes; rewriting the same inode in place would give it SIGBUS or a different, unvalidated forest."""
    dest = Path(dest)
    # a temp name of its own: builds run in parallel into one work folder (compare) and all copy the same `libtreeflat` there, so a
    # shared `dest.new` would let one writer truncate the other's file and make the second rename fail
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.new")
    try:
        if data is not None:
            tmp.write_bytes(data)
        else:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def copy_unless_same(src, dest) -> None:
    """Puts a copy of the file `src` at `dest` (as a new file, see `replace_file`) unless `dest` already holds identical bytes. The bytes are
    compared directly: `filecmp.cmp` caches its answer by size and modification time, which a rewrite within one clock tick leaves unchanged."""
    dest = Path(dest)
    if not (dest.exists() and dest.read_bytes() == Path(src).read_bytes()):
        replace_file(dest, src=src)


def replace_via_temp(dest, write) -> None:
    """`replace_file` for a writer that takes a path (xgboost's `save_model` picks the format from the suffix): `write(tmp)` makes the file
    at a name of its own beside `dest` that keeps `dest`'s suffix, which is then renamed over `dest`, so a link at `dest` is replaced,
    never followed."""
    dest = Path(dest)
    tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.{secrets.token_hex(4)}.new{dest.suffix}")
    try:
        write(tmp)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def replace_dir(dest, write) -> None:
    """The same for a model that is a folder (YDF): `write(tmp)` fills a new folder beside `dest`, which then takes the place of whatever
    `dest` was (a link is removed, not followed; an old folder is removed with its contents)."""
    dest = Path(dest)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.new")
    try:
        write(tmp)
        if dest.is_symlink() or dest.is_file():
            dest.unlink()
        elif dest.is_dir():
            shutil.rmtree(dest)
        os.replace(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def one_line(message) -> str:
    """A library's error text as one line: xgboost appends a native backtrace ("Stack trace:" and `[bt]` lines) to its message, and the
    lines of any other multi-line message are joined."""
    return " ".join(str(message).split("Stack trace:")[0].split())


def xgb():
    import xgboost

    return xgboost


def load_booster(path):
    p = Path(path)
    name = library_path(p)  # a name that is not UTF-8 is refused first, with its own message
    if p.suffix == ".ubj" and p.is_file():
        with open(p, "rb") as fh:  # xgboost segfaults on a cut UBJSON file: an object ends with its closing brace
            fh.seek(max(0, p.stat().st_size - 1))
            if fh.read(1) != b"}":
                raise ValueError(f"{p.name}: not a complete UBJSON model (truncated?)")
    b = xgb().Booster()
    try:
        b.load_model(name)
    except xgb().core.XGBoostError as e:
        raise ValueError(f"{p.name}: xgboost cannot load this model ({one_line(e)[:300]})") from None
    return b


DART_REFUSAL = "a dart model (per-tree weights, `weight_drop`) is not supported: train with plain gbtree, or rate_drop 0"


_UBJ_WEIGHT_DROP_KEY = re.compile(rb"(?:[iU]|I\x00|l\x00{3}|L\x00{7})\x0bweight_drop")


def raw_json(path) -> bytes:
    p = Path(path)
    if p.suffix == ".json":
        return p.read_bytes()
    # the library rewrites the file as JSON: one older than the writer drops the `weight_drop` of a dart model (3.4 and later) and would
    # hand back a plain gbtree, so a UBJSON file is searched for the key (11 as a length in any integer width, then the name) before the
    # library sees it; xgboost writes the length as an 8-byte `L`
    if _UBJ_WEIGHT_DROP_KEY.search(p.read_bytes()):
        raise ValueError(DART_REFUSAL + " (a .ubj file with a feature called weight_drop: save it as .json)")
    return bytes(load_booster(p).save_raw(raw_format="json"))


@dataclass
class Spec:
    """What pruning needs to know about one model (parsed once from its JSON)."""

    n_features: int
    groups: int
    kind: str  # softmax | sigmoid | identity
    base: np.ndarray  # float32 base margin per group
    tree_info: np.ndarray
    nodes: np.ndarray  # nodes per tree that a row can reach (the entries of `vals` of a tree can be more: see `off`)
    off: np.ndarray  # first entry of each tree in `vals`
    vals: np.ndarray  # split_conditions of every node (the leaf value at a leaf), float32
    parallel: int
    rf: bool = field(init=False)
    task: str = "auto"  # classification | regression; auto = regression for identity outputs (xgboost's only case)
    mean_type: bool | None = None  # trees are averaged by the format itself (LightGBM average_output, YDF random forest)

    def __post_init__(self):
        if self.n_features < 1:  # xgboost loads (does not train) a model that says so; nothing could be asked of it
            raise ValueError("the model has no input features (num_feature is 0): there is nothing to compile")
        self.rf = self.parallel > 1 if self.mean_type is None else self.mean_type
        if self.task == "auto":
            self.task = "regression" if self.kind == "identity" else "classification"


def _logf32(v: np.ndarray) -> np.ndarray:
    """log of float32 values, as the platform's own `logf` gives it (xgboost calls `logf`). numpy's float32 `log` is its own SIMD routine
    and differs from `logf` by an ulp for one input in six; where libm cannot be loaded the double log rounded to float32 is used (an
    ulp away from `logf` for one input in 200)."""
    try:
        import ctypes
        import ctypes.util

        lib = ctypes.CDLL(ctypes.util.find_library("m") or "libm.so.6")
        lib.logf.restype, lib.logf.argtypes = ctypes.c_float, [ctypes.c_float]
        return np.array([lib.logf(float(t)) for t in np.asarray(v, dtype=np.float32).ravel()], dtype=np.float32).reshape(np.shape(v))
    except (OSError, AttributeError, TypeError):
        return np.log(np.asarray(v, dtype=np.float64)).astype(np.float32)


def base_margin(param: dict, objective: str, groups: int) -> np.ndarray:
    raw = json.loads(param["base_score"])
    base = np.array(raw if isinstance(raw, list) else [raw], dtype=np.float32)
    # stored as a probability; xgboost works with its logit of the probability held in [1e-6, 1 - 1e-6]
    if objective in (
        "binary:logistic",
        "reg:logistic",
    ):
        eps = np.float32(1e-6)
        base = np.clip(base, eps, np.float32(1) - eps)
        base = -_logf32(np.float32(1) / base - np.float32(1))
    if len(base) == 1:
        base = np.repeat(base, groups)
    if len(base) != groups:
        raise ValueError("base_score does not match the number of groups")
    return base.astype(np.float32)


def check_xgb_tree(t: dict, n_features: int, i: int) -> None:
    """One xgboost JSON tree is a real tree: equal-length node arrays, children after their parent (so no cycle), at most one parent per
    node, a leaf has no children at all, a split reads a feature of the model. A broken one would loop or read outside the arrays in the
    compiler."""
    left, right = np.asarray(t["left_children"], np.int64), np.asarray(t["right_children"], np.int64)
    n = len(left)
    if n == 0 or not (len(right) == len(t["split_indices"]) == len(t["split_conditions"]) == len(t["default_left"]) == n):
        raise ValueError(f"tree {i}: the node arrays are empty or of different lengths")
    inner = left != -1
    if ((right != -1) != inner).any():
        raise ValueError(f"tree {i}: a node has one child only")
    parents = np.nonzero(inner)[0]
    lk, rk = left[parents], right[parents]
    if (lk <= parents).any() or (rk <= parents).any() or (lk >= n).any() or (rk >= n).any():
        raise ValueError(f"tree {i}: a child index is outside the tree or not after its parent")
    if len(parents) and np.bincount(np.concatenate([lk, rk]), minlength=n).max() > 1:
        raise ValueError(f"tree {i}: a node has two parents")
    si = np.asarray(t["split_indices"], np.int64)[parents]
    if len(si) and (si.min() < 0 or si.max() >= n_features):
        raise ValueError(f"tree {i}: a split reads a feature the model does not have")


def reachable_nodes(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Which nodes of one tree a row can reach from the root (children come after their parent, as `check_xgb_tree` requires). xgboost keeps
    the splits it dropped after growing them (gamma > 0 with the exact and approx methods: `num_deleted`) in the arrays, where no path
    leads."""
    n = len(left)
    if 2 * int((left != -1).sum()) == n - 1:  # every node after the root has a parent that comes before it: all are reachable
        return np.ones(n, bool)
    lt, rt = left.tolist(), right.tolist()
    seen = np.zeros(n, bool)
    seen[0] = True
    for i in range(n):
        if seen[i] and lt[i] != -1:
            seen[lt[i]] = seen[rt[i]] = True
    return seen


def model_header(raw: dict):
    """What an xgboost model document says apart from its trees: (objective, kind, groups, learner_model_param, model)."""
    lrn = raw["learner"]
    objective = lrn["objective"]["name"]
    if objective not in KINDS:
        raise ValueError(f"objective {objective} is not supported (use one of {sorted(KINDS)})")
    gb = lrn["gradient_booster"]
    if gb["name"] != "gbtree":
        raise ValueError(f"booster {gb['name']} is not supported (gbtree, including rf, only)")
    # dart keeps its per-tree weights in `weight_drop` and predict scales each tree by its weight. xgboost 3.4 writes a dart model as name
    # "gbtree" (booster=dart is deprecated; dropout is a parameter of gbtree), with `weight_drop` inside "model"; 3.2 wrote name "dart" with
    # it beside "gbtree". The file's name does not say, so the weights do; reading the trees alone would be a silent wrong answer.
    if "weight_drop" in gb or (isinstance(gb.get("model"), dict) and "weight_drop" in gb["model"]):
        raise ValueError(DART_REFUSAL)
    param = lrn["learner_model_param"]
    kind = KINDS[objective]
    groups = int(param["num_class"]) if kind == "softmax" else 1
    if kind == "softmax" and groups < 2:  # the softmax of one logit is a constant 1.0; the compiled file would be one the loader refuses
        raise ValueError(f"{objective} with num_class={groups} has a single class: nothing to classify (train with at least 2 classes)")
    return objective, kind, groups, param, gb["model"]


def make_spec(header, n_trees: int, nodes, off, vals) -> Spec:
    """The Spec of a model from its header and what was read of its trees (`nodes`: reachable nodes per tree, `off`: first entry of each
    tree in `vals`, `vals`: the split_conditions of every entry)."""
    objective, kind, groups, param, model = header
    if len(model["tree_info"]) != n_trees:
        raise ValueError(f"tree_info holds {len(model['tree_info'])} entries for {n_trees} trees")
    info = np.asarray(model["tree_info"], dtype=np.int64)
    if len(info) and (info.min() < 0 or info.max() >= groups):
        raise ValueError("tree_info names an output group the model does not have")
    return Spec(
        n_features=int(param["num_feature"]),
        groups=groups,
        kind=kind,
        base=base_margin(param, objective, groups),
        tree_info=info,
        nodes=nodes,
        off=off,
        vals=vals,
        parallel=int(model["gbtree_model_param"].get("num_parallel_tree", "1")),
    )


def spec_from_header(raw: dict, nodes, off, n_trees: int) -> Spec:
    """The Spec of a model whose trees were read apart from its document (the stream path: `select_stream`), which keeps no split_conditions table."""
    return make_spec(model_header(raw), n_trees, nodes, off, np.zeros(0, np.float32))


def parse_model(raw: dict) -> tuple[Spec, list[dict]]:
    header = model_header(raw)
    param, model = header[3], header[4]
    trees = model["trees"]
    if not trees:
        raise ValueError("the model has no trees (trained for zero rounds?)")
    for t in trees:
        if any(s != 0 for s in t["split_type"]) or t["categories"]:
            raise ValueError("categorical splits are not supported")
        if int(t["tree_param"].get("size_leaf_vector", "1")) > 1:
            raise ValueError("vector leaves are not supported")
    for i, t in enumerate(trees):
        check_xgb_tree(t, int(param["num_feature"]), i)
    sizes = np.array([len(t["left_children"]) for t in trees], dtype=np.int64)
    # `vals` holds every entry of the arrays, reachable or not: pred_leaf names a node by its array index
    off = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    nodes = np.array(
        [int(reachable_nodes(np.asarray(t["left_children"]), np.asarray(t["right_children"])).sum()) for t in trees], dtype=np.int64
    )
    vals = np.concatenate([np.asarray(t["split_conditions"], dtype=np.float32) for t in trees])
    return make_spec(header, len(trees), nodes, off, vals), trees


def transform(m: np.ndarray, kind: str) -> np.ndarray:
    """Margins (..., G) -> outputs (..., G) in float32, the way xgboost does: the arithmetic of `transform_f64` on float32 numbers."""
    return transform_f64(m.astype(np.float32, copy=False), kind)


def leaf_matrix(booster, spec: Spec, x: np.ndarray, chunk: int | None = None) -> np.ndarray:
    """(trees, rows) leaf values: one pred_leaf pass plus a table lookup (the cost of one prediction). The rows go through in chunks: a
    chunk holds its leaf numbers three times over (32 bits, then 64 twice for the lookup), 24 bytes a (tree, row), so the default chunk keeps
    that near 200 MB; 20,000 rows at most, as before."""
    out = np.empty((len(spec.nodes), len(x)), dtype=np.float32)
    chunk = chunk or max(256, min(20000, (200 << 20) // (24 * len(spec.nodes))))
    for a in range(0, len(x), chunk):
        part = np.ascontiguousarray(x[a : a + chunk], dtype=np.float32)
        # validate_features=False: a model trained on a DataFrame stores feature names and the data file is a bare matrix
        leaf = booster.predict(xgb().DMatrix(part), pred_leaf=True, validate_features=False).reshape(len(part), -1).astype(np.int64)
        out[:, a : a + chunk] = spec.vals[spec.off[None, :] + leaf].T
    return out


def predict_timed(booster, x: np.ndarray) -> np.ndarray:
    """The prediction alone, for the speed figures: `inplace_predict` is xgboost's own path without a DMatrix (honours the booster's
    nthread). `Booster.predict(DMatrix(x))` would build the DMatrix inside the timed call (0.3 to 18 ms for 3,000 rows, on all threads
    whatever the booster's nthread), which makes the "x xgboost" ratio of a small forest too good."""
    return booster.inplace_predict(np.ascontiguousarray(x, dtype=np.float32))


def predict(
    booster, x: np.ndarray, margin: bool = False, base_margin: np.ndarray | None = None, base: np.ndarray | None = None
) -> np.ndarray:
    """xgboost's own answer. `base_margin` (rows, groups) is ADDED to the model's margin, as everywhere in this repository: xgboost's
    DMatrix base_margin REPLACES the model's base_score, so the model's base margin `base` (Spec.base) is added to it here."""
    dm = xgb().DMatrix(np.ascontiguousarray(x, dtype=np.float32))
    if base_margin is not None:
        bm = np.asarray(base_margin, dtype=np.float64).reshape(len(x), -1) + np.asarray(base, dtype=np.float64)
        dm.set_base_margin(bm.astype(np.float32).reshape(-1))
    p = booster.predict(dm, output_margin=margin, validate_features=False)  # a model with feature names, a bare matrix
    return p.reshape(len(x), -1).astype(np.float32)


def finite_float(text: str) -> float:
    """argparse type for a bar or a share: a number that is neither NaN nor infinite (a NaN bar compares False with everything, so every
    model would pass it)."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"{text!r}: give a finite number")
    return v


def threads_arg(text: str):
    """argparse type for `--threads`: `all` (the default of every command) is None, meaning every usable core (`evaluator.resolve_threads`);
    a whole number forces that many threads (clamped to 1..64 by the evaluator)."""
    if text.strip().lower() == "all":
        return None
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number or 'all'") from None


def count_arg(minimum: int):
    """argparse type for a count of rows: an integer of at least `minimum` (a negative count would make numpy allocate a negative
    dimension)."""

    def parse(text: str) -> int:
        try:
            v = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from None
        if v < minimum:
            raise argparse.ArgumentTypeError(f"{v} is below the minimum of {minimum}")
        return v

    return parse


# ---------------------------------------------------------------------------------------------------- decisions
def make_rule(text: str, groups: int, kind: str, p_full: np.ndarray | None = None, tol: float | None = None, task: str = "auto"):
    """A decision rule: outputs (..., G) -> integer decisions (...). `argmax` (0.5 for one output), `threshold:T` (one output),
    `path/to/rule.py[:func]` (func(p) -> ints, p of shape (..., G): the outputs are the LAST axis, the leading axes are rows or candidates x rows), and for regression `tol` bins of width tol."""
    if (kind == "identity" and task == "auto") or task == "regression":
        width = tol or float(0.1 * np.std(p_full)) or 1.0
        return lambda p: np.rint(p[..., 0] / width).astype(np.int64)
    if text == "argmax":
        if groups == 1:
            return lambda p: (p[..., 0] >= 0.5).astype(np.int64)
        return lambda p: p.argmax(-1)
    if text.startswith("threshold:"):
        if groups != 1:
            raise ValueError("--decision threshold:T needs a single output; use a rule file for several classes")
        try:
            t = float(text.split(":", 1)[1])
        except ValueError:
            raise ValueError(f"--decision {text!r}: threshold:T needs a number T") from None
        # `p >= nan` is False for every row and `p >= -inf` True: one class for all, and an agreement of 100 % by construction
        if not math.isfinite(t):
            raise ValueError(f"--decision {text!r}: threshold:T needs a finite number T")
        return lambda p: (p[..., 0] >= t).astype(np.int64)
    path, _, name = text.partition(":")
    if not Path(path).is_file():
        raise ValueError(f"--decision {text!r}: not argmax, threshold:T or the path of a rule file ({path} is not a file)")
    spec = importlib.util.spec_from_file_location("qf_rule", path)
    if spec is None:
        raise ValueError(f"--decision {text!r}: {path} is not a Python file")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:  # the user's own rule file failed to load
        raise ValueError(f"--decision {text!r}: loading {path} failed ({type(e).__name__}: {e})") from e
    if not callable(getattr(mod, name or "decide", None)):
        raise ValueError(f"--decision {text!r}: {path} has no function {name or 'decide'}")
    return checked_rule(getattr(mod, name or "decide"), groups, f"--decision {text!r}")


def checked_rule(fn, groups: int, label: str):
    """The user's rule file as a rule the tools can trust: every call is checked (an exception, a result of the wrong shape or one that
    is not made of integers becomes one ValueError line), and it is tried once here, on a small (candidates, rows, groups) array and on a
    (rows, groups) one, because the greedy selection calls it with the first and check/prune's held-out figures with the second. A rule
    that marks an abstention with NaN is refused: NaN == NaN is False, so every row both forests abstain on would count as a disagreement."""

    def rule(p):
        try:
            out = np.asarray(fn(p))
        except Exception as e:  # the user's own code
            raise ValueError(f"{label}: the rule failed on an array of shape {p.shape} ({type(e).__name__}: {one_line(e)})") from None
        want = p.shape[:-1]
        if out.shape != want:
            raise ValueError(
                f"{label}: the rule returned shape {out.shape} for an input of shape {p.shape}; it must return one integer per row, shape "
                f"{want} (the last axis of the input holds the {groups} outputs: write p[..., 0] and p.argmax(-1), not p[:, 0])"
            )
        if out.dtype.kind in "biu":
            return out.astype(np.int64, copy=False)
        if out.dtype.kind == "f" and np.isnan(out).any():
            raise ValueError(
                f"{label}: the rule returned NaN; mark an abstention with an integer code such as -1 (NaN == NaN is False, so a row both "
                "forests abstain on would count as a disagreement)"
            )
        raise ValueError(f"{label}: the rule must return integers (a bool or int array); it returned dtype {out.dtype}")

    probe = np.random.default_rng(0).random((3, 5, groups)).astype(np.float32)
    if groups > 1:
        probe /= probe.sum(-1, keepdims=True)
    rule(probe)
    rule(probe[0])
    return rule


@dataclass
class RegBar:
    """The decision rule and bar for regression models, in place of class decisions. `space`: "log" when the model
    outputs the log of the quantity (|exp(pruned - full) - 1| is the relative difference), "raw" when it outputs the quantity.
    Bars (percent): median relative difference of pruned vs full <= `median_pct`, none above `max_pct`, and, when the data
    carries the truth `y` (in the model's output space), the pruned held-out MAE no worse than the full one's x (1 + `error_slack`)."""

    space: str = "log"
    median_pct: float = 0.5
    max_pct: float = 2.0
    error_slack: float = 0.0

    def __post_init__(self):
        for name in ("median_pct", "max_pct", "error_slack"):
            if not math.isfinite(getattr(self, name)):  # a NaN bar compares False with everything: every model would pass it
                raise ValueError(f"{name} must be a finite number, got {getattr(self, name)}")

    def describe(self) -> dict:
        return {"space": self.space, "median_pct": self.median_pct, "max_pct": self.max_pct, "error_slack": self.error_slack}

    def rel_diff(self, p_full: np.ndarray, p_pruned: np.ndarray) -> np.ndarray:
        a, b = p_full.reshape(len(p_full), -1)[:, 0].astype(np.float64), p_pruned.reshape(len(p_pruned), -1)[:, 0].astype(np.float64)
        if self.space == "log":
            return np.abs(np.expm1(b - a))
        return np.abs(b - a) / np.maximum(np.abs(a), 1e-12)

    def check(self, p_full, p_pruned, y=None) -> dict:
        """The verdict is taken on the exact figures; the rounded ones in the result are for the report only."""
        r = 100 * self.rel_diff(p_full, p_pruned)
        out = {"rows": len(r), "bars": self.describe()}
        exact = dict(zip(("median", "p90", "p99", "max"), np.percentile(r, [50, 90, 99, 100]), strict=True))
        out.update({f"{n}_pct": round(float(v), 4) for n, v in exact.items()})
        out["over_max"] = int((~(r <= self.max_pct)).sum())
        fails = []
        # every comparison is written so that a NaN (a model that answers NaN) fails the bar instead of passing it
        if not exact["median"] <= self.median_pct:
            fails.append(f"median |pruned - full| {float(exact['median']):.6g} % > bar {self.median_pct} %")
        if not exact["max"] <= self.max_pct:
            fails.append(f"{out['over_max']} rows differ by more than {self.max_pct} % (max {float(exact['max']):.6g} %)")
        if y is not None:
            e_full = float(np.abs(p_full.reshape(len(y), -1)[:, 0] - y).mean())
            e_pruned = float(np.abs(p_pruned.reshape(len(y), -1)[:, 0] - y).mean())
            out.update(mae_full=round(e_full, 6), mae_pruned=round(e_pruned, 6), error_checked=True)
            if not e_pruned <= e_full * (1 + self.error_slack):
                fails.append(f"held-out MAE pruned {e_pruned:.5f} > full {e_full:.5f} x {1 + self.error_slack:g}")
        else:
            out["error_checked"] = False
        out["failures"], out["ok"] = fails, not fails
        return out


# --------------------------------------------------------------------------------------------------------- data
PER_ROW_ARRAYS = ("y", "fold", "group", "slice", "eligible")
WHOLE_NUMBER_ARRAYS = ("fold", "group", "slice")


def npy_header_fits_file(p: Path) -> None:
    """A `.npy` whose header claims more values than the file holds is refused BEFORE numpy allocates what the header says (a 160-byte file can
    claim 10^12 values: MemoryError or a long allocation). Anything that is not a plain numeric `.npy` (an archive, a pickle, a bad header) is
    left to `np.load`, which has its own messages."""
    try:
        with open(p, "rb") as f:
            if f.read(6) != b"\x93NUMPY":
                return
            f.seek(0)
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, _, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, _, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                return
            body = os.fstat(f.fileno()).st_size - f.tell()
    except (OSError, ValueError, EOFError):
        return
    if dtype.hasobject:
        return
    need = int(np.prod([int(n) for n in shape], dtype=object)) * dtype.itemsize
    if need > body:
        raise ValueError(f"{p}: the header claims {need:,} bytes of data but the file holds {max(body, 0):,}")


# What reading a file the user gave us can raise that is not a bug of ours: a damaged compressed member (zlib), a zip method or header numpy
# cannot read (NotImplementedError, tokenize.TokenError, struct.error), text that is not text, a document nested past the parser's limit,
# a file that claims more than the machine holds. Programming errors (TypeError, KeyError, AttributeError ...) are not in the list.
DAMAGED_FILE_ERRORS = (
    zlib.error,
    zipfile.BadZipFile,
    EOFError,
    NotImplementedError,
    tokenize.TokenError,
    struct.error,
    OverflowError,
    SyntaxError,
)


@contextlib.contextmanager
def refusing_damaged(path, what: str = "file", error=ValueError):
    """The one place where a failure of READING a user file becomes the one-line refusal (a `ValueError`, or `error`, a subclass of it, naming
    the path): used by the data loader, the model readers and the artifact loader, so a new reader gets the whole family by wrapping its
    read in it. A `ValueError` raised inside is already a refusal and passes (text that is not UTF-8 gets the path); `OSError` is left to
    the callers, which print the system's own words."""
    p = Path(path)
    try:
        yield
    except UnicodeError as e:
        raise error(f"{p}: not valid text for a {what} ({one_line(e)[:200]})") from None
    except ValueError:
        raise
    except MemoryError:
        raise error(f"{p}: out of memory reading this {what} (a damaged file can claim any size)") from None
    except RecursionError:
        raise error(f"{p}: not a valid {what} (nested too deep)") from None
    except DAMAGED_FILE_ERRORS as e:
        raise error(f"{p}: not a valid {what} (damaged or unsupported: {type(e).__name__}: {one_line(e)[:200] or 'no detail'})") from None


def _load_data(path) -> dict[str, np.ndarray]:
    """`.npy` (X only) or `.npz` with X and optional fold, group, slice (0 selection, 1 held-out, -1 unused), eligible, and y
    (the truth, in the model's output space, for the regression error bar). Each of those is ONE value per row: an (n, 1) column is
    flattened to (n,) (a column would broadcast against (n,) and turn every row-wise comparison into n x n pairs), any other shape is
    refused. `base_margin` is (n,) or (n, groups). A path that is not a regular file (a FIFO would block for ever) is refused."""
    p = Path(path)
    if p.exists() and not p.is_file():
        raise ValueError(f"{p}: not a regular file (a directory, a FIFO or a device is not read)")
    try:
        npy_header_fits_file(p)
        loaded = np.load(p)  # what comes back says what the file is, whatever its name: an array (.npy) or an archive (.npz)
        if isinstance(loaded, np.ndarray):
            data = {"X": loaded}
        else:
            with loaded as d:
                data = {k: d[k] for k in d.files}
    except (OSError, EOFError, zipfile.BadZipFile) as e:  # a missing, unreadable or cut file: the library's own words, as one error
        raise ValueError(f"{p}: {one_line(e) or type(e).__name__}") from None
    except MemoryError:  # a header (or an archive member's) that claims a shape no machine holds
        raise ValueError(f"{p}: the array it claims does not fit in memory (a damaged file can claim any shape)") from None
    if "X" not in data:
        raise ValueError(f"{p}: the data file needs an array named X")
    # booleans, integers and floats of any width are read as float32; text and objects are not numbers
    if data["X"].dtype.kind not in "biuf":
        raise ValueError(f"{p}: X must hold numbers, got dtype {data['X'].dtype}")
    with np.errstate(over="ignore"):  # a number beyond the float32 range becomes infinity, which is refused below with its count
        data["X"] = np.ascontiguousarray(data["X"], dtype=np.float32)
    if data["X"].ndim != 2:
        raise ValueError(f"{p}: X must be a 2-D array (rows, features), got {data['X'].ndim} dimensions")
    if len(data["X"]) == 0:
        raise ValueError(f"{p}: X has no rows")
    if np.isinf(data["X"]).any():
        raise ValueError(
            f"{p}: X holds {int(np.isinf(data['X']).sum())} infinite values (or numbers beyond the float32 range): a value is finite or NaN"
        )
    n = len(data["X"])
    for name in PER_ROW_ARRAYS:
        if name in data:
            a = data[name]
            if a.ndim == 2 and a.shape[1] == 1:
                a = a.reshape(-1)  # the one unambiguous reading of a column
            if a.shape != (n,):
                raise ValueError(f"{p}: {name} must hold one value per row of X, shape ({n},) or ({n}, 1); got shape {a.shape}")
            require_numbers(a, name, p)
            if name in WHOLE_NUMBER_ARRAYS:
                require_whole_numbers(a, name, p)
            data[name] = a
    if "slice" in data and not np.isin(data["slice"], (-1, 0, 1)).all():
        bad = np.unique(data["slice"][~np.isin(data["slice"], (-1, 0, 1))])[:5].tolist()
        raise ValueError(f"{p}: slice must hold 0 (selection), 1 (held-out) or -1 (unused); got {bad}")
    if "base_margin" in data:
        if not (data["base_margin"].ndim in (1, 2) and len(data["base_margin"]) == n):
            raise ValueError(f"{p}: base_margin must be ({n},) or ({n}, groups); got shape {data['base_margin'].shape}")
        require_numbers(data["base_margin"], "base_margin", p)
    return data


def load_data(path) -> dict[str, np.ndarray]:
    """The data file as a dict of arrays (`.npy` with X, or `.npz`; the layout is in `_load_data`). Any failure of reading it that is the file's
    (damaged, unsupported, too deeply nested, too big) is a `ValueError` naming the path."""
    with refusing_damaged(path, "data file"):
        return _load_data(path)


def require_whole_numbers(a: np.ndarray, name: str, path) -> None:
    """`fold`, `group` and `slice` are read as integers: 0.7 would be cut to 0, and a number beyond the int64 range has no integer."""
    if a.dtype.kind == "f" and a.size and not ((a == np.rint(a)).all() and np.abs(a).max() < 2.0**62):
        raise ValueError(
            f"{path}: {name} must hold whole numbers (it is read as integers); got {a[~(a == np.rint(a))][:3].tolist() or 'a number beyond 2**62'}"
        )


def require_numbers(a: np.ndarray, name: str, path) -> None:
    """A data array that is not text or objects and holds no NaN or infinity: a NaN in the truth `y` makes every error bar compare False
    (so it passes), and a NaN in `fold`, `group` or `slice` turns into an arbitrary integer."""
    if a.dtype.kind not in "biuf":
        raise ValueError(f"{path}: {name} must hold numbers, got dtype {a.dtype}")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise ValueError(f"{path}: {name} holds {int((~np.isfinite(a)).sum())} NaN or infinite values; every value must be a finite number")


def check_width(x: np.ndarray, n_features: int, what="the data") -> None:
    """The rows must have exactly the model's number of features: the libraries differ on what they do with another width (xgboost reads
    the missing columns as missing, YDF ignores extra ones, LightGBM and numpy raise their own errors)."""
    if x.ndim != 2 or x.shape[1] != n_features:
        raise ValueError(f"{what}: X has {x.shape[-1] if x.ndim else 0} columns, the model has {n_features} features")


def data_width(path) -> int | None:
    """The number of columns of X in a `.npy` or `.npz` data file, read from the array's header without loading the rows; None when it cannot
    be told this way (anything else is left to `load_data`, which refuses it with its own message)."""
    p = Path(path)
    if not p.is_file():  # a pipe would block the peek: only a regular file is opened here
        return None
    try:
        if p.suffix == ".npz":
            with zipfile.ZipFile(p) as z, z.open("X.npy") as f:
                version = np.lib.format.read_magic(f)
                shape = (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)[0]
        else:
            with open(p, "rb") as f:
                version = np.lib.format.read_magic(f)
                shape = (np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0)(f)[0]
    except Exception:  # a peek: whatever is wrong with the file, load_data names it later
        return None
    return int(shape[1]) if len(shape) == 2 else None


def load_data_cli(path) -> dict[str, np.ndarray]:
    """`load_data` for the command-line tools: a bad data file ends the run with one message instead of a traceback."""
    try:
        return load_data(path)
    except (ValueError, OSError, EOFError) as e:
        raise SystemExit(f"{path}: {one_line(e)}") from None


def slices(data: dict, eval_frac: float, seed: int) -> np.ndarray:
    """0 = selection, 1 = held-out, -1 = unused. By the `slice` array if given, else by the parity of `group` (a document
    or customer id: rows of one group never straddle the two slices), else a seeded random split."""
    n = len(data["X"])
    if "slice" in data:
        s = data["slice"].astype(np.int64)
    elif "group" in data:
        s = (data["group"].astype(np.int64) % 2).astype(np.int64)
    else:
        s = (np.random.default_rng(seed).random(n) < eval_frac).astype(np.int64)
    if "eligible" in data:
        s = np.where(data["eligible"].astype(bool), s, -1)
    return s


# ------------------------------------------------------------------------------------------- the neutral forest
ZERO_EPS = float(np.float32(1e-35))  # LightGBM's kZeroThreshold (a float constant): |x| <= ZERO_EPS counts as zero


@dataclass
class Tree:
    group: int
    left: np.ndarray  # int32, -1 at a leaf
    right: np.ndarray
    feature: np.ndarray
    value: np.ndarray  # float32: the split threshold, or the leaf value at a leaf
    default_left: np.ndarray  # bool
    # EXTENSIONS beyond xgboost's semantics (zero_missing, cat, linear, leaf64): None everywhere for xgboost. The flat file and the
    # generated code hold all of them; the wavefront engines hold none. A Forest that uses zero_missing, cat or linear lists it in
    # `Forest.requires`, and compile.py refuses a backend whose `supports` leaves it out. leaf64 only changes how leaves are stored.
    zero_missing: np.ndarray | None = None  # bool per node: NaN and |x| <= ZERO_EPS go the `default_left` way, else `x < value`
    # node -> sorted int64 categories that go LEFT; NaN, values below 0 after truncation toward zero, and absent ones go right
    cat: dict[int, np.ndarray] | None = None
    # LightGBM linear leaves (`linear_tree=True`): leaf node -> (constant, fallback, features int32, coefficients float64), all doubles.
    # The leaf's value is `constant + sum coef * x[feature]` in that order (x as float64), or `fallback` (the plain leaf value) when any of
    # its features is NaN. Only NaN counts: 0.0 or |x| <= 1e-35 are numbers there (LightGBM refuses zero_as_missing with linear trees).
    linear: dict[int, tuple] | None = None
    # The leaf values in double, aligned with `value` (meaningful at leaves only), for the sources whose leaves are not float32 numbers
    # (LightGBM's text model holds doubles; a YDF random forest divides by the number of trees). `value` stays the float32 copy that the
    # default compile uses; the opt-in f64 leaves (`compile --f64-leaves`) store and generate these instead. None when `value` is exact.
    leaf64: np.ndarray | None = None


@dataclass
class ModelIR:
    base: np.ndarray  # float32 (groups,)
    trees: list[Tree]
    # LightGBM `sigmoid:s` (gbdt): s is folded into the leaves, so a base margin, given in raw-score units, is multiplied by it
    margin_scale: float = 1.0


@dataclass
class Forest:
    """Everything a compile backend needs. Semantics (xgboost's): `x[feature] < value` goes left, NaN goes the way
    `default_left` says; margin = base + sum of the leaves of the model's trees of that group, float32, in list order; output =
    softmax / sigmoid / identity of the margin; the forest answers the mean over `models` of the outputs (`average` =
    "output") or the transform of the mean margin ("margin"). `exp` (output = exp(margin)) and the Tree extensions are additions
    of the LightGBM and YDF adapters: `requires` lists the ones a forest uses and a backend declares the ones it holds (`supports`)."""

    n_features: int
    groups: int
    kind: str
    average: str
    models: list[ModelIR]
    task: str = "auto"
    sum64: bool = False  # the library sums in double (LightGBM, YDF): float32 sums would drift over the 1e-6 bar on hundreds of trees
    # float32 sums that add a base margin LATE, in double (YDF with float32_sums): its predictor starts from its own initial prediction
    late_margin: bool = False
    # the factor a base margin is multiplied by: the models' common `sigmoid:s` (NaN: the models disagree, a margin is refused)
    margin_scale: float = 1.0

    @property
    def n_trees(self) -> int:
        return sum(len(m.trees) for m in self.models)

    @property
    def n_nodes(self) -> int:
        return sum(len(t.left) for m in self.models for t in m.trees)

    @property
    def requires(self) -> set[str]:
        """What this forest needs beyond xgboost's float32 semantics (empty for every xgboost model). A backend compiles it only if its
        `supports` holds all of it; the wavefront engines hold none."""
        need = set()
        if self.sum64:
            need.add("sum64")
        if self.kind == "exp":
            need.add("objective:exp")
        if self.kind == "identity" and self.groups > 1:
            need.add("objective:identity-with-several-groups")  # YDF random forest classification: the outputs are already probabilities
        for m in self.models:
            for t in m.trees:
                if t.zero_missing is not None and t.zero_missing.any():
                    need.add("zero_missing")
                if t.cat:
                    need.add("categorical")
                if t.linear:
                    need.add("linear_leaves")
        return need


COUNT_OPTIONS = {"block": (1, 4096), "chunk_nodes": (1, 10**9), "jobs": (1, 256), "opt_level": (0, 3), "codegen_units": (1, 256)}


class UsageError(ValueError, SystemExit):
    """An option that cannot be used. A `ValueError` for code that calls the library and a `SystemExit` with the message as its code, so
    that every command-line tool prints it as one line and stops with its own refusal code."""

    def __init__(self, message: str):
        ValueError.__init__(self, message)
        self.code = message


def output_file(path, flag: str) -> None:
    """A file a command writes at the END of a long run (`--json`, `--md`): checked now, so that the run is not lost to a missing or read-only
    folder or to a target that is not a regular file."""
    p = Path(path)
    if p.is_dir():
        raise ValueError(f"{flag} {p}: a folder, not a file")
    if not p.parent.is_dir():
        raise ValueError(f"{flag} {p}: no such folder for the file")
    if (
        p.exists() and not p.is_file()
    ):  # /dev/null, a pipe: the report is written as a new file renamed over the target, which would replace it
        raise ValueError(f"{flag} {p}: not a regular file")
    if not os.access(p.parent, os.W_OK | os.X_OK):  # the file is written as a new file in its folder, then renamed
        raise ValueError(f"{flag} {p}: the folder is not writable")


def require_cpu(cpu) -> None:
    """`--cpu N` pins each timed run with `taskset -c N`; a CPU this process may not use would fail every run with a message of its own."""
    if cpu is not None and cpu not in os.sched_getaffinity(0):
        raise ValueError(f"--cpu {cpu}: not one of the CPUs this process may use ({sorted(os.sched_getaffinity(0))})")


def refuse_to_overwrite(out, files) -> None:
    """The artifact a command writes must not be (a link to) one of the model files it was given: it would be replaced by the compiled forest."""
    target = os.path.realpath(out)  # realpath: a link loop at the output is not an error here (the output replaces the link)
    for f in files:
        if os.path.realpath(f) == target:
            raise ValueError(f"{out}: that is the model file itself; compiling would overwrite it. Give another file name")


def parse_options(items: list[str]) -> dict:
    """`--opt key=value` pairs: digits are ints, true/false (yes/no, on/off) are booleans (`f64_leaves=false` is False, not the truthy string
    "false"), anything else stays a string. A pair without `=` is a message, not a ValueError from the unpacking."""
    out: dict = {}
    for item in items:
        if "=" not in item:
            raise UsageError(f"--opt {item!r}: expected key=value")
        k, v = item.split("=", 1)
        if not k.strip():
            raise UsageError(f"--opt {item!r}: the option has no name")
        low = v.strip().lower()
        out[k] = int(v) if v.isdecimal() else True if low in ("true", "yes", "on") else False if low in ("false", "no", "off") else v
    for key, (lowest, highest) in COUNT_OPTIONS.items():  # the numbers the generated-code backends read: a whole number in range
        if key in out and not (type(out[key]) is int and lowest <= out[key] <= highest):
            raise UsageError(f"--opt {key}: give a whole number from {lowest} to {highest}, not {out[key]!r}")
    return out


def load_forest(files: list[Path], average: str = "output", float32_sums: bool = False) -> Forest:
    """`float32_sums`: YDF models only. YDF's own predictor adds the leaves in float32 (see YdfSource.model_ir for the order), so ITS answer
    carries that rounding (up to ~3e-5 absolute at 300 trees on a log-scale target); a forest that sums the same way (the float32 path
    xgboost uses) reproduces `model.predict` bit for bit, where the default double sums differ from it by that noise."""
    import qf_formats

    models, head, task, sum64, fmts, first_names = [], None, "auto", False, set(), None
    for f in files:
        src = qf_formats.open_source(f)
        fmts.add(src.fmt)
        sum64 |= not isinstance(src, qf_formats.XgbSource)
        key = (src.spec.n_features, src.spec.groups, src.spec.kind)
        if head not in (None, key):
            raise ValueError("the models disagree on features, classes or objective")
        names = getattr(src, "names", None)  # YDF reads matrix column i as its i-th input feature BY NAME: folds must agree on the order
        if names is not None and first_names not in (None, names):
            raise ValueError(f"{f.name}: the models disagree on the names or the order of the input features ({first_names} and {names})")
        first_names = names if first_names is None else first_names
        head, task = key, src.spec.task
        with qf_formats.malformed(f, src.fmt):
            models.append(src.model_ir(True) if float32_sums and src.fmt == "ydf" else src.model_ir())
    if float32_sums:
        if fmts != {"ydf"}:
            raise ValueError(
                "float32 sums reproduce YDF's own predictor and apply to YDF models only (xgboost already sums in float32, LightGBM in "
                "double)"
            )
        sum64 = False
    scales = {m.margin_scale for m in models}
    scale = scales.pop() if len(scales) == 1 else float("nan")
    return Forest(head[0], head[1], head[2], average, models, task, sum64, late_margin=float32_sums, margin_scale=scale)


def require_rule(decision: str, forest: Forest) -> None:
    """Builds the decision rule of a classification forest only to refuse one that cannot be used, before any long work is done."""
    if forest.task == "classification":
        make_rule(decision, forest.groups, forest.kind, None, None, forest.task)


def reference_predict(files: list[Path], x: np.ndarray, average: str = "output", base_margin: np.ndarray | None = None) -> np.ndarray:
    """The library's own answer for the forest (xgboost / LightGBM / YDF): (rows, groups), averaged the way the compiled forest averages.
    float32 for xgboost, float64 for the others (their own precision). `base_margin` (rows, groups) is the same margin the compiled
    forest gets: added to each model's raw score by the library's own predictor (LightGBM / YDF: `raw_score=True` plus the margin, times
    the `sigmoid:s` of a LightGBM binary model that folds s into its leaves, then the objective's transform; xgboost: its DMatrix
    base_margin, with the model's base margin added), then averaged."""
    import qf_formats

    srcs = [qf_formats.open_source(f) for f in files]
    kind = srcs[0].spec.kind
    if base_margin is not None:
        bm = np.asarray(base_margin, dtype=np.float64).reshape(len(x), -1)
        parts = [
            s.predict(x, margin=True, base_margin=bm)
            if s.fmt == "xgboost"
            else s.predict(x, margin=True) + bm * getattr(s, "margin_scale", 1.0)
            for s in srcs
        ]
        tf = transform if parts[0].dtype == np.float32 else transform_f64
        if average == "margin":
            return tf(np.mean(parts, axis=0, dtype=parts[0].dtype), kind)
        return np.mean([tf(p, kind) for p in parts], axis=0, dtype=parts[0].dtype)
    if average == "margin":
        parts = [s.predict(x, margin=True) for s in srcs]
        tf = transform if parts[0].dtype == np.float32 else transform_f64  # `transform` rounds to float32: a double library's answer is not
        return tf(np.mean(parts, axis=0, dtype=parts[0].dtype), kind)
    parts = [s.predict(x) for s in srcs]
    return np.mean(parts, axis=0, dtype=parts[0].dtype)


def leaf_positions(t: Tree, x: np.ndarray) -> np.ndarray:
    """The leaf node every row of x (float32) ends in, by the forest's own semantics (extensions included)."""
    n = len(x)
    ar = np.arange(n)
    pos = np.zeros(n, np.int64)
    while True:
        lf = t.left[pos] == -1
        if lf.all():
            return pos
        node = np.where(lf, 0, pos)
        v = x[ar, t.feature[node]]
        nan = np.isnan(v)
        go = np.where(nan, t.default_left[node], v < t.value[node])
        if t.zero_missing is not None:
            zero = t.zero_missing[node] & (np.abs(v.astype(np.float64)) <= ZERO_EPS)
            go = np.where(zero, t.default_left[node], go)
        for k, cats in (t.cat or {}).items():
            sel = node == k
            if sel.any():
                iv = np.trunc(np.clip(np.where(nan, -1.0, v.astype(np.float64)), -1, 2.0**31 - 1)).astype(np.int64)
                go = np.where(sel, (iv >= 0) & np.isin(iv, cats), go)
        pos = np.where(lf, pos, np.where(go, t.left[node], t.right[node]))


def tree_values(t: Tree, x: np.ndarray) -> np.ndarray:
    pos = leaf_positions(t, x)
    out = t.value[pos]
    if not t.linear:
        return out
    out = out.astype(np.float64)
    for node, (const, fallback, feats, coefs) in t.linear.items():
        sel = np.flatnonzero(pos == node)
        if len(sel):
            xs = x[sel][:, feats].astype(np.float64)
            v = np.full(len(sel), const)
            for j in range(len(feats)):
                v = v + coefs[j] * xs[:, j]
            out[sel] = np.where(np.isnan(xs).any(axis=1), fallback, v)
    return out


def ir_predict(forest: Forest, x: np.ndarray, average: str | None = None, base_margin: np.ndarray | None = None) -> np.ndarray:
    """A slow numpy evaluator of the forest's own semantics, extensions included (float64 sums). It is the reference for the
    adapters' tests and the semantics a backend must implement: (rows, groups) outputs. `base_margin` (rows, groups) is added to
    every model's margin before the transform."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    outs, margins = [], []
    for m in forest.models:
        marg = np.tile(m.base.astype(np.float64), (len(x), 1))
        if base_margin is not None:
            marg = marg + np.asarray(base_margin, dtype=np.float64).reshape(len(x), -1) * m.margin_scale
        for t in m.trees:
            marg[:, t.group] += tree_values(t, x).astype(np.float64)
        margins.append(marg)
        outs.append(transform_f64(marg, forest.kind))
    if (average or forest.average) == "margin":
        return transform_f64(np.mean(margins, axis=0), forest.kind)
    return np.mean(outs, axis=0)


def transform_f64(m: np.ndarray, kind: str) -> np.ndarray:
    """The objective's transform in the dtype of `m` (float64 for the libraries that sum in double); `kind` is softmax, sigmoid, exp (LightGBM
    poisson / gamma / tweedie) or identity."""
    if kind == "softmax":
        e = np.exp(m - m.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)
    if kind == "sigmoid":
        if m.dtype == np.float32:
            # xgboost's `common::Sigmoid`: the exponent is clamped at 88.7f (a margin below -88.7 answers 3.0e-39, not 0; `np.minimum`
            # keeps a NaN margin NaN, as `std::min` does) and 1e-16f is added. LightGBM and YDF have no clamp: the double arm below.
            x = np.minimum(-m, np.float32(88.7))
            return np.float32(1) / (np.exp(x) + np.float32(1) + np.float32(1e-16))
        return 1.0 / (1.0 + np.exp(-m))
    if kind == "exp":
        return np.exp(m)
    return m
