"""Model formats behind one interface. prune, check and compile talk to a `Source` (one model file or folder); each format has an
adapter that reads it into the neutral forest (`qf_common.ModelIR`) and can write a pruned copy in its own format:

    xgboost   .ubj / .json            XgbSource (here)
    LightGBM  text model (`tree` ...) qf_lgb.LgbSource
    YDF       model folder            qf_ydf.YdfSource  (imports `ydf` lazily, at build time only)

A Source has: `fmt`, `path`, `spec` (qf_common.Spec), `leaf_matrix(x)` -> (trees, rows) float32 holding each tree's share of the
full sum, `predict(x, margin=False)` -> (rows, groups) the library's own answer, `model_ir()`, `save_pruned(kept, mean_scale, dest)`,
`size_bytes`, and optionally `unit_of` (IR tree -> pruning unit, when several IR trees must be kept together)."""

from __future__ import annotations

import contextlib
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import qf_common as qf

LGB_MAGIC = b"tree\nversion="


def is_ydf_dir(p: Path) -> bool:
    return p.is_dir() and ((p / "header.pb").is_file() or (p / "data_spec.pb").is_file())


def is_lgb_file(p: Path) -> bool:
    if not p.is_file():
        return False
    try:
        with open(p, "rb") as f:
            return f.read(len(LGB_MAGIC)) == LGB_MAGIC
    except OSError as e:  # unreadable (permission denied): a model path that cannot be used, not a traceback
        raise ModelFileError(f"{p}: {e.strerror}") from None


def natural(f: Path):
    return [int(t) if t.isdecimal() else t.casefold() for t in re.split(r"(\d+)", f.name)]  # fold0, Fold1, fold2 are in that order


class ModelFileError(ValueError, SystemExit):
    """A model path that is missing, empty, of an unknown or mixed format. It is a `ValueError` for code that calls the library (one clean
    error with a message, never a bare exit) and still a `SystemExit` with the message as its code, so the command-line tools print it
    and stop without a traceback."""

    def __init__(self, message: str):
        ValueError.__init__(self, message)
        self.code = message


def detect(path: Path) -> str:
    p = Path(path)
    if is_ydf_dir(p):
        return "ydf"
    if is_lgb_file(p):
        return "lightgbm"
    if p.suffix in (".ubj", ".json"):
        return "xgboost"
    raise ModelFileError(f"{p}: not an xgboost (.ubj/.json), LightGBM (text) or YDF (folder) model")


@contextlib.contextmanager
def malformed(path, fmt: str):
    """A model file that is not what its format says must end in ONE clean `ValueError` that names the file, whatever the parser tripped on
    (a missing field, an index out of range, a number that is not one): the parsers' internal errors never reach the caller."""
    try:
        yield
    except MemoryError:  # not a damaged file: the machine cannot hold this one (an xgboost forest prunes in a stream: LIMITS.md)
        raise ValueError(f"{Path(path).name}: out of memory reading this {fmt} model; see LIMITS.md for what each command needs") from None
    except RecursionError:  # a JSON document nested thousands of levels deep: the parser's own limit, not a model
        raise ValueError(f"{Path(path).name}: not a valid {fmt} model (nested too deep)") from None
    except (KeyError, IndexError, TypeError, AttributeError, OverflowError, ZeroDivisionError, UnicodeDecodeError) as e:
        what = f"missing field {e}" if isinstance(e, KeyError) else f"{type(e).__name__}: {e}"
        raise ValueError(f"{Path(path).name}: not a valid {fmt} model ({what[:200]})") from e


def open_source(path):
    try:
        with qf.refusing_damaged(path, "model", ModelFileError):
            return _open_source(path)
    except OSError as e:  # a model file that cannot be read (permission denied, I/O error)
        raise ModelFileError(f"{path}: {e.strerror or e}") from None


def _open_source(path):
    fmt = detect(Path(path))
    if fmt == "xgboost":
        with malformed(path, fmt):
            return XgbSource(Path(path))
    if fmt == "lightgbm":
        import qf_lgb

        with malformed(path, fmt):
            return qf_lgb.LgbSource(Path(path))
    import qf_ydf

    with malformed(path, fmt):  # the YDF reader's own refusals are ValueErrors; an index or a field it did not expect is not a traceback
        return qf_ydf.YdfSource(Path(path))


# rows of every timed call of the speed baseline, whatever size the caller's sample is (a library's rows/s moves with the batch)
BASELINE_ROWS = 20000


def library_rate(srcs, x, reps: int = 2, threads: int = 1) -> float:
    """Rows/s of the LIBRARY that wrote the model (xgboost, LightGBM or YDF), the baseline of every speed ratio (`compare`, `check
    --speed`): its own predict on ONE thread by default (`set_threads(1)`: nthread / num_threads; `threads=N` times it on N, for the
    all-cores comparison) over exactly BASELINE_ROWS rows (the sample tiled
    or cut), best of `reps`. A source with several folds is timed over all of them, like the compiled forest. The batch size is fixed
    because YDF's default is every core and its rate grows with the batch (per-call overhead): the same model would read 2.17x on 500
    rows and 0.38x on 1,000."""
    for src in srcs:
        src.set_threads(threads)
    xs = np.ascontiguousarray(x)
    xs = xs[:BASELINE_ROWS] if len(xs) >= BASELINE_ROWS else np.tile(xs, (-(-BASELINE_ROWS // len(xs)), 1))[:BASELINE_ROWS]
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        for src in srcs:
            getattr(src, "predict_timed", src.predict)(xs)  # xgboost: without the DMatrix construction
        best = min(best, time.perf_counter() - t)
    return len(xs) / best


XGB_JSON_HEAD = re.compile(rb'\s*\{\s*(?:"version"\s*:\s*\[[^\]]*\]\s*,\s*)?"learner"')


def is_xgb_json(f: Path) -> bool:
    """An xgboost JSON model, compact or re-saved with indentation (the first bytes only: a model can be hundreds of MB)."""
    if f.suffix != ".json" or not f.is_file():
        return False
    with open(f, "rb") as fh:
        return XGB_JSON_HEAD.match(fh.read(256)) is not None


def is_model_file(f: Path) -> bool:
    return (f.suffix == ".ubj" and f.is_file()) or is_xgb_json(f) or is_lgb_file(f) or is_ydf_dir(f)  # a pipe is never opened


def model_files(path) -> list[Path]:
    """A model, or the models of a folder (the folds of a k-fold ensemble) in natural order (fold0, fold1, ..., fold10)."""
    p = Path(path)
    if not p.exists():
        raise ModelFileError(f"{p}: no such model file or folder")
    if p.is_file() or is_ydf_dir(p):
        return [p]
    if not p.is_dir():  # a FIFO, a device or a socket: listing it would raise NotADirectoryError, reading it could block
        raise ModelFileError(f"{p}: not a model file or folder (a pipe or a device is not read)")
    files = [f for f in p.iterdir() if is_model_file(f)]
    if not files:
        raise ModelFileError(f"{p}: no xgboost, LightGBM or YDF model files")
    files = sorted(files, key=natural)
    formats = sorted({detect(f) for f in files})
    if len(formats) > 1:  # the folds of one k-fold ensemble are one format; a stray file would be averaged in as another fold
        raise ModelFileError(f"{p}: holds models of several formats ({', '.join(formats)}); a folder is the folds of ONE ensemble")
    seen: dict[str, str] = {}
    for f in files:
        key = f.stem.casefold()
        if key in seen:  # the same model saved twice (fold0.ubj and fold0.json) would count as two folds and shift the fold numbers
            raise ModelFileError(
                f"{p}: {seen[key]} and {f.name} look like one model saved twice (names that differ only by case sort ambiguously too); "
                "a folder holds each fold once"
            )
        seen[key] = f.name
    return files


def early_stopping_notes(j: dict, name: str) -> list[str]:
    """A model saved after early stopping keeps every round it trained plus a `best_iteration` attribute. `Booster.predict` (the bar's
    reference here) uses all the rounds, but the scikit-learn wrapper's predict stops at best_iteration + 1: the compiled forest would
    then differ from what such a caller gets from xgboost, silently."""
    try:
        best = int(j["learner"]["attributes"]["best_iteration"])
        rounds = len(j["learner"]["gradient_booster"]["model"]["iteration_indptr"]) - 1
    except (KeyError, TypeError, ValueError):
        return []
    if best + 1 >= rounds:
        return []
    return [
        f"{name}: saved after early stopping (best_iteration {best}, {rounds} rounds kept). Booster.predict and this tool use all {rounds} "
        f"rounds; the scikit-learn wrapper's predict uses only the first {best + 1}. To match that, save the model cut to the best "
        f"rounds first (booster[:{best + 1}])."
    ]


class XgbSource:
    fmt = "xgboost"

    def __init__(self, path: Path):
        self.path = path
        self.raw = qf.raw_json(path)
        j = json.loads(self.raw)
        self.spec, self._trees = qf.parse_model(j)
        self.notes = early_stopping_notes(j, path.name)
        for n in self.notes:
            warnings.warn(n, stacklevel=2)
        self._booster = None
        self.name = path.name
        self.unit_of = None

    @property
    def booster(self):
        if self._booster is None:
            self._booster = qf.load_booster(self.path)
        return self._booster

    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def leaf_matrix(self, x):
        qf.check_width(x, self.spec.n_features, self.name)
        return qf.leaf_matrix(self.booster, self.spec, x)

    def set_threads(self, n: int) -> None:
        self.booster.set_param({"nthread": n})

    def predict(self, x, margin=False, base_margin=None):
        qf.check_width(x, self.spec.n_features, self.name)
        return qf.predict(self.booster, x, margin, base_margin, self.spec.base)

    def predict_timed(self, x):
        qf.check_width(x, self.spec.n_features, self.name)
        return qf.predict_timed(self.booster, x)

    def model_ir(self) -> qf.ModelIR:
        return qf.ModelIR(self.spec.base, [xgb_tree(t, int(g)) for t, g in zip(self._trees, self.spec.tree_info, strict=True)])

    def save_pruned(self, kept, mean_scale: bool, dest: Path) -> None:
        pb = qf.xgb().Booster()
        pb.load_model(bytearray(prune_raw(self.raw, self.spec, kept, mean_scale)))
        qf.replace_via_temp(dest, lambda tmp: pb.save_model(qf.library_path(tmp)))


def xgb_tree(t: dict, group: int) -> qf.Tree:
    """One xgboost JSON tree as the neutral tree, without the nodes no path reaches (splits xgboost dropped after growing them keep their
    slots in its arrays): the children are renumbered, the order of the reachable nodes is kept."""
    left, right = np.asarray(t["left_children"], dtype=np.int32), np.asarray(t["right_children"], dtype=np.int32)
    feature, value = np.asarray(t["split_indices"], dtype=np.int32), np.asarray(t["split_conditions"], dtype=np.float32)
    default_left = np.asarray(t["default_left"], dtype=bool)
    live = qf.reachable_nodes(left, right)
    if not live.all():
        new = np.cumsum(live) - 1
        left, right = np.where(left[live] == -1, -1, new[left[live]]), np.where(right[live] == -1, -1, new[right[live]])
        feature, value, default_left = feature[live], value[live], default_left[live]
    return qf.Tree(group, left.astype(np.int32), right.astype(np.int32), feature, value, default_left)


# Attributes xgboost saves after early stopping that name rounds of the model they were written for. The scikit-learn wrapper predicts with
# `iteration_range=(0, best_iteration + 1)`, so on a pruned file (fewer rounds) they raise "Out of range for tree layers" or silently use a
# prefix of the kept rounds. `best_score` is the validation score of the original model at that round. Every other attribute is the user's.
ROUND_ATTRIBUTES = ("best_iteration", "best_score", "best_ntree_limit")


def prune_raw(raw: bytes, spec, kept, mean_scale) -> bytes:
    """The model JSON holding only the kept trees (leaves scaled by trees/kept for a random forest), as a normal xgboost model."""
    return prune_doc(json.loads(raw), spec, kept, mean_scale)


def prune_doc(j: dict, spec, kept, mean_scale, trees=None) -> bytes:
    """`prune_raw` on a parsed document. `trees` (original index -> tree dict) replaces the document's own list of trees when that holds
    none of them: the stream path (`select_stream`) reads only the kept trees from a model too big to parse whole."""
    for key in ROUND_ATTRIBUTES:
        j["learner"].get("attributes", {}).pop(key, None)
    model = j["learner"]["gradient_booster"]["model"]
    source = model["trees"] if trees is None else trees
    g_n, k = spec.groups, len(kept[0])
    order = sorted(np.concatenate(kept).tolist()) if spec.rf else [int(kept[g][r]) for r in range(k) for g in range(g_n)]
    out = []
    totals = np.bincount(spec.tree_info, minlength=g_n)
    for i, t in enumerate(order):
        tree = source[t]
        if mean_scale:
            scale = np.float32(totals[spec.tree_info[t]] / k)
            leaf = np.asarray(tree["left_children"]) == -1
            for key in ("split_conditions", "base_weights"):
                v = np.asarray(tree[key], dtype=np.float32)
                v[leaf] *= scale
                tree[key] = v.tolist()
        tree["id"] = i
        out.append(tree)
    model["trees"] = out
    model["tree_info"] = [int(spec.tree_info[t]) for t in order]
    model["gbtree_model_param"]["num_trees"] = str(len(out))
    model["gbtree_model_param"]["num_parallel_tree"] = str(k if spec.rf else 1)
    model["iteration_indptr"] = [0, len(out)] if spec.rf else list(range(0, len(out) + 1, g_n))
    # raw UTF-8, not \u escapes: xgboost's JSON reader keeps an escape as the six characters, so a feature called `prix` with an e-acute came back with a literal backslash-u escape in its name
    return json.dumps(j, ensure_ascii=False).encode()


def spec_for(n_features, groups, kind, base, tree_info, nodes, mean_type, task="auto") -> qf.Spec:
    """A Spec for the formats that need no xgboost-style leaf table (`off` and `vals` are unused by them)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    return qf.Spec(
        n_features=n_features,
        groups=groups,
        kind=kind,
        base=np.asarray(base, dtype=np.float32),
        tree_info=np.asarray(tree_info, dtype=np.int64),
        nodes=nodes,
        off=np.concatenate([[0], np.cumsum(nodes)[:-1]]) if len(nodes) else nodes,
        vals=np.zeros(0, np.float32),
        parallel=1,
        task=task,
        mean_type=mean_type,
    )
