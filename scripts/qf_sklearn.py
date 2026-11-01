"""scikit-learn adapter: decision trees, random forests and extra-trees (regression and classification, one or several outputs) reach QuickForest
through a QF-owned file, `.qfsk`, never through a pickle (unpickling runs code from the file).

    export_sklearn(estimator, "model.qfsk", X=None) -> dict      the only way in, from Python (sklearn is imported by the caller)
    SklearnSource(path)                                          what every command reads: the `.qfsk` as a neutral forest
    subforest(estimator, kept) -> estimator                      the pruned model as a real sklearn object

The file holds plain numpy arrays (np.savez, loaded with allow_pickle=False): the nodes of every tree concatenated, `tree_off` marking where
each tree starts. sklearn sends a sample left when `x <= threshold` (the float32 cell against the float64 threshold, compared in double), so
QF's `x < v` uses v = up32(floor32(threshold)), the float32 just above the largest float32 not above the threshold, exactly as LightGBM's
reader does. A regression answer is the mean over trees of the leaf values; a classification answer is the mean over trees of the leaf class
fractions of each output, all outputs side by side (`segments` marks where each output's classes start). A binary output keeps both columns,
sklearn's own `predict_proba` layout."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import qf_common as qf
import qf_formats

FORMAT = "qfsk1"
MAX_GROUPS = 1024
MIN_SKLEARN = (1, 3)
PROBE_ROWS = 2048
MAX_PROBE_CELLS = 8_000_000  # rows x features of the probe set: 2,048 rows of 30,000 features peaked at 2.3 GB while being drawn
MIN_PROBE_ROWS = 256
F32_MAX = float(np.finfo(np.float32).max)
EXPORT_TOL = 1e-9
# bench-only: compile every output of a multi-output forest as scalar trees (K copies of each tree), to measure the vector trees against them
REPLICATE_OUTPUTS = os.environ.get("QF_REPLICATE_OUTPUTS") == "1"

SUPPORTED = (
    "DecisionTreeRegressor",
    "DecisionTreeClassifier",
    "RandomForestRegressor",
    "RandomForestClassifier",
    "ExtraTreesRegressor",
    "ExtraTreesClassifier",
)
KEYS = (
    "format",
    "estimator",
    "sklearn_version",
    "n_features",
    "n_outputs",
    "task",
    "n_classes",
    "tree_off",
    "left",
    "right",
    "feature",
    "threshold",
    "missing_left",
    "value",
)


def _version_ok(text: str) -> bool:
    try:
        major, minor = (int(p) for p in text.split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= MIN_SKLEARN


def _estimator_parts(est):
    """(task, trees, n_features, n_outputs, n_classes) of a fitted supported estimator, or a one-line refusal."""
    name = type(est).__name__
    if name not in SUPPORTED:
        raise ValueError(f"{name} is not supported (decision trees, random forests and extra-trees only)")
    import sklearn

    if not _version_ok(sklearn.__version__):
        raise ValueError("sklearn 1.3 or newer: this version has no missing-value routing to export")
    trees = getattr(est, "estimators_", None)
    if trees is None and hasattr(est, "tree_"):
        trees = [est]
    if not trees:
        raise ValueError(f"{name} is not fitted")
    classification = name.endswith("Classifier")
    n_out = int(est.n_outputs_)
    n_classes = np.array(est.n_classes_ if classification else [0] * n_out, np.int64).reshape(-1)
    return ("classification" if classification else "regression"), list(trees), int(est.n_features_in_), n_out, n_classes


def _stack(trees, n_out: int, c_max: int):
    """The concatenated node arrays of the fitted trees, and the offset of each tree."""
    if {t.tree_.value.shape[1:] for t in trees} != {(n_out, c_max)}:
        raise ValueError("the trees' value arrays do not match the class counts of the model")
    sizes = np.array([t.tree_.node_count for t in trees], np.int64)
    off = np.concatenate([[0], np.cumsum(sizes)])
    cat = lambda f: np.concatenate([f(t.tree_) for t in trees])  # noqa: E731
    arrays = {
        "tree_off": off,
        "left": cat(lambda t: t.children_left).astype(np.int32),
        "right": cat(lambda t: t.children_right).astype(np.int32),
        "feature": cat(lambda t: t.feature).astype(np.int32),
        "threshold": cat(lambda t: t.threshold).astype(np.float64),
        "missing_left": cat(lambda t: t.missing_go_to_left).astype(bool),
    }
    value = cat(lambda t: t.value).astype(np.float64)
    arrays["value"] = value.reshape(int(off[-1]), n_out, c_max)
    return arrays


def _normalise(arrays: dict, n_classes: np.ndarray) -> bool:
    """A classifier's leaf rows must be fractions that sum to 1 (sklearn 1.4 and later store them so; older versions stored weighted counts):
    divide the rows that do not, in place. Returns whether any did."""
    value, leaf, changed = arrays["value"], arrays["left"] == -1, False
    for k, c in enumerate(n_classes):
        sums = value[leaf, k, : int(c)].sum(axis=1)
        if (sums <= 0).any() or not np.isfinite(sums).all():
            raise ValueError("a leaf holds no samples or a value that is not a finite number")
        off = np.abs(sums - 1.0) > 1e-6
        if off.any():
            rows = np.flatnonzero(leaf)[off]
            value[rows, k, : int(c)] /= sums[off][:, None]
            changed = True
    return changed


def probe_rows(arrays: dict, n_features: int, n: int = PROBE_ROWS, seed: int = 0, nan_rate: float = 0.06) -> np.ndarray:
    """Rows that exercise the routing: values on a threshold, one float32 step either side of it, NaN (`nan_rate` of the cells), and plain
    numbers. A model of more than MAX_PROBE_CELLS / PROBE_ROWS features gets fewer rows (at least MIN_PROBE_ROWS), so that drawing them stays
    within a few hundred MB."""
    n = min(n, max(MIN_PROBE_ROWS, MAX_PROBE_CELLS // max(1, n_features)))
    rng = np.random.default_rng(seed)
    split = np.flatnonzero((arrays["left"] != -1) & np.isfinite(arrays["threshold"]))
    x = rng.normal(size=(n, n_features)).astype(np.float32)
    if len(split):
        pick = rng.choice(split, size=(n, n_features))
        with np.errstate(over="ignore"):
            base = np.clip(arrays["threshold"][pick], -F32_MAX, F32_MAX).astype(np.float32)
        step = rng.integers(0, 5, size=(n, n_features))
        v = np.where(
            step == 0,
            base,
            np.where(step == 1, np.nextafter(base, np.float32(np.inf)), np.where(step == 2, np.nextafter(base, np.float32(-np.inf)), x)),
        ).astype(np.float32)
        mask = rng.random((n, n_features)) < 0.35  # most cells stay random so that deep nodes are reached too
        x = np.where(mask, v, x).astype(np.float32)
    x[rng.random(x.shape) < nan_rate] = np.nan
    return x


def reference_walk(arrays: dict, x: np.ndarray, classification: bool, n_classes: np.ndarray) -> np.ndarray:
    """sklearn's arithmetic on the exported arrays, float64: per tree, descend with `x <= threshold` (NaN follows `missing_left`), add the
    leaf values, divide by the number of trees. Regression: (rows, K). Classification: (rows, G), each output's fractions side by side."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    off, left, right = arrays["tree_off"], arrays["left"], arrays["right"]
    feature, thr, miss, value = arrays["feature"], arrays["threshold"], arrays["missing_left"], arrays["value"]
    rows = np.arange(len(x))
    total = np.zeros((len(x), value.shape[1], value.shape[2]))
    for a in off[:-1]:
        pos = np.full(len(x), a, np.int64)
        while True:
            node_left = left[pos]
            inner = node_left != -1
            if not inner.any():
                break
            v = x[rows, np.where(inner, feature[pos], 0)]
            go_left = np.where(np.isnan(v), miss[pos], v.astype(np.float64) <= thr[pos])
            nxt = np.where(go_left, node_left, right[pos]) + a
            pos = np.where(inner, nxt, pos)
        total += value[pos]
    total /= len(off) - 1
    if not classification:
        return total[:, :, 0]
    return np.concatenate([total[:, k, : int(c)] for k, c in enumerate(n_classes)], axis=1)


def estimator_answer(est, x: np.ndarray, classification: bool) -> np.ndarray:
    """The estimator's own `predict` / `predict_proba` as the same (rows, K or G) float64 matrix `reference_walk` gives."""
    if not classification:
        return np.asarray(est.predict(x), np.float64).reshape(len(x), -1)
    proba = est.predict_proba(x)
    parts = proba if isinstance(proba, list) else [proba]
    return np.concatenate([np.asarray(p, np.float64) for p in parts], axis=1)


def export_sklearn(estimator, path, X=None) -> dict:
    """Writes `path` (a `.qfsk`) from a fitted sklearn tree, random forest or extra-trees model, after checking that QF's own walk of the
    exported arrays reproduces the estimator on `X` (given) or on 2,048 rows built from the trees' thresholds plus NaN. Nothing is written
    when the two differ by more than 1e-9 relative. Returns {"path", "verified_rows", "max_diff", "normalised", ...}."""
    path = Path(path)
    if path.suffix != ".qfsk":
        raise ValueError(f"{path.name}: the export file must end in .qfsk")
    task, trees, n_features, n_out, n_classes = _estimator_parts(estimator)
    classification = task == "classification"
    if classification and (n_classes < 2).any():
        raise ValueError(
            "a classification output with a single class has nothing to classify: train with at least two classes in each output"
        )
    if int(n_classes.sum()) > MAX_GROUPS or n_out > MAX_GROUPS:
        raise ValueError(
            f"the model has {int(n_classes.sum()) if classification else n_out} output values; the most QF holds is {MAX_GROUPS}"
        )
    c_max = int(max(1, n_classes.max())) if classification else 1
    arrays = _stack(trees, n_out, c_max)
    normalised = _normalise(arrays, n_classes) if classification else False
    arrays.update(
        format=np.array(FORMAT),
        estimator=np.array(type(estimator).__name__),
        sklearn_version=np.array(__import__("sklearn").__version__),
        n_features=np.array(n_features, np.int64),
        n_outputs=np.array(n_out, np.int64),
        task=np.array(task),
        n_classes=n_classes,
    )
    _validate(arrays, path.name)
    rows = probe_rows(arrays, n_features)
    if X is not None:
        given = np.ascontiguousarray(X, dtype=np.float32)
        if given.ndim != 2 or given.shape[1] != n_features:
            raise ValueError(f"X must have {n_features} columns")
        rows = np.concatenate([given, rows])
    nan_verified = True
    try:
        theirs = estimator_answer(estimator, rows, classification)
    except ValueError as e:
        if "NaN" not in str(e):
            raise
        # an estimator that refuses missing values (e.g. criterion="absolute_error") cannot be asked about NaN: verify on NaN-free rows. The
        # probes are drawn again without NaN (dropping the rows that hold one would leave none at 100 features), and the caller's rows lose
        # only those that hold a NaN. QF's answer for a NaN row is then the stored routing, which no library answer has checked.
        nan_verified = False
        rows = probe_rows(arrays, n_features, nan_rate=0.0)
        if X is not None:
            rows = np.concatenate([given[~np.isnan(given).any(axis=1)], rows])
        theirs = estimator_answer(estimator, rows, classification)
    ours = reference_walk(arrays, rows, classification, n_classes)
    diff = float((np.abs(ours - theirs) / np.maximum(1.0, np.abs(theirs))).max())
    if not diff <= EXPORT_TOL:
        raise ValueError(f"export does not reproduce the estimator: largest difference {diff:.3g} on {len(rows)} rows")
    qf.replace_via_temp(path, lambda tmp: _save(tmp, arrays))
    return {
        "path": str(path),
        "verified_rows": len(rows),
        "max_diff": diff,
        "normalised": normalised,
        "n_trees": len(trees),
        "nan_verified": nan_verified,
    }


def _save(tmp, arrays: dict) -> None:
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)


def _validate(a: dict, name: str) -> None:
    """Refuses, in one line naming the file, an export that is not one of ours or is damaged: every later reader indexes by these arrays."""

    def bad(why):
        raise ValueError(f"{name}: not a valid {FORMAT} file ({why})")

    missing = [k for k in KEYS if k not in a]
    if missing:
        bad(f"missing {missing[0]}")
    if str(a["format"]) != FORMAT:
        bad(f"format {str(a['format'])[:40]!r}, expected {FORMAT}")
    for k in ("format", "estimator", "sklearn_version", "task", "n_features", "n_outputs"):
        if a[k].ndim != 0:
            bad(f"{k} is not a single value")
    for k in ("n_features", "n_outputs"):
        if a[k].dtype.kind not in "iu":
            bad(f"{k} has the wrong type")
    task, n_features, n_out = str(a["task"]), int(a["n_features"]), int(a["n_outputs"])
    if task not in ("regression", "classification"):
        bad(f"task {task[:40]!r}")
    if n_features < 1 or n_out < 1 or n_out > MAX_GROUPS:
        bad("feature or output count out of range")
    off, n_classes, value = a["tree_off"], a["n_classes"], a["value"]
    kinds = {
        "tree_off": "i",  # signed: -1 marks a leaf, and an unsigned array wraps `np.diff` and the child tests
        "left": "i",
        "right": "i",
        "feature": "i",
        "threshold": "f",
        "missing_left": "b",
        "value": "f",
        "n_classes": "iu",
    }
    for k, kind in kinds.items():
        if a[k].dtype.kind not in kind:
            bad(f"{k} has the wrong type")
    # compared pairwise, never subtracted: `np.diff` wraps for values near the int64 limits, and a wrapped offset list passed the test and
    # sent `np.repeat` below into a segmentation fault
    if off.ndim != 1 or len(off) < 2 or off[0] != 0 or (off[1:] <= off[:-1]).any():
        bad("tree_off is not increasing from 0")
    total = int(off[-1])
    for k in ("left", "right", "feature", "threshold", "missing_left"):
        if a[k].shape != (total,):
            bad(f"{k} does not have one entry per node")
    if n_classes.shape != (n_out,) or value.ndim != 3 or value.shape[:2] != (total, n_out):
        bad("value or n_classes does not match the node and output counts")
    c_max = value.shape[2]
    if task == "regression" and (c_max != 1 or n_classes.any()):
        bad("a regression has one value per output")
    if task == "classification":
        if (n_classes < 2).any():
            bad("an output with fewer than two classes")
        if (n_classes > c_max).any() or c_max != n_classes.max():
            bad("n_classes beyond the width of value")
        if int(n_classes.sum()) > MAX_GROUPS:
            bad(f"more than {MAX_GROUPS} output values")
    if not np.isfinite(value).all():
        bad("a leaf value is not a finite number")
    left, right = a["left"].astype(np.int64), a["right"].astype(np.int64)
    leaf = left == -1
    if ((right == -1) != leaf).any():
        bad("a node has one child only")
    if task == "classification":
        _check_fractions(value, leaf, n_classes, bad)
    tree_of = np.repeat(np.arange(len(off) - 1), np.diff(off))
    inner = ~leaf
    lk, rk = left[inner], right[inner]  # indexes inside the node's own tree, as sklearn stores them
    size = np.diff(off)[tree_of][inner]
    here = np.arange(total)[inner] - off[tree_of][inner]
    if (lk <= here).any() or (rk <= here).any() or (lk >= size).any() or (rk >= size).any() or (lk == rk).any():
        bad("a child is outside its tree or not after its parent")
    children = np.concatenate([lk, rk]) + np.tile(off[tree_of][inner], 2)
    if len(children) and np.bincount(children, minlength=total).max() > 1:
        bad("a node has two parents")
    if len(children) != int(total - len(off) + 1):
        bad("a node is not reachable from its root")
    feat = a["feature"][inner]
    if len(feat) and (feat.min() < 0 or feat.max() >= n_features):
        bad("a split reads a feature the model does not have")
    thr = a["threshold"][inner]
    if np.isnan(thr).any() or (thr == -np.inf).any():  # +inf is sklearn's threshold for a branch that holds only the NaN rows
        bad("a threshold is not a number")


def _check_fractions(value, leaf, n_classes, bad) -> None:
    for k, c in enumerate(n_classes):
        rows = value[leaf, k, : int(c)]
        if (rows < 0).any() or (np.abs(rows.sum(axis=1) - 1.0) > 1e-6).any():
            bad("a leaf's class fractions do not sum to 1")
        if value[leaf, k, int(c) :].any():
            bad("a leaf holds a value beyond its output's classes")


def load_arrays(path) -> dict:
    """The validated arrays of a `.qfsk`, read without unpickling anything."""
    path = Path(path)
    try:
        with open(path, "rb") as f, np.load(f, allow_pickle=False) as z:  # our own handle: numpy leaks its own on a truncated archive
            arrays = {k: z[k] for k in z.files}
    except (ValueError, OSError, EOFError, KeyError, *qf.DAMAGED_FILE_ERRORS) as e:
        raise ValueError(f"{path.name}: not a valid {FORMAT} file ({qf.one_line(e)[:160]})") from None
    _validate(arrays, path.name)
    return arrays


def up_threshold(t: np.ndarray) -> np.ndarray:
    """The QF split value for sklearn thresholds: x <= t  <=>  x < up32(floor32(t)) for a float32 x."""
    with np.errstate(over="ignore"):
        f = t.astype(np.float32)  # nearest; floor32 steps down when that rounded up
        f = np.where(f.astype(np.float64) > t, np.nextafter(f, np.float32(-np.inf)), f)
        return np.nextafter(f, np.float32(np.inf)).astype(np.float32)


class SklearnSource:
    fmt = "sklearn"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.name = self.path.name
        self.arrays = a = load_arrays(self.path)
        self.estimator = str(a["estimator"])
        self.classification = str(a["task"]) == "classification"
        self.n_outputs = int(a["n_outputs"])
        self.n_classes = a["n_classes"].astype(np.int64)
        self.n_trees = len(a["tree_off"]) - 1
        self.names = None
        n_features = int(a["n_features"])
        groups = int(self.n_classes.sum()) if self.classification else self.n_outputs
        self.segments = [0, *np.cumsum(self.n_classes).tolist()] if self.classification else None
        self.vector = groups >= 2 and not REPLICATE_OUTPUTS
        if self.classification and groups < 2:
            raise ValueError(f"{self.name}: not a valid {FORMAT} file (a classification needs at least two classes)")
        self._thr = up_threshold(a["threshold"])
        self.irs = self._trees(groups)
        info = np.zeros(len(self.irs), np.int64) if self.vector or groups == 1 else np.arange(len(self.irs)) % groups
        self.unit_of = None if self.vector else np.arange(len(self.irs)) // groups
        self.spec = qf_formats.spec_for(
            n_features,
            groups,
            "identity",
            np.zeros(groups, np.float32),
            info,
            [len(t.left) for t in self.irs],
            True,
            "classification" if self.classification else "regression",
            groups if self.vector else 0,
        )
        self.single_tree = self.n_trees == 1  # a decision tree: pruning below 100 % is refused by the caller

    def _cut(self, i: int):
        a, off = self.arrays, self.arrays["tree_off"]
        s = slice(int(off[i]), int(off[i + 1]))
        return s, a["left"][s], a["right"][s]

    def _leaf_table(self, s: slice) -> np.ndarray:
        """(nodes, G) float64: a leaf's row is the mean-ready share of the tree: value / n_trees, the outputs side by side."""
        v = self.arrays["value"][s]
        if self.classification:
            v = np.concatenate([v[:, k, : int(c)] for k, c in enumerate(self.n_classes)], axis=1)
        else:
            v = v[:, :, 0]
        return v / self.n_trees

    def _trees(self, groups: int) -> list[qf.Tree]:
        a, out = self.arrays, []
        for i in range(self.n_trees):
            s, left, right = self._cut(i)
            leaf = left == -1
            feature = np.where(leaf, 0, a["feature"][s]).astype(np.int32)
            thr = np.where(leaf, np.float32(0), self._thr[s])
            miss = np.where(leaf, False, a["missing_left"][s])
            table = np.where(leaf[:, None], self._leaf_table(s), 0.0)
            if self.vector:
                out.append(qf.Tree(qf.VECTOR_GROUP, left, right, feature, thr, miss, vector=table.astype(np.float32), vector64=table))
                continue
            for g in range(groups):  # one output (K = 1 regression) or the replicated forest of the bench switch
                col = table[:, g]
                out.append(qf.Tree(g, left, right, feature, np.where(leaf, col, thr).astype(np.float32), miss, leaf64=col.copy()))
        return out

    # ---- the Source interface
    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    def model_ir(self) -> qf.ModelIR:
        return qf.ModelIR(self.spec.base, list(self.irs))

    def forest_segments(self) -> list[int] | None:
        """`Forest.segments` for this model: only a vector classifier with several outputs carries them (the bench's replicated forest has none)."""
        return self.segments if self.vector and self.classification and self.n_outputs > 1 else None

    def predict(self, x, margin=False, base_margin=None):
        x = np.ascontiguousarray(x, dtype=np.float32)
        qf.check_width(x, self.spec.n_features, self.name)
        return reference_walk(self.arrays, x, self.classification, self.n_classes)

    def set_threads(self, n: int) -> None:
        pass

    def leaf_codes(self, x) -> np.ndarray:
        """(trees, rows) int32: the ordinal of the leaf each row ends in, among the leaves of its tree in node order (the row of `leaf_vectors()[i]`)."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        qf.check_width(x, self.spec.n_features, self.name)
        out = np.empty((self.n_trees, len(x)), np.int32)
        for i in range(self.n_trees):
            t = self.irs[i if self.vector else i * self.spec.groups]
            ordinal = np.cumsum(t.left == -1) - 1
            out[i] = ordinal[qf.leaf_positions(t, x)]
        return out

    def leaf_vectors(self) -> list[np.ndarray]:
        """Per tree, the (leaves, G) float32 table the codes of `leaf_codes` index (a leaf's share of the mean, outputs side by side)."""
        if not self.vector:
            raise ValueError(f"{self.name}: a model with one output group has scalar trees; use leaf_matrix")
        return [t.vector[t.left == -1] for t in self.irs]

    def leaf_matrix(self, x) -> np.ndarray:
        """(trees, rows) leaf values of a one-group model (scalar trees)."""
        if self.vector:
            raise ValueError(f"{self.name}: a model with several output groups has vector trees; use leaf_codes and leaf_vectors")
        x = np.ascontiguousarray(x, dtype=np.float32)
        qf.check_width(x, self.spec.n_features, self.name)
        return np.stack([t.value[qf.leaf_positions(t, x)] for t in self.irs])

    def save_pruned(self, kept, mean_scale: bool, dest: Path) -> None:
        """A `.qfsk` holding only the kept trees; the mean over what is left needs no rescaling (it divides by the kept count itself)."""
        units = (
            sorted({int(i) for g in kept for i in g}) if self.unit_of is None else sorted({int(self.unit_of[i]) for g in kept for i in g})
        )
        a, off = self.arrays, self.arrays["tree_off"]
        pick = np.concatenate([np.arange(off[u], off[u + 1]) for u in units])
        sizes = np.array([off[u + 1] - off[u] for u in units], np.int64)
        new_off = np.concatenate([[0], np.cumsum(sizes)])
        pruned = {k: a[k] for k in a}
        for k in ("left", "right", "feature", "threshold", "missing_left", "value"):
            pruned[k] = a[k][pick]
        pruned["tree_off"] = new_off
        _validate(pruned, Path(dest).name)
        qf.replace_via_temp(Path(dest), lambda tmp: _save(tmp, pruned))


def subforest(estimator, kept) -> object:
    """The pruned model as a sklearn object: a deep copy that keeps only `estimators_[i]` for i in `kept` (sklearn's own predict averages
    `estimators_`), so its answer equals the pruned `.qfsk`'s."""
    name = type(estimator).__name__
    if name not in SUPPORTED:  # Bagging, boosting and the like also carry `estimators_`, with other bookkeeping this copy would leave stale
        raise ValueError(f"{name} is not supported (decision trees, random forests and extra-trees only)")
    trees = getattr(estimator, "estimators_", None)
    if trees is None:
        raise ValueError("a single decision tree has no trees to select")
    kept = [int(i) for i in kept]
    if not kept or len(set(kept)) != len(kept) or min(kept) < 0 or max(kept) >= len(trees):
        raise ValueError(f"kept must name distinct trees among the {len(trees)} of the model")
    out = copy.deepcopy(estimator)
    out.estimators_ = [out.estimators_[i] for i in kept]
    out.n_estimators = len(kept)
    return out
