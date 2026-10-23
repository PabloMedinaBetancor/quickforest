"""LightGBM adapter: reads a LightGBM text model (gbdt, dart, goss and rf; regression, binary, multiclass, cross_entropy, poisson, gamma,
tweedie, ranking objectives; k-fold folders) into the neutral forest, and writes pruned copies as normal LightGBM text models.

Exactness. LightGBM sends a row left when `x <= threshold` (double). The forest sends it left when `x < value` (float32). For float32
inputs the two are the same decision with value = the float32 above the largest float32 <= threshold, so thresholds are converted,
not rounded. Missing values (decision_type bits): missing_type None sends NaN as 0.0, so NaN goes the way `0 <= threshold` goes;
NaN sends NaN the default way; Zero sends NaN and |x| <= 1e-35 the default way (rewritten to a plain split when that is expressible,
otherwise kept as `Tree.zero_missing`). Categorical splits become `Tree.cat`. Leaf values are summed from float32 copies of the
double leaves; `average_output` (rf) is folded into the leaves as 1/iterations, and a binary `sigmoid:s` as a factor s on them.

Inputs are float32 and finite or NaN: a +-inf input is not covered (LightGBM's `inf <= inf` is true, the forest's `inf < inf` is not).

Linear trees (`linear_tree=true`): a leaf's value is `leaf_const + sum coef * x[feature]` in double, in the order of the model file, or the
plain `leaf_value` when ANY of that leaf's features is NaN (LightGBM's predictor; only NaN counts, 0.0 is a number; LightGBM itself refuses
zero_as_missing with linear trees). They are read into `Tree.linear`; the leaf matrix pruning needs is not defined for them (compile only).

Refused with a clear message: multiclassova, cross_entropy_lambda, regression `sqrt`, custom objectives."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import qf_common as qf
import qf_formats

IDENTITY = {
    "regression",
    "regression_l2",
    "regression_l1",
    "l2",
    "l1",
    "mean_squared_error",
    "mse",
    "huber",
    "fair",
    "quantile",
    "mape",
    "mean_absolute_percentage_error",
    "lambdarank",
    "rank_xendcg",
    "regression_l2_root",
    "rmse",
    "mae",
}
SIGMOID = {"binary", "cross_entropy", "xentropy"}
EXP = {"poisson", "gamma", "tweedie"}
F32_MAX = float(np.finfo(np.float32).max)
# a line after the trees of a LightGBM text model: the feature importances (`name=count`), then the parameters block (`[name: value]`)
TAIL_LINE = re.compile(
    r"end of trees|feature_importances:|[^\n]*=\d+|parameters:|\[[a-z0-9_]+: [^\n]*\]|end of parameters|pandas_categorical:.*"
)


def floor32(t: float) -> np.float32:
    """The largest float32 <= t (-inf below -float32 max, float32 max from there up)."""
    if t >= F32_MAX:
        return np.float32(F32_MAX)
    if t < -F32_MAX:
        return np.float32(-np.inf)
    f = np.float32(t)
    return np.nextafter(f, np.float32(-np.inf)) if float(f) > t else f


def up32(f: np.float32) -> np.float32:
    with np.errstate(over="ignore"):
        return np.nextafter(f, np.float32(np.inf))


def numeric_split(t: float, default_left: bool, missing: int) -> tuple[np.float32, bool, bool]:
    """(value, default_left, needs the zero_missing flag) of a numerical LightGBM split `x <= t`, per its missing_type
    (0 None, 1 Zero, 2 NaN). LightGBM's predictor reads every |x| <= 1e-35 (the bucket B) as exactly 0.0, so B goes the way 0.0 does:
    left if 0 <= t, except for missing_type Zero, where B and NaN go the default way. The left set must be a prefix {x < value}."""
    big = floor32(t)
    zf = floor32(qf.ZERO_EPS)  # largest float32 in B; B = [-zf, zf]
    zlo = -zf
    if missing == 1:
        bucket_left, nan_left = default_left, default_left
    else:
        bucket_left = t >= 0.0
        nan_left = bucket_left if missing == 0 else default_left  # None reads NaN as 0.0
    if not bucket_left:
        if big <= zf:
            return min(up32(big), zlo), nan_left, False
    else:
        pa = np.nextafter(zlo, np.float32(-np.inf))
        if big >= pa:
            return up32(max(big, zf)), nan_left, False
    return up32(big), nan_left, True  # only a missing_type Zero node can get here


def check_tree(lc: np.ndarray, rc: np.ndarray, ni: int, nl: int, name: str) -> None:
    """Node 0 must reach every node exactly once (a LightGBM tree: a cycle or a shared child would loop the compiler)."""
    seen = np.zeros(ni + nl, bool)
    seen[0] = True
    frontier = np.array([0])
    while frontier.size:
        inner = frontier[frontier < ni]
        kids = np.concatenate([lc[inner], rc[inner]])
        if kids.size == 0:
            break
        if seen[kids].any() or len(np.unique(kids)) != len(kids):
            raise ValueError(f"{name}: a tree has a cycle or a node with two parents")
        seen[kids] = True
        frontier = kids
    if not seen.all():
        raise ValueError(f"{name}: a tree has nodes the root cannot reach")


def kv(block: bytes) -> dict[str, str]:
    out = {}
    for ln in block.decode().split("\n"):
        k, eq, v = ln.partition("=")
        if eq:
            out[k] = v
    return out


def floats(s: str | None, dtype=np.float64) -> np.ndarray:
    return np.array(s.split(), dtype=dtype) if s else np.zeros(0, dtype)


class LgbSource:
    fmt = "lightgbm"

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        data = path.read_bytes()
        if b"\0" in data:  # LightGBM's own loader never returns on a NUL byte (it reads the text as a C string)
            raise ValueError(f"{path}: a LightGBM text model holds no NUL byte (the library hangs on one)")
        end = data.find(b"end of trees")
        if end < 0:
            raise ValueError(f"{path}: not a LightGBM model (no 'end of trees' line: truncated?)")
        starts = [m.start() for m in re.finditer(rb"(?m)^Tree=\d+$", data[:end])]
        if not starts:
            raise ValueError(f"{path}: no trees")
        self.head, self.tail = data[: starts[0]], data[end:]
        try:
            self._check_tail((self.head + self.tail).decode())
        except UnicodeDecodeError:
            raise ValueError(f"{path}: not valid UTF-8 text outside the trees") from None
        self.blocks = [data[a:b] for a, b in zip(starts, [*starts[1:], end], strict=True)]
        self.h = kv(self.head.split(b"tree_sizes=")[0] if b"tree_sizes=" in self.head else self.head)
        self.rf = b"\naverage_output\n" in self.head
        self.unit_of = None
        self._booster = None
        self._threads = 0
        self._build()

    # ---- parsing
    def _build(self):
        h = self.h
        obj = h.get("objective", "").split()
        name = obj[0] if obj else ""
        opts = dict(o.split(":", 1) for o in obj[1:] if ":" in o)
        self.groups = int(h["num_tree_per_iteration"])
        if not 1 <= self.groups <= len(self.blocks):
            raise ValueError(f"{self.name}: num_tree_per_iteration={self.groups} but the file has {len(self.blocks)} trees")
        max_feature = int(h["max_feature_idx"])
        if not 0 <= max_feature < 1 << 31:
            raise ValueError(f"{self.name}: max_feature_idx={max_feature} is out of range")
        self.scale = 1.0
        if name in IDENTITY and "sqrt" not in obj:
            kind, task = "identity", "regression"
        elif name in SIGMOID:
            kind, task = "sigmoid", "classification"
            try:
                self.scale = float(opts.get("sigmoid", 1.0))
            except ValueError:
                raise ValueError(f"{self.name}: sigmoid:{opts['sigmoid']!r} is not a number") from None
            if not (math.isfinite(self.scale) and self.scale > 0):  # LightGBM: "Sigmoid parameter should be greater than zero"
                raise ValueError(f"{self.name}: sigmoid:{opts['sigmoid']} must be a finite number greater than zero (LightGBM refuses it)")
        elif name == "multiclass" or name == "softmax":
            kind, task = "softmax", "classification"
        elif name in EXP:
            kind, task = "exp", "regression"
        else:
            raise ValueError(
                f"objective '{' '.join(obj)}' is not supported (LightGBM: regression family, binary, multiclass, cross_entropy, poisson, "
                f"gamma, tweedie)"
            )
        if kind == "softmax" and self.groups < 2:
            raise ValueError("multiclass model with one tree per iteration")
        n_trees = len(self.blocks)
        if n_trees % self.groups:
            raise ValueError("the number of trees is not a multiple of the trees per iteration")
        self.n_iter = n_trees // self.groups
        self.trees = [self._tree(kv(b)) for b in self.blocks]
        self._check_header(max_feature)
        self.linear = any(t["linear"] for t in self.trees)
        self.spec = qf_formats.spec_for(
            int(h["max_feature_idx"]) + 1,
            self.groups,
            kind,
            np.zeros(self.groups),
            np.arange(n_trees) % self.groups,
            [len(t["left"]) for t in self.trees],
            self.rf,
            task,
        )

    def _check_tail(self, text: str) -> None:
        """The part after the trees is read by LightGBM's own loader (it parses the parameters block), which crashes the process on a damaged
        one. Every line must have the shape LightGBM writes, and a parameters block must be closed."""
        lines = [ln for ln in text[text.index("end of trees") :].split("\n") if ln.strip()]
        for ln in lines:
            if not TAIL_LINE.fullmatch(ln):
                raise ValueError(f"{self.name}: a line after the trees is not part of a LightGBM text model: {ln[:40]!r}")
        if "parameters:" in lines and "end of parameters" not in lines:
            raise ValueError(f"{self.name}: the parameters block after the trees is not closed (truncated?)")

    def _check_header(self, max_feature: int) -> None:
        """The header lines LightGBM's own loader trusts: it cuts the trees out of the file by `tree_sizes` and reads one name and one
        range per feature. A file where they disagree with what this reader found would compile here and then abort the process (or fail
        inside the library) the first time `prune` or `check` loads it with LightGBM, so it is refused now."""
        for key in ("feature_names", "feature_infos"):
            if key in self.h and len(self.h[key].split()) != max_feature + 1:
                raise ValueError(
                    f"{self.name}: {key} lists {len(self.h[key].split())} features but max_feature_idx={max_feature} says {max_feature + 1}"
                )
        found = re.search(rb"(?m)^tree_sizes=(.*)$", self.head)
        if found:
            try:
                sizes = [int(t) for t in found.group(1).split()]
            except ValueError:
                raise ValueError(f"{self.name}: tree_sizes holds something that is not a number") from None
            if sizes != [len(b) for b in self.blocks]:
                raise ValueError(f"{self.name}: tree_sizes does not match the trees in the file (damaged or edited by hand)")

    def _tree(self, d: dict) -> dict:
        nl = int(d["num_leaves"])
        if not 1 <= nl <= len(d["leaf_value"]) // 2 + 1:  # every leaf has a value of at least one character plus a space
            raise ValueError(f"{self.name}: num_leaves={nl} does not fit the tree's leaf_value line")
        factor = self.scale / (self.n_iter if self.rf else 1)
        leaf64 = floats(d["leaf_value"]) * factor
        if len(leaf64) != nl:
            raise ValueError(f"{self.name}: a tree has num_leaves={nl} but {len(leaf64)} leaf values")
        leaf = leaf64.astype(np.float32)
        # leaf j -> its record, as (const, fallback, feats, coefs)
        lin = self._linear(d, nl, factor) if d.get("is_linear", "0") != "0" else None
        if nl == 1:
            return {
                "left": np.array([-1], np.int32),
                "right": np.array([-1], np.int32),
                "feature": np.zeros(1, np.int32),
                "value": leaf,
                "dl": np.zeros(1, bool),
                "zm": None,
                "cat": None,
                "linear": lin and {0: lin[0]},
                "leaf64": leaf64,
            }
        ni = nl - 1

        def child(a):
            return np.where(a >= 0, a, ni + (~a)).astype(np.int32)

        left, right = floats(d["left_child"], np.int64), floats(d["right_child"], np.int64)
        dt = floats(d["decision_type"], np.int64)
        thr = floats(d["threshold"])
        feat = floats(d["split_feature"], np.int64)
        # decision_type: bit 0 categorical, bit 1 default left, bits 2-3 the missing type (0 None, 1 Zero, 2 NaN); LightGBM writes nothing else, and
        # a missing type of 3 or a higher bit would be read as one of the others here and answered differently from the library
        if len(dt) and (dt.min() < 0 or dt.max() > 15 or (((dt >> 2) & 3) == 3).any()):
            bad = int(dt[(dt < 0) | (dt > 15) | (((dt >> 2) & 3) == 3)][0])
            raise ValueError(f"{self.name}: decision_type {bad} is not one LightGBM writes (missing type 3 or a bit beyond the fourth)")
        if not len(left) == len(right) == len(dt) == len(thr) == len(feat) == ni:
            raise ValueError(f"{self.name}: a tree has num_leaves={nl} but its split lines do not all hold {ni} entries")
        for c in (left, right):
            if (c >= ni).any() or (c < -nl).any():
                raise ValueError(f"{self.name}: a child index is outside the tree")
        if len(feat) and (feat.min() < 0 or feat.max() > int(self.h["max_feature_idx"])):
            raise ValueError(f"{self.name}: a split uses a feature index outside the model")
        if np.isnan(thr).any():
            raise ValueError(f"{self.name}: a NaN threshold")
        lc, rc = child(left), child(right)
        check_tree(lc, rc, ni, nl, self.name)
        value = np.zeros(ni, np.float32)
        dl = np.zeros(ni, bool)
        zm = np.zeros(ni, bool)
        cat = {}
        if int(d.get("num_cat", "0")):
            bounds = floats(d["cat_boundaries"], np.int64)
            words = floats(d["cat_threshold"], np.int64)
            if len(bounds) < 1 or (np.diff(bounds) < 0).any() or bounds[0] < 0 or bounds[-1] > len(words):
                raise ValueError(f"{self.name}: cat_boundaries do not describe cat_threshold")
        else:
            bounds = words = None
        for i in range(ni):
            if dt[i] & 1:  # categorical: threshold is the index of the bitset of categories that go left
                ci = int(thr[i])
                if bounds is None or not 0 <= ci < len(bounds) - 1:
                    raise ValueError(f"{self.name}: a categorical split points outside cat_boundaries")
                w = words[bounds[ci] : bounds[ci + 1]]
                cat[i] = np.array([32 * j + b for j, x in enumerate(w) for b in range(32) if (int(x) >> b) & 1], np.int64)
                if not len(cat[i]):  # LightGBM loads it (every row goes right) but never writes one: a hand-edited or damaged file
                    raise ValueError(
                        f"{self.name}: a categorical split sends no category left (its bitset is empty): damaged or edited by hand"
                    )
                value[i], dl[i] = 0.0, False
            else:
                value[i], dl[i], zm[i] = numeric_split(float(thr[i]), bool(dt[i] & 2), int((dt[i] >> 2) & 3))
        return {
            "left": np.concatenate([lc, -np.ones(nl, np.int32)]),
            "right": np.concatenate([rc, -np.ones(nl, np.int32)]),
            "feature": np.concatenate([feat, np.zeros(nl, np.int64)]).astype(np.int32),
            "value": np.concatenate([value, leaf]),
            "dl": np.concatenate([dl, np.zeros(nl, bool)]),
            "zm": np.concatenate([zm, np.zeros(nl, bool)]) if zm.any() else None,
            "cat": cat or None,
            "linear": {ni + j: rec for j, rec in enumerate(lin)} if lin else None,
            "leaf64": np.concatenate([np.zeros(ni), leaf64]),
        }

    def _linear(self, d: dict, nl: int, factor: float) -> list[tuple]:
        """The linear model of every leaf of one tree: (constant, fallback, features, coefficients), scaled like the plain leaves."""
        const, fallback = floats(d.get("leaf_const")) * factor, floats(d["leaf_value"]) * factor
        cnt = floats(d.get("num_features"), np.int64)
        feats, coefs = floats(d.get("leaf_features"), np.int64), floats(d.get("leaf_coeff")) * factor
        if not (len(const) == len(cnt) == nl and cnt.min(initial=0) >= 0 and cnt.sum() == len(feats) == len(coefs)):
            raise ValueError("linear tree: leaf_const / num_features / leaf_features / leaf_coeff do not describe num_leaves leaves")
        if len(feats) and (feats.min() < 0 or feats.max() > int(self.h["max_feature_idx"])):
            raise ValueError("linear tree: a leaf uses a feature index outside the model")
        if not (np.isfinite(const).all() and np.isfinite(coefs).all() and np.isfinite(fallback).all()):
            raise ValueError("linear tree: a non-finite constant or coefficient")
        end = np.cumsum(cnt)
        return [
            (float(const[j]), float(fallback[j]), feats[end[j] - cnt[j] : end[j]].astype(np.int32), coefs[end[j] - cnt[j] : end[j]].copy())
            for j in range(nl)
        ]

    # ---- the Source interface
    @property
    def size_bytes(self) -> int:
        return self.path.stat().st_size

    @property
    def booster(self):
        if self._booster is None:
            import lightgbm

            try:
                self._booster = lightgbm.Booster(model_file=qf.library_path(self.path))
            except lightgbm.basic.LightGBMError as e:
                raise ValueError(f"{self.name}: LightGBM cannot load this model ({str(e)[:200]})") from None
        return self._booster

    def model_ir(self) -> qf.ModelIR:
        trees = [
            qf.Tree(int(g), t["left"], t["right"], t["feature"], t["value"], t["dl"], t["zm"], t["cat"], t["linear"], t["leaf64"])
            for t, g in zip(self.trees, self.spec.tree_info, strict=True)
        ]
        return qf.ModelIR(self.spec.base, trees, self.margin_scale)

    @property
    def margin_scale(self) -> float:
        """The factor a base margin is multiplied by: `sigmoid:s` is folded into the leaves, and the margin is in LightGBM's raw-score units
        (the transform is sigma(s * (raw + margin))). An rf model's margin is already the averaged, scaled one (see `predict`)."""
        return 1.0 if self.rf else self.scale

    def leaf_matrix(self, x, chunk: int = 20000) -> np.ndarray:
        if self.linear:
            raise ValueError(
                f"{self.name}: linear-leaf trees (linear_tree=true) cannot be pruned (a leaf is a function of the row, not a table entry); "
                "QuickForest only compiles them: use compile, or optimize --keep 100% --compile"
            )
        qf.check_width(x, self.spec.n_features, self.name)
        out = np.empty((len(self.trees), len(x)), np.float32)
        tables = [t["value"][t["left"] == -1] for t in self.trees]  # leaf j of the model = j-th leaf node here
        for a in range(0, len(x), chunk):
            part = np.ascontiguousarray(x[a : a + chunk], dtype=np.float64)
            idx = self.booster.predict(part, pred_leaf=True).reshape(len(part), -1)
            for i, tab in enumerate(tables):
                out[i, a : a + chunk] = tab[idx[:, i]]
        return out

    def set_threads(self, n: int) -> None:
        self._threads = n  # passed to predict: Booster.reset_parameter crashes on a booster loaded from a model file

    def predict(self, x, margin=False):
        qf.check_width(x, self.spec.n_features, self.name)
        extra = {"num_threads": self._threads} if self._threads else {}
        p = self.booster.predict(np.ascontiguousarray(x, dtype=np.float32).astype(np.float64), raw_score=margin, **extra)
        p = np.asarray(p, dtype=np.float64).reshape(len(x), -1)
        # LightGBM's raw score of an rf model is the SUM of its trees and ignores `sigmoid:s`; the forest's margin is the averaged, scaled
        # one
        if margin:
            p = p * (self.scale / (self.n_iter if self.rf else 1))
        return p

    def save_pruned(self, kept, mean_scale: bool, dest: Path) -> None:
        """A LightGBM text model holding only the kept trees, in iteration order (LightGBM's own averaging of an rf model then
        divides by the kept iterations, which is the rescale the greedy selection assumed)."""
        k, g_n = len(kept[0]), self.groups
        order = [int(kept[g][r]) for r in range(k) for g in range(g_n)]
        blocks = []
        for i, t in enumerate(order):
            b = self.blocks[t]
            nl = b.index(b"\n")
            blocks.append(f"Tree={i}\n".encode() + b[nl + 1 :])
        sizes = b"tree_sizes=" + b" ".join(str(len(b)).encode() for b in blocks)
        head = re.sub(rb"(?m)^tree_sizes=.*$", lambda _m: sizes, self.head, count=1)
        qf.replace_file(dest, data=head + b"".join(blocks) + self.importance_tail(order))

    def importance_tail(self, order) -> bytes:
        """The text after the trees, with the `feature_importances:` block (split counts of every tree) rewritten for the trees that are kept:
        `name=count` for the features that still split, most splits first, as LightGBM writes it."""
        names = self.h["feature_names"].split() if "feature_names" in self.h else []
        if not names:
            return self.tail
        counts = np.zeros(len(names), np.int64)
        for t in order:
            tree = self.trees[t]
            counts += np.bincount(tree["feature"][tree["left"] != -1], minlength=len(names))[: len(names)]
        listed = sorted(((n, int(c)) for n, c in zip(names, counts, strict=True) if c), key=lambda nc: -nc[1])
        block = "".join(f"{n}={c}\n" for n, c in listed).encode()
        return re.sub(rb"(?s)(feature_importances:\n).*?(\n\n|\Z)", lambda m: m.group(1) + block + m.group(2), self.tail, count=1)
