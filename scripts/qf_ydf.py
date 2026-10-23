"""YDF (Yggdrasil Decision Forests) adapter: reads a YDF model folder (random forest or gradient boosted trees; regression, binary and
multiclass classification) through ydf's own inspection API into the neutral forest, and writes pruned copies as normal YDF models.
`ydf` is imported here, lazily, at build time only: a compiled forest never needs it.

Matrix columns are the model's input features in `model.input_feature_names()` order (a row is (rows, features) float32, NaN = missing).
YDF's numerical condition is `value >= threshold` -> positive child; here positive = right, so `x < threshold` goes left, exact in
float32; a NaN follows the condition's own stored `missing` answer (YDF's na_value, which it set at training from the global imputation).
Random forests: regression = mean of the leaves; classification = the share of trees voting for each class (winner_takes_all) or the
mean of the leaf probabilities, so each YDF tree becomes one forest tree per output group (a binary model has one group: class 1).
Gradient boosted trees: initial prediction + sum of the leaves, then sigmoid / softmax (binomial / multinomial log-likelihood) or
identity (squared error, absolute error) or exp (poisson).

Refused with a clear message: categorical, boolean, discretized, oblique, set and vector-sequence conditions or features, ranking /
uplift / survival / anomaly tasks, other GBT losses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import qf_common as qf
import qf_formats

LOSS_KIND = {
    "SQUARED_ERROR": "identity",
    "MEAN_AVERAGE_ERROR": "identity",
    "BINOMIAL_LOG_LIKELIHOOD": "sigmoid",
    "BINARY_FOCAL_LOSS": "sigmoid",
    "MULTINOMIAL_LOG_LIKELIHOOD": "softmax",
    "POISSON": "exp",
}


YDF_SMALL_TREE_LEAVES = 64  # YDF's small-tree engine takes trees of at most 64 leaves


def ydf():
    import ydf as _ydf

    return _ydf


class YdfSource:
    fmt = "ydf"

    def __init__(self, path: Path):
        y = ydf()
        self.path = path
        self.name = path.name
        try:
            self.model = y.load_model(qf.library_path(path))
            self._threads = None  # YDF's own default (all cores) unless set_threads says otherwise
            self._read(y)
        except (RuntimeError, OSError) as e:  # ydf raises ValueError for a bad header or data spec and RuntimeError for a bad node shard
            raise ValueError(f"{path}: {qf.one_line(e)}") from None

    def _read(self, y):
        m = self.model
        gbt = isinstance(m, y.GradientBoostedTreesModel)
        if not (gbt or isinstance(m, y.RandomForestModel)):
            raise ValueError(f"{type(m).__name__}: only random forests and gradient boosted trees are supported")
        task = m.task()
        if task not in (y.Task.REGRESSION, y.Task.CLASSIFICATION):
            raise ValueError(f"task {task.name} is not supported (regression and classification only)")
        classification = task == y.Task.CLASSIFICATION
        feats = sorted(m.input_features(), key=lambda f: f.column_idx)
        bad = [f"{f.name} ({f.semantic.name})" for f in feats if f.semantic.name != "NUMERICAL"]
        if bad:
            raise ValueError(f"only numerical input features are supported; found {', '.join(bad[:5])}")
        self.names = [f.name for f in feats]
        col = {f.column_idx: i for i, f in enumerate(feats)}
        raw = m.get_all_trees()
        self.gbt, self.rf = gbt, not gbt
        n_ydf = len(raw)
        if gbt:
            loss = next((ln.split(":", 1)[1].strip() for ln in m.describe("text").splitlines() if ln.startswith("Loss:")), "?")
            if loss not in LOSS_KIND:
                raise ValueError(f"gradient boosted trees loss {loss} is not supported ({', '.join(LOSS_KIND)})")
            if classification and m.output_logits():  # apply_link_function=False: predict answers the margin, whatever the loss says
                raise ValueError(
                    "gradient boosted trees saved with apply_link_function=False answer the margin (logits), not the probabilities "
                    "the loss implies; train with the link function on"
                )
            kind, per_iter = LOSS_KIND[loss], m.num_trees_per_iteration()
            groups = per_iter
            base = np.asarray(m.initial_predictions(), np.float32)
            self.logits_ok = classification
        else:
            n_classes = len(m.label_classes()) if classification else 1
            if classification and n_classes < 2:  # a forest trained on a single class has no second class to model
                raise ValueError("a YDF random forest trained on one class has no second class to model: train with both classes")
            groups = 1 if (not classification or n_classes == 2) else n_classes
            kind, per_iter, base = "identity", 1, np.zeros(groups, np.float32)
            self.logits_ok = False
        self.wta = bool(m.winner_takes_all()) if self.rf and classification else False
        self.classification = classification
        # structure of every YDF tree once, then one forest tree per output group (RF classification) or one in all (the rest)
        self.shapes = [self._shape(t.root, col) for t in raw]
        irs, tree_info, unit_of = [], [], []
        for u, (arrs, leaves) in enumerate(self.shapes):
            if self.rf and classification:
                for g in range(groups):
                    irs.append(self._ir(arrs, self._rf_leaf(leaves, g, n_ydf)))
                    tree_info.append(g)
                    unit_of.append(u)
            else:
                v = np.array([0.0 if lf is None else lf[0] for lf in leaves], np.float64) / (n_ydf if self.rf else 1)
                irs.append(self._ir(arrs, v))
                tree_info.append(u % per_iter if gbt else 0)
                unit_of.append(u)
        self.irs, self.unit_of = irs, np.asarray(unit_of)
        self.spec = qf_formats.spec_for(
            len(feats),
            groups,
            kind,
            base,
            tree_info,
            [len(t.left) for t in irs],
            self.rf,
            "classification" if classification else "regression",
        )
        self.n_ydf = n_ydf
        if gbt and kind == "exp":
            self._check_exp_link()

    def _check_exp_link(self):
        """A Poisson GBT has no logits switch to read, but `apply_link_function=False` is saved in it all the same and `predict` then answers
        the margin: four probe rows (zeros, ones, minus ones, all missing) tell it apart. With the link on YDF answers exp(clip(margin, -19, 19))
        (its own clamp of the logit), which is exp(margin) except beyond rates of 5.6e-9 and 1.8e8; either answer is accepted, row by row, with
        a relative tolerance on both sides. The margin itself matches neither, since exp(m) - m is at least 1 and a clamped rate is far from m."""
        x = np.array([np.zeros(self.spec.n_features), np.ones(self.spec.n_features), -np.ones(self.spec.n_features)], np.float32)
        x = np.concatenate([x, np.full((1, self.spec.n_features), np.nan, np.float32)])
        margin = (self.spec.base.astype(np.float64)[0] + self.leaf_matrix(x).astype(np.float64).sum(0)).astype(np.float32)
        got = self.predict(x).ravel().astype(np.float64)
        with np.errstate(all="ignore"):  # float32 like YDF: a big margin is inf
            plain = np.exp(margin).astype(np.float64)
            clamped = np.exp(np.clip(margin, -19.0, 19.0)).astype(np.float64)
            ok = np.zeros(len(got), bool)
            for ref in (plain, clamped):
                ok |= (got == ref) | (np.abs(got - ref) <= 1e-3 * np.maximum(np.abs(got), np.abs(ref)))
        if not ok.all():
            raise ValueError(
                "gradient boosted trees with the POISSON loss saved with apply_link_function=False answer the margin (the log of the "
                "rate), not the rate; train with the link function on"
            )

    @staticmethod
    def _shape(root, col):
        """Pre-order arrays of one YDF tree and its leaves' payloads (a tuple per leaf, in node order)."""
        y = ydf().tree
        left, right, feat, thr, dl, leaves = [], [], [], [], [], []
        stack = [(root, -1, 0)]  # (node, parent, side)
        while stack:
            node, parent, side = stack.pop()
            i = len(left)
            for arr, v in ((left, -1), (right, -1), (feat, 0), (thr, 0.0), (dl, False)):
                arr.append(v)
            if parent >= 0:
                (left if side == 0 else right)[parent] = i
            if node.is_leaf:
                v = node.value
                if isinstance(v, y.ProbabilityValue):
                    leaves.append((np.asarray(v.probability, np.float64),))
                elif isinstance(v, y.RegressionValue):
                    leaves.append((float(v.value),))
                else:
                    raise ValueError(f"leaf value {type(v).__name__} is not supported")
                thr[i] = np.nan  # replaced by the leaf value in _ir
            else:
                c = node.condition
                if not isinstance(c, y.NumericalHigherThanCondition):
                    raise ValueError(f"condition {type(c).__name__} is not supported (numerical value >= threshold only)")
                feat[i], thr[i], dl[i] = col[c.attribute], float(c.threshold), not bool(c.missing)
                leaves.append(None)
                stack.append((node.pos_child, i, 1))  # popped second: right = positive
                stack.append((node.neg_child, i, 0))  # popped first: left = negative, the node after its parent
        return {
            "left": np.array(left, np.int32),
            "right": np.array(right, np.int32),
            "feature": np.array(feat, np.int32),
            "thr": np.array(thr, np.float32),
            "dl": np.array(dl, bool),
        }, leaves

    def _rf_leaf(self, leaves, g, n):
        """Per leaf node, the contribution to output group g (class share), already divided by the number of trees."""
        out = np.zeros(len(leaves), np.float64)
        for i, lf in enumerate(leaves):
            if lf is None:
                continue
            p = lf[0]
            cls = g if len(p) > 2 else 1
            out[i] = (float(np.argmax(p) == cls) if self.wta else p[cls]) / n
        return out

    @staticmethod
    def _ir(arrs, leaf_values):
        """A forest tree sharing the structure arrays; leaf nodes carry `leaf_values` (per node, ignored at internal nodes)."""
        leaf64 = np.where(arrs["left"] == -1, np.asarray(leaf_values, np.float64), 0.0)
        value = np.where(arrs["left"] == -1, leaf64, arrs["thr"]).astype(np.float32)
        return qf.Tree(0, arrs["left"], arrs["right"], arrs["feature"], value, arrs["dl"], leaf64=leaf64)

    # ---- the Source interface
    @property
    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.path.iterdir() if f.is_file())

    def model_ir(self, float32_sums: bool = False) -> qf.ModelIR:
        """`float32_sums`: the forest that reproduces YDF's own float32 predictor. YDF picks its inference engine from the trees: with every
        tree at most 64 leaves (YDF's small-tree engine) the sum runs from the initial prediction, tree by tree; with a bigger tree (the
        generic engine) it sums the leaves from zero and adds the initial prediction LAST (the two orders differ by up to 2e-5 at 75
        leaves). The initial prediction then becomes one more single-leaf tree per output group, after all the others, and the forest's
        base is zero."""
        trees = []
        for t, g in zip(self.irs, self.spec.tree_info, strict=True):
            trees.append(qf.Tree(int(g), t.left, t.right, t.feature, t.value, t.default_left, leaf64=t.leaf64))
        base = self.spec.base
        if float32_sums and self.gbt and self.generic_engine:
            for g, b in enumerate(np.asarray(base, np.float32)):
                one = np.array([float(b)])
                trees.append(
                    qf.Tree(
                        g,
                        np.array([-1], np.int32),
                        np.array([-1], np.int32),
                        np.array([0], np.int32),
                        one.astype(np.float32),
                        np.array([False]),
                        leaf64=one,
                    )
                )
            base = np.zeros_like(base)
        return qf.ModelIR(base, trees)

    @property
    def generic_engine(self) -> bool:
        """True when YDF would run this model with its generic engine (a tree of more than 64 leaves), not its small-tree one."""
        return any(int((t.left == -1).sum()) > YDF_SMALL_TREE_LEAVES for t in self.irs)

    def leaf_matrix(self, x) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        qf.check_width(x, self.spec.n_features, self.name)
        out = np.empty((len(self.irs), len(x)), np.float32)
        done = {}
        for i, (t, u) in enumerate(zip(self.irs, self.unit_of, strict=True)):
            if u not in done:  # trees of one unit share their structure: walk once
                done = {u: qf.leaf_positions(t, x)}
            out[i] = t.value[done[u]]
        return out

    def set_threads(self, n: int) -> None:
        self._threads = n  # YDF's predict(num_threads=...): the speed baseline is ONE thread, as xgboost's and LightGBM's

    def predict(self, x, margin=False):
        x = np.ascontiguousarray(x, dtype=np.float32)
        qf.check_width(x, self.spec.n_features, self.name)
        data = {n: x[:, i] for i, n in enumerate(self.names)}
        use_logits = margin and self.logits_ok
        if use_logits:
            was = self.model.output_logits()  # give the model's own value back (a model that answers logits is refused at load)
            self.model.set_output_logits(True)
        try:
            p = self.model.predict(data, **({} if self._threads is None else {"num_threads": self._threads}))
        finally:
            if use_logits:
                self.model.set_output_logits(was)
        p = np.asarray(p, np.float64).reshape(len(x), -1)
        if margin and not use_logits and self.spec.kind == "exp":
            p = np.log(p)  # a Poisson GBT answers exp(margin) and has no logits switch
        return p

    def save_pruned(self, kept, mean_scale: bool, dest: Path) -> None:
        """A YDF model holding only the kept trees (a random forest averages what is left; boosted trees keep iteration order)."""
        k, g_n = len(kept[0]), self.spec.groups
        if self.gbt:
            order = [int(self.unit_of[kept[g][r]]) for r in range(k) for g in range(g_n)]
        else:
            order = sorted({int(self.unit_of[i]) for g in kept for i in g})
        m = ydf().load_model(qf.library_path(self.path))
        trees = m.get_all_trees()
        for i, t in enumerate(order):
            m.set_tree(i, trees[t])
        for j in range(len(trees) - 1, len(order) - 1, -1):
            m.remove_tree(j)
        qf.replace_dir(dest, lambda tmp: m.save(qf.library_path(tmp)))
