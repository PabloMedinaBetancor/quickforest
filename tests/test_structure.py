"""The kept tree indices of `prune` and the structure export (`scripts/structure.py`), on small random models of the three libraries.

* `prune` / `optimize` say which trees of the ORIGINAL model they kept: the indices, applied to the original model's exported structure,
  give exactly the pruned file's structure (tree by tree, array by array; the leaves of a random forest are rescaled by the pruning);
* the exported arrays, walked in plain numpy, give the library's margin on 200 rows within the bar the repository uses for that library
  (identical for xgboost, 1e-6 relative to max(1, |value|) for LightGBM and YDF: `check.TOL`);
* refusals: categorical splits, linear leaves, a damaged file, a folder of folds, an output that cannot be written; and a path with a
  space and a non-ASCII name works.

YDF cases skip where ydf is not installed. Synthetic data only.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import check
import optimize
import prune
import qf_formats
import structure

HAS_YDF = importlib.util.find_spec("ydf") is not None
NF, N = 6, 1500
NODE_ARRAYS = ("left", "right", "feature", "threshold", "default_left", "zero_missing", "leaf_index")


def make_x(seed: int = 3, n: int = N) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, NF)).astype(np.float32)
    x[rng.random(x.shape) < 0.25] = 0.0  # zeros, for the zero-as-missing model
    x[rng.random(x.shape) < 0.06] = np.nan
    return x


def labels(x: np.ndarray):
    z = np.nan_to_num(x)
    score = np.stack([z[:, 0] + z[:, 1], np.sin(2 * z[:, 2]) - z[:, 3], z[:, 4] ** 2 - z[:, 5]], 1)
    return score.argmax(1), (z[:, 0] + z[:, 1] * z[:, 2] > 0.1).astype(int), (z[:, 0] * 2 + np.sin(z[:, 1]) + z[:, 4]).astype(np.float64)


def ydf_frame(x, y):
    import ydf  # noqa: F401

    return {**{f"f{i}": np.nan_to_num(x[:, i]) for i in range(NF)}, "y": y}


def tree_slices(a: dict, wanted: list[int]) -> list[dict]:
    """The listed trees of an export, each as its own slice of every per-node array."""
    off = a["node_offset"]
    return [{k: a[k][off[t] : off[t + 1]] for k in NODE_ARRAYS} | {"group": int(a["tree_group"][t])} for t in wanted]


def walk_margin(a: dict, x: np.ndarray) -> np.ndarray:
    """The margin of the exported structure in plain numpy: base margin plus the leaf every row reaches in every tree, trees in order,
    float32 sums for xgboost and float64 sums for the others (the libraries' own)."""
    dtype = np.float32 if str(a["format"]) == "xgboost" else np.float64
    n, rows = len(x), np.arange(len(x))
    margin = np.tile(a["base_margin"].astype(dtype), (n, 1))
    zero = float(np.float32(1e-35))
    off = a["node_offset"]
    for t in range(len(a["tree_group"])):
        lo, hi = off[t], off[t + 1]
        left, right, feat, thr = (a[k][lo:hi] for k in ("left", "right", "feature", "threshold"))
        dl, zm = a["default_left"][lo:hi], a["zero_missing"][lo:hi]
        pos = np.zeros(n, np.int64)
        while True:
            leaf = left[pos] == -1
            if leaf.all():
                break
            node = np.where(leaf, 0, pos)
            v = x[rows, np.maximum(feat[node], 0)]
            go_left = np.where(np.isnan(v), dl[node], v < thr[node])
            go_left = np.where(zm[node] & (np.abs(v.astype(np.float64)) <= zero), dl[node], go_left)
            pos = np.where(leaf, pos, np.where(go_left, left[node], right[node]))
        g = int(a["tree_group"][t])
        margin[:, g] = margin[:, g] + thr[pos].astype(dtype)
    return margin


def hugging_rows(a: dict, base: np.ndarray, rng, n: int = 400) -> np.ndarray:
    """Rows that land ON a split threshold of the exported structure, one float32 step below it and one above it (the places where a
    wrong threshold or a wrong side shows), the other features taken from the base rows."""
    inner = np.flatnonzero(a["left"] != -1)
    pick = rng.choice(inner, n)
    rows = base[rng.integers(0, len(base), n)].copy()
    thr = a["threshold"][pick]
    step = rng.integers(-1, 2, n)  # -1 below, 0 on, +1 above
    value = np.where(step < 0, np.nextafter(thr, np.float32(-np.inf)), np.where(step > 0, np.nextafter(thr, np.float32(np.inf)), thr))
    rows[np.arange(n), a["feature"][pick]] = value
    return rows


class Fixtures:
    """Small random models of every kind, trained once."""

    def __init__(self, root: Path):
        self.root = root
        self.x = make_x()
        y3, y2, yr = labels(self.x)
        self.cases: dict[str, dict] = {}
        d = root / "models"
        d.mkdir()
        xp = {"max_depth": 4, "nthread": 1, "verbosity": 0, "seed": 1, "tree_method": "hist", "eta": 0.3}
        self.add("xgb binary", xgb.train({**xp, "objective": "binary:logistic"}, xgb.DMatrix(self.x, y2), 20), d / "xgb-binary.ubj")
        self.add(
            "xgb multiclass",
            xgb.train({**xp, "objective": "multi:softprob", "num_class": 3}, xgb.DMatrix(self.x, y3), 12),
            d / "xgb-multi.json",
        )
        self.add("xgb regression", xgb.train({**xp, "objective": "reg:squarederror"}, xgb.DMatrix(self.x, yr), 25), d / "xgb-reg.ubj")
        rf = {**xp, "objective": "multi:softprob", "num_class": 3, "num_parallel_tree": 12, "subsample": 0.8, "colsample_bynode": 0.8}
        self.add("xgb random forest", xgb.train({**rf, "eta": 1}, xgb.DMatrix(self.x, y3), 1), d / "xgb-rf.ubj", rf=True)
        lp = dict(num_leaves=9, verbose=-1, num_threads=1, seed=1, min_data_in_leaf=5)
        self.add("lgb binary", lgb.train({**lp, "objective": "binary"}, lgb.Dataset(self.x, y2), 20), d / "lgb-binary.txt")
        self.add(
            "lgb multiclass", lgb.train({**lp, "objective": "multiclass", "num_class": 3}, lgb.Dataset(self.x, y3), 10), d / "lgb-multi.txt"
        )
        self.add("lgb regression", lgb.train({**lp, "objective": "regression"}, lgb.Dataset(self.x, yr), 20), d / "lgb-reg.txt")
        self.add(
            "lgb zero as missing",
            lgb.train({**lp, "objective": "binary", "zero_as_missing": True}, lgb.Dataset(self.x, y2), 20),
            d / "lgb-zero.txt",
        )
        self.add(
            "lgb random forest",
            lgb.train(
                {**lp, "objective": "multiclass", "num_class": 3, "boosting": "rf", "bagging_freq": 1, "bagging_fraction": 0.7},
                lgb.Dataset(self.x, y3),
                12,
            ),
            d / "lgb-rf.txt",
            rf=True,
        )
        if HAS_YDF:
            import ydf

            frame3, frame2, framer = ydf_frame(self.x, y3), ydf_frame(self.x, y2), ydf_frame(self.x, yr)
            cls, reg = ydf.Task.CLASSIFICATION, ydf.Task.REGRESSION
            for name, learner, frame, rf_ in (
                ("ydf gbt binary", ydf.GradientBoostedTreesLearner(label="y", task=cls, num_trees=20, max_depth=4), frame2, False),
                ("ydf gbt multiclass", ydf.GradientBoostedTreesLearner(label="y", task=cls, num_trees=10, max_depth=3), frame3, False),
                ("ydf gbt regression", ydf.GradientBoostedTreesLearner(label="y", task=reg, num_trees=20, max_depth=4), framer, False),
                ("ydf rf multiclass", ydf.RandomForestLearner(label="y", task=cls, num_trees=16, max_depth=5), frame3, True),
                ("ydf rf binary", ydf.RandomForestLearner(label="y", task=cls, num_trees=16, max_depth=5), frame2, True),
                ("ydf rf regression", ydf.RandomForestLearner(label="y", task=reg, num_trees=16, max_depth=5), framer, True),
            ):
                path = d / name.replace(" ", "-")
                learner.train(frame, verbose=0).save(str(path))
                self.cases[name] = {"path": path, "rf": rf_}
        # a k-fold folder of two xgboost folds
        (d / "kfold").mkdir()
        for k in range(2):
            xgb.train({**xp, "objective": "binary:logistic", "seed": k}, xgb.DMatrix(self.x[k::2], y2[k::2]), 10).save_model(
                str(d / "kfold" / f"fold{k}.ubj")
            )
        self.kfold = d / "kfold"
        self.data = root / "data.npz"
        np.savez(self.data, X=self.x)

    def add(self, name: str, booster, path: Path, rf: bool = False):
        booster.save_model(str(path))
        self.cases[name] = {"path": path, "rf": rf}


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fx = Fixtures(Path(cls.tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def out_dir(self, name: str) -> Path:
        return Path(self.tmp.name) / "out" / name.replace(" ", "-")


class KeptIndicesTest(Base):
    def check_pruned_is_the_selection(self, name: str, case: dict, rep: dict, out: Path, model: Path):
        ki = rep["kept_indices"][model.name]
        orig, pruned = structure.export(model), structure.export(out / model.name)
        groups = int(orig["n_groups"])
        self.assertEqual(ki["original_trees"], int(orig["source_tree"].max()) + 1, name)
        self.assertEqual(len(ki["per_group"]), groups, name)
        for lst in (*ki["per_group"], ki["flat"]):
            self.assertEqual(lst, sorted(set(lst)), f"{name}: a list of kept indices is sorted and has no repeat")
            self.assertTrue(all(isinstance(i, int) and 0 <= i < ki["original_trees"] for i in lst), name)
        self.assertEqual(ki["flat"], sorted(set().union(*ki["per_group"])), name)
        for g in range(groups):
            mine = [
                t for t in range(len(orig["tree_group"])) if orig["tree_group"][t] == g and orig["source_tree"][t] in ki["per_group"][g]
            ]
            theirs = [t for t in range(len(pruned["tree_group"])) if pruned["tree_group"][t] == g]
            self.assertEqual(len(mine), len(ki["per_group"][g]), f"{name} group {g}: one exported tree per kept library tree")
            self.assertEqual(len(mine), len(theirs), f"{name} group {g}: the pruned file holds exactly the kept trees")
            ratio = sum(orig["tree_group"] == g) / len(theirs)  # a random forest's leaves are rescaled by trees / kept
            for t_orig, t_pruned in zip(tree_slices(orig, mine), tree_slices(pruned, theirs), strict=True):
                for key in NODE_ARRAYS:
                    a, b = t_orig[key], t_pruned[key]
                    if key == "threshold":
                        inner = t_orig["left"] != -1
                        np.testing.assert_array_equal(a[inner], b[inner], f"{name} group {g}: thresholds")
                        if not case["rf"]:
                            np.testing.assert_array_equal(a, b, f"{name} group {g}: leaf values")
                        elif orig["format"] == "xgboost":  # the pruning multiplies the float32 leaf by float32(trees / kept)
                            np.testing.assert_array_equal(a[~inner] * np.float32(ratio), b[~inner], f"{name} group {g}: rescaled leaves")
                        else:
                            np.testing.assert_allclose(
                                a[~inner] * ratio, b[~inner], rtol=2e-6, err_msg=f"{name} group {g}: rescaled leaves"
                            )
                    else:
                        np.testing.assert_array_equal(a, b, f"{name} group {g}: {key}")

    def test_the_kept_indices_applied_to_the_original_structure_give_the_pruned_structure(self):
        for name, case in self.fx.cases.items():
            with self.subTest(name):
                out = self.out_dir("kept-" + name)
                rep = prune.run(case["path"], self.fx.data, out, keep="40%")
                self.assertEqual(list(rep["kept_indices"]), [case["path"].name])
                self.check_pruned_is_the_selection(name, case, rep, out, case["path"])
                self.assertEqual(json.loads((out / "report.json").read_text())["kept_indices"], rep["kept_indices"])

    def test_the_cheaper_orientation_method_names_its_kept_trees_the_same_way(self):
        for name in ("xgb multiclass", "lgb multiclass", "xgb random forest", "lgb binary"):
            with self.subTest(name):
                case = self.fx.cases[name]
                out = self.out_dir("orientation-" + name)
                rep = prune.run(case["path"], self.fx.data, out, keep="3", method="orientation")
                self.check_pruned_is_the_selection(name, case, rep, out, case["path"])
                self.assertTrue(all(len(lst) == 3 for lst in rep["kept_indices"][case["path"].name]["per_group"]))

    def test_every_index_is_listed_when_nothing_is_pruned(self):
        for name, case in self.fx.cases.items():
            with self.subTest(name):
                everything = prune.run(case["path"], self.fx.data, self.out_dir("all-" + name), keep="100%")["kept_indices"][
                    case["path"].name
                ]
                n = everything["original_trees"]
                self.assertEqual(everything["flat"], list(range(n)), name)
                src = qf_formats.open_source(case["path"])
                unit_of = getattr(src, "unit_of", None)
                for g, lst in enumerate(everything["per_group"]):
                    in_group = np.flatnonzero(src.spec.tree_info == g)
                    self.assertEqual(
                        lst, sorted({int(i) for i in (in_group if unit_of is None else unit_of[in_group])}), f"{name} group {g}"
                    )

    def test_a_kfold_folder_gives_one_entry_per_file(self):
        out = self.out_dir("kfold")
        rep = prune.run(self.fx.kfold, self.fx.data, out, keep="50%")
        self.assertEqual(sorted(rep["kept_indices"]), ["fold0.ubj", "fold1.ubj"])
        for f in qf_formats.model_files(self.fx.kfold):
            self.check_pruned_is_the_selection("kfold", {"rf": False}, rep, out, f)
            self.assertEqual(len(rep["kept_indices"][f.name]["flat"]), 5)

    def test_optimize_reports_the_kept_indices_and_the_pruned_report_agrees(self):
        model = self.fx.cases["lgb binary"]["path"]
        rep = optimize.optimize(model, self.fx.data, self.out_dir("opt"), "30%", min_agreement=1.0)
        on_disk = json.loads((self.out_dir("opt") / "report.json").read_text())
        self.assertEqual(on_disk["kept_indices"], rep["kept_indices"])
        self.assertEqual(json.loads((self.out_dir("opt") / "pruned" / "report.json").read_text())["kept_indices"], rep["kept_indices"])
        self.check_pruned_is_the_selection("optimize", {"rf": False}, rep, self.out_dir("opt") / "pruned", model)
        # keeping everything: pruned/ holds the models as they are, and the report names every tree
        full = optimize.optimize(model, self.fx.data, self.out_dir("opt-all"), "100%")
        self.assertEqual(full["kept_indices"][model.name]["flat"], list(range(full["kept_indices"][model.name]["original_trees"])))
        self.assertEqual(json.loads((self.out_dir("opt-all") / "report.json").read_text())["kept_indices"], full["kept_indices"])
        # a refused run leaves no pruned/, so it names no kept trees
        refused = optimize.optimize(model, self.fx.data, self.out_dir("opt-refused"), "30%", min_agreement=100.0)
        self.assertFalse(refused["ok"])
        self.assertNotIn("kept_indices", json.loads((self.out_dir("opt-refused") / "report.json").read_text()))


class StructureExportTest(Base):
    def test_the_arrays_walked_in_numpy_give_the_margin_of_the_library(self):
        rng = np.random.default_rng(11)
        plain = make_x(seed=5, n=200)
        plain[rng.random(plain.shape) < 0.1] = 0.0
        for name, case in self.fx.cases.items():
            with self.subTest(name):
                a = structure.export(case["path"])
                rows = np.concatenate([plain, hugging_rows(a, plain, rng)])
                got = walk_margin(a, rows)
                want = qf_formats.open_source(case["path"]).predict(rows, margin=True)
                want = want.reshape(len(rows), -1)
                self.assertEqual(got.shape, want.shape, name)
                if str(a["format"]) == "xgboost":
                    np.testing.assert_array_equal(got, want, f"{name}: xgboost is claimed identical to the bit")
                else:
                    err = np.abs(got - want) / np.maximum(1.0, np.abs(want))
                    self.assertLessEqual(float(err.max()), check.TOL, name)

    def test_the_pruned_models_walk_too(self):
        rows = make_x(seed=6, n=200)
        for name in ("xgb multiclass", "lgb zero as missing"):
            case = self.fx.cases[name]
            prune.run(case["path"], self.fx.data, self.out_dir("walk-" + name), keep="40%")
            pruned = self.out_dir("walk-" + name) / case["path"].name
            got = walk_margin(structure.export(pruned), rows)
            want = qf_formats.open_source(pruned).predict(rows, margin=True).reshape(len(rows), -1)
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=name)

    def test_trees_of_a_single_leaf(self):
        # a constant target: both libraries grow trees that are one leaf
        y = np.full(len(self.fx.x), 2.5)
        d = Path(self.tmp.name) / "constant"
        d.mkdir()
        xgb.train({"objective": "reg:squarederror", "verbosity": 0, "max_depth": 3}, xgb.DMatrix(self.fx.x, y), 4).save_model(
            str(d / "c.ubj")
        )
        lgb.train({"objective": "regression", "verbose": -1, "min_data_in_leaf": 5}, lgb.Dataset(self.fx.x, y), 4).save_model(
            str(d / "c.txt")
        )
        rows = make_x(seed=9, n=50)
        for model in (d / "c.ubj", d / "c.txt"):
            a = structure.export(model)
            self.assertEqual(a["left"].tolist(), [-1] * len(a["tree_group"]), model.name)
            self.assertEqual(a["leaf_index"].tolist(), [0] * len(a["tree_group"]), model.name)
            self.assertEqual(a["node_offset"].tolist(), list(range(len(a["tree_group"]) + 1)), model.name)
            want = qf_formats.open_source(model).predict(rows, margin=True).reshape(len(rows), -1)
            np.testing.assert_allclose(walk_margin(a, rows), want, rtol=1e-6, atol=1e-6, err_msg=model.name)

    def test_the_arrays_are_what_the_reference_says(self):
        a = structure.export(self.fx.cases["xgb multiclass"]["path"])
        self.assertEqual(str(a["format"]), "xgboost")
        self.assertEqual((str(a["kind"]), str(a["task"]), int(a["n_groups"]), int(a["n_features"])), ("softmax", "classification", 3, NF))
        self.assertEqual(a["base_margin"].dtype, np.float32)
        self.assertEqual(a["base_margin"].shape, (3,))
        n_trees, n_nodes = len(a["tree_group"]), len(a["left"])
        self.assertEqual(n_trees, 36)
        self.assertEqual(len(a["node_offset"]), n_trees + 1)
        self.assertEqual((int(a["node_offset"][0]), int(a["node_offset"][-1])), (0, n_nodes))
        self.assertEqual(a["tree_group"].tolist(), [t % 3 for t in range(36)])
        self.assertEqual(a["source_tree"].tolist(), list(range(36)))
        dtypes = {
            "left": np.int32,
            "right": np.int32,
            "feature": np.int32,
            "threshold": np.float32,
            "default_left": np.bool_,
            "leaf_index": np.int32,
        }
        for key, dtype in dtypes.items():
            self.assertEqual(a[key].dtype, dtype, key)
            self.assertEqual(a[key].shape, (n_nodes,), key)
        leaf = a["left"] == -1
        np.testing.assert_array_equal(a["right"] == -1, leaf)
        np.testing.assert_array_equal(a["feature"] == -1, leaf)
        self.assertFalse(a["default_left"][leaf].any())
        np.testing.assert_array_equal(a["leaf_index"] == -1, ~leaf)
        for t in range(n_trees):  # the leaves of every tree are numbered 0, 1, 2, ... in node order; children stay inside the tree
            lo, hi = a["node_offset"][t], a["node_offset"][t + 1]
            lf = leaf[lo:hi]
            self.assertEqual(a["leaf_index"][lo:hi][lf].tolist(), list(range(int(lf.sum()))))
            kids = np.concatenate([a["left"][lo:hi][~lf], a["right"][lo:hi][~lf]])
            self.assertTrue(((kids > 0) & (kids < hi - lo)).all())
        # LightGBM: a zero-as-missing model uses the zero_missing array, the others have none set
        self.assertTrue(structure.export(self.fx.cases["lgb zero as missing"]["path"])["zero_missing"].any())
        self.assertFalse(a["zero_missing"].any())
        # leaves count as many as the library reports
        booster = xgb.Booster()
        booster.load_model(str(self.fx.cases["xgb multiclass"]["path"]))
        self.assertEqual(
            int(leaf.sum()),
            sum(
                int((np.array(t["left_children"]) == -1).sum())
                for t in json.loads(booster.save_raw(raw_format="json"))["learner"]["gradient_booster"]["model"]["trees"]
            ),
        )

    @unittest.skipUnless(HAS_YDF, "ydf is not installed")
    def test_a_ydf_random_forest_is_one_exported_tree_per_class_and_names_its_library_tree(self):
        a = structure.export(self.fx.cases["ydf rf multiclass"]["path"])
        self.assertEqual(a["tree_group"].tolist(), [g for _ in range(16) for g in range(3)])
        self.assertEqual(a["source_tree"].tolist(), [u for u in range(16) for _ in range(3)])
        b = structure.export(self.fx.cases["ydf rf binary"]["path"])
        self.assertEqual((len(b["tree_group"]), b["source_tree"].tolist()), (16, list(range(16))))

    def test_save_writes_an_archive_that_loads_without_pickle(self):
        out = self.out_dir("save") / "s.npz"
        out.parent.mkdir(parents=True)
        arrays = structure.save(self.fx.cases["lgb multiclass"]["path"], out)
        with np.load(out, allow_pickle=False) as z:
            self.assertEqual(sorted(z.files), sorted(arrays))
            for key in z.files:
                np.testing.assert_array_equal(z[key], arrays[key], key)
                self.assertEqual(z[key].dtype, arrays[key].dtype, key)

    def test_structure_reads_the_model_without_predicting(self):
        # no prediction on the model: neither the library's predict nor the neutral forest's evaluator may be called
        from unittest import mock

        import qf_common

        for name in ("xgb binary", "lgb binary") + (("ydf gbt binary",) if HAS_YDF else ()):
            with (
                mock.patch.object(qf_common, "ir_predict", side_effect=AssertionError("predict")),
                mock.patch.object(qf_common, "leaf_positions", side_effect=AssertionError("predict")),
                mock.patch.object(xgb.Booster, "predict", side_effect=AssertionError("predict")),
                mock.patch.object(lgb.Booster, "predict", side_effect=AssertionError("predict")),
            ):
                structure.export(self.fx.cases[name]["path"])


class RefusalsTest(Base):
    def run_cli(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-s", str(SCRIPTS / "quickforest"), "structure", *map(str, args)], capture_output=True, text=True, timeout=300
        )

    def assert_one_line_refusal(self, r: subprocess.CompletedProcess, *words: str):
        lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(len(lines), 1, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        for w in words:
            self.assertIn(w, lines[0])

    def test_categorical_and_linear_models_are_refused_with_one_line_and_exit_1(self):
        x = self.fx.x
        y = labels(x)[1]
        cat = np.nan_to_num(x).copy()
        cat[:, 3] = np.random.default_rng(0).integers(0, 6, len(x))
        d = Path(self.tmp.name) / "ext"
        d.mkdir()
        lp = dict(objective="binary", num_leaves=9, verbose=-1, num_threads=1, min_data_in_leaf=5)
        lgb.train(lp, lgb.Dataset(cat, (cat[:, 3] % 2 == 0) ^ (y > 5), categorical_feature=[3]), 5).save_model(str(d / "cat.txt"))
        lgb.train({**lp, "linear_tree": True}, lgb.Dataset(np.nan_to_num(x), y), 5).save_model(str(d / "linear.txt"))
        r = self.run_cli(d / "cat.txt", d / "cat.npz")
        self.assert_one_line_refusal(r, "categorical")
        self.assertFalse((d / "cat.npz").exists())
        r = self.run_cli(d / "linear.txt", d / "linear.npz")
        self.assert_one_line_refusal(r, "linear")
        self.assertFalse((d / "linear.npz").exists())
        for model in (d / "cat.txt", d / "linear.txt"):
            with self.assertRaises(ValueError):
                structure.export(model)

    def test_a_damaged_file_is_one_line_and_exit_1(self):
        d = Path(self.tmp.name) / "damaged"
        d.mkdir()
        blob = self.fx.cases["xgb binary"]["path"].read_bytes()
        (d / "cut.ubj").write_bytes(blob[: len(blob) // 2])
        (d / "junk.ubj").write_bytes(os.urandom(300))
        (d / "empty.txt").write_bytes(b"")
        text = self.fx.cases["lgb binary"]["path"].read_text()
        (d / "cut.txt").write_text(text[: len(text) // 2])
        for name in ("cut.ubj", "junk.ubj", "empty.txt", "cut.txt", "missing.ubj"):
            with self.subTest(name):
                self.assert_one_line_refusal(self.run_cli(d / name, d / "o.npz"))
                self.assertFalse((d / "o.npz").exists())

    def test_a_folder_of_folds_and_a_bad_output_are_refused(self):
        d = Path(self.tmp.name) / "bad-out"
        d.mkdir()
        model = self.fx.cases["xgb binary"]["path"]
        self.assert_one_line_refusal(self.run_cli(self.fx.kfold, d / "o.npz"), "one model file at a time")
        self.assert_one_line_refusal(self.run_cli(model, d / "o.json"), ".npz")
        self.assert_one_line_refusal(self.run_cli(model, d / "no" / "such" / "o.npz"), "no such folder")
        (d / "folder.npz").mkdir()
        self.assert_one_line_refusal(self.run_cli(model, d / "folder.npz"), "folder")
        copy = d / "model.npz"  # a LightGBM text model may carry any name: the archive must not overwrite it
        shutil.copy(self.fx.cases["lgb binary"]["path"], copy)
        before = copy.read_bytes()
        self.assert_one_line_refusal(self.run_cli(copy, copy), "model file itself")
        self.assertEqual(copy.read_bytes(), before)
        self.assertEqual(self.run_cli().returncode, 2)  # no arguments: the option parser

    def test_an_old_archive_is_replaced_and_a_link_is_not_followed(self):
        d = Path(self.tmp.name) / "replace"
        d.mkdir()
        elsewhere = d / "elsewhere.bin"
        elsewhere.write_bytes(b"precious")
        (d / "link.npz").symlink_to(elsewhere)
        r = self.run_cli(self.fx.cases["xgb binary"]["path"], d / "link.npz")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(elsewhere.read_bytes(), b"precious")
        self.assertFalse((d / "link.npz").is_symlink())
        self.assertEqual(int(np.load(d / "link.npz")["n_features"]), NF)

    def test_a_path_with_a_space_and_a_non_ascii_name_works(self):
        d = Path(self.tmp.name) / "dir with space ü 日本"
        d.mkdir()
        for name, case in self.fx.cases.items():
            if name not in ("xgb binary", "lgb binary", "ydf gbt binary"):
                continue
            with self.subTest(name):
                src = d / f"mödel ß {case['path'].name}"
                shutil.copytree(case["path"], src) if case["path"].is_dir() else shutil.copy(case["path"], src)
                r = self.run_cli(src, d / f"trée ß {case['path'].stem}.npz")
                self.assertEqual(r.returncode, 0, r.stderr)
                a = np.load(d / f"trée ß {case['path'].stem}.npz")
                np.testing.assert_array_equal(a["threshold"], structure.export(case["path"])["threshold"])

    def test_a_name_that_is_not_valid_utf8_is_written(self):
        raw = os.fsencode(Path(self.tmp.name)) + b"/caf\xe9.npz"
        structure.save(self.fx.cases["xgb binary"]["path"], os.fsdecode(raw))
        self.assertTrue(os.path.exists(raw))
        self.assertEqual(int(np.load(os.fsdecode(raw))["n_features"]), NF)


if __name__ == "__main__":
    unittest.main()
