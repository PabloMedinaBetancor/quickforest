"""The scikit-learn reader (qf_sklearn): the `.qfsk` export is verified before it is written, the source reads it as a neutral forest whose
answers equal the estimator's (<= 1e-6, also on rows that sit on a threshold), a pruned file equals `subforest`, and a damaged file or an
unsupported estimator ends in one clean line. Models are small and seeded; pickles are used only to hold nothing (QF never reads one).

    python3 -s -m unittest discover -s tests -p "test_sklearn.py"   (a Python with numpy and scikit-learn >= 1.3)
"""

import functools
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.ensemble import (
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qf_common as qf
import qf_formats
import qf_lgb
import qf_sklearn

TOL = 1e-6
N_FEATURES = 8


def rel_err(ref, got):
    ref = np.asarray(ref, np.float64).reshape(len(ref), -1)
    got = np.asarray(got, np.float64).reshape(len(ref), -1)
    return float((np.abs(ref - got) / np.maximum(1.0, np.abs(ref))).max())


def make_data(seed, n=700, nan=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
    x[:, 3] = rng.integers(0, 6, n)  # repeated values: thresholds land between them
    if nan:
        x[rng.random(x.shape) < 0.08] = np.nan
    z = np.nan_to_num(x)
    return x, z


def targets(z, k):
    cols = [z[:, 0] * 3 + z[:, 1], z[:, 2] - 2 * z[:, 3], z[:, 4] * z[:, 5], np.sin(z[:, 6]), z[:, 7]]
    return np.stack([cols[i % 5] * (1 + i) for i in range(k)], 1)


def fit_regressor(cls, k, seed=0, nan=False, trees=12, depth=6):
    x, z = make_data(seed, nan=nan)
    y = targets(z, k)
    y = y[:, 0] if k == 1 else y
    kw = {} if cls is DecisionTreeRegressor else {"n_estimators": trees}
    return cls(max_depth=depth, random_state=seed, **kw).fit(x, y), x


def fit_classifier(classes, seed=0, trees=12, depth=6, nan=False):
    """One output per entry of `classes` (its class count)."""
    x, z = make_data(seed, nan=nan)
    cols = [
        np.digitize(z[:, i] + 0.3 * z[:, i + 1], np.quantile(z[:, i] + 0.3 * z[:, i + 1], np.linspace(0, 1, c + 1)[1:-1]))
        for i, c in enumerate(classes)
    ]
    y = np.stack(cols, 1)
    y = y[:, 0] if len(classes) == 1 else y
    return RandomForestClassifier(n_estimators=trees, max_depth=depth, random_state=seed).fit(x, y), x


@functools.cache
def models():
    """name -> (estimator, rows): the layouts of DISENO 5.5, smaller."""
    out = {}
    for k in (1, 2, 5, 16):
        out[f"rf_reg_k{k}"] = fit_regressor(RandomForestRegressor, k, trees=8 if k == 16 else 12)
    out["et_reg_k3"] = fit_regressor(ExtraTreesRegressor, 3)
    out["dt_reg_k2"] = fit_regressor(DecisionTreeRegressor, 2)
    out["rf_reg_k2_nan"] = fit_regressor(RandomForestRegressor, 2, nan=True)
    out["rf_clf_k2_c3_2"] = fit_classifier((3, 2))
    out["rf_clf_k1_c4"] = fit_classifier((4,), seed=1)
    return out


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        cls.models = models()
        cls.files = {}
        for name, (est, x) in cls.models.items():
            cls.files[name] = cls.t / f"{name}.qfsk"
            qf_sklearn.export_sklearn(est, cls.files[name], x)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def answer(self, name, rows):
        est = self.models[name][0]
        return qf_sklearn.estimator_answer(est, rows, name.startswith("rf_clf"))


class ExportTest(Base):
    def test_export_reproduces_the_estimator_on_data_rows_and_threshold_rows(self):
        for name, (est, x) in self.models.items():
            info = qf_sklearn.export_sklearn(est, self.t / f"again_{name}.qfsk", x)
            self.assertLessEqual(info["max_diff"], 1e-9, name)
            self.assertGreaterEqual(info["verified_rows"], len(x) + qf_sklearn.PROBE_ROWS, name)

    def test_the_probe_rows_hit_thresholds_and_hold_nan(self):
        arrays = qf_sklearn.load_arrays(self.files["rf_reg_k5"])
        rows = qf_sklearn.probe_rows(arrays, N_FEATURES)
        inner = np.flatnonzero(arrays["left"] != -1)
        exact = np.isin(rows[:, 0], np.float32(arrays["threshold"][inner][arrays["feature"][inner] == 0]))
        self.assertGreater(int(exact.sum()), 20)
        self.assertTrue(np.isnan(rows).any())

    def test_every_estimator_it_cannot_hold_is_refused_in_one_line_and_nothing_is_written(self):
        x, z = make_data(5)
        y = targets(z, 1)[:, 0]
        one_class = RandomForestClassifier(n_estimators=3, max_depth=3).fit(x, np.zeros(len(x), int))
        cases = {
            "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=3).fit(x, y),
            "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=3).fit(x, y),
            "BaggingRegressor": BaggingRegressor(n_estimators=3).fit(x, y),
            "Pipeline": make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=3)).fit(x, y),
            "is not fitted": RandomForestRegressor(n_estimators=3),
            "single class": one_class,
        }
        for label, est in cases.items():
            path = self.t / "refused.qfsk"
            with self.assertRaises(ValueError, msg=label) as cm:
                qf_sklearn.export_sklearn(est, path)
            self.assertNotIn("\n", str(cm.exception), label)
            self.assertFalse(path.exists(), label)
        est = self.models["rf_reg_k2"][0]
        with self.assertRaisesRegex(ValueError, r"\.qfsk"):
            qf_sklearn.export_sklearn(est, self.t / "model.pkl")

    def test_an_export_that_does_not_reproduce_the_estimator_is_not_written(self):
        est, x = self.models["rf_reg_k2"]
        real = qf_sklearn.reference_walk
        path = self.t / "wrong.qfsk"
        with mock.patch.object(qf_sklearn, "reference_walk", lambda *a, **k: real(*a, **k) + 1e-6):
            with self.assertRaisesRegex(ValueError, "export does not reproduce the estimator"):
                qf_sklearn.export_sklearn(est, path, x)
        self.assertFalse(path.exists())

    def test_a_sklearn_older_than_1_3_is_refused_in_one_line(self):
        est = self.models["rf_reg_k2"][0]
        with mock.patch("sklearn.__version__", "1.2.2"):
            with self.assertRaisesRegex(ValueError, "sklearn 1.3 or newer"):
                qf_sklearn.export_sklearn(est, self.t / "old.qfsk")

    def test_leaf_counts_are_divided_into_fractions_and_an_empty_leaf_is_refused(self):
        left = np.array([1, -1, -1], np.int32)
        value = np.zeros((3, 2, 3))
        value[1, 0, :3], value[1, 1, :2] = [2, 4, 2], [1, 3]
        value[2, 0, :3], value[2, 1, :2] = [0.2, 0.3, 0.5], [1, 0]
        arrays = {"left": left, "value": value}
        self.assertTrue(qf_sklearn._normalise(arrays, np.array([3, 2])))
        np.testing.assert_allclose(arrays["value"][1, 0, :3], [0.25, 0.5, 0.25])
        np.testing.assert_allclose(arrays["value"][1, 1, :2], [0.25, 0.75])
        self.assertFalse(qf_sklearn._normalise(arrays, np.array([3, 2])))
        arrays["value"][2, 1, :2] = 0
        with self.assertRaisesRegex(ValueError, "no samples"):
            qf_sklearn._normalise(arrays, np.array([3, 2]))

    def test_the_split_value_is_the_float32_just_above_the_largest_float32_not_above_the_threshold(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=500).astype(np.float32)
        t = np.concatenate(
            [
                rng.normal(size=500),
                base.astype(np.float64),
                np.nextafter(base, np.float32(np.inf)).astype(np.float64) - 1e-12,
                [0.0, 1e-45, -1e-45, 1e39, -1e39, np.inf],
            ]
        )
        want = np.array([qf_lgb.up32(qf_lgb.floor32(float(v))) for v in t], np.float32)
        np.testing.assert_array_equal(qf_sklearn.up_threshold(t), want)
        # the rule itself: a float32 x goes left exactly when x <= t
        x = np.concatenate([base, np.nextafter(base, np.float32(np.inf)), np.nextafter(base, np.float32(-np.inf))])
        for v, split in zip(t[:500], qf_sklearn.up_threshold(t[:500]), strict=True):
            np.testing.assert_array_equal(x < split, x.astype(np.float64) <= v)


class SourceTest(Base):
    def forest(self, name, **kw):
        return qf.load_forest([self.files[name]], **kw)

    def test_the_forest_answers_what_the_estimator_answers_on_threshold_rows(self):
        for name, (_, x) in self.models.items():
            forest = self.forest(name)
            rows = np.concatenate([x[:300], qf_sklearn.probe_rows(qf_sklearn.load_arrays(self.files[name]), N_FEATURES)])
            if not name.endswith("nan"):
                rows = rows[~np.isnan(rows).any(axis=1)] if name == "dt_reg_k2" else rows
            self.assertLessEqual(rel_err(self.answer(name, rows), qf.ir_predict(forest, rows)), TOL, name)
            src = qf_formats.open_source(self.files[name])
            self.assertLessEqual(rel_err(self.answer(name, rows), src.predict(rows)), 1e-12, name)

    def test_what_the_forest_needs_and_how_its_outputs_are_laid_out(self):
        for name in ("rf_reg_k2", "rf_reg_k5", "rf_reg_k16", "et_reg_k3", "dt_reg_k2", "rf_clf_k2_c3_2", "rf_clf_k1_c4"):
            f = self.forest(name)
            self.assertTrue({"vector_leaves", "sum64"} <= f.requires, name)
            self.assertEqual(qf_formats.open_source(self.files[name]).spec.vector_width, f.groups, name)
        one = self.forest("rf_reg_k1")
        self.assertEqual((one.groups, "vector_leaves" in one.requires), (1, False))
        self.assertTrue(all(t.vector is None and t.leaf64 is not None for t in one.models[0].trees))
        c = self.forest("rf_clf_k2_c3_2")
        self.assertEqual((c.groups, c.segments, c.task, c.kind), (5, [0, 3, 5], "classification", "identity"))
        self.assertIsNone(self.forest("rf_clf_k1_c4").segments)  # one output: one block
        self.assertIsNone(self.forest("rf_reg_k5").segments)
        # a binary output keeps both of sklearn's columns
        binary = fit_classifier((2,), seed=2)
        qf_sklearn.export_sklearn(binary[0], self.t / "bin.qfsk", binary[1])
        self.assertEqual(qf.load_forest([self.t / "bin.qfsk"]).groups, 2)

    def test_the_replicated_forest_of_the_bench_switch_answers_bit_for_bit_the_same(self):
        for name in ("rf_reg_k5", "rf_clf_k2_c3_2"):
            vec = self.forest(name)
            with mock.patch.object(qf_sklearn, "REPLICATE_OUTPUTS", True):
                rep = self.forest(name)
            self.assertEqual(rep.n_trees, vec.n_trees * vec.groups, name)
            self.assertTrue(all(t.vector is None for t in rep.models[0].trees))
            rows = self.models[name][1][:200]
            np.testing.assert_array_equal(qf.ir_predict(vec, rows), qf.ir_predict(rep, rows), name)

    def test_a_pruned_file_equals_the_pruned_sklearn_model(self):
        for name in ("rf_reg_k5", "rf_clf_k2_c3_2", "et_reg_k3"):
            est, x = self.models[name]
            src = qf_formats.open_source(self.files[name])
            kept = [0, 3, 4, 7, 9]
            dest = self.t / f"pruned_{name}.qfsk"
            src.save_pruned([np.array(kept)] * src.spec.groups, True, dest)
            pruned = qf_formats.open_source(dest)
            self.assertEqual(pruned.n_trees, len(kept))
            small = qf_sklearn.subforest(est, kept)
            self.assertEqual(len(est.estimators_), self.models[name][0].n_estimators)
            want = qf_sklearn.estimator_answer(small, x, name.startswith("rf_clf"))
            self.assertLessEqual(rel_err(want, pruned.predict(x)), 1e-12, name)
            self.assertLessEqual(rel_err(want, qf.ir_predict(qf.load_forest([dest]), x)), TOL, name)
        est = self.models["rf_reg_k2"][0]
        for bad in ([], [1, 1], [-1], [len(est.estimators_)]):
            with self.assertRaises(ValueError, msg=str(bad)):
                qf_sklearn.subforest(est, bad)
        with self.assertRaises(ValueError):
            qf_sklearn.subforest(self.models["dt_reg_k2"][0], [0])

    def test_a_decision_tree_says_it_has_one_tree(self):
        self.assertTrue(qf_formats.open_source(self.files["dt_reg_k2"]).single_tree)
        self.assertFalse(qf_formats.open_source(self.files["rf_reg_k2"]).single_tree)

    def test_leaf_codes_and_vectors_rebuild_the_forest_answer(self):
        name = "rf_reg_k5"
        src, x = qf_formats.open_source(self.files[name]), self.models[name][1][:200]
        codes, tables = src.leaf_codes(x), src.leaf_vectors()
        total = sum(tables[i][codes[i]].astype(np.float64) for i in range(src.n_trees))
        self.assertLessEqual(rel_err(self.answer(name, x), total), TOL)
        with self.assertRaises(ValueError):
            src.leaf_matrix(x)

    def test_a_folder_of_exports_is_the_folds_of_one_ensemble(self):
        folder = self.t / "folds"
        folder.mkdir()
        parts = [fit_regressor(RandomForestRegressor, 3, seed=s, trees=6) for s in (11, 12)]
        for i, (est, x) in enumerate(parts):
            qf_sklearn.export_sklearn(est, folder / f"fold{i}.qfsk", x)
        files = qf_formats.model_files(folder)
        self.assertEqual([f.name for f in files], ["fold0.qfsk", "fold1.qfsk"])
        forest = qf.load_forest(files)
        x = parts[0][1][:100]
        want = np.mean([e.predict(x) for e, _ in parts], axis=0)
        self.assertLessEqual(rel_err(want, qf.ir_predict(forest, x)), TOL)
        odd = fit_classifier((3, 2), seed=3)
        qf_sklearn.export_sklearn(odd[0], folder / "fold2.qfsk", odd[1])
        with self.assertRaisesRegex(ValueError, "disagree"):
            qf.load_forest(qf_formats.model_files(folder))
        (folder / "fold2.qfsk").unlink()
        c1, c2 = fit_classifier((3, 2), seed=4), fit_classifier((2, 3), seed=5)
        qf_sklearn.export_sklearn(c1[0], folder / "fold2.qfsk", c1[1])
        qf_sklearn.export_sklearn(c2[0], folder / "fold3.qfsk", c2[1])
        with self.assertRaisesRegex(ValueError, "disagree"):  # the same 5 columns, other class sets
            qf.load_forest(qf_formats.model_files(folder)[2:])

    def test_a_tree_whose_right_subtree_lies_65535_nodes_away_is_refused_by_the_compile_in_one_line(self):
        import compile

        rng = np.random.default_rng(0)
        x = rng.normal(size=(90000, 3)).astype(np.float32)
        tree = DecisionTreeRegressor(random_state=0).fit(
            x, rng.normal(size=len(x))
        )  # one leaf per row: the left subtree alone is ~90k nodes
        path = self.t / "deep.qfsk"
        qf_sklearn.export_sklearn(tree, path, x[:100])
        with self.assertRaisesRegex(ValueError, "65,535"):
            compile.compile_model(path, self.t / "deep.flat", backend="flat")

    def test_float32_sums_are_for_ydf_only(self):
        with self.assertRaisesRegex(ValueError, "YDF"):
            self.forest("rf_reg_k2", float32_sums=True)

    def test_a_pickle_path_is_refused_with_the_way_in(self):
        for suffix in (".pkl", ".joblib", ".pickle"):
            p = self.t / f"model{suffix}"
            p.write_bytes(b"not read")
            with self.assertRaisesRegex(ValueError, "export_sklearn"):
                qf_formats.open_source(p)


class DamagedFileTest(Base):
    """A `.qfsk` that is not a valid export ends in one `ValueError` line that names the file: never a traceback, never a read outside it."""

    def mutated(self, label, change):
        z = dict(np.load(self.files["rf_clf_k2_c3_2"], allow_pickle=False))
        z = change(z) or z
        path = self.t / f"damaged-{label.replace(' ', '-')}.qfsk"
        np.savez(path, **z)
        return path

    def test_each_damage_is_one_line_naming_the_file(self):
        def cut_offsets(z):
            z["tree_off"] = z["tree_off"][:-1]

        def late_start(z):
            z["tree_off"] = z["tree_off"] + 1

        def child_first(z):
            z["left"] = z["left"].copy()
            z["left"][0] = 0

        def short_value(z):
            z["value"] = z["value"][:-1]

        def nan_threshold(z):
            z["threshold"] = z["threshold"].copy()
            z["threshold"][np.flatnonzero(z["left"] != -1)[0]] = np.nan

        def wide_classes(z):
            z["n_classes"] = np.array([9, 2], np.int64)

        def other_format(z):
            z["format"] = np.array("qfsk2")

        def pickled(z):
            z["estimator"] = np.array([{"a": 1}], dtype=object)

        def bad_feature(z):
            z["feature"] = z["feature"].copy()
            z["feature"][np.flatnonzero(z["left"] != -1)[0]] = 99

        def infinite_value(z):
            z["value"] = z["value"].copy()
            z["value"][np.flatnonzero(z["left"] == -1)[0], 0, 0] = np.inf

        def one_class(z):
            z["n_classes"] = np.array([1, 2], np.int64)

        def no_key(z):
            del z["threshold"]

        def two_parents(z):
            z["right"] = z["right"].copy()
            i = np.flatnonzero(z["left"] != -1)[0]
            z["right"][i] = z["left"][i]

        def fractions(z):
            z["value"] = z["value"].copy()
            z["value"][np.flatnonzero(z["left"] == -1)[0], 0, :3] += 0.3

        def one_child(z):
            z["right"] = z["right"].copy()
            z["right"][np.flatnonzero(z["left"] != -1)[0]] = -1

        def text_as_number(z):
            z["n_features"] = np.array("many")

        for change in (cut_offsets, late_start, child_first, short_value, nan_threshold, wide_classes, other_format, pickled, bad_feature,
                       infinite_value, one_class, no_key, two_parents, fractions, one_child, text_as_number):  # fmt: skip
            path = self.mutated(change.__name__, change)
            with self.assertRaises(ValueError, msg=change.__name__) as cm:
                qf_formats.open_source(path)
            text = str(cm.exception)
            self.assertNotIn("\n", text, change.__name__)
            self.assertIn(path.name, text, change.__name__)

    def test_a_file_that_is_not_an_archive_is_refused(self):
        for label, content in (("empty", b""), ("text", b"hello"), ("truncated", self.files["rf_reg_k2"].read_bytes()[:200])):
            p = self.t / f"junk-{label}.qfsk"
            p.write_bytes(content)
            with self.assertRaises(ValueError, msg=label) as cm:
                qf_formats.open_source(p)
            self.assertNotIn("\n", str(cm.exception), label)


if __name__ == "__main__":
    unittest.main()
