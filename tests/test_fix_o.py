"""Fix O (QuickForest 1.2): the two bugs code check 2 found outside its share.

O1: xgboost `one_output_per_tree` with a sigmoid objective and several targets (multilabel) was read, compiled, and then died in the loader
    with the bare "counts out of range". The reader refuses it, in one line, naming the strategy that works.
O2: a one-tree-per-output identity forest (xgboost `reg:squarederror`, several targets) got no output segment table, so the flat file said ONE
    classification output where the generated code says K regression outputs. The flat file now carries the table (format 7 without the
    vector flag; the library accepts it).

python3 -m pytest tests/test_fix_o.py

Needs a library that reads the scalar segment table (rust/build.sh, then TREEFLAT_LIB=<bin>/libtreeflat-x86_64.so QF_REQUIRE_FORMAT7=1) for the
engine tests; the xgboost ones also need `xgboost`."""

import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_codegen as cg
import backend_flat
import backends
import evaluator
import qf_common as qf
from vector_forests import explode, make_vector_forest

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


def lib_reads_scalar_segments() -> bool:
    """True when the library at hand accepts a format-7 file without vector leaves (the one this fix adds)."""
    try:
        lib = evaluator.load_lib(evaluator.find_lib())
    except OSError:
        return False
    if not hasattr(lib, "tfl_segments"):
        return False
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.flat"
        try:
            blob = backend_flat.pack(
                explode(make_vector_forest(1, width=3, trees=1, kind="identity")), lambda b: evaluator.fnv1a(b, evaluator.find_lib())
            )
            out.write_bytes(blob)
            evaluator.FlatForest(out).close()
        except (ValueError, OSError):
            return False
    return True


def train(objective: str, strategy: str, targets: int, rounds: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(600, 5)).astype(np.float32)
    y = (
        (X[:, :1] + 0.5 * rng.normal(size=(600, targets)) > 0).astype(np.float32)
        if "logistic" in objective
        else X[:, :1] * 2 + rng.normal(size=(600, targets))
    )
    params = dict(objective=objective, tree_method="hist", multi_strategy=strategy, max_depth=3, nthread=1)
    return X, xgb.train(params, xgb.DMatrix(X, label=y), rounds)


@unittest.skipIf(xgb is None, "xgboost is not installed")
class MultilabelRefusalTest(unittest.TestCase):
    """O1: the reader refuses the unsupported combination, once, with one clear line."""

    def load(self, objective, strategy, targets):
        _, booster = train(objective, strategy, targets)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.ubj"
            booster.save_model(path)
            return qf.load_forest([path], "output")

    def test_one_output_per_tree_multilabel_is_refused_at_read(self):
        for objective, targets in itertools.product(("binary:logistic", "reg:logistic"), (2, 3)):
            with self.subTest(objective=objective, targets=targets):
                with self.assertRaises(ValueError) as cm:
                    self.load(objective, "one_output_per_tree", targets)
                self.assertIn("multi_strategy=multi_output_tree", str(cm.exception))
                self.assertNotIn("counts out of range", str(cm.exception))

    def test_what_works_still_reads(self):
        self.assertEqual(self.load("binary:logistic", "multi_output_tree", 3).groups, 3)  # vector trees: one sigmoid per column
        self.assertEqual(self.load("binary:logistic", "one_output_per_tree", 1).groups, 1)  # a plain binary model
        self.assertEqual(self.load("reg:squarederror", "one_output_per_tree", 3).groups, 3)


class ScalarSigmoidPackTest(unittest.TestCase):
    """O1, second line: a scalar forest with a sigmoid over several groups that reaches the writer by another road is refused there, not by the loader."""

    def test_the_writer_refuses_it_in_one_line(self):
        forest = explode(make_vector_forest(2, width=3, trees=2, kind="sigmoid"))
        with self.assertRaises(ValueError) as cm:
            backend_flat.pack(forest, lambda b: 0)
        self.assertIn("multi_strategy=multi_output_tree", str(cm.exception))


class ScalarSegmentsRuleTest(unittest.TestCase):
    """O2: the flat writer and the generated code write the same segment table for scalar forests too (DC 3.1: K regression columns)."""

    def test_flat_and_codegen_agree_on_scalar_forests(self):
        for kind, task, groups in itertools.product(("identity", "softmax", "exp"), ("auto", "regression", "classification"), (1, 2, 5)):
            if groups == 1 and kind == "softmax":
                continue
            vec = make_vector_forest(1, width=max(groups, 2), trees=1, kind=kind)
            forest = explode(vec) if groups > 1 else vec
            forest.task = task
            if groups == 1:
                continue
            with self.subTest(kind=kind, task=task, groups=groups):
                self.assertEqual(backend_flat.scalar_segments(forest) or [], cg.file_segments(forest))

    def test_the_regression_rule_and_the_classification_exception(self):
        reg = explode(make_vector_forest(1, width=5, trees=1, kind="identity"))
        reg.task = "regression"
        self.assertEqual(backend_flat.scalar_segments(reg), [0, 1, 2, 3, 4, 5])
        shares = explode(make_vector_forest(1, width=5, trees=1, kind="identity"))
        shares.task = "classification"  # random-forest class shares: ONE output of 5 columns
        self.assertIsNone(backend_flat.scalar_segments(shares))

    def test_a_margin_scale_or_exact_leaves_keep_the_old_layout(self):
        # format 7 has no margin-scale field and format 6 (linear / f64 records) no segment table: no table, as before the fix
        forest = explode(make_vector_forest(1, width=3, trees=1, kind="identity"))
        forest.task = "regression"
        forest.models[0].margin_scale = 0.5
        self.assertIsNone(backend_flat.scalar_segments(forest))
        exact = explode(make_vector_forest(1, width=3, trees=1, kind="identity", table64=True))
        exact.task = "regression"
        blob = backend_flat.pack(exact, lambda b: 0, exact=True)
        self.assertEqual(int.from_bytes(blob[8:12], "little"), 6)


@unittest.skipUnless(lib_reads_scalar_segments(), "the library at hand does not read a scalar segment table: build it with rust/build.sh")
class ScalarSegmentsEngineTest(unittest.TestCase):
    """O2 end to end: the file says K regression outputs, and the answers are the exploded twin's."""

    def flat(self, forest, folder):
        out = Path(folder) / "m.flat"
        backends.get("flat").compile(forest, out, {})
        return evaluator.FlatForest(out)

    def test_a_scalar_identity_forest_is_k_regression_outputs(self):
        forest = explode(make_vector_forest(3, width=4, trees=5, kind="identity"))
        forest.task = "regression"
        X = np.random.default_rng(0).normal(size=(50, forest.n_features)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            ff = self.flat(forest, d)
            self.assertEqual((ff.n_outputs, ff.output_kind, ff.segments), (4, "regression", [0, 1, 2, 3, 4]))
            self.assertEqual(ff.segments, cg.file_segments(forest))
            p = ff.predict(X)
            self.assertEqual(p.shape, (50, 4))
            self.assertEqual(len(ff.split_outputs(p)), 4)
            ff.close()

    def test_class_shares_stay_one_classification_output(self):
        forest = explode(make_vector_forest(3, width=4, trees=5, kind="identity"))
        forest.task = "classification"
        with tempfile.TemporaryDirectory() as d:
            ff = self.flat(forest, d)
            self.assertEqual((ff.n_outputs, ff.output_kind, ff.segments), (1, "classification", None))
            ff.close()

    @unittest.skipIf(xgb is None, "xgboost is not installed")
    def test_xgboost_multi_target_regression_flat_and_codegen_agree(self):
        X, booster = train("reg:squarederror", "one_output_per_tree", 5)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.ubj"
            booster.save_model(path)
            forest = qf.load_forest([path], "output")
            ff = self.flat(forest, d)
            self.assertEqual((ff.n_outputs, ff.output_kind), (5, "regression"))
            self.assertEqual(ff.segments, cg.file_segments(forest))
            want = booster.predict(xgb.DMatrix(X[:100]))
            np.testing.assert_allclose(ff.predict(X[:100]).reshape(100, -1), want, rtol=1e-5, atol=1e-5)
            ff.close()


if __name__ == "__main__":
    unittest.main()
