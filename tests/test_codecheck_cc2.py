"""code check 2 (QuickForest 1.2): the judge per output (decisions, argmax / threshold, rule files, check / compare). One regression per
finding, plus the guard that every way of deciding gives the same decisions for every model kind.

python3 -m pytest tests/test_codecheck_cc2.py

The engine tests need a library that reads format 7: build it with rust/build.sh and point TREEFLAT_LIB at it (QF_REQUIRE_FORMAT7=1 makes a stale
library fail instead of skip)."""

import sys
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backends
import compare
import evaluator
import prune
import qf_common as qf
import qf_formats
import qf_sklearn
from test_vector_flat import require_format7_library

MULTI = HERE / "fixtures" / "multi"

# a rule file for a model of two classifier outputs with class sets of 3 and 2: one integer per output
RULE_K2 = """import numpy as np

def decide(p):
    return np.stack([p[..., :3].argmax(-1), p[..., 3:].argmax(-1)], -1)
"""


def export_k2(folder: Path) -> tuple[Path, np.ndarray]:
    import joblib  # test-only: QF itself never reads a pickle

    est = joblib.load(MULTI / "rf_clf_k2_c3_2.joblib")
    x = np.load(MULTI / "rf_clf_k2_c3_2.npz")["x"].astype(np.float32)
    path = folder / "m.qfsk"
    qf_sklearn.export_sklearn(est, path, X=x[:500])
    return path, x


class CompareJudgesPerOutputTest(unittest.TestCase):
    """`compare.py` (and `choose.py`, which measures through it) built its decision rule without the model's segments: a rule file that
    returns one integer per output, accepted by check / prune / optimize, was refused there, and `argmax` was taken over ALL the columns
    of a model of several outputs (a column flipping inside one output's block was not a decision change)."""

    def test_a_rule_file_for_a_model_of_two_outputs_is_measured(self):
        require_format7_library()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            model, x = export_k2(d)
            (d / "rule.py").write_text(RULE_K2)
            np.savez(d / "data.npz", X=x)
            table = compare.run_table(
                model,
                ["flat"],
                d / "data.npz",
                d / "w",
                decision=str(d / "rule.py"),
                eq_rows=400,
                n_random=100,
                speed_rows=400,
                log=lambda *_: None,
            )
            row = table["rows"][0]
            self.assertEqual(row["status"], "ok", row)
            self.assertEqual(row["decision_mismatches"], 0)

    def test_the_rule_of_a_multi_output_forest_decides_per_output(self):
        forest = qf.load_forest([self.model_path()], "output")
        rule = qf.forest_rule("argmax", forest)
        p = np.array([[0.1, 0.7, 0.2, 0.6, 0.4], [0.1, 0.7, 0.2, 0.4, 0.6]], np.float32)
        self.assertEqual(rule(p).tolist(), [[1, 0], [1, 1]])

    def test_a_regression_forest_has_no_rule(self):
        import joblib

        est = joblib.load(MULTI / "rf_reg_k2.joblib")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "r.qfsk"
            qf_sklearn.export_sklearn(est, path, X=np.load(MULTI / "rf_reg_k2.npz")["x"][:300])
            self.assertIsNone(qf.forest_rule("argmax", qf.load_forest([path], "output")))

    def model_path(self) -> Path:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        return export_k2(Path(self._tmp.name))[0]


class ThresholdMessageTest(unittest.TestCase):
    """`threshold:T` is refused when the model has several outputs (multilabel, several classifiers) and when its one output has several class
    columns (a binary sklearn classifier: both columns); the two refusals must say what is true of each, not both «single output»."""

    def test_several_outputs_and_several_class_columns_are_told_apart(self):
        with self.assertRaises(ValueError) as outputs:
            qf.make_rule("threshold:0.4", 3, "sigmoid", None, None, "classification", [0, 1, 2, 3])
        with self.assertRaises(ValueError) as columns:
            qf.make_rule("threshold:0.4", 2, "identity", None, None, "classification", None)
        self.assertIn("3 outputs", str(outputs.exception))
        self.assertIn("2 class columns", str(columns.exception))
        self.assertNotIn("single output", str(columns.exception))

    def test_one_probability_column_still_takes_a_threshold(self):
        rule = qf.make_rule("threshold:0.4", 1, "sigmoid", None, None, "classification", None)
        self.assertEqual(rule(np.array([[0.3], [0.4], [0.9]], np.float32)).tolist(), [0, 1, 1])


class RuleFileCannotModifyItsInputTest(unittest.TestCase):
    """A rule file that writes into `p` changed the arrays the tools keep and judge again (the full forest's answer is reused for every
    candidate of the greedy, `agreement` reads `p` after the rule): the rule gets a read-only view and the write is a one-line refusal."""

    def rule_file(self, folder: Path, body: str) -> str:
        path = folder / "rule.py"
        path.write_text("import numpy as np\n\ndef decide(p):\n" + body)
        return str(path)

    def test_a_rule_that_writes_into_p_is_refused_and_p_is_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            text = self.rule_file(Path(d), "    p[p < 0.5] = 0.0\n    return (p[..., 0] > 0).astype(int)\n")
            with self.assertRaisesRegex(ValueError, "must not modify p"):
                qf.make_rule(text, 1, "sigmoid", None, None, "classification", None)

    def test_a_rule_that_copies_first_works(self):
        with tempfile.TemporaryDirectory() as d:
            text = self.rule_file(Path(d), "    q = p.copy()\n    q[q < 0.5] = 0.0\n    return (q[..., 0] > 0).astype(int)\n")
            rule = qf.make_rule(text, 1, "sigmoid", None, None, "classification", None)
            p = np.array([[0.2], [0.7]], np.float32)
            self.assertEqual(rule(p).tolist(), [0, 1])
            self.assertTrue(np.array_equal(p, np.array([[0.2], [0.7]], np.float32)))


class PooledAgreementTest(unittest.TestCase):
    """The folder figures `selection_pct` / `models_pct` pooled each model's WORST output: with K > 1 outputs that is not the pooled worst."""

    def test_the_pooled_figure_is_the_worst_output_over_all_the_rows(self):
        a = {"rows": 10, "same": 5, "same_per_output": [10, 5]}  # worst on output 1
        b = {"rows": 10, "same": 5, "same_per_output": [5, 10]}  # worst on output 0
        self.assertEqual(prune.pooled_pct([a, b]), 75.0)  # 15 of 20 on each output; the sum of the worsts says 10 of 20
        self.assertEqual(prune.pooled_pct([{"rows": 10, "same": 9}, {"rows": 10, "same": 7}]), 80.0)  # one decision per row: the plain sum
        self.assertEqual(prune.pooled_pct([]), 0.0)


class DecisionTableTest(unittest.TestCase):
    """DC §4/§5: ONE decision rule per model kind. For every kind the rule built from the IR (`forest_rule`), the compiled forest's own
    `argmax_outputs` and `segment_decisions` on the file's segments give the same integers, in the same shape."""

    CASES = (
        ("xgb_vec_multilabel_k2.ubj", 2),  # sigmoid, width-1 segments
        ("xgb_vec_softprob_c3.ubj", None),  # one softmax block
        ("ydf_rf_binary", None),  # one group (class 1): p >= 0.5
        ("rf_clf_k2_c3_2.joblib", 2),  # class sets 3 and 2
        ("rf_clf_k1_c4.joblib", None),  # one output, four classes
    )

    def test_every_way_of_deciding_agrees(self):
        require_format7_library()
        for name, outputs in self.CASES:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as d:
                d = Path(d)
                if name.startswith("ydf") and importlib.util.find_spec("ydf") is None:
                    continue  # reading a YDF model needs the ydf library (it has no wheel on every platform)
                if name.endswith(".joblib"):
                    import joblib

                    est = joblib.load(MULTI / name)
                    x = np.load(MULTI / name.replace(".joblib", ".npz"))["x"].astype(np.float32)
                    model = d / "m.qfsk"
                    qf_sklearn.export_sklearn(est, model, X=x[:500])
                else:
                    model, x = MULTI / name, None
                    data = MULTI / (name.split(".")[0] + ".npz")
                    if data.is_file():
                        x = np.load(data)["x"].astype(np.float32)
                forest = qf.load_forest(qf_formats.model_files(model), "output")
                if x is None:
                    x = np.random.default_rng(0).normal(size=(300, forest.n_features)).astype(np.float32)
                x = x[:300, : forest.n_features]
                art = d / "m.flat"
                backends.get("flat").compile(forest, art, {})
                flat = evaluator.FlatForest(art)
                try:
                    p = flat.predict(x).reshape(len(x), -1)
                    self.assertEqual(flat.n_outputs, outputs or 1)
                    by_rule = qf.forest_rule("argmax", forest)(p)
                    if name == "ydf_rf_binary":
                        # one identity group and no segments is also what a regression forest's file is (the 1.1 bytes must not change): the
                        # file cannot say «classifier», so the file's own argmax refuses; check / prune decide from the IR
                        self.assertEqual(flat.output_kind, "regression")
                        with self.assertRaises(ValueError):
                            flat.argmax_outputs(p)
                        by_file = (p >= 0.5).astype(np.int64)
                    else:
                        self.assertEqual(flat.output_kind, "classification")
                        by_file = flat.argmax_outputs(p)
                    by_ir = evaluator.segment_decisions(p, qf.forest_segments(forest) or [0, forest.groups])
                    shape = (len(x), outputs) if outputs else (len(x),)
                    self.assertEqual(by_rule.shape, shape)
                    self.assertTrue(np.array_equal(by_rule.reshape(len(x), -1), by_file.reshape(len(x), -1)))
                    self.assertTrue(np.array_equal(by_rule.reshape(len(x), -1), by_ir.reshape(len(x), -1)))
                    # 3-D answers (the greedy's candidates x rows x G) decide the same, as (c, rows, K)
                    three = qf.forest_rule("argmax", forest)(np.stack([p, p[::-1]]))
                    self.assertTrue(np.array_equal(three[0], by_rule))
                    self.assertTrue(np.array_equal(three[1], by_rule[::-1]))
                finally:
                    flat.close()


if __name__ == "__main__":
    unittest.main()
