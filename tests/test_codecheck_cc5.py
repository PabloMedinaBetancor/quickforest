"""code check 5 (QF 1.2): pruning per output, the regression bar and the report shape. numpy only; runs in a few seconds:

python3 -s -m pytest tests/test_codecheck_cc5.py
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import optimize
import prune
import qf_common as qf
from test_prune_multi import FakeFormats, classification_source, data_file, regression_source


def scalar_spec(groups, kind, n_trees, task):
    """A scalar forest of `groups` outputs, trees dealt to the groups in turn."""
    return qf.Spec(
        3,
        groups,
        kind,
        np.zeros(groups, np.float32),
        np.arange(n_trees) % groups,
        np.full(n_trees, 3),
        np.zeros(n_trees, np.int64),
        np.zeros(0, np.float32),
        1,
        task=task,
        mean_type=False,
    )


class ScalarSeveralOutputsGreedyTest(unittest.TestCase):
    """xgboost `multi_strategy=one_output_per_tree`: K outputs, scalar trees, tree_info = the output. Output g's greedy step must be
    scored against output g of the full model, not against all K columns of it."""

    def setUp(self):
        rng = np.random.default_rng(5)
        self.n, self.per = 400, 12
        scale = np.array([1.0, 40.0])
        self.L = np.stack([rng.normal(size=self.n) * 0.2 * scale[i % 2] + 0.3 * scale[i % 2] for i in range(2 * self.per)]).astype(
            np.float32
        )
        self.spec = scalar_spec(2, "identity", 2 * self.per, "regression")

    def test_identity_outputs_pick_what_a_model_of_that_output_alone_picks(self):
        spec = self.spec
        p_full = np.stack([self.L[spec.tree_info == g].sum(0) for g in range(2)], 1).astype(np.float32)
        rule = qf.make_rule("argmax", 2, "identity", p_full, None, "regression")
        both = prune.greedy_order(self.L, spec, 4, rule, False, p_full, rule(p_full), 1 / np.maximum(np.abs(p_full), 1e-9))
        for g in range(2):
            ix = np.flatnonzero(spec.tree_info == g)
            alone_spec = scalar_spec(1, "identity", len(ix), "regression")
            pf = p_full[:, g : g + 1]
            r1 = qf.make_rule("argmax", 1, "identity", pf, None, "regression")
            alone = prune.greedy_order(self.L[ix], alone_spec, 4, r1, False, pf, r1(pf), 1 / np.maximum(np.abs(pf), 1e-9))[0]
            self.assertEqual(sorted(both[g]), sorted(ix[alone].tolist()), f"output {g}")

    def test_sigmoid_outputs_score_the_candidate_beside_the_other_outputs(self):
        """First pick of output 0, by brute force: the candidate's probability in column 0, the other columns at their base, and the
        decision the rule takes over BOTH columns."""
        spec = scalar_spec(2, "sigmoid", 2 * self.per, "classification")
        L = self.L * 0.05
        p_full = qf.transform(np.stack([L[spec.tree_info == g].sum(0) for g in range(2)], 1), "sigmoid")
        rule = qf.make_rule("argmax", 2, "sigmoid", p_full, None, "classification")
        d_full = rule(p_full)
        best, best_score = None, -np.inf
        for t in np.flatnonzero(spec.tree_info == 0):
            p = qf.transform(np.stack([L[t], np.zeros(self.n, np.float32)], 1), "sigmoid")
            mse = ((p - p_full) ** 2).mean()
            score = (rule(p) == d_full).sum() - 0.5 * min(mse, 1.0)
            if score > best_score:
                best, best_score = int(t), score
        got = prune.greedy_order(L, spec, 1, rule, False, p_full, d_full)
        self.assertEqual(int(got[0][0]), best)


class RegBarTopLevelMaeTest(unittest.TestCase):
    def test_the_top_level_mae_is_the_one_that_fails(self):
        n = 100
        a = np.full((n, 2), 10.0)
        b = a.copy()
        b[0, 0] *= 1.015  # output 0: the largest relative difference, inside the bars, and exact against y
        b[:, 1] *= 1.0005  # output 1: tiny differences, but its MAE against the truth gets worse
        y = np.stack([b[:, 0], a[:, 1]], 1)
        res = qf.RegBar("raw", 0.5, 2.0, 0.0).check(a, b, y)
        self.assertFalse(res["ok"])
        self.assertTrue(all(m.startswith("output 1") for m in res["failures"]), res["failures"])
        self.assertGreater(res["mae_pruned"], res["mae_full"], "the report must show the figures of the output that failed")
        self.assertEqual((res["mae_full"], res["mae_pruned"]), (res["per_output"][1]["mae_full"], res["per_output"][1]["mae_pruned"]))

    def test_k1_keeps_the_figures_of_its_only_output(self):
        a, b = np.full(50, 4.0), np.full(50, 4.001)
        res = qf.RegBar("raw").check(a, b, np.full(50, 4.0))
        self.assertEqual((res["mae_full"], res["mae_pruned"]), (0.0, 0.001))
        self.assertNotIn("per_output", res)


class AutoLinesNameTheOutputTest(unittest.TestCase):
    def test_classification_rung_that_fails_says_which_output(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = classification_source(d, [0, 3, 5], trees=40)
            with FakeFormats():
                rep = prune.run(src, data_file(d), d / "out", "auto", min_agreement=99.9, keep_ladder=["5%", "10%"])
            failed = [g for g in rep["auto"]["rungs"] if not g["pass"]]
            self.assertTrue(failed, "the bar is set so that a small rung fails")
            text = "\n".join(prune.auto_lines(rep))
            for g in failed:
                low = [j for j, v in enumerate(g["agreement_pct_per_output"]) if v < 99.9]
                line = next(x for x in text.splitlines() if x.lstrip().startswith(g["keep"] + ":"))
                self.assertIn("below the bar: " + ", ".join(f"output {j}" for j in low), line)

    def test_regression_rung_that_fails_says_which_output(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = regression_source(d)
            with FakeFormats():
                rep = prune.run(
                    src,
                    data_file(d),
                    d / "out",
                    "auto",
                    regression=qf.RegBar("raw", median_pct=1e-6, max_pct=1e-6),
                    keep_ladder=["10%", "25%"],
                )
            text = "\n".join(prune.auto_lines(rep))
            self.assertIn("below the bar: output 0", text)

    def test_one_output_lines_are_the_old_ones(self):
        r = {
            "auto": {
                "bar": {"min_agreement_pct": 95.0},
                "rungs": [
                    {
                        "keep": "5%",
                        "kept_per_group": [3],
                        "kept_trees": 3,
                        "nodes": 30,
                        "heldout_rows": 100,
                        "heldout_agreement_pct": 90.0,
                        "pass": False,
                    }
                ],
                "chosen": {"keep": "100% (whole forest)", "reason": "r"},
            }
        }
        self.assertEqual(prune.auto_lines(r)[1], "     5%: [3] per group, 3 trees, 30 nodes, held-out agreement 90.0 % on 100 rows: fail")


class OptimizeKeepAllChecksTheTruthWithTheBarTest(unittest.TestCase):
    def test_wrong_width_y_is_refused_with_the_regression_bar_at_100_percent(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = regression_source(d)  # five outputs
            data = data_file(d, y=np.zeros((600, 2)))
            with FakeFormats(), self.assertRaisesRegex(ValueError, "y has 2 column"):
                optimize.optimize(src, data, d / "opt", "100%", regression=qf.RegBar("raw"))

    def test_without_the_bar_a_k_target_regression_does_not_read_y(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = regression_source(d)
            data = data_file(d, y=np.zeros((600, 2)))
            with FakeFormats():
                rep = optimize.optimize(src, data, d / "opt", "100%")
            self.assertEqual(rep["stages"]["prune"], "skipped (keep 100%)")


if __name__ == "__main__":
    unittest.main()
