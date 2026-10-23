"""`--keep auto` (the smallest rung of a ladder that passes the bar, else the whole forest) and the compile-only path beside it.

python3 -m unittest tests.test_keep_auto        (from the repository root, with a Python that has xgboost)
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import optimize
import prune
import qf_common as qf
import qf_formats
from test_pipeline import FOLDS, N, make_data

LADDER = ("5%", "10%", "25%", "50%")


class KeepAutoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.t = Path(cls.tmp.name)
        x, y, reg = make_data()
        cls.x = x
        fold = np.arange(N) % FOLDS
        rf = {
            "objective": "multi:softprob",
            "num_class": 3,
            "num_parallel_tree": 40,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "learning_rate": 1,
            "max_depth": 6,
            "reg_lambda": 1e-5,
            "tree_method": "hist",
            "nthread": 2,
        }
        (t / "rf").mkdir()
        for k in range(FOLDS):
            tr = fold != k
            xgb.train(rf, xgb.DMatrix(x[tr], y[tr]), 1).save_model(str(t / "rf" / f"fold{k}.ubj"))
        gb = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.2, "tree_method": "hist", "nthread": 2}
        xgb.train(gb, xgb.DMatrix(x[:4000], (y[:4000] == 0).astype(int)), 60).save_model(str(t / "gb.ubj"))
        xgb.train(
            {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.2, "tree_method": "hist", "nthread": 2},
            xgb.DMatrix(x[:4000], reg[:4000]),
            60,
        ).save_model(str(t / "reg.json"))
        np.savez(t / "rf.npz", X=x, fold=fold, group=np.arange(N) // FOLDS)
        np.savez(t / "all.npz", X=x, group=np.arange(N))
        np.savez(t / "reg.npz", X=x, y=reg, group=np.arange(N))
        np.savez(t / "rf-y.npz", X=x, y=y, fold=fold, group=np.arange(N) // FOLDS)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_keep(self, model, data, name, keep, **kw):
        return prune.run(self.t / model, self.t / data, self.t / name, keep, **kw)

    def test_the_first_k_picks_of_one_selection_are_the_selection_of_k(self):
        """The reason `--keep auto` runs the selection once: for greedy and orientation, on a mean-scaled forest and on a boosted one, the
        order made for K starts with exactly what a run for k < K selects."""
        x = self.x[:1500]
        for model, mean_scale in (("rf/fold0.ubj", True), ("gb.ubj", False)):
            src = qf_formats.open_source(self.t / model)
            spec = src.spec
            lm = src.leaf_matrix(x)
            margin = np.stack([spec.base[g] + lm[spec.tree_info == g].sum(0, dtype=np.float64) for g in range(spec.groups)], 1)
            p_full = qf.transform(margin, spec.kind)
            rule = qf.make_rule("argmax", spec.groups, spec.kind, p_full, None, spec.task)
            d_full = rule(p_full)
            big = prune.greedy_order(lm, spec, 12, rule, mean_scale, p_full, d_full)
            for k in (1, 2, 5, 9, 12):
                alone = prune.select_greedy(lm, spec, k, rule, mean_scale, p_full, d_full)
                for g in range(spec.groups):
                    self.assertEqual(alone[g].tolist(), sorted(int(i) for i in big[g][:k]), (model, "greedy", k, g))
            ranked = prune.orientation_order(lm, spec, 12, margin)
            for k in (1, 4, 12):
                alone = prune.select_orientation(lm, spec, k, margin)
                for g in range(spec.groups):
                    self.assertEqual(alone[g].tolist(), sorted(int(i) for i in ranked[g][:k]), (model, "orientation", k, g))

    def test_each_rung_of_auto_is_the_plain_run_of_that_share(self):
        """The rungs of one auto run hold the very trees and the very held-out figures that `--keep 5%` ... `--keep 50%` give alone."""
        auto = self.run_keep("rf", "rf.npz", "auto-rungs", "auto", min_agreement=100.5)
        a = auto["auto"]
        self.assertEqual(a["ladder"], list(LADDER))
        self.assertEqual([r["keep"] for r in a["rungs"]], list(LADDER))
        for rung in a["rungs"]:
            alone = self.run_keep("rf", "rf.npz", f"plain-{rung['keep'].strip('%')}", rung["keep"])
            self.assertEqual(rung["kept_per_group"], [m["kept_per_group"] for m in alone["models"]])
            self.assertEqual(rung["heldout_agreement_pct"], alone["heldout"]["pct"], rung["keep"])
            self.assertEqual(rung["kept_trees"], sum(m["kept_trees"] for m in alone["models"]))
            self.assertEqual(rung["nodes"], sum(m["nodes_after"] for m in alone["models"]))
        # and the models a rung would emit are those of the plain run: choose each rung by setting the bar just under its own figure
        for rung in a["rungs"]:
            alone = self.run_keep("rf", "rf.npz", f"plain2-{rung['keep'].strip('%')}", rung["keep"])
            chosen = self.run_keep(
                "rf", "rf.npz", f"chosen-{rung['keep'].strip('%')}", "auto", min_agreement=-1.0, keep_ladder=[rung["keep"]]
            )
            self.assertEqual(chosen["kept_indices"], alone["kept_indices"], rung["keep"])
            for m in alone["models"]:
                self.assertEqual(
                    (self.t / f"chosen-{rung['keep'].strip('%')}" / m["file"]).read_bytes(),
                    (self.t / f"plain2-{rung['keep'].strip('%')}" / m["file"]).read_bytes(),
                    (rung["keep"], m["file"]),
                )

    def test_the_chosen_rung_is_the_smallest_that_passes(self):
        plain = {lab: self.run_keep("gb.ubj", "all.npz", f"gb-{lab.strip('%')}", lab) for lab in LADDER}
        figures = [plain[lab]["heldout"]["pct"] for lab in LADDER]
        self.assertGreater(len(set(figures)), 1, figures)
        bars = [0.0, 50.0, figures[1] - 1e-9, figures[2], figures[3], 99.9999]
        for bar in bars:
            r = self.run_keep("gb.ubj", "all.npz", f"gb-auto-{bar}", "auto", min_agreement=bar)
            passing = [lab for lab in LADDER if prune.meets_agreement(plain[lab]["heldout"], bar)]
            c = r["auto"]["chosen"]
            if passing:
                self.assertEqual(c["keep"], passing[0], (bar, figures))
                self.assertFalse(c["whole_forest"])
                self.assertEqual(r["kept_indices"], plain[passing[0]]["kept_indices"], bar)
            else:
                self.assertTrue(c["whole_forest"], (bar, figures))
            self.assertEqual([g["pass"] for g in r["auto"]["rungs"]], [lab in passing for lab in LADDER], bar)
            self.assertFalse((self.t / f"gb-auto-{bar}" / "_rungs").exists())

    def test_a_model_that_passes_at_no_rung_ends_compiled_whole_and_exact(self):
        rep = optimize.optimize(
            self.t / "gb.ubj", self.t / "all.npz", self.t / "whole", "auto", True, True, min_agreement=100.5, backend="flat"
        )
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertEqual(rep["failures"], [])
        c = rep["stages"]["prune"]["auto"]["chosen"]
        self.assertTrue(c["whole_forest"])
        self.assertEqual(c["keep"], "100% (whole forest)")
        self.assertIn("whole forest", c["reason"])
        self.assertTrue(all(not g["pass"] for g in rep["stages"]["prune"]["auto"]["rungs"]))
        self.assertEqual(rep["stages"]["compile"]["trees"], 60)
        self.assertTrue(rep["stages"]["verify"]["ok"])
        self.assertEqual((self.t / "whole" / "pruned" / "gb.ubj").read_bytes(), (self.t / "gb.ubj").read_bytes())
        ki = rep["kept_indices"]["gb.ubj"]
        self.assertEqual(ki["flat"], list(range(ki["original_trees"])))
        self.assertEqual(rep["stages"]["prune"]["heldout_pct"], 100.0)
        text = (self.t / "whole" / "report.md").read_text()
        self.assertIn("keep auto", text)
        self.assertIn("chosen: 100% (whole forest)", text)
        self.assertEqual(json.loads((self.t / "whole" / "report.json").read_text())["stages"]["prune"]["auto"]["chosen"], c)

    def test_optimize_auto_emits_the_smallest_passing_rung_compiled_and_verified(self):
        rep = optimize.optimize(
            self.t / "rf", self.t / "rf.npz", self.t / "opt-auto", "auto", True, True, min_agreement=50.0, backend="flat"
        )
        self.assertTrue(rep["ok"], rep["failures"])
        a = rep["stages"]["prune"]["auto"]
        self.assertEqual(a["chosen"]["keep"], "5%")  # the smallest rung passes a bar this low
        self.assertEqual({m for m in a["chosen"]["kept_per_group"]}, {2})
        self.assertEqual(rep["keep"], "auto")
        self.assertTrue(rep["stages"]["verify"]["ok"])
        self.assertTrue((self.t / "opt-auto" / "pruned" / "fold0.ubj").is_file())
        self.assertTrue({"model.flat", "pruned", "report.json", "report.md"} <= {p.name for p in (self.t / "opt-auto").iterdir()})
        self.assertFalse(any(p.name.startswith(("_work", "_rungs")) for p in (self.t / "opt-auto").iterdir()))
        self.assertEqual(rep["kept_indices"]["fold0.ubj"]["original_trees"], 120)
        self.assertEqual(len(rep["kept_indices"]["fold0.ubj"]["per_group"][0]), 2)
        self.assertEqual(len(a["rungs"]), 4)
        self.assertIn("keep auto: rungs tried", (self.t / "opt-auto" / "report.md").read_text())

    def test_a_ladder_of_your_own_is_sorted_and_collapses_equal_rungs(self):
        r = self.run_keep("gb.ubj", "all.npz", "ladder-own", "auto", min_agreement=0.0, keep_ladder="50%,10%,20")
        self.assertEqual(r["auto"]["ladder"], ["50%", "10%", "20"])
        self.assertEqual([g["keep"] for g in r["auto"]["rungs"]], ["10%", "20", "50%"])
        self.assertEqual(r["auto"]["chosen"]["keep"], "10%")
        self.assertEqual({m["kept_per_group"] for m in r["models"]}, {6})
        # two entries that round to the same trees are one rung, and the report says so
        r = self.run_keep("gb.ubj", "all.npz", "ladder-twice", "auto", min_agreement=100.5, keep_ladder=["10%", "6", "50%"])
        self.assertEqual([(g["keep"], g["also"]) for g in r["auto"]["rungs"]], [("10%", ["6"]), ("50%", [])])

    def test_regression_auto_uses_the_regression_bars(self):
        loose = qf.RegBar("raw", 1e9, 1e9, 1e9)
        r = self.run_keep("reg.json", "reg.npz", "reg-loose", "auto", regression=loose)
        self.assertEqual(r["auto"]["chosen"]["keep"], "5%")
        self.assertTrue(r["auto"]["rungs"][0]["regression"]["ok"])
        tight = qf.RegBar("raw", 1e-9, 1e-9, 0.0)
        r = self.run_keep("reg.json", "reg.npz", "reg-tight", "auto", regression=tight)
        self.assertTrue(r["auto"]["chosen"]["whole_forest"])
        self.assertTrue(all(not g["pass"] for g in r["auto"]["rungs"]))
        self.assertEqual(r["regression"]["max_pct"], 0.0)
        rep = optimize.optimize(
            self.t / "reg.json", self.t / "reg.npz", self.t / "reg-opt", "auto", True, True, regression=tight, backend="flat"
        )
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertEqual(rep["stages"]["compile"]["trees"], 60)

    def test_a_bad_ladder_or_a_ladder_without_auto_is_refused_before_any_work(self):
        for bad in ("", ",", "5%,", "0", "-3", "0%", "nan", "abc", "100%", "150%", "inf", "5%,5%", "0.1", "5%, ,10%", "1e999"):
            with self.assertRaises(SystemExit, msg=bad):
                self.run_keep("gb.ubj", "all.npz", "ladder-bad", "auto", keep_ladder=bad)
            with self.assertRaises(SystemExit, msg=bad):
                optimize.optimize(self.t / "gb.ubj", self.t / "all.npz", self.t / "ladder-bad-opt", "auto", keep_ladder=bad)
        for keep in ("10%", "30", "100%"):
            with self.assertRaises(SystemExit, msg=keep):
                self.run_keep("gb.ubj", "all.npz", "ladder-bad", keep, keep_ladder="5%")
        self.assertFalse((self.t / "ladder-bad").exists() or (self.t / "ladder-bad-opt").exists())

    def test_a_folder_of_the_user_called_rungs_is_never_deleted(self):
        out = self.t / "has-rungs"
        (out / "_rungs").mkdir(parents=True)
        (out / "_rungs" / "mine.txt").write_text("keep me")
        with self.assertRaises(SystemExit):
            self.run_keep("gb.ubj", "all.npz", "has-rungs", "auto")
        self.run_keep("gb.ubj", "all.npz", "has-rungs", "10%")  # a fixed size never touches that name
        self.assertEqual((out / "_rungs" / "mine.txt").read_text(), "keep me")

    def test_compile_only_and_a_chosen_share_are_two_paths_to_the_same_answers(self):
        """The two first-class paths: `--keep 100%` compiles the whole forest and its answers are the library's; `--keep 50%` prunes."""
        whole = optimize.optimize(self.t / "gb.ubj", self.t / "all.npz", self.t / "path-whole", "100%", True, True, backend="flat")
        self.assertTrue(whole["ok"], whole["failures"])
        self.assertEqual(whole["stages"]["compile"]["trees"], 60)
        self.assertTrue(whole["stages"]["verify"]["ok"])
        half = optimize.optimize(
            self.t / "gb.ubj", self.t / "all.npz", self.t / "path-half", "50%", True, True, min_agreement=50.0, backend="flat"
        )
        self.assertTrue(half["ok"], half["failures"])
        self.assertEqual(half["stages"]["compile"]["trees"], 30)

    def test_best_n_by_count_through_the_python_api_and_accept_any(self):
        """`keep=10` (an int, a count per class) in the API, a bar the set cannot meet, and `accept_any` to keep it anyway."""
        r = prune.run(self.t / "rf", self.t / "rf-y.npz", self.t / "best10", 10)
        self.assertEqual({m["kept_per_group"] for m in r["models"]}, {10})
        self.assertEqual({len(g) for m in r["kept_indices"].values() for g in m["per_group"]}, {10})
        acc = r["heldout"]["accuracy"]  # the data holds the truth y: how often each forest hits it
        self.assertEqual(acc["rows"], r["heldout"]["rows"])
        self.assertTrue(0 < acc["pruned_pct"] <= 100 and 0 < acc["full_pct"] <= 100)
        refused = optimize.optimize(self.t / "rf", self.t / "rf-y.npz", self.t / "best10-refused", 10, min_agreement=100.5)
        self.assertFalse(refused["ok"])
        self.assertEqual(sorted(p.name for p in (self.t / "best10-refused").iterdir()), ["report.json", "report.md"])
        kept = optimize.optimize(
            self.t / "rf", self.t / "rf-y.npz", self.t / "best10-kept", 10, True, True, min_agreement=100.5, accept_any=True, backend="flat"
        )
        self.assertTrue(kept["ok"], kept["failures"])
        self.assertEqual(kept["failures"], [])
        self.assertIn("held-out agreement", kept["accept_any"]["waived"][0])
        self.assertEqual(kept["kept_indices"], r["kept_indices"])
        self.assertEqual(kept["stages"]["compile"]["trees"], 3 * 10 * 3)  # 3 folds x 3 classes x 10 trees
        self.assertTrue(kept["stages"]["verify"]["ok"])
        text = (self.t / "best10-kept" / "report.md").read_text()
        self.assertIn("ACCEPTED WITHOUT THE BAR", text)
        self.assertIn("held-out accuracy against the truth y", text)
        passing = optimize.optimize(self.t / "rf", self.t / "rf-y.npz", self.t / "best10-passes", 10, min_agreement=0.0, accept_any=True)
        self.assertEqual(passing["accept_any"]["waived"], [])
        self.assertIn("the bar would have passed", (self.t / "best10-passes" / "report.md").read_text())
        with self.assertRaises(ValueError):  # the size chosen by the bar and the size taken without it exclude each other
            optimize.optimize(self.t / "rf", self.t / "rf-y.npz", self.t / "auto-accept", "auto", accept_any=True)
        self.assertFalse((self.t / "auto-accept").exists())

    def test_accept_any_waives_the_regression_bars_too(self):
        tight = qf.RegBar("raw", 1e-9, 1e-9, 0.0)
        no = optimize.optimize(self.t / "reg.json", self.t / "reg.npz", self.t / "reg-no", 5, regression=tight)
        self.assertFalse(no["ok"])
        yes = optimize.optimize(self.t / "reg.json", self.t / "reg.npz", self.t / "reg-yes", 5, regression=tight, accept_any=True)
        self.assertTrue(yes["ok"])
        self.assertEqual(len(yes["accept_any"]["waived"]), len(no["failures"]))


if __name__ == "__main__":
    unittest.main()
