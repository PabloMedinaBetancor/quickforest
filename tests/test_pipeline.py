"""End-to-end tests on synthetic data: a multiclass xgb_rf (3 folds) and a binary gbtree are pruned to 10 %, compiled and
checked against xgboost; plus regression, the one-command pipeline and the pluggable backend. Needs numpy and xgboost.

    python3 -m unittest discover -s tests        (from the repository root, with a Python that has xgboost)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xgboost as xgb

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import backend_codegen
import backends
import check
import compile
import evaluator
import optimize
import prune
import qf_common as qf
import qf_formats
from tiled import tiled

N, NF, FOLDS = 6000, 12, 3


def make_data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(N, NF)).astype(np.float32)
    s = np.stack([x[:, 0] + x[:, 1] * x[:, 2], np.sin(2 * x[:, 3]) - x[:, 4], x[:, 5] ** 2 - x[:, 6]], 1) + rng.normal(
        scale=0.6, size=(N, 3)
    )
    y = s.argmax(1)
    reg = (x[:, 0] * 3 + np.sin(x[:, 1] * 2) + x[:, 2] * x[:, 3] + rng.normal(scale=0.3, size=N)).astype(np.float32)
    x[rng.random((N, NF)) < 0.05] = np.nan
    return x, y, reg


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = Path(cls.tmp.name)
        x, y, reg = make_data()
        cls.fold = np.arange(N) % FOLDS
        # (A) multiclass random forest, 3 folds: fold k's model trains on the other folds, so fold k's rows are its out-of-fold rows
        rf = {
            "objective": "multi:softprob",
            "num_class": 3,
            "num_parallel_tree": 100,
            "subsample": 0.8,
            "colsample_bynode": 0.8,
            "learning_rate": 1,
            "max_depth": 8,
            "reg_lambda": 1e-5,
            "tree_method": "hist",
            "nthread": 2,
        }
        (t / "rf").mkdir()
        for k in range(FOLDS):
            tr = cls.fold != k
            xgb.train(rf, xgb.DMatrix(x[tr], y[tr]), 1).save_model(str(t / "rf" / f"fold{k}.ubj"))
        # (B) binary gbtree
        yb = (y == 0).astype(int)
        gb = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.2, "tree_method": "hist", "nthread": 2}
        xgb.train(gb, xgb.DMatrix(x[:4000], yb[:4000]), 80).save_model(str(t / "gb.ubj"))
        # (C) regression gbtree
        xgb.train(
            {"objective": "reg:squarederror", "max_depth": 4, "eta": 0.2, "tree_method": "hist", "nthread": 2},
            xgb.DMatrix(x[:4000], reg[:4000]),
            60,
        ).save_model(str(t / "reg.json"))
        np.savez(t / "rf.npz", X=x, fold=cls.fold, group=np.arange(N) // FOLDS)
        np.savez(t / "all.npz", X=x, group=np.arange(N))
        cls.t = t

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def prune_compile_check(self, model, data, name, keep="10%", min_pct=0.0, min_ratio=4):
        out = self.t / f"pruned-{name}"
        r = prune.run(model, data, out, keep)
        self.assertTrue((out / "report.json").is_file())
        self.assertEqual(json.loads((out / "report.json").read_text())["totals"], r["totals"])
        self.assertGreater(r["heldout"]["rows"], 500)
        self.assertGreaterEqual(r["heldout"]["pct"], min_pct, prune.summary(r))
        self.assertGreater(r["totals"]["node_ratio"], min_ratio)
        for m in r["models"]:  # the pruned file is a normal model of the same format and name
            self.assertTrue((out / m["file"]).is_file())
        flat = self.t / f"{name}.flat"
        compile.compile_model(out, flat, backend="flat")
        for source, tag in ((out, "pruned"), (model, "full")):
            if tag == "full":
                compile.compile_model(source, flat, backend="flat")
            res = check.run(source, flat, data=data, n_random=3000)
            self.assertTrue(res["ok"], (name, tag, res["sets"]))
        return r

    def test_multiclass_rf_pruned_to_10_percent(self):
        r = self.prune_compile_check(self.t / "rf", self.t / "rf.npz", "rf", min_pct=90)
        self.assertEqual(r["heldout"]["mode"], "out-of-fold")
        self.assertEqual({m["kept_per_group"] for m in r["models"]}, {10})
        o = prune.run(self.t / "rf", self.t / "rf.npz", self.t / "pruned-rf-orient", "10%", method="orientation")
        self.assertGreater(o["heldout"]["pct"], 80)  # the fallback ordering also works (it is not guaranteed to be worse)

    def test_binary_gbtree_pruned_to_10_percent(self):
        self.prune_compile_check(self.t / "gb.ubj", self.t / "all.npz", "gb", min_pct=80)

    def test_regression_gbtree(self):
        r = self.prune_compile_check(self.t / "reg.json", self.t / "all.npz", "reg", keep="25%", min_ratio=3)
        self.assertGreater(r["heldout"]["pct"], 20)  # regression agreement = same 0.1-sd bin

    def test_custom_threshold_rule_and_margin_average(self):
        out = self.t / "pruned-thr"
        prune.run(self.t / "gb.ubj", self.t / "all.npz", out, "20%", decision="threshold:0.3")
        flat = self.t / "thr.flat"
        compile.compile_model(self.t / "rf", flat, average="margin", backend="flat")
        self.assertTrue(check.run(self.t / "rf", flat, average="margin", n_random=2000)["ok"])

    def test_pipeline_emits_or_refuses(self):
        ok = optimize.optimize(self.t / "rf", self.t / "rf.npz", self.t / "opt-ok", "10%", True, True, min_agreement=90.0, backend="flat")
        self.assertTrue(ok["ok"])
        self.assertTrue((self.t / "opt-ok" / "model.flat").is_file() and (self.t / "opt-ok" / "pruned" / "fold0.ubj").is_file())
        self.assertTrue(ok["stages"]["verify"]["ok"])
        refused = optimize.optimize(
            self.t / "rf", self.t / "rf.npz", self.t / "opt-no", "10%", True, True, min_agreement=100.5, backend="flat"
        )
        self.assertFalse(refused["ok"])
        self.assertEqual(sorted(p.name for p in (self.t / "opt-no").iterdir()), ["report.json", "report.md"])

    def test_the_float32_transform_is_the_float32_formula_and_the_double_one_keeps_double(self):
        rng = np.random.default_rng(3)
        m = rng.normal(scale=4, size=(500, 3)).astype(np.float32)
        e = np.exp(m - m.max(-1, keepdims=True))
        want = {
            "softmax": e / e.sum(-1, keepdims=True),
            "sigmoid": np.float32(1) / (np.float32(1) + np.exp(-m)),
            "exp": np.exp(m),
            "identity": m,
        }
        for kind, expected in want.items():
            got = qf.transform(m, kind)
            self.assertEqual(got.dtype, np.float32, kind)
            self.assertTrue(np.array_equal(got, expected), kind)
            self.assertEqual(qf.transform_f64(m.astype(np.float64), kind).dtype, np.float64, kind)

    def test_long_option_lists_are_keyword_only(self):
        """A swapped positional argument fails silently, so the options after the first few are keyword-only."""
        import inspect

        import compare

        for fn, positional in ((optimize.optimize, 6), (prune.run, 4), (check.run, 4), (compare.run_table, 4), (compare.measure_all, 3)):
            params = list(inspect.signature(fn).parameters.values())
            self.assertTrue(all(p.kind is p.POSITIONAL_OR_KEYWORD for p in params[:positional]), fn.__name__)
            self.assertTrue(all(p.kind is p.KEYWORD_ONLY for p in params[positional:]), fn.__name__)

    def test_the_library_copy_helper_writes_a_new_file_only_when_the_bytes_differ(self):
        src, dest = self.t / "lib-src.bin", self.t / "lib-dest.bin"
        src.write_bytes(b"one")
        qf.copy_unless_same(src, dest)
        first = dest.stat().st_ino
        qf.copy_unless_same(src, dest)
        self.assertEqual(dest.stat().st_ino, first)  # identical bytes: left alone
        src.write_bytes(b"two")
        qf.copy_unless_same(src, dest)
        self.assertEqual(dest.read_bytes(), b"two")

    def test_a_failed_verify_names_the_source_library_not_xgboost(self):
        """The compiled forest is judged against the model's own library, whichever it is: the refusal and the speed line say so, and
        the speed figure of every library is stored under `library_rows_per_s`."""
        failing = {"ok": False, "sets": [{"set": "random", "rows": 10, "ok": False, "max_err": 1e-3, "decision_mismatches": 2}]}
        with mock.patch.object(check, "run", return_value=failing):
            rep = optimize.optimize(
                self.t / "rf", self.t / "rf.npz", self.t / "opt-verify", "10%", True, True, min_agreement=90.0, backend="flat"
            )
        self.assertFalse(rep["ok"])
        self.assertEqual(len(rep["failures"]), 1)
        self.assertTrue(rep["failures"][0].startswith("compiled forest differs from the source library: "), rep["failures"])
        self.assertNotIn("xgboost", rep["failures"][0])
        speed = {"rows_from": "x", "library": "lightgbm", "library_rows_per_s": 7, "compiled_rows_per_s": 70, "speedup": 10.0}
        speed.update(load_seconds=0.1, extra_rss_mb=1.0)
        stages = {"verify": {"ok": True, "sets": [], "speed": speed}}
        report = {"ok": True, "model": "m", "keep": "10%", "seconds": 1, "failures": [], "compile_unsupported": None, "stages": stages}
        text = optimize.report_md(report, self.t)
        self.assertIn("lightgbm 7 rows/s", text)

    def test_a_refused_run_still_lists_the_rows_where_the_pruned_forest_disagrees(self):
        """The report lists the rows where the pruned model disagrees (docs/REFERENCE.md), also for a refused run: that run deletes
        pruned/report.json, where prune keeps the list, so optimize's own report has to carry it."""
        refused = optimize.optimize(self.t / "rf", self.t / "rf.npz", self.t / "opt-cases", "10%", min_agreement=100.5)
        self.assertFalse(refused["ok"])
        cases = refused["stages"]["prune"]["cases"]
        self.assertEqual(len(cases), min(50, refused["stages"]["prune"]["disagreements"]))
        self.assertGreater(len(cases), 0)
        self.assertEqual(sorted(p.name for p in (self.t / "opt-cases").iterdir()), ["report.json", "report.md"])
        md = (self.t / "opt-cases" / "report.md").read_text()
        self.assertIn("disagreeing held-out rows", md)
        self.assertIn(f"row {cases[0]['row']}:", md)

    def test_a_selection_share_or_row_cap_out_of_range_is_refused(self):
        """`prune --sub 0`, `--sub -1`, `--sub 1.5` and `--max-sel-rows 0` are refused with a message: no silent single selection row, no
        ZeroDivisionError."""
        for kw in ({"sub": 0}, {"sub": -1.0}, {"sub": 1.5}, {"max_sel_rows": 0}, {"eval_frac": 1.5}, {"tol": -1.0}):
            with self.assertRaises(SystemExit, msg=str(kw)):
                prune.run(self.t / "gb.ubj", self.t / "all.npz", self.t / "pruned-range", "10%", **kw)
        self.assertFalse((self.t / "pruned-range").exists())
        # the edges themselves are fine
        prune.run(self.t / "gb.ubj", self.t / "all.npz", self.t / "pruned-range", "10%", sub=1.0, max_sel_rows=100)

    def test_keep_edges_are_clamped_or_refused_never_silent(self):
        """A keep count larger than the number of trees keeps them all; zero, a negative number or text is refused."""
        for keep in ("150%", "500", "100%"):
            r = prune.run(self.t / "rf", self.t / "rf.npz", self.t / f"keep-{keep.strip('%')}", keep)
            self.assertEqual({m["kept_per_group"] for m in r["models"]}, {100}, keep)
            self.assertEqual(r["totals"]["node_ratio"], 1.0, keep)
        for bad in ("0%", "0", "-3", "abc", "nan"):
            with self.assertRaises(SystemExit, msg=bad):
                prune.run(self.t / "rf", self.t / "rf.npz", self.t / "keep-bad", bad)
        self.assertEqual([prune.parse_keep(k, 7) for k in ("1%", "50%", "100%", "9", "2.4", "inf")], [1, 4, 7, 7, 2, 7])
        self.assertEqual(
            [prune.keeps_all(k) for k in ("100%", "120%", " 100% ", "100", "99.9%", "1.0", "x%")],
            [True, True, True, False, False, False, False],
        )

    def test_a_plain_number_after_keep_is_a_count_not_100_percent(self):
        """`--keep 100` is 100 trees per class, not 100 %: the pruning must run."""
        x, y, _ = make_data()
        xgb.train(
            {"objective": "binary:logistic", "max_depth": 2, "eta": 0.3, "nthread": 2},
            xgb.DMatrix(x[:2000], (y[:2000] == 0).astype(int)),
            150,
        ).save_model(str(self.t / "big.ubj"))
        for i, (keep, trees) in enumerate((("100", 100), ("100%", 150), ("250%", 150))):
            rep = optimize.optimize(self.t / "big.ubj", self.t / "all.npz", self.t / f"opt-m{i}", keep, min_agreement=0.0)
            self.assertTrue(rep["ok"], rep["failures"])
            n = qf.load_forest(qf_formats.model_files(self.t / f"opt-m{i}" / "pruned")).n_trees
            self.assertEqual(n, trees, keep)

    def test_a_logistic_base_score_at_the_extremes_is_clamped_like_xgboost(self):
        """xgboost holds a logistic base_score in [1e-6, 1 - 1e-6] (a fold with one class has base_score 0 or 1); the logit of 0 would be
        -inf."""
        x, y, _ = make_data()
        for obj in ("binary:logistic", "reg:logistic"):
            b = xgb.train(
                {"objective": obj, "max_depth": 2, "nthread": 1, "base_score": 0.5}, xgb.DMatrix(x[:500], (y[:500] == 0).astype(int)), 3
            )
            j = json.loads(bytes(b.save_raw(raw_format="json")))
            for base in ("0E0", "1E-7", "1E0", "9.9999994E-1", "1E-6", "9.99999E-1"):
                j["learner"]["learner_model_param"]["base_score"] = base
                m = self.t / f"base-{obj[:3]}-{base}.json"
                m.write_text(json.dumps(j))
                flat = self.t / "base.flat"
                compile.compile_model(m, flat, backend="flat")
                res = check.run(m, flat, n_random=500)
                self.assertTrue(res["ok"], (obj, base, res["sets"]))

    def test_heldout_agreement_follows_the_average_rule_of_the_ensemble(self):
        """Models without a fold array are combined by the output mean in the report whatever `--average` the compiled forest uses."""
        for average in ("output", "margin"):
            r = prune.run(self.t / "rf", self.t / "all.npz", self.t / f"avg-{average}", "10%", average=average)
            files, pruned = qf_formats.model_files(self.t / "rf"), qf_formats.model_files(self.t / f"avg-{average}")
            h = np.where(qf.slices(qf.load_data(self.t / "all.npz"), 0.5, 15) == 1)[0]
            x = qf.load_data(self.t / "all.npz")["X"][h]
            decisions = []
            for fs in (files, pruned):
                ps = [qf_formats.open_source(f) for f in fs]
                out = (
                    np.mean([s.predict(x) for s in ps], 0)
                    if average == "output"
                    else qf.transform(np.mean([s.predict(x, margin=True) for s in ps], 0), "softmax")
                )
                decisions.append(out.argmax(1))
            self.assertEqual(r["heldout"]["rows"], len(h))
            self.assertEqual(r["heldout"]["same"], int((decisions[0] == decisions[1]).sum()), average)

    def test_the_overfit_figure_compares_the_models_own_selection_and_held_out_agreement(self):
        """The overfit the summary prints is selection agreement minus held-out agreement. The selection figure is each model's own, so the
        held-out one it is subtracted from must be each model's own too: several models without a fold array are averaged into ONE ensemble
        figure (`pct`), which is another quantity and made the printed overfit meaningless."""
        r = prune.run(self.t / "rf", self.t / "all.npz", self.t / "overfit", "10%")
        h = r["heldout"]
        own = round(
            100 * sum(m["heldout_agreement"]["same"] for m in r["models"]) / sum(m["heldout_agreement"]["rows"] for m in r["models"]), 3
        )
        self.assertNotEqual(own, h["pct"], "the fixture must make the ensemble figure differ from the models' own")
        self.assertEqual(h["models_pct"], own)
        self.assertIn(f"so overfit {h['selection_pct'] - own:+.2f} points", prune.summary(r))
        # one model, or models with a fold array: the two figures are the same quantity
        f = prune.run(self.t / "rf", self.t / "rf.npz", self.t / "overfit-fold", "10%")
        self.assertEqual(f["heldout"]["models_pct"], f["heldout"]["pct"])

    def test_a_model_with_more_features_than_the_format_holds_is_refused_at_compile_time(self):
        """The flat format holds at most 32,767 features: a wider forest is refused at compile time, not written as a file that fails to
        load ("counts out of range")."""
        tree = qf.Tree(
            0,
            np.array([1, -1, -1], np.int32),
            np.array([2, -1, -1], np.int32),
            np.array([3, 0, 0], np.int32),
            np.array([0.5, 1.0, 2.0], np.float32),
            np.array([True, False, False]),
        )
        for n_features, ok in ((32767, True), (32768, False), (40000, False)):
            f = qf.Forest(n_features, 1, "identity", "output", [qf.ModelIR(np.float32([0.0]), [tree])])
            out = self.t / "wide.flat"
            if ok:
                backends.get("flat").compile(f, out, {})
                self.assertEqual(evaluator.FlatForest(out).n_features, n_features)
            else:
                with self.assertRaisesRegex(ValueError, "at most 32,767 features"):
                    backends.get("flat").compile(f, out, {})

    def test_the_profile_rows_are_never_the_timed_rows_even_on_a_small_data_file(self):
        """With fewer rows than the speed set, the profile rows must stay distinct from the timed rows, so "held-out" is true."""
        import compare

        x, _, _ = make_data()
        x = x[:3000].copy()
        x[:, 0] = np.arange(len(x))  # a unique id per row
        np.savez(self.t / "small.npz", X=x)
        work = self.t / "prep-small"
        compare.prepare(
            self.t / "gb.ubj",
            self.t / "small.npz",
            work,
            average="output",
            eq_rows=20000,
            n_random=50,
            speed_rows=20000,
            seed=15,
            log=lambda *_: None,
        )
        timed, profile = np.load(work / "x-speed.npy")[:, 0], np.load(work / "x-pgo.npy")[:, 0]
        self.assertGreater(len(timed), 0)
        self.assertGreater(len(profile), 0)
        self.assertEqual(len(np.intersect1d(timed, profile)), 0)

    def test_a_retrained_model_of_the_same_file_name_gets_fresh_reference_answers(self):
        """The reference answers are cached by set name and length, so the cache signature must change with the model: a changed model is
        never compared with another model's answers (a correct compiled forest would read as NOT EXACT)."""
        import shutil

        import compare

        x, y, _ = make_data()
        yb = (y == 0).astype(int)
        mdir = self.t / "stale-ref"
        mdir.mkdir()
        work = self.t / "prep-stale-ref"
        for seed in (1, 2):
            p = {
                "objective": "binary:logistic",
                "max_depth": 3,
                "eta": 0.3,
                "tree_method": "hist",
                "nthread": 2,
                "seed": seed,
                "subsample": 0.7,
            }
            xgb.train(p, xgb.DMatrix(x[:2000], yb[:2000]), 10 + seed).save_model(str(mdir / "m.ubj"))
            files, _ = compare.prepare(
                mdir, self.t / "all.npz", work, average="output", eq_rows=800, n_random=50, speed_rows=1000, seed=3, log=lambda *_: None
            )
            for k in ("data", "random"):
                got = np.load(next(work.glob(f"ref-{k}-*.npy")))
                want = qf.reference_predict(files, np.load(work / f"x-{k}.npy"), "output")
                np.testing.assert_array_equal(got, want, (seed, k))
        shutil.rmtree(work)

    def compare_models(self, name):
        """Two different binary models and a data file for the compare tests (the same file name in two folders)."""
        x, y, _ = make_data()
        yb = (y == 0).astype(int)
        out = []
        for seed in (1, 2):
            d = self.t / f"{name}-{seed}"
            d.mkdir()
            p = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "tree_method": "hist", "nthread": 2, "seed": seed}
            xgb.train(p, xgb.DMatrix(x[:2000] + seed, yb[:2000] if seed == 1 else 1 - yb[:2000]), 8).save_model(str(d / "m.ubj"))
            out.append(d / "m.ubj")
        return out

    def test_compare_without_random_rows_measures_every_backend(self):
        """`--random 0` (no threshold-hugging rows) writes no x-random.npy, so the worker must not load it as the "random" timing rows."""
        import compare

        a, _ = self.compare_models("norandom")
        work = self.t / "w-norandom"
        t = compare.run_table(
            a, ["flat", "wave"], self.t / "all.npz", work, eq_rows=500, n_random=0, speed_rows=500, budget=0.3, log=lambda *_: None
        )
        self.assertEqual([r["status"] for r in t["rows"]], ["ok", "ok"], t["rows"])

    def test_compare_judges_the_decisions_of_a_ydf_random_forest_classifier(self):
        """A YDF random forest classifier is an `identity` forest with several output groups, and `compare` built no decision rule for
        identity forests, so its decision mismatches were never counted (a wrong answer still failed on max_err, but the report said 0).
        `check.py` decides by the task: so does `compare`."""
        try:
            import ydf
        except ImportError:
            self.skipTest("no ydf")
        import compare

        rng = np.random.default_rng(2)
        x = rng.normal(size=(800, 4)).astype(np.float32)
        y = (x[:, 0] + x[:, 1] > 0).astype(int) + (x[:, 2] > 0.7).astype(int)
        model = self.t / "ydf-rf"
        ydf.RandomForestLearner(label="y", task=ydf.Task.CLASSIFICATION, num_trees=12, max_depth=5, num_threads=1).train(
            {**{f"f{i}": x[:, i] for i in range(4)}, "y": y}, verbose=0
        ).save(str(model))
        np.savez(self.t / "ydf-rf.npz", X=x, group=np.arange(len(x)))
        rules = []
        real = check.compare

        def spy(name, ref, got, decision, rule, groups, absolute=False, exact=False):
            rules.append(rule)
            return real(name, ref, got, decision, rule, groups, absolute, exact)

        with mock.patch.object(check, "compare", spy):
            t = compare.run_table(
                model,
                ["flat"],
                self.t / "ydf-rf.npz",
                self.t / "w-ydf-rf",
                eq_rows=500,
                n_random=100,
                speed_rows=500,
                budget=0.3,
                log=lambda *_: None,
            )
        self.assertEqual([r["status"] for r in t["rows"]], ["ok"], t["rows"])
        self.assertTrue(
            rules and all(r is not None for r in rules), "the decisions of a classifier are compared whatever its output transform"
        )

    def test_compare_judges_an_xgboost_backend_at_difference_zero_like_check_does(self):
        """`check` and `--verify` demand the bit-identity the library is claimed to have for xgboost; `compare` (and so `--backend auto`,
        which keeps the fastest EXACT backend) judged the same backends at 1e-6, so a backend that drifted by 1e-7 would have been chosen."""
        import compare

        a, _ = self.compare_models("exactbar")
        seen = []
        real = check.compare

        def spy(name, ref, got, decision, rule, groups, absolute=False, exact=False):
            seen.append(exact)
            return real(name, ref, got, decision, rule, groups, absolute, exact)

        with mock.patch.object(check, "compare", spy):
            compare.run_table(
                a,
                ["flat"],
                self.t / "all.npz",
                self.t / "w-exactbar",
                eq_rows=300,
                n_random=50,
                speed_rows=300,
                budget=0.2,
                log=lambda *_: None,
            )
        self.assertTrue(seen and all(seen), seen)
        self.assertTrue(check.bit_exact([a], "output", qf.load_forest([a])))
        self.assertFalse(check.bit_exact([a], "output", qf.load_forest([a]), margin=True))
        self.assertFalse(check.bit_exact([self.t / "gb.ubj"] * 2, "margin", qf.load_forest([self.t / "gb.ubj"] * 2, "margin")))

    def test_measure_stage_refuses_a_work_folder_built_from_another_model(self):
        """`compare --stage measure` refuses build-*.json files made from another model, instead of judging them against the new model's
        answers."""
        import compare

        a, b = self.compare_models("mismatch")
        data, work, quiet = self.t / "all.npz", self.t / "w-mismatch", lambda *_: None
        kw = {"eq_rows": 500, "n_random": 200, "speed_rows": 500, "log": quiet}
        compare.run_table(a, ["flat"], data, work, stage="build", **kw)
        with self.assertRaises(SystemExit) as cm:
            compare.run_table(b, ["flat"], data, work, stage="measure", budget=0.3, **kw)
        self.assertIn("another model", str(cm.exception))
        with self.assertRaises(SystemExit):  # the same model folded the other way (margin instead of output) is not the same build either
            compare.run_table(a, ["flat"], data, work, average="margin", stage="measure", budget=0.3, **kw)
        t = compare.run_table(a, ["flat"], data, work, stage="measure", budget=0.3, **kw)  # its own build still measures
        self.assertEqual([r["status"] for r in t["rows"]], ["ok"], t["rows"])
        # a build file written before the signature existed cannot be checked: measured with a warning, not refused
        row = json.loads((work / "build-flat.json").read_text())
        row.pop("model_sig")
        (work / "build-flat.json").write_text(json.dumps(row))
        said = []
        t = compare.run_table(a, ["flat"], data, work, stage="measure", budget=0.3, **{**kw, "log": said.append})
        self.assertEqual([r["status"] for r in t["rows"]], ["ok"])
        self.assertTrue(any("signature" in m for m in said), said)

    def test_the_model_signature_follows_the_bytes_not_the_timestamp(self):
        """The cache signature hashes the bytes of every model file: a YDF model is a folder whose mtime and size do not change when a file
        inside is rewritten, and a same-size retrain restored with rsync -a keeps both."""
        import compare

        d = self.t / "fp"
        d.mkdir()
        f = d / "model.bin"
        f.write_bytes(b"A" * 100)
        st = f.stat()
        before = compare.model_fingerprint([f])
        f.write_bytes(b"B" * 100)
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertEqual(f.stat().st_size, st.st_size)
        self.assertNotEqual(compare.model_fingerprint([f]), before)
        inner = d / "inner"
        inner.mkdir()
        (inner / "nodes").write_bytes(b"x" * 10)
        before = compare.model_fingerprint([inner])
        (inner / "nodes").write_bytes(b"y" * 10)  # the folder's own mtime does not move
        self.assertNotEqual(compare.model_fingerprint([inner]), before)

    def test_a_regenerated_data_file_of_the_same_name_and_size_is_measured_on_its_own_rows(self):
        """The prepared test rows are cached in --work, keyed by the data file. Its name and byte size are not what it IS: `np.savez` of
        another array of the same shape writes a file of the same name and size, and the second run measured rows of the first."""
        import shutil

        import compare

        x, y, _ = make_data()
        mdir = self.t / "regen-model"
        mdir.mkdir()
        xgb.train(
            {"objective": "binary:logistic", "max_depth": 3, "nthread": 2}, xgb.DMatrix(x[:2000], (y[:2000] == 0).astype(int)), 6
        ).save_model(str(mdir / "m.ubj"))
        work, data = self.t / "prep-regen", self.t / "regen.npz"
        rng = np.random.default_rng(5)
        arrays = [rng.normal(size=(400, NF)).astype(np.float32) for _ in range(2)]
        sizes = set()
        for k, a in enumerate(arrays):
            np.savez(data, X=a)
            sizes.add(data.stat().st_size)
            compare.prepare(mdir, data, work, average="output", eq_rows=400, n_random=0, speed_rows=400, seed=3, log=lambda *_: None)
            rows = {r.tobytes() for r in a}
            for name in ("x-data", "x-speed", "x-pgo"):
                got = np.load(work / f"{name}.npy")
                self.assertTrue(
                    all(r.tobytes() in rows for r in got), f"{name}.npy holds rows that are not in the data file given (run {k + 1})"
                )
        self.assertEqual(len(sizes), 1, "the two data files must have the same size, or this test proves nothing")
        shutil.rmtree(work)

    def test_cpu_name_reads_the_fields_of_x86_and_aarch64_kernels(self):
        import platform

        import compare

        cases = {
            "processor\t: 0\nmodel name\t: Intel(R) Core(TM) i7-8550U\n": "Intel(R) Core(TM) i7-8550U",
            "processor\t: 0\nModel name\t: Neoverse-N1\n": "Neoverse-N1",
            "processor\t: 0\nBogoMIPS\t: 38.40\nHardware\t: Qualcomm Technologies, Inc SM8250\n": "Qualcomm Technologies, Inc SM8250",
            "processor\t: 0\nCPU part\t: 0xd0b\n": platform.processor() or "unknown",
        }
        for text, want in cases.items():
            f = self.t / "cpuinfo"
            f.write_text(text)
            self.assertEqual(compare.cpu_name(f), want, text)
        self.assertEqual(compare.cpu_name(self.t / "no-such-cpuinfo"), platform.processor() or "unknown")

    def test_the_default_output_folder_of_optimize_names_the_model_even_when_it_is_a_dot(self):
        """`optimize .` made `-optimized`, a name that any later command line reads as an option."""
        folds = self.t / "folds"
        folds.mkdir()
        (folds / "sub").mkdir()
        here = os.getcwd()
        try:
            os.chdir(folds)
            parent = Path.cwd().parent  # what the kernel calls the folder (a bind mount, as in proot, is not a link `resolve` follows)
            self.assertEqual(Path(optimize.default_out(".")), parent / "folds-optimized")
            os.chdir(folds / "sub")
            self.assertEqual(Path(optimize.default_out("..")), parent / "folds-optimized")
        finally:
            os.chdir(here)
        self.assertEqual(Path(optimize.default_out(self.t / "gb.ubj")), Path(os.path.realpath(self.t)) / "gb-optimized")
        with self.assertRaises(ValueError):
            optimize.default_out("/")

    def test_optimize_on_a_dot_writes_next_to_the_folder_under_its_name(self):
        import subprocess

        folds = self.t / "dotfolds"
        folds.mkdir()
        for k in range(2):
            (folds / f"fold{k}.ubj").write_bytes((self.t / "gb.ubj").read_bytes())
        r = subprocess.run(
            [
                sys.executable,
                "-s",
                str(SCRIPTS / "optimize.py"),
                ".",
                "--data",
                str(self.t / "all.npz"),
                "--keep",
                "50%",
                "--min-agreement",
                "0",
            ],
            cwd=folds,
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertTrue((self.t / "dotfolds-optimized" / "report.json").is_file())
        self.assertFalse((self.t / "-optimized").exists())

    def test_a_keep_count_below_one_is_refused_not_rounded_to_one_tree(self):
        """`--keep 0.1` is a count of 0.1 (easily meant as 10 %): it used to keep ONE tree per group without a word."""
        for bad in ("0.1", "0.5", "0.99", " 0.3 "):
            with self.assertRaises(SystemExit, msg=bad) as cm:
                prune.parse_keep(bad, 100)
            self.assertIn("percent sign", str(cm.exception.code), bad)
        self.assertEqual(
            [prune.parse_keep(k, 100) for k in ("1", "1.4", "30", "0.1%", "10%")], [1, 1, 30, 1, 10]
        )  # a share keeps at least one
        out = self.t / "keep-below-one"
        with self.assertRaises(SystemExit):
            prune.run(self.t / "gb.ubj", self.t / "all.npz", out, "0.1")
        with self.assertRaises(SystemExit):
            optimize.optimize(self.t / "gb.ubj", self.t / "all.npz", self.t / "keep-below-one-opt", "0.1")
        self.assertFalse((self.t / "keep-below-one-opt").exists())

    def test_optimize_refuses_flags_that_cannot_apply_before_it_prunes_anything(self):
        """`compile` refuses `--float32-sums` on a model that is not YDF at once; `optimize` pruned first and only then reported
        `NOT COMPILED` (exit 3, pruned/ left), and without `--compile` it ignored both flags in silence."""
        data = self.t / "all.npz"
        out = self.t / "flags-refused"
        with self.assertRaises(ValueError) as cm:
            optimize.optimize(self.t / "gb.ubj", data, out, "50%", compile_=True, float32_sums=True, backend="flat")
        self.assertIn("YDF models only", str(cm.exception))
        self.assertFalse(out.exists(), "nothing may be written before the refusal")
        for flag in ({"f64_leaves": True}, {"float32_sums": True}):
            with self.assertRaises(ValueError, msg=flag) as cm:
                optimize.optimize(self.t / "gb.ubj", data, out, "50%", **flag)
            self.assertIn("--compile", str(cm.exception))
            self.assertFalse(out.exists())
        with self.assertRaises(SystemExit) as cm:  # the command line: one line, no traceback
            optimize.main([str(self.t / "gb.ubj"), "--data", str(data), "--out", str(out), "--compile", "--float32-sums"])
        self.assertIn("optimize.py:", str(cm.exception.code))
        self.assertFalse(out.exists())
        rep = optimize.optimize(
            self.t / "gb.ubj", data, out, "50%", compile_=True, f64_leaves=True, min_agreement=0.0, backend="flat"
        )  # a no-op on xgboost
        self.assertTrue(rep["ok"], rep["failures"])

    def test_a_pretty_printed_json_model_in_a_fold_folder_is_still_a_fold(self):
        """A folder of folds is read whatever the JSON indentation: a model re-saved with indentation (`{\n  "learner"`) is still a fold."""
        d = self.t / "pretty-folds"
        d.mkdir()
        x, y, _ = make_data()
        yb = (y == 0).astype(int)
        p = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "tree_method": "hist", "nthread": 2}
        for k in range(3):
            xgb.train(p, xgb.DMatrix(x[:1500], yb[:1500]), 5).save_model(str(d / f"fold{k}.json"))
        doc = json.loads((d / "fold1.json").read_text())
        (d / "fold1.json").write_text(json.dumps(doc, indent=2))
        self.assertEqual([f.name for f in qf_formats.model_files(d)], ["fold0.json", "fold1.json", "fold2.json"])

    def test_a_folder_that_mixes_formats_or_holds_a_fold_twice_is_refused(self):
        """A stray LightGBM text file next to xgboost folds, or `fold0.ubj` next to `fold0.json`, is refused; it would be averaged in as one
        more fold and shift the fold numbers of the data onto the wrong models."""
        x, y, _ = make_data()
        yb = (y == 0).astype(int)
        b = xgb.train(
            {"objective": "binary:logistic", "max_depth": 3, "tree_method": "hist", "nthread": 2}, xgb.DMatrix(x[:1500], yb[:1500]), 5
        )
        twice = self.t / "fold-twice"
        twice.mkdir()
        b.save_model(str(twice / "fold0.ubj"))
        b.save_model(str(twice / "fold0.json"))
        with self.assertRaises(SystemExit) as cm:
            qf_formats.model_files(twice)
        self.assertIn("saved twice", str(cm.exception))
        mixed = self.t / "fold-mixed"
        mixed.mkdir()
        b.save_model(str(mixed / "fold0.ubj"))
        (mixed / "fold1.txt").write_text("tree\nversion=v4\nnum_class=1\n")
        with self.assertRaises(SystemExit) as cm:
            qf_formats.model_files(mixed)
        self.assertIn("several formats", str(cm.exception))

    def test_fold_names_that_differ_only_by_case_are_one_fold_and_sort_by_their_number(self):
        """Fold files are ordered by their number, not by ASCII order, and `Fold0.ubj` next to `fold0.ubj` is refused as a duplicate:
        otherwise the fold numbers of the data point at the wrong models."""
        x, y, _ = make_data()
        yb = (y == 0).astype(int)
        b = xgb.train(
            {"objective": "binary:logistic", "max_depth": 3, "tree_method": "hist", "nthread": 2}, xgb.DMatrix(x[:1500], yb[:1500]), 5
        )
        d = self.t / "fold-case"
        d.mkdir()
        for n in ("fold0.ubj", "Fold1.ubj", "fold2.ubj", "fold10.ubj"):
            b.save_model(str(d / n))
        self.assertEqual([f.name for f in qf_formats.model_files(d)], ["fold0.ubj", "Fold1.ubj", "fold2.ubj", "fold10.ubj"])
        b.save_model(str(d / "FOLD2.ubj"))
        with self.assertRaises(SystemExit) as cm:
            qf_formats.model_files(d)
        self.assertIn("saved twice", str(cm.exception))
        for kind in ("lightgbm", "ydf"):  # the same rule for the other formats: a text model saved under two suffixes, a model folder twice
            twin = self.t / f"fold-twice-{kind}"
            twin.mkdir()
            for name in ("m.txt", "M.model") if kind == "lightgbm" else ("m", "m.ydf"):
                if kind == "lightgbm":
                    (twin / name).write_text("tree\nversion=v4\nnum_class=1\n")
                else:
                    (twin / name).mkdir()
                    (twin / name / "header.pb").write_bytes(b"")
            with self.assertRaises(SystemExit) as cm:
                qf_formats.model_files(twin)
            self.assertIn("saved twice", str(cm.exception), kind)
        with self.assertRaises(SystemExit) as cm:  # a path that is not there is said, not a FileNotFoundError traceback
            qf_formats.model_files(self.t / "no-such-folder")
        self.assertIn("no such", str(cm.exception))

    def test_keeping_100_percent_still_validates_the_model_and_copies_only_models(self):
        """`optimize --keep 100%` copies only valid models of one format to pruned/: a text file, a folder that mixes formats or a stray
        file next to the folds (a dataset) is refused."""
        x, y, _ = make_data()
        yb = (y == 0).astype(int)
        b = xgb.train(
            {"objective": "binary:logistic", "max_depth": 3, "tree_method": "hist", "nthread": 2}, xgb.DMatrix(x[:1500], yb[:1500]), 5
        )
        data = self.t / "all.npz"
        junk = self.t / "junk.bin"
        junk.write_text("not a model")
        out = self.t / "keep100-junk"
        with self.assertRaises(SystemExit):
            optimize.optimize(junk, data, out, "100%")
        self.assertFalse(out.exists())  # nothing left behind: the retry in the same --out works
        mixed = self.t / "keep100-mixed"
        mixed.mkdir()
        b.save_model(str(mixed / "fold0.ubj"))
        (mixed / "fold1.txt").write_text("tree\nversion=v4\nnum_class=1\n")
        with self.assertRaises(SystemExit) as cm:
            optimize.optimize(mixed, data, self.t / "keep100-mixed-out", "100%")
        self.assertIn("several formats", str(cm.exception))
        ok = self.t / "keep100-ok"
        ok.mkdir()
        b.save_model(str(ok / "fold0.ubj"))
        b.save_model(str(ok / "fold1.ubj"))
        (ok / "train.csv").write_text("a,b\n1,2\n")
        rep = optimize.optimize(ok, data, self.t / "keep100-ok-out", "100%", compile_=True, verify=True, backend="flat")
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertEqual(sorted(q.name for q in (self.t / "keep100-ok-out" / "pruned").iterdir()), ["fold0.ubj", "fold1.ubj"])

    def test_a_model_saved_after_early_stopping_is_flagged(self):
        """The scikit-learn wrapper predicts with the first best_iteration + 1 rounds of a saved model; Booster.predict (the bar's
        reference) and the compiled forest use all of them. The difference must be reported, not silent."""
        import warnings

        x, y, _ = make_data()
        p = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.5, "nthread": 1}
        d = xgb.DMatrix(x[:2000], (y[:2000] == 0).astype(int))
        v = xgb.DMatrix(x[2000:3000], (y[2000:3000] == 0).astype(int))
        b = xgb.train(p, d, 80, evals=[(v, "v")], early_stopping_rounds=3, verbose_eval=False)
        self.assertLess(b.best_iteration + 1, b.num_boosted_rounds())
        b.save_model(str(self.t / "early.ubj"))
        with self.assertWarns(UserWarning) as w:
            qf_formats.open_source(self.t / "early.ubj")
        self.assertIn("early stopping", str(w.warning))
        self.assertIn(f"first {b.best_iteration + 1}", str(w.warning))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            qf_formats.open_source(self.t / "gb.ubj")  # a model trained to the end says nothing
        r = prune.run(self.t / "early.ubj", self.t / "all.npz", self.t / "pruned-early", "50%")
        self.assertIn("early stopping", r["models"][0]["notes"][0])

    def test_the_xgboost_baseline_of_a_speed_figure_times_the_prediction_not_the_dmatrix(self):
        """The speed baseline is xgboost's `inplace_predict`: timing `Booster.predict(DMatrix(x))` would include the DMatrix construction
        (0.3 to 18 ms for 3,000 rows, on all threads whatever the booster's nthread) and inflate the "x xgboost" figure of small forests."""
        import compare

        built = []

        class Counting(xgb.DMatrix):
            def __init__(self, *a, **k):
                built.append(1)
                super().__init__(*a, **k)

        x = make_data()[0][:300]
        flat = self.t / "base.flat"
        compile.compile_model(self.t / "gb.ubj", flat, backend="flat")
        with mock.patch.object(xgb, "DMatrix", Counting):
            self.assertGreater(compare.library_speed([self.t / "gb.ubj"], x)[0], 0)
            self.assertGreater(check.bench([self.t / "gb.ubj"], flat, backends.get("flat"), x, 1)["library_rows_per_s"], 0)
        self.assertEqual(built, [], "a DMatrix was built inside the timed xgboost call")

    def test_the_speed_figure_is_measured_on_real_rows_when_given(self):
        """check --speed times the real rows when real data is given, not the random threshold-hugging rows."""
        flat = self.t / "speed.flat"
        compile.compile_model(self.t / "gb.ubj", flat, backend="flat")
        self.assertEqual(
            check.run(self.t / "gb.ubj", flat, data=self.t / "all.npz", n_random=300, speed=True)["speed"]["rows_from"], "real rows"
        )
        self.assertEqual(check.run(self.t / "gb.ubj", flat, n_random=300, speed=True)["speed"]["rows_from"], "random rows")

    def test_pruning_into_a_folder_with_the_folds_of_an_earlier_run_is_refused(self):
        """prune into an OUT_DIR that still holds a fold of an earlier run with more models is refused (naming the stale file): the compiler
        must not see three folds where two were written."""
        x, y, _ = make_data()
        for k in (3, 2):
            (self.t / f"stale-m{k}").mkdir()
            for i in range(k):
                xgb.train(
                    {"objective": "multi:softprob", "num_class": 3, "max_depth": 3, "nthread": 1}, xgb.DMatrix(x[:800], y[:800]), 6
                ).save_model(str(self.t / f"stale-m{k}" / f"fold{i}.ubj"))
            np.savez(self.t / f"stale-{k}.npz", X=x[:1200], fold=np.arange(1200) % k, group=np.arange(1200) // k)
        out = self.t / "stale-out"
        prune.run(self.t / "stale-m3", self.t / "stale-3.npz", out, "50%")
        prune.run(self.t / "stale-m3", self.t / "stale-3.npz", out, "50%")  # the same run again overwrites its own files: fine
        with self.assertRaises(SystemExit) as cm:
            prune.run(self.t / "stale-m2", self.t / "stale-2.npz", out, "50%")
        self.assertIn("fold2.ubj", str(cm.exception))
        self.assertEqual(len(qf.load_forest(qf_formats.model_files(out)).models), 3)

    def test_pruning_into_the_models_own_folder_is_refused_not_overwritten(self):
        """prune refuses an OUT_DIR that is the folder of the model: the caller's model must not be replaced by the pruned one."""
        import shutil

        own = self.t / "own"
        own.mkdir()
        shutil.copy(self.t / "gb.ubj", own / "gb.ubj")
        before = (own / "gb.ubj").read_bytes()
        for model in (own / "gb.ubj", own):
            with self.assertRaises(SystemExit):
                prune.run(model, self.t / "all.npz", own, "20%")
        self.assertEqual((own / "gb.ubj").read_bytes(), before)

    def test_a_refused_argument_leaves_the_output_folder_usable(self):
        """optimize removes its work folder when prune refuses (no held-out rows), so the corrected call does not fail with "is not empty"."""
        x = np.load(self.t / "all.npz")["X"]
        np.savez(self.t / "nohold.npz", X=x, slice=np.zeros(len(x), int))
        out = self.t / "opt-retry"
        with self.assertRaises(SystemExit):
            optimize.optimize(self.t / "gb.ubj", self.t / "nohold.npz", out, "50%")
        self.assertFalse(out.exists() and any(out.iterdir()))
        rep = optimize.optimize(self.t / "gb.ubj", self.t / "all.npz", out, "50%", min_agreement=0.0)
        self.assertTrue(rep["ok"], rep["failures"])

    def test_a_closed_forest_refuses_to_predict_instead_of_crashing(self):
        """predict() on a closed FlatForest raises instead of passing a null handle to the library (a segmentation fault). Run in a child
        process."""
        import subprocess
        import textwrap

        compile.compile_model(self.t / "gb.ubj", self.t / "closed.flat", backend="flat")
        code = textwrap.dedent(
            f"""
            import sys
            import numpy as np
            sys.path.insert(0, {str(Path(backends.__file__).parent)!r})
            import evaluator
            f = evaluator.FlatForest({str(self.t / "closed.flat")!r})
            f.close()
            f.close()
            try:
                f.predict(np.zeros((3, f.n_features), np.float32))
            except ValueError:
                sys.exit(0)
            sys.exit(1)
            """
        )
        run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr[-300:])

    def test_a_forest_of_leaf_only_trees_compiles_and_verifies(self):
        """A model whose trees are all a single leaf (constant label, huge gamma) has no thresholds: check's random rows must still work."""
        x, y, _ = make_data()
        for obj, label in (("binary:logistic", (y[:500] == 0).astype(int)), ("reg:squarederror", y[:500].astype(float))):
            xgb.train({"objective": obj, "gamma": 1e9, "nthread": 1}, xgb.DMatrix(x[:500], label), 3).save_model(str(self.t / "leaf.ubj"))
            self.assertEqual(qf.load_forest([self.t / "leaf.ubj"]).n_nodes, 3)
            compile.compile_model(self.t / "leaf.ubj", self.t / "leaf.flat", backend="flat")
            for be in ("flat", "wave"):
                res = check.run(self.t / "leaf.ubj", self.t / "leaf.flat", be, n_random=500)
                self.assertTrue(res["ok"], (obj, be, res["sets"]))

    def test_a_mistyped_backend_is_named_before_any_work_is_done(self):
        """`--backend codegen_pgo` (a typo of `codegen-pgo`) is refused with a message that names the right one, before any pruning runs."""
        with self.assertRaises(ValueError) as cm:
            backends.get("codegen_pgo")
        self.assertIn("codegen-pgo", str(cm.exception))
        out = self.t / "opt-typo"
        with mock.patch.object(prune, "run", side_effect=AssertionError("pruning ran before the backend was checked")):
            with self.assertRaises(ValueError):
                optimize.optimize(self.t / "gb.ubj", self.t / "all.npz", out, "10%", True, False, backend="codegen_pgo")
        self.assertFalse(out.exists())

    def test_a_backend_can_be_swapped_without_touching_the_rest(self):
        class Reference:
            """A trivial evaluator: xgboost itself behind the backend interface."""

            name, suffix = "ref", ".txt"

            def compile(self, forest, out, options):
                out.write_text(str(options.get("src")))
                return {"artifact": str(out), "bytes": 1, "nodes": forest.n_nodes, "trees": forest.n_trees}

            def load(self, artifact, **options):
                files = qf_formats.model_files(artifact.read_text())

                class P:
                    def predict(_, x, threads=1):
                        return qf.reference_predict(files, x).squeeze()

                return P()

        backends.register(Reference())
        art = self.t / "m.txt"
        compile.compile_model(self.t / "gb.ubj", art, backend="ref", src=self.t / "gb.ubj")
        self.assertTrue(check.run(self.t / "gb.ubj", art, backend="ref", n_random=500)["ok"])
        backends.BACKENDS.pop("ref")

    def test_every_engine_gives_the_walkers_bits(self):
        """Wavefront engines (scalar, AVX2 or its fallback): bit-identical to the walker for any row count, NaN included."""
        for model, name in ((self.t / "rf", "rf"), (self.t / "gb.ubj", "gb"), (self.t / "reg.json", "reg")):
            art = self.t / f"eng-{name}.flat"
            compile.compile_model(model, art, backend="flat")
            x = check.random_rows(qf.load_forest(qf_formats.model_files(model)), 700, 3)
            walker = backends.get("flat").load(art)
            want = walker.predict(x)
            for be in ("wave", "wave-avx2"):
                pred = backends.get(be).load(art)
                for n in (1, 7, 8, 9, 300, 700):
                    np.testing.assert_array_equal(pred.predict(x[:n]), want[:n], f"{be} {name} {n} rows")
                np.testing.assert_array_equal(pred.predict(x, threads=3), want)
                self.assertTrue(check.run(model, art, backend=be, n_random=2000)["ok"], (be, name))
                pred.close()
            walker.close()
        self.assertIn(backends.get("wave-avx2").load(self.t / "eng-gb.flat").engine_used, (1, 2))

    def test_recompiling_onto_a_file_that_is_loaded_never_changes_the_loaded_forest(self):
        """The flat file is memory-mapped and the generated `.so` is dlopen-ed: compile writes a new inode instead of truncating and
        rewriting the file in place, so a process that has the old file loaded (a service, or this very process) keeps reading valid
        bytes."""
        import subprocess

        code = (
            "import sys, numpy as np\n"
            f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
            "import backends, compile, check, qf_common as qf, qf_formats\n"
            "a, b, out, be = sys.argv[1:5]\n"
            "compile.compile_model(a, out, backend=be)\n"
            "pred = backends.get(be).load(out)\n"
            "x = check.random_rows(qf.load_forest(qf_formats.model_files(a)), 300, 3)\n"
            "want = pred.predict(x)\n"
            "compile.compile_model(b, out, backend=be)\n"
            "assert np.array_equal(pred.predict(x), want), 'the loaded forest changed'\n"
            "if be == 'flat':  # dlopen of an already loaded path answers the old library, so only the mapped file can be reloaded here\n"
            "    assert not np.array_equal(backends.get(be).load(out).predict(x), want), 'the file was not replaced'\n"
            "print('ok')\n"
        )
        a, b = self.t / "gb.ubj", self.t / "reg.json"
        backs = ["flat"]
        try:
            backend_codegen.toolchain_bin()
            backs.append("codegen")
        except FileNotFoundError:
            pass
        for be in backs:
            out = self.t / f"swap-{be}{'.flat' if be == 'flat' else '.so'}"
            out.unlink(missing_ok=True)
            p = subprocess.run(
                [sys.executable, "-I", "-c", code, str(a), str(b), str(out), be],
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONPATH": ""},
            )
            self.assertEqual((p.returncode, p.stdout.strip()), (0, "ok"), (be, p.returncode, p.stderr[-300:]))

    def test_two_compiles_into_one_folder_at_once_never_collide_on_the_temp_name(self):
        """compare builds backends in parallel into one work folder (and `libtreeflat` is copied next to each flat artifact): the temp name
        is unique per writer, so one writer cannot truncate another's file and make its `os.replace` raise FileNotFoundError."""
        import threading

        payload = [bytes([65 + i]) * (6 << 20) for i in range(2)]

        def go(dest, bar, errs, i):
            try:
                bar.wait()
                qf.replace_file(dest, data=payload[i])
            except Exception as e:
                errs.append(repr(e))

        for rnd in range(6):
            dest = self.t / f"race-{rnd}.so"
            bar, errs = threading.Barrier(2), []
            ths = [threading.Thread(target=go, args=(dest, bar, errs, i)) for i in range(2)]
            [th.start() for th in ths]
            [th.join() for th in ths]
            self.assertEqual(errs, [], rnd)
            self.assertIn(dest.read_bytes(), payload)  # one writer's whole file, never a mix
            self.assertEqual([p.name for p in self.t.glob(f"race-{rnd}.so*")], [dest.name])  # no temp file left behind

    def test_recompiling_without_auto_drops_the_choice_note_of_an_earlier_auto_build(self):
        """`--backend auto` leaves `OUT.choice.json` (which backend loads OUT). A later explicit compile onto the same OUT removes that note, so
        `backends.load_artifact(OUT)` does not load a flat file with the generated-code loader (dlopen of a non-ELF file)."""
        out = self.t / "choice.flat"
        Path(f"{out}.choice.json").write_text(
            json.dumps({"backend": "codegen-pgo", "rows_per_s": 1, "speedup_vs_library": 1, "basis": "x", "cpu": "x"})
        )
        compile.compile_model(self.t / "gb.ubj", out, backend="flat")
        self.assertFalse(Path(f"{out}.choice.json").exists())
        pred = backends.load_artifact(out)
        self.assertEqual(pred.predict(np.zeros((3, NF), dtype=np.float32)).shape, (3,))

    def test_a_compiled_file_named_without_a_folder_loads_from_the_current_folder(self):
        """`compile m.ubj m.flat` then `check m.ubj m.flat` in one folder: the evaluator library sits next to the artifact and is opened by
        its path, since a bare library name is searched by dlopen in the system paths, never in the current folder."""
        here = Path.cwd()
        work = self.t / "cwd-bare"
        work.mkdir()
        try:
            os.chdir(work)
            compile.compile_model(self.t / "gb.ubj", Path("m.flat"), backend="flat")
            self.assertTrue(Path(evaluator.LIB_NAME).is_file())
            res = check.run(self.t / "gb.ubj", Path("m.flat"), data=self.t / "all.npz", n_random=500)
            self.assertTrue(res["ok"], res["sets"])
            try:
                backend_codegen.toolchain_bin()
            except FileNotFoundError:
                return
            compile.compile_model(self.t / "gb.ubj", Path("g.so"), backend="codegen-block", block=64)
            self.assertTrue(check.run(self.t / "gb.ubj", Path("g.so"), backend="codegen-block", n_random=500)["ok"])
        finally:
            os.chdir(here)

    def test_a_forest_that_fails_to_open_does_not_also_crash_its_destructor(self):
        """A FlatForest whose library or file could not be opened prints no "Exception ignored ... no attribute _h" from __del__."""
        import io
        from contextlib import redirect_stderr

        bad = self.t / "not-a-library.so"
        bad.write_text("not an ELF file")
        err = io.StringIO()
        with mock.patch.dict(os.environ, {"TREEFLAT_LIB": str(bad)}), redirect_stderr(err), self.assertRaises(OSError):
            evaluator.FlatForest(self.t / "gb.ubj")
        import gc

        gc.collect()
        self.assertNotIn("_h", err.getvalue())

    def test_generated_code_backends(self):
        """Per-tree code generation (row at a time, row blocking, profile-guided): exact, and bit-identical to the walker."""
        try:
            backend_codegen.toolchain_bin()
            backend_codegen.llvm_profdata()
        except FileNotFoundError as e:
            self.skipTest(str(e))
        pruned = self.t / "pruned-rf"
        if not pruned.is_dir():
            prune.run(self.t / "rf", self.t / "rf.npz", pruned, "10%")
        flat = self.t / "cg-ref.flat"
        compile.compile_model(pruned, flat, backend="flat")
        x = check.random_rows(qf.load_forest(qf_formats.model_files(pruned)), 500, 5)
        want = backends.get("flat").load(flat).predict(x)
        np.save(self.t / "pgo-rows.npy", tiled(x[:300]))
        for be in ("codegen", "codegen-block", "codegen-pgo"):
            art = self.t / f"cg-{be}.so"
            compile.compile_model(pruned, art, backend=be, pgo_rows=str(self.t / "pgo-rows.npy"), block=64)
            pred = backends.get(be).load(art)
            np.testing.assert_array_equal(pred.predict(x), want, be)
            np.testing.assert_array_equal(pred.predict(x[:5]), want[:5], be)
            self.assertTrue(check.run(pruned, art, backend=be, n_random=1500)["ok"], be)

    def test_the_profile_guided_backend_takes_its_profile_rows_from_the_data_file_of_compile_and_optimize(self):
        """`--backend codegen-pgo` is documented as rebuilt on a profile pass over `--data`: compile and optimize must hand it those rows, and
        refuse with one line, before any work, when there is no data file."""
        try:
            backend_codegen.toolchain_bin()
            backend_codegen.llvm_profdata()
        except FileNotFoundError as e:
            self.skipTest(str(e))
        model = self.t / "gb.ubj"
        x, _, _ = make_data()
        np.save(self.t / "pgo-data.npy", tiled(x[:300]))
        flat = self.t / "pgo-ref.flat"
        compile.compile_model(model, flat, backend="flat")
        rows = check.random_rows(qf.load_forest(qf_formats.model_files(model)), 400, 9)
        want = backends.get("flat").load(flat).predict(rows)
        art = self.t / "pgo-npy.so"
        info = compile.compile_model(model, art, backend="codegen-pgo", data=self.t / "pgo-data.npy")
        self.assertEqual(info["backend"], "codegen-pgo")
        np.testing.assert_array_equal(backends.get("codegen-pgo").load(art).predict(rows), want)
        with self.assertRaisesRegex(ValueError, "--data"):
            compile.compile_model(model, self.t / "pgo-nodata.so", backend="codegen-pgo")
        self.assertFalse((self.t / "pgo-nodata.so").exists())
        rep = optimize.optimize(model, self.t / "all.npz", self.t / "opt-pgo", "100%", True, True, backend="codegen-pgo")
        self.assertTrue(rep["ok"], rep)
        self.assertIsNone(rep["compile_unsupported"])
        self.assertEqual(rep["stages"]["compile"]["backend"], "codegen-pgo")
        big = self.t / "pgo-big.npz"
        np.savez(big, X=np.zeros((compile.PROFILE_ROWS + 5000, 3), np.float32))
        compile.profile_rows(big, self.t / "pgo-big.npy")
        self.assertEqual(np.load(self.t / "pgo-big.npy").shape, (compile.PROFILE_ROWS, 3))

    def test_optimize_without_rustc_keeps_pruned_and_exits_3(self):
        """`optimize --compile --backend codegen-*` on a machine without rustc ended in exit 1 and deleted pruned/ (ToolchainMissing is
        a FileNotFoundError, a bare OSError for the main); REFERENCE says exit 3, pruned/ kept, nothing compiled."""
        out = self.t / "opt-norustc"
        err = backend_codegen.ToolchainMissing("rustc not found (needed to compile generated code)")
        with mock.patch.object(backend_codegen, "toolchain_bin", side_effect=err):
            with self.assertRaises(SystemExit) as cm:
                optimize.main(
                    [
                        str(self.t / "gb.ubj"),
                        "--data",
                        str(self.t / "all.npz"),
                        "--out",
                        str(out),
                        "--keep",
                        "50%",
                        "--min-agreement",
                        "1",
                        "--compile",
                        "--backend",
                        "codegen-block",
                    ]
                )
        self.assertEqual(cm.exception.code, 3)
        self.assertTrue((out / "pruned").is_dir(), "the prune work is kept")
        rep = json.loads((out / "report.json").read_text())
        self.assertTrue(rep["ok"])
        self.assertIn("rustc not found", rep["compile_unsupported"])
        self.assertNotIn("compile", rep["stages"])

    def test_verify_of_a_model_too_wide_to_test_is_refused_before_anything_is_pruned(self):
        """What --verify cannot test at all (over 100,000 features) is refused at the start, with no output folder left behind."""
        np.savez(self.t / "toowide.npz", X=np.zeros((2, 100001), dtype=np.float32))
        out = self.t / "opt-toowide"
        with self.assertRaisesRegex(ValueError, "cannot be tested"):
            optimize.optimize(self.t / "gb.ubj", self.t / "toowide.npz", out, "50%", True, True, backend="flat")
        self.assertFalse(out.exists())
        self.assertEqual(qf.data_width(self.t / "toowide.npz"), 100001)
        np.save(self.t / "w.npy", np.zeros((2, 7), dtype=np.float32))
        self.assertEqual(qf.data_width(self.t / "w.npy"), 7)
        self.assertIsNone(qf.data_width(self.t / "missing.npz"))
        fifo = self.t / "pipe.npz"
        os.mkfifo(fifo)
        self.assertIsNone(qf.data_width(fifo))  # returns at once: opening a pipe with no writer would block

    def test_verify_sizes_its_random_rows_to_the_features_instead_of_refusing_a_wide_model(self):
        """`optimize --compile --verify` asked for 10,000 random rows whatever the width and refused every model over 10,000 features
        after pruning and compiling it, though LIMITS allows 32,767."""
        self.assertEqual(check.verify_rows(8), 10000)
        self.assertEqual(check.verify_rows(10000), 10000)
        self.assertEqual(check.verify_rows(10100), 9900)
        self.assertEqual(check.verify_rows(32767), 3051)
        self.assertEqual(check.verify_rows(100000), 1000)
        with self.assertRaisesRegex(ValueError, "cannot be tested"):
            check.verify_rows(100001)
        nf = 10100
        rng = np.random.default_rng(3)
        x = (rng.normal(size=(300, nf)) * (rng.random((300, nf)) < 0.02)).astype(np.float32)
        y = (x[:, 5] + x[:, 77] + 0.3 * rng.normal(size=300) > 0).astype(int)
        xgb.train(
            {"objective": "binary:logistic", "max_depth": 6, "verbosity": 0, "tree_method": "hist"}, xgb.DMatrix(x, label=y), 40
        ).save_model(str(self.t / "wide.ubj"))
        np.savez(self.t / "wide.npz", X=x[:200])
        # The run itself keeps the 10,100 features that the old code refused, but the limits are scaled 100x down: building a random set at
        # the real limit (100,000,000 values) peaks near 3.4 GB resident, which held a whole test process at the machine's memory cap.
        # At the scaled limit the 100 random rows ask for 400 path rows: 133 sampled nodes, of which 33 fit (1,010,000 values / 10,100
        # features / 3 rows each); the old budget passed the limit and the forest was refused after the pruning and the compile.
        forest = qf.load_forest(qf_formats.model_files(self.t / "wide.ubj"))
        split_nodes = sum(int((t.left != -1).sum()) for m in forest.models for t in m.trees)
        with mock.patch.object(check, "MAX_RANDOM_VALUES", 1_010_000), mock.patch.object(check, "VERIFY_MIN_ROWS", 100):
            self.assertEqual(check.verify_rows(nf), 100)
            self.assertGreater(split_nodes, 4 * 100 // 3)  # more split nodes than the path rows' budget before the width is counted
            path = check.path_rows(forest, 4 * 100, 5)
            self.assertGreater(len(path), 0)
            self.assertLessEqual(path.size, 1_010_000)  # the path rows are held to one test set's limit, as the random rows are
            rep = optimize.optimize(
                self.t / "wide.ubj", self.t / "wide.npz", self.t / "opt-wide", "50%", True, True, min_agreement=1, backend="flat"
            )
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertIsNone(rep["compile_unsupported"])
        self.assertIn("random", [s["set"] for s in rep["stages"]["verify"]["sets"]])
        self.assertTrue(rep["stages"]["verify"]["ok"])

    def test_generated_code_never_deletes_a_folder_that_was_there_before(self):
        """The sources of a build go into a folder of their own making: a user folder named `<output>-src` next to the output stays, with
        its files."""
        try:
            backend_codegen.toolchain_bin()
        except FileNotFoundError as e:
            self.skipTest(str(e))
        work = self.t / "src-clash"
        work.mkdir()
        mine = work / "model-src"
        mine.mkdir()
        (mine / "notes.txt").write_text("keep me")
        compile.compile_model(self.t / "gb.ubj", work / "model.so", backend="codegen")
        self.assertEqual((mine / "notes.txt").read_text(), "keep me")
        self.assertEqual(sorted(p.name for p in work.iterdir()), ["model-src", "model.so"])  # the build's own folder is gone
        info = compile.compile_model(self.t / "gb.ubj", work / "kept.so", backend="codegen", keep_src=True)
        kept = Path(info["src_dir"])
        self.assertTrue((kept / "lib.rs").is_file())
        self.assertEqual(kept.parent, work)
        self.assertTrue((mine / "notes.txt").is_file())

    def test_the_flat_and_generated_forests_take_the_thread_count_the_same_way(self):
        """One helper judges `threads` for both wrappers: not a whole number is a ValueError, a huge or negative one is clamped, never
        passed to ctypes (an ArgumentError for a negative number, a wrapped unsigned for a huge one)."""
        try:
            backend_codegen.toolchain_bin()
        except FileNotFoundError:
            self.skipTest("no rustc")
        art = self.t / "threads.so"
        compile.compile_model(self.t / "gb.ubj", art, backend="codegen")
        compile.compile_model(self.t / "gb.ubj", self.t / "threads.flat", backend="flat")
        forests = [backend_codegen.CodegenForest(art), backends.get("flat").load(self.t / "threads.flat")]
        x = np.zeros((5, forests[0].n_features), np.float32)
        for f in forests:
            for bad in (1.5, "2", [2]):
                with self.assertRaises(ValueError, msg=(type(f).__name__, bad)):
                    f.predict(x, threads=bad)
            want = f.predict(x, threads=1)
            for ok in (-3, 0, 2**40):
                self.assertTrue(np.array_equal(f.predict(x, threads=ok), want), (type(f).__name__, ok))

    def test_the_default_thread_count_is_every_usable_core_from_the_row_cutoff_on(self):
        """`threads=None` (the default of every predict): the processor affinity count (mocked), clamped to 1..64, from
        `evaluator.PARALLEL_MIN_ROWS` rows on, one thread below; an integer is used as given whatever the row count."""
        big, small = evaluator.PARALLEL_MIN_ROWS, evaluator.PARALLEL_MIN_ROWS - 1
        with mock.patch.object(os, "sched_getaffinity", lambda pid: {0, 1, 2}):
            self.assertEqual(evaluator.usable_cpus(), 3)
            self.assertEqual(evaluator.resolve_threads(None, big), 3)
            self.assertEqual(evaluator.resolve_threads(None, small), 1)
            self.assertEqual(evaluator.resolve_threads(None, 1), 1)
            self.assertEqual(evaluator.resolve_threads(1, big), 1)
            self.assertEqual(evaluator.resolve_threads(2, 1), 2)
        with mock.patch.object(os, "sched_getaffinity", lambda pid: set(range(200))):
            self.assertEqual(evaluator.usable_cpus(), 64)  # the cap
            self.assertEqual(evaluator.resolve_threads(None, big), 64)
        with mock.patch.object(os, "sched_getaffinity", lambda pid: set()):
            self.assertEqual(evaluator.usable_cpus(), 1)  # at least one
        with mock.patch.object(os, "sched_getaffinity", mock.Mock(side_effect=OSError)), mock.patch.object(os, "cpu_count", lambda: 6):
            self.assertEqual(evaluator.usable_cpus(), 6)  # no affinity call: the CPU count
        with mock.patch.object(os, "cpu_count", lambda: None):
            with mock.patch.object(os, "sched_getaffinity", mock.Mock(side_effect=OSError)):
                self.assertEqual(evaluator.usable_cpus(), 1)  # an unknown count is one

    def test_a_container_cpu_quota_lowers_the_usable_core_count(self):
        """`docker run --cpus 2` leaves all cores in the affinity mask: the cgroup quota (v2 `cpu.max`, v1 `cfs_quota_us`) caps the threads."""
        d = Path(tempfile.mkdtemp(dir=self.t))
        v2, q1, p1 = d / "cpu.max", d / "quota", d / "period"
        files = lambda *pairs: mock.patch.object(evaluator, "CGROUP_CPU_FILES", tuple(pairs))  # noqa: E731
        with mock.patch.object(os, "sched_getaffinity", lambda pid: set(range(8))):
            for text, want in (
                ("200000 100000", 2),
                ("250000 100000", 3),
                ("50000 100000", 1),
                ("max 100000", 8),
                ("", 8),
                ("x y", 8),
                ("0 100000", 8),
            ):
                v2.write_text(text)
                with files((str(v2), None)):
                    self.assertEqual(evaluator.usable_cpus(), want, text)
            for quota, want in (("400000", 4), ("-1", 8), ("junk", 8)):
                q1.write_text(quota)
                p1.write_text("100000")
                with files((str(q1), str(p1))):
                    self.assertEqual(evaluator.usable_cpus(), want, quota)
            with files((str(d / "missing"), None), (str(d / "missing2"), str(d / "missing3"))):
                self.assertEqual(evaluator.usable_cpus(), 8)  # no cgroup files: the affinity count
            v2.write_text("1000000 100000")  # a quota above the affinity count changes nothing
            with files((str(v2), None)):
                self.assertEqual(evaluator.usable_cpus(), 8)

    def test_both_wrappers_hand_the_resolved_thread_count_to_the_library_and_the_answers_do_not_change(self):
        class Spy:  # records the thread count of every predict call that reaches the library
            def __init__(self, lib):
                self.lib, self.threads = lib, []

            def __getattr__(self, name):
                fn = getattr(self.lib, name)
                if not name.startswith(("tfl_predict", "tfl_cg_predict")):
                    return fn
                at = 3 if name.startswith("tfl_cg_") else 4

                def call(*a):
                    self.threads.append(a[at])
                    return fn(*a)

                return call

        compile.compile_model(self.t / "gb.ubj", self.t / "def.flat", backend="flat")
        forests = [backends.get("flat").load(self.t / "def.flat"), backends.get("wave").load(self.t / "def.flat")]
        try:
            backend_codegen.toolchain_bin()
            compile.compile_model(self.t / "gb.ubj", self.t / "def.so", backend="codegen")
            forests.append(backend_codegen.CodegenForest(self.t / "def.so"))
        except FileNotFoundError:
            pass  # no rustc: the flat family only
        rng = np.random.default_rng(3)
        x = rng.normal(size=(evaluator.PARALLEL_MIN_ROWS + 5, forests[0].n_features)).astype(np.float32)
        x[rng.random(x.shape) < 0.05] = np.nan
        for f in forests:
            name = type(f).__name__
            spy = f._lib = Spy(f._lib)
            with mock.patch.object(os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 4}):
                default = f.predict(x)
                few = f.predict(x[:20])
                forced = f.predict(x, threads=1)
                two = f.predict(x[:20], threads=2)
                f64 = f.predict(x, dtype=np.float64)
            self.assertEqual(spy.threads, [5, 1, 1, 2, 5], name)
            np.testing.assert_array_equal(default, forced, name)  # bit-identical for any thread count
            np.testing.assert_array_equal(few, forced[:20], name)
            np.testing.assert_array_equal(two, forced[:20], name)
            np.testing.assert_array_equal(f64.astype(np.float32), forced, name)
            f._lib = spy.lib

    def test_the_threads_option_of_the_commands_takes_all_or_a_whole_number(self):
        import argparse

        self.assertIsNone(qf.threads_arg("all"))
        self.assertIsNone(qf.threads_arg(" ALL "))
        self.assertEqual(qf.threads_arg("4"), 4)
        self.assertEqual(qf.threads_arg("-1"), -1)  # a number is the evaluator's to clamp
        for bad in ("", "abc", "1.5", "none", "al l"):
            with self.assertRaises(argparse.ArgumentTypeError, msg=bad):
                qf.threads_arg(bad)
        compile.compile_model(self.t / "gb.ubj", self.t / "chk.flat", backend="flat")
        for threads in (None, 1, 3):  # check.run: None passes no thread count (an older backend with threads=1 still works)
            self.assertTrue(check.run(self.t / "gb.ubj", self.t / "chk.flat", n_random=300, threads=threads)["ok"], threads)
        # --speed: the library and the compiled forest run on the SAME number of threads, every usable core by default
        speed = check.run(self.t / "gb.ubj", self.t / "chk.flat", n_random=300, speed=True)["speed"]
        self.assertEqual((speed["compiled_threads"], speed["library_threads"]), (evaluator.usable_cpus(),) * 2)
        self.assertIn(f"{evaluator.usable_cpus()} thread(s)", speed["baseline"])
        for n in (1, 2):
            speed = check.run(self.t / "gb.ubj", self.t / "chk.flat", n_random=300, speed=True, threads=n)["speed"]
            self.assertEqual((speed["compiled_threads"], speed["library_threads"]), (n, n))
        with mock.patch.object(os, "sched_getaffinity", lambda pid: {0, 1, 2}), mock.patch.object(evaluator, "CGROUP_CPU_FILES", ()):
            speed = check.run(self.t / "gb.ubj", self.t / "chk.flat", n_random=300, speed=True)["speed"]
        self.assertEqual((speed["compiled_threads"], speed["library_threads"]), (3, 3))

    def test_a_keep_share_rounds_the_exact_product_half_to_even(self):
        """`--keep 7%` of 150 trees is 10.5 trees, which the documented rule (nearest, a half to the even one) keeps as 10; the product
        in floating point came to 10.500000000000002 and kept 11. The share is computed exactly from its text."""
        from fractions import Fraction

        for text, per_group, want in (
            ("7%", 150, 10),
            ("0.7%", 500, 4),
            ("3.5%", 300, 10),
            ("2.8%", 625, 18),
            ("1.4%", 250, 4),
            ("10%", 300, 30),
            ("12.5%", 8, 1),
            ("37.5%", 8, 3),
            ("0.1%", 100, 1),
            ("1e1%", 100, 10),
            ("250%", 7, 7),
            ("inf%", 7, 7),
            ("30", 100, 30),
            ("2.5", 100, 2),
            ("200", 7, 7),
        ):
            self.assertEqual(prune.parse_keep(text, per_group), want, (text, per_group))
        for k in range(1, 400):  # the rule itself, against exact fractions, for every percentage with one decimal
            for per_group in (150, 200, 500, 625):
                t = f"{k / 10:g}%"
                exact = Fraction(t[:-1]) / 100 * per_group
                self.assertEqual(prune.parse_keep(t, per_group), min(per_group, max(1, round(exact))), (t, per_group))

    def test_the_quickstart_times_all_cores_against_all_cores(self):
        """The quickstart's last line compared xgboost on one thread with `flat.predict`, whose default is now every usable core: it times both
        on the same count (`nthread` = `usable_cpus()`) and says so."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("qf_quickstart", Path(__file__).resolve().parents[1] / "examples" / "quickstart.py")
        quick = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(quick)
        flat = self.t / "quick.flat"
        compile.compile_model(self.t / "gb.ubj", flat, backend="flat")
        booster = xgb.Booster()
        booster.load_model(str(self.t / "gb.ubj"))
        x = np.tile(np.load(self.t / "all.npz")["X"], (2, 1))[: evaluator.PARALLEL_MIN_ROWS + 100]
        with mock.patch.object(os, "sched_getaffinity", lambda pid: {0, 1, 2}), mock.patch.object(evaluator, "CGROUP_CPU_FILES", ()):
            f = backends.get("flat").load(flat)
            try:
                line = quick.speed_line(booster, f, x)
            finally:
                f.close()
        self.assertTrue(line.startswith("all cores (3): xgboost"), line)
        self.assertNotIn("one thread", line)
        self.assertEqual(json.loads(booster.save_config())["learner"]["generic_param"]["nthread"], "3")

    def test_compare_times_the_library_and_the_compiled_forest_on_every_usable_core_by_default(self):
        """`compare --threads` takes N or all (default all): the library and the compiled forest run on the same count, and the table says it."""
        import compare

        a, _ = self.compare_models("threads-default")
        kw = dict(eq_rows=300, n_random=0, speed_rows=300, budget=0.2, log=lambda *_: None)
        want = {None: evaluator.usable_cpus(), 1: 1, 2: 2}
        for threads, n in want.items():
            t = compare.run_table(a, ["flat"], self.t / "all.npz", self.t / f"w-threads-{threads}", threads=threads, **kw)
            self.assertEqual(t["threads"], n, threads)
            self.assertEqual([r["status"] for r in t["rows"]], ["ok"], t["rows"])
        with mock.patch.object(sys, "argv", ["compare", "m", "--data", "d.npz", "--work", "w"]):
            seen = {}
            with mock.patch.object(compare, "run_table", lambda *a_, **k: seen.update(k) or {"rows": [], "threads": 1}):
                with mock.patch.object(compare, "table_md", lambda t: ""):
                    compare.main()
        self.assertIsNone(
            seen.get("threads", "missing")
        )  # the command's default is "all", resolved by the run on the machine that measures

    def test_a_loaded_forest_cannot_be_copied_or_pickled_and_nothing_crashes(self):
        """`copy.copy(FlatForest)` shared the raw handle, so deleting both copies freed it twice (SIGSEGV) and a copy used after the
        original was closed read freed memory. Copy, deepcopy and pickle are refused with a TypeError; the original keeps working."""
        import copy
        import pickle

        compile.compile_model(self.t / "gb.ubj", self.t / "cp.flat", backend="flat")
        forests = [backends.get("flat").load(self.t / "cp.flat"), backends.get("wave").load(self.t / "cp.flat")]
        try:
            backend_codegen.toolchain_bin()
            compile.compile_model(self.t / "gb.ubj", self.t / "cp.so", backend="codegen")
            forests.append(backend_codegen.CodegenForest(self.t / "cp.so"))
        except FileNotFoundError:
            pass
        x = np.zeros((4, forests[0].n_features), np.float32)
        for f in forests:
            want = f.predict(x)
            for attempt in (copy.copy, copy.deepcopy, pickle.dumps, lambda o: pickle.dumps(o, protocol=0)):
                with self.assertRaises(TypeError, msg=(type(f).__name__, attempt)):
                    attempt(f)
            np.testing.assert_array_equal(f.predict(x), want)
            if hasattr(f, "close"):
                f.close()

    def test_the_reported_rustc_peak_is_the_builds_own_not_the_biggest_child_the_process_ever_waited_for(self):
        """`rustc_peak_mb` is the peak of THIS build's child: `RUSAGE_CHILDREN` is the maximum over every child the process ever waited for,
        so the second backend built in a process (compare.build_all, choose) would report the first one's peak."""
        import subprocess

        try:
            backend_codegen.toolchain_bin()
        except FileNotFoundError as e:
            self.skipTest(str(e))
        big = "b = bytearray(700 << 20)\nfor i in range(0, len(b), 4096): b[i] = 1\n"  # a child that touches 700 MB, then exits
        subprocess.run([sys.executable, "-c", big], check=True)
        info = compile.compile_model(self.t / "gb.ubj", self.t / "peak.so", backend="codegen", block=64)
        self.assertGreater(info["rustc_peak_mb"], 20)  # a real figure
        self.assertLess(info["rustc_peak_mb"], 600, info)

    def test_backend_auto_keeps_the_fastest_exact_one(self):
        """`compile --backend auto`: times what can run here on held-out rows, keeps an exact winner, remembers it in OUT.choice.json."""
        pruned = self.t / "pruned-rf"
        if not pruned.is_dir():
            prune.run(self.t / "rf", self.t / "rf.npz", pruned, "10%")
        out = self.t / "auto" / "model.bin"
        out.parent.mkdir(exist_ok=True)
        with self.assertRaises(ValueError):
            compile.compile_model(pruned, out, backend="auto")  # no rows to time on
        info = compile.compile_model(pruned, out, backend="auto", data=str(self.t / "rf.npz"))
        self.assertIn(info["backend"], ("flat", "wave", "wave-avx2", "codegen-block", "codegen-pgo"))
        self.assertEqual(json.loads(Path(f"{out}.choice.json").read_text())["backend"], info["backend"])
        self.assertFalse(list(out.parent.glob(".choose-*")))
        x = check.random_rows(qf.load_forest(qf_formats.model_files(pruned)), 400, 3)
        want = qf.reference_predict(qf_formats.model_files(pruned), x, "output")
        np.testing.assert_allclose(backends.load_artifact(out).predict(x), want, atol=1e-6)
        self.assertIn("| flat |", info["table_md"])
        rep = optimize.optimize(
            self.t / "rf", self.t / "rf.npz", self.t / "opt-auto", "10%", True, True, min_agreement=90.0, backend="auto"
        )
        self.assertTrue(rep["ok"], rep["failures"])
        self.assertTrue(rep["stages"]["verify"]["ok"])
        self.assertIn("Backends timed on held-out rows", (self.t / "opt-auto" / "report.md").read_text())

    def planted(self, good, name, kind, seed):
        """A copy of the compiled file `good` with ONE defect planted in a node and the checksum rewritten (so the loader accepts it)."""
        import struct

        blob = bytearray(good.read_bytes())
        _, ng, nfo, nt, nn = struct.unpack_from("<5I", blob, 12)
        node_at = 64 + nfo * 8 + (nfo * ng * 4 + 7) // 8 * 8 + nt * 8
        words = np.frombuffer(bytes(blob[node_at : node_at + 8 * nn]), "<u8").copy()
        width = (words >> np.uint64(32)) & np.uint64(0xFFFF)  # 0 = a leaf
        rng = np.random.default_rng(seed)
        low = np.uint64(0xFFFFFFFF)

        def f32(w):
            return np.uint32(w & low).view(np.float32)

        if kind == "leaf":
            i = int(rng.choice(np.flatnonzero(width == 0)))
            new = f32(words[i]) * np.float32(1 + 1e-5)
            words[i] = (words[i] & ~low) | np.uint64(np.float32(new).view(np.uint32))
        elif kind == "threshold":
            i = int(rng.choice(np.flatnonzero(width != 0)))
            new = np.nextafter(f32(words[i]), np.float32(np.inf if rng.random() < 0.5 else -np.inf))
            words[i] = (words[i] & ~low) | np.uint64(np.float32(new).view(np.uint32))
        else:  # the default direction of a missing value
            i = int(rng.choice(np.flatnonzero(width != 0)))
            words[i] ^= np.uint64(1) << np.uint64(63)
        blob[node_at : node_at + 8 * nn] = words.astype("<u8").tobytes()
        out = good.with_name(name)
        out.write_bytes(evaluator.seal(blob))
        return out

    def test_the_verify_bar_refuses_an_xgboost_forest_that_differs_from_the_library_by_one_part_in_100000(self):
        """xgboost's answers are claimed identical to the bit, so a leaf that is 1e-5 off (a probability that moves by 3e-7, inside the 1e-6
        bar of the other libraries) must fail the check whenever it changes an answer on the caller's own rows."""
        good = self.t / "bit.flat"
        compile.compile_model(self.t / "rf", good, backend="flat")
        x = np.load(self.t / "rf.npz")["X"]
        base = evaluator.FlatForest(good)
        want = base.predict(x)
        base.close()
        changed = 0
        for seed in range(16):
            bad = self.planted(good, "bit-bad.flat", "leaf", seed)
            ff = evaluator.FlatForest(bad)
            differs = not np.array_equal(ff.predict(x), want)
            ff.close()
            if differs:
                changed += 1
                self.assertFalse(
                    check.run(self.t / "rf", bad, data=self.t / "rf.npz", n_random=2000)["ok"], f"a changed leaf (seed {seed}) passed"
                )
        self.assertGreater(changed, 4, "the planted leaves must change answers, or the test proves nothing")

    def test_the_verify_bar_reaches_the_split_nodes_of_every_tree(self):
        """A threshold one float32 step off, or a flipped default direction, changes an answer only for a row that reaches that node and
        falls on the threshold (or is missing): the check must build such rows for nodes of every tree, not for one random threshold."""
        good = self.t / "path.flat"
        compile.compile_model(self.t / "reg.json", good, backend="flat")
        rng = np.random.default_rng(3)
        many = (rng.normal(size=(100000, NF)) * 2).astype(np.float32)
        many[rng.random(many.shape) < 0.5] = np.nan  # an independent witness that a flipped default can change an answer at all
        ff = evaluator.FlatForest(good)
        want = ff.predict(many)
        ff.close()
        caught = 0
        for kind in ("threshold", "default"):
            for seed in range(12):
                bad = self.planted(good, "path-bad.flat", kind, seed)
                if kind == "default":  # a node no missing value can reach (its feature is split above it) cannot be told apart by any row
                    fb = evaluator.FlatForest(bad)
                    same = np.array_equal(fb.predict(many), want)
                    fb.close()
                    if same:
                        continue
                caught += 1
                self.assertFalse(check.run(self.t / "reg.json", bad, n_random=1500)["ok"], f"{kind} defect (seed {seed}) passed")
        self.assertGreater(caught, 18)

    def test_path_rows_reach_a_sample_of_nodes_from_every_tree(self):
        left, right = np.array([1, 3, 5, -1, -1, -1, -1], np.int32), np.array([2, 4, 6, -1, -1, -1, -1], np.int32)
        feature = np.array([0, 1, 0, -1, -1, -1, -1], np.int32)
        value = np.array([0.5, -1.25, 3.0, 1, 2, 3, 4], np.float32)
        tree = qf.Tree(0, left, right, feature, value, np.array([True, False, True, False, False, False, False]))
        forest = qf.Forest(2, 1, "identity", "output", [qf.ModelIR(np.zeros(1, np.float32), [tree, tree, tree])])
        rows = check.path_rows(forest, 2000, 1)
        self.assertEqual(rows.dtype, np.float32)
        self.assertEqual(rows.shape[1], 2)
        seen = {}
        for r in rows:
            node = 0
            while left[node] != -1:
                v = r[feature[node]]
                kind = (
                    "nan"
                    if np.isnan(v)
                    else "at"
                    if v == value[node]
                    else "below"
                    if v == np.nextafter(value[node], np.float32(-np.inf))
                    else None
                )
                if kind:
                    seen.setdefault(node, set()).add(kind)
                node = (
                    (left[node] if tree.default_left[node] else right[node])
                    if np.isnan(v)
                    else left[node]
                    if v < value[node]
                    else right[node]
                )
        # node 2 sits on the right of the root (x0 >= 0.5) and splits on the same feature: it needs a row on ITS side of the root
        # (a missing x0 goes left at the root, so node 2 below its right side cannot be reached by NaN)
        self.assertEqual({int(k): v for k, v in seen.items()}, {0: {"at", "below", "nan"}, 1: {"at", "below", "nan"}, 2: {"at", "below"}})
        small = check.path_rows(forest, 9, 1)  # fewer rows than nodes: still some rows, never more than asked
        self.assertTrue(0 < len(small) <= 9)

    def test_the_check_rows_try_negative_fractions_at_categorical_nodes_even_when_the_set_is_empty(self):
        """LightGBM truncates a category toward zero, so -1.0, -1.5 and -1.9 are all -1: a negative category, which goes right. Generated code
        once sent x in (-2, -1] left at an empty set and no check row ever had such a value."""
        for cats in (np.array([], np.int64), np.array([3, 5], np.int64)):
            tree = qf.Tree(
                0,
                np.array([1, -1, -1], np.int32),
                np.array([2, -1, -1], np.int32),
                np.array([0, 0, 0], np.int32),
                np.array([0.0, 1.0, 2.0], np.float32),
                np.array([False, False, False]),
                cat={0: cats},
            )
            forest = qf.Forest(1, 1, "identity", "output", [qf.ModelIR(np.float32([0.0]), [tree])], "auto", True)
            rows = check.random_rows(forest, 400, 3)
            self.assertTrue(((rows[:, 0] > -2) & (rows[:, 0] <= -1)).any(), f"set {cats.tolist()}: no random row in (-2, -1]")
            paths = check.path_rows(forest, 30, 1)
            self.assertIn(-1.5, paths[:, 0].tolist(), f"set {cats.tolist()}: no path row at -1.5")
            self.assertLessEqual(len(check.path_rows(forest, 8, 1)), 8)

    def test_a_corrupt_compiled_file_is_refused(self):
        flat = self.t / "wrong.flat"
        compile.compile_model(self.t / "gb.ubj", flat, backend="flat")
        res = check.run(self.t / "gb.ubj", flat, n_random=500)
        self.assertTrue(res["ok"])
        blob = bytearray(flat.read_bytes())
        blob[-9] ^= 0x40  # corrupt a leaf: the checksum must refuse the file
        flat.write_bytes(bytes(blob))
        with self.assertRaises(ValueError):
            check.run(self.t / "gb.ubj", flat, n_random=10)


class DeepTreeTest(unittest.TestCase):
    """Generated code writes a tree as ONE nested if/else expression, and rustc's parser overflows its stack on nesting deeper than about
    1,300-1,500 levels (measured: 1,300 builds, 1,400 and more crash with a backtrace). A deeper tree is refused in one line before rustc runs,
    and `compile --backend auto` leaves the generated-code backends out of the race."""

    @staticmethod
    def spine_forest(depth):
        """One tree that is a chain of `depth` splits (each has a leaf on its left), as a forest."""
        n = 2 * depth + 1
        left, right = np.full(n, -1, np.int32), np.full(n, -1, np.int32)
        for i in range(depth):
            left[2 * i], right[2 * i] = 2 * i + 1, (2 * i + 2 if i < depth - 1 else 2 * depth)
        tree = qf.Tree(0, left, right, np.zeros(n, np.int32), np.arange(n, dtype=np.float32), np.zeros(n, bool))
        return qf.Forest(1, 1, "identity", "output", [qf.ModelIR(np.float32([0.0]), [tree])], "auto", False)

    def test_a_tree_deeper_than_generated_code_can_hold_is_refused_in_one_line_before_rustc(self):
        limit = backend_codegen.MAX_TREE_DEPTH
        for name in ("codegen", "codegen-block", "codegen-pgo"):
            be = backends.get(name)
            with tempfile.TemporaryDirectory() as t, self.assertRaises(ValueError, msg=name) as cm:
                be.compile(self.spine_forest(limit + 1), Path(t) / "x.so", {})
            msg = str(cm.exception)
            self.assertNotIn("\n", msg)
            self.assertIn(f"{limit + 1:,} levels", msg)
            self.assertNotIsInstance(cm.exception, backend_codegen.BuildFailed)  # refused up front, not a rustc crash
        self.assertIsNone(backend_codegen.too_deep(self.spine_forest(limit)))
        self.assertEqual(backend_codegen.too_deep(self.spine_forest(limit + 1)), limit + 1)

    def test_the_automatic_choice_skips_the_generated_code_backends_for_such_a_forest(self):
        import choose

        names, skipped = choose.candidates(self.spine_forest(backend_codegen.MAX_TREE_DEPTH + 1))
        self.assertFalse([n for n in names if n.startswith("codegen")], names)
        self.assertIn("levels", skipped.get("codegen-block", ""))
        names, _ = choose.candidates(self.spine_forest(5))
        self.assertTrue(set(names) & {"flat"}, names)


if __name__ == "__main__":
    unittest.main()
