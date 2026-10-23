"""LightGBM and YDF adapters on synthetic data: the forest read from a model equals the library's own predictions (<= 1e-6,
relative above 1) on seeded random models and rows, the flat file does too where the model is small enough for float32 sums, and
pruning / the regression bar / the pipeline work on them. Each test is capped at about two minutes.

    python3 -s -m unittest discover -s tests -p "test_readers.py"   (a Python with numpy, xgboost, lightgbm; ydf for the YDF tests)
"""

import importlib.util
import json
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import lightgbm as lgb
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import backend_flat
import backends
import check
import compile
import evaluator
import optimize
import prune
import qf_common as qf
import qf_formats

HAS_YDF = importlib.util.find_spec("ydf") is not None
TOL = 1e-6


def data(seed, n=1500, nf=6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, nf)).astype(np.float32)
    x[:, 4] = rng.integers(0, 8, n)
    x[:, 5] = np.where(rng.random(n) < 0.4, 0, x[:, 5])
    x[rng.random((n, nf)) < 0.08] = np.nan
    f = np.nan_to_num(x[:, 0]) * 2 + np.nan_to_num(x[:, 1]) + (np.nan_to_num(x[:, 4]) % 3 == 0) * 2 + np.nan_to_num(x[:, 5])
    return rng, x, f


def make_rows(forest, rng, n=1500):
    x = check.random_rows(forest, n, int(rng.integers(1 << 30)))
    y = rng.normal(size=x.shape).astype(np.float32) * 2
    y[:, 4] = rng.integers(-1, 10, n) + rng.choice([0, 0.5, 0.99], n)  # categories incl. negative and fractional
    y[rng.random(y.shape) < 0.1] = np.nan
    y[rng.random(y.shape) < 0.1] = 0
    return np.concatenate([x, y])


def rel_err(ref, got):
    ref, got = ref.reshape(len(ref), -1), np.asarray(got).reshape(len(ref), -1)
    return float((np.abs(ref - got) / np.maximum(1.0, np.abs(ref))).max())


class LightGBMTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def train(self, seed, params, cat, kind="reg", rounds=15):
        rng, x, f = data(seed)
        y = {"reg": f, "bin": (f > 0.5).astype(int), "multi": (f > 0.5).astype(int) + (f > 2), "pos": np.exp(f / 4)}[kind]
        base = {"num_leaves": 8, "verbose": -1, "min_data_in_leaf": 10, "min_data_per_group": 5, "max_cat_to_onehot": 2, "num_threads": 2}
        b = lgb.train({**base, **params, "seed": seed}, lgb.Dataset(x, y, categorical_feature=cat), rounds)
        path = self.t / f"m{seed}.txt"
        b.save_model(str(path))
        return rng, x, y, path

    def test_forest_equals_lightgbm_on_random_models(self):
        configs = [
            ({"objective": "regression"}, [4], "reg"),
            ({"objective": "regression", "zero_as_missing": True}, [], "reg"),
            ({"objective": "regression", "use_missing": False}, [], "reg"),
            ({"objective": "binary", "sigmoid": 1.7}, [4], "bin"),
            ({"objective": "multiclass", "num_class": 3}, [4], "multi"),
            ({"objective": "regression", "boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1, "feature_fraction": 0.8}, [], "reg"),
            ({"objective": "regression", "boosting": "dart"}, [], "reg"),
            ({"objective": "poisson"}, [], "pos"),
            ({"objective": "huber"}, [], "reg"),
        ]
        t0, worst = time.time(), 0.0
        for seed in range(100, 108):
            for params, cat, kind in configs:
                if time.time() - t0 > 100:
                    break
                rng, _, _, path = self.train(seed, params, cat, kind)
                src = qf_formats.open_source(path)
                forest = qf.Forest(src.spec.n_features, src.spec.groups, src.spec.kind, "output", [src.model_ir()], src.spec.task)
                x = make_rows(forest, rng)
                err = rel_err(src.predict(x), qf.ir_predict(forest, x))
                worst = max(worst, err)
                self.assertLessEqual(err, TOL, (seed, params))
        self.assertLess(worst, TOL)

    def test_flat_file_equals_lightgbm_where_it_can_hold_the_model(self):
        for seed, params, cat, kind in [
            (1, {"objective": "regression", "use_missing": False}, [], "reg"),
            (2, {"objective": "binary"}, [], "bin"),
            (3, {"objective": "multiclass", "num_class": 3}, [], "multi"),
            (4, {"objective": "regression", "boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1}, [], "reg"),
        ]:
            _, _, _, path = self.train(seed, params, cat, kind, rounds=8)
            flat = self.t / f"f{seed}.flat"
            compile.compile_model(path, flat, backend="flat")
            res = check.run(path, flat, n_random=3000)
            self.assertTrue(res["ok"], (seed, res["sets"]))

    def test_the_flat_file_holds_categorical_exp_and_zero_as_missing_models(self):
        _, _, _, cat = self.train(5, {"objective": "regression"}, [4], "reg", rounds=60)
        _, _, _, pos = self.train(6, {"objective": "poisson"}, [], "pos", rounds=60)
        _, _, _, zm = self.train(7, {"objective": "regression", "zero_as_missing": True}, [], "reg", rounds=60)
        for i, path in enumerate((cat, pos, zm)):
            flat = self.t / f"x{i}.flat"
            compile.compile_model(path, flat, backend="flat")
            res = check.run(path, flat, n_random=3000)
            self.assertTrue(res["ok"], (i, res["sets"]))

    def test_margin_averaging_reference_follows_the_forests_margin(self):
        """`reference_predict(average="margin")` must not take LightGBM's raw score, which is the SUM for an rf model and ignores sigmoid:s,
        or check fails a correct compiled forest."""
        rf = {"boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1}
        for seed, params, kind in [
            (11, {"objective": "binary", "sigmoid": 1.7}, "bin"),
            (12, {"objective": "binary", "sigmoid": 0.6, **rf}, "bin"),
            (13, {"objective": "tweedie", **rf}, "pos"),
            (14, {"objective": "multiclass", "num_class": 3, **rf}, "multi"),
        ]:
            _, _, _, path = self.train(seed, params, [], kind, rounds=8)
            flat = self.t / f"avg{seed}.flat"
            compile.compile_model(path, flat, average="margin", backend="flat")
            res = check.run(path, flat, average="margin", n_random=2000)
            self.assertTrue(res["ok"], (seed, res["sets"]))

    def test_a_base_margin_is_added_to_the_raw_score_before_the_sigmoid_scale(self):
        """`sigmoid:s` is folded into the leaves, so a base margin added to the scaled score gave sigma(s*raw + m) where LightGBM (and
        REFERENCE.md "Base margin") say sigma(s*(raw + m)): off by 0.36 in probability at s = 1.7. The margin is now scaled by s in the
        flat file's reader, in the generated code, and in the reference and the numpy evaluator the harness checks against."""
        import backend_codegen
        import compile

        # (seed, s, extra LightGBM parameters, compile options): plain leaves, s = 1 (no scale to carry), linear leaves and exact (f64)
        # leaves, which are the files whose header bytes the scale cannot use
        for seed, s, extra, opts in (
            (41, 1.7, {}, {}),
            (42, 0.6, {}, {}),
            (43, 1.0, {}, {}),
            (44, 1.7, {"linear_tree": True}, {}),
            (45, 0.6, {}, {"f64_leaves": True}),
        ):
            _, x, f = data(seed)
            m = 0.8 * np.nan_to_num(x[:, 3]).astype(np.float64)  # the margin of a base model
            y = (f + m > 0.5).astype(int)
            params = {
                "objective": "binary",
                "sigmoid": s,
                "num_leaves": 8,
                "verbose": -1,
                "min_data_in_leaf": 10,
                "num_threads": 2,
                "seed": seed,
                **extra,
            }
            b = lgb.train(params, lgb.Dataset(x, y, init_score=m), 15)
            path = self.t / f"sm{seed}.txt"
            b.save_model(str(path))
            want = 1.0 / (1.0 + np.exp(-s * (b.predict(x.astype(np.float64), raw_score=True) + m)))
            flat = self.t / f"sm{seed}.flat"
            compile.compile_model(path, flat, backend="flat", **opts)
            if extra or opts:  # these are linear-format files: the header bytes the scale uses elsewhere hold their records
                self.assertEqual(int.from_bytes(flat.read_bytes()[8:12], "little"), backend_flat.FORMAT_LINEAR, (seed, extra, opts))
            pred = evaluator.FlatForest(flat)
            self.assertEqual(pred.margin_scale, s, (seed, s))
            got = pred.predict(x, base_margin=m, dtype=np.float64)
            self.assertLess(float(np.abs(got - want).max()), 1e-6, (seed, s))
            self.assertLess(rel_err(pred.predict(x, base_margin=m), want), 1e-6, (seed, s))
            bm = m.reshape(-1, 1)
            self.assertLess(rel_err(qf.reference_predict([path], x, "output", bm), want), 1e-6, (seed, s, "reference"))
            forest = qf.load_forest([path])
            self.assertLess(rel_err(qf.ir_predict(forest, x, base_margin=bm), want), 1e-6, (seed, s, "numpy evaluator"))
            res = check.run(path, flat, n_random=1500, margin=True)
            self.assertTrue(res["ok"], (seed, res["sets"]))
            try:
                backend_codegen.toolchain_bin()
            except FileNotFoundError:
                pass  # no Rust toolchain here: the flat file, the reference and the numpy evaluator were checked
            else:
                out = self.t / f"sm{seed}.so"
                backends.get("codegen-block").compile(forest, out, {})
                cg = backends.get("codegen-block").load(out)
                self.assertLess(rel_err(cg.predict(x, base_margin=m), want), 1e-6, (seed, s, "generated code"))
            pred.close()

    def test_a_base_margin_with_two_models_of_different_sigmoid_scales_is_refused_not_guessed(self):
        """A file that averages models with different `sigmoid:s` has no single factor for the margin: it still evaluates without one."""
        import compile

        for k, s in enumerate((1.7, 0.6)):
            _, x, f = data(50 + k)
            b = lgb.train(
                {"objective": "binary", "sigmoid": s, "num_leaves": 6, "verbose": -1, "num_threads": 2},
                lgb.Dataset(x, (f > 0.5).astype(int)),
                6,
            )
            folder = self.t / "mixed"
            folder.mkdir(exist_ok=True)
            b.save_model(str(folder / f"fold{k}.txt"))
        flat = self.t / "mixed.flat"
        compile.compile_model(self.t / "mixed", flat, backend="flat")
        pred = evaluator.FlatForest(flat)
        self.assertEqual(pred.predict(x).shape, (len(x),))
        with self.assertRaisesRegex(ValueError, "sigmoid"):
            pred.predict(x, base_margin=np.zeros(len(x)))
        pred.close()

    def test_the_equality_harness_sees_a_wrong_category_set_without_any_data(self):
        """The random rows cover every category, not category 0 only: a flat file with wrong category sets must not pass when no data is
        given."""
        import struct

        _, _, _, path = self.train(5, {"objective": "regression"}, [4], "reg", rounds=30)
        self.assertIn("categorical", qf.load_forest([path]).requires)
        flat = self.t / "cat-ok.flat"
        compile.compile_model(path, flat, backend="flat")
        self.assertTrue(check.run(path, flat, n_random=3000)["ok"])
        b = bytearray(flat.read_bytes())
        words = struct.unpack_from("<I", b, 52)[0]
        at = len(b) - (words * 4 + 7) // 8 * 8
        i = 0
        while i < words:  # the table is [n, word 0 .. word n-1] per set: flip every category but 0, keep the file valid
            n = struct.unpack_from("<I", b, at + 4 * i)[0]
            for j in range(1, n + 1):
                struct.pack_into("<I", b, at + 4 * (i + j), struct.unpack_from("<I", b, at + 4 * (i + j))[0] ^ 0xFFFFFFFE)
            i += 1 + n
        bad = self.t / "cat-bad.flat"
        bad.write_bytes(evaluator.seal(b))  # the checksum put right, so the damaged category sets are what the check has to find
        self.assertFalse(check.run(path, bad, n_random=3000)["ok"])

    def test_engines_without_the_extensions_refuse_the_model_clearly(self):
        _, _, _, cat = self.train(5, {"objective": "regression"}, [4], "reg")
        for backend in ("wave", "wave-avx2"):
            with self.assertRaisesRegex(ValueError, "categorical"):
                compile.compile_model(cat, self.t / "x.flat", backend)

    def test_prune_report_and_the_regression_bar(self):
        rng, x, f = data(11, n=4000)
        y = (f + rng.normal(scale=0.3, size=len(f))).astype(np.float64)
        path = self.t / "big.txt"
        lgb.train(
            {"objective": "regression", "num_leaves": 15, "verbose": -1, "num_threads": 2}, lgb.Dataset(x[:2000], y[:2000]), 120
        ).save_model(str(path))
        np.savez(self.t / "sel.npz", X=x, y=y, group=np.arange(len(x)))
        r = prune.run(path, self.t / "sel.npz", self.t / "pr", "50%", regression=qf.RegBar("log", 50.0, 500.0, 1.0))
        self.assertTrue(r["regression"]["ok"], prune.summary(r))
        self.assertTrue(r["regression"]["error_checked"])
        pruned = qf_formats.open_source(self.t / "pr" / "big.txt")  # a normal LightGBM model with half the trees
        self.assertEqual(len(pruned.trees), 60)
        strict = prune.run(path, self.t / "sel.npz", self.t / "pr2", "10%", regression=qf.RegBar("log", 0.001, 0.01))
        self.assertFalse(strict["regression"]["ok"])
        out = optimize.optimize(path, self.t / "sel.npz", self.t / "opt-no", "10%", regression=qf.RegBar("log", 0.001, 0.01))
        self.assertFalse(out["ok"])
        self.assertEqual(sorted(p.name for p in (self.t / "opt-no").iterdir()), ["report.json", "report.md"])

    def test_binary_and_multiclass_prune_by_decision_agreement(self):
        for params, kind, name in (({"objective": "binary"}, "bin", "b"), ({"objective": "multiclass", "num_class": 3}, "multi", "m")):
            _, x, _, path = self.train(21, params, [], kind, rounds=40)
            np.savez(self.t / f"{name}.npz", X=x, group=np.arange(len(x)))
            r = prune.run(path, self.t / f"{name}.npz", self.t / f"pr-{name}", "50%")
            self.assertGreater(r["heldout"]["pct"], 60)
            self.assertEqual(len(qf_formats.open_source(self.t / f"pr-{name}" / path.name).trees), r["models"][0]["kept_trees"])


class OutputFieldsTest(unittest.TestCase):
    """Every saved-model field that changes what `predict` returns is read exactly or refused with one line, never a silent wrong answer
    (including the YDF `apply_link_function` field). One case per field: field -> (read | refused)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        rng = np.random.default_rng(11)
        cls.x = rng.normal(size=(600, 5)).astype(np.float32)
        cls.x[rng.random(cls.x.shape) < 0.05] = np.nan
        f = np.nan_to_num(cls.x[:, 0]) * 2 + np.nan_to_num(cls.x[:, 1])
        cls.y = {
            "reg": f + 3,
            "bin": (f > 0.5).astype(int),
            "multi": (f > 0.5).astype(int) + (f > 2),
            "pos": np.exp(f / 4).round(),
            "pos_f": np.exp(f / 4) + 0.1,
            "prob": np.clip(f / 6 + 0.5, 0, 1),
            "zero": np.zeros(len(f)),
        }

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check_cases(self, cases):
        """cases: (field, expected, saved model path); expected is "read" (equal to the library's predict) or "refused"."""
        for field, expected, path in cases:
            if expected == "refused":
                with self.assertRaises(ValueError, msg=field) as cm:
                    qf_formats.open_source(path)
                self.assertNotIn("\n", str(cm.exception), field)
                continue
            src = qf_formats.open_source(path)
            forest = qf.Forest(src.spec.n_features, src.spec.groups, src.spec.kind, "output", [src.model_ir()], src.spec.task)
            self.assertLessEqual(rel_err(src.predict(self.x[:300]), qf.ir_predict(forest, self.x[:300])), TOL, field)

    def test_xgboost_objectives_base_score_and_tree_parameters(self):
        import xgboost as xgb

        cases = []

        def add(field, expected, params, kind, rounds=3, **dm):
            b = xgb.train({"max_depth": 3, "eta": 0.5, **params}, xgb.DMatrix(self.x, self.y[kind], **dm), rounds)
            path = self.t / f"x{len(cases)}.json"
            b.save_model(str(path))
            cases.append((field, expected, path))

        for obj, kind, extra in [
            ("reg:squarederror", "reg", {}),
            ("reg:logistic", "prob", {}),
            ("binary:logistic", "bin", {}),
            ("multi:softprob", "multi", {"num_class": 3}),
            ("multi:softmax", "multi", {"num_class": 3}),
            ("reg:squaredlogerror", "pos_f", {}),
            ("reg:pseudohubererror", "reg", {}),
            ("reg:absoluteerror", "reg", {}),
            ("count:poisson", "pos", {}),
            ("reg:gamma", "pos_f", {}),
            ("reg:tweedie", "pos_f", {}),
            ("binary:logitraw", "bin", {}),
            ("binary:hinge", "bin", {}),
            ("rank:pairwise", "bin", {}),
            ("reg:quantileerror", "reg", {"quantile_alpha": 0.3}),
        ]:
            ok = obj in ("reg:squarederror", "reg:logistic", "binary:logistic", "multi:softprob")
            add(f"objective {obj}", "read" if ok else "refused", {"objective": obj, **extra}, kind)
        # a softmax over one logit is a constant 1.0: the compiled file would be one the loader refuses ("counts out of range")
        add("multi:softprob with num_class=1 (one class)", "refused", {"objective": "multi:softprob", "num_class": 1}, "zero")
        add("base_score 0.2 (logistic)", "read", {"objective": "binary:logistic", "base_score": 0.2}, "bin")
        add("base_score beyond the clip (logistic)", "read", {"objective": "binary:logistic", "base_score": 0.9999999}, "bin")
        add("base_score 7 (squared error)", "read", {"objective": "reg:squarederror", "base_score": 7.0}, "reg")
        add("base_score 0.3 (softprob)", "read", {"objective": "multi:softprob", "num_class": 3, "base_score": 0.3}, "multi")
        add("training base_margin", "read", {"objective": "reg:squarederror"}, "reg", base_margin=np.full(len(self.x), 5.0, np.float32))
        add("num_parallel_tree (rf)", "read", {"objective": "binary:logistic", "num_parallel_tree": 5, "subsample": 0.7}, "bin", rounds=1)
        add("num_parallel_tree (boosted rf)", "read", {"objective": "multi:softprob", "num_class": 3, "num_parallel_tree": 4}, "multi")
        add("scale_pos_weight", "read", {"objective": "binary:logistic", "scale_pos_weight": 5}, "bin")
        add("booster dart", "refused", {"objective": "reg:squarederror", "booster": "dart", "rate_drop": 0.3}, "reg")
        add("booster gblinear", "refused", {"objective": "reg:squarederror", "booster": "gblinear"}, "reg")
        two = np.stack([self.y["reg"], self.y["reg"] * 2], 1)
        for strategy in ("multi_output_tree", "one_output_per_tree"):
            b = xgb.train(
                {"objective": "reg:squarederror", "tree_method": "hist", "multi_strategy": strategy, "max_depth": 3},
                xgb.DMatrix(self.x, two),
                2,
            )
            b.save_model(str(self.t / f"x_{strategy}.json"))
            cases.append((f"two targets, {strategy}", "refused", self.t / f"x_{strategy}.json"))
        self.check_cases(cases)

    def test_xgboost_dart_is_refused_by_what_the_file_says_under_every_layout(self):
        """Hand-made JSON, so the result does not depend on the installed xgboost. xgboost 3.2 writes a dart model as booster name "dart" with
        `weight_drop` beside "gbtree"; 3.4 writes name "gbtree" with `weight_drop` inside "model" (booster=dart is deprecated, dropout is a
        gbtree parameter): read as a plain gbtree it answered wrongly (error 1.4 on a range of 2.2, measured on xgboost 3.4.1)."""
        tree = {
            "left_children": [1, -1, -1],
            "right_children": [2, -1, -1],
            "split_indices": [0, 0, 0],
            "split_conditions": [0.5, -1.0, 2.0],
            "default_left": [0, 0, 0],
            "split_type": [0, 0, 0],
            "categories": [],
            "tree_param": {"num_nodes": "3", "size_leaf_vector": "1"},
        }

        def layout(name, weight_drop_at=None):
            model = {"trees": [tree, tree], "tree_info": [0, 0], "gbtree_model_param": {"num_trees": "2", "num_parallel_tree": "1"}}
            gb = {"name": name, "model": model}
            if weight_drop_at == "model":
                model["weight_drop"] = [0.5, 1.0]
            if weight_drop_at == "booster":
                gb = {"name": name, "gbtree": {"model": model, "name": "gbtree"}, "weight_drop": [0.5, 1.0]}
            return {
                "version": [3, 4, 1],
                "learner": {
                    "objective": {"name": "reg:squarederror"},
                    "learner_model_param": {"base_score": "[5E-1]", "num_class": "0", "num_feature": "1"},
                    "gradient_booster": gb,
                },
            }

        for field, raw, expected in [
            ("plain gbtree", layout("gbtree"), "read"),
            ("3.2 dart: name dart, weight_drop beside gbtree", layout("dart", "booster"), "refused"),
            ("3.4 dart: name gbtree, weight_drop inside model", layout("gbtree", "model"), "refused"),
            ("name dart, weight_drop inside model", layout("dart", "model"), "refused"),
            ("name dart without weights", layout("dart"), "refused"),
        ]:
            if expected == "read":
                qf.parse_model(raw)
                continue
            with self.assertRaises(ValueError, msg=field) as cm:
                qf.parse_model(raw)
            self.assertNotIn("\n", str(cm.exception), field)
            self.assertIn("dart", str(cm.exception), field)
            path = self.t / "hand_dart.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError, msg=field):
                qf_formats.open_source(path)

    def test_a_ubjson_file_with_a_weight_drop_key_is_refused_before_the_library_rewrites_it(self):
        """An xgboost older than the writer reloads a 3.4 dart `.ubj` without its weights and would save a plain gbtree: the key is
        looked for in the bytes (hand-made, no library involved)."""
        path = self.t / "hand_dart.ubj"
        for width in (b"i", b"U", b"I\x00", b"l" + bytes(3), b"L" + bytes(7)):  # xgboost writes the length as an 8-byte L
            path.write_bytes(b"{" + width + b"\x07learner{" + width + b"\x0bweight_drop[$d#U\x02" + bytes(8) + b"}}")
            with self.assertRaises(ValueError, msg=width) as cm:
                qf_formats.open_source(path)
            self.assertIn("dart", str(cm.exception))
            self.assertNotIn("\n", str(cm.exception))

    def test_lightgbm_objectives_and_header_fields(self):
        cases = []

        def add(field, expected, params, kind, rounds=8, **ds):
            b = lgb.train(
                {"num_leaves": 7, "verbose": -1, "min_data_in_leaf": 10, "seed": 1, **params},
                lgb.Dataset(self.x, self.y[kind], **ds),
                rounds,
            )
            path = self.t / f"l{len(cases)}.txt"
            b.save_model(str(path))
            cases.append((field, expected, path))

        bag = {"boosting": "rf", "bagging_fraction": 0.7, "bagging_freq": 1}
        for obj, kind, extra, ok in [
            ("regression", "reg", {}, True),
            ("regression_l1", "reg", {}, True),
            ("huber", "reg", {}, True),
            ("fair", "reg", {}, True),
            ("quantile", "reg", {"alpha": 0.3}, True),
            ("mape", "pos_f", {}, True),
            ("poisson", "pos", {}, True),
            ("gamma", "pos_f", {}, True),
            ("tweedie", "pos_f", {}, True),
            ("binary", "bin", {}, True),
            ("cross_entropy", "prob", {}, True),
            ("multiclass", "multi", {"num_class": 3}, True),
            ("cross_entropy_lambda", "prob", {}, False),
            ("multiclassova", "multi", {"num_class": 3}, False),
        ]:
            add(f"objective {obj}", "read" if ok else "refused", {"objective": obj, **extra}, kind)
        add("regression with reg_sqrt", "refused", {"objective": "regression", "reg_sqrt": True}, "pos_f")
        add("binary sigmoid:2.5", "read", {"objective": "binary", "sigmoid": 2.5}, "bin")
        add("average_output (binary rf)", "read", {"objective": "binary", **bag}, "bin")
        add("average_output with sigmoid:2", "read", {"objective": "binary", "sigmoid": 2.0, **bag}, "bin")
        add("average_output (multiclass rf)", "read", {"objective": "multiclass", "num_class": 3, **bag}, "multi")
        add("average_output with an exp link (poisson rf)", "read", {"objective": "poisson", **bag}, "pos")
        add("average_output (regression rf)", "read", {"objective": "regression", **bag}, "reg")
        add("boosting dart", "read", {"objective": "binary", "boosting": "dart"}, "bin")
        add("boosting goss", "read", {"objective": "regression", "boosting": "goss"}, "reg")
        add("linear_tree (regression)", "read", {"objective": "regression", "linear_tree": True}, "reg")
        add("linear_tree (binary)", "read", {"objective": "binary", "linear_tree": True}, "bin")
        add("boost_from_average off", "read", {"objective": "binary", "boost_from_average": False}, "bin")
        add("init_score in training", "read", {"objective": "regression"}, "reg", init_score=np.full(len(self.x), 3.0))
        add("poisson_max_delta_step", "read", {"objective": "poisson", "poisson_max_delta_step": 0.1}, "pos")
        add("tweedie_variance_power", "read", {"objective": "tweedie", "tweedie_variance_power": 1.2}, "pos_f")
        self.check_cases(cases)


class LightGbmEdgeThresholdTest(unittest.TestCase):
    def test_a_threshold_of_minus_float32_max_sends_the_row_minus_float32_max_left(self):
        """`x <= t` with t exactly -FLT_MAX: the largest float32 <= t is -FLT_MAX itself, not -infinity. A model made by hand or by another
        tool can carry such a threshold; LightGBM's own predictor sends the row -FLT_MAX left, the compiled forest sent it right."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 2)).astype(np.float32)
        y = (x[:, 0] > 0) * 3.0 + rng.normal(size=400) * 0.1
        params = dict(objective="regression", num_leaves=2, verbose=-1, num_threads=1, min_data_in_leaf=5)
        text = lgb.train(params, lgb.Dataset(x, y), 1).model_to_string()
        fmax = float(np.finfo(np.float32).max)
        text = re.sub(r"(?m)^threshold=\S+$", f"threshold={-fmax!r}", text, count=1)
        data_ = text.encode()  # tree_sizes follow the new length of the line
        end = data_.find(b"end of trees")
        starts = [m.start() for m in re.finditer(rb"(?m)^Tree=\d+$", data_[:end])]
        sizes = " ".join(str(b - a) for a, b in zip(starts, [*starts[1:], end], strict=True))
        text = re.sub(r"(?m)^tree_sizes=.*$", f"tree_sizes={sizes}", text, count=1)
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "m.txt"
            model.write_text(text)
            rows = np.array([[-fmax, 0.0], [np.nextafter(np.float32(-fmax), np.float32(0)), 0.0], [0.5, 0.0]], np.float32)
            expected = lgb.Booster(model_file=str(model)).predict(rows.astype(np.float64))
            compile.compile_model(model, Path(tmp) / "m.flat", backend="flat")
            got = evaluator.FlatForest(Path(tmp) / "m.flat").predict(rows, dtype=np.float64)
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)
        self.assertNotAlmostEqual(
            float(expected[0]), float(expected[1]), places=3
        )  # the two rows really differ, so the test sees a wrong side

    def test_a_decision_type_with_missing_type_three_or_a_stray_bit_is_refused(self):
        """decision_type holds the categorical bit, the default-left bit and a missing type 0-2 in bits 2-3. A value with missing type 3 was read
        as another type here while LightGBM treats it as None (259 of 500 rows differed): the reader refuses what LightGBM never writes."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=(300, 3)).astype(np.float32)
        y = x[:, 0] * 2 + rng.normal(size=300) * 0.1
        params = dict(objective="regression", num_leaves=5, verbose=-1, num_threads=1, min_data_in_leaf=5)
        text = lgb.train(params, lgb.Dataset(x, y), 2).model_to_string()
        for name, edit in (("missing type 3", lambda v: v | 12), ("a bit beyond the fourth", lambda v: v | 16)):
            changed = re.sub(
                r"(?m)^decision_type=(.*)$",
                lambda m, edit=edit: "decision_type=" + " ".join(str(edit(int(v))) for v in m.group(1).split()),
                text,
            )
            end = changed.find("end of trees")
            starts = [m.start() for m in re.finditer(r"(?m)^Tree=\d+$", changed[:end])]
            sizes = " ".join(str(b - a) for a, b in zip(starts, [*starts[1:], end], strict=True))
            changed = re.sub(r"(?m)^tree_sizes=.*$", f"tree_sizes={sizes}", changed, count=1)
            with tempfile.TemporaryDirectory() as tmp:
                model = Path(tmp) / "m.txt"
                model.write_text(changed)
                with self.assertRaises(ValueError, msg=name) as cm:
                    qf_formats.open_source(model)
                self.assertIn("decision_type", str(cm.exception), name)
                self.assertNotIn("\n", str(cm.exception), name)
            with tempfile.TemporaryDirectory() as tmp:  # the unchanged model still reads
                model = Path(tmp) / "m.txt"
                model.write_text(text)
                qf_formats.open_source(model)


@unittest.skipUnless(HAS_YDF, "ydf is not installed")
class YdfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_forest_equals_ydf_on_random_models(self):
        import ydf

        R, C = ydf.Task.REGRESSION, ydf.Task.CLASSIFICATION
        rf, gbt = ydf.RandomForestLearner, ydf.GradientBoostedTreesLearner
        t0 = time.time()
        for seed in range(5):
            rng, x, f = data(300 + seed, n=1200, nf=6)
            d = {f"f{i}": x[:, i] for i in range(6)}
            yb, ym = (f > 1).astype(int).astype(str), ((f > 1).astype(int) + (f > 4)).astype(str)
            for tag, learner, task, lab, kw in [
                ("rf_reg", rf, R, f, {}),
                ("rf_bin", rf, C, yb, {}),
                ("rf_bin_avg", rf, C, yb, {"winner_take_all": False}),
                ("rf_multi", rf, C, ym, {}),
                ("gbt_reg", gbt, R, f, {}),
                ("gbt_bin", gbt, C, yb, {}),
                ("gbt_multi", gbt, C, ym, {}),
                ("gbt_pois", gbt, R, np.exp(f / 4), {"loss": "POISSON"}),
            ]:
                if time.time() - t0 > 100:
                    return
                m = learner(label="y", task=task, num_trees=12, max_depth=5, min_examples=5, **kw).train({**d, "y": lab}, verbose=0)
                path = self.t / f"{tag}{seed}"
                m.save(str(path))
                src = qf_formats.open_source(path)
                forest = qf.Forest(src.spec.n_features, src.spec.groups, src.spec.kind, "output", [src.model_ir()], src.spec.task)
                xt = make_rows(forest, rng)[:, :6]
                xt = np.ascontiguousarray(xt)
                self.assertLessEqual(rel_err(src.predict(xt), qf.ir_predict(forest, xt)), TOL, (tag, seed))

    def test_a_gbt_saved_without_its_link_function_is_refused_not_compiled_as_probabilities(self):
        """`apply_link_function=False` (binary, multiclass, focal and Poisson): `predict` answers the margin, so a reader that takes the
        output kind from the loss line alone compiled a forest that answered probabilities. The field is read from the model (the
        logits switch of a classifier; for Poisson, which has no switch, one probe row against exp(margin)); with the link on, the same
        models are read exactly."""
        import ydf

        _, x, f = data(405, n=900, nf=6)
        d = {f"f{i}": x[:, i] for i in range(6)}
        yb, ym = (f > 1).astype(int).astype(str), ((f > 1).astype(int) + (f > 4)).astype(str)
        yp = np.random.default_rng(2).poisson(np.exp(np.nan_to_num(f) / 4)).astype(np.float32)
        C, R = ydf.Task.CLASSIFICATION, ydf.Task.REGRESSION
        for tag, task, label, extra in [
            ("bin", C, yb, {}),
            ("multi", C, ym, {}),
            ("focal", C, yb, {"loss": "BINARY_FOCAL_LOSS"}),
            ("pois", R, yp, {"loss": "POISSON"}),
        ]:
            for link in (True, False):
                m = ydf.GradientBoostedTreesLearner(
                    label="y", task=task, num_trees=8, max_depth=4, apply_link_function=link, **extra
                ).train({**d, "y": label}, verbose=0)
                path = self.t / f"link_{tag}_{link}"
                m.save(str(path))
                if link:
                    src = qf_formats.open_source(path)
                    forest = qf.Forest(src.spec.n_features, src.spec.groups, src.spec.kind, "output", [src.model_ir()], src.spec.task)
                    self.assertLessEqual(rel_err(src.predict(x), qf.ir_predict(forest, x)), TOL, tag)
                    continue
                with self.assertRaisesRegex(ValueError, "apply_link_function"):
                    qf_formats.open_source(path)
                with self.assertRaisesRegex(ValueError, "apply_link_function"):
                    compile.compile_model(path, self.t / f"link_{tag}.flat", backend="flat")
                self.assertFalse((self.t / f"link_{tag}.flat").exists())

    def test_a_poisson_gbt_whose_rates_reach_ydfs_own_clamp_is_read_and_one_saved_without_the_link_is_still_refused(self):
        """YDF clamps the Poisson logit to [-19, 19] before it exponentiates, so a model with the link function ON answers exp(clip(margin))
        and not exp(margin) on rates above 1.78e8 or below 5.6e-9: it is read, not refused as "saved with apply_link_function=False".
        The probe is symmetric (a relative tolerance on both sides), and a model saved without the link (it answers the margin) is refused
        at any margin, large, small or shifted either way."""
        import ydf

        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 4)).astype(np.float32)
        d = {f"f{i}": x[:, i] for i in range(4)}
        counts = rng.poisson(np.exp(0.5 * x[:, 0] + 1) * 1e8).astype(float)  # a mean of about 2.7e8: rates above e^19
        small = rng.poisson(np.exp(0.5 * x[:, 0] + 1)).astype(float)
        for tag, y, shift in [
            ("big rates", counts, None),
            ("shifted +30", small, 30.0),
            ("shifted -30", small, -30.0),
            ("ordinary", small, None),
        ]:
            for link in (True, False):
                m = ydf.GradientBoostedTreesLearner(
                    label="y", task=ydf.Task.REGRESSION, loss="POISSON", num_trees=12, max_depth=3, num_threads=1, apply_link_function=link
                ).train({**d, "y": y}, verbose=0)
                if shift is not None:
                    m.set_initial_predictions([shift])
                path = self.t / f"clamp_{tag.replace(' ', '_')}_{link}"
                m.save(str(path))
                if link:
                    src = qf_formats.open_source(path)
                    self.assertEqual(src.spec.kind, "exp", tag)
                    continue
                with self.assertRaisesRegex(ValueError, "apply_link_function", msg=tag):
                    qf_formats.open_source(path)

    def test_the_margin_prediction_gives_the_models_own_logits_switch_back(self):
        """`predict(margin=True)` flips the model's logits switch; it must leave it as it found it (not force it to False)."""
        import ydf

        _, x, f = data(406, n=600, nf=6)
        d = {f"f{i}": x[:, i] for i in range(6)}
        m = ydf.GradientBoostedTreesLearner(label="y", task=ydf.Task.CLASSIFICATION, num_trees=6, max_depth=3).train(
            {**d, "y": (f > 1).astype(int).astype(str)}, verbose=0
        )
        m.save(str(self.t / "restore"))
        src = qf_formats.open_source(self.t / "restore")
        for was in (False, True):
            src.model.set_output_logits(was)
            src.predict(x[:20], margin=True)
            self.assertEqual(src.model.output_logits(), was)

    def test_prune_counts_the_librarys_trees_for_a_ydf_classifier_with_three_classes(self):
        """A YDF random-forest classification tree is one forest tree per class: the report's trees, kept_trees and node counts are the
        library's own (its API counts them), and agree with kept_indices, not 3 times too many."""
        import ydf

        def count(model):
            trees = model.get_all_trees()

            def nodes(t):
                stack, k = [t.root], 0
                while stack:
                    nd = stack.pop()
                    k += 1
                    if not nd.is_leaf:
                        stack += [nd.pos_child, nd.neg_child]
                return k

            return len(trees), sum(nodes(t) for t in trees)

        _, x, f = data(407, n=900, nf=6)
        d = {f"f{i}": x[:, i] for i in range(6)}
        np.savez(self.t / "counts-sel.npz", X=x)
        y3, y2 = np.digitize(np.nan_to_num(f), [0.0, 3.0]), (np.nan_to_num(f) > 1).astype(int)
        for tag, learner, y in [
            ("rf3", ydf.RandomForestLearner, y3),
            ("rf2", ydf.RandomForestLearner, y2),
            ("gbt3", ydf.GradientBoostedTreesLearner, y3),
        ]:
            m = learner(label="y", task=ydf.Task.CLASSIFICATION, num_trees=21 if tag != "gbt3" else 8, max_depth=4).train(
                {**d, "y": y.astype(str)}, verbose=0
            )
            path = self.t / f"counts-{tag}"
            m.save(str(path))
            r = prune.run(path, self.t / "counts-sel.npz", self.t / f"counts-out-{tag}", "25%")
            row, ki = r["models"][0], r["kept_indices"][path.name]
            (t0, n0), (t1, n1) = count(m), count(ydf.load_model(str(self.t / f"counts-out-{tag}" / path.name)))
            self.assertEqual((row["trees"], row["kept_trees"], row["nodes_before"], row["nodes_after"]), (t0, t1, n0, n1), tag)
            self.assertEqual((row["trees"], row["kept_trees"]), (ki["original_trees"], len(ki["flat"])), tag)
            self.assertEqual((r["totals"]["nodes_before"], r["totals"]["nodes_after"]), (n0, n1), tag)

    def test_ydf_losses_and_link_fields_are_read_exactly_or_refused(self):
        import ydf

        _, x, f = data(408, n=700, nf=6)
        d = {f"f{i}": x[:, i] for i in range(6)}
        f = np.nan_to_num(f)
        yb, ym = (f > 1).astype(int).astype(str), ((f > 1).astype(int) + (f > 4)).astype(str)
        yp = np.random.default_rng(3).poisson(np.exp(f / 4)).astype(np.float32)
        C, R = ydf.Task.CLASSIFICATION, ydf.Task.REGRESSION
        G, RF = ydf.GradientBoostedTreesLearner, ydf.RandomForestLearner
        cases = [
            ("gbt squared error", "read", G, R, f, {}),
            ("gbt absolute error", "read", G, R, f, {"loss": "MEAN_AVERAGE_ERROR"}),
            ("gbt binomial", "read", G, C, yb, {}),
            ("gbt binary focal", "read", G, C, yb, {"loss": "BINARY_FOCAL_LOSS"}),
            ("gbt multinomial", "read", G, C, ym, {}),
            ("gbt poisson", "read", G, R, yp, {"loss": "POISSON"}),
            ("gbt regression apply_link_function=False (no link to apply)", "read", G, R, f, {"apply_link_function": False}),
            ("gbt l2 regularization and shrinkage", "read", G, R, f, {"l2_regularization": 5.0, "shrinkage": 0.3}),
            ("gbt best-first growth", "read", G, R, f, {"growing_strategy": "BEST_FIRST_GLOBAL", "max_num_nodes": 12}),
            ("rf binary winner-take-all", "read", RF, C, yb, {"winner_take_all": True}),
            ("rf multiclass winner-take-all", "read", RF, C, ym, {"winner_take_all": True}),
            ("rf multiclass averaged probabilities", "read", RF, C, ym, {"winner_take_all": False}),
            ("rf regression", "read", RF, R, f, {}),
            ("rf without bootstrap", "read", RF, C, ym, {"bootstrap_training_dataset": False}),
            ("rf honest trees", "read", RF, R, f, {"honest": True}),
            ("rf oblique splits", "refused", RF, R, f, {"split_axis": "SPARSE_OBLIQUE"}),
        ]
        for i, (field, expected, learner, task, label, kw) in enumerate(cases):
            m = learner(label="y", task=task, num_trees=8, max_depth=4, **kw).train({**d, "y": label}, verbose=0)
            path = self.t / f"field{i}"
            m.save(str(path))
            if expected == "refused":
                with self.assertRaises(ValueError, msg=field):
                    qf_formats.open_source(path)
                continue
            src = qf_formats.open_source(path)
            forest = qf.Forest(src.spec.n_features, src.spec.groups, src.spec.kind, "output", [src.model_ir()], src.spec.task)
            self.assertLessEqual(rel_err(src.predict(x), qf.ir_predict(forest, x)), TOL, field)

    def test_a_ydf_random_forest_trained_on_one_class_is_refused_in_one_line(self):
        """A forest trained on a single class has no second class to model: it used to compile (winner-take-all) to a forest that answered
        0.0 where YDF answers 1.0, or (averaged probabilities) end in an IndexError traceback. Both read and compile now refuse in one line."""
        import subprocess

        import ydf

        x = np.random.default_rng(0).normal(size=(200, 4)).astype(np.float32)
        d = {f"f{i}": x[:, i] for i in range(4)}
        for wta in (True, False):
            m = ydf.RandomForestLearner(
                label="y", task=ydf.Task.CLASSIFICATION, num_trees=8, max_depth=3, winner_take_all=wta, num_threads=1
            ).train({**d, "y": np.zeros(200, int).astype(str)}, verbose=0)
            path = self.t / f"oneclass-{wta}"
            m.save(str(path))
            with self.assertRaises(ValueError, msg=f"winner_take_all={wta}") as cm:
                qf_formats.open_source(path)
            self.assertNotIn("\n", str(cm.exception))
            self.assertIn("one class", str(cm.exception))
            r = subprocess.run(
                [sys.executable, "-s", str(SCRIPTS / "quickforest"), "compile", str(path), str(self.t / f"oneclass-{wta}.flat")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 3, r.stderr[-300:])
            self.assertNotIn("Traceback", r.stderr)
            self.assertFalse((self.t / f"oneclass-{wta}.flat").exists())

    def test_prune_does_not_write_a_ydf_model_through_a_link_in_the_output_folder(self):
        """A YDF model is a folder; a symlink with its name in OUT_DIR made `m.save` write into the folder it points to."""
        import os

        import ydf

        _, x, f = data(403, n=600, nf=6)
        model = ydf.RandomForestLearner(label="y", task=ydf.Task.REGRESSION, num_trees=10, max_depth=4).train(
            {**{f"f{i}": x[:, i] for i in range(6)}, "y": f}, verbose=0
        )
        path = self.t / "link-src"
        model.save(str(path))
        np.savez(self.t / "link-sel.npz", X=x)
        out, elsewhere = self.t / "link-out", self.t / "link-elsewhere"
        out.mkdir()
        elsewhere.mkdir()
        (elsewhere / "precious.txt").write_text("precious\n")
        os.symlink(elsewhere, out / path.name)
        prune.run(path, self.t / "link-sel.npz", out, "50%")
        self.assertEqual([p.name for p in elsewhere.iterdir()], ["precious.txt"])
        self.assertFalse((out / path.name).is_symlink())
        self.assertEqual(qf.load_forest([out / path.name]).n_trees, 5)

    def test_folds_trained_on_the_same_features_in_another_column_order_are_refused(self):
        """YDF reads matrix column i of every fold as that fold's own i-th input feature, so two folds that list the same features in
        another order would be averaged with swapped columns and `check` (which maps columns the same way) would pass."""
        import ydf

        rng = np.random.default_rng(4)
        a, b = rng.normal(size=600), rng.normal(size=600)
        truth = (a > 0.5).astype(int)
        folds = self.t / "order-folds"
        folds.mkdir()
        for k, order in enumerate((("a", "b"), ("b", "a"))):
            columns = {name: {"a": a, "b": b}[name] for name in order}
            learner = ydf.GradientBoostedTreesLearner(label="y", task=ydf.Task.CLASSIFICATION, num_trees=5, max_depth=3)
            learner.train({**columns, "y": truth}, verbose=0).save(str(folds / f"fold{k}"))
        with self.assertRaisesRegex(ValueError, "names or the order of the input features"):
            qf.load_forest(qf_formats.model_files(folds))
        self.assertEqual(qf.load_forest(qf_formats.model_files(folds)[:1]).n_features, 2)

    def test_the_ydf_speed_baseline_is_one_thread_on_a_fixed_batch_and_named_after_the_library(self):
        """The speed baseline of `compare` / `check --speed` for a YDF model is the library's own predict with the thread count of the compiled
        forest's run (this test pins the one-thread call, num_threads=1) on the SAME number of rows whatever the sample: YDF's multi-threaded default swings with the batch size (630k rows/s on 500 rows, 4.1M on
        1,000), which would print 2.17x in one place and 0.38x in another. The column says which library it is."""
        import compare
        import compile
        import ydf

        _, x, f = data(402, n=1200, nf=6)
        d = {f"f{i}": x[:, i] for i in range(6)}
        model = ydf.GradientBoostedTreesLearner(label="y", task=ydf.Task.REGRESSION, num_trees=10, max_depth=4).train(
            {**d, "y": f}, verbose=0
        )
        path = self.t / "baseline"
        model.save(str(path))
        flat = self.t / "baseline.flat"
        compile.compile_model(path, flat, backend="flat")
        cls, orig = type(ydf.load_model(str(path))), None
        seen = []

        def spy(self_, data, **kw):
            seen.append((len(next(iter(data.values()))), kw.get("num_threads")))
            return orig(self_, data, **kw)

        orig = cls.predict
        xs = np.ascontiguousarray(x[:, :6])
        by_sample = {}
        with mock.patch.object(cls, "predict", spy):
            for n in (300, 1100):
                seen.clear()
                rate, lib = compare.library_speed([path], xs[:n])
                self.assertEqual(lib, "ydf")
                self.assertGreater(rate, 0)
                by_sample[n] = list(seen)
                seen.clear()
                r = check.bench([path], flat, backends.get("flat"), xs[:n], 1)
                self.assertEqual(r["library"], "ydf")
                self.assertIn("ydf", r["baseline"])
                self.assertIn("1 thread", r["baseline"])
                by_sample[(n, "bench")] = list(seen)
        for key, calls in by_sample.items():
            self.assertTrue(calls, key)
            self.assertEqual({t for _, t in calls}, {1}, f"{key}: YDF predict ran on its default threads")
        self.assertEqual(
            {tuple(sorted({b for b, _ in c})) for c in by_sample.values()},
            {(by_sample[300][0][0],)},
            "the baseline batch depends on the sample",
        )
        table = {"model": "m", "trees": 1, "nodes": 1, "features": 1, "cpu": "x", "library_rows_per_s": 5, "library": "ydf", "rows": []}
        md = compare.table_md(table)
        self.assertIn("ydf 1 thread", md)
        self.assertIn("x ydf", md)
        self.assertNotIn("xgboost", md)

    def test_margin_averaging_of_a_poisson_gbt(self):
        """The margin reference of a Poisson GBT is exp(margin) (YDF has no logits switch for it), or check fails a correct forest."""
        import ydf

        _, x, f = data(401, n=1200, nf=6)
        d = {f"f{i}": x[:, i] for i in range(6)}
        y = np.random.default_rng(1).poisson(np.exp(np.nan_to_num(f) / 4)).astype(np.float32)
        folds = self.t / "pois"
        folds.mkdir()
        for k in range(2):
            lrn = ydf.GradientBoostedTreesLearner(
                label="y", task=ydf.Task.REGRESSION, loss="POISSON", num_trees=12, max_depth=3, random_seed=k
            )
            lrn.train({**d, "y": y}, verbose=0).save(str(folds / f"m{k}"))
        flat = self.t / "pois.flat"
        compile.compile_model(folds, flat, average="margin", backend="flat")
        res = check.run(folds, flat, average="margin", n_random=1000)
        self.assertTrue(res["ok"], res["sets"])

    def test_prune_compile_and_verify(self):
        import ydf

        _, x, f = data(400, n=3000, nf=6)
        d = {f"f{i}": x[:, i] for i in range(6)}
        np.savez(self.t / "s.npz", X=x[:, :6], group=np.arange(len(x)))
        ym = ((f > 1).astype(int) + (f > 4)).astype(str)
        gm = ydf.GradientBoostedTreesLearner(label="y", task=ydf.Task.CLASSIFICATION, num_trees=30, max_depth=4).train(
            {**d, "y": ym}, verbose=0
        )
        gm.save(str(self.t / "gm"))
        out = optimize.optimize(self.t / "gm", self.t / "s.npz", self.t / "o1", "50%", True, True, min_agreement=60.0, backend="flat")
        self.assertTrue(out["ok"], out["failures"])
        rm = ydf.RandomForestLearner(label="y", task=ydf.Task.CLASSIFICATION, num_trees=30, max_depth=6).train({**d, "y": ym}, verbose=0)
        rm.save(str(self.t / "rm"))
        out = optimize.optimize(self.t / "rm", self.t / "s.npz", self.t / "o2", "30%", True, True, min_agreement=60.0, backend="flat")
        self.assertTrue(out["ok"], out["failures"])  # the flat file holds the identity-with-several-groups objective since the extensions
        self.assertIsNone(out["compile_unsupported"])
        self.assertTrue(out["stages"]["verify"]["ok"], out["stages"]["verify"])
        self.assertTrue((self.t / "o2" / "pruned" / "rm").is_dir())
        self.assertEqual(len(qf_formats.open_source(self.t / "o2" / "pruned" / "rm").irs), 3 * 9)


if __name__ == "__main__":
    unittest.main()
