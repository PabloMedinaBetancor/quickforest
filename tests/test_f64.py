"""The float64 OUTPUT path (`predict(..., dtype=np.float64)`, C ABI `tfl_predict_f64` / `tfl_cg_predict_f64`) and the opt-in f64 leaves
(`compile ... f64_leaves`), against the libraries on log-scale targets (values 12 to 40, where a float32 answer alone is off by up to
4.8e-7 .. 1.9e-6 absolute).

* LightGBM, plain and linear leaves, offsets 12 / 20 / 40: absolute error of the f64 answer <= 1e-9 with f64 leaves (it is the double
  arithmetic of the library, so ~1e-14 in practice), NaN / zero / huge rows, a base margin, several threads, generated code bit-identical
* what the default compile (float32 leaves) leaves: the f64 answer is off by the leaves' rounding only (<= 1e-6 at 12), and the float32
  answer is the f64 one rounded (bit for bit)
* YDF (gbt and random forest) and xgboost: the f64 answer on every backend (xgboost's float32 result widened)
* the C ABI by ctypes: old entry points unchanged, new ones agree with them; refusals
* files without inexact leaves (xgboost, a YDF boosted model) pack byte-identically with the option on; the default pack is untouched

    PYTHONPATH=<venv with lightgbm, xgboost, ydf> python3 -s -m unittest test_f64
Skips the parts whose library is missing and the generated-code ones without rustc."""

import contextlib
import ctypes
import io
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_codegen
import backend_flat
import backends
import check
import evaluator
import qf_common as qf
import qf_formats
from test_linear import FIX, HAVE_XGB, NFEAT, data, have_rustc, lgb, rows

ABS = 1e-6


def train_log(seed, path, offset, linear, leaves=15, rounds=30, objective="regression"):
    x, y, _ = data(seed)
    params = dict(
        objective=objective,
        linear_tree=linear,
        linear_lambda=0.1,
        num_leaves=leaves,
        learning_rate=0.1,
        verbose=-1,
        num_threads=1,
        deterministic=True,
        min_data_in_leaf=25,
    )
    b = lgb.train(params, lgb.Dataset(x, y + offset, categorical_feature=[3]), rounds)
    b.save_model(str(path))
    return path


def abs_err(want, got):
    return float(np.abs(np.asarray(want, np.float64).reshape(-1) - np.asarray(got, np.float64).reshape(-1)).max())


def half_ulp32(v):
    v = np.abs(np.asarray(v, np.float64))
    return np.spacing(v.astype(np.float32)).astype(np.float64) / 2


@unittest.skipIf(lgb is None, "no lightgbm")
class LightGbmF64Test(unittest.TestCase):
    def check(self, offset, linear, seed):
        with tempfile.TemporaryDirectory() as d:
            p = train_log(seed, Path(d) / "m.txt", offset, linear)
            src = qf_formats.open_source(p)
            x = rows(seed, 700, big=1e3)
            lib = src.predict(x)
            forest = qf.load_forest([p])
            bm = np.random.default_rng(seed).uniform(-3, 3, (len(x), 1))
            plain_path, exact_path = Path(d) / "p.flat", Path(d) / "e.flat"
            backends.get("flat").compile(forest, plain_path, {})
            backends.get("flat").compile(forest, exact_path, {"f64_leaves": True})
            plain, exact = evaluator.FlatForest(plain_path), evaluator.FlatForest(exact_path)
            e64 = exact.predict(x, dtype=np.float64)
            self.assertEqual(e64.dtype, np.float64)
            self.assertLess(abs_err(lib, e64), 1e-9, f"offset {offset} linear {linear}")
            # the float32 answer of the same file is the f64 one rounded, bit for bit; the default file's float32 answer is unchanged
            self.assertTrue(np.array_equal(exact.predict(x), e64.astype(np.float32)))
            self.assertLessEqual((np.abs(e64 - exact.predict(x)) - half_ulp32(e64)).max(), 1e-12)
            # threads and a base margin
            self.assertTrue(np.array_equal(e64, exact.predict(np.tile(x, (4, 1)), 3, dtype=np.float64)[: len(x)]))
            m64 = exact.predict(x, base_margin=bm, dtype=np.float64)
            self.assertLess(abs_err(lib + bm, m64), 1e-9)
            # the default compile keeps float32 leaves: its f64 answer is off by their rounding (the first tree carries the intercept)
            p64 = plain.predict(x, dtype=np.float64)
            bound = (0.5 + 30 * 0.02) * float(np.spacing(np.float32(offset)))  # first-leaf half-ulp plus the small leaves' share
            self.assertLess(abs_err(lib, p64), max(bound, 1e-6))
            self.assertTrue(np.array_equal(plain.predict(x), p64.astype(np.float32)))
            if linear:
                self.assertLess(abs_err(lib, p64), 1e-6 + 4 * np.spacing(np.float32(offset)))  # the linear leaves are double already
            err = {"plain": abs_err(lib, p64), "f64 leaves": abs_err(lib, e64)}
            if have_rustc():
                out = Path(d) / "cg.so"
                backends.get("codegen-block").compile(forest, out, {"f64_leaves": True})
                cg = backend_codegen.CodegenForest(out)
                self.assertTrue(np.array_equal(cg.predict(x, dtype=np.float64), e64), "generated code is not bit-identical to the walker")
                self.assertTrue(np.array_equal(cg.predict(x, 3, base_margin=bm, dtype=np.float64), m64))
                self.assertTrue(np.array_equal(cg.predict(x), exact.predict(x)))
                self.assertLess(abs_err(lib, cg.predict(x, dtype=np.float64)), 1e-9)
            exact.close(), plain.close()
            return err

    def test_log_scale_offsets_plain_and_linear_leaves(self):
        for i, (offset, linear) in enumerate([(12.0, False), (20.0, False), (40.0, False), (12.0, True), (20.0, True)]):
            err = self.check(offset, linear, 100 + i)
            print(f"  lightgbm offset {offset:g} linear {linear}: max abs err plain {err['plain']:.2e}, f64 leaves {err['f64 leaves']:.2e}")

    def test_check_judges_the_f64_answer_on_the_absolute_bar(self):
        for name in ("lgb_plain.txt", "lgb_ext.txt"):
            for backend in ("flat", "codegen-block"):
                if backend.startswith("codegen") and not have_rustc():
                    continue
                forest = qf.load_forest([FIX / name])
                with tempfile.TemporaryDirectory() as d:
                    be = backends.get(backend)
                    art = Path(d) / f"m{be.suffix}"
                    be.compile(forest, art, {"f64_leaves": True})
                    np.savez(Path(d) / "x.npz", X=np.load(FIX / "x.npy"))
                    r = check.run(
                        FIX / name, art, backend, Path(d) / "x.npz", n_random=3000, absolute=True, f64=True, margin=True, speed=True
                    )
                    self.assertTrue(r["ok"], f"{name} {backend}: {r['sets']}")
                    self.assertLess(max(s["max_err"] for s in r["sets"]), 1e-9)
                    self.assertIn("(float64 output)", r["bar"])
                    self.assertIn("f64_over_f32_time", r["speed"])

    def test_leaves_stay_float32_unless_asked_and_the_pack_is_byte_identical_without_inexact_leaves(self):
        forest = qf.load_forest([FIX / "lgb_plain.txt"])
        default = backend_flat.pack(forest, lambda b: 0)
        self.assertEqual(default, backend_flat.pack(forest, lambda b: 0, False))
        exact = backend_flat.pack(forest, lambda b: 0, True)
        self.assertNotEqual(default, exact)
        self.assertEqual(exact[8], backend_flat.FORMAT_LINEAR)  # records for the leaves float32 would round
        self.assertGreater(len(exact), len(default))
        if HAVE_XGB:  # xgboost leaves are float32 numbers: nothing to keep in double
            xf = qf.load_forest([FIX / "xgb_reg.json"])
            self.assertEqual(backend_flat.pack(xf, lambda b: 0, True), backend_flat.pack(xf, lambda b: 0))


class XgboostF64Test(unittest.TestCase):
    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_xgboost_f64_is_the_float32_result_widened_on_every_backend(self):
        import xgboost as xgb

        x, y, _ = data(5)
        for offset in (12.0, 20.0):
            with tempfile.TemporaryDirectory() as d:
                b = xgb.train(
                    {"objective": "reg:squarederror", "max_depth": 5, "eta": 0.3, "base_score": offset}, xgb.DMatrix(x, y + offset), 25
                )
                m = Path(d) / "m.json"
                b.save_model(str(m))
                forest = qf.load_forest([m])
                xs = rows(3, 500, big=1e3)
                lib = qf_formats.open_source(m).predict(xs)
                for name in ("flat", "wave", "wave-avx2", "codegen", "codegen-block"):
                    if name.startswith("codegen") and not have_rustc():
                        continue
                    be = backends.get(name)
                    art = Path(d) / f"m-{name}{be.suffix}"
                    be.compile(forest, art, {"f64_leaves": True})  # nothing to keep in double: ignored
                    pred = be.load(art)
                    p64, p32 = pred.predict(xs, dtype=np.float64), pred.predict(xs)
                    self.assertEqual((p64.dtype, p32.dtype), (np.float64, np.float32))
                    self.assertTrue(np.array_equal(p64, p32.astype(np.float64)), name)
                    self.assertLess(abs_err(lib, p64), ABS, f"{name} offset {offset}")
                    r = check.run(m, art, name, None, n_random=2000, absolute=True, f64=True)
                    self.assertTrue(r["ok"], f"{name}: {r['sets']}")
                    # with a margin the float32 arithmetic of xgboost itself decides (a few ulp at 15..30): the relative bar holds
                    r = check.run(m, art, name, None, n_random=2000, f64=True, margin=True)
                    self.assertTrue(r["ok"], f"{name} + margin: {r['sets']}")


class YdfF64Test(unittest.TestCase):
    """YDF's own predictor adds the leaves in float32, tree by tree from the initial prediction: its ANSWER carries that rounding (measured
    up to 3.3e-5 absolute at 300 trees on a log-scale target of 14). Double sums (the default) and f64 leaves cannot reproduce it; the
    float32 sums option does, bit for bit."""

    @staticmethod
    def models():
        import ydf

        x, y, _ = data(8)
        d = {f"f{i}": x[:, i].astype(np.float32) for i in range(NFEAT)}
        yc = (y > np.median(y)).astype(int)
        for name, lrn, task, lab, kw in [
            ("gbt 120 trees, 12", ydf.GradientBoostedTreesLearner, "REGRESSION", y + 12, {"num_trees": 120}),
            (
                "gbt 200 trees shrinkage .02, 14",
                ydf.GradientBoostedTreesLearner,
                "REGRESSION",
                y + 14,
                {"num_trees": 200, "shrinkage": 0.02},
            ),
            ("rf 40 trees, 20", ydf.RandomForestLearner, "REGRESSION", y + 20, {"num_trees": 40}),
            ("gbt binary", ydf.GradientBoostedTreesLearner, "CLASSIFICATION", yc, {"num_trees": 60}),
            ("rf binary", ydf.RandomForestLearner, "CLASSIFICATION", yc, {"num_trees": 30}),
        ]:
            yield name, lrn(label="y", task=getattr(ydf.Task, task), max_depth=5, random_seed=1, **kw).train({**d, "y": lab}, verbose=0)

    def test_float32_sums_reproduce_ydf_and_double_sums_carry_its_noise(self):
        try:
            import ydf  # noqa: F401
        except ImportError:
            self.skipTest("no ydf")
        xs = rows(9, 800, big=1e3)
        bm = np.random.default_rng(3).uniform(-2, 2, (len(xs), 1))
        for name, m in self.models():
            with tempfile.TemporaryDirectory() as t:
                m.save(str(Path(t) / "m"))
                files = qf_formats.model_files(Path(t) / "m")
                lib = qf_formats.open_source(Path(t) / "m").predict(xs)
                got = {}
                for f32s in (False, True):
                    forest = qf.load_forest(files, "output", f32s)
                    self.assertEqual(forest.sum64, not f32s)
                    art = Path(t) / f"m{int(f32s)}.flat"
                    backends.get("flat").compile(forest, art, {"f64_leaves": True})
                    pred = evaluator.FlatForest(art)
                    got[f32s] = (abs_err(lib, pred.predict(xs, dtype=np.float64)), abs_err(lib, pred.predict(xs)))
                    if f32s:
                        self.assertEqual(
                            art.read_bytes()[48:52], bytes([8, 0, 0, 0]), "float32 sums: no double-sum flag, only the late-margin one"
                        )
                        self.assertTrue(np.array_equal(pred.predict(xs, dtype=np.float64), pred.predict(xs).astype(np.float64)))
                        # regression: bit for bit; classification: the float32 transform may differ by an ulp
                        if name.startswith(("gbt 1", "gbt 2", "rf 4")):
                            self.assertEqual(got[f32s], (0.0, 0.0), f"{name}: float32 sums are not bit-identical to YDF")
                        self.assertLess(got[f32s][0], 1e-6)
                        if name.startswith(("gbt 1", "rf 40")):  # the wavefront engines hold float32-sum files
                            self.assertTrue(np.array_equal(evaluator.FlatForest(art, engine=1).predict(xs), pred.predict(xs)))
                        if have_rustc() and name.startswith("gbt 120"):
                            out = Path(t) / "cg.so"
                            backends.get("codegen-block").compile(forest, out, {"f64_leaves": True})
                            self.assertTrue(
                                np.array_equal(
                                    backend_codegen.CodegenForest(out).predict(xs, dtype=np.float64), pred.predict(xs, dtype=np.float64)
                                )
                            )
                        # the base margin joins LATE, in double (YDF's float32 trajectory starts at its own initial prediction, not at the
                        # margin): the answer is the library's raw score + margin, transformed in double (summing the margin first would put
                        # it ~2e-5 off)
                        want = qf.reference_predict(files, xs, "output", base_margin=bm)
                        got64 = pred.predict(xs, base_margin=bm, dtype=np.float64)
                        # regression: 0.0; classification: no worse than without the margin (the float32 transform of YDF's probability
                        # differs by a few ulp)
                        self.assertLess(
                            abs_err(want, got64), 1e-12 if name.startswith(("gbt 1", "gbt 2", "rf 4")) else got[True][0] + 1e-9, name
                        )
                        self.assertTrue(np.array_equal(pred.predict(xs, base_margin=bm), got64.astype(np.float32)), name)
                        self.assertTrue(np.array_equal(pred.predict(xs, 3, base_margin=bm, dtype=np.float64), got64), f"{name}: threads")
                        if name.startswith(("gbt 1", "rf 40")):
                            wave = evaluator.FlatForest(art, engine=1)
                            self.assertTrue(np.array_equal(wave.predict(xs, base_margin=bm, dtype=np.float64), got64), f"{name}: wave")
                            wave.close()
                        if have_rustc() and name.startswith("gbt 120"):
                            cgm = backend_codegen.CodegenForest(out)
                            self.assertTrue(np.array_equal(cgm.predict(xs, 2, base_margin=bm, dtype=np.float64), got64), f"{name}: codegen")
                            self.assertTrue(
                                np.array_equal(cgm.predict(xs, base_margin=bm), got64.astype(np.float32)), f"{name}: codegen f32"
                            )
                    pred.close()
                print(
                    f"  ydf {name}: max abs err double sums {got[False][0]:.2e} (f32 out {got[False][1]:.2e}), float32 sums {got[True][0]:.2e}"
                )
                self.assertLess(got[False][0], 1e-4)

    def test_float32_sums_are_exact_for_trees_over_64_leaves(self):
        """YDF's generic engine (any tree of more than 64 leaves) adds the initial prediction LAST; summing it first puts the answer up to
        2e-5 off `model.predict` at 75 leaves. The initial prediction is a trailing single-leaf tree per group."""
        try:
            import ydf
        except ImportError:
            self.skipTest("no ydf")
        rng = np.random.default_rng(5)
        n = 800
        x = rng.normal(size=(n, 5)).astype(np.float32)
        y = 20 + 3 * x[:, 0] + np.sin(4 * x[:, 1]) * 2 + (x[:, 2] > 0) * x[:, 3] + rng.normal(size=n) * 0.2
        m = ydf.GradientBoostedTreesLearner(
            label="y", task=ydf.Task.REGRESSION, num_trees=40, max_depth=8, min_examples=2, shrinkage=0.3, random_seed=1
        ).train({**{f"f{i}": x[:, i] for i in range(5)}, "y": y}, verbose=0)
        with tempfile.TemporaryDirectory() as t:
            m.save(str(Path(t) / "m"))
            files = qf_formats.model_files(Path(t) / "m")
            forest = qf.load_forest(files, "output", True)
            self.assertGreater(
                max(int((tr.left < 0).sum()) for tr in forest.models[0].trees), 64, "the model must have a tree over 64 leaves"
            )
            art = Path(t) / "m.flat"
            backends.get("flat").compile(forest, art, {"f64_leaves": True})
            pred = evaluator.FlatForest(art)
            xs = (np.random.default_rng(11).normal(size=(800, 5)) * 3).astype(np.float32)
            xs[np.random.default_rng(12).random(xs.shape) < 0.05] = np.nan
            lib = qf_formats.open_source(Path(t) / "m").predict(xs)
            self.assertEqual(abs_err(lib, pred.predict(xs, dtype=np.float64)), 0.0)
            wave = evaluator.FlatForest(art, engine=1)
            self.assertTrue(np.array_equal(wave.predict(xs), pred.predict(xs)))
            wave.close()
            pred.close()

    def test_float32_sums_are_refused_for_the_other_libraries(self):
        files = [FIX / "lgb_plain.txt"]
        if lgb is not None:
            with self.assertRaises(ValueError) as cm:
                qf.load_forest(files, "output", True)
            self.assertIn("YDF", str(cm.exception))
        if HAVE_XGB:
            with self.assertRaises(ValueError):
                qf.load_forest([FIX / "xgb_reg.json"], "output", True)


@unittest.skipIf(lgb is None, "no lightgbm")
class XgboostBaseMarginTest(unittest.TestCase):
    """The logit of a binary:logistic base_score is xgboost's own float32 `logf` result, bit for bit: with numpy's float32 log the margin is
    an ulp off for one score in six, and so is the sigmoid of every row of such a model."""

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_logit_of_the_base_score_is_xgboosts_own(self):
        import json

        import xgboost

        rng = np.random.default_rng(5)
        x = rng.normal(size=(40, 3)).astype(np.float32)
        dm = xgboost.DMatrix(x, label=(rng.random(40) < 0.5).astype(np.float32))
        scores = rng.uniform(0.01, 0.99, 80)
        bad = 0
        for sc in scores:
            b = xgboost.train({"objective": "binary:logistic", "base_score": float(sc), "verbosity": 0, "nthread": 1}, dm, 0)
            want = b.predict(dm, output_margin=True)[0]
            param = json.loads(b.save_raw("json"))["learner"]["learner_model_param"]
            bad += float(qf.base_margin(param, "binary:logistic", 1)[0]) != float(want)
        self.assertEqual(bad, 0)


class ReferenceAveragingTest(unittest.TestCase):
    def test_margin_average_of_a_double_library_is_not_rounded_to_float32(self):
        """`reference_predict(..., "margin")` is what `check` judges a compiled forest against. For LightGBM / YDF (double answers) it must
        not pass the mean margin through the float32 `transform`: that rounds the reference (6e-8 relative; 4.8e-7 absolute on a
        log-scale target of 12) and a float64 absolute bar would judge the compiled forest against a rounded number."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            x, y, _ = data(21)
            for kind in ("binary", "regression"):
                yy = (y > np.median(y)).astype(float) if kind == "binary" else y + 12.5
                for f in range(2):
                    p = dict(objective=kind, num_leaves=7, learning_rate=0.1, verbose=-1, num_threads=1, deterministic=True, seed=f + 5)
                    lgb.train(p, lgb.Dataset(x[f::2], yy[f::2]), 15).save_model(str(d / f"{kind}{f}.txt"))
                srcs = [qf_formats.open_source(d / f"{kind}{f}.txt") for f in range(2)]
                xs = rows(22, 300)
                raw = np.mean([s.predict(xs, margin=True) for s in srcs], axis=0)
                want = qf.transform_f64(raw, "sigmoid" if kind == "binary" else "identity")
                got = qf.reference_predict([d / f"{kind}0.txt", d / f"{kind}1.txt"], xs, "margin")
                self.assertEqual(got.dtype, np.float64)
                self.assertTrue(np.array_equal(got, want), kind)


class OptionParsingTest(unittest.TestCase):
    def test_opt_pairs_booleans_are_booleans(self):
        got = qf.parse_options(["f64_leaves=false", "float32_sums=True", "block=64", "keep_src=0", "name=x y", "no_lib=on"])
        self.assertEqual(got, {"f64_leaves": False, "float32_sums": True, "block": 64, "keep_src": 0, "name": "x y", "no_lib": True})
        self.assertIs(got["f64_leaves"], False)
        with self.assertRaises(SystemExit):
            qf.parse_options(["block"])
        for name in ("compare.py", "choose.py"):  # both CLIs go through it
            self.assertIn("qf.parse_options(a.opt)", (HERE.parent / "scripts" / name).read_text())


class RefusalsAreMessagesTest(unittest.TestCase):
    """A refusal from the CLIs is a one-line message and exit 3 (as optimize's "cannot hold the model"), never a Python traceback."""

    def run_cli(self, *args):
        r = subprocess.run([sys.executable, "-s", *map(str, args)], capture_output=True, text=True, timeout=240)
        return r.returncode, r.stderr

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_compile_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            sc = HERE.parent / "scripts" / "compile.py"
            for extra, word in ((["--float32-sums"], "YDF"), (["--backend", "wave"], "cannot compile")):
                code, err = self.run_cli(sc, FIX / "lgb_plain.txt", Path(d) / "o.flat", *extra)
                self.assertEqual(code, 3, err)
                self.assertNotIn("Traceback", err)
                self.assertIn(word, err)
                self.assertNotIn(" yet", err)  # a refusal says what the backend cannot hold, not that it will one day
            self.assertFalse((Path(d) / "o.flat").exists())
            code, err = self.run_cli(sc, FIX / "lgb_plain.txt", Path(d) / "o.flat", "--backend", "auto")  # needs --data
            self.assertEqual(code, 3, err)
            self.assertNotIn("Traceback", err)


class WaveEngineRefusalTest(unittest.TestCase):
    """An explicit engine 1 / 2 (the wavefront engines) on a file they cannot hold (double sums, extended nodes, linear leaves: LightGBM /
    YDF) loads, and `predict` raises an EngineRefused that says what the C ABI's return code means, not the bare `RuntimeError:
    tfl_predict failed (1)` (the CLI turns it into one line and exit 3)."""

    def flat(self, model, d, **opts):
        out = Path(d) / "o.flat"
        backends.get("flat").compile(qf.load_forest(qf_formats.model_files(model), "output"), out, opts)
        return out

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_engine_1_2_on_a_file_the_wave_engines_cannot_hold_is_a_clear_message(self):
        with tempfile.TemporaryDirectory() as d:
            lin = train_log(3, Path(d) / "lin.txt", 12, True, leaves=7, rounds=10)
            x = rows(5, 40)
            bm = np.full((len(x), 1), 0.5)
            for name, model, opts in (
                ("plain", FIX / "lgb_plain.txt", {}),
                ("linear", lin, {}),
                ("plain f64 leaves", FIX / "lgb_plain.txt", {"f64_leaves": True}),
            ):
                art = self.flat(model, d, **opts)
                for engine in (1, 2):
                    pred = evaluator.FlatForest(art, engine=engine)  # opening is fine: the walker (engine 0) holds it
                    for what, kw in (
                        ("float32", {}),
                        ("float64", {"dtype": np.float64}),
                        ("margin", {"base_margin": bm}),
                        ("float64 + margin", {"dtype": np.float64, "base_margin": bm}),
                    ):
                        with self.subTest(f"{name} engine {engine} {what}"):
                            with self.assertRaises(evaluator.EngineRefused) as cm:
                                pred.predict(x, **kw)
                            self.assertIsInstance(cm.exception, RuntimeError)  # still a RuntimeError
                            msg = str(cm.exception)
                            self.assertNotIn("tfl_predict failed", msg)
                            for word in (f"engine {engine}", "engine=0", "double-precision sums", "extended nodes", "linear leaves"):
                                self.assertIn(word, msg)
                            self.assertNotIn("--f64-leaves", msg)  # an xgboost --f64-leaves file runs on the wave engines
                    pred.close()
                # the walker holds it, and a forest the wave engines CAN hold (xgboost) still answers on every engine
                walker = evaluator.FlatForest(art)
                self.assertEqual(len(walker.predict(x)), len(x))
                walker.close()

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_engines_still_run_a_float32_xgboost_file(self):
        with tempfile.TemporaryDirectory() as d:
            art = self.flat(FIX / "xgb_reg.json", d)
            x = rows(5, 40)
            walker = evaluator.FlatForest(art)
            want = walker.predict(x)
            for engine in (1, 2):
                pred = evaluator.FlatForest(art, engine=engine)
                self.assertTrue(np.array_equal(pred.predict(x), want))
                pred.close()
            walker.close()

    @staticmethod
    def flags(art):
        return struct.unpack_from("<I", Path(art).read_bytes(), 48)[0]

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_the_wave_engines_run_f64_leaves_and_float32_sums_files_that_hold_no_double_sums(self):
        """What the wave engines refuse is what the file's flags say (double sums 1, extended nodes 2, linear leaves 4): `--f64-leaves` on
        an xgboost model writes the very same file and runs, and a YDF `--float32-sums` file (only the late-margin flag, 8) runs with a
        base margin; a LightGBM file, which carries double sums, is refused even with `--f64-leaves`."""
        x = rows(5, 40)
        with tempfile.TemporaryDirectory() as d:
            for sub in ("a", "b"):
                (Path(d) / sub).mkdir()
            default = self.flat(FIX / "xgb_reg.json", Path(d) / "a")
            f64 = self.flat(FIX / "xgb_reg.json", Path(d) / "b", f64_leaves=True)
            self.assertEqual(default.read_bytes(), f64.read_bytes())
            self.assertEqual(self.flags(f64), 0)
            walker = evaluator.FlatForest(f64)
            want = walker.predict(x)
            walker.close()
            for engine in (1, 2):
                pred = evaluator.FlatForest(f64, engine=engine)
                self.assertTrue(np.array_equal(pred.predict(x), want))
                pred.close()
            if lgb is not None:
                (Path(d) / "c").mkdir()
                lgb_art = self.flat(FIX / "lgb_plain.txt", Path(d) / "c", f64_leaves=True)
                self.assertTrue(self.flags(lgb_art) & 1)
                pred = evaluator.FlatForest(lgb_art, engine=1)
                with self.assertRaises(evaluator.EngineRefused):
                    pred.predict(x)
                pred.close()

    def test_the_wave_engines_run_a_ydf_float32_sums_file_with_a_base_margin(self):
        try:
            import ydf  # noqa: F401
        except ImportError:
            self.skipTest("no ydf")
        name, model = next(YdfF64Test.models())
        with tempfile.TemporaryDirectory() as d:
            ydf_model = Path(d) / "m"
            model.save(str(ydf_model))
            art = Path(d) / "o.flat"
            backends.get("flat").compile(qf.load_forest(qf_formats.model_files(ydf_model), "output", True), art, {"float32_sums": True})
            self.assertEqual(self.flags(art), 8, name)
            x = rows(5, 60)
            bm = np.full((len(x), 1), 0.5)
            walker = evaluator.FlatForest(art)
            want = walker.predict(x, base_margin=bm)
            walker.close()
            for engine in (1, 2):
                pred = evaluator.FlatForest(art, engine=engine)
                self.assertTrue(np.array_equal(pred.predict(x, base_margin=bm), want), engine)
                pred.close()

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_an_unknown_engine_is_refused_when_opening(self):
        with tempfile.TemporaryDirectory() as d:
            art = self.flat(FIX / "lgb_plain.txt", d)
            for engine in (3, 7, -1):
                with self.subTest(engine=engine):
                    with self.assertRaises(ValueError) as cm:
                        evaluator.FlatForest(art, engine=engine)
                    self.assertIn("engine must be 0 (lock-step walker), 1 (scalar wavefront) or 2 (AVX2 wavefront)", str(cm.exception))

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_check_backend_wave_on_such_a_file_is_one_line_and_exit_3(self):
        with tempfile.TemporaryDirectory() as d:
            art = self.flat(FIX / "lgb_plain.txt", d)
            np.savez(Path(d) / "x.npz", X=rows(5, 60))
            for backend in ("wave", "wave-avx2"):
                with self.subTest(backend):
                    r = subprocess.run(
                        [
                            sys.executable,
                            "-s",
                            str(HERE.parent / "scripts" / "check.py"),
                            str(FIX / "lgb_plain.txt"),
                            str(art),
                            "--data",
                            str(Path(d) / "x.npz"),
                            "--backend",
                            backend,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=900,
                    )
                    self.assertEqual(r.returncode, 3, r.stderr)
                    self.assertNotIn("Traceback", r.stderr)
                    self.assertIn("cannot evaluate", r.stderr)
                    self.assertEqual(len(r.stderr.strip().splitlines()), 1, r.stderr)


class AutoBackendAnyFormatTest(unittest.TestCase):
    """`compile --backend auto` / `choose` time the candidates against the LIBRARY that wrote the model: a LightGBM (linear or not) or a YDF
    model must not be loaded as an xgboost booster, although the CLI is documented for all three."""

    def choose(self, model, d, **kw):
        import choose

        np.savez(d / "x.npz", X=rows(5, 900))
        out = d / "o.flat"
        info = choose.choose(model, out, d / "x.npz", only=["flat"], log=lambda *_: None, **kw)
        return info, out

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_auto_reports_no_parse_time_it_did_not_measure(self):
        """`--backend auto` parses inside each candidate's build: it has no parse time, and the command line says none instead of 0.0 s."""
        import compile as compile_cli

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.savez(d / "x.npz", X=rows(5, 300))
            info = compile_cli.compile_model(FIX / "lgb_plain.txt", d / "o.flat", backend="auto", data=d / "x.npz")
            self.assertNotIn("parse_seconds", info)
            self.assertIn("compile_seconds", info)
            out = io.StringIO()
            argv = ["compile.py", str(FIX / "lgb_plain.txt"), str(d / "o2.flat"), "--backend", "auto", "--data", str(d / "x.npz")]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                compile_cli.main()
            self.assertIn("compile ", out.getvalue().splitlines()[-1])
            self.assertNotIn("parse", out.getvalue().splitlines()[-1])
            argv = ["compile.py", str(FIX / "lgb_plain.txt"), str(d / "o3.flat")]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out):
                compile_cli.main()
            self.assertIn("parse ", out.getvalue().splitlines()[-1])

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_no_exact_candidate_is_a_value_error_and_a_one_line_exit_3(self):
        """The refusal is a ValueError like every other 'no backend here' (it was a RuntimeError that escaped `main` as a traceback)."""
        import choose

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.savez(d / "x.npz", X=rows(5, 300))
            with mock.patch.object(choose.compare, "best_exact", return_value=None), self.assertRaises(ValueError) as cm:
                choose.choose(FIX / "lgb_plain.txt", d / "o.flat", d / "x.npz", only=["flat"], log=lambda *_: None)
            self.assertIn("no candidate was exact", str(cm.exception))
            err = io.StringIO()
            argv = ["choose.py", str(FIX / "lgb_plain.txt"), str(d / "o2.flat"), "--data", str(d / "x.npz"), "--only", "flat"]
            with (
                mock.patch.object(choose.compare, "best_exact", return_value=None),
                mock.patch.object(sys, "argv", argv),
                contextlib.redirect_stderr(err),
                self.assertRaises(SystemExit) as ex,
            ):
                choose.main()
            self.assertEqual(ex.exception.code, 3)
            self.assertTrue(err.getvalue().startswith("choose.py: no candidate was exact"), err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_a_missing_data_file_is_a_refused_argument_for_choose_and_compile_auto(self):
        import choose
        import compile as compile_cli

        with tempfile.TemporaryDirectory() as d:
            for module, argv in (
                (choose, ["choose.py", str(FIX / "lgb_plain.txt"), str(Path(d) / "o.flat"), "--data", str(Path(d) / "absent.npz")]),
                (
                    compile_cli,
                    [
                        "compile.py",
                        str(FIX / "lgb_plain.txt"),
                        str(Path(d) / "o.flat"),
                        "--backend",
                        "auto",
                        "--data",
                        str(Path(d) / "absent.npz"),
                    ],
                ),
            ):
                err = io.StringIO()
                with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as ex:
                    module.main()
                self.assertEqual(ex.exception.code, 3, err.getvalue())
                self.assertIn("no such data file", err.getvalue())
                self.assertNotIn("Traceback", err.getvalue())
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), [])  # nothing was left behind

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_auto_on_lightgbm_plain_and_linear(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            lin = train_log(3, d / "lin.txt", 12, True, leaves=7, rounds=10)
            for name, model, opts in (("plain", FIX / "lgb_plain.txt", {}), ("linear", lin, {}), ("linear f64", lin, {"f64_leaves": True})):
                with self.subTest(name):
                    sub = d / name.replace(" ", "-")
                    sub.mkdir()
                    info, out = self.choose(model, sub, compile_options=opts)
                    self.assertEqual(info["backend"], "flat")
                    self.assertGreater(info["speedup_vs_library"], 0)
                    pred = backends.load_artifact(out)
                    x = rows(9, 300)
                    want = qf.reference_predict(qf_formats.model_files(model), x, "output")
                    got = pred.predict(x, 1, dtype=np.float64)
                    self.assertLessEqual(np.abs(np.asarray(want, np.float64).reshape(len(x), -1) - got.reshape(len(x), -1)).max(), 1e-6)

    def test_auto_on_ydf(self):
        try:
            import ydf
        except ImportError:
            self.skipTest("no ydf")
        x, y, _ = data(8)
        d = {f"f{i}": x[:, i].astype(np.float32) for i in range(NFEAT)}
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            ydf.GradientBoostedTreesLearner(label="y", task=ydf.Task.REGRESSION, num_trees=20, max_depth=4, random_seed=1).train(
                {**d, "y": y + 12}, verbose=0
            ).save(str(t / "m"))
            info, _ = self.choose(t / "m", t, compile_options={"float32_sums": True})
            self.assertEqual(info["backend"], "flat")
            self.assertGreater(info["speedup_vs_library"], 0)


class OptimizarForwardsTest(unittest.TestCase):
    """`optimize --compile` passes the two compile options on, instead of compiling with the defaults whatever was asked."""

    def test_f64_leaves_and_float32_sums_reach_the_compile_step(self):
        import optimize

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            np.savez(d / "x.npz", X=np.load(FIX / "x.npy"))
            plain = optimize.optimize(FIX / "lgb_plain.txt", d / "x.npz", d / "o1", "100%", True, True, backend="flat")
            exact = optimize.optimize(FIX / "lgb_plain.txt", d / "x.npz", d / "o2", "100%", True, True, f64_leaves=True, backend="flat")
            self.assertTrue(plain["ok"] and exact["ok"], (plain["failures"], exact["failures"]))
            self.assertNotEqual((d / "o1" / "model.flat").read_bytes()[8], backend_flat.FORMAT_LINEAR)
            self.assertEqual(
                (d / "o2" / "model.flat").read_bytes()[8], backend_flat.FORMAT_LINEAR, "the double leaves were not compiled in"
            )
            self.assertEqual(exact["stages"]["verify"]["bar"].count("float64 output"), 1)
            with self.assertRaises(ValueError) as cm:  # refused at once, before anything is pruned or written
                optimize.optimize(FIX / "lgb_plain.txt", d / "x.npz", d / "o3", "100%", True, False, float32_sums=True, backend="flat")
            self.assertIn("YDF", str(cm.exception))
            self.assertFalse((d / "o3").exists())
            r = subprocess.run([sys.executable, str(HERE.parent / "scripts" / "optimize.py"), "--help"], capture_output=True, text=True)
            self.assertIn("--f64-leaves", r.stdout)
            self.assertIn("--float32-sums", r.stdout)


class CAbiF64Test(unittest.TestCase):
    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_new_entry_points_agree_with_the_old_ones(self):
        with tempfile.TemporaryDirectory() as d:
            p = train_log(1, Path(d) / "m.txt", 12.0, True)
            forest = qf.load_forest([p])
            art = Path(d) / "m.flat"
            backends.get("flat").compile(forest, art, {"f64_leaves": True})
            lib = evaluator.load_lib(evaluator.find_lib(art))
            err = ctypes.create_string_buffer(256)
            h = lib.tfl_open(str(art).encode(), err, 256)
            self.assertTrue(h, err.value)
            x = np.ascontiguousarray(rows(2, 300, big=1e3), np.float32)
            n = len(x)
            bm = np.random.default_rng(0).uniform(-2, 2, (n, 1))
            old32, old32m, ex32 = (np.empty((n, 1), np.float32) for _ in range(3))
            self.assertEqual(lib.tfl_predict(h, x.ctypes.data, n, old32.ctypes.data, 1), 0)
            self.assertEqual(lib.tfl_predict_ex(h, x.ctypes.data, n, ex32.ctypes.data, 1, 0), 0)
            self.assertEqual(lib.tfl_predict_margin(h, x.ctypes.data, n, old32m.ctypes.data, 1, 0, bm.ctypes.data), 0)
            n64, n64m = np.empty((n, 1)), np.empty((n, 1))
            self.assertEqual(lib.tfl_predict_f64(h, x.ctypes.data, n, n64.ctypes.data, 1, 0, None), 0)
            self.assertEqual(lib.tfl_predict_f64(h, x.ctypes.data, n, n64m.ctypes.data, 3, 0, bm.ctypes.data), 0)
            self.assertTrue(np.array_equal(old32, ex32))
            self.assertTrue(np.array_equal(old32, n64.astype(np.float32)))
            self.assertTrue(np.array_equal(old32m, n64m.astype(np.float32)))
            self.assertEqual(lib.tfl_predict_f64(h, x.ctypes.data, n, n64.ctypes.data, 1, 7, None), 2)  # unknown engine
            self.assertEqual(lib.tfl_predict_f64(h, x.ctypes.data, n, n64.ctypes.data, 1, 1, None), 1)  # wave refuses a double-sum file
            lib.tfl_close(h)
            if have_rustc():
                out = Path(d) / "cg.so"
                backends.get("codegen-block").compile(forest, out, {"f64_leaves": True})
                so = ctypes.CDLL(str(out))
                for name in ("tfl_cg_predict", "tfl_cg_predict_margin", "tfl_cg_predict_f64"):
                    self.assertTrue(hasattr(so, name), name)
                so.tfl_cg_predict_f64.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
                so.tfl_cg_predict.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_uint32]
                c64, c32 = np.empty((n, 1)), np.empty((n, 1), np.float32)
                self.assertEqual(so.tfl_cg_predict_f64(x.ctypes.data, n, c64.ctypes.data, 1, None), 0)
                self.assertEqual(so.tfl_cg_predict(x.ctypes.data, n, c32.ctypes.data, 1), 0)
                self.assertTrue(np.array_equal(c64, n64) and np.array_equal(c32, old32))

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_dtype_must_be_float32_or_float64_and_a_bad_margin_is_still_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = train_log(2, Path(d) / "m.txt", 12.0, False, rounds=5)
            _, pred = (lambda f: (f, evaluator.FlatForest(self._compile(f, Path(d) / "m.flat"))))(qf.load_forest([p]))
            x = rows(1, 20)
            with self.assertRaises(ValueError):
                pred.predict(x, dtype=np.int32)
            with self.assertRaises(ValueError):
                pred.predict(x, base_margin=np.zeros(3), dtype=np.float64)
            self.assertEqual(pred.predict(x, dtype="float64").dtype, np.float64)

    @staticmethod
    def _compile(forest, art):
        backends.get("flat").compile(forest, art, {})
        return art


if __name__ == "__main__":
    unittest.main()
