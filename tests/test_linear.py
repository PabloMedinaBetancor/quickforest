"""LightGBM linear-leaf trees (`linear_tree=True`) and the base margin, against the libraries and against the numpy reference.

* random LightGBM linear-tree models (several depths and `linear_lambda`, NaN rows, zero and tiny values, a categorical split feature,
  regression / binary / poisson / multiclass, k-fold folders) read by the adapter, compiled to the flat file and to generated code, and
  compared with `lightgbm.Booster.predict` (<= 1e-6, relative to max(1, |value|)); the walker is bit-identical to the numpy reference
  for the identity objective
* the NaN fallback pinned on planted rows (a NaN in a feature of the leaf's linear model gives the plain leaf value, nothing else does)
* the base margin on every backend (walker, wave, wave-avx2, codegen, codegen-block) and through the C ABI
* old forests byte-identical (an xgboost forest = the xgboost format, LightGBM with categorical / zero-as-missing nodes = the double format)
* the refusals (wave engines, pruning)

    PYTHONPATH=<venv with lightgbm and xgboost> python3 -s -m unittest discover -s tests
Skips the LightGBM / xgboost parts when those libraries are missing and the generated-code parts without rustc."""

import ctypes
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
FIX = HERE / "fixtures"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(HERE))
import backend_codegen
import backend_flat
import backends
import check
import evaluator
import qf_common as qf
import qf_formats
from test_extensions import random_forest, random_rows

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import xgboost  # noqa: F401

    HAVE_XGB = True
except ImportError:  # pragma: no cover
    HAVE_XGB = False

TOL = 1e-6
NFEAT = 7


def have_rustc() -> bool:
    try:
        backend_codegen.toolchain_bin()
        return True
    except FileNotFoundError:
        return False


# ------------------------------------------------------------------------------------------------ LightGBM models and rows
def data(seed, n=900):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, NFEAT))
    x[:, 3] = rng.integers(0, 5, n)  # the categorical feature
    x[rng.random((n, NFEAT)) < 0.06] = np.nan
    z = np.nan_to_num(x)
    y = 3 * z[:, 0] + z[:, 1] ** 2 + (z[:, 3] == 2) * 2 + 0.5 * z[:, 4] * (z[:, 2] > 0) + rng.normal(size=n) * 0.2
    return x, y, rng


def train(seed, path, objective="regression", leaves=7, lam=0.1, rounds=12, cat=True, extra=None):
    x, y, rng = data(seed)
    if objective == "binary":
        y = (y > np.median(y)).astype(int)
    elif objective == "multiclass":
        y = np.digitize(y, np.quantile(y, [0.33, 0.66]))
    elif objective == "poisson":
        y = rng.poisson(np.exp(0.3 * np.nan_to_num(x[:, 0])))
    params = dict(
        objective=objective,
        linear_tree=True,
        linear_lambda=lam,
        num_leaves=leaves,
        learning_rate=0.1,
        verbose=-1,
        num_threads=1,
        deterministic=True,
        min_data_in_leaf=25,
        **(extra or {}),
    )
    if objective == "multiclass":
        params["num_class"] = 3
    b = lgb.train(params, lgb.Dataset(x, y, categorical_feature=[3] if cat else "auto"), rounds)
    b.save_model(str(path))
    return path


def rows(seed, n=600, big=1e6):
    """Float32 rows with NaN, exact zeros, the zero bucket (+-1e-36), huge values (+-`big`: 1e6 keeps a regression answer inside float32;
    exp and sigmoid objectives get 20), unknown categories, an all-NaN and an all-zero row."""
    rng = np.random.default_rng(seed)
    x = (rng.normal(size=(n, NFEAT)) * 2).astype(np.float32)
    pick = rng.random((n, NFEAT))
    x[pick < 0.12] = np.nan
    special = np.array([0.0, -0.0, 1e-36, -1e-36, 1e-35, big, -big], np.float32)
    m = (pick > 0.12) & (pick < 0.3)
    x[m] = rng.choice(special, m.sum())
    x[:, 3] = rng.choice(np.array([0, 1, 2, 3, 4, 7, -1, 2.5, np.nan], np.float32), n)
    x[0], x[1] = np.nan, 0.0
    return x


def flat_for(files, out, average="output"):
    forest = qf.load_forest([Path(f) for f in files], average)
    backends.get("flat").compile(forest, Path(out), {})
    return forest, evaluator.FlatForest(out)


def rel_err(want, got):
    want, got = np.asarray(want, np.float64).reshape(len(want), -1), np.asarray(got, np.float64).reshape(len(want), -1)
    return float((np.abs(want - got) / np.maximum(1.0, np.abs(want))).max())


@unittest.skipIf(lgb is None, "no lightgbm")
class LinearLeavesTest(unittest.TestCase):
    CASES: ClassVar[list] = [  # (objective, leaves, linear_lambda, rounds, extra)
        ("regression", 3, 0.0, 10, {}),
        ("regression", 7, 0.1, 12, {}),
        ("regression", 15, 10.0, 8, {}),
        ("regression", 31, 0.5, 6, {"use_missing": False}),
        ("regression", 7, 0.1, 10, {"min_gain_to_split": 1e6}),  # stumps
        ("binary", 7, 0.1, 10, {"sigmoid": 1.7}),
        ("poisson", 7, 0.1, 10, {}),
        ("multiclass", 5, 0.1, 6, {}),
    ]

    def test_parse(self):
        with tempfile.TemporaryDirectory() as d:
            p = train(1, Path(d) / "m.txt", "regression", 7)
            src = qf_formats.open_source(p)
            self.assertTrue(src.linear)
            f = qf.load_forest([p])
            self.assertEqual(f.requires, {"sum64", "linear_leaves", "categorical"})
            t = f.models[0].trees[0]
            self.assertEqual(set(t.linear), set(np.flatnonzero(t.left == -1)))  # every leaf has its linear model
            with self.assertRaises(ValueError) as cm:
                src.leaf_matrix(rows(1, 10))
            self.assertIn("cannot be pruned", str(cm.exception))

    def test_numpy_reference_walker_and_generated_code_equal_lightgbm(self):
        worst = {}
        for i, (obj, leaves, lam, rounds, extra) in enumerate(self.CASES):
            with tempfile.TemporaryDirectory() as d:
                p = train(10 + i, Path(d) / "m.txt", obj, leaves, lam, rounds, extra=extra)
                x = rows(i, big=1e6 if obj == "regression" else 20.0)
                lib = qf_formats.open_source(p).predict(x)
                forest, pred = flat_for([p], Path(d) / "m.flat")
                ir = qf.ir_predict(forest, x)
                got = pred.predict(x)
                for name, v in (("numpy reference", ir), ("walker", got)):
                    e = rel_err(lib, v)
                    worst[(obj, leaves, lam, name)] = e
                    self.assertLess(e, TOL, f"{obj} leaves {leaves} lambda {lam} {name}: {e:.2e}")
                if obj == "regression":  # identity objective: the walker IS the numpy double arithmetic, rounded to float32
                    self.assertTrue(np.array_equal(got, ir.astype(np.float32).reshape(got.shape)), f"{obj} {leaves}")
                self.assertTrue(np.array_equal(got, pred.predict(np.tile(x, (5, 1)), 3)[: len(x)]), "threads")
                pred.close()
                if have_rustc():
                    for be in ("codegen-block", "codegen"):
                        out = Path(d) / f"m-{be}.so"
                        backends.get(be).compile(forest, out, {})
                        cg = backends.get(be).load(out).predict(x)
                        if obj == "regression":
                            self.assertTrue(np.array_equal(cg, got), f"{be} {leaves} leaves not bit-identical to the walker")
                        self.assertLess(rel_err(lib, cg), TOL, be)

    def test_nan_in_a_leaf_feature_gives_the_plain_leaf_value_and_nothing_else_does(self):
        """Planted rows: NaN in a feature of the leaf's linear model but not in any split; 0.0 and the zero bucket are numbers."""
        with tempfile.TemporaryDirectory() as d:
            p = train(77, Path(d) / "m.txt", "regression", 9, 0.1, 10, cat=False)
            forest, pred = flat_for([p], Path(d) / "m.flat")
            src = qf_formats.open_source(p)
            used = {int(f) for t in forest.models[0].trees for (_, _, fs, _) in t.linear.values() for f in fs}
            split = {int(f) for t in forest.models[0].trees for f in t.feature[t.left != -1]}
            only_linear = sorted(used - split)
            self.assertTrue(only_linear or used, "the model uses no linear feature")
            base = np.full((1, NFEAT), 0.5, np.float32)
            for f in only_linear or sorted(used):
                for v in (np.nan, 0.0, 1e-36, -1e-36, 1e-35, 1.0):
                    x = base.copy()
                    x[0, f] = v
                    lib, got = src.predict(x), pred.predict(x)
                    self.assertLess(rel_err(lib, got), TOL, f"feature {f} = {v}")
            nan_rows = np.tile(base, (len(used), 1))
            for k, f in enumerate(sorted(used)):
                nan_rows[k, f] = np.nan
            self.assertLess(rel_err(src.predict(nan_rows), pred.predict(nan_rows)), TOL)
            pred.close()

    def test_k_fold_folder_both_averages(self):
        with tempfile.TemporaryDirectory() as d:
            for k in range(3):
                train(30 + k, Path(d) / f"fold{k}.txt", "regression", 5 + k, 0.1 * k, 8)
            x = rows(5)
            for avg in ("output", "margin"):
                files = qf_formats.model_files(d)
                _, pred = flat_for(files, Path(d) / f"m-{avg}.flat", avg)
                lib = qf.reference_predict(files, x, avg)
                self.assertLess(rel_err(lib, pred.predict(x)), TOL, avg)
                bm = np.random.default_rng(3).uniform(-3, 3, (len(x), 1))
                self.assertLess(rel_err(qf.reference_predict(files, x, avg, bm), pred.predict(x, base_margin=bm)), TOL, avg)
                pred.close()

    def test_base_margin_with_linear_leaves_every_objective(self):
        for i, (obj, leaves, lam, rounds, extra) in enumerate(self.CASES):
            with tempfile.TemporaryDirectory() as d:
                p = train(50 + i, Path(d) / "m.txt", obj, leaves, lam, rounds, extra=extra)
                x = rows(i, 3000, 1e6 if obj == "regression" else 20.0)
                forest, pred = flat_for([p], Path(d) / "m.flat")
                bm = np.random.default_rng(i).uniform(-2, 2, (len(x), forest.groups))
                want = qf.reference_predict([p], x, "output", bm)
                got = pred.predict(x, base_margin=bm)
                self.assertLess(rel_err(want, got), TOL, obj)
                self.assertTrue(np.array_equal(got, pred.predict(x, 3, base_margin=bm)), f"{obj}: threads change the margin result")
                self.assertLess(rel_err(qf.ir_predict(forest, x, base_margin=bm), got), TOL, obj)
                self.assertGreater(np.abs(got - pred.predict(x)).max(), 0, "the margin changed nothing")
                if have_rustc() and obj in ("regression", "multiclass"):
                    out = Path(d) / "m.so"
                    backends.get("codegen-block").compile(forest, out, {})
                    cg = backends.get("codegen-block").load(out)
                    self.assertTrue(np.array_equal(cg.predict(x, base_margin=bm), got) or obj != "regression")
                    self.assertLess(rel_err(want, cg.predict(x, 3, base_margin=bm)), TOL, obj)
                pred.close()

    def test_absolute_error_is_the_float32_rounding_of_the_librarys_answer(self):
        """A 1e-6 ABSOLUTE bar on a log-scale target (~12): the compiled answer is the library's, rounded to float32 (<= half an ulp:
        4.8e-7 for values in 8..16, 9.5e-7 in 16..32), whatever the margin."""
        with tempfile.TemporaryDirectory() as d:
            p = train(9, Path(d) / "m.txt", "regression", 15, 1.0, 20)
            x = rows(9, 2000, big=3.0)
            _, pred = flat_for([p], Path(d) / "m.flat")
            bm = 12.0 + np.random.default_rng(1).uniform(-3, 3, len(x))
            want = qf.reference_predict([p], x, "output", bm.reshape(-1, 1)).reshape(-1)
            got = pred.predict(x, base_margin=bm).astype(np.float64)
            half_ulp = 0.5 * np.spacing(np.abs(want).astype(np.float32)).astype(np.float64)
            self.assertTrue((np.abs(got - want) <= half_ulp * 1.01).all(), f"worst {np.abs(got - want).max():.2e}")
            self.assertTrue(np.abs(got - want).max() > 0)
            pred.close()

    def test_a_margin_of_the_wrong_shape_or_not_finite_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = train(2, Path(d) / "m.txt")
            _, pred = flat_for([p], Path(d) / "m.flat")
            x = rows(1, 20)
            for bad in (np.zeros((19, 1)), np.zeros((20, 2)), np.full((20, 1), np.nan), np.full(20, np.inf)):
                with self.assertRaises(ValueError):
                    pred.predict(x, base_margin=bad)
            self.assertEqual(pred.predict(x, base_margin=np.zeros(20)).shape, (20,))  # (rows,) is fine for one group
            pred.close()

    def test_wave_engines_and_pruning_refuse_linear_leaves(self):
        import compile

        with tempfile.TemporaryDirectory() as d:
            p = train(3, Path(d) / "m.txt")
            for be in ("wave", "wave-avx2"):
                with self.assertRaises(ValueError) as cm:
                    compile.compile_model(p, Path(d) / "m.art", be)
                self.assertIn("linear_leaves", str(cm.exception))
            import choose

            names, skipped = choose.candidates(qf.load_forest([p]))
            self.assertNotIn("wave", names)
            self.assertIn("linear_leaves", skipped["wave"])
            self.assertIn("flat", names)
            # the C ABI: the wavefront engine on a linear file is an error, an unknown engine is 2, a null margin is the plain call
            lib = evaluator.load_lib(evaluator.find_lib())
            _, pred = flat_for([p], Path(d) / "m.flat")
            x = rows(1, 50)
            out = np.empty((50, 1), np.float32)
            call = lambda engine, bm=None: lib.tfl_predict_margin(  # noqa: E731
                pred._h, x.ctypes.data, 50, out.ctypes.data, 1, engine, bm.ctypes.data if bm is not None else None
            )
            self.assertEqual(call(0), 0)
            self.assertTrue(np.array_equal(out[:, 0], pred.predict(x)))
            self.assertEqual(call(1), 1)
            self.assertEqual(call(9), 2)
            bm = np.full((50, 1), 5.0)
            self.assertEqual(call(0, bm), 0)
            self.assertLess(rel_err(out, pred.predict(x) + 5.0), 1e-6)
            pred.close()
            # prune: a clear message, not a traceback
            y = np.nan_to_num(rows(4, 400))
            np.savez(Path(d) / "sel.npz", X=y)
            r = subprocess.run(
                [
                    sys.executable,
                    "-s",
                    str(SCRIPTS / "prune.py"),
                    str(p),
                    str(Path(d) / "sel.npz"),
                    str(Path(d) / "out"),
                    "--keep",
                    "50%",
                    "--regression",
                    "raw",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("cannot be pruned", r.stdout + r.stderr)

    def test_broken_linear_files_are_refused_by_the_loader(self):
        with tempfile.TemporaryDirectory() as d:
            p = train(4, Path(d) / "m.txt")
            _, pred = flat_for([p], Path(d) / "m.flat")
            pred.close()
            blob = (Path(d) / "m.flat").read_bytes()
            self.assertEqual(blob[8], backend_flat.FORMAT_LINEAR)  # written only because of the linear leaves
            lib = evaluator.load_lib(evaluator.find_lib())
            err = ctypes.create_string_buffer(256)

            def opens(b):
                q = Path(d) / "x.flat"
                q.write_bytes(b)
                h = lib.tfl_open(str(q).encode(), err, len(err))
                if h:
                    lib.tfl_close(h)
                return bool(h)

            self.assertTrue(opens(blob))
            self.assertFalse(opens(blob[:-8]))  # truncated
            sums = evaluator.seal  # the checksum put right, so that the check behind it is the one that refuses
            body = bytearray(blob)
            body[-1] ^= 0x01
            self.assertFalse(opens(bytes(body)))  # checksum
            bad = bytearray(blob)
            bad[-16:-12] = (NFEAT + 5).to_bytes(4, "little")  # the last coefficient's feature out of range
            self.assertFalse(opens(sums(bytes(bad))))
            bad = bytearray(blob)
            bad[-8:] = np.float64(np.nan).tobytes()  # NaN coefficient
            self.assertFalse(opens(sums(bytes(bad))))
            bad = bytearray(blob)
            bad[8] = backend_flat.FORMAT_DOUBLE  # the flag without the format
            self.assertFalse(opens(sums(bytes(bad))))
            bad = bytearray(blob)
            bad[48] &= ~1  # linear leaves without double sums
            self.assertFalse(opens(sums(bytes(bad))))


class LinearSyntheticTest(unittest.TestCase):
    """numpy-only random forests with linear leaves attached (deep trees, many features per leaf, folds): the walker and the generated
    code against `ir_predict`."""

    @staticmethod
    def with_linear(forest, seed):
        rng = np.random.default_rng(seed)
        for m in forest.models:
            for t in m.trees:
                t.linear = {}
                for k in np.flatnonzero(t.left == -1):
                    n = int(rng.integers(0, 6))
                    feats = np.sort(rng.choice(7, n, replace=False)).astype(np.int32)  # not the category-code columns (7, 8)
                    scale = rng.choice([1e-3, 1.0, 1e3]) if forest.kind == "identity" else 0.02  # exp and softmax overflow otherwise
                    t.linear[int(k)] = (float(rng.normal()), float(np.float32(t.value[k])), feats, rng.normal(size=n) * scale)
        return forest

    def test_walker_and_generated_code_against_the_numpy_reference(self):
        for seed, kind, groups, folds, avg in [(1, "identity", 1, 1, "output"), (2, "exp", 1, 2, "margin"), (3, "softmax", 3, 2, "output")]:
            f = self.with_linear(random_forest(seed, kind, groups, 30, True, True, True, folds, avg), seed)
            x = random_rows(seed, 2500)
            bm = np.random.default_rng(seed).uniform(-1, 1, (len(x), groups))
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "m.flat"
                backends.get("flat").compile(f, out, {})
                pred = evaluator.FlatForest(out)
                for margin in (None, bm):
                    want = qf.ir_predict(f, x, base_margin=margin).reshape(len(x), -1)
                    got = pred.predict(x, base_margin=margin).reshape(len(x), -1)
                    self.assertLess(rel_err(want, got), TOL, f"{kind} walker, margin {margin is not None}")
                    self.assertTrue(np.array_equal(got, pred.predict(x, 3, base_margin=margin).reshape(len(x), -1)))
                    if kind == "identity":
                        self.assertTrue(np.array_equal(got, want.astype(np.float32)))
                pred.close()
                if have_rustc():
                    so = Path(d) / "m.so"
                    backends.get("codegen-block").compile(f, so, {})
                    cg = backends.get("codegen-block").load(so)
                    self.assertLess(rel_err(qf.ir_predict(f, x, base_margin=bm), cg.predict(x, base_margin=bm)), TOL, kind)
                    if kind == "identity":
                        self.assertTrue(np.array_equal(cg.predict(x), qf.ir_predict(f, x).astype(np.float32).reshape(-1)))


class BaseMarginTest(unittest.TestCase):
    """The base margin on every backend, with the library's own answer as the reference (check --margin)."""

    def run_all(self, model, backend_names):
        x = np.load(FIX / "x.npy")
        for name in backend_names:
            if name.startswith("codegen") and not have_rustc():
                continue
            be = backends.get(name)
            with tempfile.TemporaryDirectory() as d:
                art = Path(d) / f"m{be.suffix}"
                forest = qf.load_forest(qf_formats.model_files(model))
                if forest.requires - set(be.supports):
                    continue
                be.compile(forest, art, {})
                np.savez(Path(d) / "x.npz", X=x)
                r = check.run(model, art, name, Path(d) / "x.npz", n_random=2000, margin=True)
                self.assertTrue(r["ok"], f"{name}: {r['sets']}")
                self.assertTrue(all(s["base_margin"] for s in r["sets"]))

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_xgboost_forests_on_every_backend(self):
        for m in ("xgb_reg.json", "xgb_multi.ubj"):
            self.run_all(FIX / m, ["flat", "wave", "wave-avx2", "codegen", "codegen-block"])

    @unittest.skipIf(lgb is None, "no lightgbm")
    def test_lightgbm_forests(self):
        for m in ("lgb_plain.txt", "lgb_ext.txt"):
            self.run_all(FIX / m, ["flat", "codegen-block"])

    def test_ydf_forests_with_a_margin(self):
        try:
            import ydf
        except ImportError:
            self.skipTest("no ydf")
        x, y, _ = data(8)
        d = {f"f{i}": x[:, i].astype(np.float32) for i in range(NFEAT)}
        for lrn in (ydf.RandomForestLearner, ydf.GradientBoostedTreesLearner):
            m = lrn(label="y", task=ydf.Task.REGRESSION, num_trees=15, max_depth=5, random_seed=1).train({**d, "y": y}, verbose=0)
            with tempfile.TemporaryDirectory() as t:
                m.save(str(Path(t) / "m"))
                forest = qf.load_forest(qf_formats.model_files(Path(t) / "m"))
                art = Path(t) / "m.flat"
                backends.get("flat").compile(forest, art, {})
                np.savez(Path(t) / "x.npz", X=x.astype(np.float32))
                r = check.run(Path(t) / "m", art, "flat", Path(t) / "x.npz", n_random=1000, margin=True)
                # relative bar; YDF's own float32 sums drift ~1e-6 ABSOLUTE even at residual scale (measured 7e-7 on real-like rows, 1.2e-6
                # on threshold-hugging ones)
                self.assertTrue(r["ok"], r["sets"])
                self.assertTrue(all(s["base_margin"] for s in r["sets"]))

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_a_margin_is_added_to_xgboost_not_swapped_for_its_base_score(self):
        f = FIX / "xgb_reg.json"
        x = np.load(FIX / "x.npy")
        with tempfile.TemporaryDirectory() as d:
            _, pred = flat_for([f], Path(d) / "m.flat")
            zero = pred.predict(x, base_margin=np.zeros(len(x)))
            self.assertTrue(np.array_equal(zero, pred.predict(x)))
            shifted = pred.predict(x, base_margin=np.full(len(x), 10.0))
            self.assertLess(np.abs((shifted - zero) - 10.0).max(), 1e-4)
            pred.close()


class OldForestsStayTheSameTest(unittest.TestCase):
    """expected.json holds the SHA-256 of the packed file (checksum field 0) of each fixture. It was made by the compiler BEFORE linear leaves and the
    base margin; the format numbers moved up by three when the checksum began to cover the header (the only bytes that changed, byte 8)."""

    def test_byte_identical(self):
        expected = json.loads((FIX / "expected.json").read_text())
        for name, want in expected.items():
            if (name.startswith("xgb") and not HAVE_XGB) or (name.startswith("lgb") and lgb is None):
                continue
            forest = qf.load_forest([FIX / name])
            blob = backend_flat.pack(forest, lambda b: 0)
            self.assertEqual(hashlib.sha256(blob).hexdigest(), want, name)
            fx, fd = backend_flat.FORMAT_XGB, backend_flat.FORMAT_DOUBLE
            self.assertEqual(blob[8], {"xgb_reg.json": fx, "xgb_multi.ubj": fx, "lgb_plain.txt": fd, "lgb_ext.txt": fd}[name], name)


if __name__ == "__main__":
    unittest.main()
