"""The flat file and its walker against the numpy reference (`qf_common.ir_predict`) on random forests that use every LightGBM / YDF
extension: double sums, exp, identity with several groups, zero-as-missing nodes and categorical nodes. Synthetic only; needs numpy
(no lightgbm, no ydf, no xgboost).

    python3 -s -m unittest discover -s tests
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import backend_codegen
import backend_flat
import backends
import compile
import evaluator
import qf_common as qf
import qf_formats

NF = 9
CAT_FEATURES = (7, 8)  # these features hold integer-like category codes


def random_tree(rng, group, depth, cat, zm):
    left, right, feat, val, dl, zmask, cats = [], [], [], [], [], [], {}

    def node(d):
        i = len(left)
        for a in (left, right, feat, val, dl, zmask):
            a.append(0)
        if d == 0 or (d < depth - 1 and rng.random() < 0.15):
            left[i], right[i], feat[i], val[i] = -1, -1, 0, rng.normal() * 0.3
            return i
        if cat and rng.random() < 0.3:
            f = int(rng.choice(CAT_FEATURES))
            cats[i] = np.sort(rng.choice(40, size=int(rng.integers(1, 12)), replace=False)).astype(np.int64)
            val[i] = 0.0
        else:
            f = int(rng.integers(0, 7))
            val[i] = np.float32(rng.choice([rng.normal(), 0.0, 1e-36, -1e-36, 2e-35]))
            zmask[i] = bool(zm and rng.random() < 0.4)
        feat[i], dl[i] = f, bool(rng.random() < 0.5)
        left[i] = node(d - 1)
        right[i] = node(d - 1)
        return i

    node(depth)
    return qf.Tree(
        group,
        np.array(left, np.int32),
        np.array(right, np.int32),
        np.array(feat, np.int32),
        np.array(val, np.float32),
        np.array(dl, bool),
        np.array(zmask, bool) if zm else None,
        cats or None,
    )


def random_forest(seed, kind, groups, n_trees, cat, zm, sum64, folds=1, average="output"):
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(folds):
        trees = [random_tree(rng, t % groups, int(rng.integers(2, 7)), cat, zm) for t in range(n_trees)]
        models.append(qf.ModelIR(np.float32(rng.normal(size=groups) * 0.1), trees))
    return qf.Forest(NF, groups, kind, average, models, "auto", sum64)


def random_rows(seed, n):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, NF)).astype(np.float32)
    pick = rng.random((n, NF))
    x[pick < 0.08] = np.nan
    x[(pick > 0.08) & (pick < 0.2)] = rng.choice(
        np.array([0.0, -0.0, 1e-36, -1e-36, 2e-35, 1e-35], np.float32), size=((pick > 0.08) & (pick < 0.2)).sum()
    )
    for f in CAT_FEATURES:
        x[:, f] = rng.choice(
            np.array([0, 1, 2, 3, 5, 8, 13, 21, 31, 32, 33, 39, 40, 41, 100, 5000, -1, -3, -0.5, 3.7, 12.99, np.nan], np.float32), size=n
        )
    return x


def flat_predict(forest, x, threads=1):
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.flat"
        backends.get("flat").compile(forest, out, {})
        pred = backends.get("flat").load(out)
        got = pred.predict(x, threads)
        pred.close()
    return got


class ExtensionsTest(unittest.TestCase):
    def check(self, forest, tol, n=1500, seed=3):
        x = random_rows(seed, n)
        want = qf.ir_predict(forest, x)
        got = flat_predict(forest, x)
        want = want.reshape(len(x), -1)
        got = got.reshape(len(x), -1)
        err = np.abs(got.astype(np.float64) - want) / np.maximum(1.0, np.abs(want))
        self.assertLess(err.max(), tol, f"max relative error {err.max():.2e}")
        self.assertTrue(np.array_equal(got, flat_predict(forest, x, threads=3).reshape(len(x), -1)))

    def test_double_sums_over_500_trees(self):
        # float32 sums drift over the bar here (LightGBM, 500 trees); the double-sum flag keeps it under 1e-6
        self.check(random_forest(1, "identity", 1, 500, False, False, True), 1e-6)

    def test_exp_objective(self):
        self.check(random_forest(2, "exp", 1, 120, False, False, True), 1e-6)

    def test_identity_with_several_groups(self):
        self.check(random_forest(3, "identity", 4, 60, False, False, True), 1e-6)

    def test_zero_as_missing_nodes(self):
        self.check(random_forest(4, "sigmoid", 1, 150, False, True, True), 1e-6)

    def test_categorical_nodes(self):
        self.check(random_forest(5, "identity", 1, 150, True, False, True), 1e-6)

    def test_everything_at_once_softmax_two_folds_margin_average(self):
        f = random_forest(6, "softmax", 3, 90, True, True, True, folds=2, average="margin")
        self.check(f, 1e-6)
        self.check(random_forest(7, "softmax", 3, 90, True, True, True, folds=2, average="output"), 1e-6)

    def test_float32_sums_with_extensions(self):
        # xgboost-style float32 sums still work with the extended nodes (compared with a loose bound: the reference sums in double)
        self.check(random_forest(8, "exp", 1, 30, True, True, False), 1e-4)

    def test_plain_xgboost_style_forest_stays_in_the_xgboost_format(self):
        f = random_forest(9, "sigmoid", 1, 20, False, False, False)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.flat"
            backends.get("flat").compile(f, out, {})
            self.assertEqual(int.from_bytes(out.read_bytes()[8:12], "little"), backend_flat.FORMAT_XGB)

    def test_generated_code_runs_the_extensions_bit_for_bit(self):
        try:
            backend_codegen.toolchain_bin()
        except FileNotFoundError:
            self.skipTest("no rustc")
        f = random_forest(11, "softmax", 3, 40, True, True, True, folds=2)
        x = random_rows(5, 800)
        want = flat_predict(f, x)
        for name in ("codegen", "codegen-block"):
            with tempfile.TemporaryDirectory() as d:
                be, out = backends.get(name), Path(d) / "m.so"
                be.compile(f, out, {})
                got = be.load(out).predict(x)
            self.assertTrue(np.array_equal(got, want), name)
        g = random_forest(12, "exp", 1, 40, True, True, False)
        want = flat_predict(g, x)
        with tempfile.TemporaryDirectory() as d:
            be, out = backends.get("codegen-block"), Path(d) / "m.so"
            be.compile(g, out, {})
            self.assertTrue(np.array_equal(be.load(out).predict(x), want))

    def test_infinite_thresholds_run_in_the_flat_file_and_the_generated_code(self):
        """LightGBM writes `inf` as the threshold of a NaN-versus-the-rest split; the generated code accepts it."""
        inf = np.float32(np.inf)
        tree = qf.Tree(
            0,
            np.array([1, 3, -1, -1, -1], np.int32),
            np.array([2, 4, -1, -1, -1], np.int32),
            np.array([0, 1, 0, 0, 0], np.int32),
            np.array([inf, -inf, 5.0, 1.0, 2.0], np.float32),
            np.array([False, True, False, False, False]),
        )
        f = qf.Forest(NF, 1, "identity", "output", [qf.ModelIR(np.float32([0.25]), [tree])], "auto", True)
        x = np.zeros((12, NF), np.float32)
        x[:, 0] = [0, 1, -1, 3e38, -3e38, np.nan, np.nan, 0, 1, np.nan, 5, -5]
        x[:, 1] = [0, 1, -1, 3e38, -3e38, np.nan, 0, np.nan, -3e38, 3e38, np.nan, 7]
        want = qf.ir_predict(f, x).astype(np.float32).reshape(-1)
        self.assertTrue(np.array_equal(flat_predict(f, x).reshape(-1), want))
        self.assertIn("f32::INFINITY", backend_codegen.tree_expr(tree))
        try:
            backend_codegen.toolchain_bin()
        except FileNotFoundError:
            return
        with tempfile.TemporaryDirectory() as d:
            be, out = backends.get("codegen-block"), Path(d) / "m.so"
            be.compile(f, out, {})
            self.assertTrue(np.array_equal(be.load(out).predict(x).reshape(-1), want))

    def test_engines_that_cannot_run_the_extensions_refuse_cleanly(self):
        f = random_forest(10, "identity", 1, 20, True, False, True)
        self.assertTrue(f.requires)
        for name in ("wave", "wave-avx2"):
            self.assertTrue(f.requires - set(getattr(backends.get(name), "supports", ())), name)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.flat"
            backends.get("flat").compile(f, out, {})
            lib = evaluator.load_lib(evaluator.find_lib())
            self.assertIsNotNone(lib)
            pred = evaluator.FlatForest(out, engine=1)
            with self.assertRaises(RuntimeError):
                pred.predict(random_rows(1, 50))
            pred.close()


def one_split_forest(feature: int, n_features: int, zero_missing: bool = True) -> qf.Forest:
    """One tree that splits on `feature` (x < 0.5 goes left to 1.0, else 2.0); a zero-as-missing node makes the file use extended nodes."""
    tree = qf.Tree(
        0,
        np.array([1, -1, -1], np.int32),
        np.array([2, -1, -1], np.int32),
        np.array([feature, 0, 0], np.int32),
        np.array([0.5, 1.0, 2.0], np.float32),
        np.array([False, False, False]),
        np.array([zero_missing, False, False]) if zero_missing else None,
    )
    return qf.Forest(n_features, 1, "identity", "output", [qf.ModelIR(np.float32([0.0]), [tree])], "auto", True)


class ExtendedFeatureLimitTest(unittest.TestCase):
    """Extended nodes (categorical or zero-as-missing) hold a 13-bit feature index: a split on index 8191 (the model then has 8,192
    features) works, one on index 8192 is refused with the message that says 'at most 8,192 features' (as LIMITS.md does; 8,191 was
    one below what the code accepts). Plain nodes hold 15 bits."""

    LAST = 0x1FFF

    def test_a_split_on_the_last_extended_feature_works(self):
        f = one_split_forest(self.LAST, self.LAST + 1)
        x = np.zeros((4, self.LAST + 1), np.float32)
        x[:, self.LAST] = [0.0, 0.25, 0.75, np.nan]  # zero and NaN take the node's default way (right); 0.25 goes left
        got = flat_predict(f, x).reshape(-1)
        self.assertTrue(np.array_equal(got, qf.ir_predict(f, x).astype(np.float32).reshape(-1)))
        self.assertEqual(got.tolist(), [2.0, 1.0, 2.0, 2.0])

    def test_one_feature_over_is_refused_with_the_documented_message(self):
        f = one_split_forest(self.LAST + 1, self.LAST + 2)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as cm:
                backends.get("flat").compile(f, Path(d) / "m.flat", {})
        msg = str(cm.exception)
        self.assertIn("feature index is out of range", msg)
        self.assertIn("at most 8,192 features with categorical or zero-as-missing nodes", msg)

    def test_the_same_split_in_a_plain_model_is_accepted(self):
        f = one_split_forest(self.LAST + 1, self.LAST + 2, zero_missing=False)  # no extended node: 15-bit index
        x = np.zeros((3, self.LAST + 2), np.float32)
        x[:, self.LAST + 1] = [0.25, 0.75, np.nan]
        self.assertEqual(flat_predict(f, x).reshape(-1).tolist(), [1.0, 2.0, 2.0])

    def test_the_command_line_refusal_is_one_line_and_exit_3(self):
        f = one_split_forest(self.LAST + 1, self.LAST + 2)
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            argv = ["compile.py", "model.txt", str(Path(d) / "m.flat"), "--backend", "flat"]
            patches = (
                mock.patch.object(qf_formats, "model_files", lambda m: [m]),
                mock.patch.object(qf, "load_forest", lambda *a: f),
                mock.patch.object(sys, "argv", argv),
            )
            with contextlib.ExitStack() as stack, contextlib.redirect_stderr(err):
                for p in patches:
                    stack.enter_context(p)
                with self.assertRaises(SystemExit) as cm:
                    compile.main()
        self.assertEqual(cm.exception.code, 3)
        self.assertEqual(len(err.getvalue().strip().splitlines()), 1)
        self.assertIn("at most 8,192 features", err.getvalue())


def need_rustc(test):
    try:
        backend_codegen.toolchain_bin()
    except FileNotFoundError as e:
        test.skipTest(str(e))


class EmptyCategorySetTest(unittest.TestCase):
    """A categorical node whose set holds no category sends every row right (LightGBM does; the walker does). The generated code wrote
    `matches!(x as i64, -1)` for an empty set, which sent x in (-2, -1] left."""

    def forest(self):
        tree = qf.Tree(
            0,
            np.array([1, -1, -1], np.int32),
            np.array([2, -1, -1], np.int32),
            np.array([0, 0, 0], np.int32),
            np.array([0.0, 1.0, 2.0], np.float32),
            np.array([False, False, False]),
            None,
            {0: np.array([], np.int64)},
        )
        return qf.Forest(1, 1, "identity", "output", [qf.ModelIR(np.float32([0.0]), [tree])], "auto", True)

    def test_every_backend_sends_every_row_right(self):
        need_rustc(self)
        x = np.array([[v] for v in (-2.0, -1.9, -1.5, -1.0, -0.5, 0.0, 1.0, 40.0, np.nan)], np.float32)
        f = self.forest()
        self.assertEqual(qf.ir_predict(f, x).reshape(-1).tolist(), [2.0] * len(x))
        self.assertEqual(flat_predict(f, x).reshape(-1).tolist(), [2.0] * len(x))
        with tempfile.TemporaryDirectory() as d:
            backends.get("codegen-block").compile(f, Path(d) / "m.so", {})
            pred = backends.get("codegen-block").load(Path(d) / "m.so")
            self.assertEqual(pred.predict(x).reshape(-1).tolist(), [2.0] * len(x))


class GeneratedCodeOutputPathTest(unittest.TestCase):
    """rustc and the linker read a leading `-` and a `%` in a path as an option and an escape: the sources are built in a clean temp
    folder and only the finished library is copied to the output, so the output path may be called anything."""

    def build(self, out, cwd=None):
        need_rustc(self)
        f = one_split_forest(0, 2, zero_missing=False)
        here = os.getcwd()
        try:
            if cwd:
                os.chdir(cwd)
            backends.get("codegen-block").compile(f, Path(out), {})
            pred = backends.get("codegen-block").load(Path(out))
            got = pred.predict(np.array([[0.25, 0.0], [0.75, 0.0]], np.float32)).reshape(-1).tolist()
        finally:
            os.chdir(here)
        self.assertEqual(got, [1.0, 2.0])

    def test_a_relative_path_whose_first_part_starts_with_a_dash(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "-lead").mkdir()
            self.build("./-lead/m.so", cwd=d)

    def test_a_percent_sign_in_the_path(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "50%done").mkdir()
            self.build(Path(d) / "50%done" / "m.so")

    def test_keep_src_copies_the_sources_next_to_the_output_and_nothing_else_stays_in_it(self):
        need_rustc(self)
        f = one_split_forest(0, 2, zero_missing=False)
        with tempfile.TemporaryDirectory() as d:
            info = backends.get("codegen-block").compile(f, Path(d) / "m.so", {"keep_src": True})
            self.assertEqual(Path(info["src_dir"]).parent, Path(d))
            self.assertTrue((Path(info["src_dir"]) / "lib.rs").is_file())
            info = backends.get("codegen-block").compile(f, Path(d) / "n.so", {})
            self.assertEqual(sorted(p.name for p in Path(d).iterdir() if p.name.startswith("n")), ["n.so"])

    def test_a_failed_build_is_one_line_and_exit_3(self):
        """rustc failing is a RuntimeError (not the user's input) that is also a ValueError, so every command reports it like a refusal."""
        need_rustc(self)
        self.assertTrue(issubclass(backend_codegen.BuildFailed, (RuntimeError, ValueError)))
        f = one_split_forest(0, 2, zero_missing=False)
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            argv = ["compile.py", "model.txt", str(Path(d) / "m.so"), "--backend", "codegen-block"]
            patches = (
                mock.patch.object(qf_formats, "model_files", lambda m: [m]),
                mock.patch.object(qf, "load_forest", lambda *a: f),
                mock.patch.object(backend_codegen, "run_measured", lambda cmd: (1, "boom", 0.0)),
                mock.patch.object(sys, "argv", argv),
            )
            with contextlib.ExitStack() as stack, contextlib.redirect_stderr(err):
                for p in patches:
                    stack.enter_context(p)
                with self.assertRaises(SystemExit) as cm:
                    compile.main()
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), [])  # the failed build left nothing behind
        self.assertEqual(cm.exception.code, 3)
        lines = [ln for ln in err.getvalue().strip().splitlines() if not ln.startswith("RAM: building with")]  # the plan's line comes first
        self.assertEqual(len(lines), 1)
        self.assertIn("rustc failed", err.getvalue())


if __name__ == "__main__":
    unittest.main()
