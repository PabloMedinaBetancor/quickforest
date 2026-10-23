"""Regression tests for the cases the random-model differential run (tests/differential/run_differential.py) failed on. Each class
reproduces a failing case, or its cause, in a few lines, so the fix stays fixed without a three-hour run.

    PYTHONPATH=<venv with lightgbm, xgboost> python3 -s -m unittest test_differential_pins
Skips the parts whose library is missing."""

import contextlib
import io
import itertools
import subprocess
import sys
import tempfile
import textwrap
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE / "differential"))
import backends
import qf_common as qf
import qf_formats
import run_differential as rd
from test_linear import HAVE_XGB, have_rustc, lgb


def large_input_linear_forest():
    """One linear leaf, const + c0 x0 + c1 x1 with |x| up to 1e6, whose answer is near zero on every row: (forest, float32 rows)."""
    const, c0, c1 = -1.0e5, 0.123456789123, -0.0987654321
    x0 = np.random.default_rng(5).uniform(8e5, 9.9e5, 400).astype(np.float32)
    x1 = (-(const + c0 * x0.astype(np.float64)) / c1).astype(np.float32)
    leaf = (const, 0.0, np.array([0, 1], np.int32), np.array([c0, c1]))
    tree = qf.Tree(
        0,
        np.array([-1], np.int32),
        np.array([-1], np.int32),
        np.array([0], np.int32),
        np.array([0.0], np.float32),
        np.array([False]),
        linear={0: leaf},
    )
    forest = qf.Forest(2, 1, "identity", "output", [qf.ModelIR(np.float32([0.0]), [tree])], "auto", True)
    return forest, np.stack([x0, x1], 1)


def replay(case, seed0=1):
    """The failures of one differential case on the flat walker and the wave engines (no generated code)."""
    with tempfile.TemporaryDirectory() as w, contextlib.redirect_stdout(io.StringIO()):
        try:
            chk = rd.run_case(case, seed0, Path(w), rd.new_stat(), "none")
        except rd.Skip as e:
            raise unittest.SkipTest(str(e)) from e
    return chk.failures


class ToleranceTest(unittest.TestCase):
    """The harness's own bars and oracles (a bar that lets a planted defect through proves nothing)."""

    @unittest.skipUnless(lgb is not None, "no lightgbm")
    def test_a_lightgbm_leaf_rounded_the_wrong_way_fails_the_exact_default_file_bar(self):
        """The default-file bar of LightGBM used to be the proven bound of the float32 rounding of the leaves (up to 1e-6 and more), so a
        writer that rounded toward zero instead of to nearest (twice the error, 1e-7 here) passed. Now the oracle is the library on the same
        float32 leaves and the bar is exact."""
        rng = np.random.default_rng(3)
        x = rng.normal(size=(400, 3))
        xs = rng.normal(size=(300, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as w:
            b = lgb.train({"objective": "regression", "num_leaves": 15, "verbose": -1}, lgb.Dataset(x, 700 + 50 * x[:, 0] * x[:, 1]), 8)
            path = Path(w) / "m.txt"
            b.save_model(str(path))
            (oracle,) = rd.lgb_float32_leaf_files([path], Path(w) / "r32")
            want = qf.reference_predict([oracle], xs, "output", None)
            with contextlib.redirect_stdout(io.StringIO()):
                forest = qf.load_forest([path], "output", False)
            scale = rd.forest_scale(forest)
            key, lim = rd.bar("lightgbm", {"folds": 1}, {}, "regression", False, "f64", False, scale)
            self.assertEqual((key, lim), ("lgb", 0.0))
            results = []
            for broken in (False, True):
                if broken:  # round toward zero: the writer's rounding error doubled
                    for t in forest.models[0].trees:
                        leaf = t.left == -1
                        v32 = t.leaf64.astype(np.float32)
                        over = np.abs(v32.astype(np.float64)) > np.abs(t.leaf64)
                        t.value = np.where(leaf & over, np.nextafter(v32, np.float32(0)), np.where(leaf, v32, t.value)).astype(np.float32)
                art = Path(w) / f"{broken}.flat"
                backends.get("flat").compile(forest, art, {})
                pred = backends.get("flat").load(art)
                try:
                    got = pred.predict(xs, 1, dtype=np.float64)
                finally:
                    pred.close()
                chk = rd.Checker(0, "x", Path("."), {}, "none")
                chk.compare(key, "f64 answer vs the library", "flat", {}, want, got, lim)
                results.append(len(chk.failures))
            self.assertEqual(results, [0, 1], "the honest file passes, the truncated one fails")

    def test_threshold_rows_reach_every_split_of_the_last_tree_and_fall_on_both_sides_of_it(self):
        """`check.random_rows` hugs a threshold of the whole forest at random, so a given node of the LAST tree was hit by luck. These rows
        reach each split node and set its feature to the threshold, a float32 step below, a step above and NaN."""
        import qf_common

        left, right = np.array([1, 3, 5, -1, -1, -1, -1], np.int32), np.array([2, 4, 6, -1, -1, -1, -1], np.int32)
        # the root and node 2 split on feature 0: the second one must be reached on its side
        feature = np.array([0, 1, 0, -1, -1, -1, -1], np.int32)
        value = np.array([0.5, -1.25, 3.0, 1, 2, 3, 4], np.float32)
        tree = qf_common.Tree(0, left, right, feature, value, np.array([True, False, True, False, False, False, False]))
        forest = qf_common.Forest(2, 1, "identity", "output", [qf_common.ModelIR(np.zeros(1, np.float32), [tree, tree])])
        rows = rd.threshold_rows(forest, 2)
        self.assertEqual(rows.dtype, np.float32)

        def visited(r):
            node, seen = 0, []
            while left[node] != -1:
                seen.append(node)
                v = r[feature[node]]
                node = (
                    (left[node] if tree.default_left[node] else right[node])
                    if np.isnan(v)
                    else left[node]
                    if v < value[node]
                    else right[node]
                )
            return seen

        hit = {}
        for r in rows:
            for n in visited(r):
                v = r[feature[n]]
                kind = (
                    "nan"
                    if np.isnan(v)
                    else "at"
                    if v == value[n]
                    else "below"
                    if v == np.nextafter(value[n], np.float32(-np.inf))
                    else "above"
                    if v == np.nextafter(value[n], np.float32(np.inf))
                    else None
                )
                if kind:
                    hit.setdefault(int(n), set()).add(kind)
        # node 2 splits on feature 0 again below the root's split on it: a NaN there would take the root's default direction and never get
        # to it
        self.assertEqual(
            {n: sorted(k) for n, k in hit.items()},
            {0: ["above", "at", "below", "nan"], 1: ["above", "at", "below", "nan"], 2: ["above", "at", "below"]},
        )

    @unittest.skipUnless(lgb is not None, "no lightgbm")
    @unittest.skipUnless(lgb is not None, "no lightgbm")
    def test_the_lightgbm_oracle_rounds_the_leaves_after_the_factors_the_compile_folds_in(self):
        """Found by the full run (90 cases): the default file stores float32 of `leaf * sigmoid_s / iterations`, not float32 of the leaf
        that the library then scales in double, so an oracle that rounded the file's leaf was off by a leaf rounding (1e-8 .. 1e-9 of a
        probability) for every binary model with s != 1 and every rf model. The oracle folds the factor in first (`sigmoid:1`, no
        `average_output`), and a base margin goes to it scaled the way the compile scales it."""
        rng = np.random.default_rng(7)
        x = rng.normal(size=(500, 4))
        xs = rng.normal(size=(300, 4)).astype(np.float32)
        bm = rng.uniform(-1, 1, (300, 1))
        cases = {
            "binary-s2": ({"objective": "binary", "sigmoid": 2.0, "num_leaves": 15}, (x[:, 0] + x[:, 1] > 0).astype(float), 12),
            "rf": (
                {
                    "boosting_type": "rf",
                    "bagging_freq": 1,
                    "bagging_fraction": 0.7,
                    "feature_fraction": 0.8,
                    "num_leaves": 15,
                    "objective": "regression",
                },
                30 + 7 * x[:, 0] * x[:, 1] + x[:, 2],
                7,
            ),
        }
        with tempfile.TemporaryDirectory() as w:
            for name, (params, y, rounds) in cases.items():
                path = Path(w) / f"{name}.txt"
                lgb.train({**params, "verbose": -1}, lgb.Dataset(x, y), rounds).save_model(str(path))
                with contextlib.redirect_stdout(io.StringIO()):
                    forest = qf.load_forest([path], "output", False)
                (oracle,) = rd.lgb_float32_leaf_files([path], Path(w) / "r32")
                art = Path(w) / f"{name}.flat"
                backends.get("flat").compile(forest, art, {})
                pred = backends.get("flat").load(art)
                try:
                    got = pred.predict(xs, 1, dtype=np.float64)
                    got_bm = pred.predict(xs, 1, base_margin=bm, dtype=np.float64)
                finally:
                    pred.close()
                want = qf.reference_predict([oracle], xs, "output", None)
                want_bm = qf.reference_predict([oracle], xs, "output", bm * rd.lgb_oracle_folds_factor([path], forest))
                for what, a, b in (("plain", want, got), ("base margin", want_bm, got_bm)):
                    self.assertLessEqual(rd.rel_err(a, b), 1e-12, f"{name} {what}: {rd.rel_err(a, b):.3e}")
                if name == "rf":  # identity objective: the answer is a sum of float32 leaves, exactly the library's on the oracle
                    self.assertTrue(np.array_equal(want.reshape(300, -1), got.reshape(300, -1)))

    def test_the_lightgbm_oracle_rounds_plain_leaves_and_leaves_linear_trees_alone(self):
        """The default LightGBM file stores each plain leaf as a float32, a linear tree keeps its constants in double. The oracle is the
        library on a copy with the plain leaves rounded the same way, so the f64 answer must be EXACTLY the oracle's (no bound), also for
        the one-leaf plain tree of a constant target next to linear trees."""
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, 3))
        with tempfile.TemporaryDirectory() as w:
            lin = lgb.train(
                {"objective": "regression", "linear_tree": True, "num_leaves": 7, "verbose": -1}, lgb.Dataset(x, 5 * x[:, 0] + 3), 5
            )
            const = lgb.train({"objective": "binary", "verbose": -1}, lgb.Dataset(x, np.ones(300)), 3)  # one-leaf trees, log(1e-15)
            plain = lgb.train({"objective": "regression", "num_leaves": 7, "verbose": -1}, lgb.Dataset(x, x[:, 0] * 0.123456789 + 1e-3), 6)
            xs = rng.normal(size=(200, 3)).astype(np.float32)
            for name, b in (("lin", lin), ("const", const), ("plain", plain)):
                path = Path(w) / f"{name}.txt"
                b.save_model(str(path))
                (rounded,) = rd.lgb_float32_leaf_files([path], Path(w) / "r32")
                # byte sizes: dropped, the numbers change
                keep = [ln for ln in path.read_text().splitlines() if not ln.startswith("tree_sizes=")]
                for old, new in zip(keep, rounded.read_text().splitlines(), strict=True):
                    if old.startswith("leaf_value=") and name != "lin":
                        self.assertEqual(
                            [float(np.float32(float(v))) for v in old[11:].split()], [float(v) for v in new[11:].split()], name
                        )
                    if name == "lin":
                        self.assertEqual(old, new, "a linear tree's block is left as it is")
                with contextlib.redirect_stdout(io.StringIO()):
                    forest = qf.load_forest([path], "output", False)
                art = Path(w) / f"{name}.flat"
                backends.get("flat").compile(forest, art, {})
                pred = backends.get("flat").load(art)
                try:
                    got = pred.predict(xs, 1, dtype=np.float64)
                finally:
                    pred.close()
                want = qf.reference_predict([rounded], xs, "output", None)
                if name == "const" or (name == "lin" and rd.fusing_machine()):
                    # const: a sigmoid of a margin, numpy's exp against the evaluator's, an ulp of a double. lin on aarch64: LightGBM's
                    # wheel fuses the multiply-add of a linear leaf (see fused_slack), the evaluator rounds twice: 4e-16 on aarch64
                    self.assertLessEqual(rd.rel_err(want, got), 1e-12, name)
                else:
                    self.assertTrue(
                        np.array_equal(want.reshape(len(xs), -1), got.reshape(len(xs), -1)), f"{name}: {rd.rel_err(want, got):.3e}"
                    )

    def test_the_table_limits_are_the_same_on_every_machine(self):
        """The aarch64 rule is a per-row allowance (`fused_slack`), not a different table limit: `bar` answers the same on x86-64 and on
        ARM, so the x86 bar is exactly what it was (exact 0.0 for the linear-leaf float64 answer)."""

        def limits(machine):
            out = {}
            scale = (1.0, 10, 0.5)
            with mock.patch.object(rd.platform, "machine", return_value=machine):
                for fmt, lin, f64l, rf, kind, tr, dt, bm in itertools.product(
                    ("xgboost", "lightgbm", "ydf"),
                    (False, True),
                    (False, True),
                    (False, True),
                    ("regression", "binary", "multiclass"),
                    (False, True),
                    ("f32", "f64"),
                    (False, True),
                ):
                    meta = {"lin": lin, "folds": 2, "rf": rf}
                    out[fmt, lin, f64l, rf, kind, tr, dt, bm] = rd.bar(fmt, meta, {"f64_leaves": f64l}, kind, tr, dt, bm, scale, "output")
            return out

        x86 = limits("x86_64")
        for machine in ("AMD64", "i686", "riscv64", "aarch64", "arm64", "AARCH64"):
            self.assertEqual(limits(machine), x86, machine)
        with mock.patch.object(rd.platform, "machine", return_value="x86_64"):
            key, lim = rd.bar("lightgbm", {"lin": True, "folds": 1}, {}, "regression", False, "f64", False, (1.0, 10, 0.5), "output")
        self.assertEqual((key, lim), ("lgb-lin", 0.0))

    def test_the_arm_allowance_exists_only_on_fusing_machines(self):
        forest, x = large_input_linear_forest()
        ref = np.zeros((len(x), 1))
        for machine in ("x86_64", "AMD64", "riscv64"):
            with mock.patch.object(rd.platform, "machine", return_value=machine):
                self.assertIsNone(rd.fused_slack(forest, x, ref), machine)
        for machine in ("aarch64", "arm64", "AARCH64"):
            with mock.patch.object(rd.platform, "machine", return_value=machine):
                slack = rd.fused_slack(forest, x, ref)
            self.assertEqual(slack.shape, (len(x),), machine)
            self.assertTrue((slack > 0).all())

    def test_the_arm_allowance_covers_a_fused_answer_near_zero_with_1e6_sized_terms(self):
        """The proof: a linear leaf const + c0 x0 + c1 x1 with inputs near 1e6 and an
        answer near zero. Rounding each multiply-add once (math.fma, here the exact rational sum rounded once) gives what LightGBM's ARM
        wheel returns; the evaluator rounds twice, as LightGBM's x86 wheel does. They differ by about an ulp of the 1e5-sized terms,
        several times the old fixed bar of 1e-12 on some rows, and always inside the per-row allowance; the x86 bar stays exactly 0 and
        flags the difference."""
        forest, x = large_input_linear_forest()
        (const, _fallback, _feats, (c0, c1)) = forest.models[0].trees[0].linear[0]
        with tempfile.TemporaryDirectory() as d:
            backends.get("flat").compile(forest, Path(d) / "m.flat", {})
            pred = backends.get("flat").load(Path(d) / "m.flat")
            try:
                got = pred.predict(x, 1, dtype=np.float64).reshape(-1)
            finally:
                pred.close()

        def fma(a, b, c):  # math.fma (Python 3.13+) without needing it: the exact value, rounded once
            return float(Fraction(a) * Fraction(b) + Fraction(c))

        fused = np.array([fma(c1, float(b), fma(c0, float(a), const)) for a, b in x.astype(np.float64)])
        unfused = np.array([(const + c0 * float(a)) + c1 * float(b) for a, b in x.astype(np.float64)])
        self.assertTrue(np.array_equal(got, unfused), "the evaluator rounds the multiply and the add separately")
        self.assertLess(np.abs(fused).max(), 1.0)  # an answer near zero ...
        self.assertGreater(np.abs(x).max(), 5e5)  # ... from inputs near 1e6
        err = rd.rel_err(fused, got)
        self.assertGreater(err, 1e-12, "the case must exceed the old fixed bar")
        with mock.patch.object(rd.platform, "machine", return_value="aarch64"):
            slack = rd.fused_slack(forest, x, fused.reshape(-1, 1))
        chk = rd.Checker(0, "x", Path("."), {}, "none")
        chk.compare("lgb-lin", "fused vs the evaluator", "flat", {}, fused.reshape(-1, 1), got.reshape(-1, 1), 0.0, slack)
        self.assertEqual(chk.failures, [], "ARM: inside the per-row allowance")
        chk.compare("lgb-lin", "fused vs the evaluator", "flat", {}, fused.reshape(-1, 1), got.reshape(-1, 1), 0.0)
        self.assertEqual(len(chk.failures), 1, "x86: the bar is still exact")
        old = np.abs(got - fused) / np.maximum(1.0, np.abs(fused))
        self.assertTrue((old > 1e-12).any() and (old <= slack).all())
        self.assertLess(slack.max() / 1e-12, 2e6)  # the allowance is 1e-12 x the terms (about 1e5 here), not unbounded

    def test_the_printed_table_states_the_arm_rule(self):
        row = next(r for r in rd.TABLE if r[0] == "lgb-lin")
        self.assertIn("aarch64", row[3])
        self.assertIn("multiply-add", row[4])
        self.assertIn("aarch64 / arm64", rd.table_markdown())

    def test_slack_only_where_the_margin_is_a_sum_of_something_else(self):
        # compare(): without slack the table limit rules; with it a row's own allowance is added
        chk = rd.Checker(0, "x", Path("."), {}, "none")
        ref = np.array([[1.0], [1.0]])
        got = np.array([[1.0], [1.0 + 5e-12]])
        chk.compare("lgb-lin-tr", "t", "flat", {}, ref, got, 1e-12)
        self.assertEqual(len(chk.failures), 1)
        chk.compare("lgb-lin-tr", "t", "flat", {}, ref, got, 1e-12, slack=np.array([0.0, 1e-11]))
        self.assertEqual(len(chk.failures), 1)  # no new failure
        chk.compare("lgb-lin-tr", "t", "flat", {}, ref, got, 1e-12, slack=np.array([1e-11, 0.0]))
        self.assertEqual(len(chk.failures), 2)  # the allowance of another row does not cover this one

    def test_an_answer_beyond_float32_must_be_inf_or_float_max_with_the_libraries_sign_never_error_zero(self):
        """The error of a row whose library answer is beyond float32's range used to be set to 0 whatever the compiled answer was (a 0, a
        NaN or the wrong sign passed). Only +-inf and +-FLT_MAX with the library's sign are as close as float32 can come to it."""
        f32 = np.float32
        fmax = float(np.finfo(np.float32).max)
        for ref in (4.0e38, -5.0e38, np.inf, -np.inf):
            sign = np.sign(ref)
            for good in (sign * np.inf, sign * fmax):
                self.assertEqual(rd.rel_err([ref], np.array([good], f32)), 0.0, (ref, good))
            for bad in (0.0, np.nan, -sign * np.inf, -sign * fmax, 1.0, sign * 3.1e38):
                err = rd.rel_err([ref], np.array([bad], f32))
                self.assertGreater(err, 1e-3, (ref, bad))
        # a library answer just under float32's end is judged by the ordinary error, not skipped
        self.assertLess(rd.rel_err([3.2e38], np.array([f32(3.2e38)], f32)), 1e-7)
        self.assertGreater(rd.rel_err([3.2e38], np.array([0.0], f32)), 0.5)
        # float32 can hold 3.1e38, so FLT_MAX or inf against it is a 9.8 % difference, not saturation (only an answer beyond FLT_MAX is excused)
        for edge in (fmax, np.inf):
            self.assertGreater(rd.rel_err([3.1e38], np.array([edge], f32)), 0.05, edge)
            self.assertGreater(rd.rel_err([-3.1e38], np.array([-edge], f32)), 0.05, edge)
        self.assertEqual(rd.rel_err([fmax], np.array([fmax], f32)), 0.0)
        # and the second oracle that skipped the same rows (the per-row allowance of `Checker.compare`)
        with tempfile.TemporaryDirectory() as w:
            chk = rd.Checker(0, "pin", Path(w), {}, "none")
            chk.compare("k", "label", "flat", {}, [4.0e38, 1.0], np.array([0.0, 1.0], f32), 0.0, slack=np.zeros(2))
            self.assertEqual(len(chk.failures), 1, "a 0 for a library answer beyond float32 passed the per-row oracle")
            chk.failures.clear()
            chk.compare("k", "label", "flat", {}, [4.0e38, 1.0], np.array([np.inf, 1.0], f32), 0.0, slack=np.zeros(2))
            self.assertEqual(chk.failures, [])


class LibraryCopiesTest(unittest.TestCase):
    """A process aborted with `fatal runtime error: out of TLS keys`. Every copy of libtreeflat that spawns a thread takes a pthread key it
    never gives back (Rust's std), a process has 1,024, and every compiled model folder carries its own copy of the library. Copies of one
    build share one loaded instance (`evaluator.load_lib`)."""

    SCRIPT = textwrap.dedent(
        """
        import ctypes, shutil, sys
        from pathlib import Path
        import numpy as np
        sys.path.insert(0, sys.argv[1])
        import compile, evaluator
        root, model, x = Path(sys.argv[2]), Path(sys.argv[3]), np.load(sys.argv[4])
        compile.compile_model(model, root / "m.flat", backend="flat")
        x = np.tile(x, (600 // len(x) + 1, 1))[:600].astype(np.float32)
        lib = evaluator.find_lib()
        libc = ctypes.CDLL("libc.so.6")
        held, k = [], ctypes.c_uint()
        while libc.pthread_key_create(ctypes.byref(k), None) == 0:
            held.append(k.value)
        for v in held[:6]:  # six keys left for the whole process
            libc.pthread_key_delete(ctypes.c_uint(v))
        for i in range(40):  # forty copies of the library, each one predicting on three threads with the wave engine
            d = root / f"c{i}"
            d.mkdir()
            shutil.copy(lib, d / lib.name)
            shutil.copy(root / "m.flat", d / "m.flat")
            f = evaluator.FlatForest(d / "m.flat", engine=1)
            f.predict(x, 3)
            f.close()
        print("ok")
        """
    )

    def test_forty_copies_of_the_library_need_one_instance(self):
        fix = HERE / "fixtures"
        with tempfile.TemporaryDirectory() as w:
            r = subprocess.run(
                [sys.executable, "-s", "-c", self.SCRIPT, str(HERE.parent / "scripts"), w, str(fix / "xgb_reg.json"), str(fix / "x.npy")],
                capture_output=True,
                text=True,
                timeout=300,
            )
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "ok"), r.stderr[-400:])


class ResumeStateTest(unittest.TestCase):
    """A recorded result belongs to a seed0 and a generator, not to a case number: a full run resumed in an --out written under another
    --seed0 (or before the generator changed) used to count those models as done and report them as this run's."""

    def test_results_of_another_seed_or_generator_are_not_reused(self):
        import json

        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            records = [
                {"case": 1, "ok": True, **rd.stamp(1)},  # this run
                {"case": 2, "ok": True, **rd.stamp(2)},  # the same case number under another seed0: another model
                {"case": 3, "ok": True, "seed0": 1, "gen": "an-older-generator"},
                {"case": 4, "ok": True},  # written before records were stamped
                {"trained": 5, **rd.stamp(1)},
                {"trained": 6, **rd.stamp(2)},
                {"start": 7},
            ]
            (out / "worker-0.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\nnot json\n")
            self.assertEqual(sorted(rd.collect(out, 1)), [1])
            self.assertEqual(sorted(rd.collect(out, 2)), [2])
            self.assertTrue(rd.trained(out, 5, 1))
            self.assertFalse(rd.trained(out, 6, 1))
            self.assertFalse(rd.trained(out, 5, 2))


class ReplayedCasesTest(unittest.TestCase):
    """Four failing cases of the first full run. They depend on the libraries' versions (the models are retrained); a version that trains
    other models must still pass them, though they are no longer the same cases."""

    @unittest.skipUnless(lgb is not None, "no lightgbm")
    def test_case_28_exp_of_a_margin_built_from_terms_of_5e4(self):
        self.assertEqual(replay(28), [])

    @unittest.skipUnless(lgb is not None, "no lightgbm")
    def test_case_892_one_leaf_float32_tree_in_a_linear_forest(self):
        self.assertEqual(replay(892), [])

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_case_1071_sigmoid_one_ulp_from_the_base_margin(self):
        self.assertEqual(replay(1071), [])

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_case_1767_sigmoid_one_ulp_from_the_base_margin(self):
        self.assertEqual(replay(1767), [])

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_case_2178_base_margin_over_80_trees_with_answers_near_100(self):
        self.assertEqual(replay(2178), [])

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_case_1230_sigmoid_one_ulp_from_the_base_margin(self):
        self.assertEqual(replay(1230), [])


@unittest.skipUnless(HAVE_XGB, "no xgboost")
class XgboostSoftmaxIsBitExactTest(unittest.TestCase):
    """xgboost's `common::Softmax` sums the exponentials in a DOUBLE and divides by that sum rounded to float32. A float32 sum drifts from
    it as the class count grows (3.6e-7 at 30 classes), so every backend must give xgboost's own bits, at 3 and at 30 classes."""

    def test_multiclass_answers_equal_the_library_bit_for_bit_on_every_backend(self):
        import xgboost as xgb

        for classes in (3, 30):
            rng = np.random.default_rng(classes)
            x = rng.normal(size=(1500, 4)).astype(np.float32)
            score = x[:, 0] + 0.3 * x[:, 1]
            y = np.digitize(score, np.quantile(score, np.linspace(0, 1, classes + 1)[1:-1]))
            booster = xgb.train(
                {"objective": "multi:softprob", "num_class": classes, "max_depth": 3, "eta": 0.5, "nthread": 1}, xgb.DMatrix(x, label=y), 5
            )
            xs = x[:300]
            want = booster.predict(xgb.DMatrix(xs)).reshape(len(xs), -1)
            with tempfile.TemporaryDirectory() as d:
                model = Path(d) / "m.json"
                booster.save_model(str(model))
                forest = qf.load_forest([model])
                for name in ("flat", "wave", "wave-avx2", "codegen-block"):
                    if name.startswith("codegen") and not have_rustc():
                        continue
                    be = backends.get(name)
                    art = Path(d) / f"m-{name}{be.suffix}"
                    be.compile(forest, art, {})
                    pred = be.load(art)
                    got = pred.predict(xs)
                    self.assertTrue(np.array_equal(got, want), f"{name}, {classes} classes: max {np.abs(got - want).max():.2e}")
                    if hasattr(pred, "close"):
                        pred.close()


def library_answer(model, xs, bm):
    """The model library's own answer for rows `xs` and a base margin `bm` added to its margin (the same meaning on both sides)."""
    return qf_formats.open_source(model).predict(xs, base_margin=bm)


class TransformsFollowTheLibraryAtTheEdgesTest(unittest.TestCase):
    """The objective's transform is the library's own code at the ends of its range, not the textbook formula: margins from -1e4 to 1e4,
    and the ones that overflow a float32 sum (xgboost: `std::min(-x, 88.7f)` in the sigmoid, so a margin below -88.7 answers 3.0e-39 and
    not 0; the softmax subtracts the largest margin first). LightGBM's and YDF's transforms have no clamp and double arithmetic."""

    MARGINS = np.array(
        [-1e4, -1000, -200, -100, -89, -88.7, -88.6, -50, -1, 0, 0.3, 50, 88.6, 88.7, 89, 100, 1000, 1e4, 3e38, -3e38, 1e39, -1e39]
    )

    @staticmethod
    def xgb_data(seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(300, 4)).astype(np.float32)
        return x, (x[:, 0] > 0).astype(int), np.digitize(x[:, 0], [-0.5, 0.5])

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_xgboost_binary_sigmoid_equals_the_library_bit_for_bit_on_every_backend(self):
        import xgboost as xgb

        x, y, _ = self.xgb_data()
        booster = xgb.train({"objective": "binary:logistic", "max_depth": 2}, xgb.DMatrix(x, y), 3)
        xs, bm = x[: len(self.MARGINS)], self.MARGINS
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "m.json"
            booster.save_model(str(model))
            want = library_answer(model, xs, bm).reshape(len(xs))
            self.assertTrue((want[bm < -100] > 0).all() and (want[bm < -100] < 1e-38).all(), "xgboost answers a tiny number, not 0")
            forest = qf.load_forest([model])
            for name in ("flat", "wave", "wave-avx2", "codegen-block"):
                if name.startswith("codegen") and not have_rustc():
                    continue
                be = backends.get(name)
                art = Path(d) / f"m-{name}{be.suffix}"
                be.compile(forest, art, {})
                pred = be.load(art)
                got = pred.predict(xs, base_margin=bm)
                self.assertTrue(np.array_equal(got, want), f"{name}: {np.c_[bm, got, want][got != want][:4]}")
                if hasattr(pred, "close"):
                    pred.close()

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_xgboost_softmax_equals_the_library_bit_for_bit_on_every_backend(self):
        import xgboost as xgb

        x, _, y = self.xgb_data()
        booster = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 2}, xgb.DMatrix(x, y), 3)
        bm = np.array(
            [
                *([1e4, 0, -1e4], [-1e4, -1e4, -1e4], [88, 89, -1000], [100, 100, 100], [0, 0, 0], [-100, -100, 0], [-1e4, 0, 0]),
                *([0, 1e4, 1e4], [3e38, 3e38, 0], [-3e38, -3e38, -3e38], [1e39, 0, 0], [0, 0, -1e39], [1e4, 1e4, 1e4]),
            ]
        )
        xs = x[: len(bm)]
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "m.json"
            booster.save_model(str(model))
            want = library_answer(model, xs, bm)
            forest = qf.load_forest([model])
            for name in ("flat", "wave", "wave-avx2"):
                be = backends.get(name)
                art = Path(d) / f"m-{name}{be.suffix}"
                be.compile(forest, art, {})
                pred = be.load(art)
                got = pred.predict(xs, base_margin=bm)
                self.assertTrue(np.array_equal(got, want, equal_nan=True), f"{name}: {np.c_[got, want][:6]}")
                pred.close()

    @unittest.skipUnless(HAVE_XGB, "no xgboost")
    def test_a_margin_below_minus_88_7_is_the_libraries_answer_also_in_the_references_and_the_checks(self):
        """A valid binary model whose margins reach -120 (an offset tree, as a stacked or calibrated model may
        carry). xgboost answers 3.0e-39 on those rows; the compiled forest, `quickforest check`, `optimize --verify` and the differential's
        float32 reference used to answer 0 and refuse it."""
        import json

        import xgboost as xgb

        rng = np.random.default_rng(0)
        x = rng.normal(size=(600, 4)).astype(np.float32)
        y = (x[:, 0] > 0).astype(int)
        b = xgb.train(
            {"objective": "binary:logistic", "max_depth": 2, "eta": 1.0, "lambda": 0, "min_child_weight": 0}, xgb.DMatrix(x, y), 120
        )
        j = json.loads(b.save_raw("json"))
        t0 = j["learner"]["gradient_booster"]["model"]["trees"][0]
        t0["split_conditions"] = [
            v - 90.0 if left == -1 else v for v, left in zip(t0["split_conditions"], t0["left_children"], strict=True)
        ]
        b = xgb.Booster()
        b.load_model(bytearray(json.dumps(j).encode()))
        margin = b.predict(xgb.DMatrix(x), output_margin=True)
        self.assertGreater(int((margin < -88.7).sum()), 100)
        want = b.predict(xgb.DMatrix(x))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            b.save_model(str(d / "m.ubj"))
            np.savez(d / "sel.npz", X=x)
            backends.get("flat").compile(qf.load_forest([d / "m.ubj"]), d / "m.flat", {})
            pred = backends.get("flat").load(d / "m.flat")
            self.assertTrue(np.array_equal(pred.predict(x), want), "the compiled forest")
            pred.close()
            # numpy's float32 transform and the differential's reference (libm expf) are xgboost's too, to an ulp of expf
            tiny = margin < -88.7
            self.assertTrue(np.allclose(qf.transform(margin, "sigmoid")[tiny], want[tiny], rtol=1e-5, atol=0))
            self.assertTrue(np.array_equal(rd.xgb_margin_average_reference([d / "m.ubj"], x, "sigmoid").ravel(), want))
            for command in (["check", str(d / "m.ubj"), str(d / "m.flat"), "--data", str(d / "sel.npz")],):
                run = subprocess.run(
                    [sys.executable, "-s", str(HERE.parent / "scripts" / "quickforest"), *command], capture_output=True, text=True
                )
                self.assertEqual(run.returncode, 0, run.stdout[-400:] + run.stderr[-400:])

    def test_the_numpy_transform_clamps_like_xgboost_and_not_like_the_double_one(self):
        m = self.MARGINS[np.abs(self.MARGINS) < 1e30].astype(np.float32)
        got = qf.transform(m[:, None], "sigmoid")[:, 0]
        x = np.minimum(-m, np.float32(88.7))
        want = np.float32(1) / (rd.libm_expf(x) + np.float32(1) + np.float32(1e-16))
        self.assertTrue(np.allclose(got, want, rtol=1e-5, atol=0))
        self.assertTrue((got[m < -88.7] > 0).all(), "float32 sigmoid below the clamp is not 0")
        self.assertTrue(np.isnan(qf.transform(np.array([[np.nan]], np.float32), "sigmoid")).all(), "a NaN margin stays NaN")
        # the double transform (LightGBM, YDF) has no clamp
        with np.errstate(over="ignore"):
            d = qf.transform_f64(np.array([-1e4, -745.0, -100.0]), "sigmoid")
        self.assertEqual(d[0], 0.0)
        self.assertEqual(d[2], 1.0 / (1.0 + np.exp(100.0)))

    @unittest.skipUnless(lgb is not None, "no lightgbm")
    def test_lightgbm_transforms_equal_the_library_at_the_edges(self):
        """Binary (also with sigmoid:1.7, which is folded into the leaves), multiclass, poisson and cross-entropy forests whose first tree is
        shifted by -1e4 ... 1e4: the compiled double answer is LightGBM's (no clamp; overflow is inf, underflow 0 or a subnormal)."""
        import re

        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 4))
        yb, y3 = (x[:, 0] > 0).astype(int), np.digitize(x[:, 0], [-0.5, 0.5])
        models = {
            "binary": ({"objective": "binary"}, yb),
            "binary-sigmoid-1.7": ({"objective": "binary", "sigmoid": 1.7}, yb),
            "multiclass": ({"objective": "multiclass", "num_class": 3}, y3),
            "poisson": ({"objective": "poisson"}, np.abs(x[:, 0]) + 1),
            "cross-entropy": ({"objective": "cross_entropy"}, yb.astype(float)),
        }

        def shifted(text, delta):
            parts = re.split(r"(?m)^(?=Tree=\d+$)", text)
            parts[0] = re.sub(r"(?m)^tree_sizes=.*\n", "", parts[0])
            parts[1] = re.sub(
                r"(?m)^leaf_value=(.*)$", lambda m: "leaf_value=" + " ".join(repr(float(v) + delta) for v in m.group(1).split()), parts[1]
            )
            return "".join(parts)

        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, (params, y) in models.items():
                booster = lgb.train({**params, "num_leaves": 4, "verbose": -1}, lgb.Dataset(x, y), 3)
                text = booster.model_to_string()
                for delta in (-1e4, -800, -745, -710, -100, -30, 30, 100, 710, 800, 1e4):
                    path = d / f"{name}-{delta}.txt"
                    path.write_text(shifted(text, delta))
                    want = lgb.Booster(model_file=str(path)).predict(x[:40])
                    backends.get("flat").compile(qf.load_forest([path]), d / "m.flat", {"f64_leaves": True, "no_lib": True})
                    pred = backends.get("flat").load(d / "m.flat")
                    got = pred.predict(x[:40].astype(np.float32), dtype=np.float64).reshape(want.shape)
                    pred.close()
                    # exact where the libraries agree to the bit, else the folded sigmoid scale's rounding (1e-14 of a tiny number)
                    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0, err_msg=f"{name}, first tree shifted by {delta}")

    def test_a_ydf_classifier_with_a_logit_above_88_7_answers_the_exact_softmax_where_ydf_itself_answers_nan(self):
        """YDF's multi-class softmax exponentiates in float32 without subtracting the largest logit, so a logit above 88.7 overflows it to NaN
        (inf / inf). The compiled forest answers the exact softmax; and a binary logit down to -1e4 or up to 1e4 gives 0 or 1."""
        try:
            import ydf
        except ImportError:
            self.skipTest("no ydf")
        import logging

        logging.disable(logging.CRITICAL)
        rng = np.random.default_rng(0)
        x = rng.normal(size=(60, 4)).astype(np.float32)
        cols = {f"f{i}": x[:, i] for i in range(4)}
        for classes, labels in ((2, (x[:, 0] > 0).astype(int)), (3, np.digitize(x[:, 0], [-0.5, 0.5]))):
            model = ydf.GradientBoostedTreesLearner(label="y", task=ydf.Task.CLASSIFICATION, num_trees=5, max_depth=3, num_threads=1).train(
                {**cols, "y": labels.astype(str)}, verbose=0
            )
            start = np.asarray(model.initial_predictions(), np.float64)
            with tempfile.TemporaryDirectory() as d:
                for k, delta in enumerate((-1e4, -200, 100, 1e4)):
                    moved = start.copy()
                    moved[0] += delta
                    model.set_initial_predictions(moved)
                    model.save(str(Path(d) / f"m{k}"))
                    backends.get("flat").compile(qf.load_forest([Path(d) / f"m{k}"], "output", False), Path(d) / "m.flat", {"no_lib": True})
                    pred = backends.get("flat").load(Path(d) / "m.flat")
                    got = pred.predict(x).reshape(len(x), -1)
                    pred.close()
                    self.assertTrue(np.isfinite(got).all(), (classes, delta))
                    if classes == 3:
                        self.assertTrue(np.allclose(got.sum(1), 1.0, atol=1e-6), (classes, delta))
                    if abs(delta) >= 100:  # the moved logit saturates: probability 1 above, 0 below
                        self.assertTrue((got[:, 0] == (1.0 if delta > 0 else 0.0)).all(), (classes, delta, got[:3, 0]))


class XgboostPrunedNodesTest(unittest.TestCase):
    """With `gamma` > 0 xgboost's exact tree method drops splits after growing them but keeps the dropped nodes in the model's arrays
    (`num_deleted`), where no path reaches them. Only the nodes a row can reach belong in the compiled forest and in the node counts."""

    @staticmethod
    def reachable(tree):
        left, right = tree["left_children"], tree["right_children"]
        seen, stack = 0, [0]
        while stack:
            i = stack.pop()
            seen += 1
            if left[i] != -1:
                stack += [left[i], right[i]]
        return seen

    def trained(self, method):
        import xgboost as xgb

        rng = np.random.default_rng(1)
        x = rng.normal(size=(400, 6)).astype(np.float32)
        y = (x[:, 0] + 0.5 * x[:, 1] * x[:, 2] + 0.3 * rng.normal(size=400) > 0).astype(int)
        params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.3, "gamma": 1.0, "tree_method": method, "nthread": 1}
        return x, xgb.train(params, xgb.DMatrix(x, label=y), 12)

    def test_a_model_with_dropped_nodes_compiles_bit_exact_on_every_backend_and_counts_only_the_reachable_nodes(self):
        import json

        import xgboost as xgb

        for method in ("exact", "approx"):
            x, booster = self.trained(method)
            trees = json.loads(bytes(booster.save_raw(raw_format="json")))["learner"]["gradient_booster"]["model"]["trees"]
            dead = sum(int(t["tree_param"]["num_deleted"]) for t in trees)
            reach = sum(self.reachable(t) for t in trees)
            if method == "exact":  # the one method that keeps them in the xgboost 3.x tested here; approx must still pass unchanged
                self.assertGreater(dead, 0, "the test model must keep dropped nodes in its arrays")
                self.assertLess(reach, sum(len(t["left_children"]) for t in trees))
            want = booster.predict(xgb.DMatrix(x)).reshape(len(x), -1)
            with tempfile.TemporaryDirectory() as d:
                model = Path(d) / "m.json"
                booster.save_model(str(model))
                forest = qf.load_forest([model])
                self.assertEqual(forest.n_nodes, reach, method)
                self.assertEqual(int(qf_formats.open_source(model).spec.nodes.sum()), reach, method)
                for name in ("flat", "wave", "codegen-block"):
                    if name.startswith("codegen") and not have_rustc():
                        continue
                    be = backends.get(name)
                    art = Path(d) / f"m-{name}{be.suffix}"
                    info = be.compile(forest, art, {})
                    self.assertEqual(info["nodes"], reach, f"{method} {name}")
                    pred = be.load(art)
                    got = pred.predict(x).reshape(len(x), -1)
                    self.assertTrue(np.array_equal(got, want), f"{method} {name}: max {np.abs(got - want).max():.2e}")
                    if hasattr(pred, "close"):
                        pred.close()


def stump_text(index, feature, threshold, left, right):
    """One LightGBM tree with a single split `x[feature] <= threshold` (missing type NaN, NaN goes right) and two leaves."""
    return (
        f"Tree={index}\nnum_leaves=2\nnum_cat=0\nsplit_feature={feature}\nsplit_gain=1\nthreshold={threshold!r}\ndecision_type=8\n"
        f"left_child=-1\nright_child=-2\nleaf_value={left!r} {right!r}\nleaf_weight=1 1\nleaf_count=1 1\ninternal_value=0\n"
        "internal_weight=2\ninternal_count=2\nis_linear=0\nshrinkage=1\n\n\n"
    )


@unittest.skipIf(lgb is None, "no lightgbm")
class Float64NearThresholdPinTest(unittest.TestCase):
    """LIMITS.md, 'Inputs are float32': LightGBM compares a float64 input with its float64 threshold, QuickForest rounds the input to
    float32 (to nearest) first. This pins WHICH inputs fall on which side today, so a silent change shows up; it is a documented limit,
    not a bug.

    Two stumps, answer = A + B: A on feature 0 with threshold 0.1 (left 1, right 2), B on feature 1 with threshold 1 + 1e-8 (left 10,
    right 20). Around 0.1 the float32 neighbours are 0.0999999940 (below) and 0.1000000015 (above, the value QuickForest compares with);
    0.1 rounds UP to the float32 above, so inputs in (0.09999999776, 0.1] go left in LightGBM and right in QuickForest. Around 1 + 1e-8
    the float32 neighbours are 1.0 and 1.00000012; the threshold rounds DOWN to 1.0, so inputs in (1 + 1e-8, 1 + 5.96e-8) go right in
    LightGBM and left in QuickForest. Everything else, and every input that was rounded to float32 first, agrees."""

    ROWS = (  # (x0, x1); the other feature is far from its threshold (-5: left)
        (0.09999999403953552, -5.0),  # the float32 below 0.1
        (0.0999999970, -5.0),  # nearer to it than to the float32 above: left on both sides
        (0.0999999990, -5.0),  # DIFFERENT: nearer to the float32 above, so QuickForest goes right
        (0.1, -5.0),  # DIFFERENT: the threshold itself, <= goes left in LightGBM
        (0.1 + 1e-12, -5.0),  # right on both
        (0.10000000149011612, -5.0),  # the float32 above 0.1: right on both
        (-5.0, 1.0),
        (-5.0, 1.0 + 1e-8),  # the threshold itself: left on both (it rounds to 1.0)
        (-5.0, 1.0 + 3e-8),  # DIFFERENT: LightGBM right, QuickForest rounds to 1.0 and goes left
        (-5.0, 1.0 + 5e-8),  # DIFFERENT, the same
        (-5.0, 1.0 + 7e-8),  # nearer to the float32 above 1: right on both
        (-5.0, 1.0000001192092896),  # the float32 above 1
    )
    LIGHTGBM = (11.0, 11.0, 11.0, 11.0, 12.0, 12.0, 11.0, 11.0, 21.0, 21.0, 21.0, 21.0)
    QUICKFOREST = (11.0, 11.0, 12.0, 12.0, 12.0, 12.0, 11.0, 11.0, 11.0, 11.0, 21.0, 21.0)

    def test_which_float64_inputs_go_the_other_way(self):
        text = (
            "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\nmax_feature_idx=1\nobjective=regression\n"
            "feature_names=a b\nfeature_infos=none none\n\n"
            + stump_text(0, 0, 0.1, 1.0, 2.0)
            + stump_text(1, 1, 1.0 + 1e-8, 10.0, 20.0)
            + "end of trees\n\nend of parameters\n\npandas_categorical:null\n"
        )
        x = np.array(self.ROWS, np.float64)
        with tempfile.TemporaryDirectory() as d:
            model = Path(d) / "m.txt"
            model.write_text(text)
            booster = lgb.Booster(model_file=str(model))
            backends.get("flat").compile(qf.load_forest([model]), Path(d) / "m.flat", {})
            pred = backends.get("flat").load(Path(d) / "m.flat")
            try:
                lib64, qf64 = booster.predict(x), pred.predict(x)
                lib32 = booster.predict(x.astype(np.float32).astype(np.float64))
            finally:
                pred.close()
        self.assertEqual(tuple(lib64), self.LIGHTGBM)
        self.assertEqual(tuple(qf64), self.QUICKFOREST)
        self.assertEqual(
            [i for i in range(len(x)) if lib64[i] != qf64[i]], [2, 3, 8, 9], "the rows where the two sides differ on float64 inputs"
        )
        self.assertTrue(np.array_equal(lib32, qf64), "rounded to float32 first, LightGBM and QuickForest agree on every row")


if __name__ == "__main__":
    unittest.main()


class CodegenPlanCoversEveryFormatTest(unittest.TestCase):
    """The generated-code builds of the differential run (`--codegen-every N`, `--pgo-every N`) were picked by `case % N == N - 1`, and the
    format is `case % 3`: with N a multiple of 3 (the defaults 6, 12, 18 and 60) every pick was a YDF case, so generated code was never built
    from an xgboost or a LightGBM model (multiclass, f32 sums, categorical, linear leaves, exp, sigmoid scale) outside the unit tests."""

    FORMATS = ("xgboost", "lightgbm", "ydf")

    def picks(self, kind, every, pgo_every, cases):
        return [c for c in range(cases) if rd.plan_codegen(c, "full", every, pgo_every) == kind]

    def test_every_format_gets_generated_code_in_the_quick_and_the_full_plan(self):
        for mode, block, pgo, cases, kinds in (("quick", 6, 18, 36, ("block",)), ("full", 12, 60, 3000, ("block", "pgo"))):
            for kind in kinds:
                picked = self.picks(kind, block, pgo, cases)
                self.assertEqual({self.FORMATS[c % 3] for c in picked}, set(self.FORMATS), (mode, kind, picked))
            # the share of the run is the one the flags say: one case in N (pgo takes precedence over block on a clash)
            n_pgo, n_block = len(self.picks("pgo", block, pgo, cases)), len(self.picks("block", block, pgo, cases))
            self.assertEqual(n_pgo, cases // pgo, mode)
            self.assertLessEqual(abs(n_block + n_pgo - cases // block), n_pgo, mode)

    def test_a_period_that_is_not_a_multiple_of_three_and_the_degenerate_periods_still_pick(self):
        self.assertEqual(self.picks("block", 7, 0, 70).__len__(), 10)
        self.assertEqual(self.picks("block", 1, 0, 5), [0, 1, 2, 3, 4])
        self.assertEqual(len(self.picks("block", 2, 0, 20)), 10)
        self.assertEqual(self.picks("block", 0, 0, 20), [])
