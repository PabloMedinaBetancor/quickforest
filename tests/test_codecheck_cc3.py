"""Regression tests of code check 3 (sklearn `.qfsk` export, verification and reader hardening): each failed before its fix.

python -I -m pytest -q -p no:cacheprovider tests/test_codecheck_cc3.py     (a Python with numpy and scikit-learn >= 1.3)
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

try:
    from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("scikit-learn is not installed") from None

import qf_formats
import qf_sklearn


def small_forest(**kw):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    return RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0, **kw).fit(x, x[:, 0] + rng.normal(size=200)), x


class OffsetWrapTest(unittest.TestCase):
    """code check 3, finding 1 (critical): `tree_off` values near the int64 limits wrapped `np.diff` to positive numbers, the "increasing" test passed, and
    `np.repeat` in `_validate` died with a segmentation fault (a damaged or hostile file killed the process)."""

    def test_wrapped_tree_offsets_are_refused_in_one_line_not_a_crash(self):
        est, _ = small_forest()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.qfsk"
            qf_sklearn.export_sklearn(est, path)
            code = (
                "import sys, numpy as np\n"
                f"sys.path.insert(0, {str(SCRIPTS)!r})\n"
                "import qf_sklearn as q\n"
                f"a = q.load_arrays({str(path)!r})\n"
                "big = 2**63 - 1\n"
                "a['tree_off'] = np.array([0, big, -2, len(a['left'])], np.int64)\n"  # diffs: big, big (wrapped), small: all 'positive'
                "try:\n"
                "    q._validate(a, 'forged.qfsk')\n"
                "    print('ACCEPTED')\n"
                "except ValueError as e:\n"
                "    print('REFUSED', e)\n"
            )
            r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, f"the child died (rc {r.returncode}): {r.stderr[-200:]}")
        self.assertTrue(r.stdout.startswith("REFUSED forged.qfsk"), r.stdout)
        self.assertEqual(len(r.stdout.strip().splitlines()), 1)


class _NanRefusingForest(RandomForestRegressor):
    """A forest that refuses NaN by construction (newer scikit-learn releases accept NaN in forests, so the library cannot be relied on to refuse)."""

    def predict(self, X):
        if np.isnan(np.asarray(X, dtype=np.float64)).any():
            raise ValueError("Input X contains NaN. This estimator does not accept missing values")
        return super().predict(X)


_NanRefusingForest.__name__ = "RandomForestRegressor"  # the exporter reads the class name


class NanRefusingEstimatorTest(unittest.TestCase):
    """code check 3, finding 2 (major): an estimator that refuses NaN (RandomForest with criterion="absolute_error") made the export drop every probe row
    that held a NaN. With 6 % of the cells NaN that is every row from about 100 features: the export crashed on an empty array (sklearn's
    own multi-line message), and with fewer features it verified a fraction of the rows. The probes are now drawn again without NaN, and the
    result says the NaN routing was not checked against the library."""

    def fit(self, n_features):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, n_features)).astype(np.float32)
        return _NanRefusingForest(4, max_depth=4, random_state=0).fit(x, x[:, 0]), x

    def test_many_features_still_export_and_verify_on_full_probe_set(self):
        est, _x = self.fit(200)
        with tempfile.TemporaryDirectory() as d:
            r = qf_sklearn.export_sklearn(est, Path(d) / "m.qfsk")
            self.assertTrue((Path(d) / "m.qfsk").exists())
        self.assertEqual(r["verified_rows"], qf_sklearn.PROBE_ROWS)
        self.assertEqual(r["max_diff"], 0.0)
        self.assertFalse(r["nan_verified"])

    def test_rows_of_the_caller_that_hold_nan_are_the_only_ones_dropped(self):
        est, x = self.fit(6)
        given = x[:50].copy()
        given[:10, 0] = np.nan
        with tempfile.TemporaryDirectory() as d:
            r = qf_sklearn.export_sklearn(est, Path(d) / "m.qfsk", given)
        self.assertEqual(r["verified_rows"], 40 + qf_sklearn.PROBE_ROWS)
        self.assertFalse(r["nan_verified"])

    def test_an_estimator_that_takes_nan_is_verified_with_it(self):
        est, _ = small_forest()
        with tempfile.TemporaryDirectory() as d:
            r = qf_sklearn.export_sklearn(est, Path(d) / "m.qfsk")
        self.assertTrue(r["nan_verified"])
        self.assertEqual(r["verified_rows"], qf_sklearn.PROBE_ROWS)


class WideModelProbeTest(unittest.TestCase):
    """code check 3, finding 5 (minor): the probe set was always 2,048 rows, so a model of 30,000 features drew 61 M cells and peaked at 2.3 GB (the
    test machine has 7.6 GB and a memory watchdog). Wide models now get fewer probe rows, never fewer than MIN_PROBE_ROWS."""

    def test_the_probe_set_is_bounded_in_cells(self):
        est, _ = small_forest()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.qfsk"
            qf_sklearn.export_sklearn(est, path)
            arrays = qf_sklearn.load_arrays(path)
        wide = qf_sklearn.probe_rows(arrays, 30000)
        self.assertLessEqual(wide.size, qf_sklearn.MAX_PROBE_CELLS)
        self.assertGreaterEqual(len(wide), qf_sklearn.MIN_PROBE_ROWS)
        self.assertEqual(len(qf_sklearn.probe_rows(arrays, 4)), qf_sklearn.PROBE_ROWS)  # a normal model keeps all of them


class SubforestTypeTest(unittest.TestCase):
    """code check 3, finding 3 (minor): `subforest` took anything that carries `estimators_` (Bagging, boosting) and returned a copy whose other bookkeeping
    (`estimators_features_`, the stage arrays) was left stale: a model that answers wrongly or fails later."""

    def test_only_the_estimators_the_reader_holds(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 3))
        y = x[:, 0]
        for est in (BaggingRegressor(n_estimators=4, random_state=0).fit(x, y), GradientBoostingRegressor(n_estimators=4).fit(x, y)):
            with self.assertRaisesRegex(ValueError, "not supported"):
                qf_sklearn.subforest(est, [0, 1])
        est, _ = small_forest()
        self.assertEqual(len(qf_sklearn.subforest(est, [2, 0]).estimators_), 2)


class PickleInFolderTest(unittest.TestCase):
    """code check 3, finding 4 (minor): a pickle fold in a folder of exports was skipped without a word (the ensemble averaged one fold fewer than the user
    thinks), and a folder of pickles only said "no model files". DC 3.1: `.pkl`/`.joblib` are refused anywhere."""

    def test_a_pickle_fold_without_an_export_is_refused_with_the_way_in(self):
        est, _ = small_forest()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            qf_sklearn.export_sklearn(est, d / "fold0.qfsk")
            (d / "fold1.joblib").write_bytes(b"not read")
            with self.assertRaisesRegex(ValueError, "export_sklearn"):
                qf_formats.model_files(d)
            only = d / "only"
            only.mkdir()
            (only / "fold0.pkl").write_bytes(b"not read")
            with self.assertRaisesRegex(ValueError, "export_sklearn"):
                qf_formats.model_files(only)

    def test_the_pickle_an_export_came_from_may_stay_beside_it(self):
        est, _ = small_forest()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            qf_sklearn.export_sklearn(est, d / "fold0.qfsk")
            (d / "fold0.joblib").write_bytes(b"not read")
            self.assertEqual([f.name for f in qf_formats.model_files(d)], ["fold0.qfsk"])


class ThresholdSweepTest(unittest.TestCase):
    """Checked, no bug (kept as a pin): `x < up_threshold(t)` equals sklearn's `float64(x) <= t` for every float32 x around any threshold.
    The one exception is t = +inf with x = +inf (sklearn is left, QF right): sklearn's own predict refuses an infinite input, so no
    library answer exists, and LIMITS.md already says infinite inputs are not covered."""

    def test_float32_neighbourhood_of_thresholds_of_every_scale(self):
        rng = np.random.default_rng(5)
        f32max = float(np.finfo(np.float32).max)
        e = rng.integers(-160, 130, size=20000)
        f = rng.normal(size=5000).astype(np.float32)
        mid = (f.astype(np.float64) + np.nextafter(f, np.float32(np.inf)).astype(np.float64)) / 2  # exactly between two float32 values
        edge = np.array(
            [0.0, -0.0, 1e-46, -1e-46, 7e-46, -7e-46, f32max, -f32max, f32max * (1 + 2**-25), -f32max * (1 + 2**-25), 3.5e38, -3.5e38]
        )
        t = np.concatenate([rng.choice([-1, 1], 20000) * rng.random(20000) * 2.0**e, mid, edge])
        u = qf_sklearn.up_threshold(t)
        with np.errstate(all="ignore"):
            base = t.astype(np.float32)
            for step in (0, 1, -1, 2, -2):
                x = base.copy()
                for _ in range(abs(step)):
                    x = np.nextafter(x, np.float32(np.inf if step > 0 else -np.inf))
                self.assertTrue(((x.astype(np.float64) <= t) == (x < u)).all(), step)
        for x in np.array([0, -0.0, -np.inf, f32max, -f32max, 1.4e-45, -1.4e-45], np.float32):
            self.assertTrue(((np.float64(x) <= t) == (x < u)).all(), x)


if __name__ == "__main__":
    unittest.main()
