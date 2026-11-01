"""Regression tests of review pass 2: each failed before its fix."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


def run_cli(*args):
    """quickforest CLI in a child process: (returncode, combined output)."""
    r = subprocess.run([sys.executable, "-I", str(SCRIPTS / "quickforest"), *map(str, args)], capture_output=True, text=True, timeout=300)
    return r.returncode, (r.stdout + r.stderr).strip()


@unittest.skipUnless(xgb is not None, "xgboost not installed")
class PruneFolderTest(unittest.TestCase):
    def test_a_folder_whose_models_have_different_output_counts_is_refused_in_one_line(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(600, 5)).astype(np.float32)
        y = np.stack([x[:, 0], x[:, 1] * 2, x[:, 2] - x[:, 3]], 1).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "folds").mkdir()
            for name, k in (("a", 3), ("b", 3), ("c", 2)):
                m = xgb.XGBRegressor(tree_method="hist", multi_strategy="multi_output_tree", n_estimators=6, max_depth=3, n_jobs=1)
                m.fit(x, y[:, :k])
                m.save_model(str(d / "folds" / f"{name}.ubj"))
            np.savez(d / "data.npz", X=x)
            rc, out = run_cli("prune", d / "folds", d / "data.npz", d / "out", "--keep", "50%", "--regression", "log")
        self.assertNotEqual(rc, 0)
        self.assertIn("disagree", out)
        self.assertNotIn("Traceback", out)
        self.assertNotIn("inhomogeneous", out)
        self.assertEqual(len(out.splitlines()), 1)


class QfskDamageTest(unittest.TestCase):
    """A hostile or damaged .qfsk ends in a one-line ValueError from the Python reader and the CLI alike."""

    @classmethod
    def setUpClass(cls):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            raise unittest.SkipTest("scikit-learn is not installed") from None
        import qf_sklearn

        cls.qf_sklearn = qf_sklearn
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 4)).astype(np.float32)
        est = RandomForestRegressor(n_estimators=3, max_depth=3, random_state=0).fit(x, np.stack([x[:, 0], x[:, 1]], 1))
        cls.dir = tempfile.TemporaryDirectory()
        cls.path = Path(cls.dir.name) / "m.qfsk"
        qf_sklearn.export_sklearn(est, cls.path, X=x)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def write(self, name, **changes):
        with np.load(self.path, allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        a.update(changes)
        out = self.path.with_name(name)
        with open(out, "wb") as f:
            np.savez(f, **a)
        return out, a

    def test_a_truncated_file_is_a_value_error_from_the_python_reader(self):
        raw = self.path.read_bytes()
        bad = self.path.with_name("trunc.qfsk")
        bad.write_bytes(raw[:2000])
        with self.assertRaisesRegex(ValueError, "not a valid qfsk1 file"):
            self.qf_sklearn.load_arrays(bad)

    def test_a_flipped_compression_method_is_a_value_error(self):
        raw = bytearray(self.path.read_bytes())
        at = raw.find(b"PK\x01\x02")  # the first central-directory entry: its compression method is 10 bytes in
        self.assertGreater(at, 0)
        raw[at + 10] = 99
        bad = self.path.with_name("method.qfsk")
        bad.write_bytes(bytes(raw))
        with self.assertRaisesRegex(ValueError, "not a valid qfsk1 file"):
            self.qf_sklearn.load_arrays(bad)

    def test_unsigned_node_arrays_are_refused_not_walked(self):
        for key in ("left", "right", "feature", "tree_off"):
            with np.load(self.path, allow_pickle=False) as z:
                v = z[key]
            out, _ = self.write(f"u_{key}.qfsk", **{key: v.astype(np.uint64)})
            with self.subTest(key=key):
                rc, text = run_cli("compile", out, out.with_suffix(".flat"), "--backend", "flat")
                self.assertNotEqual(rc, 0)
                self.assertNotIn("Traceback", text)
                self.assertIn("wrong type", text)

    def test_a_float_feature_count_is_refused(self):
        out, _a = self.write("f_nf.qfsk", n_features=a_float(self.path, "n_features"))
        with self.assertRaisesRegex(ValueError, "n_features has the wrong type"):
            self.qf_sklearn.load_arrays(out)


class FuzzHarnessTest(unittest.TestCase):
    def test_the_segment_mutation_survives_a_stacked_huge_segment_count(self):
        """`fuzz_loader.py --mode full` reported HANG (vec_cols / vec_ext64, multi:258 and 259): the harness, not the loader, looped over 2^32 offsets."""
        import time

        sys.path.insert(0, str(HERE / "fuzz"))
        import fuzz_loader

        b = bytearray(4000)
        L = {"format": 7, "n_segments": 0xFFFFFFFF, "seg_at": 100, "n_groups": 5}
        t0 = time.perf_counter()
        for seed in range(20):
            out = fuzz_loader.mutate_vector(bytearray(b), "seg", np.random.default_rng(seed), L)
            self.assertEqual(len(out), len(b))
        self.assertLess(time.perf_counter() - t0, 5)


def a_float(path, key):
    with np.load(path, allow_pickle=False) as z:
        return z[key].astype(np.float64)


if __name__ == "__main__":
    unittest.main()
