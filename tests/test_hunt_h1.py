"""Regression tests of review pass 1: each failed before its fix."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import qf_sklearn
import structure


def walk(a, x):
    """Plain-numpy walk of a structure export: (rows, groups) margin of a forest of vector trees (x < threshold goes left)."""
    off, out = a["node_offset"], np.tile(a["base_margin"].astype(np.float64), (len(x), 1))
    for t in range(len(off) - 1):
        s = slice(off[t], off[t + 1])
        left, right, feat, thr, dl = a["left"][s], a["right"][s], a["feature"][s], a["threshold"][s], a["default_left"][s]
        for r in range(len(x)):
            n = 0
            while left[n] != -1:
                v = x[r, feat[n]]
                go_left = dl[n] if np.isnan(v) else v < thr[n]
                n = left[n] if go_left else right[n]
            out[r] += a["leaf_vector"][s][n]
    return out


class StructureVectorTest(unittest.TestCase):
    def test_structure_of_vector_trees_carries_the_leaf_vectors(self):
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 5)).astype(np.float32)
        y = np.stack([x[:, 0] + x[:, 1], x[:, 2] * x[:, 3], np.sin(x[:, 4])], 1)
        est = RandomForestRegressor(n_estimators=6, max_depth=4, random_state=0).fit(x, y)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.qfsk"
            qf_sklearn.export_sklearn(est, str(path), X=x)
            a = structure.export(path)
        self.assertEqual(a["n_groups"], 3)
        self.assertTrue((a["tree_group"] == -1).all())
        self.assertIn("leaf_vector", a)
        self.assertEqual(a["leaf_vector"].shape, (len(a["left"]), 3))
        self.assertEqual(a["leaf_vector"].dtype, np.float32)
        self.assertFalse(a["leaf_vector"][a["left"] != -1].any())  # zero at the inner nodes
        np.testing.assert_allclose(walk(a, x[:40]), est.predict(x[:40]), rtol=1e-5, atol=1e-6)

    def test_structure_of_scalar_trees_has_no_leaf_vector(self):
        import xgboost as xgb

        rng = np.random.default_rng(1)
        x = rng.normal(size=(200, 4)).astype(np.float32)
        m = xgb.XGBRegressor(n_estimators=3, max_depth=3).fit(x, x[:, 0])
        with tempfile.TemporaryDirectory() as d:
            m.save_model(str(Path(d) / "m.ubj"))
            a = structure.export(Path(d) / "m.ubj")
        self.assertNotIn("leaf_vector", a)


if __name__ == "__main__":
    unittest.main()
