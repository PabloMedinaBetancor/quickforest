"""code check 4 (QuickForest 1.2): the xgboost and YDF vector readers, the stream path and the structure export.

python -I -m pytest -q -p no:cacheprovider tests/test_codecheck_cc4.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import qf_formats
import select_stream

try:
    import xgboost as xgb
except ImportError:  # the readers' tests need the library they read
    xgb = None


def train_vector(path: Path, rounds: int, parallel: int = 1, k: int = 3, seed: int = 0, **extra):
    """A small `multi_output_tree` regression model of K targets saved to `path`; returns the rows it was trained on."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(600, 6)).astype(np.float32)
    x[rng.random(x.shape) < 0.05] = np.nan
    z = np.nan_to_num(x)
    y = np.stack([z[:, 0] + z[:, 1] ** 2, np.sin(z[:, 2]), z[:, 3] * z[:, 4]][:k], 1)
    params = dict(tree_method="hist", nthread=1, verbosity=0, seed=1, max_depth=3, multi_strategy="multi_output_tree", **extra)
    params |= dict(objective="reg:squarederror", num_parallel_tree=parallel)
    xgb.train(params, xgb.DMatrix(x, label=y), rounds).save_model(str(path))
    return x


class StreamEstimateTest(unittest.TestCase):
    """`select_stream.wanted`: does the in-memory path fit `--memory-mb`? A vector forest holds one int32 leaf code a (tree, row), as a
    scalar one holds one float32 (`prune.vector_leaves`: the (trees, rows, W) float array never exists; measured: 300 trees x 40,000 rows
    x W = 5 peak at 69 MB where W times the scalar estimate said 229 MB), so W must not multiply the (tree, row) term."""

    @staticmethod
    def stub(d: Path, name: str, trees: int, width: int) -> Path:
        p = d / name
        p.write_text(json.dumps({"gbtree_model_param": {"num_trees": str(trees)}, "tree_param": {"size_leaf_vector": str(width)}}))
        return p

    def test_a_vector_forest_is_not_charged_w_times_for_its_leaf_array(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            one, wide = self.stub(d, "one.json", 400000, 1), self.stub(d, "wide.json", 400000, 10)
            # 400,000 trees x 1,000 rows: 1.6 GB of leaf codes either way; the 3,000 MB budget holds that and the 250 MB of working space
            self.assertFalse(select_stream.wanted(one, 1000, 3000))
            self.assertFalse(select_stream.wanted(wide, 1000, 3000), "W = 10 must not turn 1.6 GB of codes into 16 GB")

    def test_the_estimate_still_grows_with_trees_and_rows(self):
        with tempfile.TemporaryDirectory() as d:
            wide = self.stub(Path(d), "wide.json", 400000, 10)
            self.assertTrue(select_stream.wanted(wide, 5000, 3000))  # 8 GB of codes
            self.assertTrue(select_stream.wanted(wide, 1000, 1500))


@unittest.skipIf(xgb is None, "xgboost is not installed")
class XgbVectorPruneTest(unittest.TestCase):
    """`XgbSource.save_pruned` on REAL vector-leaf models (the prune tests use a fake source): xgboost must load the file and answer the sum
    of the kept trees, a random forest's leaves scaled by trees / kept, from the same leaf codes and tables the selection used."""

    def check(self, path: Path, x, kept: list[int], scale: float):
        src = qf_formats.open_source(path)
        dest = path.with_name(path.stem + "-pruned.json")
        src.save_pruned([list(kept)] * src.spec.groups, bool(src.spec.rf), dest)
        got = xgb.Booster(model_file=str(dest)).predict(xgb.DMatrix(x), output_margin=True).reshape(len(x), -1)
        codes, tables = src.leaf_codes(x), src.leaf_vectors()
        expect = np.tile(src.spec.base, (len(x), 1)).astype(np.float32)
        for i in kept:
            expect += (tables[i][codes[i]] * np.float32(scale)).astype(np.float32)
        np.testing.assert_allclose(got, expect, atol=2e-6, rtol=0)
        again = qf_formats.open_source(dest)
        self.assertEqual((len(again.spec.nodes), again.spec.parallel), (len(kept), len(kept) if src.spec.rf else 1))

    def test_a_vector_model_has_no_scalar_leaf_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gb.json"
            x = train_vector(p, 2, eta=0.3)
            with self.assertRaisesRegex(ValueError, "no scalar leaf matrix"):
                qf_formats.open_source(p).leaf_matrix(x)

    def test_a_pruned_vector_random_forest_is_rescaled_and_loads(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "rf.json"
            x = train_vector(p, 1, parallel=4, eta=1, subsample=0.63, colsample_bynode=0.8)
            self.check(p, x, [3, 1], 2.0)
            self.check(p, x, [2], 4.0)

    def test_a_pruned_boosted_vector_model_keeps_any_trees_unscaled(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "gb.json"
            x = train_vector(p, 8, eta=0.3)
            self.check(p, x, [0, 1, 2, 3], 1.0)
            self.check(p, x, [7, 2, 5], 1.0)  # not a prefix, not in order

    def test_unreachable_slots_keep_codes_tables_and_the_pruned_file_aligned(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "orphan.json"
            x = train_vector(p, 3, eta=0.3)
            doc = json.loads(p.read_text())
            t = doc["learner"]["gradient_booster"]["model"]["trees"][1]
            n, w = len(t["left_children"]), int(t["tree_param"]["size_leaf_vector"])
            leaves = len(t["leaf_weights"]) // w
            # an orphan split with two leaves, then an orphan leaf, appended behind the real nodes: no path reaches them
            extra = [(n + 1, n + 2, 0), (-1, leaves, n), (-1, leaves + 1, n), (-1, leaves + 2, 2147483647)]
            for left, right, parent in extra:
                for key, val in (
                    ("left_children", left),
                    ("right_children", right),
                    ("split_indices", 0),
                    ("split_conditions", 0.25),
                    ("default_left", 1),
                    ("parents", parent),
                    ("split_type", 0),
                    ("loss_changes", 0.0),
                    ("sum_hessian", 1.0),
                ):
                    t[key].append(val)
            t["leaf_weights"] += [100.0, 200.0, 300.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            t["base_weights"] += [0.0] * 3 + [100.0, 200.0, 300.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
            t["tree_param"]["num_nodes"] = str(n + 4)
            p.write_text(json.dumps(doc))
            src = qf_formats.open_source(p)
            self.assertEqual(int(src.spec.nodes[1]), n)  # the reachable nodes only
            ref = xgb.Booster(model_file=str(p)).predict(xgb.DMatrix(x), output_margin=True).reshape(len(x), -1)
            codes, tables = src.leaf_codes(x), src.leaf_vectors()
            total = np.tile(src.spec.base, (len(x), 1)).astype(np.float32)
            for i, v in enumerate(tables):
                total += v[codes[i]]
            np.testing.assert_allclose(total, ref, atol=2e-6, rtol=0)
            self.check(p, x, [1, 0], 1.0)


@unittest.skipIf(xgb is None, "xgboost is not installed")
class XgbHeaderTest(unittest.TestCase):
    """A header field that sizes arrays (`num_target`, `num_class`) is checked against the trees BEFORE anything is allocated from it."""

    def edited(self, d: Path, **param) -> Path:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 4)).astype(np.float32)
        p = d / "m.json"
        xgb.train({"verbosity": 0, "nthread": 1, "max_depth": 2}, xgb.DMatrix(x, label=x[:, 0]), 3).save_model(str(p))
        doc = json.loads(p.read_text())
        doc["learner"]["learner_model_param"].update(param)
        p.write_text(json.dumps(doc))
        return p

    def test_outputs_declared_beyond_the_trees_are_refused_in_one_line(self):
        with tempfile.TemporaryDirectory() as d:
            for field in ("num_target", "num_class"):
                p = self.edited(Path(d), **({"num_target": "100000000"} if field == "num_target" else {"num_class": "100000000"}))
                if field == "num_class":
                    doc = json.loads(p.read_text())
                    doc["learner"]["objective"] = {"name": "multi:softprob", "softmax_multiclass_param": {"num_class": "100000000"}}
                    p.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError, "declares 100000000 outputs but the model holds 3 trees"):
                    qf_formats.open_source(p)

    def test_a_model_with_a_tree_for_every_output_still_opens(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(qf_formats.open_source(self.edited(Path(d))).spec.groups, 1)


if __name__ == "__main__":
    unittest.main()
