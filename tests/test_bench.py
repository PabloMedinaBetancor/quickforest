"""Tests for the bench scripts (bench/bench_e2e.py). Building the README-size forest must not hold every fold's forest in Python at once
(~2.8 GB) before the optimize child process starts, which needs the same again. `count_nodes` reads one file at a time and must give the
same total as loading the whole forest.

    PYTHONPATH=<venv with xgboost> python3 -s -m unittest test_bench"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parent / "bench"))
import qf_common as qf
import qf_formats
from test_linear import HAVE_XGB


@unittest.skipUnless(HAVE_XGB, "xgboost not installed")
class CountNodesTest(unittest.TestCase):
    def test_one_file_at_a_time_equals_the_whole_forest(self):
        import bench_e2e
        import common as c

        x, y = c.make_data(1500, 6, 3, 7)
        ids = c.kfold_ids(len(x), 3, 7)
        with tempfile.TemporaryDirectory() as d:
            c.train_xgboost(x, y, ids, Path(d), 3, 4, 3, True, 7)
            files = qf_formats.model_files(Path(d))
            self.assertEqual(len(files), 3)
            whole = qf.load_forest(files)
            expect = sum(len(t.left) for m in whole.models for t in m.trees)
            self.assertGreater(expect, 0)
            self.assertEqual(bench_e2e.count_nodes(qf, files), expect)

    def test_the_sizes_are_small_and_the_readme_figure(self):
        import bench_e2e

        self.assertEqual(sorted(bench_e2e.SIZES), ["readme", "small"])
        self.assertEqual(bench_e2e.SIZES["readme"]["trees"], 67)  # the size behind the figures in bench/README-NUMBERS.md


class TrainedModelsAreReusedOnlyForTheSameTraining(unittest.TestCase):
    """bench_e2e keeps its trained models in --work. Fold files of the right count were enough to reuse them: the same folder given another
    --size, --data, --noise or --seed was timed on models that are not the ones the banner names."""

    def args(self, **kw):
        import argparse

        return argparse.Namespace(**{"data": "sparse", "noise": 0.1, "seed": 15, **kw})

    @unittest.skipUnless(HAVE_XGB, "xgboost not installed")
    def test_the_stamp_names_what_the_models_were_trained_from(self):
        import bench_e2e

        base = bench_e2e.training_stamp("small", self.args())
        for other in (self.args(seed=16), self.args(noise=0.35), self.args(data="dense")):
            self.assertNotEqual(bench_e2e.training_stamp("small", other), base)
        self.assertNotEqual(bench_e2e.training_stamp("readme", self.args()), base)
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / "full").mkdir()
            for k in range(base["folds"]):
                (work / "full" / f"fold{k}.ubj").write_bytes(b"x")
            (work / "sel.npz").write_bytes(b"x")
            self.assertFalse(bench_e2e.trained_here(work, base), "fold files without a stamp are another run's")
            (work / "training.json").write_text(json.dumps(base))
            self.assertTrue(bench_e2e.trained_here(work, base))
            self.assertFalse(bench_e2e.trained_here(work, bench_e2e.training_stamp("small", self.args(seed=16))))
            (work / "full" / "fold0.ubj").unlink()
            self.assertFalse(bench_e2e.trained_here(work, base), "a missing fold is a training to redo")


if __name__ == "__main__":
    unittest.main()
