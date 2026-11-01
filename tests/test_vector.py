"""Vector trees (QuickForest 1.2). The `ir` tests need nothing but numpy and run in well under a second:

python3 -s -m pytest tests/test_vector.py -k ir
"""

import dataclasses
import itertools
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_wave
import qf_common as qf
import qf_formats
from vector_forests import explode, make_vector_forest


def stump(group, low, high, vector=None):
    """One split on feature 0 at 0.5 (x < 0.5 goes left); scalar leaves `low`/`high`, or the two rows of `vector`."""
    value = np.array([0.5, 0, 0], np.float32)
    if vector is None:
        value[1:] = low, high
    return qf.Tree(
        group,
        np.array([1, -1, -1], np.int32),
        np.array([2, -1, -1], np.int32),
        np.zeros(3, np.int32),
        value,
        np.array([True, False, False]),
        vector=None if vector is None else np.vstack([np.zeros((1, vector.shape[1]), np.float32), vector]),
    )


def forest_of(trees, groups, **kw):
    return qf.Forest(1, groups, "identity", "output", [qf.ModelIR(np.zeros(groups, np.float32), trees)], **kw)


def rows(n=300, features=6, seed=1):
    x = np.random.default_rng(seed).normal(0, 1, (n, features)).astype(np.float32)
    x[::7, 0] = np.nan
    x[::11, 1] = 0.0
    x[::5, 2] = np.random.default_rng(seed).integers(0, 45, len(x[::5, 2]))
    return x


class IrExplodeTest(unittest.TestCase):
    def test_ir_the_exploded_forest_answers_bit_for_bit(self):
        cases = itertools.product((2, 5, 16), ("identity", "softmax", "sigmoid", "exp"), ("output", "margin"), (False, True))
        for seed, (width, kind, average, ext) in enumerate(cases):
            with self.subTest(width=width, kind=kind, average=average, ext=ext):
                f = make_vector_forest(
                    seed,
                    width=width,
                    depth=1 + seed % (12 if width < 16 else 5),
                    trees=3,
                    kind=kind,
                    average=average,
                    folds=1 + seed % 3,
                    ext=ext,
                )
                x = rows(120)
                bm = np.random.default_rng(seed).normal(0, 1, (len(x), width))
                for base in (None, bm):
                    got, want = qf.ir_predict(f, x, base_margin=base), qf.ir_predict(explode(f), x, base_margin=base)
                    self.assertEqual(got.dtype, want.dtype)
                    self.assertTrue(np.array_equal(got.view(np.uint64), want.view(np.uint64)))

    def test_ir_explode_lays_the_groups_out_tree_by_tree(self):
        f = make_vector_forest(3, width=3, trees=2, table64=True)
        out = explode(f)
        trees = out.models[0].trees
        self.assertEqual([t.group for t in trees], [0, 1, 2, 0, 1, 2])
        for k, t in enumerate(trees):
            v = f.models[0].trees[k // 3]
            self.assertIsNone(t.vector)
            leaf = v.left == -1
            self.assertTrue(np.array_equal(t.value[leaf], v.vector[leaf, k % 3]))
            self.assertTrue(np.array_equal(t.value[~leaf], v.value[~leaf]))
            self.assertTrue(np.array_equal(t.leaf64, v.vector64[:, k % 3]))
            self.assertIs(t.left, v.left)
        self.assertEqual(out.requires, qf.Forest.requires.fget(dataclasses.replace(f, models=out.models)))

    def test_ir_explode_keeps_a_scalar_tree_and_drops_the_segments(self):
        f = make_vector_forest(0, width=4, trees=1, segments=[0, 2, 4])
        out = explode(f)
        self.assertIsNone(out.segments)
        self.assertEqual(len(out.models[0].trees), 4)

    def test_ir_the_known_values_of_two_stumps(self):
        v = stump(qf.VECTOR_GROUP, 0, 0, np.array([[1.0, 10.0], [2.0, 20.0]], np.float32))
        scalar = stump(1, 100.0, 200.0)
        x = np.array([[0.0], [1.0], [np.nan]], np.float32)  # NaN goes left (default_left at the root)
        got = qf.ir_predict(forest_of([v], 2), x)
        self.assertEqual(got.tolist(), [[1.0, 10.0], [2.0, 20.0], [1.0, 10.0]])
        mixed_parts = qf.ir_predict(forest_of([scalar], 2), x)
        self.assertEqual(mixed_parts.tolist(), [[0.0, 100.0], [0.0, 200.0], [0.0, 100.0]])

    def test_ir_tree_values_of_a_vector_tree_is_float32_rows_by_width(self):
        t = make_vector_forest(2, width=5).models[0].trees[0]
        out = qf.tree_values(t, rows())
        self.assertEqual((out.dtype, out.shape), (np.float32, (300, 5)))


class IrContractTest(unittest.TestCase):
    def test_ir_requires_vector_leaves_and_the_walker_backends_declare_it(self):
        f = make_vector_forest(0)
        self.assertIn("vector_leaves", f.requires)
        self.assertNotIn("vector_leaves", explode(f).requires)
        self.assertIn("vector_leaves", qf.FEATURES)
        self.assertIn("vector_leaves", f.requires - set(backend_wave.WaveBackend.supports))
        self.assertEqual(qf.VECTOR_GROUP, -1)

    def test_ir_a_valid_vector_forest_passes_in_every_variant(self):
        for seed, (ext, t64, seg) in enumerate(itertools.product((False, True), (False, True), (None, [0, 2, 5]))):
            qf.check_vector(make_vector_forest(seed, ext=ext, table64=t64, segments=seg))

    def test_ir_a_scalar_forest_has_no_vector_requirement(self):
        f = forest_of([stump(0, 1.0, 2.0)], 1)
        qf.check_vector(f)
        self.assertNotIn("vector_leaves", f.requires)

    def test_ir_check_vector_refuses_each_broken_contract(self):
        def changed(**tree_fields):
            f = make_vector_forest(0, width=3, trees=2, table64=True)
            f.models[0].trees[1] = dataclasses.replace(f.models[0].trees[1], **tree_fields)
            return f

        def with_vector(edit):
            f = make_vector_forest(0, width=3, trees=2, table64=True)
            t = f.models[0].trees[1]
            v = t.vector.copy()
            edit(v, t)
            f.models[0].trees[1] = dataclasses.replace(t, vector=v)
            return f

        t0 = make_vector_forest(0, width=3, trees=2, table64=True).models[0].trees[1]
        leaf = int(np.flatnonzero(t0.left == -1)[0])
        internal = int(np.flatnonzero(t0.left != -1)[0])
        nan64 = t0.vector64.copy()
        nan64[leaf, 0] = np.inf
        scalar = stump(0, 1.0, 2.0)
        nonzero_leaf = t0.value.copy()
        nonzero_leaf[leaf] = 1.0
        bad_internal = t0.vector.copy()
        bad_internal[internal, 0] = 1.0
        cases = {
            "mixes": forest_of([make_vector_forest(0, width=2, trees=1).models[0].trees[0], scalar], 2),
            "one group": make_vector_forest(0, width=2, trees=1).__class__(
                1, 1, "identity", "output", make_vector_forest(0, width=2).models
            ),
            "wrong group": changed(group=0),
            "shape": changed(vector=t0.vector[:, :2]),
            "rows": changed(vector=t0.vector[:-1]),
            "dtype": changed(vector=t0.vector.astype(np.float64)),
            "dtype64": changed(vector64=t0.vector64.astype(np.float32)),
            "nan": with_vector(lambda v, t: v.__setitem__((leaf, 1), np.nan)),
            "inf64": changed(vector64=nan64),
            "internal row": changed(vector=bad_internal),
            "leaf value": changed(value=nonzero_leaf),
            "linear": changed(linear={leaf: (0.0, 0.0, np.zeros(0, np.int32), np.zeros(0))}),
            "leaf64": changed(leaf64=np.zeros(len(t0.left))),
            "scale": make_vector_forest(0, width=2, trees=1),
            "segments none vector": forest_of([scalar], 2, segments=[0, 1, 2]),
        }
        cases["scale"].models[0].margin_scale = 0.5
        for name, f in cases.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, "."):
                qf.check_vector(f)

    def test_ir_check_vector_refuses_bad_segments(self):
        for seg in ([1, 5], [0, 2, 2, 5], [0, 3, 2, 5], [0, 4], [0, 2, 6], [], [0], [5]):
            with self.subTest(seg=seg), self.assertRaisesRegex(ValueError, "segments"):
                qf.check_vector(self.with_segments(seg))

    @staticmethod
    def with_segments(seg):
        f = make_vector_forest(0, width=5)
        return dataclasses.replace(f, segments=seg)

    def test_ir_check_vector_refuses_a_scalar_tree_outside_its_groups(self):
        for group in (-1, 2, 7):
            with self.subTest(group=group), self.assertRaisesRegex(ValueError, "group"):
                qf.check_vector(forest_of([stump(group, 1.0, 2.0)], 2))
        orphan = dataclasses.replace(stump(0, 1.0, 2.0), vector64=np.zeros((3, 2)))
        with self.assertRaisesRegex(ValueError, "group"):
            qf.check_vector(forest_of([orphan], 2))

    def test_ir_predict_never_indexes_with_the_vector_group(self):
        f = forest_of([stump(qf.VECTOR_GROUP, 1.0, 2.0)], 2)  # a scalar tree wearing the vector marker
        with self.assertRaises(ValueError):
            qf.ir_predict(f, np.zeros((4, 1), np.float32))

    def test_ir_spec_vector_width_is_zero_or_the_group_count(self):
        def spec(width, groups=3):
            return qf.Spec(2, groups, "identity", np.zeros(groups, np.float32), np.zeros(1, np.int64), np.ones(1, np.int64), np.zeros(1, np.int64),
                           np.zeros(1, np.float32), 1, vector_width=width)  # fmt: skip

        self.assertEqual((spec(0).vector_width, spec(3).vector_width), (0, 3))
        for width in (1, 2, -1, 4):
            with self.subTest(width=width), self.assertRaisesRegex(ValueError, "vector_width"):
                spec(width)
        with self.assertRaisesRegex(ValueError, "vector_width"):
            spec(1, groups=1)


class IrLoadForestTest(unittest.TestCase):
    @staticmethod
    def source(forest):
        spec = SimpleNamespace(n_features=forest.n_features, groups=forest.groups, kind=forest.kind, task="auto")
        return SimpleNamespace(fmt="ydf", spec=spec, model_ir=lambda *a: forest.models[0], names=None)

    def load(self, forest):
        with mock.patch.object(qf_formats, "open_source", return_value=self.source(forest)):
            return qf.load_forest([Path("m.ydf")])

    def test_ir_load_forest_refuses_a_forest_that_mixes_vector_and_scalar_trees(self):
        v = make_vector_forest(0, width=2, trees=1).models[0].trees[0]
        mixed = forest_of([v, stump(0, 1.0, 2.0)], 2)
        with self.assertRaisesRegex(ValueError, "mixes"):
            self.load(mixed)

    def test_ir_load_forest_passes_a_vector_forest_and_carries_nothing_else(self):
        f = make_vector_forest(4, width=3)
        got = self.load(f)
        self.assertIn("vector_leaves", got.requires)
        self.assertIsNone(got.segments)


if __name__ == "__main__":
    unittest.main()
