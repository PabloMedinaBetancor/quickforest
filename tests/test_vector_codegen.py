"""Generated code for vector trees (QuickForest 1.2, DISENO A5/A6): the source it writes, its refusals, and V1, the generated library against the
scalar forest `explode` makes of the same trees, bit for bit, on f32 and f64 answers, 1/3/8 threads and the row counts of the compressed design.

The sections that build run rustc (about 2-4 minutes per build on a laptop): they skip without a toolchain, and set QF_VECTOR_VARIANTS=name,name to
build only those variants of `VARIANTS` (a partial run; the gate runs all)."""

import hashlib
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_codegen as cg
import backends
import qf_common as qf
from vector_forests import explode, make_vector_forest

ROW_COUNTS = (1, 3, 4, 1023, 1025, 2100)
THREADS = (1, 3, 8)
NF = 6


def rows(n, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1.5, (n, NF)).astype(np.float32)
    x[::7, 0] = np.nan
    x[::11, 1] = 0.0
    x[::13, 3] = 1e-36
    x[::5, 2] = rng.integers(-2, 45, len(x[::5, 2]))  # categorical codes, some out of range
    return x


def toolchain() -> bool:
    try:
        cg.toolchain_bin()
    except FileNotFoundError:
        return False
    return True


def stump(low, high, group=qf.VECTOR_GROUP, vector=None):
    value = np.zeros(3, np.float32)
    value[0] = 0.5
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


class SourceTest(unittest.TestCase):
    """What the generator writes: needs no rustc."""

    def test_leaf_ordinals_follow_the_emission_order_and_the_table_follows_them(self):
        f = make_vector_forest(3, width=4, depth=5, trees=1)
        t = f.models[0].trees[0]
        expr, table = cg.vector_expr(t, False, 7)
        ordinals = [int(v) for v in re.findall(r"&V7\[(\d+)\]", expr)]
        self.assertEqual(ordinals, list(range(int((t.left == -1).sum()))))  # each leaf once, numbered as it is written
        # the table's j-th row is the row of the j-th leaf the expression reaches: walk the tree in emission order
        order, stack = [], [0]
        while stack:
            v = stack.pop()
            if t.left[v] == -1:
                order.append(v)
            else:
                stack += [int(t.right[v]), int(t.left[v])]
        self.assertTrue(table.startswith(f"static V7: [[f32; G]; {len(order)}] = [["))
        cells = [
            float(v.removesuffix("f32"))
            for row in table.split("= [[", 1)[1].rstrip("];\n").split("],[")
            for v in row.strip("[]").split(",")
        ]
        self.assertEqual(np.float32(cells).tolist(), t.vector[order].reshape(-1).tolist())

    def test_a_double_table_is_used_only_for_exact_trees_with_rows_in_double(self):
        f = make_vector_forest(4, width=3, depth=3, trees=1, table64=True)
        t = f.models[0].trees[0]
        self.assertEqual(cg.vector_type(t, True), "f64")
        self.assertEqual(cg.vector_type(t, False), "f32")
        self.assertIn("f64; G]", cg.vector_expr(t, True, 0)[1])
        plain = make_vector_forest(4, width=3, depth=3, trees=1).models[0].trees[0]
        self.assertEqual(cg.vector_type(plain, True), "f32")  # no vector64: nothing to widen

    def test_a_root_leaf_is_a_valid_vector_tree(self):
        t = qf.Tree(
            qf.VECTOR_GROUP,
            np.array([-1], np.int32),
            np.array([-1], np.int32),
            np.zeros(1, np.int32),
            np.zeros(1, np.float32),
            np.array([False]),
            vector=np.array([[1, 2]], np.float32),
        )
        expr, table = cg.vector_expr(t, False, 0)
        self.assertEqual(expr, "&V0[0]")
        self.assertIn("[[1f32,2f32]]", table.replace("1.0f32", "1f32").replace("2.0f32", "2f32"))

    def test_row_and_block_chunks_use_the_strided_vector_add(self):
        f = make_vector_forest(5, width=3, trees=2)
        trees = f.models[0].trees
        row = cg.chunk_source(trees, 0, 0, NF, 3, "row")
        block = cg.chunk_source(trees, 0, 0, NF, 3, "block")
        self.assertIn("-> &'static [f32; G]", row)
        self.assertIn("for g in 0..G { a[g] += v[g]; }", row)
        self.assertIn("*acc.add(g * stride + r) += v[g];", block)
        self.assertIn("b0(xs, rows, m, stride);", block)  # the base pointer, not m.add(group * stride)
        self.assertNotIn("m.add(", block)
        # double sums over a float32 table cast each entry; a double table needs no cast
        self.assertIn("v[g] as f64", cg.chunk_source(trees, 0, 0, NF, 3, "row", acc="f64"))
        wide = make_vector_forest(5, width=3, trees=2, table64=True).models[0].trees
        self.assertNotIn("as f64", cg.chunk_source(wide, 0, 0, NF, 3, "row", acc="f64", exact=True))
        with self.assertRaises(ValueError):
            cg.chunk_source(wide, 0, 0, NF, 3, "row", acc="f32", exact=True)  # f64 rows need double sums

    def test_modules_import_the_group_count_only_when_there_are_vector_trees(self):
        vec = make_vector_forest(6, width=2, trees=4).models[0].trees
        self.assertIn("use super::{NF, G};", cg.chunk_source(vec, 0, 0, NF, 2, "block", units=2))
        scalar = [stump(0.1, 0.2, 0), stump(0.3, 0.4, 1)]
        src = cg.chunk_source(scalar, 0, 0, NF, 2, "block", units=2)
        self.assertIn("use super::NF;", src)
        self.assertNotIn("use super::{NF, G};", src)

    def test_the_driver_exports_segments_for_vector_forests_only(self):
        seg = make_vector_forest(7, width=5, segments=[0, 2, 5])
        src = cg.driver_source(seg, [[0]], 64)
        self.assertIn("static SEGMENTS: [u32; 3] = [0,2,5];", src)
        self.assertIn("tfl_cg_segments(out: *mut u32, cap: usize) -> u32", src)
        plain = make_vector_forest(7, width=3, kind="softmax")
        self.assertIn("static SEGMENTS: [u32; 0] = [];", cg.driver_source(plain, [[0]], 64))
        scalar = qf.Forest(NF, 1, "identity", "output", [qf.ModelIR(np.zeros(1, np.float32), [stump(0.1, 0.2, 0)])])
        self.assertNotIn("tfl_cg_segments", cg.driver_source(scalar, [[0]], 64))

    def test_the_segment_table_follows_rule_r_seg(self):
        def forest(**kw):
            return make_vector_forest(7, width=4, trees=1, **kw)

        self.assertEqual(cg.file_segments(forest(segments=[0, 1, 4])), [0, 1, 4])  # the forest's own table first
        self.assertEqual(cg.file_segments(forest()), [0, 1, 2, 3, 4])  # a K-output regressor: identity, task auto -> regression
        rf = forest()
        rf.task = "classification"
        self.assertEqual(cg.file_segments(rf), [])  # YDF random forest multi-class: ONE output
        self.assertEqual(cg.file_segments(forest(kind="sigmoid")), [0, 1, 2, 3, 4])  # xgboost multilabel
        self.assertEqual(cg.file_segments(forest(kind="softmax")), [])  # softprob: one softmax over the classes
        self.assertEqual(cg.file_segments(forest(kind="exp")), [])

    def test_scalar_forests_generate_the_source_they_always_did(self):
        """K = 1 and every 1.1 forest: the generated sources are byte-identical to the ones the pre-1.2 generator wrote (pins taken at v1.1)."""
        f = scalar_forest()
        got = {}
        for mode in ("row", "block"):
            for acc in ("f32", "f64"):
                chunk = cg.chunk_source(f.models[0].trees, 0, 0, NF, 3, mode, acc, False, 1)
                multi = cg.chunk_source(f.models[0].trees, 0, 0, NF, 3, mode, acc, False, 3)
                got[f"{mode}-{acc}"] = hashlib.sha256((chunk + multi).encode()).hexdigest()[:16]
        got["driver"] = hashlib.sha256(cg.driver_source(f, [[0]], 64, "f32", "T").encode()).hexdigest()[:16]
        self.assertEqual(got, SCALAR_SOURCE_PINS)

    def test_a_vector_tree_weighs_its_leaf_table(self):
        t = make_vector_forest(8, width=16, depth=6, trees=1).models[0].trees[0]
        self.assertEqual(cg.tree_weight(t), len(t.left) + int((t.left == -1).sum()) * 16 // 8)
        self.assertEqual(cg.tree_weight(stump(0, 1, 0)), 3)

    def test_chunks_are_cut_by_weight(self):
        f = make_vector_forest(9, width=16, depth=7, trees=12)
        plan, _ = cg.split_chunks(f, 2_000)
        self.assertGreater(len(plan), 1)
        for _, _, trees in plan:
            if len(trees) > 1:
                self.assertLessEqual(sum(cg.tree_weight(t) for t in trees), 2_000)
        self.assertEqual(sum(len(t) for _, _, t in plan), 12)

    def test_the_table_cap_refuses_and_points_to_the_flat_backend(self):
        f = make_vector_forest(10, width=4, depth=3, trees=3)
        values = cg.vector_table_values(f)
        self.assertGreater(values, 0)
        with mock.patch.object(cg, "CODEGEN_MAX_VECTOR_VALUES", values - 1), tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError) as e:
                backends.get("codegen-block").compile(f, Path(d) / "m.so", {})
            self.assertIn("--backend flat", str(e.exception))
            self.assertIn(f"{values:,}", str(e.exception))
            self.assertEqual(list(Path(d).iterdir()), [])  # refused before anything is written
        self.assertEqual(cg.CODEGEN_MAX_VECTOR_VALUES, 50_000_000)

    def test_a_forest_that_mixes_vector_and_scalar_trees_is_refused(self):
        f = make_vector_forest(11, width=2, trees=1)
        f.models[0].trees.append(stump(0.1, 0.2, 0))
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError) as e:
            backends.get("codegen").compile(f, Path(d) / "m.so", {})
        self.assertIn("mixes", str(e.exception))

    def test_a_group_index_of_minus_one_never_reaches_the_scalar_generator(self):
        t = stump(0.1, 0.2, qf.VECTOR_GROUP)  # a scalar-looking tree carrying the vector group
        f = qf.Forest(NF, 2, "identity", "output", [qf.ModelIR(np.zeros(2, np.float32), [t])])
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            backends.get("codegen").compile(f, Path(d) / "m.so", {})


def scalar_forest():
    """Three groups, two folds' worth of shapes in one model: deterministic, built from `make_vector_forest`'s exploded trees."""
    return explode(make_vector_forest(21, width=3, depth=4, trees=3, ext=True))


SCALAR_SOURCE_PINS = {
    "row-f32": "d7b1e9e5a9d7a029",
    "row-f64": "98c7c620ae022fd9",
    "block-f32": "56f9abc542e89997",
    "block-f64": "277ac166bc641748",
    "driver": "2412d22c926acaab",
}


def reference(forest, x, base_margin, dtype, exact, threads):
    """The answer of the scalar forest `explode` makes, on the flat engine; a sigmoid forest per group (one-group code is today's code)."""
    scalar = explode(forest)
    options = {"f64_leaves": True} if exact else {}

    def flat(fo, bm):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "m.flat"
            backends.get("flat").compile(fo, out, options)
            pred = backends.get("flat").load(out)
            try:
                return pred.predict(x, threads, base_margin=bm, dtype=dtype)
            finally:
                pred.close()

    if forest.kind != "sigmoid":
        return flat(scalar, base_margin)
    cols = []
    for g in range(forest.groups):
        models = []
        for m in scalar.models:
            trees = [qf.Tree(0, *dataclass_fields(t)) for t in m.trees if t.group == g]
            models.append(qf.ModelIR(m.base[g : g + 1].copy(), trees))
        one = qf.Forest(forest.n_features, 1, "sigmoid", forest.average, models, "auto", scalar.sum64, scalar.late_margin)
        bm = None if base_margin is None else np.asarray(base_margin)[:, g]
        cols.append(flat(one, bm))
    return np.stack(cols, axis=1)


def dataclass_fields(t):
    return t.left, t.right, t.feature, t.value, t.default_left, t.zero_missing, t.cat, None, t.leaf64


VARIANTS = {
    # name: (forest keywords, compile options, exact)
    "ident-seg": (dict(width=5, depth=4, trees=6, segments=[0, 2, 5]), {}, False),
    "softmax-folds": (dict(width=3, depth=5, trees=5, folds=2, kind="softmax"), {}, False),
    "sigmoid-ext64": (dict(width=16, depth=6, trees=4, folds=2, kind="sigmoid", ext=True, table64=True), {"f64_leaves": True}, True),
    "exp-margin-deep": (dict(width=2, depth=12, trees=3, kind="exp", average="margin"), {}, False),
    "ident-sum64": (dict(width=4, depth=3, trees=9, sum64=True), {}, False),
}


@unittest.skipUnless(toolchain(), "no rustc")
class BuiltTest(unittest.TestCase):
    """V1: the generated library against the exploded forest, bit for bit. One build per variant and mode (the longest part of the suite)."""

    def check_variant(self, name, mode, **extra):
        key = (name, mode, tuple(sorted(extra.items())))
        if key in self._built:  # a build is minutes: the tests that need a checked library again reuse it
            return self._built[key]
        kw, options, exact = VARIANTS[name]
        forest = make_vector_forest(sum(map(ord, name)), **kw)
        if "segments" in kw:
            forest.task = "classification"
        be = backends.get(mode)
        with tempfile.TemporaryDirectory() as d:
            info = be.compile(forest, Path(d) / "m.so", {**options, **extra})
            pred = be.load(Path(d) / "m.so")
            self.assertEqual(pred.n_groups, forest.groups)
            base = np.random.default_rng(5).normal(0, 0.3, (max(ROW_COUNTS), forest.groups))
            for n in ROW_COUNTS:
                x = rows(n, seed=n)
                for threads in THREADS:
                    for dtype in (np.float32, np.float64):
                        for bm in (None, base[:n]):
                            if n in (1, 3, 4, 1025) and threads == 8 and bm is not None:
                                continue  # bound the reference builds, not the cases: every row count still runs at 1 and 3 threads
                            got = pred.predict(x, threads, base_margin=bm, dtype=dtype)
                            want = self.cached_reference(forest, x, bm, dtype, exact, threads, name)
                            view = np.uint32 if dtype == np.float32 else np.uint64
                            self.assertTrue(
                                np.array_equal(got.view(view), want.view(view)),
                                f"{name} {mode} rows={n} threads={threads} {dtype.__name__} base_margin={bm is not None}",
                            )
            self._built[key] = info, pred
            return info, pred

    _refs: ClassVar[dict] = {}
    _built: ClassVar[dict] = {}

    def cached_reference(self, forest, x, bm, dtype, exact, threads, name):
        # the reference runs one thread (the flat engine is bit-identical across thread counts): one flat compile per call is the cost, so
        # the answers are kept per (variant, rows, base margin, dtype)
        key = (name, len(x), bm is not None, dtype)
        if key not in self._refs:
            self._refs[key] = reference(forest, x, bm, dtype, exact, 1)
        return self._refs[key]

    def run_variants(self, mode):
        names = [n for n in os.environ.get("QF_VECTOR_VARIANTS", "").split(",") if n] or list(VARIANTS)
        for name in names:
            with self.subTest(variant=name, mode=mode):
                self.check_variant(name, mode)

    def test_v1_block_mode_matches_the_exploded_forest(self):
        self.run_variants("codegen-block")

    def test_v1_row_mode_matches_the_exploded_forest(self):
        self.run_variants("codegen")

    def test_v1_one_codegen_unit_asked_by_hand(self):
        self.check_variant("softmax-folds", "codegen-block", compile_preset="1.0", lto="off", codegen_units=1)

    def test_output_segments_and_the_per_output_helpers(self):
        _, pred = self.check_variant("ident-seg", "codegen-block")
        self.assertEqual(pred.segments, [0, 2, 5])
        self.assertEqual(pred.n_outputs, 2)
        self.assertEqual(pred.output_kind, "classification")
        p = np.random.default_rng(2).random((7, 5)).astype(np.float32)
        a, b = pred.split_outputs(p)
        self.assertTrue(np.array_equal(a, p[:, :2]) and np.array_equal(b, p[:, 2:]))
        want = np.stack([p[:, :2].argmax(1), p[:, 2:].argmax(1)], axis=1)
        self.assertTrue(np.array_equal(pred.argmax_outputs(p), want))
        with self.assertRaises(ValueError):
            pred.split_outputs(p[:, :4])  # a wrong width is refused
        _, reg = self.check_variant("ident-sum64", "codegen-block")
        self.assertEqual((reg.segments, reg.output_kind, reg.n_outputs), ([0, 1, 2, 3, 4], "regression", 4))
        cols = reg.split_outputs(np.zeros((3, 4), np.float32))
        self.assertEqual(len(cols), 4)
        _, multilabel = self.check_variant("sigmoid-ext64", "codegen-block")
        self.assertEqual((multilabel.output_kind, multilabel.n_outputs), ("classification", 16))
        decided = multilabel.argmax_outputs(np.tile(np.array([0.2, 0.5, 0.9, 0.49] * 4, np.float32), (2, 1)))
        self.assertEqual(decided.tolist(), [[0, 1, 1, 0] * 4] * 2)  # one column per output: p >= 0.5
        with self.assertRaises(ValueError):
            reg.argmax_outputs(np.zeros((3, 4), np.float32))

    def test_a_scalar_forest_has_no_segments_and_one_output(self):
        f = scalar_forest()
        f.task = "classification"
        with tempfile.TemporaryDirectory() as d:
            backends.get("codegen-block").compile(f, Path(d) / "m.so", {})
            pred = backends.get("codegen-block").load(Path(d) / "m.so")
        self.assertEqual((pred.segments, pred.n_outputs, pred.output_kind), (None, 1, "classification"))


if __name__ == "__main__":
    unittest.main()
