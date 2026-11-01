"""The flat file of vector trees (QuickForest 1.2, format 7): the writer `backend_flat.pack`, `evaluator.FlatForest` on it, and its refusals.

The `layout` tests read the bytes with a small struct parser of DESIGN A1 and need numpy only (well under a second):

python3 -s -m pytest tests/test_vector_flat.py -k layout

The `engine` tests load the file through the Rust library and skip themselves when the library at hand does not read format 7 (a build
from before 1.2); build it with rust/build.sh or point TREEFLAT_LIB at it.
"""

import dataclasses
import itertools
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_flat
import backend_wave
import evaluator
import qf_common as qf
from vector_forests import explode, make_vector_forest

HEADER = 64
VECTOR_GROUP_U32 = 0xFFFF_FFFF


def fnv1a(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def pack(forest, exact=False):
    return backend_flat.pack(forest, fnv1a, exact)


def parse(blob: bytes) -> dict:
    """The sections of a format-7 file by A1, checking its size, its checksum and the 8-byte alignment of every table."""
    head = struct.unpack_from("<8sIIIIIIII", blob, 0)
    magic, fmt, n_features, groups, n_folds, n_trees, n_nodes, objective, average = head
    checksum, flags, cat_words, vec_leaves, n_segments = struct.unpack_from("<QIIII", blob, 40)
    assert magic == b"TREEFLT1"
    at = HEADER
    folds = [struct.unpack_from("<II", blob, at + 8 * i) for i in range(n_folds)]
    at += 8 * n_folds
    base_at, at = at, at + -(-(n_folds * groups * 4) // 8) * 8
    trees = [struct.unpack_from("<II", blob, at + 8 * i) for i in range(n_trees)]
    at += 8 * n_trees
    node_at, at = at, at + 8 * n_nodes
    cats_at, at = at, at + -(-(cat_words * 4) // 8) * 8
    width = 8 if flags & 32 else 4
    vec_at, at = at, at + -(-(vec_leaves * groups * width) // 8) * 8
    seg_at, at = at, at + (-(-((n_segments + 1) * 4) // 8) * 8 if n_segments else 0)
    assert at == len(blob), (at, len(blob))
    assert all(x % 8 == 0 for x in (node_at, cats_at, vec_at, seg_at))
    assert checksum == fnv1a(blob[:40] + bytes(8) + blob[48:])
    table = np.frombuffer(blob, "<f8" if width == 8 else "<f4", vec_leaves * groups, vec_at).reshape(vec_leaves, groups)
    segments = list(struct.unpack_from(f"<{n_segments + 1}I", blob, seg_at)) if n_segments else None
    nodes = np.frombuffer(blob, "<u8", n_nodes, node_at)
    base = np.frombuffer(blob, "<f4", n_folds * groups, base_at).reshape(n_folds, groups)
    return dict(
        format=fmt,
        n_features=n_features,
        groups=groups,
        n_trees=n_trees,
        n_nodes=n_nodes,
        objective=objective,
        average=average,
        flags=flags,
        folds=folds,
        trees=trees,
        nodes=nodes,
        table=table,
        segments=segments,
        base=base,
        vec_leaves=vec_leaves,
        n_segments=n_segments,
    )


def walk(f: dict, tree: int, x: np.ndarray) -> np.ndarray:
    """The vector-table row that the row `x` reaches in tree number `tree` of a parsed file without extended nodes (x < threshold goes
    left, NaN where the default says); the leaf is the table row it names."""
    i = f["trees"][tree][0]
    while True:
        w = int(f["nodes"][i])
        dist = (w >> 32) & 0xFFFF
        if dist == 0:
            assert w >> 32 == 0
            return f["table"][w & 0xFFFFFFFF]
        v = x[(w >> 48) & 0x7FFF]
        go_left = bool(w >> 63) if np.isnan(v) else v < np.uint32(w & 0xFFFFFFFF).view(np.float32)
        i += 1 if go_left else dist


def rows(n=80, features=6, seed=1):
    x = np.random.default_rng(seed).normal(0, 1.5, (n, features)).astype(np.float32)
    x[np.random.default_rng(seed + 1).random(x.shape) < 0.1] = np.nan
    return x


class LayoutTest(unittest.TestCase):
    def test_layout_the_header_and_tables_of_a_known_forest(self):
        # two stumps and a depth-2 tree, width 3: 2 + 2 + 3 = 7 vector leaves in file order
        def tree(left, right, rows_):
            n = len(left)
            vector = np.zeros((n, 3), np.float32)
            for node, row in rows_.items():
                vector[node] = row
            return qf.Tree(
                qf.VECTOR_GROUP,
                np.array(left, np.int32),
                np.array(right, np.int32),
                np.zeros(n, np.int32),
                np.zeros(n, np.float32),
                np.ones(n, bool),
                None,
                None,
                vector=vector,
            )

        a = tree([1, -1, -1], [2, -1, -1], {1: [1, 2, 3], 2: [4, 5, 6]})
        # breadth-first numbering: root 0 -> (1, 2); 1 -> (3, 4); leaves 2, 3, 4. File (depth-first) order is 3, 4, 2.
        b = tree([1, 3, -1, -1, -1], [2, 4, -1, -1, -1], {2: [7, 8, 9], 3: [10, 11, 12], 4: [13, 14, 15]})
        forest = qf.Forest(
            4, 3, "softmax", "output", [qf.ModelIR(np.array([0.5, 0, -0.5], np.float32), [a, b, a])], "auto", False, segments=[0, 1, 3]
        )
        f = parse(pack(forest))
        self.assertEqual((f["format"], f["groups"], f["n_trees"], f["n_nodes"], f["objective"]), (7, 3, 3, 11, 2))
        self.assertEqual(f["flags"], backend_flat.FLAG_VECTOR)
        self.assertEqual((f["vec_leaves"], f["n_segments"], f["segments"]), (7, 2, [0, 1, 3]))
        self.assertEqual([g for _, g in f["trees"]], [VECTOR_GROUP_U32] * 3)
        self.assertEqual(f["table"].tolist(), [[1, 2, 3], [4, 5, 6], [10, 11, 12], [13, 14, 15], [7, 8, 9], [1, 2, 3], [4, 5, 6]])
        # the leaf words count 0, 1, ... in file order: tree a (nodes 0-2), tree b (3-7), tree a again (8-10)
        leaf_words = [int(w) for w in f["nodes"] if (int(w) >> 32) & 0xFFFF == 0]
        self.assertEqual(leaf_words, list(range(7)))
        self.assertEqual(f["base"].tolist(), [[0.5, 0, -0.5]])

    def test_layout_a_file_walked_by_hand_gives_the_forests_answer(self):
        for seed, (width, depth, folds) in enumerate(itertools.product((2, 5, 16), (1, 4, 9), (1, 3))):
            with self.subTest(width=width, depth=depth, folds=folds):
                forest = make_vector_forest(seed, width=width, depth=depth, trees=4, folds=folds, sum64=True)
                f = parse(pack(forest))
                x = rows(40)
                want = qf.ir_predict(forest, x)
                for r in range(len(x)):
                    got = np.zeros(width)
                    for first, count in f["folds"]:
                        margin = f["base"][f["folds"].index((first, count))].astype(np.float64)
                        for t in range(first, first + count):
                            margin = margin + walk(f, t, x[r])
                        got += margin / len(f["folds"])
                    np.testing.assert_allclose(got, want[r], rtol=1e-12, atol=1e-12)

    def test_layout_a_float64_table_is_written_only_when_float32_cannot_hold_it(self):
        wide = make_vector_forest(3, width=4, table64=True)
        exact, narrow = parse(pack(wide, exact=True)), parse(pack(wide, exact=False))
        self.assertEqual(exact["flags"] & backend_flat.FLAG_VEC64, backend_flat.FLAG_VEC64)
        self.assertEqual(exact["flags"] & backend_flat.FLAG_SUM64, backend_flat.FLAG_SUM64)
        self.assertEqual(narrow["flags"] & backend_flat.FLAG_VEC64, 0)
        self.assertEqual(exact["table"].dtype, np.dtype("<f8"))
        self.assertEqual(narrow["table"].dtype, np.dtype("<f4"))
        # every row of the f64 table is some tree's vector64 row, in the same order as the f32 table
        np.testing.assert_array_equal(exact["table"].astype(np.float32), narrow["table"])
        self.assertTrue((exact["table"] != narrow["table"]).any())
        # a forest whose f64 rows are all float32 numbers packs identically with and without --f64-leaves
        same = make_vector_forest(4, width=3, table64=True)
        for m in same.models:
            for t in m.trees:
                t.vector64 = t.vector.astype(np.float64)
        self.assertEqual(pack(same, exact=True), pack(same, exact=False))
        # float32 sums have nothing to keep in double
        plain = make_vector_forest(5, width=3)
        self.assertEqual(pack(plain, exact=True), pack(plain, exact=False))

    def test_layout_a_one_output_forest_has_no_segment_table(self):
        forest = make_vector_forest(6, width=5, kind="softmax")
        f = parse(pack(forest))
        self.assertEqual((f["n_segments"], f["segments"]), (0, None))
        shares = dataclasses.replace(make_vector_forest(6, width=5), task="classification")  # random-forest class shares
        self.assertEqual(parse(pack(shares))["segments"], None)
        with_segments = dataclasses.replace(forest, segments=[0, 2, 5])
        g = parse(pack(with_segments))
        self.assertEqual((g["n_segments"], g["segments"]), (2, [0, 2, 5]))
        self.assertEqual(len(pack(with_segments)) - len(pack(forest)), 16)  # three u32 offsets padded to 16 bytes

    def test_layout_the_writer_derives_one_segment_per_column_for_regressions_and_sigmoids(self):
        for kind, task, want in (
            ("identity", "auto", [0, 1, 2, 3, 4]),
            ("identity", "regression", [0, 1, 2, 3, 4]),
            ("sigmoid", "auto", [0, 1, 2, 3, 4]),
            ("identity", "classification", None),
            ("softmax", "auto", None),
            ("exp", "auto", None),
        ):
            with self.subTest(kind=kind, task=task):
                forest = dataclasses.replace(make_vector_forest(13, width=4, kind=kind), task=task)
                self.assertEqual(backend_flat.output_segments(forest), want)
                self.assertEqual(parse(pack(forest))["segments"], want)
        given = dataclasses.replace(make_vector_forest(13, width=4), segments=[0, 3, 4])
        self.assertEqual(parse(pack(given))["segments"], [0, 3, 4])

    def test_layout_every_objective_and_sum_type_keeps_its_flags(self):
        for kind, sum64, ext in itertools.product(("identity", "softmax", "sigmoid", "exp"), (False, True), (False, True)):
            with self.subTest(kind=kind, sum64=sum64, ext=ext):
                forest = make_vector_forest(7, width=3, kind=kind, sum64=sum64, ext=ext)
                f = parse(pack(forest))
                want = backend_flat.FLAG_VECTOR | (backend_flat.FLAG_SUM64 if forest.sum64 else 0) | (backend_flat.FLAG_EXT if ext else 0)
                self.assertEqual(f["flags"], want)
                self.assertEqual(f["objective"], qf.OBJECTIVE[kind])

    def test_layout_a_scalar_forest_never_gets_a_vector_mark(self):
        # a scalar forest never carries the vector flags; it is format 7 only when it has a segment table to carry (K regression outputs, fix O2)
        scalar = explode(make_vector_forest(8, width=4, ext=True))
        scalar.task = "classification"  # class shares of one output: no table, so below format 7
        f = struct.unpack_from("<I", pack(scalar), 8)[0], struct.unpack_from("<I", pack(scalar), 48)[0]
        self.assertLess(f[0], 7)
        self.assertEqual(f[1] & (backend_flat.FLAG_VECTOR | backend_flat.FLAG_VEC64), 0)
        scalar.task = "regression"
        blob = pack(scalar)
        f = struct.unpack_from("<I", blob, 8)[0], struct.unpack_from("<I", blob, 48)[0]
        self.assertEqual(f[0], 7)
        self.assertEqual(f[1] & (backend_flat.FLAG_VECTOR | backend_flat.FLAG_VEC64), 0)
        self.assertEqual(struct.unpack_from("<II", blob, 56), (0, 4))

    def test_layout_a_deep_tree_keeps_the_leaf_order_of_its_depth_first_nodes(self):
        forest = make_vector_forest(9, width=2, depth=12, trees=1, n_features=3, sum64=True)
        f = parse(pack(forest))
        leaves = [int(w) for w in f["nodes"] if (int(w) >> 32) & 0xFFFF == 0]
        self.assertEqual(leaves, list(range(len(leaves))))
        self.assertEqual(f["vec_leaves"], len(leaves))


class RefusalTest(unittest.TestCase):
    def test_layout_the_writer_refuses_each_forest_it_must_not_write(self):
        good = make_vector_forest(10, width=3, sum64=True)
        scalar_tree = explode(good).models[0].trees[0]
        mixed = dataclasses.replace(good, models=[qf.ModelIR(good.models[0].base, [*good.models[0].trees, scalar_tree])])
        scaled = dataclasses.replace(good, models=[qf.ModelIR(good.models[0].base, good.models[0].trees, margin_scale=0.5)])
        nan_row = make_vector_forest(10, width=3, sum64=True)
        nan_row.models[0].trees[0].vector[np.flatnonzero(nan_row.models[0].trees[0].left == -1)[0], 0] = np.nan
        bad_segments = dataclasses.replace(good, segments=[0, 2])
        for what, forest in (("mixed", mixed), ("margin scale", scaled), ("nan row", nan_row), ("segments", bad_segments)):
            with self.subTest(what), self.assertRaises(ValueError) as cm:
                pack(forest)
            self.assertTrue(str(cm.exception))

    def test_layout_a_vector_forest_goes_to_no_backend_that_cannot_hold_it(self):
        forest = make_vector_forest(11, width=3)
        self.assertIn("vector_leaves", forest.requires - set(backend_wave.WaveBackend.supports))
        self.assertIn("vector_leaves", backend_flat.FlatBackend.supports)


MULTI = HERE / "fixtures" / "multi"


class OneOutputFilesStayTheSameTest(unittest.TestCase):
    """V5: a model with one output compiles, with 1.2's writer, to the bytes that 1.1 wrote (fixtures/multi, built from tag v1.1)."""

    def compile_sha(self, model):
        import hashlib

        import backends
        import qf_formats

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m.flat"
            backends.get("flat").compile(qf.load_forest(qf_formats.model_files(model), "output"), out, {"no_lib": True})
            return hashlib.sha256(out.read_bytes()).hexdigest()

    def test_k1_xgboost_models_of_both_strategies_compile_to_the_1_1_bytes(self):
        try:
            import xgboost  # noqa: F401
        except ImportError:
            self.skipTest("xgboost is not installed")
        want = (MULTI / "k1_xgb_mot.sha256").read_text().split()[0]
        self.assertEqual((MULTI / "k1_xgb_oopt.sha256").read_text().split()[0], want)
        for name in ("k1_xgb_mot_model.ubj", "k1_xgb_oopt_model.ubj"):
            with self.subTest(name):
                self.assertEqual(self.compile_sha(MULTI / name), want)

    def test_a_binary_ydf_random_forest_compiles_to_the_1_1_bytes(self):
        try:
            import ydf  # noqa: F401
        except ImportError:
            self.skipTest("ydf is not installed")
        want = (MULTI / "k1_ydf_rf_binary.sha256").read_text().split()[0]
        self.assertEqual(self.compile_sha(MULTI / "ydf_rf_binary"), want)


def format7_library_ok() -> bool:
    """True when the library at hand opens a format-7 file."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "probe.flat"
            path.write_bytes(backend_flat.pack(make_vector_forest(0, width=2, trees=1, depth=1), lambda b: evaluator.fnv1a(b)))
            evaluator.FlatForest(path).close()
        return True
    except (ValueError, FileNotFoundError):
        return False


def bits(a):
    return np.ascontiguousarray(a).view(np.uint64 if a.dtype == np.float64 else np.uint32)


ROW_COUNTS = (1, 3, 4, 1023, 1025, 2100)


STALE_LIBRARY = "the evaluator library at hand does not read format 7 (build the current one with rust/build.sh)"


def require_format7_library():
    """Skip when the library is stale; FAIL instead when QF_REQUIRE_FORMAT7=1 (a fresh worktree ships a pre-format-7 .so)."""
    if format7_library_ok():
        return
    if os.environ.get("QF_REQUIRE_FORMAT7") == "1":
        raise AssertionError(STALE_LIBRARY)
    raise unittest.SkipTest(STALE_LIBRARY)


class StaleLibraryTest(unittest.TestCase):
    """A stale library skips by default and fails loudly under QF_REQUIRE_FORMAT7=1."""

    def test_stale_library_skips_by_default_and_fails_when_required(self):
        from unittest import mock

        with mock.patch.object(sys.modules[__name__], "format7_library_ok", return_value=False):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("QF_REQUIRE_FORMAT7", None)
                with self.assertRaises(unittest.SkipTest):
                    require_format7_library()
            with mock.patch.dict(os.environ, {"QF_REQUIRE_FORMAT7": "1"}):
                with self.assertRaisesRegex(AssertionError, "rust/build.sh"):
                    require_format7_library()
        with mock.patch.object(sys.modules[__name__], "format7_library_ok", return_value=True):
            with mock.patch.dict(os.environ, {"QF_REQUIRE_FORMAT7": "1"}):
                require_format7_library()


class EngineTest(unittest.TestCase):
    """V1, V2, V8, V9: the Rust library on the files `pack` writes."""

    @classmethod
    def setUpClass(cls):
        require_format7_library()
        cls.dir = tempfile.TemporaryDirectory()
        cls.tmp = Path(cls.dir.name)

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def open(self, forest, name, exact=False, engine=0):
        path = self.tmp / f"{name}.flat"
        path.write_bytes(backend_flat.pack(forest, lambda b: evaluator.fnv1a(b), exact))
        return evaluator.FlatForest(path, engine=engine)

    def compare(self, forest, label, exact=False, reference=None):
        """The vector file against the file of the exploded forest: bit for bit, every thread count, row count, base margin, answer type."""
        vec, ref = (
            self.open(forest, f"{label}-v", exact),
            self.open(explode(forest) if reference is None else reference, f"{label}-e", exact),
        )
        self.addCleanup(vec.close)
        self.addCleanup(ref.close)
        x = rows(max(ROW_COUNTS), forest.n_features, seed=len(label))
        bm = np.random.default_rng(1).normal(0, 1, (len(x), forest.groups))
        for n, threads, use_bm, dtype in itertools.product(ROW_COUNTS, (1, 3), (False, True), (np.float32, np.float64)):
            kw = dict(threads=threads, base_margin=bm[:n] if use_bm else None, dtype=dtype)
            with self.subTest(label=label, rows=n, threads=threads, base_margin=use_bm, dtype=dtype.__name__):
                got, want = vec.predict(x[:n], **kw), ref.predict(x[:n], **kw)
                self.assertEqual(got.shape, (n, forest.groups))
                self.assertTrue(np.array_equal(bits(got), bits(want)))

    def test_engine_the_flat_walker_answers_what_the_exploded_forest_answers(self):
        cases = [
            dict(width=2, depth=1, trees=1),
            dict(width=2, depth=12, trees=3, n_features=4),
            dict(width=5, depth=4, trees=6, folds=3),
            dict(width=5, depth=6, trees=4, average="margin", folds=2, kind="softmax"),
            dict(width=16, depth=5, trees=3, kind="softmax"),
            dict(width=5, depth=5, trees=4, kind="exp"),
            dict(width=3, depth=5, trees=5, sum64=True, folds=2),
            dict(width=5, depth=7, trees=4, ext=True, kind="softmax", folds=3),
            dict(width=16, depth=3, trees=2, ext=True, average="margin", folds=2),
            dict(width=4, depth=6, trees=4, table64=True, kind="exp", folds=2),
        ]
        for seed, kw in enumerate(cases):
            self.compare(make_vector_forest(seed, **kw), f"case{seed}")

    def test_engine_a_float64_table_answers_what_the_exploded_f64_leaves_answer(self):
        for seed, kw in enumerate(
            [dict(width=4, depth=5, trees=4, table64=True), dict(width=6, depth=3, trees=3, table64=True, ext=True, folds=2)]
        ):
            self.compare(make_vector_forest(100 + seed, **kw), f"f64-{seed}", exact=True)

    def test_engine_the_flat_walker_matches_the_reference_walk_within_the_bar(self):
        for seed, kw in enumerate([dict(width=5, depth=6, trees=5, ext=True), dict(width=3, depth=4, trees=8, kind="softmax", folds=2)]):
            forest = make_vector_forest(200 + seed, **kw)
            fo = self.open(forest, f"ir{seed}")
            self.addCleanup(fo.close)
            x = rows(300, forest.n_features, seed=seed)
            want = qf.ir_predict(forest, x)
            np.testing.assert_allclose(fo.predict(x, threads=1), want, rtol=1e-6, atol=1e-6)

    def test_engine_a_sigmoid_over_several_groups_is_a_sigmoid_per_column(self):
        forest = make_vector_forest(300, width=3, depth=4, trees=5, kind="sigmoid", sum64=True)
        raw = self.open(dataclasses.replace(forest, kind="identity"), "sig-raw")
        fo = self.open(forest, "sig")
        self.addCleanup(raw.close)
        self.addCleanup(fo.close)
        x = rows(200, forest.n_features, seed=5)
        np.testing.assert_allclose(fo.predict(x), 1 / (1 + np.exp(-raw.predict(x).astype(np.float64))), rtol=1e-6, atol=1e-7)

    def test_engine_segments_and_outputs_are_read_from_the_file(self):
        forest = make_vector_forest(400, width=5, segments=[0, 2, 5])
        fo = self.open(forest, "seg")
        self.addCleanup(fo.close)
        self.assertEqual(fo.segments, [0, 2, 5])
        self.assertEqual((fo.n_outputs, fo.output_kind), (2, "classification"))
        plain = self.open(dataclasses.replace(make_vector_forest(401, width=5), task="classification"), "noseg")
        self.addCleanup(plain.close)
        self.assertEqual((plain.segments, plain.n_outputs, plain.output_kind), (None, 1, "classification"))
        reg = self.open(make_vector_forest(403, width=3), "reg")
        self.addCleanup(reg.close)
        self.assertEqual((reg.segments, reg.n_outputs, reg.output_kind), ([0, 1, 2, 3], 3, "regression"))
        sig = self.open(make_vector_forest(404, width=3, kind="sigmoid"), "sig-cols")
        self.addCleanup(sig.close)
        self.assertEqual((sig.n_outputs, sig.output_kind), (3, "classification"))
        scalar = self.open(explode(make_vector_forest(402, width=1 + 1)), "scalar")
        self.addCleanup(scalar.close)
        # a scalar regression forest (one tree per output) names its K outputs the same way (fix O2)
        self.assertEqual((scalar.segments, scalar.n_outputs, scalar.output_kind), ([0, 1, 2], 2, "regression"))
        shares = self.open(dataclasses.replace(explode(make_vector_forest(405, width=2)), task="classification"), "scalar-shares")
        self.addCleanup(shares.close)
        self.assertEqual((shares.segments, shares.n_outputs, shares.output_kind), (None, 1, "classification"))


if __name__ == "__main__":
    unittest.main()
