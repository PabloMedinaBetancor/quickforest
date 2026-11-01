"""code check 1 (QuickForest 1.2): vector-leaf compile, flat format 7 and generated code. One regression per finding, plus the guards that keep
the duplicated rules (output segments, MAX_GROUPS) from drifting apart.

python3 -m pytest tests/test_codecheck_cc1.py

The `engine` section needs a library that reads format 7: build it with rust/build.sh and point TREEFLAT_LIB at it (QF_REQUIRE_FORMAT7=1 makes
a stale library fail instead of skip)."""

import dataclasses
import itertools
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import backend_codegen as cg
import backend_flat
import evaluator
import qf_common as qf
import qf_sklearn
from vector_forests import explode, make_vector_forest


class ReloadSamePathTest(unittest.TestCase):
    """A generated library compiled again to the same path and loaded in the same process must be the NEW library: dlopen matches a name against
    the libraries already mapped before it looks at the file, so `ctypes.CDLL(path)` hands back the one the file replaced (the service that
    swaps its model file, a compare/choose loop that rebuilds into one output name)."""

    def lib_copy(self, folder: Path, extra: bytes) -> Path:
        # any shared object does: bytes after the ELF image are ignored by the loader, and they make this file another one
        dest = folder / "m.so"
        qf.replace_file(dest, data=Path(evaluator.find_lib()).read_bytes() + extra)
        return dest

    def test_a_replaced_file_loads_as_a_new_library(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d)
            first = cg.dlopen_current(str(self.lib_copy(folder, b"")))
            same = cg.dlopen_current(str(folder / "m.so"))
            self.assertEqual(first._handle, same._handle, "an unchanged file is the library already loaded")
            self.lib_copy(folder, bytes(8))
            second = cg.dlopen_current(str(folder / "m.so"))
            self.assertNotEqual(first._handle, second._handle, "the file was replaced: the new library must be mapped")
            again = cg.dlopen_current(str(folder / "m.so"))
            self.assertEqual(second._handle, again._handle)
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ["m.so"])

    def test_plain_dlopen_is_the_bug(self):
        """Documents why `dlopen_current` exists: the plain call returns the library the file replaced."""
        import ctypes

        with tempfile.TemporaryDirectory() as d:
            path = self.lib_copy(Path(d), b"")
            first = ctypes.CDLL(str(path))
            self.lib_copy(Path(d), bytes(16))
            self.assertEqual(first._handle, ctypes.CDLL(str(path))._handle)


class DuplicatedRulesTest(unittest.TestCase):
    """Rules written once in Python for the flat writer and once for the generated code (and once in Rust) must agree on every input."""

    def test_the_flat_writer_and_the_generated_code_write_the_same_segments(self):
        for kind, task, groups, segments in itertools.product(
            ("identity", "sigmoid", "softmax", "exp"), ("auto", "regression", "classification"), (2, 5), (None, "own")
        ):
            forest = make_vector_forest(1, width=groups, trees=1, kind=kind, segments=[0, 1, groups] if segments else None)
            forest.task = task
            flat = backend_flat.output_segments(forest)
            self.assertEqual(flat or [], cg.file_segments(forest), (kind, task, groups, segments))

    def test_max_groups_is_one_number_in_python_and_rust(self):
        rust = (HERE.parent / "rust" / "src" / "lib.rs").read_text()
        want = int(re.search(r"^const MAX_GROUPS: usize = (\d+);", rust, re.M).group(1))
        self.assertEqual(backend_flat.MAX_GROUPS, want)
        self.assertEqual(qf_sklearn.MAX_GROUPS, want)


def lib_reads_format_7() -> bool:
    try:
        lib = evaluator.find_lib()
        return hasattr(evaluator.load_lib(lib), "tfl_segments")
    except OSError:
        return False


@unittest.skipUnless(lib_reads_format_7(), "the library at hand does not read format 7: build it with rust/build.sh")
class EngineMatrixTest(unittest.TestCase):
    """Invariant 1 of the plan: a vector forest answers bit for bit as its exploded twin at every row count (the lane tail and the block edge),
    thread count, answer type and with or without a base margin; f64 tables, extended nodes, several folds, late margins included."""

    ROWS = (1, 3, 4, 5, 1023, 1025)

    def check(self, forest, exact=False):
        lib = evaluator.find_lib()

        def fnv(b):
            return evaluator.fnv1a(b, lib)

        with tempfile.TemporaryDirectory() as d:
            paths = []
            for name, f in (("v", forest), ("s", explode(forest))):
                paths.append(Path(d) / f"{name}.flat")
                paths[-1].write_bytes(backend_flat.pack(f, fnv, exact))
            vec, scalar = (evaluator.FlatForest(p, lib) for p in paths)
            try:
                for n in self.ROWS:
                    rng = np.random.default_rng(n)
                    x = rng.normal(0, 1.5, (n, forest.n_features)).astype(np.float32)
                    x[rng.random(x.shape) < 0.08] = np.nan
                    bm = rng.normal(0, 1, (n, forest.groups))
                    for threads, dtype, margin in itertools.product((1, 3, 8), (np.float32, np.float64), (None, bm)):
                        got = vec.predict(x, threads, base_margin=margin, dtype=dtype)
                        want = scalar.predict(x, threads, base_margin=margin, dtype=dtype)
                        self.assertTrue(np.array_equal(got, want), (n, threads, dtype.__name__, margin is not None))
            finally:
                vec.close()
                scalar.close()

    def test_float32_sums_with_folds(self):
        self.check(make_vector_forest(31, n_features=7, folds=3, trees=5))

    def test_double_sums_with_extended_nodes_and_f64_tables(self):
        self.check(make_vector_forest(32, n_features=7, ext=True, table64=True, depth=6), exact=True)

    def test_late_margin_with_a_softmax(self):
        f = make_vector_forest(33, n_features=7, kind="softmax", folds=2, average="margin")
        self.check(dataclasses.replace(f, late_margin=True, sum64=False))

    def test_a_wide_forest(self):
        self.check(make_vector_forest(34, n_features=5, width=300, depth=2, trees=2))


if __name__ == "__main__":
    unittest.main()
