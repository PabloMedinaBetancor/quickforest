"""Greedy selection on forests too big to hold (scripts/select_stream.py): the streamed path must select exactly the trees the in-memory path
selects, with memory that does not grow with the trees. Synthetic data only; needs numpy and xgboost.

    python3 -m unittest tests.test_scale        (from the repository root, with a Python that has xgboost)
"""

import json
import mmap
import os
import platform
import re
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import xgboost as xgb

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_pages
import prune
import qf_common as qf
import qf_formats
import select_stream as ss

N, NF = 1800, 8


def make_data():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(N, NF)).astype(np.float32)
    s = np.stack([x[:, 0] + x[:, 1] * x[:, 2], np.sin(2 * x[:, 3]), x[:, 4] ** 2 - x[:, 5]], 1) + rng.normal(scale=0.5, size=(N, 3))
    x[rng.random(x.shape) < 0.06] = np.nan  # missing values go the way each split says
    return x, s.argmax(1), (s[:, 0] > 0).astype(int), (3 * s[:, 0] + s[:, 1]).astype(np.float32)


def fit(x, y, rounds, **params):
    return xgb.train(
        {"tree_method": "hist", "max_depth": 5, "seed": 3, "nthread": 2, **params},
        xgb.DMatrix(x[: N // 2], y[: N // 2]),
        num_boost_round=rounds,
    )


class ScaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.t = Path(cls.tmp.name)
        x, y3, y2, reg = make_data()
        cls.x = x
        np.savez(t / "d.npz", X=x)
        rf = {"num_parallel_tree": 90, "subsample": 0.8, "colsample_bynode": 0.7, "learning_rate": 1}
        models = {
            "binrf": fit(x, y2, 1, objective="binary:logistic", **rf),
            "multirf": fit(x, y3, 1, objective="multi:softprob", num_class=3, **{**rf, "num_parallel_tree": 50}),
            "bingb": fit(x, y2, 40, objective="binary:logistic", learning_rate=0.3),
            "multigb": fit(x, y3, 12, objective="multi:softprob", num_class=3, learning_rate=0.3),
            "reg": fit(x, reg, 40, objective="reg:squarederror", learning_rate=0.3),
        }
        cls.paths = {}
        for name, b in models.items():
            for ext in ("ubj", "json"):
                b.save_model(str(t / f"{name}.{ext}"))
                cls.paths[f"{name}.{ext}"] = t / f"{name}.{ext}"
        # the same JSON re-saved with indentation: other tools write models that way
        for name in ("binrf", "multirf"):
            doc = json.loads((t / f"{name}.json").read_text())
            (t / f"{name}-pretty.json").write_text(json.dumps(doc, indent=2))
            cls.paths[f"{name}-pretty.json"] = t / f"{name}-pretty.json"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def stream(self, name, **kw):
        return ss.StreamSource(self.paths[name], scratch=Path(tempfile.mkdtemp(dir=self.t)), **kw)

    # ------------------------------------------------------------------------------------------------------------- the reader
    def test_the_scan_reads_what_parse_model_reads(self):
        for name, path in self.paths.items():
            with self.subTest(name):
                classic, stream = qf_formats.open_source(path), self.stream(name)
                a, b = classic.spec, stream.spec
                for f in ("n_features", "groups", "kind", "parallel", "rf", "task"):
                    self.assertEqual(getattr(a, f), getattr(b, f), f)
                for f in ("base", "tree_info", "nodes"):
                    np.testing.assert_array_equal(getattr(a, f), getattr(b, f), f)
                self.assertEqual(stream.size_bytes, classic.size_bytes)
                for i in (0, len(a.nodes) // 2, len(a.nodes) - 1):  # a tree read again from its position is the tree the library holds
                    got = stream.mf.tree_at(int(stream.tree_pos[i]))
                    for key in ("left_children", "right_children", "split_indices", "split_conditions", "default_left"):
                        dt = (
                            np.float32 if key == "split_conditions" else None
                        )  # a .ubj tree holds float32 values, the library's JSON their shortest text
                        np.testing.assert_array_equal(np.asarray(got[key], dt), np.asarray(classic._trees[i][key], dt), f"tree {i} {key}")
                stream.close()

    # ---------------------------------------------------------------------------------------------------- the leaf values
    def test_the_leaf_codes_hold_the_leaf_matrix_exactly(self):
        for name in ("binrf.ubj", "multirf.json", "bingb.ubj", "multigb.json", "reg.ubj", "binrf-pretty.json"):
            classic = qf_formats.open_source(self.paths[name])
            xs, xe = self.x[:700], self.x[700:900]
            full = classic.leaf_matrix(xs)
            ref = np.stack(
                [classic.spec.base[g] + full[classic.spec.tree_info == g].sum(0, dtype=np.float64) for g in range(classic.spec.groups)], 1
            )
            for cells, budget in (
                (1, None),
                (37, 0.0001),
                (500, None),
                (ss.CHUNK_CELLS, 0.0001),
            ):  # chunk sizes; codes in memory and in a file
                with self.subTest(name=name, chunk_cells=cells, budget=budget):
                    stream = self.stream(name, chunk_cells=cells)
                    lm = stream.leaf_matrix(xs, extra=xe, code_budget_mb=budget)
                    self.assertEqual(lm.on_disk, budget is not None)
                    self.assertEqual(lm.shape, full.shape)
                    np.testing.assert_array_equal(lm[np.arange(full.shape[0])], full)
                    np.testing.assert_array_equal(lm[np.array([5, 2, 9]) % full.shape[0]], full[np.array([5, 2, 9]) % full.shape[0]])
                    np.testing.assert_array_equal(lm[3], full[3])
                    np.testing.assert_array_equal(
                        prune.group_margin(lm, classic.spec), ref
                    )  # bitwise: the same additions in the same order
                    held = classic.predict(xe)
                    mine = qf.transform(classic.spec.base[None, :] + lm.extra_margin, classic.spec.kind)
                    np.testing.assert_allclose(mine, held, rtol=2e-5, atol=2e-6)
                    lm.close()
                    stream.close()

    def test_a_missing_or_extreme_value_goes_the_way_xgboost_sends_it(self):
        classic, stream = qf_formats.open_source(self.paths["binrf.ubj"]), self.stream("binrf.ubj")
        rows = self.x[:60].copy()
        rows[:20, :] = np.nan
        rows[20:30, 0] = 3e38  # xgboost refuses a DMatrix with inf in it; the largest finite values still sort the same way
        rows[30:40, 1] = -3e38
        rows[40:50] = 0.0
        np.testing.assert_array_equal(stream.leaf_matrix(rows)[np.arange(len(classic.spec.nodes))], classic.leaf_matrix(rows))
        stream.close()

    # -------------------------------------------------------------------------------------------------------- the selection
    def test_the_selection_does_not_depend_on_how_many_candidates_are_scored_at_once(self):
        for name in ("binrf.ubj", "multirf.ubj", "bingb.json"):
            classic = qf_formats.open_source(self.paths[name])
            spec, rows = classic.spec, self.x[:600]
            lm = classic.leaf_matrix(rows)
            margin = prune.group_margin(lm, spec)
            p_full = qf.transform(margin, spec.kind)
            rule = qf.make_rule("argmax", spec.groups, spec.kind, p_full, None, spec.task)
            per = int(np.bincount(spec.tree_info).min())
            ref = None
            for block in (1, 2, 7, 64, None):
                with (
                    self.subTest(name=name, block=block),
                    mock.patch.object(prune, "candidate_block", (lambda n, g, b=block: b) if block else prune.candidate_block),
                ):
                    order = prune.greedy_order(lm, spec, min(6, per), rule, spec.rf, p_full, rule(p_full))
                    ref = ref or order
                    for g in range(spec.groups):
                        self.assertEqual(list(order[g]), list(ref[g]))

    def test_the_streamed_selection_is_the_in_memory_selection(self):
        for name in ("binrf.ubj", "multirf.json", "bingb.ubj", "multigb.json", "reg.json"):
            classic = qf_formats.open_source(self.paths[name])
            spec, rows = classic.spec, self.x[:600]
            lm = classic.leaf_matrix(rows)
            stream = self.stream(name, chunk_cells=300)
            codes = stream.leaf_matrix(rows, code_budget_mb=0.0001)
            for scale in ("rf scaling", "none"):
                if scale == "rf scaling" and not spec.rf:
                    continue
                mean_scale = scale == "rf scaling"
                p_full = qf.transform(prune.group_margin(lm, spec), spec.kind)
                rule = qf.make_rule("argmax", spec.groups, spec.kind, p_full, None, spec.task)
                per = int(np.bincount(spec.tree_info).min())
                for k in (1, 4, min(9, per)):
                    with self.subTest(name=name, scale=scale, k=k):
                        want = prune.greedy_order(lm, spec, k, rule, mean_scale, p_full, rule(p_full))
                        got = prune.greedy_order(codes, spec, k, rule, mean_scale, p_full, rule(p_full))
                        self.assertEqual([list(g) for g in got], [list(g) for g in want])
            orient = prune.orientation_order(lm, spec, 5, prune.group_margin(lm, spec))
            self.assertEqual(
                [sorted(g) for g in prune.orientation_order(codes, spec, 5, prune.group_margin(codes, spec))], [sorted(g) for g in orient]
            )
            codes.close()
            stream.close()

    def test_the_walk_and_the_selection_do_not_depend_on_the_number_of_threads(self):
        for name in ("binrf.ubj", "multirf.json"):
            classic = qf_formats.open_source(self.paths[name])
            spec, rows = classic.spec, self.x[:500]
            full = classic.leaf_matrix(rows)
            p_full = qf.transform(prune.group_margin(full, spec), spec.kind)
            rule = qf.make_rule("argmax", spec.groups, spec.kind, p_full, None, spec.task)
            want = prune.greedy_order(full, spec, 5, rule, spec.rf, p_full, rule(p_full))
            for threads in (1, 2, 5):
                with self.subTest(name=name, threads=threads):
                    stream = self.stream(name, chunk_cells=700, threads=threads)
                    codes = stream.leaf_matrix(rows, extra=self.x[500:600], code_budget_mb=0.0001)
                    np.testing.assert_array_equal(codes[np.arange(full.shape[0])], full)
                    got = prune.greedy_order(codes, spec, 5, rule, spec.rf, p_full, rule(p_full), threads=threads)
                    self.assertEqual([list(g) for g in got], [list(g) for g in want])
                    self.assertEqual(
                        [list(g) for g in prune.greedy_order(full, spec, 5, rule, spec.rf, p_full, rule(p_full), threads=threads)],
                        [list(g) for g in want],
                    )
                    codes.close()
                    stream.close()

    # ------------------------------------------------------------------------------------------------------------ end to end
    def test_prune_run_streams_a_model_over_the_budget_and_keeps_the_same_trees(self):
        for name, keep in (("binrf.ubj", "10%"), ("multirf.json", "6"), ("bingb.ubj", "25%"), ("multirf.ubj", "auto")):
            with self.subTest(name=name, keep=keep):
                a = prune.run(self.paths[name], self.t / "d.npz", self.t / f"c-{name}-{keep}", keep)
                b = prune.run(self.paths[name], self.t / "d.npz", self.t / f"s-{name}-{keep}", keep, memory_mb=1)
                self.assertNotIn("scale", a["models"][0])
                self.assertEqual(b["models"][0]["scale"]["mode"], "stream")
                self.assertIn("read in a stream", prune.summary(b))
                self.assertNotIn("read in a stream", prune.summary(a))
                self.assertEqual(a["kept_indices"], b["kept_indices"])
                for f in ("trees", "kept_trees", "nodes_before", "nodes_after", "bytes_before", "bytes_after"):
                    self.assertEqual(a["models"][0][f], b["models"][0][f], f)
                self.assertEqual(a["heldout"]["rows"], b["heldout"]["rows"])
                self.assertLess(abs(a["heldout"]["pct"] - b["heldout"]["pct"]), 0.5)  # the full forest's answers come from the walker
                da, db = qf.load_booster(self.t / f"c-{name}-{keep}" / name), qf.load_booster(self.t / f"s-{name}-{keep}" / name)
                d = xgb.DMatrix(self.x[:400])
                np.testing.assert_array_equal(da.predict(d), db.predict(d))  # the same kept trees, scaled the same
                left = sorted(q.name for q in (self.t / f"s-{name}-{keep}").iterdir())
                self.assertEqual(left, sorted(["report.json", name]), "the working files are gone")

    def test_the_selection_reads_only_the_selection_rows(self):
        """The held-out figure the report prints is honest only because the greedy pick reads the selection rows alone; a pick that also
        saw the held-out rows (or the unused ones) would score 1,800 rows here instead of the selection slice's."""
        r = prune.run(self.paths["binrf.ubj"], self.t / "d.npz", self.t / "sel-only", "10%")
        sl = qf.slices({"X": self.x}, 0.5, 15)  # the split run() draws by default
        self.assertEqual(r["models"][0]["selection_rows"], int((sl == 0).sum()))
        self.assertEqual(r["heldout"]["rows"], int((sl == 1).sum()))
        self.assertLess(r["models"][0]["selection_rows"], len(self.x))

    def test_outputs_that_overflow_float32_are_refused_in_one_line_naming_the_cause(self):
        """A regression target of 1e30 made every candidate score NaN, `best` stayed None and the run ended in numpy's "non-broadcastable
        output operand with shape ..."; both greedy selections now say what is wrong."""
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 4)).astype(np.float32)
        y = (x[:, 0] + 0.1 * rng.normal(size=300)) * 1e30
        model = fit(x, y, 10, objective="reg:squarederror", max_depth=3)
        model.save_model(str(self.t / "huge.json"))
        np.savez(self.t / "huge.npz", X=x)
        with self.assertRaisesRegex(ValueError, "not finite or overflow float32"):
            prune.run(self.t / "huge.json", self.t / "huge.npz", self.t / "huge-out", "50%")
        n, g_n = 10, 2  # the same refusal in the selection that keeps a YDF classification tree whole (one forest tree per class)
        spec = mock.Mock(groups=g_n, base=np.zeros(g_n, np.float32), tree_info=np.array([0, 1] * 3))
        leaves = np.random.default_rng(1).normal(size=(6, n)).astype(np.float32)
        p_full = np.full((n, g_n), np.nan, np.float32)
        with self.assertRaisesRegex(ValueError, "not finite or overflow float32"):
            prune.greedy_units_order(leaves, spec, np.array([0, 0, 1, 1, 2, 2]), 2, lambda p: p.argmax(-1), p_full, np.zeros(n, np.int64))

    def test_a_streamed_run_answers_only_the_rows_it_was_given_budget_for(self):
        r = prune.run(self.paths["binrf.ubj"], self.t / "d.npz", self.t / "cap", "10%", memory_mb=1, max_held_rows=300, max_sel_rows=400)
        self.assertEqual(r["heldout"]["rows"], 300)
        self.assertEqual(r["models"][0]["selection_rows"], 400)
        self.assertEqual(r["models"][0]["scale"]["heldout_rows_used"], 300)
        c = prune.run(self.paths["binrf.ubj"], self.t / "d.npz", self.t / "nocap", "10%", max_sel_rows=400)
        held = int((qf.slices({"X": self.x}, 0.5, 15) == 1).sum())
        self.assertGreater(held, 400)
        self.assertEqual(c["heldout"]["rows"], held)  # the in-memory path still uses every held-out row

    def _folder_of_three(self, name):
        """Three multi-class forests (different seeds) in one folder, to be averaged as one ensemble (no `fold` array)."""
        _, y3, _, _ = make_data()
        folder = self.t / name
        folder.mkdir()
        rf = {
            "num_parallel_tree": 30,
            "subsample": 0.7,
            "colsample_bynode": 0.6,
            "learning_rate": 1,
            "objective": "multi:softprob",
            "num_class": 3,
        }
        for k in range(3):
            fit(self.x, y3, 1, **{**rf, "seed": k}).save_model(str(folder / f"fold{k}.ubj"))
        return folder

    def _ensemble_agreement_on(self, folder, pruned, rows):
        """Recomputed with xgboost itself: the mean of the full models and the mean of the pruned ones, on exactly `rows`."""
        d = xgb.DMatrix(self.x[rows])

        def mean(src):
            outs = []
            for f in sorted(Path(src).glob("fold*.ubj")):
                b = xgb.Booster()
                b.load_model(str(f))
                outs.append(b.predict(d))
            return np.mean(outs, axis=0)

        full, small = mean(folder), mean(pruned)
        return full, int((full.argmax(1) == small.argmax(1)).sum())

    def test_every_model_of_a_folder_is_judged_on_the_same_held_out_rows_when_the_run_is_streamed(self):
        # Each streamed model drew its OWN subset of the held-out rows and the ensemble averaged unrelated rows
        folder = self._folder_of_three("same-rows-stream")
        out = self.t / "same-rows-stream-out"
        r = prune.run(folder, self.t / "d.npz", out, "30%", memory_mb=1, max_held_rows=100, save_probs=True)
        self.assertTrue(all("scale" in m for m in r["models"]))
        held = int((qf.slices({"X": self.x}, 0.5, 15) == 1).sum())
        self.assertGreater(held, 100)
        z = np.load(out / "heldout_probs.npz")
        self.assertEqual(len(z["rows"]), 100)
        true_full, true_same = self._ensemble_agreement_on(folder, out, z["rows"])
        self.assertLess(float(np.abs(z["p_full"] - true_full).max()), 1e-5)
        self.assertEqual(r["heldout"]["same"], true_same)
        self.assertEqual({tuple(m["scale"]["heldout_rows_used"] for m in r["models"])}, {(100, 100, 100)})

    def test_every_model_of_a_folder_is_judged_on_the_same_held_out_rows_in_memory(self):
        # the in-memory twin: no cap, every model answers every held-out row, and the figure is the ensemble's on the rows it names
        folder = self._folder_of_three("same-rows-memory")
        out = self.t / "same-rows-memory-out"
        r = prune.run(folder, self.t / "d.npz", out, "30%", save_probs=True)
        self.assertFalse(any("scale" in m for m in r["models"]))
        z = np.load(out / "heldout_probs.npz")
        true_full, true_same = self._ensemble_agreement_on(folder, out, z["rows"])
        self.assertLess(float(np.abs(z["p_full"] - true_full).max()), 1e-5)
        self.assertEqual(r["heldout"]["same"], true_same)

    def test_a_folder_with_one_streamed_and_one_in_memory_model_shares_the_held_out_rows(self):
        # the cap is a property of the run, not of the model: the models that fit in memory are cut to the same rows
        folder = self._folder_of_three("same-rows-mixed")
        out = self.t / "same-rows-mixed-out"
        real = ss.wanted
        with mock.patch.object(ss, "wanted", side_effect=lambda p, n, m: Path(p).name == "fold1.ubj" or real(p, n, m)):
            r = prune.run(folder, self.t / "d.npz", out, "30%", max_held_rows=100, save_probs=True)
        z = np.load(out / "heldout_probs.npz")
        self.assertEqual(
            ["scale" in m for m in r["models"]], [False, True, False]
        )  # the streamed model is the middle one: the mixing happened
        self.assertEqual(len(z["rows"]), 100)
        true_full, true_same = self._ensemble_agreement_on(folder, out, z["rows"])
        self.assertLess(float(np.abs(z["p_full"] - true_full).max()), 1e-5)
        self.assertEqual(r["heldout"]["same"], true_same)

    def test_the_codes_go_to_a_file_when_they_do_not_fit_the_budget(self):
        stream = self.stream("multirf.ubj", memory_mb=1)
        lm = stream.leaf_matrix(self.x[:900], code_budget_mb=0.01)
        self.assertTrue(lm.on_disk and lm.codes.dtype == np.uint8)
        self.assertEqual(lm.code_bytes, 150 * 900)  # one byte per tree and row
        lm.close()
        again = stream.leaf_matrix(self.x[:900], code_budget_mb=64)
        self.assertFalse(again.on_disk)
        again.close()
        stream.close()

    def test_the_memory_of_a_streamed_walk_does_not_grow_with_the_trees(self):
        """1,500 trees x 4,000 rows: the float32 leaf matrix is 24 MB; the walk with its codes in a file stays far below that."""
        rng = np.random.default_rng(5)
        x = rng.normal(size=(4000, NF)).astype(np.float32)
        y = (x[:, 0] * x[:, 1] + x[:, 2] > 0).astype(int)
        b = xgb.train(
            {
                "tree_method": "hist",
                "max_depth": 5,
                "objective": "binary:logistic",
                "num_parallel_tree": 1500,
                "subsample": 0.6,
                "colsample_bynode": 0.6,
                "learning_rate": 1,
                "nthread": 1,
            },
            xgb.DMatrix(x[:1500], y[:1500]),
            num_boost_round=1,
        )
        path = self.t / "big.ubj"
        b.save_model(str(path))
        matrix_bytes = 1500 * 4000 * 4
        stream = ss.StreamSource(path, scratch=Path(tempfile.mkdtemp(dir=self.t)), chunk_cells=1 << 14)
        tracemalloc.start()
        lm = stream.leaf_matrix(x, code_budget_mb=0.001)
        walk_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.reset_peak()
        block = prune.candidate_block(4000, 1)  # what the greedy loop reads: blocks of candidate rows
        for a in range(0, 1500, block):
            lm[np.arange(a, min(a + block, 1500))]
        read_peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        self.assertTrue(lm.on_disk)
        self.assertLess(walk_peak, matrix_bytes / 4, f"the walk used {walk_peak:,} bytes, the matrix it replaces is {matrix_bytes:,}")
        self.assertLess(read_peak, matrix_bytes / 2, f"a block read used {read_peak:,} bytes")
        self.assertEqual(lm.code_bytes, matrix_bytes // 4)  # one byte where the matrix has four
        lm.close()
        stream.close()

    @staticmethod
    def resident_mb(path):
        """The resident megabytes of the mappings of `path`, from /proc/self/smaps, which walks the page tables. (The RssFile counter of
        /proc/self/status is exact on recent kernels only: older ones add a page fault to it in batches of up to 64.)"""
        inode, total, inside = str(os.stat(path).st_ino), 0, False  # by inode: a sandbox that rewrites paths shows another name
        with open("/proc/self/smaps") as fh:
            for line in fh:
                if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                    inside = line.split()[4] == inode
                elif inside and line.startswith("Rss:"):
                    total += int(line.split()[1])
        return total / 1024

    def mapping_unit_mb(self, path):
        """What ONE page fault maps on this machine, in a mapping set up as the product sets it up (`ss.map_file`), of THIS model file: the
        resident bytes after touching one byte, the largest of a few offsets. The file itself is measured because how the kernel holds it in
        its page cache depends on how it was written: a file written in one big write is held in 2 MB folios on recent kernels, and one fault in
        such a folio maps all of it, while a file written in 1 MB pieces (the first version of this probe) maps 64 KB. The product cannot let go
        of pages a fault has mapped ahead of the cursor."""
        unit, size = 0.0, path.stat().st_size
        with open(path, "rb") as fh:
            mm = ss.map_file(fh)
            try:
                for frac in (0.2, 0.35, 0.5, 0.65, 0.8):  # far from the ends and from each other, in different 2 MB stretches of the file
                    before = self.resident_mb(path)
                    mm[int(size * frac) + 12345]
                    unit = max(unit, self.resident_mb(path) - before)
                    mm.madvise(mmap.MADV_DONTNEED)
            finally:
                mm.close()
        return unit

    def test_the_pages_of_the_model_file_are_let_go_of_as_they_are_read(self):
        """The file is mapped; the pages already read must leave the resident set (a 949 MB model once cost 957 MB of RSS). What may stay is bounded
        by the release window plus what a page fault maps ahead of the cursor, whatever the size of the file, and the test says so: the bar is the
        window plus two mapping units of this machine (measured on the model file) plus a slack, the file is made at least four times the bar so
        that a file that is NOT let go of fails it, and a run with the release switched off must fail the same bar (the mutation is part of the
        test). A failure prints which stretches of the file were resident at the peak (tests/diag_pages.py reports the same on its own)."""
        try:
            self.resident_mb(__file__)
        except OSError:  # not Linux, or a sandbox without /proc/self/smaps
            self.skipTest("needs /proc/self/smaps")
        window_mb, slack_mb = (1 << 18) / 2**20, 0.25
        x = np.random.default_rng(2).normal(size=(3000, NF)).astype(np.float32)
        dmat = xgb.DMatrix(x, (x[:, 0] * x[:, 1] > 0).astype(int))
        path = self.t / "pages.ubj"

        def save(trees):
            xgb.train(
                {
                    "tree_method": "hist",
                    "max_depth": 7,
                    "objective": "binary:logistic",
                    "num_parallel_tree": trees,
                    "subsample": 0.9,
                    "learning_rate": 1,
                    "nthread": 2,
                },
                dmat,
                num_boost_round=1,
            ).save_model(str(path))
            return path.stat().st_size / 2**20

        trees = 1500
        for _ in range(3):  # a machine that maps more at a time needs a bigger file to tell "let go of" from "kept"
            size_mb = save(trees)
            unit = self.mapping_unit_mb(path)
            bar = window_mb + 2 * unit + slack_mb
            if size_mb >= 4 * bar:
                break
            trees = int(np.ceil(trees * 4.4 * bar / size_mb))
        self.assertGreaterEqual(
            size_mb, 4 * bar, f"a model of {trees} trees is {size_mb:.1f} MB; the bar of {bar:.2f} MB needs a file four times as big"
        )

        def peak_resident_mb():
            """(the peak, where the resident stretches of the file were at that moment)"""
            with mock.patch.object(ss, "RELEASE_BYTES", 1 << 18):
                before = self.resident_mb(path)
                mf = ss.ModelFile(path)  # finding where the list of trees ends reads all of it
                peak, where = self.resident_mb(path) - before, diag_pages.describe_residency(path)
                for i, _ in enumerate(mf.trees(ss.WANT)):
                    if i % 25 == 0:
                        now = self.resident_mb(path) - before
                        if now > peak:
                            peak, where = now, diag_pages.describe_residency(path)
                mf.close()
            return peak, where

        detail = (
            f"(mapping unit {unit:.2f} MB, window {window_mb:.2f} MB, bar {bar:.2f} MB, file {size_mb:.1f} MB, kernel {platform.release()})"
        )
        peak, where = peak_resident_mb()
        self.assertLess(peak, bar, f"{peak:.1f} MB of the {size_mb:.1f} MB file stayed resident {detail}; at the peak {where}")
        with mock.patch.object(ss.ModelFile, "_release", lambda self, upto: None):  # the release switched off: the same bar must catch it
            kept, _ = peak_resident_mb()
        self.assertGreater(kept, bar, f"with the release switched off {kept:.1f} MB stayed resident, under the bar {detail}")

    @staticmethod
    def vm_flags(path):
        """The VmFlags of the mapping of `path` from /proc/self/smaps (None when it is not mapped)."""
        inode, flags, inside = str(os.stat(path).st_ino), None, False
        with open("/proc/self/smaps") as fh:
            for line in fh:
                if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                    inside = line.split()[4] == inode
                elif inside and line.startswith("VmFlags:"):
                    flags = line.split()[1:]
        return flags

    def test_the_model_file_is_mapped_without_huge_pages(self):
        """A kernel that holds a file in 2 MB folios of its page cache maps all of one on a fault, and again after the pages were let go of: the
        mapping asks for small ones (`nh` among the flags of the mapping), a plain mapping of the same file does not."""
        if not hasattr(mmap, "MADV_NOHUGEPAGE") or not os.path.isdir("/sys/kernel/mm/transparent_hugepage"):
            self.skipTest("this kernel has no huge-page support to switch off")
        path = self.paths["binrf.ubj"]
        with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as plain:
            plain[0]
            self.assertNotIn("nh", self.vm_flags(path) or [])
        mf = ss.ModelFile(path)
        try:
            self.assertIn("nh", self.vm_flags(path) or [])
        finally:
            mf.close()

    # --------------------------------------------------------------------------------------------------------------- refusals
    def test_a_cut_model_is_refused_with_one_line(self):
        for name in ("binrf.ubj", "binrf.json"):
            data = self.paths[name].read_bytes()
            for cut in (len(data) // 2, len(data) - 40, 40):
                bad = self.t / f"cut{cut}-{name}"
                bad.write_bytes(data[:cut])
                with self.subTest(name=name, cut=cut), self.assertRaises(ValueError) as e:
                    ss.StreamSource(bad, scratch=Path(tempfile.mkdtemp(dir=self.t))).close()
                self.assertNotIn("Traceback", str(e.exception))
                self.assertLess(len(str(e.exception)), 300)
        empty = self.t / "empty.json"
        empty.write_bytes(b"")
        with self.assertRaises(ValueError):
            ss.StreamSource(empty, scratch=Path(tempfile.mkdtemp(dir=self.t)))

    def test_a_dart_model_is_refused_as_the_in_memory_path_refuses_it(self):
        x, _, y2, _ = make_data()
        dart = fit(x, y2, 5, objective="binary:logistic", booster="dart", rate_drop=0.3, learning_rate=0.5)
        dart.save_model(str(self.t / "dart.ubj"))  # xgboost 3.2 names the booster "dart"; 3.4 keeps "gbtree" and adds `weight_drop`
        doc = json.loads(self.paths["binrf.json"].read_text())
        doc["learner"]["gradient_booster"]["model"]["weight_drop"] = [1.0] * 90
        (self.t / "dart34.json").write_text(json.dumps(doc))
        for name in ("dart.ubj", "dart34.json"):
            with self.subTest(name):
                with self.assertRaises(ValueError) as classic:
                    qf_formats.open_source(self.t / name)
                with self.assertRaises(ValueError) as stream:
                    ss.StreamSource(self.t / name, scratch=Path(tempfile.mkdtemp(dir=self.t)))
                self.assertIn("dart", str(classic.exception).lower())
                self.assertIn("dart", str(stream.exception).lower())

    def test_options_are_checked(self):
        for kw in ({"memory_mb": 0}, {"memory_mb": -5}, {"max_held_rows": 0}):
            with self.subTest(kw), self.assertRaises(SystemExit):
                prune.run(self.paths["binrf.ubj"], self.t / "d.npz", self.t / "bad", "10%", **kw)
            self.assertFalse((self.t / "bad").exists())

    def test_wanted_follows_the_budget(self):
        p = self.paths["multirf.ubj"]
        self.assertTrue(ss.wanted(p, 5000, 1))
        self.assertFalse(ss.wanted(p, 5000, 100000))
        self.assertFalse(ss.wanted(self.t / "missing.ubj", 5000, 1))
        self.assertEqual(ss._num_trees(p), 150)
        self.assertEqual(ss._num_trees(self.paths["multirf.json"]), 150)

    def test_the_command_line_takes_the_budget(self):
        out = self.t / "cli"
        r = subprocess.run(
            [
                sys.executable,
                "-I",
                str(SCRIPTS / "quickforest"),
                "prune",
                str(self.paths["binrf.ubj"]),
                str(self.t / "d.npz"),
                str(out),
                "--keep",
                "10%",
                "--memory-mb",
                "1",
                "--max-held-rows",
                "400",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("pruned 1 model(s)", r.stdout)
        self.assertEqual(json.loads((out / "report.json").read_text())["models"][0]["scale"]["heldout_rows_used"], 400)
        bad = subprocess.run(
            [
                sys.executable,
                "-I",
                str(SCRIPTS / "quickforest"),
                "prune",
                str(self.paths["binrf.ubj"]),
                str(self.t / "d.npz"),
                str(self.t / "cli2"),
                "--memory-mb",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertNotEqual(bad.returncode, 0)
        self.assertEqual(len(bad.stderr.strip().splitlines()), 1, bad.stderr)


if __name__ == "__main__":
    unittest.main()
