"""Pruning per output, decision segments, check segments and the API pieces of QuickForest 1.2 (part P4).

Everything here uses a FAKE source built from the IR (`tests/vector_forests.py`) and synthetic data, so it needs nothing but numpy (and
scikit-learn for the two estimator tests) and runs in a few seconds on one thread:

python3 -s -m pytest tests/test_prune_multi.py
"""

import itertools
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))
import check
import evaluator
import prune
import qf_common as qf
import qf_formats
import select_stream
from vector_forests import make_vector_forest

REGISTRY: dict[str, "FakeVecSource"] = {}  # placeholder file path -> the source it stands for


class FakeVecSource:
    """The source interface `prune.run` uses, over an IR forest of vector trees: the model the readers (P2) hand to pruning."""

    unit_of = None
    notes = None
    linear = False
    size_bytes = 1000

    def __init__(self, forest, *, task, mean, fmt="fakevec", segments=None):
        self.forest, self.fmt, self.segments = forest, fmt, segments
        trees = forest.models[0].trees
        w = forest.groups
        self.spec = qf.Spec(
            forest.n_features,
            w,
            forest.kind,
            forest.models[0].base,
            np.zeros(len(trees), np.int64),
            np.array([len(t.left) for t in trees]),
            np.zeros(len(trees), np.int64),
            np.zeros(0, np.float32),
            1,
            task=task,
            mean_type=mean,
            vector_width=w,
        )
        self.mean = mean

    def _margin(self, x):
        m = np.repeat(self.forest.models[0].base[None], len(x), 0).astype(np.float32)
        for t in self.forest.models[0].trees:
            m += qf.tree_values(t, x)
        return m

    def leaf_matrix(self, x):
        raise ValueError("a vector forest hands out leaf_codes / leaf_vectors")  # as the P2 sources do

    def leaf_codes(self, x):
        x = np.asarray(x, np.float32)
        out = []
        for t in self.forest.models[0].trees:
            leaves = np.flatnonzero(t.left == -1)
            out.append(np.searchsorted(leaves, qf.leaf_positions(t, x)).astype(np.int32))
        return np.stack(out)

    def leaf_vectors(self):
        return [t.vector[t.left == -1] for t in self.forest.models[0].trees]

    def predict(self, x, margin=False, base_margin=None):
        m = self._margin(np.asarray(x, np.float32))
        return m if margin else qf.transform(m, self.forest.kind)

    def save_pruned(self, kept, mean_scale, dest):
        ids = [int(i) for i in kept[0]]
        assert all(list(map(int, g)) == ids for g in kept), "a vector tree is one unit: every group keeps the same trees"
        factor = len(self.forest.models[0].trees) / len(ids) if mean_scale else 1.0
        trees = []
        for i in sorted(ids):
            t = self.forest.models[0].trees[i]
            vec = (t.vector * np.float32(factor)).astype(np.float32)
            trees.append(_with_vector(t, vec))
        f = make_like(self.forest, trees)
        REGISTRY[str(dest)] = FakeVecSource(f, task=self.spec.task, mean=self.mean, fmt=self.fmt, segments=self.segments)
        Path(dest).write_text("fake")


def _with_vector(t, vec):
    import dataclasses

    return dataclasses.replace(t, vector=vec)


def make_like(forest, trees):
    import dataclasses

    return dataclasses.replace(forest, models=[qf.ModelIR(forest.models[0].base, trees)])


class FakeFormats:
    """Context manager: `qf_formats` knows the placeholder files of REGISTRY."""

    def __enter__(self):
        self.patches = [
            mock.patch.object(qf_formats, "open_source", lambda p: REGISTRY[str(p)]),
            mock.patch.object(qf_formats, "model_files", lambda p: [Path(p)]),
            mock.patch.object(qf_formats, "is_model_file", lambda p: str(p) in REGISTRY),
            mock.patch.object(select_stream, "wanted", lambda *a, **k: False),
        ]
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *a):
        for p in self.patches:
            p.stop()


def regression_source(tmp: Path, trees=40, width=5, seed=1, scales=None, name="model.fakevec"):
    scales = np.array(scales if scales is not None else [1, 1e3, 1e-3, 1, 1], np.float32)[:width]
    f = make_vector_forest(seed, width=width, trees=trees, depth=3, n_features=4)
    ms = f.models[0]
    for t in ms.trees:
        t.vector = (t.vector * 0.01 * scales[None]).astype(np.float32)
    ms.base = (5 * scales).astype(np.float32)
    f.kind, f.task = "identity", "regression"
    path = tmp / name
    path.write_text("fake")
    REGISTRY[str(path)] = FakeVecSource(f, task="regression", mean=False)
    return path


def classification_source(tmp: Path, segments, trees=40, seed=2, name="clf.fakevec", fmt="fakevec", mean=True, kind="identity"):
    w = segments[-1]
    f = make_vector_forest(seed, width=w, trees=trees, depth=3, n_features=4, kind=kind)
    ms = f.models[0]
    rng = np.random.default_rng(seed)
    for t in ms.trees:
        v = rng.random(t.vector.shape).astype(np.float32) + 0.05
        for a, b in itertools.pairwise(segments):  # class fractions of each output sum to 1 at a leaf
            v[:, a:b] /= v[:, a:b].sum(1, keepdims=True)
        v[t.left != -1] = 0
        t.vector = (v / trees).astype(np.float32)
    ms.base = np.zeros(w, np.float32)
    f.task = "classification"
    f.segments = list(segments) if len(segments) > 2 else None
    path = tmp / name
    path.write_text("fake")
    REGISTRY[str(path)] = FakeVecSource(f, task="classification", mean=mean, fmt=fmt, segments=f.segments)
    return path


def data_file(tmp: Path, n=600, features=4, seed=3, **extra):
    x = np.random.default_rng(seed).random((n, features)).astype(np.float32)
    path = tmp / "data.npz"
    np.savez(path, X=x, **extra)
    return path


class RulesTest(unittest.TestCase):
    def test_segments_helper(self):
        self.assertIsNone(qf.decision_segments(5, "identity", 5, None))
        self.assertIsNone(qf.decision_segments(3, "softmax", 0, [0, 3]))
        self.assertEqual(qf.decision_segments(5, "identity", 5, [0, 3, 5]), [0, 3, 5])
        self.assertEqual(qf.decision_segments(2, "sigmoid", 2, None), [0, 1, 2])  # xgboost multilabel: a width-1 segment per column
        self.assertIsNone(qf.decision_segments(2, "sigmoid", 0, None))  # a scalar forest keeps 1.1's rule

    def test_argmax_per_segment_and_width_one(self):
        p = np.array([[0.1, 0.7, 0.2, 0.9, 0.1], [0.5, 0.2, 0.3, 0.2, 0.8]], np.float32)
        rule = qf.make_rule("argmax", 5, "identity", None, None, "classification", [0, 3, 5])
        self.assertEqual(rule(p).tolist(), [[1, 0], [0, 1]])
        self.assertEqual(rule(p[None]).shape, (1, 2, 2))
        sig = qf.make_rule("argmax", 2, "sigmoid", None, None, "classification", [0, 1, 2])
        self.assertEqual(sig(np.array([[0.5, 0.49], [0.2, 0.9]], np.float32)).tolist(), [[1, 0], [0, 1]])
        # argmax over G would be wrong: the second output's winner is index 3 or 4, never read against the first output's columns
        self.assertEqual(rule(np.array([[0.0, 0.0, 0.1, 0.0, 0.9]], np.float32)).tolist(), [[2, 1]])

    def test_threshold_refused_with_several_outputs(self):
        with self.assertRaisesRegex(ValueError, "single output"):
            qf.make_rule("threshold:0.5", 2, "sigmoid", None, None, "classification", [0, 1, 2])

    def test_rule_file_must_return_k_integers(self):
        with tempfile.TemporaryDirectory() as d:
            ok, bad, nan = Path(d) / "ok.py", Path(d) / "bad.py", Path(d) / "nan.py"
            ok.write_text("import numpy as np\ndef decide(p):\n    return np.stack([p[..., :3].argmax(-1), p[..., 3:].argmax(-1)], -1)\n")
            bad.write_text("def decide(p):\n    return p.argmax(-1)\n")
            nan.write_text("import numpy as np\ndef decide(p):\n    return np.full(p.shape[:-1] + (2,), np.nan)\n")
            rule = qf.make_rule(str(ok), 5, "identity", None, None, "classification", [0, 3, 5])
            self.assertEqual(rule(np.ones((4, 5), np.float32)).shape, (4, 2))
            with self.assertRaisesRegex(ValueError, "one integer per row and output"):
                qf.make_rule(str(bad), 5, "identity", None, None, "classification", [0, 3, 5])
            with self.assertRaisesRegex(ValueError, "NaN"):
                qf.make_rule(str(nan), 5, "identity", None, None, "classification", [0, 3, 5])
            # without segments the same file is the 1.1 contract: one integer per row
            self.assertEqual(qf.make_rule(str(bad), 5, "identity", None, None, "classification")(np.ones((4, 5), np.float32)).shape, (4,))

    def test_regression_rule_bins_every_column_on_its_own_width(self):
        p = np.stack([np.linspace(0, 1, 50), np.linspace(0, 1000, 50)], 1)
        rule = qf.make_rule("argmax", 2, "identity", p, None, "regression")
        d = rule(p)
        self.assertEqual(d.shape, (50, 2))
        self.assertEqual(int(d[:, 0].max()), int(d[:, 1].max()))  # each column spans the same number of bins

    def test_split_and_argmax_outputs(self):
        p = np.array([[0.1, 0.7, 0.2, 0.9, 0.1], [0.5, 0.2, 0.3, 0.2, 0.8]], np.float32)
        parts = evaluator.split_outputs(p, 5, [0, 3, 5], "classification")
        self.assertEqual([a.shape for a in parts], [(2, 3), (2, 2)])
        self.assertEqual(evaluator.argmax_outputs(p, 5, [0, 3, 5], "classification").tolist(), [[1, 0], [0, 1]])
        reg = evaluator.split_outputs(p, 5, None, "regression")
        self.assertEqual(len(reg), 5)
        np.testing.assert_array_equal(reg[2], p[:, 2])
        self.assertEqual(len(evaluator.split_outputs(p[:, 0], 1, None, "regression")), 1)  # predict() of one group is (rows,)
        with self.assertRaisesRegex(ValueError, "5 column"):
            evaluator.split_outputs(p[:, :4], 5, [0, 3, 5], "classification")
        with self.assertRaisesRegex(ValueError, "regressor"):
            evaluator.argmax_outputs(p, 5, None, "regression")
        stub = SimpleNamespace(n_groups=5, segments=[0, 3, 5], output_kind="classification")
        stub._output_kind = lambda: stub.output_kind
        self.assertEqual(len(evaluator.FlatForest.split_outputs(stub, p)), 2)
        self.assertEqual(evaluator.FlatForest.argmax_outputs(stub, p).shape, (2, 2))

    def test_check_ambiguous_per_segment(self):
        p = np.array([[0.4, 0.4, 0.2, 0.5, 0.5], [0.9, 0.05, 0.05, 0.1, 0.9]], np.float32)
        amb = check.ambiguous("argmax", p, 5, [0, 3, 5])
        self.assertEqual(amb.tolist(), [[True, True], [False, False]])
        amb = check.ambiguous("argmax", np.array([[0.5, 0.9]], np.float32), 2, [0, 1, 2])
        self.assertEqual(amb.tolist(), [[True, False]])
        res = check.compare(
            "t",
            p,
            p.copy(),
            "argmax",
            qf.make_rule("argmax", 5, "identity", None, None, "classification", [0, 3, 5]),
            5,
            segments=[0, 3, 5],
        )
        self.assertTrue(res["ok"])
        other = p.copy()
        other[1, 3:] = [0.9, 0.1]  # the second output's decision flips, unambiguously
        rule = qf.make_rule("argmax", 5, "identity", None, None, "classification", [0, 3, 5])
        self.assertEqual(check.compare("t", p, other, "argmax", rule, 5, segments=[0, 3, 5])["decision_mismatches"], 1)

    def test_estimator_diff_lays_out_classifier_outputs_side_by_side(self):
        class Est:
            def predict_proba(self, x):
                return [np.full((len(x), 3), 1 / 3), np.full((len(x), 2), 0.5)]

        ref = np.hstack([np.full((4, 3), 1 / 3), np.full((4, 2), 0.5)])
        self.assertEqual(check.estimator_diff(Est(), np.zeros((4, 2)), ref), 0.0)
        self.assertGreater(check.estimator_diff(Est(), np.zeros((4, 2)), ref + 1e-3), 5e-4)
        with self.assertRaisesRegex(ValueError, "5 column"):
            check.estimator_diff(Est(), np.zeros((4, 2)), ref[:, :4])


class RegBarTest(unittest.TestCase):
    def test_one_output_dict_is_the_old_shape(self):
        a = np.linspace(1, 2, 100)
        r = qf.RegBar("raw").check(a[:, None], a[:, None] * 1.001, a * 1.001)
        self.assertNotIn("per_output", r)
        self.assertTrue(r["ok"] and r["error_checked"])
        self.assertEqual(r["rows"], 100)

    def test_every_output_must_pass_and_the_top_level_is_the_worst(self):
        rng = np.random.default_rng(0)
        a = rng.random((200, 3)) + 1
        b = a * (1 + rng.normal(0, 0.0005, a.shape))
        b[:, 2] *= 1.03
        r = qf.RegBar("raw").check(a, b)
        self.assertFalse(r["ok"])
        self.assertEqual(len(r["per_output"]), 3)
        self.assertTrue(r["per_output"][0]["ok"] and r["per_output"][1]["ok"] and not r["per_output"][2]["ok"])
        self.assertEqual(r["median_pct"], r["per_output"][2]["median_pct"])
        self.assertTrue(all(m.startswith("output 2:") for m in r["failures"]))

    def test_nan_in_any_column_fails(self):
        a = np.ones((50, 3))
        b = a.copy()
        b[7, 1] = np.nan
        r = qf.RegBar("raw").check(a, b)
        self.assertFalse(r["ok"])
        self.assertFalse(r["per_output"][1]["ok"])
        self.assertTrue(np.isnan(r["max_pct"]) or r["max_pct"] > 2)

    def test_y_must_have_one_column_per_output(self):
        a = np.ones((50, 3))
        with self.assertRaisesRegex(ValueError, "1 column"):
            qf.RegBar("raw").check(a, a, np.ones(50))
        r = qf.RegBar("raw").check(a, a, np.ones((50, 3)))
        self.assertTrue(r["ok"] and r["error_checked"])


class LeafCodesVectorTest(unittest.TestCase):
    def test_vector_leaf_values_and_bytes(self):
        lv = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)  # 5 leaves in total, W = 3
        codes = np.array([[0, 1, 1, 0], [1, 0, 0, 1]], np.uint8)  # two trees of two leaves each... leaf_off 0 and 2
        lm = select_stream.LeafCodes(codes, lv, np.array([0, 2]), np.zeros((4, 3)), None)
        self.assertEqual(lm[1].shape, (4, 3))
        np.testing.assert_array_equal(lm[1], lv[2 + codes[1]])
        self.assertEqual(lm[np.array([0, 1])].shape, (2, 4, 3))
        np.testing.assert_array_equal(lm[np.array([0, 1])][0], lv[codes[0]])
        self.assertEqual(lm.leaf_value_bytes, 5 * 3 * 4)

    def test_scalar_leaf_values_unchanged(self):
        lv = np.array([1, 2, 3, 4, 5], np.float32)
        lm = select_stream.LeafCodes(np.array([[0, 1], [1, 0]], np.uint8), lv, np.array([0, 2]), np.zeros((2, 1)), None)
        self.assertEqual(lm[np.array([0, 1])].tolist(), [[1, 2], [4, 3]])


class StreamRefusalTest(unittest.TestCase):
    def test_vector_leaf_models_are_refused_before_a_scratch_folder(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            js = d / "m.json"
            js.write_text(json.dumps({"gbtree_model_param": {"num_trees": "3"}, "tree_param": {"size_leaf_vector": "2"}}))
            ubj = d / "m.ubj"
            ubj.write_bytes(b"{" + b"i\x0fgbtree_model_paramnum_trees" + b"i\x10size_leaf_vectorSL" + struct.pack(">q", 1) + b"3}")
            one = d / "one.json"
            one.write_text(json.dumps({"gbtree_model_param": {"num_trees": "400000"}, "tree_param": {"size_leaf_vector": "1"}}))
            wide = d / "wide.json"
            wide.write_text(json.dumps({"gbtree_model_param": {"num_trees": "400000"}, "tree_param": {"size_leaf_vector": "2"}}))
            self.assertEqual(select_stream.vector_leaf_width(js), 2)
            self.assertEqual(select_stream.vector_leaf_width(ubj), 3)
            self.assertEqual(select_stream.vector_leaf_width(one), 1)
            scratch = d / "scratch"
            scratch.mkdir()
            for f in (js, ubj):
                with self.assertRaisesRegex(ValueError, "vector leaves.*raise --memory-mb"):
                    prune.open_model(f, 1000, 1, scratch)
                self.assertEqual(list(scratch.iterdir()), [], "a refused model leaves no scratch file")
            # the in-memory estimate: one leaf code a (tree, row), W or not (code check 4: `test_codecheck_cc4.StreamEstimateTest`)
            self.assertTrue(select_stream.wanted(wide, 1000, 1500))
            self.assertFalse(select_stream.wanted(one, 1000, 3000))


class MultiRegressionPruneTest(unittest.TestCase):
    def run_prune(self, d, keep="50%", **kw):
        src = regression_source(d, **kw.pop("source", {}))
        out = d / "out"
        with FakeFormats():
            return prune.run(
                src, data_file(d, **kw.pop("data", {})), out, keep, regression=kw.pop("regression", qf.RegBar("raw")), **kw
            ), out

    def test_k5_raw_per_output(self):
        with tempfile.TemporaryDirectory() as d:
            rep, out = self.run_prune(Path(d), "50%")
            reg = rep["regression"]
            self.assertEqual(len(reg["per_output"]), 5)
            self.assertEqual(rep["per_output"], reg["per_output"])
            self.assertTrue(reg["ok"], reg["failures"])
            self.assertEqual(reg["median_pct"], max(o["median_pct"] for o in reg["per_output"]))
            per_group = rep["kept_indices"]["model.fakevec"]["per_group"]
            self.assertEqual(len(per_group), 5)
            self.assertTrue(all(g == per_group[0] for g in per_group), "the same vector trees feed every group")
            self.assertEqual(len(per_group[0]), 20)
            m = rep["models"][0]
            self.assertEqual((m["trees"], m["kept_trees"], m["kept_per_group"]), (40, 20, 20))
            for key in ("heldout", "totals", "params", "models", "kept_indices", "disagreements"):  # the old keys are still there
                self.assertIn(key, rep)
            self.assertIn("median_pct", rep["regression"])
            self.assertEqual(json.loads((out / "report.json").read_text())["regression"]["median_pct"], reg["median_pct"])
            self.assertIn("worst of 5 outputs", prune.summary(rep))

    def test_pruned_copy_holds_exactly_the_kept_trees_and_renormalises_nothing_for_boosting(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            rep, out = self.run_prune(d, "50%")
            kept = rep["kept_indices"]["model.fakevec"]["per_group"][0]
            full = REGISTRY[str(d / "model.fakevec")]
            pruned = REGISTRY[str(out / "model.fakevec")]
            self.assertEqual(len(pruned.forest.models[0].trees), len(kept))
            for t, i in zip(pruned.forest.models[0].trees, kept, strict=True):
                np.testing.assert_array_equal(t.vector, full.forest.models[0].trees[i].vector)

    def test_a_bad_output_fails_the_bar_and_is_named(self):
        with tempfile.TemporaryDirectory() as d:
            tight = qf.RegBar("raw", median_pct=1e-6, max_pct=1e-6)
            rep, _ = self.run_prune(Path(d), "10%", regression=tight)
            reg = rep["regression"]
            self.assertFalse(reg["ok"])
            self.assertTrue(any(m.startswith("output ") for m in reg["failures"]))

    def test_y_of_one_column_on_k5_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = regression_source(d)
            with FakeFormats(), self.assertRaisesRegex(SystemExit, "y has 1 column"):
                prune.run(src, data_file(d, y=np.ones(600)), d / "out", "50%", regression=qf.RegBar("raw"))
            with FakeFormats():
                rep = prune.run(
                    src, data_file(d, y=np.ones((600, 5)) * [1, 1e3, 1e-3, 1, 1]), d / "out2", "50%", regression=qf.RegBar("raw")
                )
            self.assertTrue(rep["regression"]["error_checked"])
            self.assertEqual(len(rep["regression"]["per_output"]), 5)

    def test_order_is_independent_of_the_output_units(self):
        """`raw` weighs every entry by 1/|full|: scaling a column by 1e6 must not change which trees are picked."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "a").mkdir()
            (d / "b").mkdir()
            a, _ = self.run_prune(d / "a", "40%", source={"scales": [1, 1, 1, 1, 1]})
            b, _ = self.run_prune(d / "b", "40%", source={"scales": [1, 1e6, 1, 1e-6, 1]})
            self.assertEqual(a["kept_indices"]["model.fakevec"]["per_group"][0], b["kept_indices"]["model.fakevec"]["per_group"][0])


class MultiClassificationPruneTest(unittest.TestCase):
    def test_sklearn_like_3_and_2_classes_auto(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = classification_source(d, [0, 3, 5], trees=40)
            with FakeFormats():
                rep = prune.run(src, data_file(d), d / "out", "auto", min_agreement=90.0)
            h = rep["heldout"]
            self.assertEqual(len(h["agreement_pct_per_output"]), 2)
            self.assertEqual([o["agreement_pct"] for o in rep["per_output"]], h["agreement_pct_per_output"])
            self.assertLessEqual(h["all_outputs_pct"], min(h["agreement_pct_per_output"]))
            self.assertEqual(h["pct"], min(h["agreement_pct_per_output"]))  # the top level is the worst output
            chosen = rep["auto"]["chosen"]
            self.assertTrue(all(h_ >= 90.0 for h_ in h["agreement_pct_per_output"]) or chosen["whole_forest"])
            for rung in rep["auto"]["rungs"]:
                self.assertEqual(len(rung["agreement_pct_per_output"]), 2)
                # the smallest passing rung: every output reaches the bar, or the rung did not pass
                self.assertEqual(rung["pass"], all(p_ >= 90.0 for p_ in rung["agreement_pct_per_output"]))
            self.assertIn("worst of 2 outputs", prune.summary(rep))

    def test_auto_picks_the_smallest_rung_where_every_output_passes(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = classification_source(d, [0, 3, 5], trees=40)
            with FakeFormats():
                rep = prune.run(src, data_file(d), d / "out", "auto", min_agreement=95.0, keep_ladder=["10%", "25%", "50%", "75%"])
            rungs = rep["auto"]["rungs"]
            first = next((r for r in rungs if r["pass"]), None)
            chosen = rep["auto"]["chosen"]
            if first is None:
                self.assertTrue(chosen["whole_forest"])
            else:
                self.assertEqual(chosen["kept_trees"], first["kept_trees"])

    def test_accuracy_per_output_when_y_is_given(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = classification_source(d, [0, 3, 5], trees=30)
            y = np.random.default_rng(0).integers(0, 2, (600, 2))
            with FakeFormats():
                rep = prune.run(src, data_file(d, y=y), d / "out", "50%")
            acc = rep["heldout"]["accuracy"]
            self.assertEqual(len(acc["full_pct_per_output"]), 2)
            self.assertEqual(acc["full_pct"], min(acc["full_pct_per_output"]))

    def test_xgboost_multilabel_k2_decides_per_column_and_the_copy_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = classification_source(d, [0, 2], trees=30, kind="sigmoid", mean=False, name="ml.fakevec")
            reg = REGISTRY[str(src)]
            for t in reg.forest.models[0].trees:  # boosting-like margins around 0
                t.vector = ((t.vector * 30 - 0.5) * 2).astype(np.float32)
                t.vector[t.left != -1] = 0
            reg.forest.segments = reg.segments = None
            with FakeFormats():
                rep = prune.run(src, data_file(d), d / "out", "50%")
            h = rep["heldout"]
            self.assertEqual(len(h["agreement_pct_per_output"]), 2)  # one decision per sigmoid column
            pruned = REGISTRY[str(d / "out" / "ml.fakevec")]
            kept = rep["kept_indices"]["ml.fakevec"]["per_group"][0]
            x = np.random.default_rng(5).random((50, 4)).astype(np.float32)
            expect = reg.forest.models[0].base[None].repeat(50, 0).astype(np.float32)
            for i in kept:
                expect = expect + qf.tree_values(reg.forest.models[0].trees[i], x)
            np.testing.assert_array_equal(
                pruned.predict(x), qf.transform(expect, "sigmoid")
            )  # no re-normalisation for boosting, bit for bit

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = classification_source(d, [0, 2], trees=10, kind="sigmoid", mean=False, name="ml.fakevec")
            with FakeFormats():
                with self.assertRaisesRegex(ValueError, "single output"):
                    prune.run(src, data_file(d), d / "o1", "50%", decision="threshold:0.5")
                with self.assertRaisesRegex(SystemExit, "regression bar needs a regression model"):
                    prune.run(src, data_file(d), d / "o2", "50%", regression=qf.RegBar("raw"))
                with self.assertRaisesRegex(SystemExit, "orientation"):
                    prune.run(src, data_file(d), d / "o3", "50%", method="orientation")

    def test_single_decision_tree_below_100_percent_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = classification_source(d, [0, 3, 5], trees=1, name="dt.fakevec", fmt="sklearn")
            with FakeFormats():
                with self.assertRaisesRegex(SystemExit, "single decision tree"):
                    prune.run(src, data_file(d), d / "o1", "30%")
                rep = prune.run(src, data_file(d), d / "o2", "100%")
            self.assertEqual(rep["models"][0]["kept_trees"], 1)

    def test_a_vector_forest_never_reaches_greedy_order(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = classification_source(d, [0, 3, 5], trees=12)
            with FakeFormats(), mock.patch.object(prune, "greedy_order", side_effect=AssertionError("scalar path reached")):
                prune.run(src, data_file(d), d / "o1", "50%")
                prune.run(src, data_file(d), d / "o2", "auto")


class VectorGreedyTest(unittest.TestCase):
    def test_picks_are_the_same_list_for_every_group_and_a_prefix_of_a_longer_run(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = REGISTRY[str(classification_source(d, [0, 3, 5], trees=20))]
            x = np.random.default_rng(1).random((300, 4)).astype(np.float32)
            lm = prune.vector_leaves(src, x)
            p_full = src.predict(x)
            np.testing.assert_allclose(src.spec.base[None] + lm.margin, src.predict(x, margin=True), rtol=1e-5, atol=1e-6)
            self.assertEqual(lm[np.array([0, 3])].shape, (2, 300, 5))
            rule = qf.make_rule("argmax", 5, "identity", p_full, None, "classification", [0, 3, 5])
            long = prune.greedy_vector_order(lm, src.spec, 12, rule, True, p_full, rule(p_full))
            short = prune.greedy_vector_order(lm, src.spec, 5, rule, True, p_full, rule(p_full))
            self.assertTrue(all(np.array_equal(g, long[0]) for g in long))
            np.testing.assert_array_equal(short[0], long[0][:5])
            self.assertEqual(len(set(long[0].tolist())), 12)


class EstimatorApiTest(unittest.TestCase):
    def test_not_a_model(self):
        with self.assertRaisesRegex(ValueError, "neither a model path nor a fitted sklearn estimator"):
            prune.export_estimator(object(), Path("/nonexistent"), "x.npz")

    def test_unsupported_and_unfitted_sklearn(self):
        try:
            from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        except ImportError:
            self.skipTest("scikit-learn is not installed here")
        with self.assertRaisesRegex(ValueError, "GradientBoostingRegressor: not supported"):
            prune.export_estimator(GradientBoostingRegressor(), Path("/nonexistent"), "x.npz")
        with self.assertRaisesRegex(ValueError, "call fit first"):
            prune.export_estimator(RandomForestRegressor(), Path("/nonexistent"), "x.npz")

    def test_run_exports_then_prunes_the_file(self):
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            self.skipTest("scikit-learn is not installed here")
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            est = RandomForestRegressor(n_estimators=2, random_state=0).fit(np.random.default_rng(0).random((30, 4)), np.arange(30.0))
            seen = {}

            def fake_export(estimator, path, X=None):
                seen["rows"] = len(X)
                src = regression_source(Path(path).parent, name=Path(path).name)
                return {"verified_rows": 3, "max_diff": 0.0, "normalised": False, "src": str(src)}

            with FakeFormats(), mock.patch.dict(sys.modules, {"qf_sklearn": SimpleNamespace(export_sklearn=fake_export)}):
                rep = prune.run(est, data_file(d), d / "out", "50%", regression=qf.RegBar("raw"))
            self.assertEqual(rep["library_diff"], 0.0)
            self.assertTrue(rep["model"].endswith("source/model.qfsk"))
            self.assertEqual(seen["rows"], 600)
            self.assertTrue((d / "out" / "model.qfsk").exists())


if __name__ == "__main__":
    unittest.main()
