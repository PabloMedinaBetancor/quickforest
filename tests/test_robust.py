"""Pinning tests for the failures the fuzz harness (tests/fuzz/fuzz_loader.py) found: a damaged model, data, choice or flat file ends in ONE
clean exception that names the problem (never a KeyError from inside a parser, a hang, a memory blow-up, a crash or code run from a data
file)."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import backends
import evaluator
import qf_common as qf
import qf_formats

FIX = HERE / "fixtures"


def flat_file(t: Path) -> Path:
    forest = qf.load_forest([FIX / "lgb_plain.txt"])
    out = t / "m.flat"
    backends.get("flat").compile(forest, out, {})
    return out


class LightGbmParserTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.text = (FIX / "lgb_plain.txt").read_text()

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, text: str):
        p = self.t / "m.txt"
        p.write_text(text)
        return qf.load_forest([p])

    def test_the_unchanged_model_loads(self):
        self.assertGreater(len(self.load(self.text).models[0].trees), 0)

    def test_damage_is_a_value_error_that_names_the_file(self):
        first = self.text.index("Tree=0")
        second = self.text.index("Tree=1")
        tree0 = self.text[first:second]
        for what, text in {
            "truncated (no 'end of trees')": self.text[: self.text.index("end of trees")],
            "no max_feature_idx": re.sub(r"(?m)^max_feature_idx=.*\n", "", self.text),
            "no num_tree_per_iteration": re.sub(r"(?m)^num_tree_per_iteration=.*\n", "", self.text),
            "no split_feature": self.text.replace(tree0, re.sub(r"(?m)^split_feature=.*\n", "", tree0)),
            "no leaf_value": self.text.replace(tree0, re.sub(r"(?m)^leaf_value=.*\n", "", tree0)),
            "child out of range": self.text.replace(tree0, tree0.replace("left_child=2 ", "left_child=99 ")),
            "feature out of range": self.text.replace(tree0, tree0.replace("split_feature=0 ", "split_feature=500 ")),
            "fewer leaf values": self.text.replace(tree0, re.sub(r"(?m)^leaf_value=(\S+) \S+ ", r"leaf_value=\1 ", tree0)),
            "non-numeric threshold": self.text.replace(tree0, tree0.replace("threshold=-0.0", "threshold=abc", 1)),
            "NaN threshold": self.text.replace(tree0, tree0.replace("threshold=-0.0", "threshold=nan ", 1)),
            "more trees per iteration than trees": self.text.replace("num_tree_per_iteration=1", "num_tree_per_iteration=9999"),
        }.items():
            with self.subTest(what), self.assertRaises(ValueError) as cm:
                self.load(text)
            self.assertTrue(str(cm.exception), what)

    def test_a_model_with_no_trees_says_so(self):
        text = self.text[: self.text.index("Tree=0")] + self.text[self.text.index("end of trees") :]
        text = re.sub(r"(?m)^tree_sizes=.*\n", "", text)
        with self.assertRaises(ValueError) as cm:
            self.load(text)
        self.assertIn("no trees", str(cm.exception))

    def test_a_header_that_lightgbm_trusts_but_this_reader_ignored_is_refused(self):
        """LightGBM cuts the trees out of the file by `tree_sizes` and reads one name and one range per feature: a file where those lines are
        wrong compiled here, then `prune` and `check` loaded it with LightGBM and the process aborted (SIGABRT) or failed in the library."""
        for what, text in {
            "tree_sizes larger": re.sub(r"(?m)^tree_sizes=(\d+)", lambda m: f"tree_sizes={int(m.group(1)) + 7}", self.text, count=1),
            "tree_sizes smaller": re.sub(r"(?m)^tree_sizes=(\d+)", lambda m: f"tree_sizes={int(m.group(1)) - 7}", self.text, count=1),
            "tree_sizes not numbers": re.sub(r"(?m)^tree_sizes=.*", "tree_sizes=a b", self.text, count=1),
            "one tree_sizes entry missing": re.sub(r"(?m)^(tree_sizes=\d+(?: \d+)*) \d+$", r"\1", self.text, count=1),
            "feature_infos one short": re.sub(r"(?m)^(feature_infos=.*) \S+$", r"\1", self.text, count=1),
            "feature_names one short": re.sub(r"(?m)^(feature_names=.*) \S+$", r"\1", self.text, count=1),
            "feature_names one more": re.sub(r"(?m)^(feature_names=.*)$", r"\1 extra", self.text, count=1),
        }.items():
            self.assertNotEqual(text, self.text, what)
            with self.subTest(what), self.assertRaisesRegex(ValueError, "tree_sizes|feature_names|feature_infos"):
                self.load(text)
        self.assertEqual(qf_formats.open_source(FIX / "lgb_plain.txt").booster.num_trees(), len(self.load(self.text).models[0].trees))

    def test_a_damaged_tail_after_the_trees_is_refused_because_lightgbm_crashes_on_it(self):
        """LightGBM parses the parameters block that follows the trees when it loads a model, and the process died (SIGSEGV) on a block that
        was cut or whose lines were damaged, a part of the file this reader does not otherwise need. A NUL byte made the loader hang."""
        cut = self.text[: self.text.index("[objective_seed")] + "[objec"
        for what, text in {
            "parameters cut inside a line": cut,
            "parameters not closed": self.text.replace("end of parameters", "", 1),
            "a parameter line with = in place of the colon": self.text.replace(
                "[max_cat_threshold: 32]", "[max_cat_threshold=1 2 3 -5 7]", 1
            ),
            "text that is not a parameter line": self.text.replace("end of parameters", "junk junk\nend of parameters", 1),
            "a NUL byte": self.text + "\0",
            "bytes that are not UTF-8": self.text.replace("pandas_categorical:null", "pandas_categorical:\udcff", 1),
        }.items():
            self.assertNotEqual(text, self.text, what)
            p = self.t / "tail.txt"
            p.write_bytes(text.encode("utf-8", "surrogateescape"))
            with self.subTest(what), self.assertRaises(ValueError):
                qf_formats.open_source(p)

    def test_a_model_the_library_itself_rejects_is_a_value_error_not_the_librarys_exception(self):
        import lightgbm

        source = qf_formats.open_source(FIX / "lgb_plain.txt")
        with mock.patch.object(lightgbm, "Booster", side_effect=lightgbm.basic.LightGBMError("Model format error")):
            with self.assertRaises(ValueError) as cm:
                _ = source.booster
        self.assertIn("lgb_plain.txt: LightGBM cannot load this model (Model format error)", str(cm.exception))

    def test_a_huge_num_leaves_is_refused_before_anything_is_allocated(self):
        tree0 = self.text[self.text.index("Tree=0") : self.text.index("Tree=1")]
        with self.assertRaises(ValueError) as cm:
            self.load(self.text.replace(tree0, tree0.replace("num_leaves=7", "num_leaves=99999999999")))
        self.assertIn("num_leaves", str(cm.exception))

    def test_a_cycle_or_a_shared_child_does_not_hang_the_compiler(self):
        tree0 = self.text[self.text.index("Tree=0") : self.text.index("Tree=1")]
        for what, bad in {
            "cycle": tree0.replace("left_child=2 5 -1 -4 -5 -2", "left_child=0 5 -1 -4 -5 -2"),
            "two parents": tree0.replace("right_child=1 -3 3 4 -6 -7", "right_child=2 -3 3 4 -6 -7"),
        }.items():
            with self.subTest(what), self.assertRaises(ValueError) as cm:
                self.load(self.text.replace(tree0, bad))
            self.assertRegex(str(cm.exception), "cycle|two parents|reach")


class XgboostParserTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.j = json.loads((FIX / "xgb_reg.json").read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, j):
        p = self.t / "m.json"
        p.write_text(json.dumps(j))
        return qf.load_forest([p])

    def tree(self, j=None):
        return (j or self.j)["learner"]["gradient_booster"]["model"]["trees"][0]

    def test_the_unchanged_model_loads(self):
        self.assertGreater(len(self.load(self.j).models[0].trees), 0)

    def test_damage_is_a_value_error(self):
        def mut(fn):
            j = json.loads(json.dumps(self.j))
            fn(j)
            return j

        cases = {
            "no split_indices": lambda j: self.tree(j).pop("split_indices"),
            "no objective": lambda j: j["learner"].pop("objective"),
            "no tree_info": lambda j: j["learner"]["gradient_booster"]["model"].pop("tree_info"),
            "no base_score": lambda j: j["learner"]["learner_model_param"].pop("base_score"),
            "tree_info too short": lambda j: j["learner"]["gradient_booster"]["model"]["tree_info"].pop(),
            "children of different length": lambda j: self.tree(j)["right_children"].pop(),
            "cycle": lambda j: self.tree(j)["left_children"].__setitem__(0, 0),
            "child before its parent": lambda j: self.tree(j)["left_children"].__setitem__(2, 1),
            "child out of range": lambda j: self.tree(j)["left_children"].__setitem__(0, 9999),
            "two parents": lambda j: self.tree(j)["right_children"].__setitem__(0, self.tree(j)["left_children"][0]),
            "feature out of range": lambda j: self.tree(j)["split_indices"].__setitem__(0, 9999),
            "empty tree": lambda j: [self.tree(j)[k].clear() for k in ("left_children", "right_children")],
        }
        for what, fn in cases.items():
            with self.subTest(what), self.assertRaises(ValueError) as cm:
                self.load(mut(fn))
            self.assertTrue(str(cm.exception), what)

    def test_a_model_with_no_trees_says_so(self):
        """num_boost_round=0 saves a model without trees; numpy's "need at least one array to concatenate" was all the user read."""
        j = json.loads(json.dumps(self.j))
        model = j["learner"]["gradient_booster"]["model"]
        model["trees"], model["tree_info"], model["iteration_indptr"] = [], [], [0]
        model["gbtree_model_param"]["num_trees"] = "0"
        with self.assertRaises(ValueError) as cm:
            self.load(j)
        self.assertIn("no trees", str(cm.exception))

    def test_a_model_with_no_input_features_is_refused_by_every_format(self):
        """xgboost refuses to train a model with `num_feature` 0 but loads one that was edited or damaged; it compiled without a word and
        the generated code then divided by the number of features and aborted the process on the first prediction."""
        j = json.loads(json.dumps(self.j))
        j["learner"]["learner_model_param"]["num_feature"] = "0"
        for t in j["learner"]["gradient_booster"]["model"]["trees"]:  # stumps: nothing else in the model reads a feature
            t.update(left_children=[-1], right_children=[-1], split_indices=[0], split_conditions=[0.5], default_left=[0], split_type=[0])
            t["categories"] = []
        with self.assertRaises(ValueError) as cm:
            self.load(j)
        self.assertIn("no input features", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            qf_formats.spec_for(0, 1, "identity", [0.0], [0], [1], None)
        self.assertIn("no input features", str(cm.exception))
        import backend_codegen

        forest = qf.Forest(0, 1, "identity", "output", [qf.ModelIR(np.zeros(1, np.float32), [])])  # built by hand, past the Spec
        self.assertIn("if NF == 0 { return 1; }", backend_codegen.driver_source(forest, [[0]], 8))


class DataAndChoiceFilesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_data_file_without_x_is_a_value_error_and_the_cli_helper_exits_with_the_message(self):
        p = self.t / "d.npz"
        np.savez(p, y=np.zeros(3))
        with self.assertRaises(ValueError):
            qf.load_data(p)
        with self.assertRaises(SystemExit) as cm:
            qf.load_data_cli(p)
        self.assertIn("array named X", str(cm.exception))
        np.savez(p, X=np.zeros(3))
        with self.assertRaises(ValueError):
            qf.load_data(p)

    def test_a_fifo_or_a_directory_as_the_data_file_is_refused_not_waited_on(self):
        """LIMITS.md (Safety): opening a FIFO or a directory is refused rather than blocking. The data file is opened by np.load, which
        blocked for ever on a FIFO: the check runs in a child process with a timeout so that a hang fails the test instead of the run."""
        fifo = self.t / "pipe.npz"
        os.mkfifo(fifo)
        code = (
            "import sys; sys.path.insert(0, sys.argv[1]); import qf_common as qf\n"
            "for p in sys.argv[2:]:\n"
            "    try:\n        qf.load_data(p)\n    except ValueError as e:\n        print('refused', e)\n"
            "    else:\n        print('accepted')\n"
        )
        for path in (fifo, self.t):
            p = subprocess.run(
                [sys.executable, "-s", "-c", code, str(HERE.parent / "scripts"), str(path)], capture_output=True, text=True, timeout=60
            )
            self.assertEqual(p.stdout.split()[:1], ["refused"], (path, p.stdout, p.stderr))
            self.assertIn("regular file", p.stdout)
        with self.assertRaises(SystemExit) as cm:
            qf.load_data_cli(self.t)
        self.assertIn("regular file", str(cm.exception))
        with self.assertRaises(SystemExit) as cm:
            qf.load_data_cli(self.t / "missing.npz")  # a missing file keeps the operating system's own message
        self.assertIn("missing.npz", str(cm.exception))

    def test_the_optional_arrays_of_a_data_file_must_be_one_per_row(self):
        """`y`, `fold`, `group`, `slice` and `eligible` broadcast against (n,) when they are (n, 1): the MAE of the error bar was taken over
        all n x n pairs and gave PASS for a pruned model that fails. An (n, 1) column is flattened (the one unambiguous reading);
        anything else that is not one value per row is refused in the data file's own message. `base_margin` is (n,) or (n, groups)."""
        n = 12
        x = np.zeros((n, 3), np.float32)
        col = np.arange(n, dtype=np.float64)
        for name in ("y", "fold", "group", "slice", "eligible"):
            good = col % 2 if name == "slice" else col  # a slice holds 0, 1 or -1 only
            np.savez(self.t / "ok.npz", X=x, **{name: good.reshape(-1, 1)})
            got = qf.load_data(self.t / "ok.npz")[name]
            self.assertEqual(got.shape, (n,), name)
            self.assertTrue(np.array_equal(got, good), name)
            for bad in (col[:-1], np.zeros((n, 2)), np.zeros((n, 1, 1)), np.float64(3.0), np.zeros((1, n))):
                np.savez(self.t / "bad.npz", X=x, **{name: bad})
                with self.subTest(name=name, shape=np.shape(bad)), self.assertRaises(ValueError) as cm:
                    qf.load_data(self.t / "bad.npz")
                self.assertIn(name, str(cm.exception))
            np.savez(self.t / "bad.npz", X=x, **{name: col[:-1]})
            with self.assertRaises(SystemExit) as cm:
                qf.load_data_cli(self.t / "bad.npz")
            self.assertEqual(len(str(cm.exception).splitlines()), 1)
        for ok in (np.zeros(n), np.zeros((n, 1)), np.zeros((n, 4))):
            np.savez(self.t / "ok.npz", X=x, base_margin=ok)
            self.assertEqual(qf.load_data(self.t / "ok.npz")["base_margin"].shape[0], n)
        for bad in (np.zeros(n + 1), np.zeros((n, 2, 2)), np.float64(1.0)):
            np.savez(self.t / "bad.npz", X=x, base_margin=bad)
            with self.assertRaises(ValueError) as cm:
                qf.load_data(self.t / "bad.npz")
            self.assertIn("base_margin", str(cm.exception))

    def test_a_truth_column_gives_the_same_error_bar_verdict_as_a_flat_truth(self):
        """The reproduction through prune.run: y as (n, 1) and as (n,) must give the same MAEs and the same verdict."""
        import prune

        x = np.load(FIX / "x.npy")
        y = np.nan_to_num(x[:, 0]) * 2 + 1.0
        rep = {}
        for tag, yy in (("flat", y), ("column", y.reshape(-1, 1))):
            np.savez(self.t / f"sel-{tag}.npz", X=x, y=yy)
            r = prune.run(
                FIX / "xgb_reg.json", self.t / f"sel-{tag}.npz", self.t / f"out-{tag}", "50%", regression=qf.RegBar("raw", 50, 100, 0.0)
            )
            rep[tag] = {k: r["regression"][k] for k in ("mae_full", "mae_pruned", "ok", "rows")}
        self.assertEqual(rep["flat"], rep["column"])
        self.assertLess(rep["flat"]["mae_full"], 10)

    def test_a_choice_file_is_data_and_never_runs_a_file(self):
        art = flat_file(self.t)
        marker = self.t / "ran"
        evil = self.t / "evil.py"
        evil.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\nclass Backend:\n    name='evil'\n")
        choice = Path(f"{art}.choice.json")
        for body in (
            json.dumps({"backend": str(evil)}),
            json.dumps({"backend": f"{evil}:Backend"}),
            json.dumps({}),
            json.dumps([1]),
            json.dumps({"backend": 7}),
            "null",
            "{",
        ):
            choice.write_text(body)
            with self.subTest(body), self.assertRaises(ValueError) as cm:
                backends.load_artifact(art)
            self.assertTrue(str(cm.exception))
            self.assertFalse(marker.exists(), "a choice file made the loader execute a Python file")


class OutputFolderLinksTest(unittest.TestCase):
    """Nothing is written outside an output folder through a link that sits in it: prune wrote the pruned model, report.json and
    heldout_probs.npz in place, so a planted symlink sent them anywhere; every output is now a NEW file renamed over the name (as compile
    does)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        (self.t / "out").mkdir()
        (self.t / "elsewhere").mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def plant(self, *names):
        for name in names:
            (self.t / "elsewhere" / name).write_text("precious\n")
            os.symlink(self.t / "elsewhere" / name, self.t / "out" / name)

    def assert_nothing_escaped(self, names):
        for name in names:
            self.assertEqual((self.t / "elsewhere" / name).read_text(), "precious\n", f"{name}: written through the link")
            self.assertFalse((self.t / "out" / name).is_symlink(), f"{name} is still a link")
            self.assertTrue((self.t / "out" / name).is_file())
        self.assertEqual(sorted(p.name for p in (self.t / "elsewhere").iterdir()), sorted(names))

    def test_prune_writes_its_outputs_as_new_files_for_xgboost_and_lightgbm(self):
        import prune

        np.savez(self.t / "sel.npz", X=np.load(FIX / "x.npy"))
        for model in ("xgb_reg.json", "lgb_plain.txt"):
            for p in list((self.t / "out").iterdir()) + list((self.t / "elsewhere").iterdir()):
                p.unlink()
            names = (model, "report.json", "heldout_probs.npz")
            self.plant(*names)
            prune.run(FIX / model, self.t / "sel.npz", self.t / "out", "50%", save_probs=True)
            self.assert_nothing_escaped(names)
            self.assertGreater(len(qf.load_forest([self.t / "out" / model]).models[0].trees), 0)
            self.assertEqual(json.loads((self.t / "out" / "report.json").read_text())["params"]["keep"], "50%")

    def test_the_choice_file_next_to_a_compiled_artifact_is_not_written_through_a_link(self):
        import choose

        art = self.t / "out" / "m.flat"
        os.symlink(self.t / "elsewhere" / "victim.json", Path(f"{art}.choice.json"))
        (self.t / "elsewhere" / "victim.json").write_text("precious\n")
        np.save(self.t / "rows.npy", np.load(FIX / "x.npy"))
        choose.choose(FIX / "lgb_plain.txt", art, self.t / "rows.npy", only=["flat"], log=lambda *_: None)
        self.assertEqual((self.t / "elsewhere" / "victim.json").read_text(), "precious\n")
        self.assertFalse(Path(f"{art}.choice.json").is_symlink())
        self.assertEqual(json.loads(Path(f"{art}.choice.json").read_text())["backend"], "flat")

    def test_check_and_compare_reports_and_work_files_are_not_written_through_a_link(self):
        flat = self.t / "m.flat"
        backends.get("flat").compile(qf.load_forest([FIX / "lgb_plain.txt"]), flat, {})
        np.savez(self.t / "x.npz", X=np.load(FIX / "x.npy"))
        work = self.t / "out" / "work"
        work.mkdir()
        planted = {
            self.t / "out": ("check.json", "report.json", "report.md"),
            work: (
                "sets.json",
                "prepared.json",
                "x-data.npy",
                "x-random.npy",
                "x-speed.npy",
                "x-pgo.npy",
                "build-flat.json",
                "spec-flat.json",
                "res-flat.json",
                "flat-data.npy",
                "flat-random.npy",
            ),
        }
        for folder, names in planted.items():
            for name in names:
                victim = self.t / "elsewhere" / f"{folder.name}-{name}"
                victim.write_text("precious\n")
                os.symlink(victim, folder / name)
        out = self.t / "out"
        scripts = HERE.parent / "scripts"
        done = subprocess.run(
            [
                sys.executable,
                "-I",
                str(scripts / "check.py"),
                str(FIX / "lgb_plain.txt"),
                str(flat),
                "--random",
                "50",
                "--json",
                str(out / "check.json"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-400:])
        done = subprocess.run(
            [
                sys.executable,
                "-I",
                str(scripts / "compare.py"),
                str(FIX / "lgb_plain.txt"),
                "--data",
                str(self.t / "x.npz"),
                "--backends",
                "flat",
                "--work",
                str(work),
                "--eq-rows",
                "100",
                "--random",
                "50",
                "--speed-rows",
                "100",
                "--json",
                str(out / "report.json"),
                "--md",
                str(out / "report.md"),
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-400:])
        for folder, names in planted.items():
            for name in names:
                victim = self.t / "elsewhere" / f"{folder.name}-{name}"
                self.assertEqual(victim.read_text(), "precious\n", f"{folder.name}/{name}: written through the link")
                if (folder / name).exists():
                    self.assertFalse((folder / name).is_symlink(), f"{folder.name}/{name} is still a link")
        self.assertIn('"ok"', (out / "check.json").read_text())
        self.assertIn('"rows"', (out / "report.json").read_text())


class EvaluatorApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        cls.art = flat_file(cls.t)
        cls.fo = evaluator.FlatForest(cls.art)
        cls.x = np.zeros((4, cls.fo.n_features), np.float32)

    @classmethod
    def tearDownClass(cls):
        cls.fo.close()
        cls.tmp.cleanup()

    def test_arrays_numpy_holds_at_an_odd_address_reach_the_library_aligned(self):
        """A float32 view of a byte buffer can sit at any address; Rust builds slices from the pointer, and a misaligned one is undefined
        behaviour. The answer must be the aligned copy's, and every pointer the library receives must be aligned."""
        seen = []

        class Spy:
            def __init__(self, lib):
                self._lib = lib

            def __getattr__(self, name):
                fn = getattr(self._lib, name)
                if not name.startswith("tfl_predict"):
                    return fn

                def call(h, x, n, out, threads, *rest):
                    seen.append((name, x, *(r for r in rest if r)))
                    return fn(h, x, n, out, threads, *rest)

                call.argtypes = fn.argtypes
                return call

        nf, rows = self.fo.n_features, 6
        rng = np.random.default_rng(0)
        want_x = rng.normal(size=(rows, nf)).astype(np.float32)
        want_bm = rng.normal(size=rows)
        raw_x = np.zeros(want_x.nbytes + 8, np.uint8)
        raw_bm = np.zeros(want_bm.nbytes + 8, np.uint8)
        for offset in (1, 2, 3, 5):
            x = raw_x[offset : offset + want_x.nbytes].view(np.float32).reshape(rows, nf)
            bm = raw_bm[offset : offset + want_bm.nbytes].view(np.float64)
            x[:], bm[:] = want_x, want_bm
            self.assertFalse(x.flags.aligned)
            forest = evaluator.FlatForest(self.art)
            forest._lib = Spy(forest._lib)
            try:
                for kw in ({}, {"base_margin": bm}, {"dtype": np.float64}, {"base_margin": bm, "dtype": np.float64}):
                    seen.clear()
                    got = forest.predict(x, **kw)
                    ref = forest.predict(want_x, **{k: (want_bm if k == "base_margin" else v) for k, v in kw.items()})
                    self.assertTrue(np.array_equal(got, ref), (offset, kw))
                    self.assertTrue(seen, kw)
                    for name, *pointers in seen:
                        self.assertEqual([p for p in pointers if p % 8], [], (offset, kw, name))
            finally:
                forest.close()

    def test_misuse_is_a_value_error(self):
        with self.assertRaises(ValueError):
            evaluator.FlatForest(str(self.art) + "\0x")
        with self.assertRaises(ValueError):
            self.fo.predict(self.x, dtype="nonsense")
        with self.assertRaises(ValueError):
            self.fo.predict(self.x, threads=1.5)
        self.assertEqual(self.fo.predict(self.x, threads=2**40).shape, (4,))  # clamped, not wrapped into the C unsigned
        self.assertEqual(self.fo.predict(self.x, threads=-3).shape, (4,))

    def test_complex_and_masked_arrays_are_refused_not_read_through(self):
        """numpy turns a complex array into float32 with a warning (the imaginary part is dropped) and `ascontiguousarray` hands a masked array's
        raw data on (the mask is ignored): both answered from numbers the caller did not mean. Rows and base margin; the shared helper also
        serves CodegenForest."""
        x = np.asarray(self.x, np.float32)
        bad = {
            "complex rows": x.astype(np.complex64),
            "masked rows": np.ma.masked_array(x, mask=x > 0),
            "complex list": [list(r) for r in x.astype(np.complex128)],
        }
        for name, rows in bad.items():
            with self.assertRaises(ValueError, msg=name) as cm:
                self.fo.predict(rows)
            self.assertNotIn("\n", str(cm.exception), name)
            with self.assertRaises(ValueError, msg=name):
                evaluator.float32_rows(rows)
        bm = np.zeros(len(x))
        for name, margin in {"complex": bm.astype(complex), "masked": np.ma.masked_array(bm, mask=bm == 0)}.items():
            with self.assertRaises(ValueError, msg=f"base_margin {name}"):
                self.fo.predict(x, base_margin=margin, dtype=np.float64)
        self.assertEqual(self.fo.predict(np.ma.masked_array(x, mask=False).filled()).shape, (len(x),))  # the way out the message names

    def test_the_reason_a_file_is_refused_survives_a_path_of_hundreds_of_bytes(self):
        """The library writes `<path>: <reason>` into a buffer of 512 bytes and cuts it there: a long path pushed the reason out, or split a
        multi-byte character and made the caller's decode raise UnicodeDecodeError instead of the ValueError."""
        for pad in range(3):  # the cut falls on every position inside a 3-byte character
            folder = self.t / ("x" * pad) / ("日" * 80) / ("本" * 80) / ("語" * 40)
            folder.mkdir(parents=True)
            (folder / "empty.flat").write_bytes(b"")
            with self.subTest(path_bytes=len(os.fsencode(folder / "empty.flat"))), self.assertRaises(ValueError) as cm:
                evaluator.FlatForest(folder / "empty.flat")
            self.assertNotIsInstance(cm.exception, UnicodeDecodeError)
            self.assertIn("empty file", str(cm.exception))

    def test_every_entry_point_is_bound_when_the_library_loads(self):
        lib = evaluator.load_lib(evaluator.find_lib(self.art))
        for name in ("tfl_open", "tfl_info", "tfl_predict", "tfl_predict_ex", "tfl_engine_ok", "tfl_predict_margin", "tfl_predict_f64"):
            self.assertIsNotNone(getattr(lib, name).argtypes, name)
        self.assertEqual(lib.tfl_engine_ok(0), 1)

    def test_a_file_that_is_not_the_evaluator_is_refused_with_one_message_and_not_kept(self):
        """A stale TREEFLAT_LIB, or any other shared object, lacks the entry points: say so, and do not cache a half-bound library."""
        fake = self.t / "not-the-evaluator.so"
        fake.write_bytes(b"not a library, a unique digest")
        for _ in range(2):  # the second call must not be answered from a cache of the first
            with (
                mock.patch.object(ctypes, "CDLL", return_value=types.SimpleNamespace()),
                self.assertRaises(ValueError) as cm,
            ):
                evaluator.load_lib(fake)
            self.assertIn("is not a build of this repository's evaluator", str(cm.exception))
            self.assertIn("rust/build.sh", str(cm.exception))

    def test_a_file_name_that_is_not_utf8_is_shown_with_an_escape_in_every_message(self):
        """Latin-1 names from an old archive are legal on Linux; a message that carried the raw characters could not be written out."""
        odd = os.path.join(os.fsdecode(bytes(self.t)), os.fsdecode(b"caf\xe9.flat"))
        self.assertEqual(evaluator.shown(odd).split(os.sep)[-1], "caf\\xe9.flat")
        with open(odd, "wb") as fh:
            fh.write(b"not a flat file")
        with self.assertRaises(ValueError) as cm:
            evaluator.FlatForest(odd)
        self.assertIn("caf\\xe9.flat", str(cm.exception))
        str(cm.exception).encode("utf-8")  # writable: no raw surrogate left in the message

    def test_a_fifo_or_a_directory_is_refused_instead_of_blocking(self):
        fifo = self.t / "pipe"
        os.mkfifo(fifo)
        for path in (fifo, self.t):
            with self.assertRaises(ValueError) as cm:
                evaluator.FlatForest(path)
            if path == fifo:
                self.assertIn("regular file", str(cm.exception))

    def test_close_while_another_thread_predicts_waits_instead_of_crashing_the_process(self):
        """ctypes drops the GIL during tfl_predict and close() freed and unmapped the model under the running walk: SIGSEGV. A service that
        swaps a recompiled forest closes the old one while requests are in flight. The race runs in a child process so a crash is an exit
        status here, not the end of the test run: the predicting thread must either finish or get the 'closed' ValueError, and the
        process must exit 0."""
        code = (
            "import sys, threading, time\n"
            "sys.path.insert(0, sys.argv[1]); import numpy as np, evaluator\n"
            "x = np.random.default_rng(0).normal(size=(300_000, int(sys.argv[3]))).astype(np.float32)\n"
            "for trial in range(6):\n"
            "    f = evaluator.FlatForest(sys.argv[2]); seen = []\n"
            "    def run():\n"
            "        try:\n"
            "            while True:\n                f.predict(x)\n"
            "        except ValueError as e:\n            seen.append(str(e))\n"
            "    t = threading.Thread(target=run); t.start(); time.sleep(0.02 * (trial + 1))\n"
            "    f.close(); t.join(60)\n"
            "    assert not t.is_alive() and seen == ['this forest was closed'], (trial, seen)\n"
            "print('survived')\n"
        )
        p = subprocess.run(
            [sys.executable, "-s", "-c", code, str(HERE.parent / "scripts"), str(self.art), str(self.fo.n_features)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual((p.returncode, p.stdout.strip()), (0, "survived"), p.stderr[-500:])

    def test_close_is_idempotent_and_a_closed_forest_refuses_predict(self):
        f = evaluator.FlatForest(self.art)
        f.close()
        f.close()
        with self.assertRaises(ValueError):
            f.predict(self.x)

    def test_the_c_functions_answer_one_for_sizes_that_overflow_and_for_null_pointers(self):
        lib, h = self.fo._lib, self.fo._h
        out = (ctypes.c_float * 4)()
        xs = self.x.ctypes.data
        for rows in (2**62, 2**63 + 5, 2**64 - 1):
            self.assertEqual(lib.tfl_predict(h, xs, rows, ctypes.addressof(out), 1), 1, rows)
        self.assertEqual(lib.tfl_predict(h, None, 4, ctypes.addressof(out), 1), 1)
        self.assertEqual(lib.tfl_predict(h, xs, 4, None, 1), 1)
        self.assertEqual(lib.tfl_predict(None, xs, 4, ctypes.addressof(out), 1), 1)
        self.assertEqual(lib.tfl_predict(h, None, 0, None, 1), 0)
        err = ctypes.create_string_buffer(64)
        self.assertFalse(lib.tfl_open(None, err, len(err)))
        self.assertIn(b"null", err.value)


class QuickforestHelpTest(unittest.TestCase):
    """`quickforest --help` is a request, not a failure: the usage goes to stdout and the exit code is 0."""

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-s", str(HERE.parent / "scripts" / "quickforest"), *args], capture_output=True, text=True)

    def test_help_prints_the_usage_to_stdout_and_exits_0(self):
        for flag in ("--help", "-h", "help"):
            r = self.run_cli(flag)
            self.assertEqual((r.returncode, r.stderr), (0, ""), flag)
            self.assertIn("quickforest", r.stdout)
            self.assertIn("compile", r.stdout)

    def test_no_command_or_an_unknown_one_still_fails_with_the_usage_on_stderr(self):
        for args in ((), ("nosuch",)):
            r = self.run_cli(*args)
            self.assertNotEqual(r.returncode, 0, args)
            self.assertEqual(r.stdout, "", args)
            self.assertIn("quickforest", r.stderr)


class LibraryNotFoundTest(unittest.TestCase):
    def test_the_message_names_the_machine_and_the_way_out(self):
        with (
            tempfile.TemporaryDirectory() as t,
            mock.patch.object(evaluator, "HERE", Path(t) / "scripts"),
            mock.patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("TREEFLAT_LIB", None)
            for machine in ("aarch64", "riscv64"):
                with mock.patch.object(platform, "machine", return_value=machine), self.assertRaises(FileNotFoundError) as cm:
                    evaluator.find_lib()
                self.assertIn(machine, str(cm.exception))
                self.assertIn("QUICK=1 rust/build.sh", str(cm.exception))
            with mock.patch.object(platform, "machine", return_value="x86_64"), self.assertRaises(FileNotFoundError) as cm:
                evaluator.find_lib()
            self.assertNotIn("QUICK=1", str(cm.exception))


class ModelPathErrorsTest(unittest.TestCase):
    def test_a_folder_name_that_is_not_utf8_ends_in_a_result_or_one_line_never_a_traceback(self):
        """A folder called `caf\\xe9` in Latin-1 is legal on Linux: compile wrote the file and then crashed printing its name, and check /
        optimize crashed encoding it for the library, which wants UTF-8 and says so. Compile now refuses such an output folder up front
        (the evaluator could never open the file it wrote)."""
        import shutil

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        bad = Path(os.fsdecode(os.fsencode(root) + b"/caf\xe9"))
        bad.mkdir()
        np.savez(root / "x.npz", X=np.load(FIX / "x.npy"))
        quickforest = str(HERE.parent / "scripts" / "quickforest")

        def run(*args):
            return subprocess.run([sys.executable, "-s", quickforest, *args], capture_output=True, text=True)

        done = run("compile", str(FIX / "xgb_reg.json"), str(bad / "m.flat"))
        self.assertEqual((done.returncode, "Traceback" in done.stderr), (3, False), done.stderr[-400:])
        self.assertIn("not valid UTF-8", done.stderr)
        self.assertFalse((bad / "m.flat").exists())
        run("compile", str(FIX / "xgb_reg.json"), str(root / "good.flat"))
        shutil.copy(root / "good.flat", bad / "m.flat")  # a file that was put there some other way is still refused in one line by check
        for args in (
            ("check", str(FIX / "xgb_reg.json"), str(bad / "m.flat"), "--random", "50"),
            ("optimize", str(FIX / "xgb_reg.json"), "--data", str(root / "x.npz"), "--compile", "--verify", "--out", str(bad / "o")),
        ):
            done = run(*args)
            self.assertNotIn("Traceback", done.stderr, args[0])
            self.assertIn("not UTF-8", done.stderr + done.stdout, args[0])
        (bad / "m.ubj").write_bytes(b"")  # a model library reads UTF-8 paths only: a model in such a folder is refused in one line too
        done = run("compile", str(bad / "m.ubj"), str(root / "m.flat"))
        self.assertNotIn("Traceback", done.stderr)
        self.assertIn("not UTF-8", done.stderr)

    def test_a_pipe_or_a_folder_named_like_a_model_in_a_fold_folder_is_not_a_fold(self):
        """Only regular files are folds: opening a named pipe (or a link to one) called `fold1.ubj` blocked the command for ever."""
        import shutil

        d = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, d, True)
        (d / "fold0.json").write_bytes((FIX / "xgb_reg.json").read_bytes())
        os.mkfifo(d / "fold1.ubj")
        os.symlink(d / "fold1.ubj", d / "fold2.ubj")
        (d / "fold3.ubj").mkdir()
        self.assertEqual([f.name for f in qf_formats.model_files(d)], ["fold0.json"])
        self.assertFalse(qf_formats.is_model_file(d / "fold1.ubj"))
        self.assertFalse(qf_formats.is_model_file(d / "fold2.ubj"))
        only_a_pipe = d / "only"
        only_a_pipe.mkdir()
        os.mkfifo(only_a_pipe / "fold0.ubj")
        with self.assertRaises(ValueError) as cm:
            qf_formats.model_files(only_a_pipe)
        self.assertIn("no xgboost, LightGBM or YDF model files", str(cm.exception))

    def test_a_model_path_that_is_not_a_model_is_a_value_error_and_still_a_clean_exit(self):
        """A LightGBM file with a damaged first line makes `open_source` raise a ValueError with the message, not call `SystemExit`. The
        command-line tools still stop with that message (the error is also a SystemExit)."""
        import qf_formats

        with tempfile.TemporaryDirectory() as t:
            bad = Path(t) / "model.txt"
            bad.write_bytes(b"X" + (FIX / "lgb_plain.txt").read_bytes()[1:])
            empty = Path(t) / "empty"
            empty.mkdir()
            for call, path in (
                (qf_formats.open_source, bad),
                (qf_formats.detect, bad),
                (qf_formats.model_files, empty),
                (qf_formats.model_files, Path(t) / "absent"),
            ):
                with self.assertRaises(ValueError) as cm:
                    call(path)
                self.assertTrue(str(cm.exception).strip())
                self.assertIsInstance(cm.exception, SystemExit)
                self.assertEqual(cm.exception.code, str(cm.exception))


class BarsAreJudgedOnExactFiguresTest(unittest.TestCase):
    """A bar is compared with the exact figure; the rounded one in a report is for reading. And a bar that is not a number is refused,
    because NaN compares False with everything and every model would pass it."""

    def test_the_regression_bar_sees_a_miss_smaller_than_the_rounding_of_the_report(self):
        full = np.full((1000, 1), 100.0, np.float32)
        bar = qf.RegBar("raw", 0.5, 2.0, 0.0)
        one_row = full.copy()
        one_row[0, 0] = 102.00004  # 2.00004 % off: the report rounds it to 2.0
        r = bar.check(full, one_row)
        self.assertEqual((r["over_max"], r["ok"]), (1, False))
        self.assertIn("1 rows differ", r["failures"][0])
        median = np.full_like(full, 100.50004)  # every row 0.50004 % off
        r = bar.check(full, median)
        self.assertFalse(r["ok"])
        self.assertIn("median", r["failures"][0])
        self.assertTrue(bar.check(full, np.full_like(full, 100.4))["ok"])

    def test_the_agreement_bar_is_judged_on_the_row_counts(self):
        import prune

        rule = lambda p: (p[..., 0] >= 0.5).astype(np.int64)  # noqa: E731
        a = np.ones((200001, 1), np.float32)
        b = a.copy()
        b[:10001] = 0  # 190,000 of 200,001 agree: 94.9995 %, reported as 95.0
        held, *_ = prune.agreement(rule, a, b)
        self.assertEqual(held["pct"], 95.0)
        self.assertFalse(prune.meets_agreement(held, 95.0))
        self.assertTrue(prune.meets_agreement(held, 94.99))
        exact, *_ = prune.agreement(rule, a[:200000], a[:200000])
        self.assertTrue(prune.meets_agreement(exact, 100.0))
        self.assertFalse(prune.meets_agreement({"rows": 0, "same": 0, "pct": None}, 0.0))

    def test_an_agreement_exactly_on_the_bar_meets_it_whatever_the_bar_is_in_binary(self):
        """`bar * rows` in floating point can round one ulp above the exact value, so 644 of 1000 (exactly 64.4 %) was refused by a 64.4 bar
        (optimize exited 2 printing "64.4000 % < bar 64.4"). The bar is compared as the decimal the user typed."""
        import prune

        for text, rows, same in (("64.4", 1000, 644), ("50.52", 2500, 1263), ("0.28", 2500, 7), ("95", 200, 190), ("99.9", 1000, 999)):
            self.assertEqual(Fraction(same, rows) * 100, Fraction(text), "not an exact tie")
            self.assertTrue(prune.meets_agreement({"rows": rows, "same": same}, float(text)), f"{same} of {rows} against {text}")
            self.assertFalse(prune.meets_agreement({"rows": rows, "same": same - 1}, float(text)), f"{same - 1} of {rows} against {text}")

    def test_optimize_refuses_a_run_that_misses_the_agreement_bar_by_less_than_the_rounding(self):
        import optimize
        import prune

        held = {"rows": 200001, "same": 190000, "pct": 95.0}
        fake = {"totals": {"nodes_before": 2, "nodes_after": 1}, "heldout": held, "disagreements": []}
        with tempfile.TemporaryDirectory() as t:
            with mock.patch.object(prune, "run", return_value=fake), mock.patch.object(prune, "summary", return_value="pruned"):
                rep = optimize.optimize("m", "d", Path(t) / "o", "10%", min_agreement=95.0)
        self.assertFalse(rep["ok"])
        self.assertIn("94.9995", rep["failures"][0])

    def test_a_bar_that_is_not_a_finite_number_is_refused_by_every_entry_point(self):
        import optimize
        import prune

        for bad in ("nan", "inf", "-inf"):
            for flag in ("--min-agreement", "--med-tol", "--max-tol", "--error-slack"):
                err = refused_stderr(optimize.main, ["m.json", "--data", "d.npz", f"{flag}={bad}"])
                self.assertIn("finite number", err, (flag, bad))
            for flag in ("--med-tol", "--max-tol", "--error-slack", "--tol"):
                err = refused_stderr(prune.main, ["m.json", "d.npz", "out", f"{flag}={bad}"])
                self.assertIn("finite number", err, (flag, bad))
        for name in ("median_pct", "max_pct", "error_slack"):
            with self.assertRaises(ValueError):
                qf.RegBar("raw", **{name: float("nan")})
        with self.assertRaises(ValueError):
            import optimize

            optimize.optimize("m", "d", "o", min_agreement=float("nan"))


def refused_stderr(main, argv) -> str:
    """What a command-line `main(argv)` prints to stderr when it refuses its arguments (it must exit, never run)."""
    import contextlib
    import io

    err = io.StringIO()
    with contextlib.redirect_stderr(err), mock.patch.object(sys, "argv", ["tool", *argv]):
        try:
            if main.__code__.co_argcount:
                main(argv)
            else:
                main()
        except SystemExit as e:
            assert e.code != 0, "the tool accepted the argument"
            return err.getvalue()
    raise AssertionError("the tool did not refuse the argument")


def cli_message(main, argv) -> str:
    """Runs a command's `main` with these arguments and returns what it says when it refuses them: the one-line message it exits with, or,
    for an argument the option parser refuses (exit 2) or a refusal that exits 3 (compile), the text on stderr. Anything else (a
    traceback, success) fails."""
    import contextlib
    import io

    err = io.StringIO()
    with contextlib.redirect_stderr(err), mock.patch.object(sys, "argv", ["tool", *argv]):
        try:
            if main.__code__.co_argcount:
                main(argv)
            else:
                main()
        except SystemExit as e:
            if isinstance(e.code, str):
                assert e.code.strip() and "\n" not in e.code.strip(), f"not a one-line message: {e.code!r}"
                return e.code
            assert e.code in (2, 3) and "Traceback" not in err.getvalue(), f"exit {e.code!r}: {err.getvalue()[-300:]}"
            return err.getvalue()
    raise AssertionError("the command accepted its arguments")


class RefusedInputsTest(unittest.TestCase):
    """A data file, an argument or a path that cannot be used ends in ONE line that names it (never a traceback, and never a PASS or an OK
    that tested or checked nothing)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.x = np.load(FIX / "x.npy")
        self.model = FIX / "xgb_reg.json"
        self.multi = FIX / "xgb_multi.ubj"
        import xgboost as xgb

        label = (np.nan_to_num(self.x[:, 0]) > 0).astype(int)
        self.binary = self.t / "binary.json"
        xgb.train({"objective": "binary:logistic", "max_depth": 2, "verbosity": 0}, xgb.DMatrix(self.x, label=label), 3).save_model(
            str(self.binary)
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_data_file_without_rows_or_with_text_in_x_is_refused_for_npz_and_npy_alike(self):
        for name, save, arr, word in (
            ("empty.npz", lambda p, a: np.savez(p, X=a), self.x[:0], "no rows"),
            ("empty.npy", np.save, self.x[:0], "no rows"),
            ("text.npz", lambda p, a: np.savez(p, X=a), np.array([["a", "b"]]), "numbers"),
            ("text.npy", np.save, np.array([["a", "b"]]), "numbers"),
            ("flat.npy", np.save, self.x[:, 0], "2-D"),
        ):
            save(self.t / name, arr)
            with self.assertRaises(ValueError, msg=name) as cm:
                qf.load_data(self.t / name)
            self.assertIn(word, str(cm.exception), name)
        for kind in (bool, np.int8, np.float16, np.float64):  # numbers of other widths are read as float32
            np.save(self.t / "wide.npy", (self.x[:5] > 0).astype(kind))
            self.assertEqual(qf.load_data(self.t / "wide.npy")["X"].dtype, np.float32)

    def test_a_report_path_that_cannot_be_written_is_refused_before_the_run_and_a_late_failure_is_one_line(self):
        """`check --json` and `compare --json/--md` write at the END of a long run: into a folder the user cannot write they used to print the
        verdict and then end in a traceback with exit 1 (on a PASS run too); `/dev/null` was `replace`d. Now the path is refused first, and a
        write that fails anyway says so in one line."""
        import os

        import check
        import compare

        art = flat_file(self.t)
        np.savez(self.t / "d.npz", X=self.x)
        ro = self.t / "ro"
        ro.mkdir()
        ro.chmod(0o555)
        self.addCleanup(lambda: ro.exists() and ro.chmod(0o755))
        targets = [("/dev/null", "not a regular file")] + ([(str(ro / "x.json"), "not writable")] if os.geteuid() else [])
        for target, word in targets:
            with self.assertRaises(ValueError, msg=target) as cm:
                qf.output_file(target, "--json")
            self.assertIn(word, str(cm.exception))
            self.assertIn(
                word,
                cli_message(check.main, [str(FIX / "lgb_plain.txt"), str(art), "--random", "20", "--json", target]),
            )
            for flag in ("--json", "--md"):
                msg = cli_message(
                    compare.main,
                    [
                        str(FIX / "lgb_plain.txt"),
                        "--data",
                        str(self.t / "d.npz"),
                        "--backends",
                        "flat",
                        "--work",
                        str(self.t / "w"),
                        flag,
                        target,
                    ],
                )
                self.assertIn(word, msg)
        self.assertFalse((self.t / "w").exists())  # refused before any work
        real = qf.replace_file

        def failing(dest, *a, **k):
            if str(dest).endswith("late.json"):
                raise PermissionError(13, "Permission denied", str(dest))
            return real(dest, *a, **k)

        with mock.patch.object(qf, "replace_file", failing):
            msg = cli_message(check.main, [str(FIX / "lgb_plain.txt"), str(art), "--random", "20", "--json", str(self.t / "late.json")])
        self.assertTrue(msg.startswith("check.py: PASS, but --json "), msg)

    def test_a_json_model_nested_thousands_of_levels_deep_is_refused_in_one_line(self):
        """json.loads ran out of recursion on `{"learner": [[[...` and every command ended in a RecursionError traceback."""
        import compile as qcompile
        import structure

        nest = self.t / "nest.json"
        nest.write_text('{"learner": ' + "[" * 100000 + "]" * 100000 + "}")
        for name, main, argv in (
            ("compile", qcompile.main, [str(nest), str(self.t / "o.flat")]),
            ("structure", structure.main, [str(nest), str(self.t / "o.npz")]),
        ):
            msg = cli_message(main, argv)
            self.assertIn("nested too deep", msg, name)

    def test_a_npy_whose_header_claims_more_data_than_the_file_holds_is_refused_in_one_line(self):
        """A 160-byte file claiming 10^12 float32 values ended in numpy's MemoryError traceback in prune, check, optimize and compare."""
        import struct

        import prune

        header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (1000000, 1000000), }"
        header += b" " * (64 - (10 + len(header) + 1) % 64) + b"\n"
        huge = self.t / "huge.npy"
        huge.write_bytes(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * 64)
        with self.assertRaises(ValueError) as cm:
            qf.load_data(huge)
        self.assertIn("header claims", str(cm.exception))
        self.assertIn("header claims", cli_message(prune.main, [str(self.model), str(huge), str(self.t / "out")]))
        archive = self.t / "huge.npz"  # the same claim inside an archive member
        import zipfile

        with zipfile.ZipFile(archive, "w") as z:
            z.writestr("X.npy", huge.read_bytes())
        with self.assertRaises(ValueError):
            qf.load_data(archive)

    def test_every_way_a_user_file_can_be_damaged_ends_in_the_one_line_refusal_at_the_three_entry_points(self):
        """A flipped byte in a compressed archive (zlib.error), an unsupported zip method (NotImplementedError), a .npy header that is
        never closed (tokenize.TokenError), a choice file nested 200,000 deep (RecursionError) or not text, and a model nested too deep for
        the streamed reader escaped the net of load_data / open_source / load_artifact and ended in a traceback."""
        import io
        import shutil
        import struct
        import zipfile

        import check
        import prune

        x = np.random.default_rng(1).normal(size=(400, 6)).astype(np.float32)  # the size at which a flipped byte breaks the deflate stream
        buf = io.BytesIO()
        np.savez_compressed(buf, X=x)
        flipped = bytearray(buf.getvalue())
        flipped[100] ^= 0xFF
        method = io.BytesIO()
        with zipfile.ZipFile(method, "w") as z:
            z.writestr("X.npy", b"")
        raw = bytearray(method.getvalue())
        for sig, off in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
            i = raw.find(sig)
            raw[i + off : i + off + 2] = struct.pack("<H", 99)
        header = b"{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4, }"
        header += b" " * (64 - (10 + len(header) + 1) % 64) + b"\n"
        unclosed = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + b"\0" * 48
        for name, blob in (("flipped.npz", bytes(flipped)), ("method99.npz", bytes(raw)), ("unclosed.npy", unclosed)):
            path = self.t / name
            path.write_bytes(blob)
            with self.assertRaises(ValueError, msg=name) as cm:
                qf.load_data(path)
            self.assertIn(str(path), str(cm.exception), name)
            self.assertNotIn("\n", str(cm.exception), name)
            msg = cli_message(prune.main, [str(self.model), str(path), str(self.t / f"out-{name}")])
            self.assertIn(name, msg)
            self.assertEqual(len(msg.splitlines()), 1, msg)
        art = flat_file(self.t)
        for name, body, word in (("deep", b"[" * 200000, "nested too deep"), ("binary", b"\xff\xfe\x00", "not valid text")):
            shutil.copy(art, self.t / f"{name}.flat")
            (self.t / f"{name}.flat.choice.json").write_bytes(body)
            with self.assertRaises(ValueError, msg=name) as cm:
                backends.load_artifact(self.t / f"{name}.flat")
            self.assertIn(word, str(cm.exception), name)
            self.assertIn(word, cli_message(check.main, [str(FIX / "lgb_plain.txt"), str(self.t / f"{name}.flat"), "--random", "20"]), name)
        deep = self.t / "deep.json"  # a model too deep for the streamed reader too (--memory-mb 1 sends a forest there)
        deep.write_text('{"learner": ' + "[" * 100000 + "]" * 100000 + "}")
        data = self.t / "x.npz"
        np.savez(data, X=x)
        msg = cli_message(prune.main, [str(deep), str(data), str(self.t / "out-deep"), "--memory-mb", "1"])
        self.assertEqual(len(msg.splitlines()), 1, msg)

    def test_the_shared_reading_net_does_not_hide_a_programming_error(self):
        """The net names the failures of READING a file; a TypeError raised by our own code inside it still shows its traceback."""
        with self.assertRaises(TypeError), qf.refusing_damaged(self.t / "x", "data file"):
            raise TypeError("a bug")
        with self.assertRaises(KeyError), qf.refusing_damaged(self.t / "x", "data file"):
            raise KeyError("a bug")

    def test_check_refuses_to_pass_with_no_rows_to_test_and_a_negative_row_count(self):
        import check

        art = flat_file(self.t)
        for kwargs, word in (({"n_random": 0}, "nothing to test"), ({"n_random": -5}, "cannot be negative")):
            with self.assertRaises(ValueError) as cm:
                check.run(FIX / "lgb_plain.txt", art, **kwargs)
            self.assertIn(word, str(cm.exception))
        np.savez(self.t / "empty.npz", X=self.x[:0])
        self.assertIn(
            "no rows", cli_message(check.main, [str(FIX / "lgb_plain.txt"), str(art), "--data", str(self.t / "empty.npz"), "--random", "0"])
        )
        self.assertIn("nothing to test", cli_message(check.main, [str(FIX / "lgb_plain.txt"), str(art), "--random", "0"]))
        self.assertIn("random", cli_message(check.main, [str(FIX / "lgb_plain.txt"), str(art), "--random", "-5"]))

    def test_optimize_keeping_everything_still_reads_the_data_and_checks_the_regression_bar(self):
        import optimize

        out = self.t / "o"
        np.savez(self.t / "d.npz", X=self.x)
        msg = cli_message(optimize.main, [str(self.model), "--data", str(self.t / "missing.npz"), "--keep", "100%", "--out", str(out)])
        self.assertIn("missing.npz", msg)
        self.assertFalse(out.exists() and any(out.iterdir()))
        msg = cli_message(
            optimize.main, [str(self.multi), "--data", str(self.t / "d.npz"), "--keep", "100%", "--regression", "log", "--out", str(out)]
        )
        self.assertIn("regression", msg)
        np.savez(self.t / "empty.npz", X=self.x[:0])
        self.assertIn(
            "no rows",
            cli_message(optimize.main, [str(self.model), "--data", str(self.t / "empty.npz"), "--keep", "100%", "--out", str(out)]),
        )

    def test_optimize_arguments_that_cannot_work_are_refused_before_any_work(self):
        import optimize

        np.savez(self.t / "d.npz", X=self.x)
        (self.t / "file").write_text("x")
        base = [str(self.binary), "--data", str(self.t / "d.npz")]
        self.assertIn("file", cli_message(optimize.main, [*base, "--out", str(self.t / "file")]))
        self.assertIn("nosuch", cli_message(optimize.main, [*base, "--backend", "nosuch", "--out", str(self.t / "o1")]))
        self.assertIn("nosuch", cli_message(optimize.main, [*base, "--backend", "nosuch", "--compile", "--out", str(self.t / "o2")]))
        self.assertIn("threshold", cli_message(optimize.main, [*base, "--decision", "threshold:abc", "--out", str(self.t / "o3")]))
        self.assertIn(
            "single output",
            cli_message(
                optimize.main,
                [str(self.multi), "--data", str(self.t / "d.npz"), "--decision", "threshold:0.5", "--out", str(self.t / "o4")],
            ),
        )
        self.assertEqual(sorted(p.name for p in self.t.iterdir() if p.name.startswith("o")), [])

    def test_prune_and_check_refuse_a_decision_rule_that_cannot_be_used_in_one_line(self):
        import prune

        np.savez(self.t / "d.npz", X=self.x)
        self.assertIn(
            "threshold",
            cli_message(prune.main, [str(self.binary), str(self.t / "d.npz"), str(self.t / "p1"), "--decision", "threshold:abc"]),
        )
        self.assertIn(
            "single output",
            cli_message(prune.main, [str(self.multi), str(self.t / "d.npz"), str(self.t / "p2"), "--decision", "threshold:0.5"]),
        )

    def test_an_output_path_or_a_rule_file_that_cannot_be_used_is_refused_in_one_line(self):
        import compare
        import compile
        import prune

        np.savez(self.t / "d.npz", X=self.x)
        (self.t / "file").write_text("x")
        for out in (self.t / "no-folder" / "m.flat", self.t, self.t / "file" / "m.flat"):
            with self.assertRaises(ValueError, msg=str(out)):
                compile.compile_model(self.model, out, backend="flat")
        self.assertIn("compile.py", cli_message(compile.main, [str(self.model), str(self.t / "no-folder" / "m.flat")]))
        self.assertIn("file", cli_message(prune.main, [str(self.binary), str(self.t / "d.npz"), str(self.t / "file")]))
        for rule in (str(self.t / "nosuch.py"), f"{self.t / 'nosuch.py'}:fn", str(self.t / "d.npz")):
            with self.assertRaises(ValueError, msg=rule):
                qf.make_rule(rule, 1, "sigmoid")
        (self.t / "rule.py").write_text("raise RuntimeError('broken rule')\n")
        with self.assertRaisesRegex(ValueError, "broken rule"):
            qf.make_rule(str(self.t / "rule.py"), 1, "sigmoid")
        (self.t / "rule2.py").write_text("x = 1\n")
        with self.assertRaisesRegex(ValueError, "no function decide"):
            qf.make_rule(str(self.t / "rule2.py"), 1, "sigmoid")
        for flag, bad in (("--random", "-5"), ("--eq-rows", "0"), ("--speed-rows", "-1")):
            self.assertIn(
                "minimum",
                cli_message(
                    compare.main, [str(self.binary), "--data", str(self.t / "d.npz"), "--work", str(self.t / "w"), f"{flag}={bad}"]
                ),
            )

    def test_a_pipe_or_a_device_as_the_model_path_is_refused_not_listed(self):
        fifo = self.t / "model.json"
        os.mkfifo(fifo)
        for path in (fifo, Path("/dev/null")):
            with self.assertRaises(ValueError) as cm:
                qf_formats.model_files(path)
            self.assertIn(str(path), str(cm.exception))
            self.assertIsInstance(cm.exception, SystemExit)


def three_flat_files(t: Path) -> dict:
    """An xgboost binary model, a LightGBM binary model with `sigmoid:1.7` and a 2-fold xgboost softmax ensemble, compiled to flat files
    (the three whose header fields were once changed silently): {name: path}. Empty when xgboost or LightGBM is missing."""
    try:
        import lightgbm as lgb
        import xgboost as xgb
    except ImportError:
        return {}
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 5)).astype(np.float32)
    binary, three = (x[:, 0] > 0).astype(int), np.digitize(x[:, 0], [-0.5, 0.5])
    (t / "folds").mkdir()
    for k in range(2):
        xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 3}, xgb.DMatrix(x, three), 4 + k).save_model(
            str(t / "folds" / f"fold{k}.ubj")
        )
    xgb.train({"objective": "binary:logistic", "max_depth": 3}, xgb.DMatrix(x, binary), 5).save_model(str(t / "bin.ubj"))
    lgb.train({"objective": "binary", "sigmoid": 1.7, "verbose": -1, "num_leaves": 5}, lgb.Dataset(x, binary), 5).save_model(
        str(t / "lgb.txt")
    )
    out = {}
    for name, model, average in (
        ("xgboost binary", t / "bin.ubj", "output"),
        ("LightGBM sigmoid:1.7", t / "lgb.txt", "output"),
        ("2-fold xgboost softmax", t / "folds", "output"),
    ):
        out[name] = t / (name.replace(" ", "_").replace(":", "") + ".flat")
        backends.get("flat").compile(qf.load_forest(qf_formats.model_files(model), average), out[name], {"no_lib": True})
    return out


class FlatHeaderIsCheckedTest(unittest.TestCase):
    """The checksum covers the whole file. It used to cover the bytes after the header only, so a bit flipped in the objective, the average
    rule, the flags or the margin scale (each valid over a range of values) loaded and answered differently: sigmoid became identity and the
    file returned raw margins where it should return probabilities."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.files = three_flat_files(self.t)
        if not self.files:
            self.skipTest("no xgboost or lightgbm")

    def tearDown(self):
        self.tmp.cleanup()

    def test_no_single_flipped_bit_of_the_header_of_three_files_loads(self):
        for name, path in self.files.items():
            good = path.read_bytes()
            evaluator.FlatForest(path).close()  # the unchanged file loads
            bad = self.t / "bad.flat"
            bad.write_bytes(good)
            with open(
                bad, "r+b"
            ) as fh:  # one byte changed in place (no new file, no truncation: 512 files per model would be slow on a busy disk)
                for byte in range(64):
                    for bit in range(8):
                        fh.seek(byte)
                        fh.write(bytes([good[byte] ^ (1 << bit)]))
                        fh.flush()
                        with self.assertRaises(ValueError, msg=f"{name}: byte {byte} bit {bit}"):
                            evaluator.FlatForest(bad)
                    fh.seek(byte)
                    fh.write(good[byte : byte + 1])
                    fh.flush()
            evaluator.FlatForest(bad).close()  # and the file is back to what it was

    def test_the_old_objective_to_identity_edit_is_refused_and_a_repaired_checksum_loads_a_forest_that_answers_differently(self):
        """The objective byte changed from sigmoid to identity: refused. With the checksum put right by hand it loads, and then it IS another
        forest: that is what the fuzz and mutation oracles count apart."""
        path = self.files["xgboost binary"]
        good = path.read_bytes()
        x = np.random.default_rng(1).normal(size=(30, 5)).astype(np.float32)
        raw = bytearray(good)
        raw[32] = 0
        (self.t / "id.flat").write_bytes(raw)
        with self.assertRaises(ValueError) as cm:
            evaluator.FlatForest(self.t / "id.flat")
        self.assertIn("checksum", str(cm.exception))
        (self.t / "id.flat").write_bytes(evaluator.seal(raw))
        sealed, original = evaluator.FlatForest(self.t / "id.flat"), evaluator.FlatForest(path)
        self.assertFalse(np.array_equal(sealed.predict(x), original.predict(x)))
        sealed.close()
        original.close()

    def test_a_file_of_the_first_layout_is_refused_with_a_line_that_says_to_compile_again(self):
        """Formats 1 to 3 had a checksum of the body only; there is no reader for them."""
        import struct

        for name, path in self.files.items():
            raw = bytearray(path.read_bytes())
            struct.pack_into("<I", raw, 8, struct.unpack_from("<I", raw, 8)[0] - 3)
            raw[40:48] = struct.pack("<Q", evaluator.fnv1a(bytes(raw[64:])))  # the checksum the first layout wrote
            old = self.t / "old.flat"
            old.write_bytes(raw)
            with self.assertRaises(ValueError) as cm:
                evaluator.FlatForest(old)
            self.assertIn("compile the model again", str(cm.exception), name)
            self.assertNotIn("\n", str(cm.exception).strip(), name)

    def test_the_margin_scale_is_read_from_the_file_the_library_verified(self):
        pred = evaluator.FlatForest(self.files["LightGBM sigmoid:1.7"])
        self.assertEqual(pred.margin_scale, 1.7)
        pred.close()
        pred = evaluator.FlatForest(self.files["xgboost binary"])
        self.assertEqual(pred.margin_scale, 1.0)
        pred.close()
        self.assertFalse(hasattr(evaluator, "file_margin_scale"), "no second, unverified read of the header is left")


class FuzzHarnessTest(unittest.TestCase):
    def test_stacked_mutations_on_a_file_with_a_huge_node_count_end_quickly(self):
        """A header count mutated to ~4e9 must not make the loops over it run in Python for ever (the fuzz run saw 39 false HANGs,
        deterministic by seed): the loops are bounded by the file's own size."""
        import struct

        sys.path.insert(0, str(HERE / "fuzz"))
        import fuzz_loader as fz

        with tempfile.TemporaryDirectory() as t:
            data = bytearray(flat_file(Path(t)).read_bytes())
        struct.pack_into("<I", data, fz.OFF["n_nodes"], 0xFFFFFFFE)
        t0 = time.perf_counter()
        for seed in range(30):
            for kind in ("child", "nan-thr", "feature"):
                fz.mutate(bytes(data), kind, np.random.default_rng([seed, fz.KINDS.index(kind)]))
        self.assertLess(time.perf_counter() - t0, 20)

    def test_a_loaded_mutant_is_clean_only_when_it_answers_like_the_undamaged_file(self):
        """The oracle of the flat target used to count every file that loaded and gave the same bits on all engines as clean, so a header
        field changed to another valid value (the objective, the average rule, the flags, the margin scale) went through. A mutant with
        its checksum repaired may load a different forest (counted apart); with the original checksum it may not load at all."""
        import struct

        sys.path.insert(0, str(HERE / "fuzz"))
        import fuzz_loader as fz

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            good = flat_file(t).read_bytes()
            nf = evaluator.FlatForest(t / "m.flat").n_features
            x = rng.normal(size=(40, nf)).astype(np.float32)
            bm = rng.uniform(-1, 1, (40, 1))

            def loaded(data, name):
                (t / name).write_bytes(data)
                return evaluator.FlatForest(t / name)

            original = evaluator.FlatForest(t / "m.flat")
            self.assertIsNone(fz.loaded_verdict(good, good, original, original, x, bm), "the same file is clean")
            # a changed file that still has the original checksum is a damaged file accepted (the loader refuses it; the oracle says so too)
            damaged = bytearray(good)
            damaged[-1] ^= 1
            self.assertIn("DIVERGE", fz.loaded_verdict(bytes(damaged), good, original, original, x, bm))
            # the header's margin scale bytes are valid over a range: a repaired checksum loads another forest, which is not clean
            other = bytearray(good)
            n_folds, n_groups = struct.unpack_from("<II", other, 20)[0], struct.unpack_from("<I", other, 16)[0]
            base_at = fz.HEADER + 8 * n_folds
            struct.pack_into("<f", other, base_at, struct.unpack_from("<f", other, base_at)[0] + 5.0)
            other = evaluator.seal(other)
            self.assertIn("different forest", fz.loaded_verdict(other, good, loaded(other, "o.flat"), original, x, bm))
            # padding after the base margins is not read: a repaired file that answers bit for bit like the original is clean
            if (n_folds * n_groups * 4) % 8:
                padded = bytearray(good)
                padded[base_at + n_folds * n_groups * 4] ^= 0x55
                padded = evaluator.seal(padded)
                self.assertIsNone(fz.loaded_verdict(padded, good, loaded(padded, "p.flat"), original, x, bm))
            original.close()

    def test_a_loaded_mutant_that_refuses_a_base_margin_with_a_message_is_a_different_forest_not_unclean(self):
        """A header whose margin scale reads NaN and whose checksum was repaired loads; the library then refuses a base margin with a
        ValueError ("the scales differ, a base margin is ambiguous"). That is a clean refusal of a different forest: the flat target used
        to label the case "UNCLEAN after load" (60 cases of the full run), in the engine loop and in `loaded_verdict`."""
        import struct

        sys.path.insert(0, str(HERE / "fuzz"))
        import fuzz_loader as fz

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            good = flat_file(t).read_bytes()
            nan_scale = bytearray(good)
            struct.pack_into("<d", nan_scale, 56, float("nan"))
            nan_scale = evaluator.seal(nan_scale)
            (t / "n.flat").write_bytes(nan_scale)
            fo = evaluator.FlatForest(t / "n.flat")
            original = evaluator.FlatForest(t / "m.flat")
            x = np.random.default_rng(0).normal(size=(10, fo.n_features)).astype(np.float32)
            bm = np.random.default_rng(1).uniform(-1, 1, (10, fo.n_groups))
            with self.assertRaises(ValueError):  # the premise: this file loads and refuses a base margin
                fo.predict(x, 1, base_margin=bm, dtype=np.float64)
            self.assertIn("different forest", fz.loaded_verdict(nan_scale, good, fo, original, x, bm))
            fo.close()
            original.close()
            # the whole flat child: the same case ends as a different forest, not as an unclean one
            (t / "bases").mkdir()
            (t / "tmp").mkdir()
            (t / "bases" / "m.flat").write_bytes(good)
            results = []
            with mock.patch.object(fz, "mutate", return_value=nan_scale):
                fz.child_flat(["flat:m:hdr:1"], t, lambda *parts: results.append(parts))
            self.assertEqual([r[2] for r in results if r[0] == "RESULT"], ["LOADED a different forest (checksum repaired)"])

    def test_a_library_that_crashes_on_a_text_file_is_counted_apart_and_the_next_file_is_probed(self):
        """QF's parser can accept a damaged text model that xgboost or LightGBM then aborts on (SIGABRT, SIGSEGV): the fuzz target probes the
        library in a process of its own, answers "SOURCE LIBRARY CRASH", which is not a failure, and the next file gets a fresh process."""
        sys.path.insert(0, str(HERE / "fuzz"))
        import fuzz_loader as fz

        aborting = "import os, sys\nfor line in sys.stdin:\n    os.abort()\n"
        probe = fz.SourceProbe(aborting)
        try:
            self.assertEqual(probe.check("a.txt"), "SOURCE LIBRARY CRASH (exit -6, signal SIGABRT)")
            self.assertEqual(probe.check("b.txt"), "SOURCE LIBRARY CRASH (exit -6, signal SIGABRT)")  # a new process each time
        finally:
            probe.close()
        silent = fz.SourceProbe("import sys\nfor line in sys.stdin:\n    pass\n", timeout=0.5)
        try:
            self.assertTrue(silent.check("c.txt").startswith("HANG"))
        finally:
            silent.close()
        self.assertEqual(fz.count_non_clean({"SOURCE LIBRARY CRASH": 5, "LOADED": 3}), 0)
        probe = fz.SourceProbe()  # the real probe: a good file is OK, a damaged one a clean refusal with the library's own message
        try:
            self.assertEqual(probe.check(FIX / "lgb_plain.txt"), "OK")
            with tempfile.TemporaryDirectory() as t:
                bad = Path(t) / "bad.txt"
                bad.write_text("tree\nnot a model\n")
                self.assertTrue(probe.check(bad).startswith("CLEAN"), probe.check(bad))
        finally:
            probe.close()

    def test_a_run_with_unclean_or_diverging_results_fails(self):
        """The summary keys are the head of a verdict line ("UNCLEAN KeyError: ..." is counted under "UNCLEAN KeyError"); the exit code
        used to compare them with "UNCLEAN" as a whole, so those results never failed the run."""
        sys.path.insert(0, str(HERE / "fuzz"))
        import fuzz_loader as fz

        for key in ("UNCLEAN KeyError", "DIVERGE engine 1 differs from the walker", "DIVERGE", "CRASH", "HANG", "UNCLEAN after load"):
            self.assertEqual(fz.count_non_clean({key: 3, "CLEAN ValueError": 100, "LOADED": 50}), 3, key)
        self.assertEqual(fz.count_non_clean({"CLEAN ValueError": 100, "LOADED": 50, "LOADED a different forest": 7}), 0)

    def test_the_header_bit_flip_kind_never_yields_a_file_that_loads(self):
        sys.path.insert(0, str(HERE / "fuzz"))
        import fuzz_loader as fz

        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            good = flat_file(t).read_bytes()
            (t / "x.flat").write_bytes(good)
            for seed in range(200):
                data = fz.mutate(good, "hdr-flip", np.random.default_rng([seed, fz.KINDS.index("hdr-flip")]))
                if data == good:  # two flips of the same bit
                    continue
                with open(t / "x.flat", "r+b") as fh:  # same length: rewritten in place
                    fh.write(data)
                with self.assertRaises(ValueError, msg=f"seed {seed}"):
                    evaluator.FlatForest(t / "x.flat")


class ValidatedInputsTest(unittest.TestCase):
    """The input layer: one place says what a data file, a decision rule or a path may hold, and every command that reads them refuses the
    rest in one line (tests/test_input_matrix.py runs the whole table of commands and damages)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.t = Path(self.tmp.name)
        self.x = np.load(FIX / "x.npy")

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_truth_or_index_array_holding_nan_or_infinity_is_refused_by_name(self):
        for name in ("y", "fold", "group", "slice", "base_margin"):
            for bad in (np.nan, np.inf, -np.inf):
                col = np.zeros(len(self.x))
                col[3] = bad
                np.savez(self.t / "d.npz", X=self.x, **{name: col})
                with self.subTest(name=name, bad=bad), self.assertRaisesRegex(ValueError, f"{name} holds 1 NaN or infinite"):
                    qf.load_data(self.t / "d.npz")
        for name in ("y", "fold", "group", "slice", "eligible", "base_margin"):
            np.savez(self.t / "d.npz", X=self.x, **{name: np.array(["a"] * len(self.x))})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"{name} must hold numbers"):
                qf.load_data(self.t / "d.npz")

    def test_one_nan_in_the_truth_cannot_pass_the_regression_bar(self):
        """`e_pruned > e_full` is False when both are NaN, so a NaN in the truth made a failing error bar pass."""
        full = np.linspace(0, 1, 50, dtype=np.float32).reshape(-1, 1)
        pruned = full + np.float32(0.5)
        y = full[:, 0].astype(np.float64)
        bar = qf.RegBar("raw", 1e9, 1e9, 0.0)
        self.assertFalse(bar.check(full, pruned, y)["ok"])
        y[0] = np.nan
        self.assertFalse(bar.check(full, pruned, y)["ok"])
        nan_out = pruned.copy()
        nan_out[0] = np.nan  # a model that answers NaN fails every bar too
        self.assertFalse(qf.RegBar("raw", 1e9, 1e9, 0.0).check(full, nan_out)["ok"])

    def test_a_decision_threshold_that_is_not_finite_is_refused(self):
        """`p >= nan` is False for every row and `p >= -inf` True: the decisions are one class, so the agreement is 100 % by construction."""
        for text in ("threshold:nan", "threshold:inf", "threshold:-inf", "threshold:1e999"):
            with self.subTest(text), self.assertRaisesRegex(ValueError, "finite"):
                qf.make_rule(text, 1, "sigmoid")
        self.assertEqual(qf.make_rule("threshold:0.5", 1, "sigmoid")(np.array([[0.2], [0.7]])).tolist(), [0, 1])

    def test_data_of_another_width_than_the_model_is_refused_in_one_line_by_every_command(self):
        import check
        import compare
        import optimize
        import prune

        art = flat_file(self.t)  # the LightGBM fixture: 7 features
        for width in (6, 8):
            np.savez(self.t / f"w{width}.npz", X=np.zeros((50, width), np.float32))
            data = str(self.t / f"w{width}.npz")
            for model in (FIX / "lgb_plain.txt", FIX / "xgb_reg.json"):
                for main, argv in (
                    (prune.main, [str(model), data, str(self.t / f"p{width}{model.stem}")]),
                    (optimize.main, [str(model), "--data", data, "--out", str(self.t / f"o{width}{model.stem}")]),
                    (optimize.main, [str(model), "--data", data, "--keep", "100%", "--out", str(self.t / f"k{width}{model.stem}")]),
                    (compare.main, [str(model), "--data", data, "--work", str(self.t / f"c{width}{model.stem}")]),
                ):
                    with self.subTest(model=model.name, width=width, command=main.__module__):
                        self.assertRegex(cli_message(main, argv), rf"X has {width} columns, the model has 7 features")
            msg = cli_message(check.main, [str(FIX / "lgb_plain.txt"), str(art), "--data", data, "--random", "10"])
            self.assertRegex(msg, rf"X has {width} columns, the model has 7 features")
        for source in (qf_formats.open_source(FIX / "lgb_plain.txt"), qf_formats.open_source(FIX / "xgb_reg.json")):  # the Python API too
            for method in (source.predict, source.leaf_matrix):
                with self.assertRaisesRegex(ValueError, "the model has 7 features"):
                    method(np.zeros((3, 6), np.float32))

    def test_compile_refuses_an_output_path_that_is_not_utf8(self):
        """The evaluator opens UTF-8 paths only: compile wrote `caf\\xe9/m.flat` and then check and optimize --verify could not open it."""
        import compile

        folder = os.fsdecode(os.fsencode(self.t) + b"/caf\xe9")
        os.mkdir(folder)
        with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
            compile.compile_model(FIX / "xgb_reg.json", Path(folder) / "m.flat", backend="flat")
        self.assertEqual(os.listdir(folder), [])
        self.assertIn("UTF-8", cli_message(compile.main, [str(FIX / "xgb_reg.json"), str(Path(folder) / "m.flat")]))


if __name__ == "__main__":
    unittest.main()
