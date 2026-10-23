"""What QuickForest writes must be what a user reads, for models trained the way people really train them.

Two classes of failure, one file:

* A model file that `prune` or `optimize` writes is reloaded by its OWN library and predicted through the DEFAULT path a user would call
  (xgboost: `Booster.predict` without `iteration_range` and the scikit-learn wrapper's `load_model` + `predict`; LightGBM:
  `Booster(model_file).predict`; YDF: `load_model().predict`), then compared with QuickForest's own answer for the pruned forest. Every path a
  `report.json` names exists, and every artifact `compile`, `choose` and `optimize` write is judged by `check` with the command
  docs/REFERENCE.md shows (no `--backend`).
* The models are drawn the way they are really trained: xgboost on a pandas DataFrame (feature names and types stored), the scikit-learn
  wrappers with early stopping (so `best_iteration` is set), LightGBM on a DataFrame with a pandas category column.

A case whose library or helper (pandas, scikit-learn, ydf, a Rust toolchain) is missing is skipped with the reason printed, never dropped.

    python3 -m unittest discover -s tests -p "test_written_files.py"
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import backend_codegen
import backends
import check
import compile
import optimize
import prune
import qf_common as qf
import qf_formats

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None
try:
    import ydf
except ImportError:  # pragma: no cover
    ydf = None
try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None
try:
    import sklearn  # noqa: F401  (the scikit-learn wrappers of xgboost and LightGBM need it)

    HAVE_SKLEARN = True
except ImportError:  # pragma: no cover
    HAVE_SKLEARN = False

QUICKFOREST = str(SCRIPTS / "quickforest")
N, NF = 1500, 6
NAMES = [f"feat {i}" + ("\u00e9" if i == 0 else "") for i in range(NF)]  # a space and a non-ASCII letter, as a DataFrame brings


def have_rustc() -> bool:
    try:
        backend_codegen.toolchain_bin()
        return True
    except Exception:
        return False


def make_rows():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(N, NF)).astype(np.float32)
    score = x[:, 0] + 0.8 * x[:, 1] * x[:, 2] - x[:, 3] + rng.normal(scale=1.2, size=N)
    return x, score


def qf_answer(model, art: Path, x):
    """QuickForest's own answer for a model file or folder: compile it, load it back, predict."""
    compile.compile_model(model, art, backend="flat")
    return backends.load_artifact(art).predict(x)


def paths_in(obj, out=None):
    """Every absolute path a JSON report names."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for v in obj.values():
            paths_in(v, out)
    elif isinstance(obj, list):
        for v in obj:
            paths_in(v, out)
    elif isinstance(obj, str) and os.path.isabs(obj) and "\n" not in obj:
        out.append(obj)
    return out


class Reload:
    """The DEFAULT predict path a user calls on a model file, per library. Each returns QuickForest-shaped answers: (rows,) for one output."""

    @staticmethod
    def xgb_booster(path, x, names=None):
        b = xgb.Booster()
        b.load_model(str(path))
        dm = xgb.DMatrix(pd.DataFrame(x, columns=names) if names else x)  # no iteration_range: what Booster.predict(dm) gives
        return b.predict(dm)

    @staticmethod
    def xgb_wrapper(path, x, kind, names=None):
        m = {"classifier": xgb.XGBClassifier, "regressor": xgb.XGBRegressor}[kind]()
        m.load_model(str(path))
        data = pd.DataFrame(x, columns=names) if names else x
        if kind == "regressor":
            return m.predict(data)
        p = m.predict_proba(data)
        return p[:, 1] if p.shape[1] == 2 else p

    @staticmethod
    def lgb_booster(path, x):
        return lgb.Booster(model_file=str(path)).predict(x)

    @staticmethod
    def ydf_model(path, x):
        return np.asarray(ydf.load_model(str(path)).predict({f"f{i}": x[:, i] for i in range(x.shape[1])}))


@unittest.skipUnless(xgb is not None, "no xgboost")
class XgboostFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        x, score = make_rows()
        cls.x = x
        cls.yb = (score > np.median(score)).astype(int)
        cls.y3 = np.digitize(score, np.quantile(score, [0.33, 0.66]))
        cls.reg = score.astype(np.float32)
        np.savez(cls.t / "d.npz", X=x)
        tr, va = slice(0, 1000), slice(1000, N)
        base = {"max_depth": 3, "eta": 0.4, "nthread": 1, "verbosity": 0, "tree_method": "hist"}
        # (1) xgb.train with a validation set and early stopping: best_iteration / best_score are saved, rounds past the best are kept
        b = xgb.train(
            {**base, "objective": "binary:logistic"},
            xgb.DMatrix(x[tr], cls.yb[tr]),
            200,
            evals=[(xgb.DMatrix(x[va], cls.yb[va]), "v")],
            early_stopping_rounds=8,
            verbose_eval=False,
        )
        assert b.best_iteration + 1 < b.num_boosted_rounds()
        b.save_model(str(cls.t / "es_binary.json"))
        b = xgb.train(
            {**base, "objective": "multi:softprob", "num_class": 3},
            xgb.DMatrix(x[tr], cls.y3[tr]),
            200,
            evals=[(xgb.DMatrix(x[va], cls.y3[va]), "v")],
            early_stopping_rounds=8,
            verbose_eval=False,
        )
        assert b.best_iteration + 1 < b.num_boosted_rounds()
        b.save_model(str(cls.t / "es_multi.ubj"))
        cls.cases = {
            "es_binary.json": ("booster", {"binary": True}),
            "es_multi.ubj": ("booster", {}),
        }
        if pd is not None:
            # (2) trained on a DataFrame (feature names and types stored in the file)
            frame = pd.DataFrame(x, columns=NAMES)
            b = xgb.train({**base, "objective": "binary:logistic"}, xgb.DMatrix(frame[tr], cls.yb[tr]), 60)
            assert b.feature_names == NAMES
            b.save_model(str(cls.t / "frame_binary.json"))
            cls.cases["frame_binary.json"] = ("booster", {"binary": True, "names": NAMES})
            if HAVE_SKLEARN:
                # (3) the scikit-learn wrappers with early stopping, saved through the wrapper (feature names, scikit_learn meta, best_iteration)
                clf = xgb.XGBClassifier(
                    n_estimators=200,
                    early_stopping_rounds=8,
                    n_jobs=1,
                    verbosity=0,
                    **{k: v for k, v in base.items() if k in ("max_depth", "tree_method")},
                    learning_rate=0.4,
                )
                clf.fit(frame[tr], cls.yb[tr], eval_set=[(frame[va], cls.yb[va])], verbose=False)
                assert clf.best_iteration + 1 < clf.get_booster().num_boosted_rounds()
                clf.save_model(str(cls.t / "skl_classifier.json"))
                reg = xgb.XGBRegressor(n_estimators=200, early_stopping_rounds=8, n_jobs=1, verbosity=0, max_depth=3, learning_rate=0.4)
                reg.fit(frame[tr], cls.reg[tr], eval_set=[(frame[va], cls.reg[va])], verbose=False)
                assert reg.best_iteration + 1 < reg.get_booster().num_boosted_rounds()
                reg.save_model(str(cls.t / "skl_regressor.ubj"))
                cls.cases["skl_classifier.json"] = ("classifier", {"binary": True, "names": NAMES})
                cls.cases["skl_regressor.ubj"] = ("regressor", {"names": NAMES})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def read_back(self, pruned: Path, name: str, kind: str, opts: dict, art: Path):
        """The pruned file through xgboost's `Booster.predict` (no iteration_range) against QuickForest's answer for it. Returns that answer."""
        f = pruned / name
        want = qf_answer(f, art, self.x)
        np.testing.assert_array_equal(
            Reload.xgb_booster(f, self.x, opts.get("names")),
            want,
            f"{name}: Booster.predict (no iteration_range) differs from the compiled pruned forest",
        )
        return want

    def read_back_with_the_wrapper(self, pruned: Path, name: str, kind: str, opts: dict, want):
        """The same file through the scikit-learn wrapper's `load_model` + `predict` (a classifier for a binary objective)."""
        if kind == "booster" and not opts.get("binary"):
            return  # a multiclass Booster-trained file has no wrapper meta: nothing a user would load that way
        got = Reload.xgb_wrapper(pruned / name, self.x, "regressor" if kind == "regressor" else "classifier", opts.get("names"))
        np.testing.assert_array_equal(got, want, f"{name}: the scikit-learn wrapper's predict differs from the compiled pruned forest")

    def pruned_and_read(self, keep: str, name: str, kind: str, opts: dict, wrapper: bool):
        out = self.t / f"pruned-{keep[:-1]}-{name}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # the ORIGINAL early-stopped model warns, as it should
            prune.run(self.t / name, self.t / "d.npz", out, keep)
        attrs = json.loads((out / name).read_text() if name.endswith(".json") else _ubj(out / name))["learner"].get("attributes", {})
        for key in ("best_iteration", "best_score", "best_ntree_limit"):
            self.assertNotIn(key, attrs, f"the pruned file still carries {key} of the original model")
        if opts.get("names"):
            b = xgb.Booster()
            b.load_model(str(out / name))
            self.assertEqual(b.feature_names, opts["names"], "the pruned file renamed the features")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the pruned file is not "saved after early stopping" any more
            want = self.read_back(out, name, kind, opts, self.t / f"{keep[:-1]}-{name}.flat")
        if wrapper:
            self.read_back_with_the_wrapper(out, name, kind, opts, want)

    def test_every_pruned_xgboost_file_predicts_like_the_pruned_forest_through_booster_predict(self):
        """An early-stopped model keeps `best_iteration` / `best_score`; the pruned file has fewer rounds, so a predict that honours
        them either raised or silently used a prefix of the pruned rounds, a third model nobody measured. Also: the feature names survive."""
        for keep in ("50%", "90%"):
            for name, (kind, opts) in self.cases.items():
                with self.subTest(model=name, keep=keep):
                    self.pruned_and_read(keep, name, kind, opts, wrapper=False)

    @unittest.skipUnless(pd is not None and HAVE_SKLEARN, "no pandas / scikit-learn: the wrapper cases are skipped")
    def test_every_pruned_xgboost_file_predicts_like_the_pruned_forest_through_the_scikit_learn_wrapper(self):
        for keep in ("50%", "90%"):
            for name, (kind, opts) in self.cases.items():
                with self.subTest(model=name, keep=keep):
                    self.pruned_and_read(keep, name, kind, opts, wrapper=True)

    def test_optimize_writes_the_same_readable_pruned_files(self):
        """The `pruned/` folder of `optimize` is read back by the library exactly as `prune`'s files are."""
        for name, (kind, opts) in self.cases.items():
            with self.subTest(model=name):
                out = self.t / f"opt-{name}"
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    rep = optimize.optimize(self.t / name, self.t / "d.npz", out, "50%", True, True, min_agreement=0.0, backend="flat")
                self.assertTrue(rep["ok"], rep["failures"])
                want = self.read_back(out / "pruned", name, kind, opts, self.t / f"opt-{name}.check.flat")
                if pd is not None and HAVE_SKLEARN:
                    self.read_back_with_the_wrapper(out / "pruned", name, kind, opts, want)
                # the compiled forest `optimize` kept answers the same as the one rebuilt from the pruned file
                np.testing.assert_array_equal(backends.load_artifact(out / "model.flat").predict(self.x), want)

    def test_every_path_a_report_names_exists(self):
        """`optimize` stores the artifact path while the artifact is in `_work/`, then moves the folder: the report named a dead path."""
        name = "es_binary.json"
        ok = self.t / "rep-ok"
        optimize.optimize(self.t / name, self.t / "d.npz", ok, "50%", True, True, min_agreement=0.0, backend="flat")
        refused = self.t / "rep-refused"
        r = optimize.optimize(self.t / name, self.t / "d.npz", refused, "50%", True, False, min_agreement=100.5, backend="flat")
        self.assertFalse(r["ok"])
        for folder in (ok, ok / "pruned", refused):
            report = json.loads((folder / "report.json").read_text())
            for p in paths_in(report):
                self.assertTrue(os.path.exists(p), f"{folder / 'report.json'} names {p}, which does not exist")
        stages = json.loads((ok / "report.json").read_text())["stages"]
        self.assertEqual(Path(stages["compile"]["artifact"]), ok / "model.flat")
        for m in json.loads((ok / "pruned" / "report.json").read_text())["models"]:
            self.assertTrue((ok / "pruned" / m["file"]).is_file())
        # a bar that fails AFTER the compile deletes the artifact: the report must not name it
        bad = self.t / "rep-verify-refused"
        with mock.patch.object(
            check,
            "run",
            return_value={"ok": False, "sets": [{"set": "data", "rows": 10, "max_err": 1.0, "decision_mismatches": 3, "ok": False}]},
        ):
            r = optimize.optimize(self.t / name, self.t / "d.npz", bad, "50%", True, True, min_agreement=0.0, backend="flat")
        self.assertFalse(r["ok"])
        for p in paths_in(json.loads((bad / "report.json").read_text())):
            self.assertTrue(os.path.exists(p), f"a refused run names {p}, which was deleted with the work folder")


def _ubj(path: Path) -> str:
    """The JSON text of a .ubj model (xgboost re-saves it as JSON)."""
    b = xgb.Booster()
    b.load_model(str(path))
    return bytes(b.save_raw("json")).decode()


@unittest.skipUnless(lgb is not None, "no lightgbm")
class LightgbmFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        x, score = make_rows()
        cls.x = x
        cls.yb = (score > np.median(score)).astype(int)
        np.savez(cls.t / "d.npz", X=x)
        tr, va = slice(0, 1000), slice(1000, N)
        p = {"objective": "binary", "num_leaves": 7, "learning_rate": 0.4, "verbose": -1, "num_threads": 1, "min_data_in_leaf": 5}
        b = lgb.train(
            p,
            lgb.Dataset(x[tr], cls.yb[tr]),
            200,
            valid_sets=[lgb.Dataset(x[va], cls.yb[va])],
            callbacks=[lgb.early_stopping(8, verbose=False)],
            keep_training_booster=True,  # else LightGBM cuts the returned booster to best_iteration and there is nothing past it to save
        )
        assert b.best_iteration < b.current_iteration()  # LightGBM's best_iteration counts from 1
        b.save_model(str(cls.t / "es_all_rounds.txt"), num_iteration=-1)  # every round, past the best (the "uncut" file)
        b.save_model(str(cls.t / "es_best.txt"))  # LightGBM's default: the rounds up to best_iteration
        cls.cases = ["es_all_rounds.txt", "es_best.txt"]
        cls.codes = x
        if pd is not None and HAVE_SKLEARN:
            frame = pd.DataFrame(x, columns=NAMES)
            frame["colour"] = pd.Categorical(np.random.default_rng(3).choice(["red", "green", "blue", "grey"], size=N))
            codes = frame["colour"].cat.codes.to_numpy().astype(np.float32)
            xc = np.column_stack([x, codes]).astype(np.float32)
            cls.xc = xc
            yc = (score + (codes == 1) * 1.5 > np.median(score)).astype(int)
            clf = lgb.LGBMClassifier(n_estimators=200, num_leaves=7, learning_rate=0.4, verbose=-1, n_jobs=1, min_child_samples=5)
            clf.fit(
                frame[tr],
                yc[tr],
                eval_set=[(frame[va], yc[va])],
                callbacks=[lgb.early_stopping(8, verbose=False)],
                categorical_feature=["colour"],
            )
            assert (
                clf.best_iteration_ > 0
            )  # the wrapper cuts its booster to the best rounds: the file holds those (LightGBM's own behaviour)
            clf.booster_.save_model(str(cls.t / "skl_frame_category.txt"))
            assert "pandas_categorical:[[" in (cls.t / "skl_frame_category.txt").read_text()
            np.savez(cls.t / "dc.npz", X=xc)
            cls.cases.append("skl_frame_category.txt")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_pruned_lightgbm_file_predicts_like_the_pruned_forest_through_the_default_path(self):
        for name in self.cases:
            x, data = (self.xc, "dc.npz") if "category" in name else (self.x, "d.npz")
            for keep in ("50%", "90%"):
                with self.subTest(model=name, keep=keep):
                    out = self.t / f"pruned-{keep[:-1]}-{name}"
                    prune.run(self.t / name, self.t / data, out, keep)
                    want = qf_answer(out / name, self.t / f"{keep[:-1]}-{name}.flat", x)
                    got = Reload.lgb_booster(out / name, x)
                    np.testing.assert_allclose(got, want, rtol=0, atol=1e-6, err_msg=f"{name}: Booster(model_file).predict differs")
                    rep = optimize.optimize(
                        self.t / name,
                        self.t / data,
                        self.t / f"opt-{keep[:-1]}-{name}",
                        keep,
                        True,
                        True,
                        min_agreement=0.0,
                        backend="flat",
                    )
                    self.assertTrue(rep["ok"], rep["failures"])
                    np.testing.assert_allclose(
                        Reload.lgb_booster(self.t / f"opt-{keep[:-1]}-{name}" / "pruned" / name, x), got, rtol=0, atol=1e-9
                    )

    def test_the_importance_block_of_a_pruned_lightgbm_file_describes_the_pruned_trees(self):
        """The text file carries `feature_importances:` (split counts); a pruned file holding 3 trees must not still list the original's."""
        name = "es_all_rounds.txt"
        out = self.t / "pruned-importance"
        prune.run(self.t / name, self.t / "d.npz", out, "50%")
        text = (out / name).read_text()
        block = text.split("feature_importances:\n", 1)[1].split("\n\n", 1)[0].splitlines()
        listed = {k: int(v) for k, v in (ln.split("=") for ln in block)}
        booster = lgb.Booster(model_file=str(out / name))
        counts = booster.feature_importance("split")
        want = {n: int(c) for n, c in zip(booster.feature_name(), counts, strict=True) if c}
        self.assertEqual(listed, want)


@unittest.skipUnless(ydf is not None, "no ydf")
class YdfFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        x, score = make_rows()
        cls.x = x
        y = (score > np.median(score)).astype(int)
        np.savez(cls.t / "d.npz", X=x)
        cols = {f"f{i}": x[:, i] for i in range(NF)}
        for tag, learner in (
            ("gbt", ydf.GradientBoostedTreesLearner(label="y", num_trees=60, max_depth=3, num_threads=1)),
            ("rf", ydf.RandomForestLearner(label="y", num_trees=60, max_depth=6, num_threads=1)),
        ):
            learner.train({**cols, "y": y}, verbose=0).save(str(cls.t / tag))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_pruned_ydf_folder_predicts_like_the_pruned_forest_through_load_model(self):
        for tag in ("gbt", "rf"):
            for keep in ("50%", "90%"):
                with self.subTest(model=tag, keep=keep):
                    out = self.t / f"pruned-{keep[:-1]}-{tag}"
                    prune.run(self.t / tag, self.t / "d.npz", out, keep)
                    want = qf_answer(out / tag, self.t / f"{keep[:-1]}-{tag}.flat", self.x)
                    got = Reload.ydf_model(out / tag, self.x)
                    np.testing.assert_allclose(
                        got.reshape(want.shape), want, rtol=0, atol=2e-5, err_msg=f"{tag}: ydf.load_model().predict differs"
                    )
                    self.assertEqual(ydf.load_model(str(out / tag)).num_trees(), qf.load_forest(qf_formats.model_files(out / tag)).n_trees)


@unittest.skipUnless(xgb is not None, "no xgboost")
class ArtifactsAreCheckedAsTheReferenceShowsTest(unittest.TestCase):
    """`compile --backend auto` and `choose.py` write OUT and OUT.choice.json; the command docs/REFERENCE.md shows,
    `quickforest check MODEL ARTIFACT --data D` (no `--backend`), failed with 'not a treeflat file (magic)' whenever the winner was generated code."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.t = Path(cls.tmp.name)
        x, score = make_rows()
        y = (score > np.median(score)).astype(int)
        xgb.train({"objective": "binary:logistic", "max_depth": 3, "verbosity": 0, "nthread": 1}, xgb.DMatrix(x, y), 40).save_model(
            str(cls.t / "m.ubj")
        )
        np.savez(cls.t / "d.npz", X=x)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def quickforest(self, *args):
        return subprocess.run([sys.executable, "-s", QUICKFOREST, *map(str, args)], capture_output=True, text=True, timeout=900)

    def check_it(self, art: Path):
        r = self.quickforest("check", self.t / "m.ubj", art, "--data", self.t / "d.npz", "--random", "300")
        self.assertEqual(r.returncode, 0, f"{art.name}: {(r.stdout + r.stderr)[-300:]}")
        self.assertIn("PASS", r.stdout)

    def test_check_without_a_backend_judges_every_artifact_compile_and_choose_write(self):
        flat, wave = self.t / "a.flat", self.t / "a-wave.flat"
        compile.compile_model(self.t / "m.ubj", flat, backend="flat")
        compile.compile_model(self.t / "m.ubj", wave, backend="wave")
        for art in (flat, wave):
            self.check_it(art)
        if not have_rustc():
            self.skipTest("no rustc: the generated-code artifacts were not built or checked")
        so = self.t / "a-block.so"
        compile.compile_model(self.t / "m.ubj", so, backend="codegen-block")
        self.check_it(so)  # no choice file: the suffix says which loader
        chosen = self.t / "chosen.bin"
        r = subprocess.run(
            [
                sys.executable,
                "-s",
                str(SCRIPTS / "choose.py"),
                str(self.t / "m.ubj"),
                str(chosen),
                "--data",
                str(self.t / "d.npz"),
                "--only",
                "codegen-block",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.assertEqual(json.loads(Path(f"{chosen}.choice.json").read_text())["backend"], "codegen-block")
        self.check_it(chosen)
        out = self.t / "opt-block"  # the artifact `optimize` writes next to pruned/, judged against the pruned models it was built from
        rep = optimize.optimize(self.t / "m.ubj", self.t / "d.npz", out, "50%", True, True, min_agreement=0.0, backend="codegen-block")
        self.assertTrue(rep["ok"], rep["failures"])
        self.check_it_for(out / "pruned", out / "model.so")

    def test_an_explicit_backend_still_wins_over_the_choice_file(self):
        flat = self.t / "forced.flat"
        compile.compile_model(self.t / "m.ubj", flat, backend="flat")
        Path(f"{flat}.choice.json").write_text(json.dumps({"backend": "codegen-block"}))
        r = self.quickforest("check", self.t / "m.ubj", flat, "--backend", "flat", "--data", self.t / "d.npz", "--random", "100")
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-300:])

    def test_check_run_loads_an_unnamed_backend_through_the_artifact(self):
        flat = self.t / "viaartifact.flat"
        compile.compile_model(self.t / "m.ubj", flat, backend="flat")
        with mock.patch.object(backends, "load_artifact", wraps=backends.load_artifact) as spy:
            res = check.run(self.t / "m.ubj", flat, data=self.t / "d.npz", n_random=100)
        self.assertTrue(res["ok"])
        self.assertTrue(spy.called, "check.run without a backend must load the artifact through backends.load_artifact")

    def test_the_pruned_models_optimize_writes_feed_compare_check_and_compile_as_the_reference_shows(self):
        out = self.t / "opt-feed"
        optimize.optimize(self.t / "m.ubj", self.t / "d.npz", out, "50%", True, True, min_agreement=0.0, backend="flat")
        pruned = out / "pruned"
        art = self.t / "from-pruned.flat"
        r = self.quickforest("compile", pruned, art)
        self.assertEqual(r.returncode, 0, r.stderr[-300:])
        self.check_it_for(pruned, art)
        r = subprocess.run(
            [
                sys.executable,
                "-s",
                str(SCRIPTS / "compare.py"),
                str(pruned),
                "--data",
                str(self.t / "d.npz"),
                "--work",
                str(self.t / "cmp-work"),
                "--backends",
                "flat,wave",
                "--random",
                "100",
                "--eq-rows",
                "300",
                "--speed-rows",
                "300",
            ],
            capture_output=True,
            text=True,
            timeout=900,
        )
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-300:])
        self.assertNotIn("NOT EXACT", r.stdout)
        shutil.rmtree(self.t / "cmp-work")

    def check_it_for(self, model: Path, art: Path):
        r = self.quickforest("check", model, art, "--data", self.t / "d.npz", "--random", "300")
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-300:])


class CachesAreKeyedByContentTest(unittest.TestCase):
    """The class behind two bugs (a compare cache keyed by the data file's name and size; models reused by their fold count): whatever the
    repository remembers about a file is keyed by the file's bytes, never by its name, size or modification time. The data-file cache has its
    pin in test_pipeline.py, the benchmark's in test_bench.py, the differential run's resume state in test_differential_pins.py."""

    def test_the_library_digest_follows_the_bytes_not_the_timestamp(self):
        import evaluator

        with tempfile.TemporaryDirectory() as d:
            lib = Path(d) / "libcopy.so"
            shutil.copy(evaluator.find_lib(), lib)
            st = lib.stat()
            before = evaluator.lib_digest(str(lib))
            blob = bytearray(lib.read_bytes())
            blob[-1] ^= 0xFF  # the same size...
            lib.write_bytes(bytes(blob))
            os.utime(lib, ns=(st.st_atime_ns, st.st_mtime_ns))  # ...and the same modification time, as `rsync -a` restores them
            self.assertEqual((lib.stat().st_size, lib.stat().st_mtime_ns), (st.st_size, st.st_mtime_ns))
            self.assertNotEqual(evaluator.lib_digest(str(lib)), before)


if __name__ == "__main__":
    unittest.main()
