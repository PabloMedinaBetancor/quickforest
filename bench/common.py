"""Shared helpers of the bench scripts: synthetic data (numpy only, no dataset download), the three model trainers, the
machine banner every script prints, and a runner for `scripts/compare.py`. Nothing here depends on the machine it runs on."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def cpu_model() -> str:
    try:
        for ln in Path("/proc/cpuinfo").read_text().splitlines():
            if ln.startswith(("model name", "Model name", "Hardware")):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def lib_versions() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("xgboost", "lightgbm", "ydf"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = "not installed"
    return out


def banner(title: str, threads: int = 1, **extra) -> dict:
    """The machine, the threads and the batch size, printed first by every script and kept in its JSON (no host name, no user name)."""
    import qf_formats

    info = {
        "title": title,
        "machine": platform.machine(),
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "os": f"{platform.system()} {platform.release()}",
        "threads_used": threads,
        "baseline": f"the source library's own predict, {threads} thread{'s' if threads > 1 else ''} (the compiled forest on the same {threads})",
        "batch_rows": qf_formats.BASELINE_ROWS,
        "date": time.strftime("%Y-%m-%d"),
        **lib_versions(),
        **extra,
    }
    print("=" * 100)
    for k, v in info.items():
        print(f"{k:14s} {v}")
    print("=" * 100, flush=True)
    return info


# ------------------------------------------------------------------------------------------------------------- synthetic data
def make_data(rows: int, features: int, classes: int, seed: int, noise: float = 0.35):
    """Mixed numeric / integer / missing features and a noisy non-linear label (classes >= 2) or target (classes == 0)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features)).astype(np.float32)
    x[:, features // 2 :: 5] = rng.integers(0, 12, size=x[:, features // 2 :: 5].shape).astype(np.float32)
    z = np.nan_to_num(x)
    w = rng.normal(size=(features, max(classes, 1)))
    score = np.tanh(z @ w * 0.4) + 0.6 * np.sin(z[:, :1] * z[:, 1:2]) + 0.25 * (z[:, 2:3] > 0.3) * z[:, 3:4]
    score = score + rng.normal(size=score.shape) * noise
    x[rng.random(x.shape) < 0.04] = np.nan
    if classes >= 2:
        return x, score.argmax(1).astype(np.int64)
    return x, score[:, 0].astype(np.float64) * 2 + 8.0


def make_sparse_data(rows: int, features: int, classes: int, seed: int, noise: float = 0.1):
    """Tree-friendly label: each class score is a few threshold terms on a few features (rule-like), plus noise. Same feature mix as
    `make_data` (numeric, integer, 4 % missing). The dense `make_data` label cannot be pruned to 90 % agreement
    (bench/README-NUMBERS.md)."""
    assert classes == 3
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, features)).astype(np.float32)
    x[:, features // 2 :: 5] = rng.integers(0, 12, size=x[:, features // 2 :: 5].shape).astype(np.float32)
    z = np.nan_to_num(x)
    s = np.zeros((rows, classes))
    s[:, 0] = (z[:, 0] > 0.2) * 1.0 + (z[:, 1] < -0.3) * 0.8 + (z[:, features // 2] > 6) * 0.6
    s[:, 1] = (z[:, 2] > 0.0) * 1.0 + (z[:, 3] > 0.5) * (z[:, 0] < 0) * 1.0 + (z[:, features // 2 + 5] < 3) * 0.5
    s[:, 2] = (z[:, 4] < 0.1) * 0.9 + (z[:, 5] > -0.2) * (z[:, 1] > 0) * 1.0 + (z[:, features // 2 + 10] > 8) * 0.7
    s = s + rng.normal(size=s.shape) * noise
    x[rng.random(x.shape) < 0.04] = np.nan
    return x, s.argmax(1).astype(np.int64)


def kfold_ids(rows: int, folds: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(rows) % folds


# ------------------------------------------------------------------------------------------------------------------ trainers
def train_xgboost(x, y, folds_ids, folder: Path, classes: int, trees: int, depth: int, rf: bool, seed: int):
    """One model file per fold, trained on the rows of the other folds. rf=True: a random forest of `trees` parallel trees per class."""
    import xgboost as xgb

    folder.mkdir(parents=True, exist_ok=True)
    for k in range(int(folds_ids.max()) + 1):
        tr = folds_ids != k if folds_ids.max() > 0 else np.ones(len(x), bool)  # one fold: train on everything
        p = {
            "objective": "multi:softprob" if classes > 2 else "binary:logistic" if classes == 2 else "reg:squarederror",
            "max_depth": depth,
            "nthread": os.cpu_count() or 1,
            "verbosity": 0,
            "tree_method": "hist",
            "seed": seed + k,
        }
        if classes > 2:
            p["num_class"] = classes
        if rf:
            p.update(num_parallel_tree=trees, subsample=0.63, colsample_bynode=0.6, eta=1.0, min_child_weight=1, reg_lambda=1e-5)
            rounds = 1
        else:
            p.update(eta=0.1, subsample=0.8)
            rounds = trees
        b = xgb.train(p, xgb.DMatrix(x[tr], label=y[tr]), rounds)
        b.save_model(str(folder / f"fold{k}.ubj"))


def train_lightgbm(x, y, folder: Path, classes: int, trees: int, leaves: int, seed: int):
    import lightgbm as lgb

    folder.mkdir(parents=True, exist_ok=True)
    p = {
        "objective": "multiclass" if classes > 2 else "binary" if classes == 2 else "regression",
        "num_leaves": leaves,
        "learning_rate": 0.1,
        "verbose": -1,
        "num_threads": os.cpu_count() or 1,
        "seed": seed,
        "deterministic": True,
        "min_data_in_leaf": 10,
    }
    if classes > 2:
        p["num_class"] = classes
    lgb.train(p, lgb.Dataset(x, y), trees).save_model(str(folder / "model.txt"))


def train_ydf(x, y, folder: Path, classes: int, trees: int, depth: int, seed: int, forest: bool = False):
    import ydf

    cols = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    label = y.astype(str) if classes >= 2 else y
    learner = ydf.RandomForestLearner if forest else ydf.GradientBoostedTreesLearner
    m = learner(
        label="y",
        task=ydf.Task.CLASSIFICATION if classes >= 2 else ydf.Task.REGRESSION,
        num_trees=trees,
        max_depth=depth,
        random_seed=seed,
        num_threads=os.cpu_count() or 1,
    ).train({**cols, "y": label}, verbose=0)
    m.save(str(folder))


def make_multi_targets(x, k: int, seed: int):
    """(rows, k) float64 regression targets for the multi-output rows: the `make_data` generator with one score column per target."""
    rng = np.random.default_rng(seed)
    z = np.nan_to_num(x)
    w = rng.normal(size=(x.shape[1], k))
    score = np.tanh(z @ w * 0.4) + 0.6 * np.sin(z[:, :1] * z[:, 1:2]) + 0.25 * (z[:, 2:3] > 0.3) * z[:, 3:4]
    return (score + rng.normal(size=score.shape) * 0.35) * 2 + 8.0


def train_xgboost_vector(x, y, folder: Path, rounds: int, depth: int, seed: int):
    """Row `xgboost-vector`: reg:squarederror with multi_strategy=multi_output_tree, K = y.shape[1] outputs, one vector tree per round."""
    import xgboost as xgb

    folder.mkdir(parents=True, exist_ok=True)
    p = {
        "objective": "reg:squarederror",
        "multi_strategy": "multi_output_tree",
        "tree_method": "hist",
        "max_depth": depth,
        "eta": 0.1,
        "subsample": 0.8,
        "nthread": os.cpu_count() or 1,
        "verbosity": 0,
        "seed": seed,
    }
    xgb.train(p, xgb.DMatrix(x, label=y), rounds).save_model(str(folder / "model.ubj"))


def train_sklearn_rf(x, y, folder: Path, trees: int, depth: int, seed: int):
    """Row `sklearn-rf`: RandomForestRegressor, K = y.shape[1] outputs. Writes model.joblib (the bench's own copy, for the library timing; QF
    never reads it) and model.qfsk through QF's exporter (skipped with a note on a tree without `qf_sklearn`, i.e. before 1.2)."""
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    folder.mkdir(parents=True, exist_ok=True)
    rf = RandomForestRegressor(n_estimators=trees, max_depth=depth, n_jobs=os.cpu_count() or 1, random_state=seed).fit(x, y)
    joblib.dump(rf, folder / "model.joblib")
    try:
        import qf_sklearn
    except ImportError:  # a tree before 1.2: the library-only timing needs just the joblib file
        print("note: qf_sklearn not found, no model.qfsk written (library-only rows)")
        return
    qf_sklearn.export_sklearn(rf, str(folder / "model.qfsk"), X=x[:2000])


def library_only_rate(lib: str, folder: Path, x, threads: int, runs: int = 3, reps: int = 2) -> list[float]:
    """Rows/s of the library's own predict over BASELINE_ROWS rows, one value per run (best of `reps` passes each): the baseline of the
    rows whose compiled side does not exist in the tree under test (1.1 cannot read sklearn or vector-leaf xgboost). Same batch and thread
    rules as `qf_formats.library_rate`."""
    import qf_formats

    rows = qf_formats.BASELINE_ROWS
    xs = np.ascontiguousarray(x)
    xs = xs[:rows] if len(xs) >= rows else np.tile(xs, (-(-rows // len(xs)), 1))[:rows]
    if lib.startswith("sklearn-rf"):
        import joblib

        m = joblib.load(folder / "model.joblib")
        m.set_params(n_jobs=threads)
        call = lambda: m.predict(xs)  # noqa: E731
    elif lib.startswith("xgboost-vector"):
        import xgboost as xgb

        b = xgb.Booster(model_file=str(folder / "model.ubj"))
        b.set_param({"nthread": threads})
        d = xgb.DMatrix(xs)
        call = lambda: b.predict(d)  # noqa: E731
    else:
        raise ValueError(lib)
    call()  # warm-up
    out = []
    for _ in range(runs):
        best = float("inf")
        for _ in range(reps):
            t = time.perf_counter()
            call()
            best = min(best, time.perf_counter() - t)
        out.append(rows / best)
    return out


def save_selection(path: Path, x, folds_ids=None):
    d = {"X": np.ascontiguousarray(x, dtype=np.float32)}
    if folds_ids is not None:
        d["fold"] = folds_ids.astype(np.int64)
    np.savez(path, **d)


# -------------------------------------------------------------------------------------------------------------------- runners
def run(cmd: list[str], log: Path | None = None) -> float:
    """Run one of the scripts, return the seconds. Output goes to `log` (or is discarded); a failure stops the bench with the log's tail."""
    t = time.perf_counter()
    with open(log or os.devnull, "w") as f:
        r = subprocess.run([sys.executable, "-s", *cmd], stdout=f, stderr=subprocess.STDOUT, check=False)
    if r.returncode not in (0, 3):  # 3: a backend cannot hold the model (reported in the table, not a failure of the bench)
        tail = Path(log).read_text()[-1500:] if log else ""
        raise SystemExit(f"{' '.join(cmd[:2])} failed ({r.returncode})\n{tail}")
    return time.perf_counter() - t


BASELINE_BUILD = ("compile_preset=1.0",)  # QF 1.0's build: the baseline of every figure in README-NUMBERS.md


def compare(
    model: Path,
    data: Path,
    work: Path,
    backends: str,
    out_json: Path,
    decision: str = "argmax",
    cpu: int | None = None,
    stage: str = "all",
    threads: int = 1,
    opts: tuple[str, ...] = BASELINE_BUILD,
):
    """Runs scripts/compare.py. `opts` are its `--opt key=value` build options: by default QF 1.0's build (`compile_preset=1.0`), the one the figures in
    README-NUMBERS.md were measured with, whatever the library's own default build is; pass `()` for the library's default build or other options."""
    cmd = [
        str(SCRIPTS / "compare.py"),
        str(model),
        "--data",
        str(data),
        "--backends",
        backends,
        "--work",
        str(work),
        "--json",
        str(out_json),
        "--decision",
        decision,
    ]
    if cpu is not None:
        cmd += ["--cpu", str(cpu)]
    cmd += [
        "--threads",
        str(threads),
    ]  # the library and the compiled forest are both timed on this many threads (compare.py defaults to all)
    for o in opts:
        cmd += ["--opt", o]
    if stage != "all":  # "build" compiles (needs rustc), "measure" times what was built, possibly on another machine
        cmd += ["--stage", stage]
    run(cmd, work.with_suffix(".log"))
    return {} if stage == "build" else json.loads(out_json.read_text())


def fmt_table(rows: list[list], head: list[str]) -> str:
    w = [max(len(str(r[i])) for r in [head, *rows]) for i in range(len(head))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |"  # noqa: E731
    return "\n".join([line(head), "|" + "|".join("-" * (x + 2) for x in w) + "|", *map(line, rows)])
