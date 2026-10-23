"""Quickstart: train a random forest on synthetic data, prune it to 10 % of its trees, compile it, and compare it with the original.

    python3 examples/quickstart.py [WORKDIR]

Needs numpy and xgboost. Takes under a minute on a quiet machine and writes everything under WORKDIR (default: a new temporary folder).
The synthetic label is a few threshold rules plus noise, which a pruned forest can follow; real data decides how far your own
forest prunes (see LIMITS.md).
"""

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import backends  # noqa: E402
import evaluator  # noqa: E402

FEATURES, CLASSES = 20, 3


def make_data(rows, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, FEATURES)).astype(np.float32)
    score = np.stack(
        [
            (x[:, 0] > 0.2) + 0.8 * (x[:, 1] < -0.3),
            (x[:, 2] > 0.0) + (x[:, 3] > 0.5) * (x[:, 0] < 0),
            0.9 * (x[:, 4] < 0.1) + (x[:, 5] > -0.2) * (x[:, 1] > 0),
        ],
        axis=1,
    )
    y = (score + rng.normal(size=score.shape) * 0.1).argmax(1)
    x[rng.random(x.shape) < 0.04] = np.nan  # missing values are part of the data
    return x, y


def rows_per_second(predict, x, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        predict(x)
        best = min(best, time.perf_counter() - t)
    return len(x) / best


def speed_line(booster, flat, x):
    """The last line: the full xgboost forest and the compiled, pruned one, both on every usable core (flat.predict's own default on a
    batch of this size; `nthread` is set to the same count)."""
    cores = evaluator.usable_cpus()
    booster.set_param({"nthread": cores})
    slow = rows_per_second(lambda a: booster.inplace_predict(a, validate_features=False), x)
    fast = rows_per_second(flat.predict, x)
    return f"all cores ({cores}): xgboost, full forest {slow:,.0f} rows/s; compiled, pruned forest {fast:,.0f} rows/s ({fast / slow:.1f}x)"


def main():
    work = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="quickforest-"))
    work.mkdir(parents=True, exist_ok=True)
    print(f"work folder: {work}", flush=True)

    # 1. A random forest: one round of 100 parallel trees per class, 300 trees in all.
    x_train, y_train = make_data(30_000, seed=1)
    params = dict(
        objective="multi:softprob",
        num_class=CLASSES,
        tree_method="hist",
        max_depth=8,
        num_parallel_tree=100,
        subsample=0.63,
        colsample_bynode=0.6,
        eta=1.0,
        reg_lambda=1e-5,
        nthread=1,
        verbosity=0,
        seed=1,
    )
    booster = xgb.train(params, xgb.DMatrix(x_train, label=y_train), 1)
    model = work / "model.ubj"
    booster.save_model(str(model))

    # 2. The selection file: rows the forest has not seen. QuickForest splits them into selection rows and held-out rows.
    x_sel, _ = make_data(10_000, seed=2)
    np.savez(work / "selection.npz", X=x_sel)

    # 3. One command: prune to 10 % of the trees, compile, and prove the compiled forest equals xgboost.
    out = work / "out"
    cmd = [sys.executable, str(ROOT / "scripts" / "quickforest"), "optimize", str(model)]
    cmd += [
        "--data",
        str(work / "selection.npz"),
        "--keep",
        "10%",
        "--min-agreement",
        "95",
        "--compile",
        "--verify",
        "--out",
        str(out),
    ]
    print("$", " ".join(cmd), flush=True)
    done = subprocess.run(cmd, check=False)
    if done.returncode != 0:  # exit 2: a bar failed and only the report was written; read out/report.md
        sys.exit(f"optimize exited with {done.returncode}; see {out / 'report.md'}")

    # 4. Use the result on fresh rows and compare it with the full forest.
    x_new, _ = make_data(20_000, seed=3)
    artifact = next(
        out.glob("model.*")
    )  # model.so (generated code for the portable x86-64-v2 CPU) or model.flat, whichever the machine could build
    flat = backends.load_artifact(artifact)
    compiled = flat.predict(x_new)
    full = booster.inplace_predict(x_new, validate_features=False)
    agree = (compiled.argmax(1) == full.argmax(1)).mean()
    print(f"\nfresh rows: the compiled forest and the full forest give the same class on {100 * agree:.2f} % of {len(x_new):,} rows")
    print(f"model file: {model.stat().st_size / 1e6:.2f} MB -> compiled file ({artifact.name}): {artifact.stat().st_size / 1e6:.2f} MB")

    print(speed_line(booster, flat, x_new))
    print("(timing on a shared or busy machine is only a rough guide)")


if __name__ == "__main__":
    main()
