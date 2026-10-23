#!/usr/bin/env python3
"""End-to-end speed, file size and load time: a k-fold xgboost random forest (3 classes, `trees` parallel trees per class per
fold) against the same forest pruned to 10 % of its trees and compiled with the generated-code backend with profile-guided optimisation.

    bench_e2e.py --work DIR [--size small|readme] [--keep 10%] [--backend codegen-pgo] [--cpu N] [--threads N] [--stage all|build|measure] [--json OUT]

`--stage build` trains, prunes and compiles (needs rustc and RAM, any machine, timing is not used); `--stage measure` times everything on the
machine it runs on, from a copy of the work folder (the generated-code library is built for the portable x86-64-v2 CPU, no Rust toolchain is needed there). Use
the two stages when the build machine is busy: the speed figures are only valid from a quiet machine.

Steps (all through this repository's own scripts): train the forest on synthetic numpy data -> `optimize` (prune on the selection rows,
held-out agreement)
-> `compare` on the pruned model (compile + exactness + speed) -> time the FULL forest with xgboost (`inplace_predict`, 1 thread or --threads N, 20,000-row batch). With --threads N the compiled forest
is timed on N threads too (do not pin one CPU with --cpu then).
The figures: end-to-end = pruned compiled rows/s over full-forest xgboost rows/s; file size = model files on disk over the flat file; load
time = xgboost loading the full model files over the flat file loaded by the evaluator. `--size readme` is the size of the figure in the
README (67 trees per class per fold, 5 folds, depth 14: 2.0 million nodes in all on the default sparse data, 4.5 million on the dense data;
about 12 minutes to train). `--data sparse` (default, label noise 0.1) is a label made of threshold rules, which a pruned forest can follow;
`--data dense --noise 0.35` is a smoother label whose 10 % prune agrees on only 70.8 % (see README-NUMBERS.md). More than 67 trees per class
would be too big for an 8 GB machine to build (300 trees per class is 4 million nodes PER fold)."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import common as c
import numpy as np

SIZES = {
    "small": dict(rows=8000, features=30, folds=3, trees=40, depth=8),
    "readme": dict(rows=60000, features=42, folds=5, trees=67, depth=14),
}


def count_nodes(qf, files):
    """Total nodes of the model files, one file at a time: a forest held whole costs ~450 bytes of Python arrays per node, which on an 8 GB
    machine can exhaust memory and leave no room for the optimize child process that needs the same again."""
    n = 0
    for f in files:
        forest = qf.load_forest([f])
        n += sum(len(t.left) for m in forest.models for t in m.trees)
        del forest
    return n


def training_stamp(size: str, a) -> dict:
    """What the trained models in a work folder were made from. A folder with the right number of fold files is not enough: the same
    --work reused with another --size, --data, --noise or --seed would be timed as if it held the models of the banner."""
    import xgboost

    return {"size": size, **SIZES[size], "data": a.data, "noise": a.noise, "seed": a.seed, "xgboost": xgboost.__version__}


def trained_here(work: Path, stamp: dict) -> bool:
    """Whether `work` holds the models and rows trained from exactly `stamp` (the stamp is written when training ends)."""
    try:
        done = json.loads((work / "training.json").read_text())
    except (OSError, ValueError):
        return False  # a folder from before the stamp existed, or an interrupted training: train again
    return done == stamp and (work / "sel.npz").exists() and len(list((work / "full").glob("fold*.ubj"))) == stamp["folds"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--size", choices=sorted(SIZES), default="small")
    ap.add_argument("--keep", default="10%")
    ap.add_argument("--backend", default="codegen-pgo")
    ap.add_argument("--min-agreement", type=float, default=90.0, help="held-out agreement bar of the pruning, percent")
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--threads", type=int, default=1, help="threads of the timing for xgboost and the compiled forest alike")
    ap.add_argument("--stage", choices=("all", "build", "measure"), default="all")
    ap.add_argument(
        "--data",
        choices=("sparse", "dense"),
        default="sparse",
        help="synthetic label: sparse = threshold rules (prunable), dense = tanh(x@w) (see bench/README-NUMBERS.md)",
    )
    ap.add_argument("--noise", type=float, default=0.1, help="label noise of the synthetic data")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--json")
    a = ap.parse_args()
    s, work = SIZES[a.size], Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    info = c.banner(
        "end to end: full xgboost forest vs pruned + compiled",
        size=a.size,
        threads=a.threads,
        keep=a.keep,
        backend=a.backend,
        label_data=a.data,
        label_noise=a.noise,
        **s,
    )

    import qf_common as qf
    import qf_formats

    full, sel = work / "full", work / "sel.npz"
    out, state = work / "optimize", work / "build-state.json"
    files = qf_formats.model_files(full) if full.is_dir() else []
    if a.stage != "measure":
        stamp = training_stamp(a.size, a)
        if not trained_here(work, stamp):
            (work / "training.json").unlink(missing_ok=True)
            shutil.rmtree(full, ignore_errors=True)  # fold files of an earlier, bigger training would be read as folds
            t = time.perf_counter()
            x, y = (c.make_sparse_data if a.data == "sparse" else c.make_data)(s["rows"], s["features"], 3, a.seed, a.noise)
            ids = c.kfold_ids(len(x), s["folds"], a.seed)
            c.train_xgboost(x, y, ids, full, 3, s["trees"], s["depth"], True, a.seed)
            c.save_selection(sel, x, ids)
            (work / "training.json").write_text(json.dumps(stamp))
            print(f"trained {s['folds']} folds in {time.perf_counter() - t:.0f} s", flush=True)
        if out.exists():
            shutil.rmtree(out)  # optimize refuses a non-empty --out (a refused earlier run leaves its report there)
        files = qf_formats.model_files(full)
        nodes_full = count_nodes(qf, files)
        c.run(
            [
                str(c.SCRIPTS / "optimize.py"),
                str(full),
                "--data",
                str(sel),
                "--keep",
                a.keep,
                "--out",
                str(out),
                "--min-agreement",
                str(a.min_agreement),
            ],
            work / "optimize.log",
        )
        rep = json.loads((out / "report.json").read_text())
        if not (out / "pruned").is_dir():
            raise SystemExit(f"pruning failed its bar, see {out / 'report.md'}")
        state.write_text(
            json.dumps(
                {
                    "nodes_full": nodes_full,
                    "nodes_pruned": count_nodes(qf, qf_formats.model_files(out / "pruned")),
                    "agree": 100.0
                    if isinstance(rep["stages"]["prune"], str)
                    else rep["stages"]["prune"]["heldout_pct"],  # a string: skipped (--keep 100%)
                    "full_mb": sum(f.stat().st_size for f in files) / 1e6,
                }
            )
        )
    elif not state.exists():
        raise SystemExit(f"nothing built in {work} (run --stage build first, then copy the work folder here)")
    st = json.loads(state.read_text())
    nodes_full, nodes_pruned, agree, full_mb = st["nodes_full"], st["nodes_pruned"], st["agree"], st["full_mb"]
    pruned = out / "pruned"

    table = c.compare(
        pruned,
        sel,
        work / "cmp",
        "flat" if a.backend == "flat" else f"flat,{a.backend}",
        work / "table.json",
        cpu=a.cpu,
        stage=a.stage,
        threads=a.threads,
    )
    if a.stage == "build":
        print("built; copy the work folder to the measuring machine and run --stage measure")
        return
    row = next(r for r in table["rows"] if r["backend"] == a.backend)
    if row.get("status") != "ok":
        raise SystemExit(f"{a.backend}: {row.get('status')}")

    srcs = [qf_formats.open_source(f) for f in files]
    x_all = np.load(sel)["X"]
    full_rate = qf_formats.library_rate(srcs, x_all, threads=a.threads)
    flat = next(r for r in table["rows"] if r["backend"] == "flat")

    # load time: the library reading the full model files vs the evaluator opening the flat file of the pruned forest
    t = time.perf_counter()
    boosters = [qf.load_booster(f) for f in files]
    load_full = time.perf_counter() - t
    del boosters
    res = {
        "info": info,
        "threads": a.threads,
        "nodes_full": nodes_full,
        "nodes_pruned": nodes_pruned,
        "prune_agreement_percent": agree,
        "xgboost_full_rows_per_s": round(full_rate),
        "xgboost_pruned_rows_per_s": table["library_rows_per_s"],
        f"{a.backend}_pruned_rows_per_s": row["rows_per_s"],
        "flat_pruned_rows_per_s": flat["rows_per_s"],
        "end_to_end_speedup": round(row["rows_per_s"] / full_rate, 1),
        "model_files_full_mb": round(full_mb, 1),
        "flat_file_pruned_mb": flat["artifact_mb"],
        "size_ratio": round(full_mb / flat["artifact_mb"], 1),
        "load_seconds_xgboost_full": round(load_full, 2),
        "load_seconds_flat_pruned": flat["load_seconds"],
        "max_error_vs_xgboost_pruned": row["max_err"],
        "compile_seconds": row.get("compile_seconds"),
    }
    print(json.dumps(res, indent=1, default=str))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
