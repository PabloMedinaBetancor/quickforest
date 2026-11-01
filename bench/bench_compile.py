#!/usr/bin/env python3
"""Compile-only speedups per library (no pruning): a synthetic model of each library, compiled with the flat walker, the wavefront engine
and the generated-code backends, against that library's OWN predict on the same number of threads (`--threads`, default 1) over 20,000-row batches.

    bench_compile.py --work DIR [--lib xgboost,lightgbm,ydf,ydf-rf] [--size small|full] [--backends flat,wave,codegen-block] [--cpu N] [--threads N]
                     [--stage all|build|measure] [--replicate-outputs] [--library-only] [--json OUT]

1.2 rows (multi-output trees): `ydf-rf` (now one vector tree per YDF tree), `ydf-rf-replicated` (the same model, 900 per-class trees),
`sklearn-rf` (RandomForestRegressor, 5 targets, exported to .qfsk), `sklearn-rf-replicated` (the same trees as 5 x 300 scalar trees) and
`xgboost-vector` (reg:squarederror, multi_output_tree, 5 targets). A `-replicated` row reuses the model of its base row and compiles it with
the replicate switch (`--replicate-outputs` does the same for any row). The switch crosses the process boundary of compare.py as the
environment variable QF_REPLICATE_OUTPUTS=1, which `qf_ydf` / `qf_sklearn` read into REPLICATE_CLASSES / REPLICATE_OUTPUTS at import.
`--library-only` times only the library's predict for the rows a tree cannot compile yet (the 1.1 baseline: sklearn-rf, xgboost-vector).

`--stage build` trains and compiles (needs rustc, any machine), `--stage measure` times what was built on the machine it runs on: copy the work
folder to a quiet machine without a Rust toolchain (the generated-code libraries are built for the portable x86-64-v2 CPU) and measure there.

Public data only: a synthetic 3-class problem from numpy (see common.make_data). Prints the machine, the threads and the batch size first.
Exactness is checked by compare (max error against the library, bit-identity with the walker), a row with `status` other than ok is not a
speed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import common as c

SIZES = {  # rows, features, trees (boosted rounds), depth / leaves
    "small": dict(rows=6000, features=30, trees=60, depth=6, leaves=31),
    "full": dict(rows=40000, features=42, trees=300, depth=8, leaves=63),
}


MULTI_K = 5  # targets of the sklearn-rf and xgboost-vector rows
NEW_LIBS = ("ydf-rf-replicated", "sklearn-rf", "sklearn-rf-replicated", "xgboost-vector")
LIBRARY_ONLY_LIBS = ("sklearn-rf", "xgboost-vector")  # no compiled side in 1.1


def base_lib(lib: str) -> str:
    return lib.removesuffix("-replicated")


def build(lib: str, work: Path, s: dict, seed: int, train: bool = True):
    if lib.endswith("-replicated"):  # the same trained model as the vector row, compiled with the switch
        return build(base_lib(lib), work.parent / base_lib(lib), s, seed, train)
    folder = work / f"model-{lib}"
    sel = work / "sel.npz"
    if not train:  # the measure stage: the models and the rows are already there
        if not (folder.is_dir() and sel.exists()):
            raise SystemExit(f"{lib}: nothing built in {work} (run --stage build first, then copy the work folder here)")
        return folder, sel
    x, y = c.make_data(s["rows"], s["features"], 3, seed)
    if lib in ("sklearn-rf", "xgboost-vector"):
        ym = c.make_multi_targets(x, MULTI_K, seed)
        if lib == "sklearn-rf":
            c.train_sklearn_rf(x, ym, folder, s["trees"], s["depth"], seed)
        else:
            c.train_xgboost_vector(x, ym, folder, s["trees"] * 3, s["depth"] - 2, seed)  # 900 rounds of depth 6 at the full size
    elif lib == "xgboost":
        c.train_xgboost(x, y, c.kfold_ids(len(x), 1, seed), folder, 3, s["trees"], s["depth"], False, seed)
    elif lib == "lightgbm":
        c.train_lightgbm(x, y, folder, 3, s["trees"], s["leaves"], seed)
    else:  # ydf (gradient boosted trees) or ydf-rf (random forest)
        c.train_ydf(x, y, folder, 3, s["trees"], s["depth"], seed, forest=lib == "ydf-rf")
    c.save_selection(sel, x[: s["rows"] // 2])
    return folder, sel


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--lib", default="xgboost,lightgbm,ydf,ydf-rf")
    ap.add_argument("--size", choices=sorted(SIZES), default="small")
    ap.add_argument("--backends", default="flat,wave,codegen-block")
    ap.add_argument("--cpu", type=int, help="pin to this logical CPU (taskset); use a quiet machine")
    ap.add_argument("--threads", type=int, default=1, help="threads of the timing for the library and the compiled forest alike")
    ap.add_argument("--stage", choices=("all", "build", "measure"), default="all")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--replicate-outputs", action="store_true", help="compile every row as replicated scalar trees (comparison rows)")
    ap.add_argument("--library-only", action="store_true", help="only the library's own predict rate, for rows this tree cannot compile")
    ap.add_argument("--json")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    info = c.banner("compile-only speed per library (no pruning)", threads=a.threads, size=a.size, backends=a.backends, cpu_pinned=a.cpu)
    out, rows = {"info": info, "libs": {}}, []
    for lib in a.lib.split(","):
        (work / lib).mkdir(exist_ok=True)
        try:
            folder, sel = build(lib, work / lib, SIZES[a.size], a.seed, train=a.stage != "measure")
        except ImportError as e:
            print(f"{lib}: skipped ({e})")
            continue
        if a.library_only:
            if lib in LIBRARY_ONLY_LIBS and a.stage != "build":
                import numpy as np

                rates = c.library_only_rate(lib, folder, np.load(sel)["X"], a.threads)
                out["libs"][lib] = {"library_rows_per_s_runs": [round(r) for r in rates], "threads": a.threads}
                print(f"{lib}: library rows/s per run {out['libs'][lib]['library_rows_per_s_runs']} ({a.threads} threads)", flush=True)
            continue
        if a.replicate_outputs or lib.endswith("-replicated"):
            os.environ["QF_REPLICATE_OUTPUTS"] = "1"
        else:
            os.environ.pop("QF_REPLICATE_OUTPUTS", None)
        t = c.compare(folder, sel, work / lib / "cmp", a.backends, work / lib / "table.json", cpu=a.cpu, stage=a.stage, threads=a.threads)
        if a.stage == "build":
            print(f"{lib}: built")
            continue
        out["libs"][lib] = t
        for r in t["rows"]:
            if "rows_per_s" in r:
                rows.append(
                    [
                        lib,
                        r["backend"],
                        f"{t['library_rows_per_s']:,}",
                        f"{r['rows_per_s']:,}",
                        r["speedup"],
                        f"{r['max_err']:.1e}",
                        r["status"],
                        r.get("compile_seconds", ""),
                        r.get("artifact_mb", ""),
                    ]
                )
            else:
                rows.append(
                    [
                        lib,
                        r["backend"],
                        "",
                        "",
                        "",
                        "",
                        "not run: " + str(r["status"]).replace("run failed (exit 1): ", "")[:90],
                        r.get("compile_seconds", ""),
                        "",
                    ]
                )
    if a.stage == "build" or a.library_only:
        if a.library_only and a.json:
            Path(a.json).write_text(json.dumps(out, indent=1, default=str))
        return
    print(
        c.fmt_table(
            rows,
            [
                "library",
                "backend",
                f"library rows/s ({a.threads} thread{'s' if a.threads > 1 else ''})",
                "compiled rows/s",
                "speedup",
                "max err",
                "status",
                "compile s",
                "MB",
            ],
        )
    )
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
