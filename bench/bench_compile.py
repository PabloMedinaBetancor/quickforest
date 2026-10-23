#!/usr/bin/env python3
"""Compile-only speedups per library (no pruning): a synthetic model of each library, compiled with the flat walker, the wavefront engine
and the generated-code backends, against that library's OWN predict on the same number of threads (`--threads`, default 1) over 20,000-row batches.

    bench_compile.py --work DIR [--lib xgboost,lightgbm,ydf,ydf-rf] [--size small|full] [--backends flat,wave,codegen-block] [--cpu N] [--threads N]
                     [--stage all|build|measure] [--json OUT]

`--stage build` trains and compiles (needs rustc, any machine), `--stage measure` times what was built on the machine it runs on: copy the work
folder to a quiet machine without a Rust toolchain (the generated-code libraries are built for the portable x86-64-v2 CPU) and measure there.

Public data only: a synthetic 3-class problem from numpy (see common.make_data). Prints the machine, the threads and the batch size first.
Exactness is checked by compare (max error against the library, bit-identity with the walker), a row with `status` other than ok is not a
speed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common as c

SIZES = {  # rows, features, trees (boosted rounds), depth / leaves
    "small": dict(rows=6000, features=30, trees=60, depth=6, leaves=31),
    "full": dict(rows=40000, features=42, trees=300, depth=8, leaves=63),
}


def build(lib: str, work: Path, s: dict, seed: int, train: bool = True):
    folder = work / f"model-{lib}"
    sel = work / "sel.npz"
    if not train:  # the measure stage: the models and the rows are already there
        if not (folder.is_dir() and sel.exists()):
            raise SystemExit(f"{lib}: nothing built in {work} (run --stage build first, then copy the work folder here)")
        return folder, sel
    x, y = c.make_data(s["rows"], s["features"], 3, seed)
    if lib == "xgboost":
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
    if a.stage == "build":
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
