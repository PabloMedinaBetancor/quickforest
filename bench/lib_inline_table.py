#!/usr/bin/env python3
"""Evaluator library with and without `-C llvm-args=-inline-threshold=3000`: flat and wave rows/s on the models of a cspgo_table work folder.

    lib_inline_table.py --work WCS --old OLD.so --new NEW.so [--libs a,b] [--threads 1,all] [--runs 3] --out OUT.json

Same models, same rows, same machine; the two libraries alternate inside every run (old,new / new,old / old,new) and the median of the runs is
reported. Flat and wave need no rustc. Run it alone on a quiet machine."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

LIBS = "xgboost,lightgbm,ydf,ydf-rf,sklearn-rf,xgboost-vector"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--libs", default=LIBS)
    ap.add_argument("--threads", default="1,all")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--backends", default="flat,wave")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work, libs = Path(a.work), a.libs.split(",")
    paths = {"old": str(Path(a.old).resolve()), "new": str(Path(a.new).resolve())}
    res: dict = {"info": c.banner("evaluator inline-threshold 3000", libs=a.libs, runs=a.runs), "libraries": paths, "cells": {}}
    for tspec in a.threads.split(","):
        n = (os.cpu_count() or 1) if tspec == "all" else int(tspec)
        for lib in libs:
            folder, sel = work / lib / f"model-{lib}", work / lib / "sel.npz"
            rates = {(v, b): [] for v in paths for b in a.backends.split(",")}
            for r in range(a.runs):
                for v in ("old", "new") if r % 2 == 0 else ("new", "old"):
                    os.environ["TREEFLAT_LIB"] = paths[v]
                    j = work / lib / f"inl-{v}-t{n}-r{r}.json"
                    d = c.compare(folder, sel, work / lib / f"cmp-inl-{v}", a.backends, j, stage="all", threads=n, opts=())
                    for row in d["rows"]:
                        if row["backend"] in a.backends.split(",") and "rows_per_s" in row:
                            rates[(v, row["backend"])].append(row["rows_per_s"])
                print(f"threads {n} {lib} run {r + 1}/{a.runs} done", flush=True)
            for b in a.backends.split(","):
                o, w = rates[("old", b)], rates[("new", b)]
                if o and w:
                    res["cells"][f"{lib}/{b}/t{n}"] = {
                        "old_rows_per_s": o,
                        "new_rows_per_s": w,
                        "old_median": statistics.median(o),
                        "new_median": statistics.median(w),
                        "new_vs_old_pct": round((statistics.median(w) / statistics.median(o) - 1) * 100, 1),
                    }
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in res["cells"].items():
        print(f"{k:40s} old {v['old_median']:>10,.0f}  new {v['new_median']:>10,.0f}  {v['new_vs_old_pct']:+.1f} %")


if __name__ == "__main__":
    main()
