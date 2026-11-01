#!/usr/bin/env python3
"""The whole compile-only table, several runs, medians: one build, then `--runs` measure passes at each thread count, one JSON.

    run_table.py --work DIR --out table.json [--libs xgboost,lightgbm,ydf,ydf-rf] [--library-only-libs sklearn-rf,xgboost-vector]
                 [--threads 1,all] [--runs 3] [--size full] [--backends flat,wave,codegen-block]

It is the 1.0 protocol of bench/README-NUMBERS.md section 2 (20,000-row batch, median of three runs, library and compiled forest in separate
processes) as one command, used twice: the 1.1 baseline (`bench/baseline-1.1-2core.json`: the old rows compiled, the new libraries' own
predict rate timed) and the 1.2 release gate. `all` = every logical CPU of the machine. Run it ALONE on a quiet machine.
The JSON keeps every run's figures (the per-run range of the gate: each run's speedup must be >= 1.0), the median, and the machine banner.
Needs the Rust toolchain for the build stage (codegen-block); the measure stages do not."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import common as c

HERE = Path(__file__).resolve().parent


def run_py(args: list[str], log: Path) -> None:
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, "-s", str(HERE / "bench_compile.py"), *args], stdout=f, stderr=subprocess.STDOUT, check=False)
    if r.returncode:
        raise SystemExit(f"bench_compile {' '.join(args[:6])} failed ({r.returncode}); tail:\n{log.read_text()[-1500:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--libs", default="xgboost,lightgbm,ydf,ydf-rf")
    ap.add_argument(
        "--library-only-libs", default="", help="rows with no compiled side in the tree under test (1.1: sklearn-rf,xgboost-vector)"
    )
    ap.add_argument("--threads", default="1,all")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--size", default="full")
    ap.add_argument("--backends", default="flat,wave,codegen-block")
    ap.add_argument("--replicate-outputs", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    info = c.banner(
        "compile-only table, several runs", size=a.size, runs=a.runs, libs=a.libs, library_only=a.library_only_libs, note=a.note
    )
    base = ["--work", str(work), "--size", a.size, "--backends", a.backends]
    flag = ["--replicate-outputs"] if a.replicate_outputs else []
    if a.libs:
        run_py([*base, "--lib", a.libs, "--stage", "build", *flag], work / "build.log")
        print(f"built in {time.perf_counter() - t0:.0f} s", flush=True)
    if a.library_only_libs:  # trains the models of the library-only rows (no compile) and times nothing yet
        run_py([*base, "--lib", a.library_only_libs, "--stage", "build", "--library-only"], work / "build-lib.log")
    res: dict = {"info": info, "runs": a.runs, "threads": {}}
    for tspec in a.threads.split(","):
        n = os.cpu_count() or 1 if tspec == "all" else int(tspec)
        per_run, lib_only = [], {}
        for r in range(a.runs):
            if a.libs:
                j = work / f"run-t{n}-r{r}.json"
                run_py(
                    [*base, "--lib", a.libs, "--stage", "measure", "--threads", str(n), "--json", str(j), *flag],
                    work / f"run-t{n}-r{r}.log",
                )
                per_run.append(json.loads(j.read_text())["libs"])
                print(f"threads {n} run {r + 1}/{a.runs} done at {time.perf_counter() - t0:.0f} s", flush=True)
        if a.library_only_libs:
            j = work / f"libonly-t{n}.json"
            run_py(
                [*base, "--lib", a.library_only_libs, "--stage", "measure", "--library-only", "--threads", str(n), "--json", str(j)],
                work / f"libonly-t{n}.log",
            )
            lib_only = json.loads(j.read_text())["libs"]
        rows: dict = {}
        for lib in a.libs.split(",") if a.libs else []:
            for runs in zip(*[[row for row in pr[lib]["rows"] if "rows_per_s" in row] for pr in per_run if lib in pr], strict=True):
                be = runs[0]["backend"]
                rows[f"{lib}/{be}"] = {
                    "library_rows_per_s_runs": [pr[lib]["library_rows_per_s"] for pr in per_run],
                    "rows_per_s_runs": [x["rows_per_s"] for x in runs],
                    "speedup_runs": [x["speedup"] for x in runs],
                    "speedup_median": statistics.median(
                        x["rows_per_s"] / pr[lib]["library_rows_per_s"] for x, pr in zip(runs, per_run, strict=False)
                    ),
                    "max_err_worst": max(x["max_err"] for x in runs),
                    "status": sorted({x["status"] for x in runs}),
                    "artifact_mb": runs[0].get("artifact_mb"),
                }
        res["threads"][str(n)] = {"rows": rows, "library_only": lib_only}
    res["seconds_total"] = round(time.perf_counter() - t0)
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {a.out} in {res['seconds_total']} s")


if __name__ == "__main__":
    main()
