#!/usr/bin/env python3
"""CS-PGO measurement (31/10/2026): codegen-block (control), codegen-pgo (QF 1.2 level 1), codegen-cspgo (level 1 + CS-PGO by the linker
route) and codegen-pgo-lpl (level 1 built the same linker-plugin way: the fair control of level 2), on the six models of the 1.2 table, held-out
rows, 1 and all threads, `--runs` passes in a rotated arm order, medians.

    cspgo_table.py --work DIR --out table.json [--stage build|measure|all] [--libs ...] [--threads 1,all] [--runs 3]

Needs the patched scripts (backend_codegen.py with the cs stages) in the tree it runs from. Run it ALONE on a quiet machine."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import common as c
import numpy as np

ARMS = ["arm-a", "arm-b", "arm-c", "arm-d", "arm-c-lpl"]


def orders(arms):
    """Three pass orders: forward, reverse, rotated by half (each arm meets different neighbours)."""
    h = len(arms) // 2
    return [arms, arms[::-1], arms[h:] + arms[:h]]


ALL_LIBS = "xgboost,lightgbm,ydf,ydf-rf,sklearn-rf,xgboost-vector"


def model_dir(work: Path, lib: str):
    return work / lib / f"model-{lib}", work / lib / "sel.npz"


def build_stage(work: Path, libs: list[str], arms: list[str]) -> None:
    for lib in libs:
        t = time.perf_counter()
        folder, sel = model_dir(work, lib)
        os.environ.pop("QF_REPLICATE_OUTPUTS", None)
        c.compare(folder, sel, work / lib / "cmp", ",".join(arms), work / lib / "table.json", stage="build", threads=1)
        print(f"{lib}: built in {time.perf_counter() - t:.0f} s", flush=True)


def verify(work: Path, lib: str, arms: list[str]) -> dict:
    """Bit-identity of every arm to codegen-block on the held-out speed rows, the exactness sets and the profile-free random rows."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    import backends

    cmp = work / lib / "cmp"
    res = {}
    x = np.load(cmp / "x-speed.npy")
    ref = None
    for arm in arms:
        art = cmp / f"{arm}.so"
        if not art.is_file():
            res[arm] = "missing"
            continue
        be = backends.get(arm)
        pred = be.load(art)
        y = np.asarray(pred.predict(x, threads=1))
        pred.close() if hasattr(pred, "close") else None
        if ref is None:
            ref = y
            res[arm] = {"speed_rows": len(x), "control": True}
            continue
        ok = ref.shape == y.shape and ref.dtype == y.dtype and ref.tobytes() == y.tobytes()
        sets = {}
        for s in ("data", "random"):
            a, b = cmp / f"{arms[0]}-{s}.npy", cmp / f"{arm}-{s}.npy"
            if a.is_file() and b.is_file():
                aa, bb = np.load(a), np.load(b)
                sets[s] = bool(aa.shape == bb.shape and aa.tobytes() == bb.tobytes())
        res[arm] = {"speed_rows": len(x), "bit_identical_speed_rows": bool(ok), "sets_identical": sets}
    return res


def measure_stage(work: Path, libs: list[str], threads: list[str], runs: int, arms: list[str], out: Path) -> dict:
    t0 = time.perf_counter()
    res: dict = {"info": c.banner("CS-PGO measurement", libs=",".join(libs), runs=runs, arms=",".join(arms)), "threads": {}}
    for tspec in threads:
        n = (os.cpu_count() or 1) if tspec == "all" else int(tspec)
        per_run = []
        for r in range(runs):
            order = orders(arms)[r % 3]
            one = {}
            for lib in libs:
                folder, sel = model_dir(work, lib)
                j = work / lib / f"cs-run-t{n}-r{r}.json"
                os.environ.pop("QF_REPLICATE_OUTPUTS", None)
                one[lib] = c.compare(folder, sel, work / lib / "cmp", ",".join(order), j, stage="measure", threads=n)
            per_run.append(one)
            print(f"threads {n} run {r + 1}/{runs} (order {','.join(order)}) done at {time.perf_counter() - t0:.0f} s", flush=True)
        rows = {}
        for lib in libs:
            lib_rates = [pr[lib]["library_rows_per_s"] for pr in per_run]
            lib_med = statistics.median(lib_rates)
            for arm in arms:
                vals, sp = [], []
                status = set()
                for pr in per_run:
                    row = next((x for x in pr[lib]["rows"] if x["backend"] == arm), None)
                    if row is None or "rows_per_s" not in row:
                        status.add((row or {}).get("status", "missing"))
                        continue
                    vals.append(row["rows_per_s"])
                    sp.append(row["rows_per_s"] / pr[lib]["library_rows_per_s"])
                    status.add(row["status"])
                rows[f"{lib}/{arm}"] = {
                    "rows_per_s_runs": vals,
                    "rows_per_s": statistics.median(vals) if vals else None,
                    "library_rows_per_s_runs": lib_rates,
                    "library_rows_per_s": lib_med,
                    "speedup_vs_library": statistics.median(sp) if sp else None,
                    "status": sorted(status),
                }
            ctl = rows[f"{lib}/{arms[0]}"]["rows_per_s"]
            for arm in arms:
                v = rows[f"{lib}/{arm}"]["rows_per_s"]
                rows[f"{lib}/{arm}"]["vs_block_pct"] = None if not (v and ctl) else round((v / ctl - 1) * 100, 2)
        res["threads"][str(n)] = {"rows": rows}
    res["verify"] = {lib: verify(work, lib, arms) for lib in libs}
    builds = {}
    for lib in libs:
        for arm in arms:
            f = work / lib / "cmp" / f"build-{arm}.json"
            if f.is_file():
                builds[f"{lib}/{arm}"] = json.loads(f.read_text())
        st = work / lib / "cmp" / "arm-d.cs-stats.json"
        if st.is_file():
            builds[f"{lib}/cs-stats"] = json.loads(st.read_text())
    res["builds"] = builds
    res["seconds_total"] = round(time.perf_counter() - t0)
    out.write_text(json.dumps(res, indent=1, default=str))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=("build", "measure", "all"), default="all")
    ap.add_argument("--libs", default=ALL_LIBS)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--threads", default="1,all")
    ap.add_argument("--runs", type=int, default=3)
    a = ap.parse_args()
    work, libs, arms = Path(a.work), a.libs.split(","), a.arms.split(",")
    if a.stage in ("build", "all"):
        build_stage(work, libs, arms)
    if a.stage in ("measure", "all"):
        measure_stage(work, libs, a.threads.split(","), a.runs, arms, Path(a.out))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
