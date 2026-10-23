#!/usr/bin/env python3
"""Paired, interleaved timing of several builds of the SAME 10 % model, in ONE process (run it pinned or with N threads).

    python3 -s bench/pair.py --threads N --rounds R --out OUT.json NAME=CMP_DIR [NAME=CMP_DIR ...] [--twin NAME]

Each CMP_DIR is a compare work folder holding codegen-pgo.so and x-speed.npy. A twin is a byte copy of a build's folder timed as if it were another
build: its ratio to the original is the noise floor of the method. Each round times every build once (best of --reps passes over the batch), in an
order that rotates from round to round, so a drift of the machine hits every build alike. Writes every sample; the ratios and their 95 % interval are
made by pair_stats.py from the JSON.

Making the builds (one pruned model, several sets of build options; make the WORK folders first, `compare.py` checks the folder of `--json` before it makes `--work`): `scripts/compare.py PRUNED_DIR --data DATA.npz --backends flat,codegen-pgo
--decision argmax --threads 1 --work WORK/cmp --json WORK/table.json --stage build`, once per variant with the same PRUNED_DIR and DATA.npz and
different options (for example none for the default, `--opt shape=top`, `--opt target_cpu=native`, `--opt opt_level=2 --opt target_cpu=x86-64 --opt shape=top`, the earlier defaults);
WORK/cmp then holds `codegen-pgo.so` and `x-speed.npy`. To pin the process to one CPU, start it with `taskset -c N`.
"""

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import backends  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, required=True)
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--twin", help="name of the build to copy and time again as the noise-floor twin")
ap.add_argument("--out", required=True)
ap.add_argument("builds", nargs="+", help="NAME=CMP_DIR")
a = ap.parse_args()

dirs = dict(b.split("=", 1) for b in a.builds)
if a.twin:
    src = Path(dirs[a.twin])
    twin_dir = Path(tempfile.mkdtemp(prefix="quickforest-pair-")) / f"twin-of-{a.twin}"
    shutil.copytree(src, twin_dir)
    dirs[f"{a.twin}-twin"] = str(twin_dir)
be = backends.get("codegen-pgo")
preds, xs = {}, None
for name, d in dirs.items():
    preds[name] = be.load(Path(d) / "codegen-pgo.so")
    x = np.ascontiguousarray(np.load(Path(d) / "x-speed.npy"))
    xs = x if xs is None else xs
    assert xs.shape == x.shape and np.array_equal(np.nan_to_num(xs), np.nan_to_num(x)), f"{name}: another batch than the first build's"
names = list(dirs)
for n in names:  # warm-up, not counted: page-in of the library and of the batch
    preds[n].predict(xs, a.threads)
samples = {n: [] for n in names}
for r in range(a.rounds):
    k = r % len(names)
    order = names[k:] + names[:k]
    if (r // len(names)) % 2:
        order = order[::-1]
    for n in order:
        best = 1e30
        for _ in range(a.reps):
            t = time.perf_counter()
            preds[n].predict(xs, a.threads)
            best = min(best, time.perf_counter() - t)
        samples[n].append(len(xs) / best)
Path(a.out).write_text(
    json.dumps({"threads": a.threads, "rows": len(xs), "reps": a.reps, "rounds": a.rounds, "dirs": dirs, "rows_per_s": samples}, indent=1)
)
print({n: round(float(np.median(v))) for n, v in samples.items()})
