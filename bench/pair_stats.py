#!/usr/bin/env python3
"""Ratios of the builds in pair.py JSON files, paired by round: median ratio and a 95 % interval (bootstrap of the median, 10,000 resamples)
and the t-interval of the mean log ratio. usage: python3 -s bench/pair_stats.py REFERENCE_NAME FILE.json [FILE.json ...]"""

import json
import sys

import numpy as np

ref = sys.argv[1]
rng = np.random.default_rng(1)
for f in sys.argv[2:]:
    d = json.load(open(f))
    s = {k: np.array(v) for k, v in d["rows_per_s"].items()}
    print(
        f"\n{f}: {d['threads']} thread(s), {d['rounds']} rounds, best of {d['reps']} passes of {d['rows']:,} rows; reference {ref} median {np.median(s[ref]):,.0f} rows/s"
    )
    for n, v in s.items():
        if n == ref:
            continue
        lr = np.log(v / s[ref])
        boots = np.median(rng.choice(lr, (10000, len(lr))), axis=1)
        lo, hi = np.exp(np.percentile(boots, [2.5, 97.5]))
        m = lr.mean()
        se = lr.std(ddof=1) / np.sqrt(len(lr))
        t = 2.045 if len(lr) >= 30 else {2: 12.7, 3: 4.30, 4: 3.18, 5: 2.78, 10: 2.26, 20: 2.09}.get(len(lr), 2.0)
        print(
            f"  {n} / {ref}: median {np.exp(np.median(lr)):.4f}  (95 % bootstrap {lo:.4f} to {hi:.4f}); mean {np.exp(m):.4f} (95 % t {np.exp(m - t * se):.4f} to {np.exp(m + t * se):.4f}); {n} median {np.median(v):,.0f} rows/s"
        )
