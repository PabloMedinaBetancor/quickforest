#!/usr/bin/env python3
"""usage: python3 -s bench/foreign.py OUTDIR NAME 'SHELL COMMAND' -- runs the command, measures the CPU that OTHER processes used meanwhile (all busy CPU seconds
from /proc/stat minus the command's own user+system time, per second of wall time) and runs it again (up to 3 attempts) while that is above 0.3
cores. Every attempt is logged in OUTDIR/NAME.foreign.json; the log of the accepted (last) attempt is OUTDIR/NAME.log."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

if len(sys.argv) != 4:
    sys.exit(__doc__)
outd, name, cmd = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
tck = os.sysconf("SC_CLK_TCK")


def busy():
    f = open("/proc/stat").readline().split()[1:]
    v = [int(x) for x in f]
    return (sum(v[:3]) + sum(v[5:8])) / tck  # user nice system + irq softirq steal


attempts = []
rc = 1
for k in range(3):
    b0, t0 = busy(), time.time()
    with open(outd / f"{name}.log", "w") as log:
        p = subprocess.Popen(["bash", "-c", cmd], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        _, status, ru = os.wait4(p.pid, 0)
    wall, b1 = time.time() - t0, busy()
    own = ru.ru_utime + ru.ru_stime
    foreign = max(0.0, ((b1 - b0) - own) / wall)
    rc = os.waitstatus_to_exitcode(status)
    attempts.append(
        {
            "attempt": k + 1,
            "exit": rc,
            "wall_s": round(wall, 1),
            "own_cpu_s": round(own, 1),
            "foreign_cores": round(foreign, 2),
            "load_avg_1m": os.getloadavg()[0],
        }
    )
    (outd / f"{name}.foreign.json").write_text(json.dumps(attempts, indent=1))
    if rc != 0 or foreign <= 0.3 or time.time() > float(os.environ.get("QF_DEADLINE", "1e18")):
        break
sys.exit(rc)
