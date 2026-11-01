# QF 1.1 baseline table (2-core machine)

Machine: Intel(R) Celeron(R) N4020 CPU @ 1.10GHz, 2 logical CPUs, Linux 6.8.0-138-generic, Python 3.10.13, numpy 2.2.6, xgboost 3.2.0, lightgbm 4.7.0, ydf 0.16.1.
Date 2026-10-31 (re-run 12:21-13:30 machine clock after an earlier run overlapped another job). Model size `full`, batch 20000 rows, 3 runs (median speedup). Tree: QF 1.1 code, bench/ of the release tree. Run alone on the 2-core machine; `all` threads = 2 here (os.cpu_count()).
Speedup = compiled rows/s over the source library's own predict at the same thread count. Max error is the worst over the runs; all rows status ok.

Command: `python3 -s bench/run_table.py --work W --out baseline-1.1-2core.json --libs xgboost,lightgbm,ydf,ydf-rf --library-only-libs sklearn-rf,xgboost-vector --threads 1,all --runs 3` (1573 s total).

## 1 thread(s)

| library/backend | library rows/s (median) | compiled rows/s (median) | speedup (median) | max err | artifact MB |
|---|---|---|---|---|---|
| xgboost/flat | 13,061 | 29,147 | 2.24 | 0.0e+00 | 2.48 |
| xgboost/wave | 13,061 | 28,409 | 2.18 | 0.0e+00 | 2.48 |
| xgboost/codegen-block | 13,061 | 38,316 | 2.93 | 0.0e+00 | 5.3 |
| lightgbm/flat | 10,771 | 28,540 | 2.66 | 5.4e-08 | 0.91 |
| lightgbm/codegen-block | 10,771 | 65,650 | 6.06 | 5.4e-08 | 2.69 |
| ydf/flat | 28,870 | 99,159 | 3.44 | 5.4e-07 | 0.45 |
| ydf/codegen-block | 28,870 | 157,139 | 5.45 | 5.4e-07 | 1.51 |
| ydf-rf/flat | 27,726 | 32,119 | 1.16 | 8.3e-07 | 1.82 |
| ydf-rf/codegen-block | 27,726 | 48,676 | 1.75 | 8.3e-07 | 1.73 |

Library-only rows (sklearn-rf, xgboost-vector) are left out: the harness reported 24,213 and 24,580,046 rows/s (1 thread), against 6,900 and 18,699 in the 1.2 table for the same libraries, and the gap was not explained. The 1.2 table has the library rates of those two.

## 2 thread(s)

| library/backend | library rows/s (median) | compiled rows/s (median) | speedup (median) | max err | artifact MB |
|---|---|---|---|---|---|
| xgboost/flat | 24,759 | 44,084 | 1.78 | 0.0e+00 | 2.48 |
| xgboost/wave | 24,759 | 43,989 | 1.80 | 0.0e+00 | 2.48 |
| xgboost/codegen-block | 24,759 | 60,099 | 2.45 | 0.0e+00 | 5.3 |
| lightgbm/flat | 15,301 | 43,598 | 2.85 | 5.4e-08 | 0.91 |
| lightgbm/codegen-block | 15,301 | 99,583 | 6.51 | 5.4e-08 | 2.69 |
| ydf/flat | 41,607 | 151,860 | 3.64 | 5.4e-07 | 0.45 |
| ydf/codegen-block | 41,607 | 292,399 | 6.68 | 5.4e-07 | 1.51 |
| ydf-rf/flat | 41,218 | 49,513 | 1.20 | 8.3e-07 | 1.82 |
| ydf-rf/codegen-block | 41,218 | 76,825 | 1.88 | 8.3e-07 | 1.73 |

Library-only rows (sklearn-rf, xgboost-vector) are left out: the harness reported 24,213 and 24,580,046 rows/s (1 thread), against 6,900 and 18,699 in the 1.2 table for the same libraries, and the gap was not explained. The 1.2 table has the library rates of those two.
