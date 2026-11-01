# QF 1.2 table (2-core machine)

Machine: Intel(R) Celeron(R) N4020 CPU @ 1.10GHz, 2 logical CPUs, Linux 6.8.0-138-generic, Python 3.10.13, numpy 2.2.6, xgboost 3.2.0, lightgbm 4.7.0, ydf 0.16.1. Date 2026-10-31 (re-run on a quiet machine). `full` size, 20,000 rows, 3 runs (median), run alone. Prebuilt library bin/libtreeflat-x86_64.so (sha256 f42293ea...). `all` threads = 2 here.
Speedup = compiled rows/s over the source library's own predict at the same thread count. Column `speedup 1.1`: the 1.1 baseline on the same machine ([README-BASELINE-1.1.md](README-BASELINE-1.1.md)); 1.2 adds sklearn-rf and xgboost-vector compiled rows (no compiled side in 1.1).

Command: `python3 -s bench/run_table.py --work W --out table-1.2-2core.json --libs xgboost,lightgbm,ydf,ydf-rf,sklearn-rf,xgboost-vector --threads 1,all --runs 3` (1912 s total).

## 1 thread(s)

| library/backend | library rows/s | compiled rows/s | speedup 1.2 | speedup 1.1 | max err | artifact MB |
|---|---|---|---|---|---|---|
| xgboost/flat | 12,798 | 28,660 | 2.24 | 2.24 | 0.0e+00 | 2.48 |
| xgboost/wave | 12,798 | 28,338 | 2.20 | 2.18 | 0.0e+00 | 2.48 |
| xgboost/codegen-block | 12,798 | 38,216 | 2.99 | 2.93 | 0.0e+00 | 5.3 |
| lightgbm/flat | 10,940 | 28,391 | 2.60 | 2.66 | 5.4e-08 | 0.91 |
| lightgbm/codegen-block | 10,940 | 65,380 | 6.00 | 6.06 | 5.4e-08 | 2.69 |
| ydf/flat | 26,016 | 98,614 | 3.79 | 3.44 | 5.4e-07 | 0.45 |
| ydf/codegen-block | 26,016 | 157,000 | 6.03 | 5.45 | 5.4e-07 | 1.51 |
| ydf-rf/flat | 27,688 | 82,709 | 2.98 | 1.16 | 8.3e-07 | 1.06 |
| ydf-rf/codegen-block | 27,688 | 93,981 | 3.40 | 1.75 | 8.3e-07 | 2.07 |
| sklearn-rf/flat | 6,900 | 70,530 | 10.24 | n/a (library only) | 6.1e-08 | 2.76 |
| sklearn-rf/codegen-block | 6,900 | 65,509 | 9.52 | n/a (library only) | 6.1e-08 | 4.27 |
| xgboost-vector/flat | 18,699 | 28,729 | 1.54 | n/a (library only) | 0.0e+00 | 2.02 |
| xgboost-vector/codegen-block | 18,699 | 50,132 | 2.68 | n/a (library only) | 0.0e+00 | 3.44 |

## 2 thread(s)

| library/backend | library rows/s | compiled rows/s | speedup 1.2 | speedup 1.1 | max err | artifact MB |
|---|---|---|---|---|---|---|
| xgboost/flat | 24,691 | 44,165 | 1.80 | 1.78 | 0.0e+00 | 2.48 |
| xgboost/wave | 24,691 | 44,505 | 1.80 | 1.80 | 0.0e+00 | 2.48 |
| xgboost/codegen-block | 24,691 | 60,554 | 2.42 | 2.45 | 0.0e+00 | 5.3 |
| lightgbm/flat | 15,482 | 43,139 | 2.79 | 2.85 | 5.4e-08 | 0.91 |
| lightgbm/codegen-block | 15,482 | 98,478 | 6.45 | 6.51 | 5.4e-08 | 2.69 |
| ydf/flat | 43,252 | 152,068 | 3.53 | 3.64 | 5.4e-07 | 0.45 |
| ydf/codegen-block | 43,252 | 268,413 | 6.25 | 6.68 | 5.4e-07 | 1.51 |
| ydf-rf/flat | 41,198 | 125,823 | 3.09 | 1.20 | 8.3e-07 | 1.06 |
| ydf-rf/codegen-block | 41,198 | 147,491 | 3.56 | 1.88 | 8.3e-07 | 2.07 |
| sklearn-rf/flat | 6,845 | 116,454 | 17.28 | n/a (library only) | 6.1e-08 | 2.76 |
| sklearn-rf/codegen-block | 6,845 | 110,615 | 16.29 | n/a (library only) | 6.1e-08 | 4.27 |
| xgboost-vector/flat | 29,858 | 44,194 | 1.48 | n/a (library only) | 0.0e+00 | 2.02 |
| xgboost-vector/codegen-block | 29,858 | 75,784 | 2.55 | n/a (library only) | 0.0e+00 | 3.44 |
