# Evaluator library: 16 codegen units or one

Decision of 31/10/2026 (BUILD-STANDARD rule 1): every build runs in parallel, so `libtreeflat` is built with fat LTO, 16 codegen units and inline threshold 3000. This note records what the change cost or gained; it is not a gate.

Method: `bench/lib_inline_table.py` (the two libraries alternate inside every run, 3 runs, median rows/s), the six models of the CS-PGO table, flat and wave backends, 1 and 2 threads, Intel Celeron N4020 (2 cores, 1.10 GHz), quiet machine. Both libraries come from the same source and toolchain (rustc 1.99.1); the only difference is `codegen-units` in `rust/Cargo.toml` (old = 1, new = 16). Raw: [cgu-table-2core.json](cgu-table-2core.json).

Build wall time of the evaluator crate (`cargo build --release`, same machine, nothing else running): 16 codegen units 14.3 s, one codegen unit 14.9 s.

| model / backend | threads | 1 unit rows/s (median) | 16 units rows/s (median) | 16 vs 1 | 1 unit, 3 runs | 16 units, 3 runs |
|---|---|---|---|---|---|---|
| xgboost flat | 1 | 29,108 | 28,937 | -0.6 % | 29,104 to 29,138 | 28,832 to 28,939 |
| xgboost wave | 1 | 28,866 | 28,484 | -1.3 % | 28,491 to 28,902 | 28,369 to 28,499 |
| lightgbm flat | 1 | 28,525 | 28,467 | -0.2 % | 28,508 to 28,527 | 28,417 to 28,489 |
| ydf flat | 1 | 98,961 | 99,009 | +0.0 % | 98,811 to 99,099 | 98,685 to 99,046 |
| ydf-rf flat | 1 | 82,166 | 82,796 | +0.8 % | 81,852 to 82,209 | 82,595 to 83,029 |
| sklearn-rf flat | 1 | 60,459 | 70,243 | +16.2 % | 51,758 to 70,271 | 70,188 to 70,276 |
| xgboost-vector flat | 1 | 28,812 | 28,860 | +0.2 % | 28,686 to 28,897 | 28,851 to 28,950 |
| xgboost flat | 2 | 43,894 | 44,123 | +0.5 % | 43,850 to 44,711 | 43,788 to 44,220 |
| xgboost wave | 2 | 43,769 | 44,310 | +1.2 % | 43,052 to 44,066 | 43,917 to 44,736 |
| lightgbm flat | 2 | 43,632 | 43,632 | +0.0 % | 42,428 to 44,328 | 43,073 to 44,338 |
| ydf flat | 2 | 152,790 | 151,292 | -1.0 % | 151,312 to 153,879 | 151,189 to 157,488 |
| ydf-rf flat | 2 | 126,368 | 126,882 | +0.4 % | 125,310 to 132,264 | 126,586 to 140,244 |
| sklearn-rf flat | 2 | 116,273 | 108,790 | -6.4 % | 107,897 to 116,860 | 108,010 to 132,542 |
| xgboost-vector flat | 2 | 43,817 | 43,660 | -0.4 % | 43,288 to 44,410 | 43,505 to 44,263 |

Reading: 12 of the 14 cells are within 1.3 % either way (noise). The two `sklearn-rf` cells move by +16.2 % and -6.4 % in opposite directions, and their own three runs spread by 10 to 26 % (one unit: 51,758 to 70,271 rows/s at one thread), so the median difference is not resolved by this method. Speed is therefore not a reason for one codegen unit; nothing here asks for it. This table covers the evaluator library only; for the generated code the earlier measurement is in [README-GENERATED-SHAPE.md](README-GENERATED-SHAPE.md) (one unit against 16: 1.003 and 1.005, inside the noise).
