# Evaluator library: inline threshold 3000 or not

Question: does building `libtreeflat` with `-C llvm-args=-inline-threshold=3000` (rust/.cargo/config.toml, BUILD-STANDARD rule 1) change the speed of the flat and wave backends?

Method: `bench/lib_inline_table.py`; the six models of the CS-PGO table, same rows, 1 and 2 threads (Intel Celeron N4020, 2 cores, quiet machine), 3 runs, the two libraries alternating inside every run, median rows/s. Both libraries are built from the same source (the 1.2 release tree) with the same toolchain (rustc 1.99.1), fat LTO, codegen-units 1; the only difference is the flag (the first has `rust/.cargo/config.toml`, the second does not; `cargo -v` shows the flag only in the first). Raw data: `inline-table-fb3ad80.json`. Wave exists for xgboost forests only.

| model / backend | threads | without flag rows/s | with flag rows/s | with vs without |
|---|---|---|---|---|
| xgboost flat | 1 | 28,902 | 29,122 | +0.8 % |
| xgboost wave | 1 | 28,283 | 28,381 | +0.3 % |
| lightgbm flat | 1 | 28,385 | 28,466 | +0.3 % |
| ydf flat | 1 | 98,556 | 98,346 | -0.2 % |
| ydf-rf flat | 1 | 82,568 | 81,995 | -0.7 % |
| sklearn-rf flat | 1 | 70,240 | 69,884 | -0.5 % |
| xgboost-vector flat | 1 | 28,755 | 28,774 | +0.1 % |
| xgboost flat | 2 | 44,105 | 44,874 | +1.7 % |
| xgboost wave | 2 | 43,549 | 44,155 | +1.4 % |
| lightgbm flat | 2 | 43,251 | 43,562 | +0.7 % |
| ydf flat | 2 | 151,976 | 150,511 | -1.0 % |
| ydf-rf flat | 2 | 126,256 | 123,761 | -2.0 % |
| sklearn-rf flat | 2 | 116,683 | 115,305 | -1.2 % |
| xgboost-vector flat | 2 | 43,974 | 44,021 | +0.1 % |

Result: within +-2 %, sign mixed: no measurable effect of the flag on the evaluator at 1 or 2 threads (an earlier run on the 1.2 release library, before the last evaluator format change, gave the same picture within +-3 %). The flag stays because BUILD-STANDARD rule 1 asks for it (decision of 31/10), not because it speeds the evaluator up; no README figure may be attributed to it.

Chunks or one crate (`DEFAULT_COMPILE_UNIT`): measured in `README-CSPGO.md` (bs/cs against b/c): equal at 1 thread, noise at 2 threads (one outlier cell). One crate needs one big rustc process (more RAM) and falls back to chunks when RAM is short, so the default stays `chunks`.
