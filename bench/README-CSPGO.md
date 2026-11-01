# QuickForest CS-PGO measurement (31/10/2026)

Question: does context-sensitive PGO (CS-PGO, lld linker route) beat plain PGO on the QF standard build?
Rule in `docs/BUILD-STANDARD.md`: used by default only if it gains more than 1 % over PGO on most of the six models.

## Arms

| arm | build |
|---|---|
| a | QF 1.2 fast shape (preset 1.0: no LTO, 16 codegen units) |
| b | standard (preset 1.1: fat LTO, 1 unit, inline 3000), 5 chunks |
| c | b + PGO |
| d | c + CS-PGO (second instrumented build on the PGO-optimised code, merged profile) |
| c-lpl | control: c rebuilt through the same linker route (lld, level-1 profile only), so d is compared with a build that differs only by the CS profile |
| bs, cs | b and c as ONE crate (`--unit single`) |

Models: the six of the 1.2 table. Profile rows (`x-pgo.npy`) are disjoint from the timed rows (`x-speed.npy`). CS route proof: xgboost has 934 CS functions
(`llvm-profdata show --showcs`; plain `show` lists only the level-1 ones).

## Verdict

- **CS-PGO vs PGO (d vs c, d vs c-lpl): within +-1 % at 1 thread on all six models** (-0.4 to +0.3 %). That is below the 1 % rule.
- At 2 threads the differences are -2.0 to +4.2 % with no sign pattern (ydf -2.0 and sklearn-rf +4.2 on the same arms swing in opposite directions, c-lpl vs c differ by up to 6 % there): noise on a 2-core Celeron, not an effect.
- Cost: CS-PGO compile is 2 to 4 times PGO's (xgboost 713 s -> 1081 s, sklearn-rf 311 s -> 951 s), peak rustc RSS up to 1.07 GB.
- PGO itself (c vs b) is worth +2.2 to +9.8 % at 1 thread (+10.1 % xgboost at 2 threads). Standard over fast (b vs a): +0.1 to +0.8 % at 1 thread, i.e. none worth quoting.
- One crate vs chunks (bs/cs vs b/c): equal at 1 thread; at 2 threads ydf cs 321899 vs c 266336 is a single outlier cell (ydf d/c-lpl 2-thread cells also swing), the other five models are within noise.
- Measured on the measurable evidence the rule says NO. **Decision (31/10): CS-PGO stays in the standard**, because the product will be used with many threads and the multi-thread result is the one that matters; this machine has only 2 cores, so the 2-thread rows are the nearest we have and they show no loss, no clear gain. The 1-thread figures above are as measured; they do not show a gain.

## Machine and method

Intel Celeron N4020 @ 1.10 GHz, 2 logical CPUs, Linux 6.8, quiet slot. 20000 held-out rows, median of 3 runs, arm order rotated per run. Threads: `1` and `2` (= all).
Compile seconds are whole-arm; PGO arms include their profile run. The first ydf compile times overlapped a 28-minute test run (contended); ydf was re-timed alone (see the table notes below).
Driver: `bench/cspgo_table.py`.

Machine: Intel(R) Celeron(R) N4020 CPU @ 1.10GHz, 2 logical CPUs, Linux 6.8.0-138-generic. 20000 held-out rows, median of 3 runs, arm order rotated per run. Threads: `1` and `2` (= all).

### Speed, 1 thread (rows/s; speed-up over the source library in brackets)

| model | library | a fast 1.2 | b standard | c b+PGO | d c+CS-PGO | c-lpl (control) | b vs a % | c vs b % | d vs c % | d vs c-lpl % |
|---|---|---|---|---|---|---|---|---|---|---|
| xgboost | 12853 | 38284 (2.98x) | 38371 (2.99x) | 42116 (3.28x) | 42206 (3.28x) | 42092 (3.27x) | +0.2 | +9.8 | +0.2 | +0.3 |
| lightgbm | 11063 | 65237 (5.90x) | 65732 (5.94x) | 68851 (6.22x) | 68874 (6.23x) | 68913 (6.23x) | +0.8 | +4.7 | +0.0 | -0.1 |
| ydf | 29142 | 157361 (5.40x) | 158214 (5.43x) | 172896 (5.93x) | 172612 (5.92x) | 172295 (5.91x) | +0.5 | +9.3 | -0.2 | +0.2 |
| ydf-rf | 27779 | 94174 (3.39x) | 94274 (3.39x) | 96830 (3.49x) | 96707 (3.48x) | 96634 (3.48x) | +0.1 | +2.7 | -0.1 | +0.1 |
| sklearn-rf | 6836 | 65740 (9.62x) | 65924 (9.64x) | 67346 (9.85x) | 67226 (9.83x) | 66894 (9.79x) | +0.3 | +2.2 | -0.2 | +0.5 |
| xgboost-vector | 18471 | 50048 (2.71x) | 50338 (2.73x) | 53306 (2.89x) | 53113 (2.88x) | 53140 (2.88x) | +0.6 | +5.9 | -0.4 | -0.1 |

### Speed, 2 threads (rows/s; speed-up over the source library in brackets)

| model | library | a fast 1.2 | b standard | c b+PGO | d c+CS-PGO | c-lpl (control) | b vs a % | c vs b % | d vs c % | d vs c-lpl % |
|---|---|---|---|---|---|---|---|---|---|---|
| xgboost | 24774 | 60151 (2.43x) | 60579 (2.45x) | 66718 (2.69x) | 66201 (2.67x) | 66174 (2.67x) | +0.7 | +10.1 | -0.8 | +0.0 |
| lightgbm | 15343 | 100138 (6.53x) | 100172 (6.53x) | 105253 (6.86x) | 104243 (6.79x) | 105940 (6.90x) | +0.0 | +5.1 | -1.0 | -1.6 |
| ydf | 41817 | 292323 (6.99x) | 287027 (6.86x) | 275843 (6.60x) | 270447 (6.47x) | 291327 (6.97x) | -1.8 | -3.9 | -2.0 | -7.2 |
| ydf-rf | 41077 | 144874 (3.53x) | 143964 (3.50x) | 148751 (3.62x) | 147552 (3.59x) | 148248 (3.61x) | -0.6 | +3.3 | -0.8 | -0.5 |
| sklearn-rf | 6856 | 110490 (16.12x) | 113301 (16.53x) | 107712 (15.71x) | 112229 (16.37x) | 109689 (16.00x) | +2.5 | -4.9 | +4.2 | +2.3 |
| xgboost-vector | 30006 | 76875 (2.56x) | 76043 (2.53x) | 81263 (2.71x) | 82783 (2.76x) | 81769 (2.73x) | -1.1 | +6.9 | +1.9 | +1.2 |

### Build cost (compile seconds, whole arm; PGO arms include their profile run; peak rustc RSS in MB)

| model | a s | b s | c s | d s | c-lpl s | d peak MB | CS functions | level-1 functions | CS profile bytes |
|---|---|---|---|---|---|---|---|---|---|
| xgboost | 164.8 | 185.7 | 712.7 | 1080.6 | 603.9 | 997 | 934 | 904 | 2758152 |
| lightgbm | 36.0 | 52.3 | 154.9 | 366.7 | 144.6 | 750 | 931 | 902 | 1099248 |
| ydf | 27.3 | 49.7 | 137.7 | 295.1 | 130.3 | 415 | 322 | 293 | 517496 |
| ydf-rf | 33.8 | 49.9 | 126.3 | 396.5 | 113.8 | 661 | 332 | 302 | 682216 |
| sklearn-rf | 65.5 | 80.7 | 311.2 | 951.3 | 298.1 | 947 | 333 | 303 | 1312176 |
| xgboost-vector | 52.2 | 68.6 | 193.0 | 515.1 | 181.7 | 1069 | 933 | 902 | 1097584 |

### Training time vs compile + PGO time (same machine, same trainers, alone)

| model | train s | a compile s | b compile s | c compile+PGO s (profile run inside) | d compile+CS-PGO s | b / train | c / train |
|---|---|---|---|---|---|---|---|
| xgboost | 82.8 | 164.8 | 185.7 | 712.7 | 1080.6 | 2.2 | 8.6 |
| lightgbm | 27.4 | 36.0 | 52.3 | 154.9 | 366.7 | 1.9 | 5.7 |
| ydf | 134.1 | 27.3 | 49.7 | 137.7 | 295.1 | 0.4 | 1.0 |
| ydf-rf | 54.3 | 33.8 | 49.9 | 126.3 | 396.5 | 0.9 | 2.3 |
| sklearn-rf | 446.7 | 65.5 | 80.7 | 311.2 | 951.3 | 0.2 | 0.7 |
| xgboost-vector | 211.3 | 52.2 | 68.6 | 193.0 | 515.1 | 0.3 | 0.9 |

### One crate (unit single) vs 5 chunks, 1 and 2 threads (rows/s)

| model | a t1 | b t1 | bs t1 | c t1 | cs t1 | a t2 | b t2 | bs t2 | c t2 | cs t2 |
|---|---|---|---|---|---|---|---|---|---|---|
| xgboost | 38285 | 38359 | 38400 | 42160 | 42125 | 60546 | 60570 | 60948 | 65959 | 66857 |
| lightgbm | 65537 | 65610 | 65700 | 68702 | 68795 | 100232 | 99718 | 99655 | 104883 | 105180 |
| ydf | 156799 | 157720 | 158295 | 172777 | 172277 | 247421 | 247488 | 238905 | 266336 | 321899 |
| ydf-rf | 94197 | 94125 | 94072 | 96464 | 96637 | 148516 | 144658 | 144509 | 147804 | 149287 |
| sklearn-rf | 65697 | 66138 | 65998 | 67406 | 67655 | 106737 | 109339 | 111069 | 106513 | 107566 |
| xgboost-vector | 50112 | 50147 | 50037 | 53146 | 53099 | 76862 | 76108 | 76646 | 81637 | 81087 |


Notes:
- ydf re-timed alone (compile seconds): a 18.2, b 33.7, c 92.7, d 179.1. The first-run ydf figures in the build-cost and training tables (27.3 / 49.7 / 137.7 / 295.1) were contended; use the re-timed ones.
- Training time of the source model: same machine, trained alone by the same trainer. (compile + PGO) / training: b/train is the plain standard build, c/train adds the profile run. Trees trained by xgboost/lightgbm are cheap to train relative to a PGO build (c is 5.7 to 8.6 times training); for ydf, sklearn-rf and xgboost-vector the PGO build costs about as much as the training or less (0.7 to 1.0).
- CS-PGO compile is unconditional extra time on top of c: at 2 cores it is 2 to 4 times c.
