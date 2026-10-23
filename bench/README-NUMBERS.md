# Numbers: what the benchmark scripts reproduce, and what they do not

Everything below was measured on one machine in a slot reserved for the purpose, with nothing else running on purpose: an Intel Core i7-8550U CPU @ 1.80GHz
(4 cores, 8 threads, AVX2, 1.80 GHz base), x86-64 Linux 6.8, Python 3.10.13, numpy 2.2.6, xgboost 3.2.0, lightgbm 4.7.0, ydf 0.16.1, on 2026-10-23.
The prebuilt evaluator library is `bin/libtreeflat-x86_64.so`, SHA-256 `b0f290b1ab8ab312ce30110c07736e65aa435bee78eb0d89f2cac9ebbb87211e`; the
generated code is built by the build defaults of this release (`opt-level` 3, the `fast` shape, the portable x86-64-v2 CPU, profile-guided
with the selection rows) on the machine that times it. The batch is 20,000 rows. The same scripts build the models on any machine
(`--stage build`); the timing (`--stage measure`) needs a quiet one. Nothing measured on a shared machine is a figure here.

**Two ways to read every speed figure.** Each is the compiled forest against the source library's own `predict` on the same rows, and the two
sides always get the same number of threads:

- **all cores against all cores**: 8 threads each, one per logical CPU of this machine (xgboost `nthread=8`, LightGBM `num_threads=8`, YDF
  `num_threads=8`, QuickForest `predict(threads=8)`); this is what `predict` does by default here for a batch of this size. The machine has 4
  cores with 2 hardware threads each, so 8 threads share 4 cores; a machine with more or fewer cores will scale differently on both sides.
- **one thread against one thread**: both sides on one thread, the process pinned to one logical CPU (`--cpu 1`, `taskset`).

Each row was run three times; each speedup in the tables is the ratio of two medians (the median rows/s of the three QuickForest runs over the median rows/s of the three runs of xgboost, or of the library in section 2), not the median of three per-run ratios. The one-thread runs were pinned to one CPU: the three runs of a row
differ by up to 14 % for QuickForest and up to 20 % for xgboost. The all-cores rows were run a second time with the CPU used by every other process logged during each run (the
difference between the machine's busy CPU time and the benchmark's own, per second); a run during which others used more than 0.3 of a core was
thrown away and redone, and the log of every attempt is kept with the results (17 timed steps, the quickstart and the paired test included, 4 reruns, highest foreign CPU of a kept run 0.30 cores). The first pass at the all-cores rows is not used: it
ran with about one other core busy, which slows an 8-thread call that splits its rows evenly across the threads, and its xgboost figures
varied by 2x between runs.

## 1. End to end: one forest, two paths (`bench/bench_e2e.py`)

Forest: 5 folds x 67 trees per class x 3 classes, depth 14, 42 features, 60,000 rows, xgboost random forest; 1,999,767 nodes, 68.7 MB of model
files; xgboost loads it in about 0.06 s. The same forest goes through both paths:

- **exact compile, no pruning** (`--keep 100%`): all the trees, compiled; the answers are those of xgboost to the last bit (max error 0 on the
  20,000 timed rows and on the test sets of the run), so there is no agreement figure (100 % by construction);
- **prune to the size you choose** (`--keep 50%`, `25%`, `10%`; greedy on the selection rows), then compile: held-out agreement is the share
  of rows (not used for the selection) on which the pruned forest decides as the full one.

"Speedup" is QuickForest's rows/s over xgboost's rows/s **on the full forest**, so the pruned rows include the gain from the smaller
forest. "Load" is the time to open the compiled flat file (memory-mapped); "flat file" is its size (the generated-code backends write a
shared library as well). The `max error` column is against xgboost on the same (pruned) model: 0 everywhere, bit-identical. The `flat` rows are
the evaluator library reading the flat file (what `--backend flat` gives, and what `best` falls back to without rustc); the other row of each
keep is the generated code (`codegen-pgo` with the selection rows as the profile; `codegen-block` for the exact row, where a profile-guided build of 2.0M
nodes was not built).

### All cores (8) against all cores (8)

| keep | QuickForest backend | nodes | held-out agreement | xgboost, full forest, rows/s | QuickForest rows/s | speedup | flat file | load | max error | runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 % (no pruning) | flat | 1,999,767 | 100.0 % | 76,414 | 91,750 | 1.2x | 16.01 MB | 0.061 s | 0 | 3 |
| 100 % (no pruning) | codegen-block | 1,999,767 | 100.0 % | 76,414 | 115,954 | 1.5x | 16.01 MB | 0.061 s | 0 | 3 |
| 50 % | flat | 1,015,056 | 98.4 % | 73,475 | 207,220 | 2.8x | 8.12 MB | 0.035 s | 0 | 3 |
| 50 % | codegen-pgo | 1,015,056 | 98.4 % | 73,475 | 284,288 | 3.9x | 8.12 MB | 0.035 s | 0 | 3 |
| 25 % | flat | 507,515 | 97.4 % | 79,235 | 357,491 | 4.5x | 4.06 MB | 0.015 s | 0 | 3 |
| 25 % | codegen-pgo | 507,515 | 97.4 % | 79,235 | 570,560 | 7.2x | 4.06 MB | 0.015 s | 0 | 3 |
| 10 % | flat | 208,649 | 95.9 % | 76,645 | 871,698 | 11.4x | 1.67 MB | 0.035 s | 0 | 3 |
| 10 % | codegen-pgo | 208,649 | 95.9 % | 76,645 | 1,412,848 | 18.4x | 1.67 MB | 0.035 s | 0 | 3 |

### One thread against one thread

| keep | QuickForest backend | nodes | held-out agreement | xgboost, full forest, rows/s | QuickForest rows/s | speedup | flat file | load | max error | runs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 % (no pruning) | flat | 1,999,767 | 100.0 % | 12,021 | 31,263 | 2.6x | 16.01 MB | 0.032 s | 0 | 3 |
| 100 % (no pruning) | codegen-block | 1,999,767 | 100.0 % | 12,021 | 29,393 | 2.4x | 16.01 MB | 0.032 s | 0 | 3 |
| 50 % | flat | 1,015,056 | 98.4 % | 13,286 | 62,105 | 4.7x | 8.12 MB | 0.017 s | 0 | 3 |
| 50 % | codegen-pgo | 1,015,056 | 98.4 % | 13,286 | 62,532 | 4.7x | 8.12 MB | 0.017 s | 0 | 3 |
| 25 % | flat | 507,515 | 97.4 % | 13,701 | 128,173 | 9.4x | 4.06 MB | 0.034 s | 0 | 3 |
| 25 % | codegen-pgo | 507,515 | 97.4 % | 13,701 | 130,753 | 9.5x | 4.06 MB | 0.034 s | 0 | 3 |
| 10 % | flat | 208,649 | 95.9 % | 13,448 | 296,271 | 22.0x | 1.67 MB | 0.008 s | 0 | 3 |
| 10 % | codegen-pgo | 208,649 | 95.9 % | 13,448 | 332,860 | 24.8x | 1.67 MB | 0.008 s | 0 | 3 |

What these tables say:

- The pruned rows are what pruning buys: at 10 % of the trees the held-out agreement is 95.9 % and the end-to-end gain is 18.4x on all
  cores and 24.8x on one thread. At 50 % and 25 % the agreement is 98.4 % and 97.4 % and the gain
  3.9x and 7.2x on all cores (4.7x and 9.5x on one thread).
  Keeping more trees costs speed and buys agreement: the choice of `--keep` is yours.
- The exact path gains less, and nothing disagrees: 1.2x to 1.5x on all cores, 2.4x to 2.6x on one thread (generated code
  and flat file), with answers identical to xgboost's. On a forest this size the flat file (16.01 MB) is larger than the processor's cache
  (8 MB) and is read from memory on every row.
- Going from 1 to 8 threads, xgboost speeds up 5.7x and QuickForest 4.2x on the 10 % row, so the ratio is lower on all
  cores than on one thread (18.4x against 24.8x). The 8 threads share 4 cores; this was not measured further here. On a machine with more cores
  the library's own scaling decides the ratio: it is the number to measure on yours.

- Why these figures are lower than the ones measured earlier on another machine (2026-10-22, a 2-core Intel Celeron N4020: 43.8x on 2 cores and 55.0x on one thread at 10 %): that was another machine. The build options are not the cause: in the paired test below the earlier build
  defaults (`opt-level` 2, baseline x86-64) read 1.004 of the current ones on the same CPU. On one thread the Celeron timed xgboost on the full forest at 3,623 rows/s and
  QuickForest at 199,137; this machine times them at 13,448 and 332,860: xgboost gained 3.7x from the faster CPU and QuickForest 1.7x, so the ratio is lower.

Caveats, stated plainly:
- The 10 % rows depend on the data: the speedup holds only where the 10 % prune passes the agreement bar of YOUR data. That is a property of
  the data, and the next subsection shows how much it moves.
- The generated-code backend (`codegen-pgo`) is built with a profile taken on rows of the same synthetic data; the held-out rows used for the
  timing are not the rows of the profile (the scripts also time the profile rows and random rows; `compare.py` reports them).
- Three runs per row, on one machine, on one synthetic forest.

### Build options on the 10 % forest, paired

The same pruned 10 % forest (208,649 nodes) compiled with different build options and timed in ONE process, interleaved: each round times every
build once (best of 3 passes over the same 20,000-row batch), in an order that rotates between rounds, for 30 rounds. Ratios are to the default
build (`fast` shape, `opt-level` 3, portable x86-64-v2); the interval is a 95 % bootstrap interval of the median of the paired ratios.
The "twin" is a byte copy of the default build timed as if it were another build: its ratio is the noise floor of the method.

| build | 1 thread, ratio to default | all cores (8), ratio to default |
|---|---:|---:|
| default (`fast` shape, opt 3, x86-64-v2) | 1 (315,240 rows/s) | 1 (1,254,915 rows/s) |
| twin of the default (noise floor) | 0.996 (0.979 to 1.004) | 1.001 (0.999 to 1.004) |
| `--shape top` (one codegen unit everywhere) | 1.003 (0.995 to 1.018) | 1.005 (1.000 to 1.009) |
| `--native` (this CPU only) | 0.932 (0.926 to 0.948) | 0.955 (0.948 to 0.959) |
| earlier defaults (`opt-level` 2, baseline x86-64, one codegen unit per chunk) | 1.004 (0.978 to 1.018) | 1.000 (0.993 to 1.002) |

On this forest and this CPU the medians of the `top` shape and of the earlier defaults are within 0.5 % of the default's, which is the size of the
spread of the twin (0.4 %): the shape of the build, the optimisation level and the instruction-set level made no measurable difference here. `--native` ran 7 % slower on one thread and 4.5 % slower on all cores (ratio 0.955).
The answers of every build are bit-identical to the default's on the 20,000 timed rows (checked). The one-thread pair was timed in the first window; the all-cores pair was timed again in the second window with the other processes' CPU logged (0.3 cores or less).

To repeat the paired test: build the same pruned model once per set of build options with `scripts/compare.py` (same `PRUNED_DIR` and `DATA.npz`,
`--stage build`, a different `--work` folder each time, and the options `--opt shape=top`, `--opt target_cpu=native`,
`--opt opt_level=2 --opt target_cpu=x86-64 --opt shape=top` (the earlier defaults), or none for the default), then time the builds together and print the ratios:

```
mkdir -p WORK/default WORK/top      # compare.py checks the folder of --json before it makes --work
python3 -s scripts/compare.py PRUNED_DIR --data DATA.npz --backends flat,codegen-pgo --decision argmax --threads 1 --stage build --work WORK/default/cmp --json WORK/default/table.json
python3 -s scripts/compare.py PRUNED_DIR --data DATA.npz --backends flat,codegen-pgo --decision argmax --threads 1 --stage build --work WORK/top/cmp --json WORK/top/table.json --opt shape=top
taskset -c 1 python3 -s bench/pair.py --threads 1 --rounds 30 --twin default --out pair-t1.json default=WORK/default/cmp top=WORK/top/cmp   # one thread, pinned
python3 -s bench/pair.py --threads 8 --rounds 30 --twin default --out pair-t8.json default=WORK/default/cmp top=WORK/top/cmp                # every logical CPU of an 8-thread machine
python3 -s bench/pair_stats.py default pair-t1.json pair-t8.json      # median ratio, 95 % bootstrap interval, mean with its t interval
```

`bench/foreign.py OUTDIR NAME 'COMMAND'` wraps one timed command: it logs the CPU that other processes used while the command ran (all busy CPU time of
the machine from `/proc/stat` minus the command's own, per second) and runs the command again, up to three attempts, while that is above 0.3 cores;
`OUTDIR/NAME.foreign.json` keeps every attempt. It is how the "foreign CPU" figures of this file were taken. Run on a machine with nothing else
on it, it logs one attempt.

### Build time of the generated code on this machine

Compile seconds and the largest `rustc` process, from the build files (`build-*.json`). The build plans its parallel `rustc` jobs against the RAM
that is free, so the number of jobs differs between rows (the machine had other work running while these were built): times are not comparable between rows.

| model | nodes | build | seconds | jobs | largest rustc | generated library |
|---|---:|---|---:|---:|---:|---:|
| keep 10 % | 208,649 | codegen-pgo, `fast` shape | 156 | 3 | 608 MB | 3.42 MB |
| keep 10 % | 208,649 | codegen-pgo, `top` shape | 179 | 3 | 327 MB | 3.42 MB |
| keep 10 % | 208,649 | codegen-pgo, `--native` | 139 | 3 | 602 MB | 3.34 MB |
| keep 25 % | 507,515 | codegen-pgo, `fast` shape | 314 | 2 | 833 MB | 7.58 MB |
| keep 50 % | 1,015,056 | codegen-pgo, `fast` shape | 689 | 1 | 1003 MB | 14.56 MB |
| keep 100 % (no pruning) | 1,999,767 | codegen-block, `fast` shape | 432 | 1 | 1001 MB | 27.76 MB |

### How the generator was chosen (disclosure: it was chosen AFTER the first attempts failed the bar)

The first generator (`common.make_data`, kept unchanged, noise 0.35) draws a dense label `argmax(tanh(x@w) + ...) + noise`. Trees learn that
label poorly, and a forest cut to a fraction of its trees disagrees with the full one on many rows. At the README's scale (same forest size as above),
held-out agreement of the pruned model, by fraction of trees kept:

| Generator, noise | 10 % | 20 % | 30 % | 40 % | 50 % |
|---|---|---|---|---|---|
| dense, 0.35 (the original `--size readme` data) | 70.8 % | 77.1 % | 82.1 % | 85.3 % | 87.9 % |
| dense, 0.1 (lower noise only) | 74.7 % | 80.7 % | 84.9 % | 87.8 % | (not run) |

No keep reached the 90 % bar on either (the bar is `optimize`'s `--min-agreement`, 90 here), so `optimize` refused to emit a model and there is
no end-to-end figure on the dense data: pruning to 10 % does not reach 90 % agreement on the dense synthetic label. Lower noise alone was not
the cause. A small-scale probe (20,000 rows, 3 folds, 30 trees per class, depth 10; agreement at 10 % / 20 % keep) then tried a label made of a
few threshold rules on a few features, like a rule-based classifier:

| Generator, noise (small probe) | 10 % | 20 % |
|---|---|---|
| dense, 0.1 | 85.3 % | 89.5 % |
| sparse, 0.1 | 94.2 % | 95.9 % |
| sparse, 0.35 | 89.5 % | 92.7 % |

The sparse generator (`common.make_sparse_data`) at noise 0.1 was chosen because it passed, and the README-scale row in section 1 is its single
build (95.9 % at 10 %). It was selected so that a pruned model could pass the bar; it is not evidence that every real model prunes to 10 % at that
agreement. Your own data decides it, and `optimize` refuses to emit a model below your bar.

Command (build and measure on the same quiet machine, or build anywhere with rustc and measure on a quiet one; one work folder per `--keep`, the trained forest of the first one is copied to the others
so that it is trained once: `full/`, `sel.npz`, `training.json`):
```
python3 -s bench/bench_e2e.py --work DIR10 --size readme --data sparse --noise 0.1 --keep 10% --backend codegen-pgo --stage build
python3 -s bench/bench_e2e.py --work DIR10 --size readme --keep 10% --backend codegen-pgo --stage measure --cpu 1 --threads 1 --json OUT.json   # one thread
python3 -s bench/bench_e2e.py --work DIR10 --size readme --keep 10% --backend codegen-pgo --stage measure --threads 8 --json OUT.json          # every logical CPU of an 8-thread machine
```
(`--keep 100% --backend codegen-block` is the exact path; `--keep 50%` and `25%` as above. `--data dense --noise 0.35` reproduces the
refused runs; `--min-agreement 0` lets a dense build go on to compile.)

## 2. Compile only, no pruning: speedup over the source library's own predict (`bench/bench_compile.py`)

One model per library, 42 features, trained on the dense generator at noise 0.35 (no pruning, so there is no agreement figure; the compiled
forest is bit-compared with the flat walker and the max error against the library is shown). Rows/s, 20,000-row batch; the speedup is the
compiled rows/s over the library's rows/s on the same number of threads. Median of three runs. The library and the compiled forest are
timed in separate processes, one after the other. `codegen-block` is the generated code built by this release's defaults (`fast` shape, opt 3,
x86-64-v2); the wave engines are the prebuilt library's. This CPU's Downfall (GDS) microcode makes gathers slow, so `wave-avx2` ran the scalar wave engine here (`engine_used` 1 in the run's JSON): the two wave columns are the same engine, and the gap between them is run-to-run noise.

### All cores (8) against all cores (8)

| library | trees / nodes | library rows/s | flat | wave | wave-avx2 | codegen-block | max error vs library | runs |
|---|---|---:|---:|---:|---:|---:|---|---:|
| xgboost | 900 / 309,004 | 122,322 | 128,468 (1.05x) | 165,139 (1.35x) | 166,203 (1.36x) | 205,796 (1.68x) | 0 (bit-identical) | 3 |
| lightgbm | 900 / 112,500 | 60,350 | 122,711 (2.03x) | - | - | 339,697 (5.63x) | 5.4e-08 | 3 |
| ydf (GBT) | 291 / 55,603 | 171,171 | 298,377 (1.74x) | - | - | 1,075,376 (6.28x) | 5.4e-07 | 3 |
| ydf-rf | 300 (900 per-class trees compiled) / 226,980 | 119,988 | 139,896 (1.17x) | - | - | 237,390 (1.98x) | 8.3e-07 | 3 |

### One thread against one thread

| library | trees / nodes | library rows/s | flat | wave | wave-avx2 | codegen-block | max error vs library | runs |
|---|---|---:|---:|---:|---:|---:|---|---:|
| xgboost | 900 / 309,004 | 35,953 | 57,216 (1.59x) | 61,465 (1.71x) | 66,437 (1.85x) | 63,054 (1.75x) | 0 (bit-identical) | 3 |
| lightgbm | 900 / 112,500 | 14,266 | 53,987 (3.78x) | - | - | 101,109 (7.09x) | 5.4e-08 | 3 |
| ydf (GBT) | 291 / 55,603 | 36,291 | 192,626 (5.31x) | - | - | 238,189 (6.56x) | 5.4e-07 | 3 |
| ydf-rf | 300 (900 per-class trees compiled) / 226,980 | 33,644 | 62,202 (1.85x) | - | - | 66,047 (1.96x) | 8.3e-07 | 3 |

### Per-run range of the compile-only speedups

The cells above are ratios of medians; these are the speedups of the three runs themselves (in run order). On 8 threads the `flat` engine is within the
run-to-run noise of the library for xgboost (1.05x, 0.86x, 1.14x in the three runs) and a YDF random forest (2.23x, 0.92x, 0.75x in the three runs): single runs fall below 1x, so those cells are no measured gain. Every other cell stayed above 1x in every run.

| library | backend | 8 threads, speedup of each of the 3 runs | one thread, speedup of each of the 3 runs |
|---|---|---|---|
| LightGBM | `codegen-block` | 6.27x, 5.63x, 5.72x | 7.13x, 12.45x, 7.06x |
| LightGBM | `flat` | 2.39x, 2.03x, 2.00x | 3.78x, 6.89x, 3.75x |
| xgboost | `codegen-block` | 1.68x, 1.38x, 1.71x | 1.81x, 1.76x, 1.66x |
| xgboost | `flat` | 1.05x, 0.86x, 1.14x | 1.64x, 1.60x, 1.58x |
| YDF boosted trees | `codegen-block` | 4.74x, 7.60x, 6.01x | 6.02x, 6.56x, 7.12x |
| YDF boosted trees | `flat` | 2.59x, 2.09x, 1.67x | 4.82x, 5.40x, 5.83x |
| a YDF random forest | `codegen-block` | 2.84x, 1.98x, 1.68x | 1.96x, 1.71x, 2.11x |
| a YDF random forest | `flat` | 2.23x, 0.92x, 0.75x | 1.85x, 1.71x, 1.95x |

The wave engines refuse LightGBM and YDF models (extended nodes or linear leaves) by design; those use `flat` or `codegen-block`.
The script's `--json` output holds every number above.

```
python3 -s bench/bench_compile.py --work DIR --size full --lib xgboost,lightgbm,ydf,ydf-rf --backends flat,wave,wave-avx2,codegen-block --stage build
python3 -s bench/bench_compile.py --work DIR --size full --lib xgboost,lightgbm,ydf,ydf-rf --backends flat,wave,wave-avx2,codegen-block --stage measure --cpu 1 --threads 1 --json OUT.json   # one thread
python3 -s bench/bench_compile.py --work DIR --size full --lib xgboost,lightgbm,ydf,ydf-rf --backends flat,wave,wave-avx2,codegen-block --stage measure --threads 8 --json OUT.json          # every logical CPU of an 8-thread machine
```
Install lightgbm and ydf in the interpreter (the library timing needs the library).
