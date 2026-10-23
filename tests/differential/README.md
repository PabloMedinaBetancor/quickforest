# Differential test

`run_differential.py` builds thousands of small random models, compiles each one with every public backend, and compares the compiled
answers with the source library's own `predict` (xgboost, LightGBM, YDF) and with each other. A failure is a model, a backend and a number.

    python3 -s tests/differential/run_differential.py --mode quick     # 36 cases, ~1 minute, runs in CI
    python3 -s tests/differential/run_differential.py --mode full --cases 3000 --workers 1 --seed0 1 --codegen-every 12 --pgo-every 60
    python3 -s tests/differential/run_differential.py --case 224 --seed0 1       # replay one case: prints every check and keeps the files

Needs numpy, xgboost, lightgbm, ydf, pandas and scikit-learn (a missing library skips its cases and the summary says so) and rustc for the generated-code backends
(skipped with a note when absent). `--out DIR` holds `results.jsonl` (one line per finished case; a full run resumes from it, but only from results of the
same `--seed0` and the same version of `run_differential.py`: a case number alone is not a model; `--fresh` forgets them all), `failures.txt` (one line per failed check) and `summary.md`. Scratch files go to `--work`. Synthetic data only.
Exit code 1 when any check failed or a case crashed. Rust compile time dominates the full run: `--codegen-every N` also builds
codegen-block on one case in N and `--pgo-every N` codegen-pgo on one in N (about 80 minutes for 3,000 cases at 12 and 60 with one worker; 32 minutes with three workers on an 8-thread machine).

## What a case is

Everything is drawn from `numpy.random.default_rng([seed0, case])`, so a case number plus `--seed0` is a complete bug report.

- Format: xgboost (every third case), LightGBM, YDF.
- Provenance: one case in six of xgboost and of LightGBM is trained the way people really train, not through the native call on a bare matrix
  (quick mode has two of each). Flavor 0 goes through pandas: an xgboost `DMatrix` built from a DataFrame (feature names and types are stored
  in the file; one name has a space and a non-ASCII letter), a LightGBM `Dataset` with `category` columns (the file ends with a
  `pandas_categorical` line). Flavor 1 is a scikit-learn wrapper fitted on a DataFrame with early stopping (`XGBClassifier`, `XGBRegressor`,
  `LGBMClassifier`, `LGBMRegressor`; the xgboost file then carries `best_iteration`, `best_score` and the rounds past the best). Such a case
  also prunes each model file and reloads the pruned copy with its own library through the default path (`Booster.predict`, the wrapper's
  `load_model` + `predict`, `lightgbm.Booster(model_file).predict`), against the compiled pruned forest: the `written` row. Without pandas
  (or, for flavor 1, scikit-learn) the case is skipped and `summary.md` counts it with that reason.
- Model: regression, binary or multiclass (and per library poisson, gamma, tweedie, quantile, ranking); xgboost with the hist, exact
  and approx tree methods and `gamma` 0 to 4 (exact keeps the splits it dropped in the model's arrays); boosted or random forest; 1-3 k-fold
  members averaged as output or as margin; 1 to 150+ trees; stumps to depth 12 or 255 leaves; NaN in the data (0, 5, 30 %); constant and collinear
  columns; LightGBM categorical splits, `zero_as_missing`, linear leaves; YDF winner-take-all and averaged forests.
- Rows: normal, NaN, exact zeros, the zero bucket (1e-36), +-1e6, +-3e38 (no linear leaf), unseen categories, and rows built from the
  model's own thresholds (exactly, and one float32 step either side).
- Compiled: flat walker (engine 0), wave (1) and wave-avx2 (2) where the file can hold them, codegen-block and codegen-pgo on a share of
  the cases, with the opt-in `--f64-leaves` (LightGBM) and `--float32-sums` (YDF); float32 and float64 answers; no margin and a random
  base margin; 1 and 3 threads.

## What is checked

(a) compiled against the library, within the tolerance table below; (b) every backend bit-identical to the walker; (c) 3 threads equal 1
thread; (d) the float32 answer equals the float64 answer rounded to float32; (e) the engines that cannot hold a file refuse it with
`EngineRefused` or a one-line `ValueError`, never a crash.

## Tolerance table

Error = |compiled - library| / max(1, |library|), maximum over rows and outputs. "exact" means 0.0. A library answer beyond float32's range (above FLT_MAX in magnitude) has no float32 twin: there the compiled answer must be +-inf or +-FLT_MAX with the library's sign (error 0), and any other answer keeps its error. `B`, `C` and `N` are the proven
float32 bounds computed per model from its leaves (see `forest_scale()`). The same table is printed in `summary.md`.

| key | what | float32 answer | float64 answer | claim |
|---|---|---|---|---|
| xgb | xgboost regression / binary / multi:softprob, no margin (folds averaged as output or as margin) | exact (0) | exact (0) | docs/REFERENCE.md: xgboost's own arithmetic is float32 (the softmax normaliser is summed in double, as xgboost's common::Softmax does), the f64 answer is its float32 result widened |
| xgb+bm | xgboost, base margin | max(1e-6, S) | max(1e-6, S) | docs/REFERENCE.md: with a base margin xgboost's float32 order decides. S = N x 2^-24 x (B + init + 3) x 1.01, the bound of a sequential float32 sum of N trees (measured 1.2e-6 on an 80-tree model with answers near 100) |
| lgb | LightGBM, default file (float32 leaves), identity objective | 6.1e-8 | exact (0); 1e-12 with a margin | docs/REFERENCE.md: the default file stores each leaf rounded to float32 after the factors the compile folds in (sigmoid s, rf 1/iterations). The reference is the library's own Booster on a copy of the model whose leaves are folded and rounded the same way (`lgb_float32_leaf_files`): float32 of its own leaves, summed in double in tree order, so the answer is bit-identical to it, like --f64-leaves; f32 = half an ulp of the answer; the margin is added in a different order than numpy's raw + margin |
| lgb-tr | LightGBM, default file, sigmoid / softmax / exp | 6.1e-8 | 1e-12 | same, but the transform is numpy's exp against the evaluator's (an ulp of a double) |
| lgb-f64l | LightGBM --f64-leaves, identity objective | 6.1e-8 | exact (0); 1e-12 with a margin | docs/REFERENCE.md: bit-identical to Booster.predict; f32 = half an ulp of the answer; the margin is added in a different order than numpy's raw + margin |
| lgb-f64l-tr | LightGBM --f64-leaves, sigmoid / softmax / exp | 6.1e-8 | 1e-12 | same, but the transform is numpy's exp against the evaluator's (an ulp of a double) |
| lgb-lin | LightGBM linear leaves, identity objective | 6.1e-8 | exact (0); 1e-12 with a margin; on aarch64 / arm64 also 1e-12 x (|answer| + sum of the terms' magnitudes) / max(1, |answer|) per row | docs/REFERENCE.md: constants and coefficients stored in f64, 0.0 against Booster.predict. A plain (non-linear) tree inside a linear forest, e.g. the one-leaf tree of a constant target, keeps its leaf in float32 without --f64-leaves: the reference rounds that tree's leaves the same way (as for the key lgb). Averaging folds or adding a base margin is exact only to 8 x 2^-52 x S per row, S = the sum of |every tree's value| (the largest partial sum: a 1e6 input times a coefficient). On aarch64 / arm64 a row also gets an allowance of 1e-12 x (|answer| + P) / max(1, |answer|), P = the sum over the trees of |constant| + the sum of |coefficient x input| of the leaf the row reaches (a platform rule that scales with the terms, not a looser bar elsewhere): LightGBM's aarch64 wheel fuses const + coeff x x into one multiply-add (GCC's default there), so the library's own answer differs by one rounding of the largest partial sum (about 1e-16 of it) from the evaluator's, which never fuses, and from LightGBM's x86 build; with 1e6 inputs and an answer near zero that is far above a fixed 1e-12 |
| lgb-lin-tr | LightGBM linear leaves, sigmoid / softmax / exp | 6.1e-8 | 1e-12 | same, with the transform |
| ydf-dbl | YDF, double sums (default): the documented float32-summation class | C + 2.6e-7 | C + 2e-7 | C = N x 2^-24 x (|init| + sum of max abs leaf) x 1.01: bound of YDF's own sequential float32 sum; reported separately from the exact rows |
| ydf-f32s-reg | YDF --float32-sums, regression | exact (0) without margin; 6.1e-8 with margin | exact (0), one fold; 6.1e-8 for several folds | docs/REFERENCE.md: reproduces model.predict bit for bit; the reference rounds to float32 where the margin or the fold average is added |
| ydf-f32s-cls | YDF --float32-sums, classification (and Poisson) | boosted 2.4e-7, random forest 5e-7 | boosted 2.4e-7, random forest 5e-7 | docs/REFERENCE.md: the float32 sums are YDF's own bit for bit, the probability is not: YDF's float32 transform differs by an ulp or three (boosted trees: measured 1.8e-7); a random forest's leaves are divided by the number of trees before the float32 sum here and after it in YDF (measured 3.9e-7 at 30 trees) |
| backend | any backend against the flat walker | exact (0) | exact (0) | docs/REFERENCE.md: all backends bit-identical to the walker |
| threads | 3 threads against 1 thread | exact (0) | exact (0) | rows are independent |
| round | float32 answer against float64 answer rounded to float32 | exact (0) | - | every file: the f32 answer is the f64 one rounded |
| written | a pruned model file (real-world provenance cases) reloaded by its own library and predicted through the default path (xgboost Booster.predict and the scikit-learn wrapper, LightGBM Booster(model_file).predict) against the compiled pruned forest | xgboost exact (0); LightGBM as lgb-f64l | xgboost exact (0); LightGBM as lgb-f64l | docs/REFERENCE.md: prune writes a normal model file of the same format; a user reads it back with the library's own predict |
| refusal | wave engines on a file they cannot hold | EngineRefused | EngineRefused | docs/REFERENCE.md: raises EngineRefused, never a crash or a wrong answer |

## Reading a failure

A line of `failures.txt` names `case`, the check key (a row of the table), the option set and `err` against `limit`. Replay with
`--case N --seed0 S` (add `--format` only if you changed the plan; the case determines its own format). The first thing to rule out is
the harness: a tolerance too tight for a documented float32 effect is a table fix, with the claim in the last column; everything else is
a bug in the compiler or the evaluator and gets a pinning unit test before the fix. Bugs this test found are listed in `../../CHANGELOG.md`
and each has its pinning test in `tests/` (`test_differential_pins.py`, `test_robust.py`, `test_f64.py`, `test_readers.py` or `test_pipeline.py`).

## Last full run

The run below was made on 2026-10-23 on the release code (the 1.0 build defaults: generated code at `opt-level` 3, the `fast` shape, the profile-guided
build by default), with the generated-code backend built on every 12th case and the profile-guided build on every 60th, as in the command at the top.
The full run of 2026-10-22, made before these defaults, had the same single failed check (below). The quick differential
test with generated code on every case and the profile-guided build on every 6th (1,483 checks, 0 failed, on two commits of the release candidate) and the aarch64 runs
(see "Platforms" in [`../../README.md`](../../README.md#platforms)) were made as well.

`--mode full --seed0 1` (3000 cases: 1000 xgboost, 1000 LightGBM, 1000 YDF) on x86-64 (Intel Core i7-8550U, 4 cores, 8 threads, AVX2),
three workers, 1916 s wall; Python 3.10.13, numpy 2.2.6, xgboost 3.2.0, LightGBM 4.7.0, YDF 0.16.1:

- Cases finished 3000; checks run 97,402; **failed checks 1** (below); **cases that crashed the process 0**.
- The one failed check: case 2071 (LightGBM `tweedie`, random forest of 400 trees with linear leaves, default float32 file), float64 answer
  1.137e-12 against the fixed 1e-12 bound of `lgb-lin-tr`. Its answers lie beyond the float32 range (margins above 88), and the exp of a margin
  that large carries the margin's own rounding (a few double ulps of a sum of 400 trees whose 1/400 average the compile folds into the leaves)
  into the relative error (a hypothesis: the case was not traced further); the float32 answer is inside its bound (5.9e-08) and with `--f64-leaves` the float64 answer is 3.4e-13. This looks like a bound of the
  harness that is too tight for exp transforms of very large margins, not a wrong answer.
- One case (766, LightGBM `rf` + `linear_tree`) is skipped by the harness because the TRAINING library itself segfaults inside `Booster.update` before QuickForest sees a model; that is a LightGBM crash, not ours.
- The wave engines really ran: the scalar wave engine in 5696 engine runs and the AVX2 gather engine in 5696 (a run counts when `FlatForest.engine_used` equals the engine asked for).
- Backend identity (every backend against the flat walker, bit for bit): 13,018 checks, max error 0. Thread identity (3 threads against 1): 14,322 checks, max error 0. Float32 answer against the float64 answer rounded to float32: 14,322 checks, 0. Refusals (a wave engine asked for a file it cannot hold must raise `EngineRefused`): 26,392 checks, all correct. Written model files reloaded by their own library: 704 checks, max error 2.1e-15.

| key | checks | max observed error | checks over 1e-6 |
|---|---|---|---|
| xgb | 2998 | 0 | 0 |
| xgb+bm | 2998 | 7.6e-06 (inside its own float32-sum bound) | 12 |
| lgb / lgb-tr (default, float32 leaves) | 2860 / 1692 | 6.0e-08 (half an ulp of the float32 answer) | 0 |
| lgb-f64l / -tr | 2860 / 1692 | 5.9e-08 | 0 |
| lgb-lin / -tr | 1168 / 1584 | 5.9e-08 (one float64 check of -tr at 1.1e-12, above) | 0 |
| ydf-dbl (double sums) | 5396 | 2.8e-05 (inside its bound C) | 350 |
| ydf-f32s-reg | 2316 | 6.0e-08 | 0 |
| ydf-f32s-cls | 3080 | 4.1e-07 | 0 |

The `checks over 1e-6` of `xgb+bm` and `ydf-dbl` are the documented float32 effects (xgboost's float32 order with a base margin, YDF's float32 summation): they are inside the bound the tolerance table proves for each case, and the exact modes (the LightGBM files, `--float32-sums`) have none.
