# What QuickForest does not do, and where its answers differ

Everything here is a measured or a designed limit. The tolerances are the ones the differential test (`tests/differential`) enforces; the
fuzz test (`tests/fuzz`) covers damaged input. [docs/REFERENCE.md](docs/REFERENCE.md) describes the commands and options these notes refer to;
the bugs the release tests found, each with its pinning test, are in [CHANGELOG.md](CHANGELOG.md).

## Accuracy

What "equal" means: the compiled forest's answers against the source library's own `predict` **on the same (pruned) model**. Pruning
changes the model; how much is the held-out agreement the run reports. `--verify` and `check` judge the compiled forest against the
library with a bar of 1e-6 (relative to `max(1, |value|)`), and for xgboost with a bar of 0 (the table below says which cases are exact); a model that cannot meet it is refused, not emitted.

| model | answer against the source library's own predict |
|---|---|
| xgboost regression / binary / multi:softprob | identical to the bit (float32, xgboost's own arithmetic; the sigmoid clamps its exponent at 88.7 as xgboost's `common::Sigmoid` does, so a margin below -88.7 answers 3.0e-39 and not 0; the softmax normaliser is summed in double and rounded to float32 once, as xgboost's `common::Softmax` does), also with folds averaged as margin, down to margins of -1e4 and up to 1e4 |
| xgboost with a base margin | xgboost adds in float32 in its own order; the difference is bounded by N x 2^-24 x (sum of the leaves' magnitudes) for N trees. The largest observed was 1.2e-6 relative (80 trees, answers near 100), above the 1e-6 bar |
| LightGBM, default (float32 leaves) | bounded by B = 2^-24 x (sum over a group's trees of the largest leaf). A target with a large scale or offset (leaves of about 1000 adding up to about 1) can break the 1e-6 bar, and the run then refuses: standardise the target, or compile with `--f64-leaves`. Against the library run on the same float32-rounded leaves the answer is exact (the differential test's oracle) |
| LightGBM `--f64-leaves`, linear leaves | identical to `Booster.predict` in double on x86-64 (on aarch64 see Platforms); the float32 answer is half an ulp; with a base margin or a transform (sigmoid, softmax, exp) 1e-12, except an exp of a margin above 88 (an answer beyond the float32 range), where the last-bit rounding of the margin can reach 1.1e-12 (seen once in 95,776 differential checks) |
| YDF default (double sums) | differs by YDF's own sequential float32 summation drift, bound C = N x 2^-24 x (the initial prediction + sum of the largest leaves), for N trees |
| YDF `--float32-sums` | bit-exact with `model.predict` for regression; classification within 2.4e-7 for boosted trees and 5e-7 for a random forest (the float32 transform can differ by an ulp or three; a forest's leaves are divided by the number of trees before the float32 sum, YDF divides after it). Trees over 64 leaves use YDF's generic engine, and the initial prediction is added last |

The largest errors seen in the last full differential run (3,000 cases) were 2.8e-5 for YDF with double sums (inside its proven bound) and
7.6e-6 for xgboost with a base margin (inside its float32-sum bound); LightGBM (default file, `--f64-leaves`, linear leaves), xgboost without a
base margin and `--float32-sums` regression stayed at or below 6e-8 in the float32 answer, and the exact rows were exact (see
`tests/differential/README.md`).

- **Inputs are float32, finite or NaN.** The evaluator rounds its input to float32 and compares float32 values, as xgboost does. LightGBM
  compares in float64, so a float64 value within float32 rounding of a threshold can go the other way in LightGBM than in QuickForest. A check
  with rows placed that close to LightGBM thresholds changed the answer for about a quarter of them; with the same rows rounded to float32
  first, the two sides agreed to 3e-8. Rows that close to a threshold are rare in real data, but cast to float32 before you compare the two
  sides. Infinite inputs are not covered (a comparison against infinity is not tested against the libraries): a selection data file that holds
  one, or a number beyond the float32 range, is refused. `predict` itself passes an infinite row through, and at a LightGBM node whose
  threshold is +infinity (a split above every value the training saw) LightGBM sends +inf left (`inf <= inf`) while QuickForest sends it right
  (`inf < inf` is false), so the two answer differently for that row (measured: 25 of the first 70 random models with +inf in the rows; the same
  rows with 1e30 instead of +inf gave no difference). -inf and xgboost agree. Clip infinite values to a large finite number before `predict`.
  Complex and masked arrays are refused with one line (a complex array lost its imaginary part and a masked one was read through its mask).
- **Output groups (classes):** at most 1,024 (the evaluator's loader refuses a file with more).
- **Features:** at most 32,767, or 8,192 for a model with categorical or zero-as-missing nodes. A node's right child must sit within 65,535 nodes
  after its parent. A model outside these limits is refused at compile time with the limit named.
- **Memory of the random test set.** `check --random N` and `optimize --verify` build N random rows (10,000 by default; `--verify` takes
  fewer for a wide model, so that rows x features stay under 100,000,000 values, 400 MB as float32). Building the set holds several float64
  and int64 temporaries of the set's size at once: the peak is about 36 bytes per value, nine times rows x features x 4 bytes. The path rows
  built after it (up to 4 x N rows, held to the same value limit) add up to their own size again. By the code's count; a test run of a
  10,100-feature model, 9,900 rows, was held at 3.4 GB resident on 2026-10-23, and the estimate for 1,000 features is about 0.4 GB. On a
  machine without that memory, run `check --random N` with a smaller N (`optimize --verify` has no rows option) or leave `--verify` out.
- Linear leaves keep their constants and coefficients in double; the float32 answer is rounded like every other forest (above 3.4e38 it is inf).
- **YDF's transforms at the ends of their range.** YDF's multi-class softmax exponentiates in float32 without subtracting the largest logit,
  so a logit above 88.7 turns its own answer into NaN; the compiled forest answers the exact softmax there (1 for that class, 0 for the others).
  A binary logit below -88.7 gives 0 in YDF's float32 sigmoid and a subnormal below 1.2e-38 in the compiled forest (an absolute difference
  under 1.2e-38). YDF's Poisson loss clamps its logit to [-19, 19] before it exponentiates (rates from 5.6e-9 to 1.8e8); the compiled forest
  does not clamp, so beyond those rates it answers `exp(margin)` where YDF saturates.
- Target transforms (a ratio, a value per unit of area, an index) are the caller's: apply them before training and after prediction.

## Refused (with a one-line reason, never a wrong answer)

- xgboost: categorical splits, `dart` and `gblinear` boosters, vector leaves; objectives other than binary:logistic, reg:logistic, multi:softprob, reg:squarederror.
- LightGBM: multiclassova, cross_entropy_lambda, regression with `reg_sqrt`, custom objectives; a `sigmoid:s` whose s is not a finite number above zero (LightGBM refuses it too); a categorical split with an empty category set.
- A model with no trees (xgboost trained for zero rounds).
- YDF: categorical, boolean, discretised and oblique features or conditions; losses other than the supported ones; a boosted model saved with
  `apply_link_function=False` (its `predict` answers the margin, not the probabilities or the Poisson rate the loss implies).
- `prune` and `optimize` on a model file called `report.json` or `heldout_probs.npz` (any case): the report written next to the pruned models would replace it.
- Linear-leaf forests: not pruned (compile them whole), and the wave engines refuse them; use the walker or the generated-code backends.
- `--float32-sums` on anything but YDF.
- A folder that mixes model formats, or holds one fold twice, or YDF folds that list their input features in another order (a column would
  mean a different feature in each fold).
- A selection data file whose X has another number of columns than the model has features (the libraries differ on what they do with one:
  xgboost reads a missing column as missing, YDF ignores an extra one, LightGBM raises), and a `y`, `fold`, `group`, `slice` or
  `base_margin` that holds NaN or infinity (a NaN in the truth made every error bar pass).
- `--decision threshold:T` with a T that is not a finite number (`p >= nan` is False for every row: one class for all, agreement 100 %).
- A LightGBM text model whose `tree_sizes`, `feature_names` or `feature_infos` disagree with its trees (LightGBM's own loader trusts them
  and aborts the process on a wrong `tree_sizes`), that holds a NUL byte (LightGBM never returns from loading one) or text outside the
  trees that is not UTF-8. A file edited by hand has to keep `tree_sizes` right.
- A path that is not valid UTF-8 (a Latin-1 `caf\xe9` folder) as the output of `compile`, `choose` or `compare`: the evaluator opens only
  UTF-8 paths, so the file would be written and could never be opened. The model and data paths may be any bytes the file system takes.
- A damaged file of any kind: xgboost JSON whose children come before their parent (xgboost itself never writes that), cycles, shared
  children, out-of-range indices, NUL bytes in paths.

## Pruning

- Pruning keeps DECISION agreement with the full ensemble, not accuracy against the truth. Agreement is measured on rows you give it; if they
  do not look like the rows the model will see, the figure does not transfer. Run your own real cases through the pruned model.
- Pruning a boosted model without a refit fails a regression accuracy bar: the pruned model's values drift from the full one's. Prune random
  forests and classifiers; for boosted regression, compile whole (`--keep 100%`).
- How far a forest prunes depends on the data. On the dense synthetic label of `bench/`, a forest cut to 10 % agreed with the full one on only
  70.8 % of held-out rows (`bench/README-NUMBERS.md`); `optimize` reported that and wrote nothing.
- The held-out agreement is the honest figure; selection overfits by about a point. The caller decides whether it meets the bar.
- Pruning is optional. `--keep 100%` (or `compile`) keeps every tree and gives the source library's answers, with no agreement figure to read.
  `--keep auto` tries 5 %, 10 %, 25 % and 50 % (your `--keep-ladder` overrides) and keeps the smallest that passes the bar, else the whole
  forest; its time is that of `--keep 50%` (the selection runs once, to the largest rung) plus one prediction of the held-out rows per rung.
  `--keep 10 --accept-any` keeps exactly the ten you asked for whatever the figure: read the agreement in the report before using them.
- The greedy selection scores every remaining tree against the selection rows at every pick, so its time grows with the number of trees
  times the number kept times the rows (`--sub` and `--max-sel-rows` shrink the rows). The benchmark forest has about a thousand trees
  (`bench/README-NUMBERS.md`); a forest of hundreds of thousands of trees is measured [below](#pruning-a-forest-of-hundreds-of-thousands-of-trees).
- An xgboost model saved after early stopping is compiled with all trained rounds (as `Booster.predict` does), not up to `best_iteration` (as the
  scikit-learn wrapper does).

### Pruning a forest of hundreds of thousands of trees

The case: a random forest of 400,000 trees from which the best 10 or 100 are wanted.

- **Where the in-memory path stops.** It holds the float32 leaf values of the selection rows, trees x rows x 4 bytes (400,000 x 10,136 is
  16 GB), and the model parsed besides. On the 4-thread laptop below, for a binary forest of 10,000 trees it needed 2,581 MB before the leaf lookup ran in chunks
  of rows (it made 24 bytes of transient arrays per tree and row) and 960 MB now; for 30,000 trees
  2,727 MB and 17 minutes then (the machine swapped: 940 s for the leaf values, against 33 s
  now) and 2,169 MB and 96 s now. For 100,000 trees the model alone did not parse in 2.8 GB of address space (MemoryError, on a laptop with other jobs running).
- **From `--memory-mb` on (default 1500) the xgboost `.json` or `.ubj` is read in a stream** and the same trees are kept
  ([REFERENCE.md](docs/REFERENCE.md#a-forest-too-big-to-hold---memory-mb)). The peak resident size, leaf codes apart, is 90 to 280 MB from
  30,000 to 1,200,000 trees (117 to 278 MB with the codes in a file; the codes add their own size while they stay in memory, up to half of
  `--memory-mb`): it does not depend on the number of trees. Time and disk do.
  The model file is mapped and its pages are let go of behind the reader (32 MB at a time), so what stays resident of it is that window plus
  what one page fault maps: 64 KB on most kernels, 2 MB where the kernel holds the file in 2 MB folios of its page cache (measured on GitHub's
  runners, Linux 6.17: a file written in one write; one written in pieces or read back from disk maps 64 KB). The mapping asks for small pages
  (`MADV_NOHUGEPAGE`); it does not grow with the file. `tests/diag_pages.py` prints, for any machine, which stretches
  of a model file stay mapped.

Measured on a 4-thread laptop (Intel i3-6006U, 2 cores, 3.3 GB of RAM, 2.2 GB for the user; nothing else running during these runs, 2026-10-22):
synthetic random forests trained by xgboost (depth 5, about 50 nodes a tree, merged from chunks of 10,000 trees with the leaves rescaled
as xgboost does), 10,136 selection rows and 9,864 held-out rows, 2 threads (the default of a streamed run), `prune MODEL DATA OUT --keep K`.
"Read" is the scan of the model file (for an in-memory run, xgboost's own load); "leaf values" the walk over the selection and held-out rows
(in memory: xgboost's `pred_leaf` and the table lookup); "selection" the greedy picks. The agreement is the pruned forest's with the full
forest's on the held-out rows; the labels are synthetic, so it shows that the selection works and not what your data will give.

| forest | trees | keep | path | read | walk | selection | total | peak memory | leaf codes | held-out agreement |
|---|---|---|---|---|---|---|---|---|---|---|
| binary | 10,000 | 10 | in memory | 4 s | 9 s | 11 s | 28 s | 960 MB | - | 98.2 % |
| binary | 10,000 | 100 | in memory | 4 s | 9 s | 110 s | 2.1 min | 960 MB | - | 99.2 % |
| binary | 30,000 | 10 | stream | 8 s | 49 s | 37 s | 95 s | 395 MB | 304 MB in memory | 98.3 % |
| binary | 100,000 | 10 | stream | 30 s | 2.6 min | 2.3 min | 5.5 min | 117 MB | 1.0 GB in a file | 98.2 % |
| binary, `.json` file | 100,000 | 10 | stream | 34 s | 2.3 min | 2.5 min | 5.4 min | 170 MB | 1.0 GB in a file | 98.2 % |
| binary | 400,000 | 10 | stream | 112 s | 10.6 min | 10.1 min | 22.6 min | 162 MB | 4.1 GB in a file | 98.5 % |
| 3 classes | 30,000 (10,000 a class) | 10 a class | stream | 8 s | 49 s | 2.3 min | 3.3 min | 393 MB | 304 MB in memory | 93.7 % |
| 3 classes, 2,000 selection and 2,000 held-out rows | 300,000 (100,000 a class) | 10 a class | stream | 86 s | 90 s | 4.5 min | 7.5 min | 721 MB | 600 MB in memory | 91.8 % |
| 3 classes, 2,000 selection and 2,000 held-out rows | 1,200,000 (400,000 a class) | 10 a class | stream | 6.0 min | 5.2 min | 20.5 min | 31.9 min | 278 MB | 2.4 GB in a file | 91.8 % |

- **Cost, measured:** reading 0.28 ms a tree; the walk about 80 ns per (tree, row); a pick 12 to 15 ns per (candidate tree, selection
  row) for a binary forest and 46 ns for 3 classes (a softmax per cell). So a run takes about T x 0.28 ms + T x (R + H) x 80 ns +
  K x T x R x 15 ns for T trees, R selection rows, H held-out rows and K trees kept per class (x 3.1 on the last term for 3
  classes). A forest of 1.2 million trees (3 classes, 400,000 a class) was pruned with 2,000 selection rows in 32 minutes (table). Computed from these figures, not run: 100 trees of a binary forest of 400,000 take 1.7 hours of selection (1.9 in all);
  ten per class of a 3-class forest of 400,000 trees a class (1,200,000 trees) with 10,136 rows, 2.2 hours.
  The time is proportional to the rows: `--max-sel-rows` and `--sub` divide it.
- **Threads:** a second thread speeds the walk (1.6x here: 82 s with one thread, 52 with two, 48 with three, 50 with four for 30,000 trees) and
  the selection by almost nothing: a pick costs 15.3 ns per (tree, row) with one thread, 13.5 with two, 15.9 with three and 17.6 with four
  (blocks of 65,536 numbers; 16.7, 16.3, 16.9 and 18.3 with blocks of 262,144). numpy's calls on blocks this small hold the interpreter lock for a good part of
  their time, and the machine has two cores. A streamed run uses 2 threads by default (`--threads`).
- **Disk:** the leaf codes take one byte a tree and row (two for a tree of more than 256 leaves), in memory up to half of `--memory-mb` and
  in a file of the scratch folder beyond it (4.1 GB for 400,000 trees x 10,136 rows; 2.4 GB measured for 1.2 million trees x 2,000 rows, 12 GB at 10,136), plus 13 bytes a node of the model
  (20.1 million nodes, 0.26 GB, for 400,000 trees). `--scratch DIR` chooses the place; the folder is removed at the end, and a killed run leaves a hidden
  `.qf-scratch-*` folder in OUT_DIR to delete.
- **Held-out answers:** the full forest's answers on the held-out rows come from the walk (leaves summed in double, rounded to float32; xgboost sums in
  float32), for at most `--max-held-rows` rows (20,000 by default) in a run that streams any model (a folder without a `fold` array then judges every model on the same rows, so its models that fit in memory are cut to them too); a run with no streamed model uses every row. A row exactly at a
  decision edge can go either way.
- **Not covered:** LightGBM and YDF forests are read by their libraries into memory as before, and compiling a forest whole (`compile`, `--keep 100%`,
  the whole-forest fallback of `--keep auto`) needs it in memory (a few hundred bytes a node). `optimize` runs this `prune` with the defaults.

## Speed: where the speedup shrinks

- The headline figures (at 10 % of the trees: 18.4x with all cores against all cores, 8 threads, and 24.8x one thread against one thread;
  medians of three runs) are on synthetic data whose label (a few threshold rules plus noise) was chosen so that a 10 % prune passes its
  bar; the same forest size on a denser label did not prune to 10 % at 90 % agreement. Read `bench/README-NUMBERS.md` before quoting them.
  The exact path (nothing pruned) gains less on the same forest: 1.5x on all cores, 2.4x on one thread.
- Most of the pruned end-to-end figure is the node reduction (10x) times the compiled walker's gain. The compile-only gains over the library's
  own predict (`bench/bench_compile.py`, no pruning) range from 1.59x (xgboost, `flat`) to 7.09x (LightGBM, `codegen-block`) on one
  thread, and from 1.68x (xgboost, `codegen-block`) to 6.28x (YDF boosted trees, `codegen-block`) on all 8 threads of the test machine (the `flat` and `codegen-block` columns, cells whose three runs all beat the library; the wave engines are in section 2 of `bench/README-NUMBERS.md` and read lower on 8 threads).
  On 8 threads the `flat` engine is within the run-to-run noise of the library for xgboost (1.05x, 0.86x, 1.14x in the three runs) and a YDF random forest (2.23x, 0.92x, 0.75x in the three runs): the median ratio is above 1x but single runs fall below it, so
  read it as no measured gain, not as a gain; the generated code and the other cells stayed above 1x in every run (per-run ranges: `bench/README-NUMBERS.md`, section 2).
  YDF has its own optimised inference engine; the numbers were measured on one CPU model (with AVX2) only.
- Every published ratio gives the library and the compiled forest the SAME number of threads: all cores against all cores (8 on the test
  machine) and one thread against one thread. `check --speed`, `compare.py` and the speed line of `optimize` do the same, every usable core
  by default; `--threads 1` gives the one-thread figures. On the test machine the ratios are lower on all cores than on one thread
  (18.4x against 24.8x at 10 %; 1.5x against 2.4x for the exact compile), because xgboost speeds up 5.7x from one thread to eight and
  QuickForest 4.2x on the 10 % row. The library's own multithreaded predict was compared on those 8 threads (4 cores) and a 20,000-row batch only: with more
  cores or a larger batch it may gain more, so measure on the machine you will use. Small forests gain little: on the quickstart's 30-tree
  model the compiled forest reads 0.9x of xgboost on 8 threads here (0.91x on 2 threads and 1.6x on one thread, measured 2026-10-22 under the earlier defaults, `opt-level` 2 and the baseline CPU, on a 2-core Intel Celeron N4020). How `predict` scales with the cores (a table of one
  thread against all cores) is in `docs/REFERENCE.md`, Threads.
- `wave-avx2` is off, and slower, on CPUs whose Downfall (GDS) microcode slows gathers; `Model::engine_ok(2)` says so.
- On aarch64 there is no gather engine: `wave-avx2` silently runs the scalar wave engine (`engine_used` = 1, `tfl_engine_ok(2)` = 0).
- **Both libraries are built for a portable CPU by default; building for the machine is an opt-in.** `-C target-cpu=native` was measured on an
  aarch64 device for the flat evaluator library only: -1.6 % on a big core, -13 % on a little core, and the library's code did not change, so it
  stays baseline (it is shipped to other machines). The generated code (`codegen-block`, `codegen-pgo`) builds at `opt-level` 3 for `x86-64-v2`
  (`generic` on arm) unless asked otherwise: AVX is never compiled in by default, so a library built here runs on Intel Nehalem and AMD Bulldozer (2008-2011) and the mainstream x86 cores after them (x86-64-v2 needs SSE4.2 and POPCNT, which a few small cores of those years, AMD Bobcat and the early Intel Atoms, lack: `--target-cpu x86-64` is the widest build).
  `--native` (API: `target_cpu="native"`) builds for the CPU of the machine: the build only runs on that CPU type, and it is refused, in one line,
  on a CPU that lacks features it was built with (checked from `/proc/cpuinfo`, so on a system without it no check is made; features whose name
  the check does not know are not checked). The generated code only compares and adds floats, so the ones that matter are the vector and
  bit-manipulation sets. On the 10 % forest of the benchmark, on an Intel Core i7-8550U (AVX2), `--native` ran 6.8 % SLOWER than the default on one thread (ratio 0.932, 95 % interval 0.926 to 0.948) and 4.5 % slower on all 8 threads (0.955, 0.948 to 0.959), in interleaved rounds of one process;
  the twin of the default build, the noise floor of that method, read 0.996 and 1.001. One forest on one CPU: it is not a rule, and it is not a reason to expect a gain. The answers are bit-identical either way.
- Link-time optimisation across the chunks of generated code is not used (planned for 1.1): the driver reaches a chunk through a table of function pointers
  once per block of rows, so there is no call to inline across chunks, and one crate for the whole forest would give up the bounded memory of
  separate chunks. Not measured. Neither build shape (`fast`, `top`) has it, so `top` here means one codegen unit everywhere (chunks and driver) and no LTO across chunks, not fat LTO; fat LTO needs the one-crate build (planned for 1.1).
- **A profile-guided build is only as good as the rows it was profiled on, and the checks on them are about the number of rows and the agreement of the
  rows with themselves, not about speed.** `codegen-pgo` refuses fewer than 2,048 rows and a profile whose two halves (two disjoint halves of the rows,
  profiled apart) agree on less than 82 % of the edge counts (`llvm-profdata overlap`); `best` then builds `codegen-block` with a notice, and the named
  backend fails with that line. The figures behind the two numbers, measured on aarch64 on 2026-10-23 (counts, not timing; three xgboost models of 1,436,
  3,180 and 35,538 nodes on synthetic mixture data; the instrumented build at `opt-level` 3 like the real one; a profile of n rows against a 20,000-row profile
  of other rows of the same data, and the agreement of its own two halves; the lowest of three draws, percent):

  | profile rows | 1,436 nodes: vs 20,000 rows / halves | 3,180 nodes | 35,538 nodes |
  |---|---|---|---|
  | 10 | 65.4 / 75.1 | 61.8 / 66.3 | 32.4 / 55.5 |
  | 100 | 89.4 / 82.0 | 87.8 / 82.2 | 60.3 / 50.1 |
  | 500 | 95.4 / 92.4 | 93.5 / 90.5 | 77.6 / 66.4 |
  | 1,000 | 97.1 / 94.2 | 95.7 / 92.9 | 83.3 / 73.1 |
  | 2,048 | 98.1 / 96.7 | 97.3 / 94.7 | 87.9 / 79.7 |
  | 5,000 | 98.9 / 97.7 | 98.2 / 97.2 | 91.5 / 85.7 |
  | 10,000 | 99.0 / 98.3 | 98.7 / 97.6 | 93.4 / 89.7 |
  | 20,000 | 99.1 / 98.8 | 98.9 / 98.5 | 94.8 / 92.4 |

  From 2,048 rows on, the two smaller forests are within 1.0 and 1.6 points of their 20,000-row figure and their halves agree on 94.7 % or more: that is the
  floor. The 82 % threshold is set by the halves figure alone, between the profiles that must be refused and the ones that must be accepted: 10 rows reach at
  most 79.4 % (any of the three, over three draws), 5,000 rows at least 85.7 % (the largest forest, the lowest of them), and 82 % is the middle, 3.2 points from
  each side; 100 rows are told apart by the floor. The largest forest is not near its plateau at 20,000 rows (6.9 points under it at 2,048, 3.3 at 5,000, 1.4 at
  10,000) and its halves read 79.7 % at 2,048 rows (refused, just under), 85.7 % at 5,000 and 92.4 % at 20,000: such a forest needs a few thousand rows more
  than the floor, and below that `best` builds `codegen-block` and says so, which is the cautious side. The weighted program-level overlap of
  `llvm-profdata` is the only agreement measured; a branch-direction agreement weighted by counts needs the control-flow graph of each function and was not built.
  The 20,000-row line is the sampling noise of the method itself (a second 20,000-row sample against the reference). The same counts from an `opt-level` 0 instrumented build agree far more (the largest forest's halves: 95.9 % at 2,048 rows
  against 79.7 % here), because LLVM instruments the optimised code, which has fewer and hotter counters: the figures are from the real build's level. The
  share of the counts each function holds (function-level overlap) was 94.9 % or more in every run and 97.4 % or more from 100 rows, since every tree function runs once per row block; it is
  reported (`profile_function_overlap`) and does not gate. Not measured: how much running speed each point of overlap is worth (a timing of the floor on x86
  is planned for 1.1), rows that do not look like the ones that will be predicted (two halves of the wrong rows agree with each other: the check cannot see
  it), other data and other forests than these three, and x86 (the counts come from the branch structure, not the instruction set, but it was not run there).
- **The `fast` build shape (opt 3, no LTO, 16 codegen units, profile pass in the same shape) was the best of the shapes compared on a text-feature
  preparation crate, not on generated forest code.** On that crate (x86-64-v2, a shared laptop, 28 interleaved rounds): a profile taken and used in the fast
  shape ran +14.6 % faster than no profile (95 % interval +11.8 to +17.1 %) and +4.0 % faster than a profile taken and used in the slow top-tier shape
  (fat LTO, one codegen unit; interval +1.0 to +6.2 %, and the run could not resolve differences under about 3 points, so this one is marginal),
  with 67 % less process time for the profile-guided build; a profile taken in the fast shape and applied to a final build in the slow one matched none
  of the hot functions and ran as fast as no profile. Details and caveats:
  [bench/README-GENERATED-SHAPE.md](bench/README-GENERATED-SHAPE.md). On the benchmark's 10 % forest (the same CPU, 30 interleaved rounds) the `top` shape ran at 1.003 of the `fast` build's speed on one thread (95 % interval 0.995 to 1.018) and 1.005 on all 8 threads (1.000 to 1.009), the earlier defaults (`opt-level` 2, baseline x86-64, one codegen unit per chunk) at 1.004 and 1.000, and the twin of `fast` at 0.996 and 1.001: no difference beyond the noise of the method; the answers are bit-identical either way.
- The profile-applied check of `codegen-pgo` is about HOT functions only: the fewest functions that together hold 90 % of the profile's counts (in
  practice the generated trees). It fails when the profile has no counts, when a hot generated function is not in the code, or when LLVM reports that a hot
  function lost its profile; LLVM's "no profile data" and hash-mismatch warnings about other, cold functions (an error path, a chunk's entry, a tree the
  profile run did not reach) are counted, never fatal. A profile that matches the hot code but not some cold code is therefore accepted, and that cold
  code runs without the profile's help.
- The build's memory plan is an estimate: it keeps the `rustc` jobs within 85 % of the RAM that is free, from a model calibrated on small chunks (up
  to 69,000 nodes) and set to cover the 850 MB per `rustc` read on 150,000-node chunks of a 4.5-million-node forest; a 150,000-node chunk at 16 units
  has not been measured (`backend_codegen.rustc_peak_estimate_mb`), so it can be too cautious (fewer jobs than the machine could run) or, on a forest shaped unlike
  those, too optimistic; `ram_mb` / `QF_RAM_MB` override it.
- Compile cost: rustc for `codegen-block` and `codegen-pgo` grows with the node count and takes minutes, not seconds, on a pruned forest of
  the benchmark's size. Measured on an Intel Core i7-8550U with the `fast` shape at `opt-level` 3 (the build plans its `rustc` jobs against the free RAM, so the
  job counts differ and the times are not comparable between rows): `codegen-pgo` on the 208,649-node forest 156 s (3 jobs, 608 MB largest `rustc`), on 507,515 nodes 314 s
  (2 jobs, 833 MB), on 1,015,056 nodes 689 s (1 job, 1003 MB); `codegen-block` on the 1,999,767-node exact forest 432 s (1 job, 1001 MB).
  Memory, not time, is the wall for bigger forests; the generated code is split into chunks to bound it. The `top` shape took 179 s on the
  208,649-node forest (327 MB). On the 880-node test fixture `codegen-block` takes 1.8 s at level 3 against 1.6 s at level 2 (median of 3, shared machine; 148 MB peak either way).
  The `fast` shape builds each chunk's codegen units in parallel: on one 68,682-node chunk of an xgboost forest of depth-10 trees (one run each, a
  shared laptop, opt 3) `codegen-block` took 17.3 s at one unit and 10.4 s at 16, and `codegen-pgo` 68.7 s against 24.1 s; the peak of the chunk's
  `rustc` went from 346 to 386 MB and from 384 to 439 MB.
- Generated code holds a tree at most 1,000 levels deep (a chain of 1,000 splits: it writes a tree as ONE nested if/else expression, and rustc's
  parser overflows its stack past about 1,300 levels on the right spine, 1,500 on the left: 140 MB and 2-4 s at 1,300). A deeper tree is
  refused in one line before rustc runs, `--backend auto` leaves the generated-code backends out, and `flat`, `wave` and `wave-avx2` hold it
  (the flat format's own limit is the 65,535-node distance to a right child). Such a tree needs a leaf-wise forest with thousands of leaves on one chain.
- Models are read into numpy arrays before compiling, at a few hundred bytes per node, so compiling a forest of millions of nodes needs
  a few GB of RAM. `prune` of an xgboost model streams a forest too big for the memory ([below](#pruning-a-forest-of-hundreds-of-thousands-of-trees)).

## Platforms

- Linux x86-64: the prebuilt library (`bin/libtreeflat-x86_64.so`, baseline instruction set, no AVX needed). It needs glibc 2.34 or newer (the highest
  symbol version it imports, from `objdump -T`; it links only `libc`, `libm` and `libgcc_s`): Ubuntu 22.04, Debian 12, RHEL 9 and newer load it; on an older glibc
  `dlopen` fails with a "version GLIBC_2.34 not found" error and `QUICK=1 rust/build.sh` builds one that loads there. musl (Alpine) is untested.
- Linux aarch64: tested on a Samsung Galaxy Tab A9+ (Cortex-A78 and A55 cores), Ubuntu 26.04 under proot, Python 3.14.4, rustc 1.93.1 from
  the distribution, xgboost 3.4.1, LightGBM 4.7.0 (last verified at the earlier build defaults, opt-level 2 and the baseline CPU, on 2026-10-22; on 2026-10-23, with the release defaults, the build was repeated, the generated-code tests (`test_codegen_target`, 46 tests, and `test_profdata`, 10 tests) passed and the quick differential test with generated code on every 2nd case and the profile-guided build on every 6th ran 36 cases, 748 checks, 0 failed, 16 cases skipped: no YDF wheel, no pandas; the whole unit suite, the quick fuzz test and the 450-case slice were not re-run with them and keep their 2026-10-22 results):
  the build, 351 unit tests (22 skipped: no YDF wheel, pandas not installed), the quick differential test (680 checks) and fuzz test (3,191 cases) and a 450-case
  slice of the full differential test (7,496 checks) all pass, 0 failed.
  No prebuilt library ships: build it with `QUICK=1 rust/build.sh` (15 s on that device; rustc and cargo built with 1.93 and 1.99, 1.88 is the declared minimum: rustup, or `apt install rustc cargo` where the distribution ships one that new, as that device's Ubuntu 26.04 does with 1.93). `evaluator.find_lib` names the machine
  when no library fits. YDF has no linux-aarch64 wheel, so YDF models cannot be read there. `wave-avx2` runs the scalar wave engine.
  `codegen-pgo` needs `llvm-profdata` from the same LLVM major version as `rustc` (rustup's `llvm-tools`, or `apt install llvm`); it looks in rustc's own toolchain (`rustc --print sysroot`), then on `PATH` and in `/usr/lib/llvm-*/bin`, and takes only a tool whose own `--version` is the LLVM major version of `rustc` (`rustc -vV`); with none, `best` builds `codegen-block` with a notice naming both versions and `--backend codegen-pgo` refuses in one line. A tool of another LLVM version is never used (its profiles are rejected or lose their counts).
- **LightGBM linear leaves on aarch64 differ from the library by one ulp of a double** (5e-17 to 9e-16): LightGBM's ARM wheel fuses
  `const + coef * x` into a multiply-add, and QuickForest's evaluator, like LightGBM on x86, does not. The differential harness allows these rows
  on aarch64 only 1e-12 times the magnitude of their terms (|answer| plus the sum of the leaf terms), so inputs near 1e6 with an answer near zero are judged
  against the size of the numbers that were added, not against a fixed 1e-12; x86-64 keeps the exact bar.
- macOS and Windows: untested.

## Safety

- A memory-mapped `.flat` file that another process truncates while it is loaded gives SIGBUS on the next access. The compilers in this repository
  replace files by rename (never in place), so a running service keeps the old forest. Do not edit a `.flat` in place.
- A generated `.so` is code. Never load one you did not build. The same goes for `libtreeflat-<machine>.so`: `FlatForest(path)` also loads the library it finds next to the `.flat` (`TREEFLAT_LIB`, then the model's folder, then `bin/`), so a `.flat` shared together with a library beside it runs that library. Share the `.flat` alone and let the receiver use the library of this repository. A `choice.json` can name only built-in backends; `--backend file.py:Class`,
  `--decision rule.py:func` and `backends.get('file.py')` execute that file and are for trusted input only.
- Damaged data files (flat, LightGBM text, xgboost JSON, npz, `choice.json`) end in a Python exception with a message. The fuzzer
  (`tests/fuzz`) and a Rust mutation test over nine fixture files back that statement; `CodegenForest.predict` is not fuzzed. The command
  line is covered by a table (`tests/test_input_matrix.py`): every command against every kind of damaged model, data array, artifact,
  output path and numeric option must end in the documented exit code, one line on stderr, no traceback, no signal and no hang.
- A damaged model file that QF's parser accepts can still crash xgboost or LightGBM when a QF command asks that library to load it (`check`, `compare` and `prune` run the library in the same process): the library aborts or segfaults and the command ends with that signal. The fuzzer loads such files with the library in a process of its own and counts these crashes apart.
- Opening a FIFO, a device or (for a flat file and the selection data file) a directory is refused with one line rather than blocking or listing it; a directory given as the model path is read as a folder of fold models.
- Output files are written as new files renamed over the name, never through a link that sits in the output folder.
- No telemetry and no network access: the scripts and the evaluator contain no network code. The libraries you train with are yours.

## Roadmap (planned for 1.1, none of it measured or built yet)

- Speed against the number of profile rows: a paired timing on x86 of builds profiled on 100 to 20,000 rows, to set the floor by speed instead of by the agreement of the counts.
- A memory-lean `check.random_rows`: on a wide forest the random rows of `optimize --verify` hold about 9 times the matrix in temporaries at the cap (it is the peak of the verify step).
- `PARALLEL_MIN_ROWS` (8,192 rows): measure again on 8 threads.
- Link-time optimisation across the chunks of generated code (one crate for the whole forest, which gives up the bounded memory of separate chunks).
- The embed crate: the evaluator as a Rust library for programs that run a compiled forest without Python.
