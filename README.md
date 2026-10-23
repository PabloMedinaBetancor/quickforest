# QuickForest

QuickForest makes a large tree ensemble (xgboost, LightGBM or YDF) cheap to load and fast to predict with. It has two paths, which you can
use alone or together, and you choose the size:

- **Compile only, no pruning.** All the trees are kept and compiled to generated Rust code that rustc builds on your machine (or, without rustc,
  to one flat file that a small Rust library evaluates). The compiled
  answers equal the source library's (to the bit for xgboost; the bound per library is in [LIMITS.md](LIMITS.md#accuracy)). There is nothing to
  agree or disagree about, because no tree is removed. On a synthetic 2.0M-node xgboost forest this path predicted 1.5x faster than xgboost
  on all cores of the test machine with the generated code (115,954 against 76,414 rows/s; 2.4x on one thread), and the model file went from
  68.7 MB to a 16.0 MB flat file (the flat format; the generated library that the 1.5x belongs to is a `.so` of 27.8 MB; the flat file with the `flat` backend gives 1.2x). The speedup for compilation alone, per library, is in section 2 ("Compile only, no pruning") of
  [bench/README-NUMBERS.md](bench/README-NUMBERS.md) and in the table "Compilation alone" below.
- **Prune to the size you choose, then compile.** Keep a share of the trees (`--keep 50%`, `25%`, `10%`), a count (`--keep 10`: ten trees
  per class), or let it choose (`--keep auto`: the smallest of 5 %, 10 %, 25 %, 50 % that passes your bar, else the whole forest). The run
  reports how often the pruned forest's decisions differ from the full forest's on rows it did not use for the choice, and refuses to
  write a model that fails the bar you set.

On the same forest pruned to 10 % of its trees (95.9 % held-out agreement), prediction went from 76,645 to 1,412,848 rows/s with all cores
(8 threads) on both sides, 18.4x; on one thread against one thread, from 13,448 to 332,860 rows/s, 24.8x (median of three runs on a quiet Intel
Core i7-8550U; the table under Results has both paths side by side, and method, caveats and the cases where it does not work are in
[bench/README-NUMBERS.md](bench/README-NUMBERS.md)).

## The problem

A random forest or a boosted model with hundreds of trees is large on disk, slow to load (the whole model is parsed into memory) and slow
to predict with. Often a good part of that is redundant: a small subset of the trees gives the same answer on almost every input. QuickForest
picks that subset, measures how often it disagrees with the full forest on rows it did not use for the choice, and compiles it into a layout
that is cheap to load (memory-mapped) and to evaluate.

## What it does

Two commands, one per path (both through `scripts/quickforest`):

```
# (a) compile only: no pruning, the compiled answers equal the library's
python3 scripts/quickforest compile MODEL OUT.so
python3 scripts/quickforest optimize MODEL --data SELECTION.npz --keep 100% --compile --verify   # the same, with report.md and the equality test

# (b) prune to the size you choose, then compile
python3 scripts/quickforest optimize MODEL --data SELECTION.npz --keep 10% --compile --verify
```

`--keep` takes a **share** or a **count**:

| `--keep` | what it keeps | example |
|---|---|---|
| a share, `50%`, `25%`, `10%` | that share of the trees of each class (of each output group) | `--keep 25%` |
| a count, `10`, `30` | that many trees of each class | `--keep 10` |
| `auto` | the smallest of 5 %, 10 %, 25 %, 50 % (or your `--keep-ladder 2%,5%,10%`) that passes the bar; none passes: the whole forest, compiled exactly | `--keep auto` |
| `100%` | all the trees: no pruning | `--keep 100% --compile` |

A count is per class: `--keep 10` on a 3-class model keeps 10 trees for each class, 30 trees in all (a binary or a regression model has one
class, so 10 trees; for a YDF random-forest classifier one tree serves every class, so 10 trees). A share of the trees of each class is
rounded to the nearest whole tree (a half goes to the even one), never below 1, never above what the class has; a share of 100 % or more
means no pruning. A plain `100` is 100 trees, not 100 %, and `0.1` without a percent sign is refused (write `10%`). The same value goes to the
Python API: `optimize.optimize(model, data, out, keep=10)` (see [docs/REFERENCE.md](docs/REFERENCE.md#using-it-from-python)).

The pipeline:

1. **Prune.** Greedy forest selection: starting from nothing, it adds the one tree that most improves the agreement of the kept trees with
   the full forest's decisions on a set of selection rows, until the requested share is kept. It prunes the forest, not each tree.
   (Skipped with `--keep 100%`.)
2. **Check on held-out rows.** Agreement is measured on rows that were not used for the selection and reported (`report.md`, with the
   first disagreeing rows). If it is below `--min-agreement` (default 95 %) the run stops. With `--keep auto` the bar picks the size:
   every rung tried is listed in the report with its figures. If you ask for a size and want it whatever the figure, add `--accept-any`
   (the figure is still reported).
3. **Compile.** The kept trees are turned into generated Rust code that rustc builds on your machine (the default), or, without rustc, written
   to one flat file (packed 8-byte nodes) read by a Rust library with no dependencies.
4. **Verify.** The compiled forest is compared with the source library's own `predict` on your rows and on rows built from the model's own
   thresholds (the value, one float32 step above and below, and NaN). If a bar fails, only the report is written and the exit code is 2.

**Train a big random forest, keep the best 10.** Train as many trees as you like, then keep the ten that together make the decisions the
whole forest makes (a count per class; `--accept-any` because you asked for exactly ten, whatever the agreement turns out to be):

```
python3 scripts/quickforest optimize big_forest.ubj --data SELECTION.npz --keep 10 --accept-any --compile --verify
```

`report.md` states the held-out agreement of the ten (and, if the data file holds the truth `y`, the accuracy of the full forest and of the
ten against it); `kept_indices` in `report.json` names the ten trees of the original model. Selecting from a very large forest takes time
that grows with the number of trees and of selection rows; an xgboost forest too big for the memory is read in a stream (`--memory-mb`) and
keeps the same trees: 400,000 trees and 10,136 selection rows were pruned to ten in 23 minutes with a 162 MB peak on a 3 GB machine
(measured; [LIMITS.md](LIMITS.md#pruning-a-forest-of-hundreds-of-thousands-of-trees)).

What the guarantee is, per library (full table in [LIMITS.md](LIMITS.md#accuracy)): the compiled answers equal the source library's
**for the pruned model**; the pruned model is not the original, and how often the two disagree is the held-out agreement the run reports.
For xgboost the compiled answers are identical to the bit (with a base margin, xgboost's own float32 summation order decides and the
difference is bounded; see the table). For LightGBM they are bounded by the rounding of the leaves to float32 unless you compile with
`--f64-leaves`. For YDF they differ by YDF's own float32 summation drift unless you compile with `--float32-sums`.

## Requirements and install

There is no installer: the scripts run from the repository folder.

- Python 3.10 or newer and `numpy`.
- At build time only, the library whose models you read: `xgboost`, `lightgbm` and/or `ydf`. A compiled forest needs numpy only.
- `rustc` only to rebuild the evaluator (rustc and cargo; built with 1.93 and 1.99, and 1.88 is the declared minimum, which has not been tried), or for the generated-code backends (`codegen-block`, `codegen-pgo`),
  which may also build with an older compiler (`codegen-block` built with rustc 1.75.0 on 2026-10-22, before the build defaults of 1.0 (`opt-level` 3, 16 codegen units); not tried with them; a build that fails on an old compiler is a one-line refusal, exit 3, not a fallback to the flat format). `codegen-pgo` also needs
  `llvm-profdata` from the same LLVM major version as `rustc` (rustup's `llvm-tools`, or `apt install llvm`). Its profile is taken from the rows of `--data` (at most 20,000): the build refuses fewer than 2,048 rows and a profile whose two halves agree on less than 82 % of the edge counts (`best` then builds `codegen-block` and says why; a named `--backend codegen-pgo` fails in one line),
  and the code is tuned for the rows it saw, so give rows that look like the ones you will predict ([LIMITS.md](LIMITS.md) has the figures).
- The evaluator is prebuilt for x86-64 Linux with glibc 2.34 or newer (`bin/libtreeflat-x86_64.so`, baseline instruction set, no AVX needed; glibc 2.34
  is Ubuntu 22.04, Debian 12, Fedora 35, RHEL 9 and newer). On an older glibc (Ubuntu 20.04, Debian 11, RHEL 8) or on aarch64 Linux build it
  once with `QUICK=1 rust/build.sh` (15 s on the test device; needs `rustc` and `cargo`, built with 1.93 and 1.99 (the declared minimum is 1.88): install them with rustup (https://rustup.rs), or with
  `apt install rustc cargo` where the distribution ships one that new (Ubuntu 26.04 ships 1.93; Ubuntu 22.04 ships 1.75, which cannot build it)).
  The prebuilt library is built for the baseline instruction set because it is shipped to other machines (a `-C target-cpu=native` build of it
  measured no faster on an aarch64 device, see [LIMITS.md](LIMITS.md#speed-where-the-speedup-shrinks)). The generated code of `codegen-block` and `codegen-pgo` is built
  at `opt-level` 3 in the `fast` shape (no link-time optimisation, 16 codegen units, the profile-guided pass built the same way; `--shape top` builds one codegen unit everywhere and no LTO across chunks) for the portable x86-64-v2 CPU (Intel Nehalem and AMD Bulldozer, 2008-2011, and later; `generic` on arm), so it runs on any CPU of that level. `--native` builds for the
  CPU of the machine instead: the build only runs on that CPU type, and it measured no faster than the portable build on our workloads (a library
  built for features another CPU lacks is refused when loaded, in one line). Before it builds, it plans its parallel `rustc` jobs against the RAM that is free and prints
  `RAM: building with N jobs / M units` (fewer jobs, then fewer units, when memory is short; it never swaps).

```
sudo apt install python3-venv                   # Debian and Ubuntu ship `venv` as a separate package; skip it where `python3 -m venv` works
python3 -m venv venv && . venv/bin/activate     # from the repository folder
pip install numpy xgboost                       # enough to run the quickstart
pip install numpy xgboost lightgbm ydf pandas scikit-learn   # everything the test suite uses (ydf has no wheel for every platform: skip it there;
                                                #   without pandas and scikit-learn the real-world model cases of the tests skip, and say so)
```

On x86-64 Linux the `xgboost` wheel pulls in a package of about 350 MB (`nvidia-nccl`); the install takes a while on a slow link. `pip install xgboost-cpu` installs a build without it (the same Python API, no GPU support).

## Quickstart

On aarch64 Linux build the evaluator first (`QUICK=1 rust/build.sh`, see above); the prebuilt library is for x86-64.

```
python3 examples/quickstart.py
```

This trains a small xgboost random forest (300 trees) on synthetic data, runs path (b) above (pruned to 10 %), and compares the result with the original on
fresh rows. It takes under a minute of processor time on a quiet machine. After two lines naming its work folder and the command it runs, it prints the lines below,
unedited, from a run on an Intel Core i7-8550U (4 cores, 8 threads; Python 3.10.13, xgboost 3.2.0, rustc 1.99.0). The counts depend on the library versions, and the timings and the thread count in the speed lines on the machine:

```
RAM: building with 1 job / 16 units
# optimize: OK
model /tmp/quickforest-np322xu9/model.ubj, keep 10%, 7.6 s
- prune: pruned 1 model(s): nodes 107,274 -> 10,700 (10.03x), bytes 3,845,451 -> 384,369; held-out agreement 96.421 % on 4,889 rows (175 disagree; selection rows 97.652 %, so overfit +1.23 points); 1.9 s
- disagreeing held-out rows (first 10 of 175): row 96: 0 -> 2; row 115: 0 -> 1; row 149: 1 -> 0; row 334: 0 -> 2; row 443: 1 -> 0; row 508: 0 -> 1; row 650: 0 -> 1; row 652: 1 -> 2; row 715: 1 -> 0; row 729: 1 -> 0
- compile (codegen-pgo, built for x86-64-v2 CPU, fast shape): 0.54 MB, 30 trees, 10,700 nodes
- verify: PASS, data 10,000 rows max err 0.0e+00, decisions differ 0, random 10,000 rows max err 0.0e+00, decisions differ 0, paths 16,005 rows max err 0.0e+00, decisions differ 0
- speed (8 thread(s), real rows): xgboost 3,855,477 rows/s, compiled 3,452,155 rows/s (0.9x); load 0.001 s, extra RSS 0.0 MB
Emitted: pruned/ and the compiled artifact

fresh rows: the compiled forest and the full forest give the same class on 96.14 % of 20,000 rows
model file: 3.85 MB -> compiled file (model.so): 0.54 MB
all cores (8): xgboost, full forest 475,879 rows/s; compiled, pruned forest 6,800,665 rows/s (14.3x)
(timing on a shared or busy machine is only a rough guide)
```

The compile step now also prints a `- profile:` line (the number of profile rows and the agreement of the profile's two halves) right after the `- compile` line; the block above was recorded on 2026-10-23 just before that line was added.

The `speed` line of the report compares the compiled forest with the library predicting with the same pruned model (the speed of the
compilation alone); the last line compares it with the full original forest (pruning and compilation together). On this small forest and 8 threads the compilation alone reads no faster than xgboost's own multi-threaded predict (0.9x in the `speed` line; the run-to-run noise of such a short call is large); the pruning is what makes the last line 14.3x. Without `rustc` the compile line says `flat` and a `note:` line says why (the flat format is slower than generated code). To use the result:

```python
import sys

sys.path.insert(0, "scripts")
from backends import load_artifact  # needs numpy only

forest = load_artifact("OUT/model.so")  # what optimize built (OUT/model.so; OUT/model.flat where generated code could not be built)
probabilities = forest.predict(x)  # float32 rows, NaN for missing; like xgboost's predict; all cores by default (threads=1 for one)
```

The other path, compile only, on the same forest (no pruning, nothing to disagree about; the file is bigger and the speedup is the one of
the "Compilation alone" table below):

```
mkdir -p OUT && python3 scripts/quickforest compile WORKDIR/model.ubj OUT/model.so       # then load_artifact("OUT/model.so"), as above
```

The commands, options, backends and file formats are in [docs/REFERENCE.md](docs/REFERENCE.md).

## Results

The figures below are from one quiet Intel Core i7-8550U (4 cores with 2 hardware threads each, AVX2), with the build defaults of this release
(generated code at `opt-level` 3, the `fast` shape, the portable x86-64-v2 CPU, profile-guided with the selection rows), on 2026-10-23.

Every figure below comes from a script in `bench/`; [bench/README-NUMBERS.md](bench/README-NUMBERS.md) has the commands, the machine, the
caveats and the runs that did not work out. The compiled forest and the source library always get the same number of threads. The headline
is **all cores against all cores** (here 8 threads, one per logical CPU of that machine); one thread against one
thread is the last column. A 20,000-row batch, Linux, Python 3.10.13, xgboost 3.2.0, lightgbm 4.7.0, ydf 0.16.1, median of three runs. The prebuilt
evaluator library is the one with SHA-256 `b0f290b1ab8ab312...` (`bin/libtreeflat-x86_64.so`).

End to end, `bench/bench_e2e.py`: a synthetic xgboost random forest of 5 folds x 67 trees per class x 3 classes (depth 14, 42 features,
2.0M nodes, 68.7 MB of model files), the same forest through both paths:

| path | nodes | held-out agreement | xgboost, full forest, rows/s (8 threads) | QuickForest rows/s (8 threads) | speedup, 8 threads | speedup, 1 thread | flat file |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact compile, nothing pruned (`--keep 100%`, codegen-block) | 1,999,767 | 100 % (by construction) | 76,414 | 115,954 | **1.5x** | 2.4x | 16.01 MB |
| keep 50 % of the trees (codegen-pgo) | 1,015,056 | 98.4 % | 73,475 | 284,288 | **3.9x** | 4.7x | 8.12 MB |
| keep 25 % (codegen-pgo) | 507,515 | 97.4 % | 79,235 | 570,560 | **7.2x** | 9.5x | 4.06 MB |
| keep 10 % (codegen-pgo) | 208,649 | 95.9 % | 76,645 | 1,412,848 | **18.4x** | 24.8x | 1.67 MB |

Compiled answers equal xgboost's to the last bit in every row (max error 0 against xgboost on the same model). The held-out agreement is
that of the pruned forest with the full one, on rows not used for the choice; the exact path prunes nothing. Opening the flat file takes
0.008 s to 0.061 s, against about 0.06 s for xgboost to read the model files. The label of this data is a few threshold rules plus noise (0.1),
chosen after a denser label failed to prune to 10 % at 90 % agreement (70.8 % at 10 % kept); that is disclosed, with the sweeps, in
`bench/README-NUMBERS.md`. How far a forest of yours prunes depends on your data; `optimize` tells you and refuses below your bar. On all
cores the ratios are lower than on one thread (xgboost speeds up 5.7x going from 1 to 8 threads, QuickForest 4.2x on the 10 % row); on a machine
with another number of cores, measure yours.

Compilation alone, no pruning (`bench/bench_compile.py`, dense synthetic data, speedup over the source library's own predict, both sides on
the same number of threads):

| library | trees / nodes | flat, 8 threads | codegen-block, 8 threads | flat, 1 thread | codegen-block, 1 thread |
|---|---|---:|---:|---:|---:|
| xgboost | 900 / 309,004 | 1.05x | 1.68x | 1.59x | 1.75x |
| LightGBM | 900 / 112,500 | 2.03x | 5.63x | 3.78x | 7.09x |
| YDF (GBT) | 291 / 55,603 | 1.74x | 6.28x | 5.31x | 6.56x |
| YDF (random forest) | 300 (900 per-class trees compiled) / 226,980 | 1.17x | 1.98x | 1.85x | 1.96x |

## How it is tested

- **Unit tests** (`tests/`, synthetic data only): the model readers, pruning, every backend, linear leaves, base margins, float64
  output, damaged input, and one test per bug found by the release tests (listed in [CHANGELOG.md](CHANGELOG.md)).
- **Differential test** (`tests/differential`): thousands of random small models of the three libraries, every backend against the
  library's own `predict` with a stated tolerance per case. The last full run (2026-10-23, 97,402 checks) had one check over its fixed bound (a float64 exp of a margin above 88, explained in
  [LIMITS.md](LIMITS.md#accuracy)) and no crash; its size, the machine and the
  observed error per kind of check are in [tests/differential/README.md](tests/differential/README.md#last-full-run).
- **Fuzzing** (`tests/fuzz`): damaged and hostile files and API misuse must end in a clean exception, never a crash, a hang or a wrong
  answer. The last full run (2026-10-23, 123,063 cases) had no crash, no hang and no unclean case (counts in [tests/fuzz/README.md](tests/fuzz/README.md#full-run-seed0-1)).
  The Rust side has its own tests, one of which runs the evaluator on damaged copies of nine fixture files (1,500 mutations each by default) in
  the debug profile, where overflow and out-of-bounds accesses panic.
- **`unsafe`**: the Rust evaluator has `unsafe` blocks and functions only where it reads memory-mapped or caller-supplied buffers; each
  carries a safety note, and the loader checks a file's checksum and structure before it reads anything else.
- **CI and lint** (`.github/workflows/ci.yml`): `ruff`, `rustfmt`, `clippy -D warnings`, the unit tests, the quick differential test and the
  quick fuzz test, configured for x86-64 and aarch64 and for Python 3.10 to 3.12.

## Platforms

Linux x86-64 is the main target. Linux aarch64 was tested on a Samsung Galaxy Tab A9+ (Cortex-A78 and A55 cores), Ubuntu 26.04 under
proot, Python 3.14.4, rustc 1.93.1, xgboost 3.4.1, LightGBM 4.7.0. These results were last verified at the earlier build defaults (opt-level 2,
baseline CPU) on 2026-10-22. On 2026-10-23, with the release defaults, the build (`QUICK=1 rust/build.sh`) was repeated, the generated-code tests (`test_codegen_target`, 46 tests, and `test_profdata`, 10 tests) passed, and the quick differential test with generated code built on every 2nd case and the profile-guided build on every 6th ran 36 cases (16 skipped: no YDF wheel, no pandas), 748 checks, 0 failed. Not re-run with the release defaults: the whole unit suite, the quick fuzz test and the 450-case slice, which keep their 2026-10-22 results:

- the build (`QUICK=1 rust/build.sh`) OK; unit tests: 351 run, all pass (22 skipped: YDF has no wheel there, and pandas was not installed);
- quick differential test: 36 cases, 680 checks, 0 failed (2026-10-22; 748 checks, 0 failed on 2026-10-23, above); quick fuzz test: 3,191 cases, 0 non-clean (2026-10-22);
- a 450-case slice of the full differential test (xgboost and LightGBM; YDF skipped): 7,496 checks, 0 failed.

The one difference from x86-64 is the last-bit behaviour of LightGBM linear leaves (see [LIMITS.md](LIMITS.md#platforms)). YDF has no linux-aarch64 wheel, so YDF models cannot be read there, and the AVX2 engine
is absent (`wave-avx2` runs the scalar wave engine). macOS and Windows are untested.

## Limits

- Pruning keeps **decisions**, not values. It suits random forests and classifiers; a boosted regression model pruned without a refit
  fails an accuracy bar (compile it whole instead).
- Some model features are refused with a one-line reason, never answered wrongly: categorical splits in xgboost and YDF, xgboost dart,
  several LightGBM objectives, oblique YDF splits.
- The speedup depends on the model, the batch and the number of cores: the compile-only gains measured on the benchmark machine range from
  1.59x (xgboost, `flat`) to 7.09x (LightGBM, `codegen-block`) on one thread, and from 1.68x (xgboost, `codegen-block`) to 6.28x (YDF boosted trees, `codegen-block`) on 8 threads (the wave engines read lower on 8 threads: 1.35x and 1.36x for xgboost); on 8 threads the `flat` engine is within the run-to-run noise of the library for xgboost (1.05x, 0.86x, 1.14x in the three runs) and a YDF random forest (2.23x, 0.92x, 0.75x in the three runs), which is not a gain
  (the generated code and the other cells stayed above 1x in every run); the per-run ranges are in `bench/README-NUMBERS.md`, section 2.
  Both sides were compared on all cores only on a machine with 8 logical CPUs; on more cores the library's own scaling
  decides the ratio. On a very small model the compiled forest can read no faster than the library on several threads (the quickstart's
  30-tree model reads 0.9x on 8 threads here; it read 0.91x on 2 threads, measured 2026-10-22 under the earlier defaults (`opt-level` 2, baseline CPU) on a 2-core Intel Celeron N4020).

The complete list, with the accuracy table per library, is [LIMITS.md](LIMITS.md).

## Prior art and credits

QuickForest is a toolkit assembled from known methods, not a new algorithm.

- Greedy forward selection of ensemble members by their effect on the ensemble: R. Caruana, A. Niculescu-Mizil, G. Crew, A. Ksikes,
  "Ensemble selection from libraries of models", ICML 2004. Ordering the members of an ensemble and keeping a prefix: G. Martinez-Munoz and
  A. Suarez, "Pruning in ordered bagging ensembles", ICML 2006.
- Compiling trees to flat arrays or to generated code: [treelite](https://github.com/dmlc/treelite) (Apache-2.0) and
  [lleaves](https://github.com/siboehm/lleaves) (MIT).
- The libraries whose models it reads and checks against: [xgboost](https://github.com/dmlc/xgboost),
  [LightGBM](https://github.com/microsoft/LightGBM) and [YDF](https://github.com/google/yggdrasil-decision-forests).

## Licence

MIT, see [LICENSE](LICENSE). Nothing from the projects below is bundled or copied: they are installed by you and used through their public
interfaces. The licences are the ones their installed packages declare.

| component | used for | licence |
|---|---|---|
| the Rust evaluator (`rust/`) | the flat evaluator library | no dependencies besides the standard library |
| numpy | everything | BSD-3-Clause |
| xgboost | reading and checking xgboost models | Apache-2.0 |
| LightGBM | reading and checking LightGBM models | MIT |
| YDF | reading and checking YDF models | Apache-2.0 |
| pandas, scikit-learn | tests only | BSD-3-Clause |
| ruff | lint in CI only | MIT |
| Rust toolchain (`rustc`, `cargo`) | building the evaluator and the generated code | Apache-2.0 or MIT |
