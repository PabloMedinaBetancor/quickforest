# QuickForest reference

This is the reference for the command-line tools and the Python modules. [README.md](../README.md) has the overview and a quickstart,
[LIMITS.md](../LIMITS.md) what is refused and where answers differ, [../bench/README-NUMBERS.md](../bench/README-NUMBERS.md) the measured
speeds. A statement about a number below names the script in `bench/` or the test in `tests/` that backs it.

## Contents

1. [Commands](#commands)
2. [Models: formats and what is refused](#models-formats-and-what-is-refused)
3. [The selection data file](#the-selection-data-file)
4. [`optimize`: the pipeline](#optimize-the-pipeline)
5. [`--keep`: the size you choose](#--keep-the-size-you-choose)
6. [The four bars](#the-four-bars)
7. [`prune`: pruning](#prune-pruning)
8. [Using it from Python](#using-it-from-python)
9. [Regression models](#regression-models)
10. [`compile`: backends](#compile-backends)
11. [`check`: the equality harness](#check-the-equality-harness)
12. [Linear leaves and the base margin](#linear-leaves-and-the-base-margin)
13. [Float64 output and exact sums](#float64-output-and-exact-sums)
14. [Using a compiled forest from Python](#using-a-compiled-forest-from-python)
15. [The flat file](#the-flat-file)
16. [Structure export](#structure-export)
17. [Rebuilding and testing](#rebuilding-and-testing)

## Commands

```
python3 scripts/quickforest optimize MODEL --data SELECTION.npz --keep 10% --compile --verify [--out DIR]   # prune to the size you choose, then compile
python3 scripts/quickforest optimize MODEL --data SELECTION.npz --keep 100% --compile --verify             # no pruning: compile the whole forest
python3 scripts/quickforest prune    MODEL SELECTION.npz OUT_DIR --keep 10%
python3 scripts/quickforest compile  MODEL OUT [--data SELECTION.npz] [--backend best|flat|auto|...]       # no pruning; --data makes the build profile-guided
python3 scripts/quickforest check    MODEL ARTIFACT --data SELECTION.npz [--speed]
python3 scripts/quickforest structure MODEL OUT.npz                  # one model's trees as numpy arrays, nothing evaluated
python3 scripts/compare.py MODEL --data SELECTION.npz --work DIR     # every backend side by side
python3 scripts/choose.py  MODEL OUT --data SELECTION.npz            # the fastest exact backend on this machine
```

Each command is also a script (`scripts/optimize.py`, ...) with its own `--help`. They need `numpy`, and the library that wrote the model
(`xgboost`, `lightgbm` or `ydf`) to read it; `compare.py` and `--backend auto` time each candidate in a `python -I` worker, which ignores
`PYTHONPATH` and the user site, so `numpy` has to be installed in the interpreter itself.

### Exit codes

Every command ends in one of four codes, and a refusal is one line on stderr, never a traceback.

| code | meaning | commands |
|---|---|---|
| 0 | success | all (`prune` exits 0 whenever it finishes: the bars are judged by `optimize`; `optimize --keep auto` exits 0 also when no rung passed and it compiled the whole forest, the reason is in the report; so does `--accept-any` with a set that missed the bar) |
| 1 | an argument or an input that cannot be used (a path, a rule file, a value that is not a finite number, a data file without rows or with the wrong number of columns, a damaged model, a model of an unsupported kind); for `check`, also the compiled forest differing from the library beyond the bar | `prune`, `check`, `optimize`, `compare`, `structure` (also a model with categorical splits or linear leaves) |
| 2 | `optimize`: a bar failed (only `report.md` and `report.json` are written); any command: the option parser refused an argument | `optimize`, all |
| 3 | no backend here can hold the model, or none was exact, or a toolchain is missing, or a profile-guided build was refused because its profile did not match the code, or too little RAM is free to build generated code (`--shape`, `ram_mb`: see the build plan below); `optimize` keeps `pruned/` and compiles nothing. For `compile` and `choose` this is also the code of a refused argument or input (a `--target-cpu` rustc does not know, `--native` together with `--target-cpu`). A generated-code library built for CPU features this machine lacks is refused at load time with one line and the same code by `check`; `optimize --verify` reports such a build as not compiled (`compile_unsupported`, `pruned/` kept) and exits 3 | `compile`, `choose`, `optimize`, `check` (an engine that cannot run the file) |

## Models: formats and what is refused

| format | read | supported | refused with a one-line reason | pruned copy |
|---|---|---|---|---|
| xgboost `.ubj` / `.json` | `XgbSource` | gbtree and random-forest boosters; binary:logistic, reg:logistic, multi:softprob, reg:squarederror | categorical splits, vector leaves, dart, other objectives | same format |
| LightGBM text model | `qf_lgb.LgbSource` | gbdt, dart, goss, rf; regression family, binary (`sigmoid:s`), multiclass, cross_entropy, poisson, gamma, tweedie, ranking; numerical splits with every missing type, `zero_as_missing`, categorical splits; `average_output`; linear trees (`linear_tree=true`), compile only | multiclassova, cross_entropy_lambda, regression sqrt, custom objectives | LightGBM text with the kept trees in iteration order; never for linear trees |
| YDF model folder | `qf_ydf.YdfSource` (imports `ydf`) | random forest (winner-take-all or averaged probabilities) and gradient-boosted trees; regression, binary, multiclass; squared and absolute error, binomial, multinomial, poisson | categorical, boolean, discretised or oblique features or conditions; ranking, uplift, survival, anomaly detection; other losses; boosted trees saved with `apply_link_function=False` | YDF folder (`remove_tree`) |

The backing tests are `tests/test_readers.py` (readers and refusals), `tests/test_extensions.py` and `tests/test_robust.py`.

- A folder of model files of one format is a k-fold ensemble: each file is pruned on its own, and the compiled forest averages them. A
  folder that mixes formats, or holds one fold twice (`fold0.ubj` and `fold0.json`), is refused. The fold number of a model is its position in natural name order, counting from 0 (`fold1.ubj`, `fold2.ubj`: fold 0 is `fold1.ubj`).
- Inputs are float32, finite or NaN. YDF matrix columns are `model.input_feature_names()` in that order, and the folds of a YDF folder must
  all list the same input features in the same order (a folder whose folds differ is refused: a column would mean another feature in each).
- The forest reproduces LightGBM's `x <= t` as `x < up32(floor32(t))`, its missing types and its zero bucket (`|x| <= float32(1e-35)` is
  read as 0.0).
- An xgboost model saved after early stopping is compiled with all the trained rounds, as `Booster.predict` does; the scikit-learn wrapper
  predicts with `best_iteration + 1` rounds. The reader warns and `prune` records it under `notes` in its report. Cut the model first
  (`booster[:best+1]`) if you predict through the wrapper. A pruned copy carries neither `best_iteration` nor `best_score` (its rounds are
  the kept ones), so `Booster.predict` and the wrapper's `predict` use all of them and agree with the compiled pruned forest. A pruned YDF
  folder keeps the original's `validation_loss` and training logs (the `ydf` API cannot edit them); its trees are the kept ones.
- A model may have at most 32,767 features (8,192 when it has categorical or zero-as-missing nodes) and 1,024 output groups (classes), and a node's right child must sit
  within 65,535 nodes after its parent; the compiler refuses a model that does not fit and says which limit it hit
  (`tests/test_pipeline.py` pins the 32,767 limit).
- What goes beyond a plain float32 forest (double sums, `exp`, an identity output with several groups, zero-as-missing nodes, categorical
  splits, linear leaves) is listed by the reader in `Forest.requires`. `flat`, `auto` and the generated-code backends hold all of it; only
  the wave engines (`wave`, `wave-avx2`) cannot, and `compile --backend wave` refuses with the reason (exit 3) while `optimize` keeps
  `pruned/` and exits 3.

## The selection data file

`SELECTION.npz` (or a `.npy` holding X alone) is a numpy archive of real rows, as your model will see them:

| array | meaning |
|---|---|
| `X` | (rows, features), float32, NaN for missing. Required, at least one row, exactly as many columns as the model has features (any other width is refused with both numbers). Booleans, integers and floats of any width are read as float32 (the same for a `.npy`); text and objects are refused, and so is infinity (or a number beyond the float32 range). |
| `fold` | optional; row -> the fold model whose out-of-fold row it is. With k-fold files, fold k is selected and judged only on its own rows. |
| `group` | optional; a document or customer id. Even ids are selection rows, odd ids are held out, so a group never straddles both. |
| `slice` | optional; 0 selection, 1 held-out, -1 unused, row by row. Any other value is refused. |
| `eligible` | optional; bool. Only the rows where it is true are used (for example only the rows a rule leaves undecided). |
| `y` | optional; the truth in the model's output space, for the regression error bar. |
| `base_margin` | optional; (rows,) or (rows, groups), used by `check --margin`. |

`fold`, `group`, `slice`, `eligible`, `y` and `base_margin` hold numbers, never text and never NaN or infinity (a NaN in `y` made the error bar pass); `fold`, `group` and `slice` hold whole numbers (they are read as integers: a 0.7 would be cut to 0). The file is read by what it holds, not by its name: an array is X alone, an archive is the table above. `fold`, `group`, `slice`, `eligible` and `y` hold ONE value per row: shape (rows,), or a column (rows, 1), which is flattened to (rows,) (a
column would broadcast against (rows,) and compare every row with every row). Any other shape, or a `base_margin` whose first dimension is not
the number of rows, ends the run with one line naming the array. A data path that exists but is not a regular file (a FIFO, a directory) is refused.

Without `fold`, `group` or `slice` the split is random (`--eval-frac` is the held-out share, `--seed` the seed). Selection and held-out rows
are always different rows. The choice of trees is judged on the rows you give it, so give it rows like the ones the model will see.

## `optimize`: the pipeline

```
python3 scripts/quickforest optimize MODEL --data SELECTION.npz --keep 10% --compile --verify [--out DIR]
    [--decision argmax|threshold:T|rule.py[:func]] [--min-agreement 95] [--backend best|flat|...|auto] [--method greedy|orientation]
    [--native | --target-cpu x86-64-v3|...] [--shape fast|top]
    [--f64-leaves] [--float32-sums] [--regression log|raw] [--keep-ladder 5%,10%,25%,50%] [--accept-any]
```

It runs: prune, held-out agreement, compile, equality test, and writes in `--out` (default `MODEL-optimized` next to the model, named by the model's real folder name for `optimize .`; it must be
empty or new):

- `report.md` and `report.json`: the settings, the pruning summary, the first 10 (md) or 50 (json) rows where the pruned forest disagrees
  with the full one, the compile summary, the verification and the speed. Written also when a bar fails.
- `pruned/`: the pruned models, same format and file names (plus `report.json`).
- `model.so` (generated code, what the default `best` backend builds), or `model.flat` and the evaluator library `libtreeflat-<machine>.so` next to it where
  generated code could not be built or `--backend flat` asked for it. The `artifact` path in
  `report.json` is where the file is in `--out`; a run refused after the compile deleted it and names no artifact.

If any bar fails nothing but the report is left, and the exit code is 2. How many trees are kept is the next section.
`--f64-leaves` and `--float32-sums` shape the compiled forest: they need `--compile`, and `--float32-sums` is refused at once for a
model that is not YDF (as `compile` refuses it), before anything is pruned.

## `--keep`: the size you choose

Pruning is a choice, not a duty: compile only (no pruning) and pruning to a size you pick are two equal paths. `--keep` (in `optimize` and
`prune`; the API parameter `keep`, a string or a number) says which.

| `--keep` | meaning |
|---|---|
| `100%` (or more) | **no pruning.** `optimize` validates the model, copies its files to `pruned/` and compiles the whole forest; the compiled answers equal the source library's (the bound per library: [LIMITS.md](../LIMITS.md#accuracy)). No agreement figure exists because no tree is removed. `compile MODEL OUT.so` does the same without data and without a report (so without PGO: `codegen-block`, one notice line). |
| a share, `50%`, `25%`, `10%`, `2.5%` | that share of the trees of each class (each output group) |
| a count, `10`, `30` | that many trees of each class |
| `auto` | the smallest rung of the ladder that passes the bar; none passes: the whole forest (below) |

**The rule, as the code does it** (`prune.parse_keep`). Trees are counted per output group (class): a multiclass model with 3 classes and
`--keep 10` keeps 10 trees for each class, 30 forest trees (a binary, a regression or any one-output model has one group: 10 trees; a YDF random-forest classification tree serves every class, so there the 10 are 10 library trees). When the groups hold different
numbers of trees the smallest group sets the base of a share. A k-fold folder is pruned fold by fold with the same rule. A share is
`share x trees per group`, rounded to the nearest whole tree (Python's `round`: a half goes to the even number, so `50%` of 7 is 4 and of 5
is 2), at least 1 and at most all of the group. A count above what the group holds keeps them all. A plain number is a count, never a share:
`100` is 100 trees, and only `100%` or more means no pruning. Zero, a negative number, text, NaN and a count below 1 are refused with one
line (`0.1` without a percent sign is not 10 %: write `10%`). The report names the result: `kept_per_group` in each model entry.

**`--keep auto`.** It tries a ladder of shares, smallest first, `--keep-ladder` (default `5%,10%,25%,50%`; shares or counts, comma
separated, each below 100 %; entries that round to the same trees are one rung), and keeps the SMALLEST rung whose held-out figures pass
the bar of the run (`--min-agreement`, or the regression bars). The greedy selection runs once, to the largest rung, because every smaller
rung is the front of the same order of picks (`tests/test_keep_auto.py` pins that the first k picks of a run for K are the selection of a
run for k); each rung's pruned models are then predicted on the held-out rows. If no rung passes, the whole forest is kept and compiled,
exactly as `--keep 100%`, the exit code is 0 and the report says why. `report.json` and `report.md` hold, under `auto` (`stages.prune.auto`
in `optimize`'s), the `ladder`, the `bar`, every rung (`keep`, `also`, `kept_per_group`, `kept_trees`, `nodes`, `heldout_rows`,
`heldout_agreement_pct`, the regression figures if `--regression`, `pass`) and `chosen` (`keep`, `whole_forest`, `reason`, and the
figures of the chosen set).

**`--accept-any`** (`optimize`; API `accept_any=True`). The bar normally stops a run that misses it. When you ask for an exact size and want
it whatever the figure (for example the best 10 trees of a very large random forest), `--accept-any` lets it through: the held-out
agreement (and the regression figures) are still computed and written, the report says `ACCEPTED WITHOUT THE BAR` and lists what the bar
would have refused (`accept_any.waived` in `report.json`), and the exit code is 0. The equality test of the compiled forest (`--verify`) is
not waived. It is refused with `--keep auto`, which chooses the size by the bar. If the data holds the truth `y` of a classifier (the class of
each row), the held-out `accuracy` of the full and of the pruned forest against it is reported too (`heldout.accuracy`, and a line in
`report.md`): agreement with the full forest is not accuracy.

Examples: `--keep 50%` (half), `--keep 10` (ten per class), `--keep auto --min-agreement 98`, `--keep 100% --compile` (compile only),
`--keep 10 --accept-any --compile` (the best 10 of a big forest).

## The four bars

Bars 1 and 4 are checks the tools make and enforce; bars 2 and 3 are things for you to judge, which no command checks.

1. **Held-out agreement.** The pruned forest's decisions agree with the full forest's on at least `--min-agreement` percent of the held-out
   rows (default 95). "Decision" is `--decision`: `argmax` (default; 0.5 for a single output), `threshold:T` (a finite number), or a Python file
   `rule.py[:func]` with `func(p) -> ints` for cascades with thresholds or abstentions (the file is executed: trusted input only). The rule gets `p`, an array of float32 outputs whose LAST axis is the output groups (`p[..., 0]` is the one output of a single-output model, `p.argmax(-1)` the class) and whose leading axes are rows, or candidate trees x rows during the greedy selection; it returns one integer or bool per leading position, shape `p.shape[:-1]`. Mark an abstention with an integer code such as -1, never NaN (`NaN == NaN` is False). The rule is tried once on a (3, 5, groups) and a (5, groups) array when it is loaded, and every answer is checked; one that raises, returns another shape or returns floats ends the run with one line. Set the
   bar from what your use can accept. Selection overfits by about a point, so the held-out figure is the honest one.
2. **Real cases.** Run your own cases through the pruned model before shipping. Agreement with the full forest is not accuracy, and the
   tools cannot check this one for you.
3. **Cost.** Greedy selection scores every remaining tree against the selection rows once per tree it keeps, so its time grows with the number
   of trees, the number kept and the number of selection rows; `--sub` uses a fraction of the rows.
4. **Exactness** (`--verify`, or `check`): the compiled forest against the source library's predict on your rows, on random rows built
   from the model's own thresholds (the value, one float32 step above and below) and NaN, and on rows that reach split nodes (sampled with a budget, see `check`) and
   fall on their threshold. The maximum difference must be <= 1e-6 (relative to `max(1, |value|)`) and every unambiguous decision identical;
   for xgboost it must be 0, the bit-identity the library is claimed to have, except with a base margin or folds averaged as a margin through
   a sigmoid or softmax, which keep the 1e-6 bar. Exceptions per library, and how to tighten them, are in the accuracy
   table of [LIMITS.md](../LIMITS.md#accuracy).

## `prune`: pruning

```
python3 scripts/quickforest prune MODEL SELECTION.npz OUT_DIR [--keep 10%|30|auto] [--decision ...] [--method greedy|orientation]
    [--scale auto|mean|none] [--sub F] [--seed N] [--eval-frac F] [--cases N] [--keep-ladder 5%,10%,25%,50%] [--min-agreement 95]
    [--max-sel-rows N] [--memory-mb N] [--threads N|all] [--max-held-rows N] [--scratch DIR]
```

It prunes the forest, not each tree. Greedy selection adds, one at a time, the tree that most improves the decision agreement of the kept
trees with the full ensemble on the selection rows; `--method orientation` is a cheaper and weaker ordering. The trees of a random forest are
rescaled (leaves x trees/kept); boosted models are not (`--scale` overrides). `--sub` uses a fraction of the selection rows. `OUT_DIR`
receives the pruned models and `report.json` (agreement on selection and held-out rows, node and byte counts, `--cases` disagreeing rows,
and `kept_indices`, below); a model file called `report.json` or `heldout_probs.npz` is refused, as the report would replace the pruned model. The tree and node counts are the library's own: a YDF random-forest classification tree is counted once, not once per class. Linear-leaf forests are refused: a leaf there is a function of the row, not a table entry; compile them whole.

**Which trees were kept.** `report.json` (and the dict `prune.run` returns) holds `kept_indices`: one entry per model file name (a
k-fold folder gives one per fold), `{"original_trees": N, "per_group": [[...], ...], "flat": [...]}`. The lists are sorted 0-based indices
into the ORIGINAL model's own tree list (xgboost `trees`, LightGBM `Tree=` order, YDF tree order), one list per output group and one of all
of them; `--keep 100%` lists every index (so does the whole-forest fallback of `--keep auto`). A YDF random-forest classification tree serves every class, so its index is in every group's list.
`optimize` writes the same under `kept_indices` in its `report.json` and in `pruned/report.json` (with `--keep 100%` only in the first); a
refused run, which leaves no `pruned/`, has none. Applied to the original model's `structure` export, the indices give the pruned model's
trees group by group in order (random-forest leaves are rescaled by the pruning); `tests/test_structure.py` pins this for the three libraries.

### A forest too big to hold: `--memory-mb`

`prune` normally loads an xgboost model with xgboost and holds the float32 leaf values of the selection rows (trees x rows x 4 bytes), and
the model parsed besides. When that would need more than `--memory-mb` (default 1500) for the model and the rows at hand, the model is
read in a stream instead and the same trees are kept (`tests/test_scale.py` pins it: the same picks as the in-memory path on binary and
multi-class forests, random forests and boosted, `.ubj` and `.json`, for every chunk size and block size):

- The model file is read once, tree by tree, and never loaded whole: each tree is checked as everywhere else, and its structure goes to
  flat files in a scratch folder (`--scratch`; by default a hidden folder inside `OUT_DIR`, removed when the run ends, also after an error).
- The selection rows are walked through the trees a chunk of trees at a time. What is kept per tree and row is one byte (two for trees of
  more than 256 leaves) naming the leaf the row ends in, plus one table of leaf values per tree: a quarter of the float32 matrix. The codes
  stay in memory while they fit half of `--memory-mb`; beyond that they are written to a file in the scratch folder and read back a block of
  trees at a time with positional reads, so they never count as the process's resident memory.
- The model file is mapped read-only (`select_stream.map_file`, which asks the kernel for small pages) and the pages already read are let
  go of with `madvise(MADV_DONTNEED)` every 32 MB, in the scan and again in the second pass; what stays resident of it is that window plus
  what one page fault maps, whatever the size of the file. `tests/diag_pages.py` is the report to run when a machine shows more: the pages
  of the mapping that are present, as file-offset ranges, at several points of a read, and every `madvise` with what stays resident of its
  range.
- The greedy loop is the function of the in-memory path, run on the leaf rows it asks for. `--threads N|all` runs the walk and the scoring
  on that many threads (default 1; a streamed run uses 2: more helps the walk a little and the selection not at all, [LIMITS.md](../LIMITS.md); below 1 is 1, above 64 is 64); the
  picks do not depend on the number. With more than one thread a decision rule from a file (`--decision rule.py`) is called from several
  threads at once: it must not keep state between calls.
- The full forest's answers on the held-out rows come from the same walk (the leaves added in tree order in double, rounded to float32; xgboost
  adds in float32, so a row exactly at a decision edge can go either way), for at most `--max-held-rows` rows (default 20,000) in a run that
  streams any model; a folder without a `fold` array then judges every model on the same rows, so its models that fit in memory are cut to them too; a run with no streamed model uses every held-out row. The agreement on the selection rows is measured on the rows used. The pruned model
  is written and predicted by xgboost, as always.
- `report.json` has, for a streamed model, `models[i].scale`: `mode` ("stream"), `memory_mb`, `leaf_codes_bytes`, `leaf_codes_on_disk`,
  `selection_rows`, `heldout_rows_used`, `scan_seconds`. An in-memory run has no such entry.

xgboost `.json` and `.ubj` models only: LightGBM and YDF models are read by their libraries as before. A forest of hundreds of thousands
of trees needs time, not memory: what a run costs, measured, is in [LIMITS.md](../LIMITS.md#pruning-a-forest-of-hundreds-of-thousands-of-trees).
Fewer selection rows (`--max-sel-rows`, `--sub`) shorten it in proportion. `optimize` runs this `prune` with the defaults of these options
(1500 MB, 2 threads for a streamed forest); call `prune` to change them.

## Using it from Python

The two commands are also two functions, with the same names for the options (a flag `--min-agreement` is the parameter `min_agreement`). Put
`scripts/` on the path:

```python
import sys

sys.path.insert(0, "scripts")
import optimize
import prune

# keep the best 10 trees of each class of a big random forest; compile; verify; take it whatever the held-out figure
report = optimize.optimize("big_forest.ubj", "SELECTION.npz", "OUT", keep=10, compile_=True, verify=True, accept_any=True)
# let the bar choose the size
report = optimize.optimize("forest.ubj", "SELECTION.npz", "OUT2", keep="auto", min_agreement=98.0, compile_=True, verify=True)
# only the pruning, no bar
report = prune.run("forest.ubj", "SELECTION.npz", "PRUNED", keep="25%")
```

- `optimize.optimize(model, data, out, keep="10%", compile_=False, verify=False, *, decision="argmax", method="greedy", min_agreement=95.0,
  backend="best", average="output", target_cpu=None, shape=None, seed=15, sub=1.0, threads=None, regression=None, f64_leaves=False,
  float32_sums=False, keep_ladder=None, accept_any=False)` runs the pipeline of `optimize`, writes `out`, and returns the report dict (`ok`, `failures`, `stages`, `kept_indices`,
  `seconds`, ...). Unlike the command it does not call `sys.exit`: read `report["ok"]` (the command's exit 2) and
  `report["compile_unsupported"]` (its exit 3). An argument that cannot be used raises `ValueError` or `SystemExit` with one line.
- `prune.run(model, data_path, out_dir, keep="10%", *, decision="argmax", method="greedy", scale="auto", sub=1.0, seed=15, eval_frac=0.5,
  max_sel_rows=20000, tol=None, cases=50, save_probs=False, regression=None, average="output", min_agreement=95.0, keep_ladder=None,
  memory_mb=1500, scratch=None, max_held_rows=None, threads=None)`
  prunes and returns the report dict (`models`, `heldout`, `totals`, `kept_indices`, `auto`, ...). `min_agreement` and `keep_ladder` matter
  only to `keep="auto"`; `prune.run` never refuses a run for a figure.
- `keep` is a string (`"25%"`, `"10"`, `"auto"`) or a number: `keep=10` is a count of 10 trees per class, `keep="10%"` a share, with the
  rounding rule of [the `--keep` section](#--keep-the-size-you-choose). `regression` is `qf_common.RegBar(space, median_pct, max_pct,
  error_slack)` with `space` `"log"` or `"raw"` (the flags `--regression`, `--med-tol`, `--max-tol`, `--error-slack`).
- A compiled forest is used with `evaluator.FlatForest` ([below](#using-a-compiled-forest-from-python)).

## Regression models

`--regression log|raw` (`optimize` and `prune`) replaces the class-agreement bar: the model outputs the log of a positive
quantity (`log`: relative difference = `|exp(pruned - full) - 1|`) or the quantity itself (`raw`). The bars on the held-out rows are: the median
`|pruned - full|` at most `--med-tol` percent (default 0.5), no row above `--max-tol` percent (default 2), and, if the data holds `y`,
the held-out MAE no worse than the full model's by more than `--error-slack`. Selection minimises the squared difference of the outputs.

Boosted models lose accuracy when pruned without refitting, and these bars fail for them. The tests that exercise this are in
`tests/test_pipeline.py`. Prune random forests; for a boosted regression model compile whole (`--keep 100%`).

## `compile`: backends

```
python3 scripts/quickforest compile MODEL OUT [--backend NAME] [--average output|margin] [--data SELECTION.npz]
    [--native | --target-cpu x86-64-v2|x86-64-v3|x86-64] [--shape fast|top] [--f64-leaves] [--float32-sums]
```

`--average` says how the folds of a k-fold folder combine: `output` is the mean of each fold's outputs, `margin` the transform of the mean
margin. Backends (`scripts/backends.py`); all produce answers that equal the walker bit for bit and the library's within the accuracy table:

| `--backend` | what it is | needs | holds |
|---|---|---|---|
| `best` (default) | the fastest build this machine can make: `codegen-pgo` when `--data` gives representative rows (`optimize` always has them), else `codegen-block` with one notice line; `flat` only when `rustc` is missing, too little RAM is free or the model is beyond what generated code holds, and `codegen-block` when there is no `llvm-profdata` of `rustc`'s LLVM major version (one notice line names rustc's version and the versions found). Writes `OUT.choice.json`, so the loader comes from what was built, not from the suffix of `OUT`; when the suffix says the other kind (`.flat` on generated code, `.so` on the flat format) the notice line says what the file holds and which `--backend` gives the other | as the backend it picks | every supported model |
| `flat` | packed 8-byte nodes, lock-step walker in the Rust library | nothing | every supported model |
| `wave` | the same file; all rows of a block walk one tree level by level, branch-free | nothing | xgboost semantics only |
| `wave-avx2` | `wave` with AVX2 gathers | AVX2, and gathers that the CPU's Downfall (GDS) microcode does not slow | xgboost, and YDF with `--float32-sums` |
| `codegen-block` | every tree becomes Rust `if/else` with constant thresholds, rows blocked per tree; rustc builds one `.so` | `rustc` at build time | every supported model |
| `codegen-pgo` | the same, rebuilt with rustc's profile-guided optimisation on a profile run over `--data` (asked for by name, it refuses in one line before any build, exit 3, without `--data` or without an `llvm-profdata` of rustc's LLVM major version; `best` falls back to `codegen-block` with a notice) | `rustc` and `llvm-profdata` of the same LLVM major version | every supported model |
| `codegen` | the same, one row at a time | `rustc` | slower than the others everywhere; `auto` never tries it. It stays as the plainest generated form, which the unit tests compare with the walker, bit for bit, on every model feature |
| `auto` | builds every candidate that can hold the model and run here, times them on held-out real rows, keeps the fastest exact one | `--data` | writes `OUT.choice.json`; load with `backends.load_artifact(OUT)` |

- `wave-avx2` falls back to the scalar wave engine where the CPU has no AVX2 or the gathers are slow (`engine_used` reads 1 and
  `tfl_engine_ok(2)` is 0); on aarch64 it is always the scalar engine. `TREEFLAT_FORCE_GATHER=1` overrides the check for tests.
- `wave` and `wave-avx2` refuse a file whose header flags hold double sums, extended nodes or linear leaves: every LightGBM forest, and a YDF
  forest unless it was compiled with `--float32-sums`. An xgboost forest runs on them whatever its options (`--f64-leaves` writes the same file),
  and so does a YDF `--float32-sums` file, with a base margin too. Forcing them (`FlatForest(path, engine=1|2)`) raises `evaluator.EngineRefused`, a
  `RuntimeError` that names the way out. An `engine` other than 0, 1 or 2 is a `ValueError`.
- **Generated code is built portable, at the highest optimisation; building for the machine is an opt-in.** Every chunk is built at `opt-level`
  3 in the build shape below (the driver's link was already 3), for the CPU `x86-64-v2` (SSE4.2 and POPCNT: Intel Nehalem and AMD Bulldozer, 2008-2011, and the mainstream x86 cores after them; a few small cores of those years, AMD Bobcat and the early Intel Atoms, lack them, and `--target-cpu x86-64` is the widest build; `generic` on
  arm). AVX2 and later are never compiled in by default. `--native` (API: `target_cpu="native"`) builds for the CPU of the machine instead, and
  **the build only runs on that CPU type**; `--target-cpu CPU` names any other (`x86-64-v3`, `x86-64`, anything `rustc --print target-cpus` lists;
  an unknown name is refused, rustc alone would only warn). `--native` together with another `--target-cpu` is refused. API:
  `options["target_cpu"]`; in `compare` and `choose` `--opt target_cpu=native`. The result (`info["target_cpu"]`, the `compile` line, `report.md`)
  names the CPU. Default, native and widest-CPU builds are bit-identical (Rust does not fuse a multiply and an add without fast-math):
  `tests/test_codegen_target.py` and the quick differential pin it. The prebuilt evaluator library of the `flat` backends is baseline too (its AVX2
  engine is chosen at run time); `--target-cpu` and `--native` are refused with `--backend flat|wave|wave-avx2`.
- **Build shape: `fast` (default) or `top`.** `fast` builds every chunk with `-C opt-level=3 -C lto=off -C codegen-units=16`. A chunk's trees are
  written into up to 16 modules, because rustc makes one codegen unit per module and `codegen-units=16` splits nothing in a one-module source; the
  units compile in parallel inside each `rustc`. `codegen-pgo` builds its instrumented binary, runs its profile and builds the final one in the SAME
  shape, since a profile only matches code that was built the same way (the stale-profile gate below fails the build otherwise). `top` (`--shape top`;
  API `options["shape"] = "top"`) is one codegen unit everywhere, in the chunks (one module each) and in the driver's link (`-C codegen-units=1`), and no LTO across chunks: slower to build; `fast` leaves the driver's link at rustc's own defaults. Chunks are
  separate crates joined at link time, so neither shape has link-time optimisation across chunks (LIMITS.md). The shape changes how the code is
  built, never what it computes: both give bit-identical answers (`tests/test_codegen_target.py::BuildShapeTest`). `--shape` is refused with
  `--backend flat|wave|wave-avx2`; an unknown shape is refused in one line by the API (a `ValueError`) and by the option parser of the commands (exit 2). The result (`info["shape"]`, `info["codegen_units"]`, the `compile` line,
  `report.md`) names the shape. Why `fast`: [bench/README-GENERATED-SHAPE.md](../bench/README-GENERATED-SHAPE.md).
- **The build plans its jobs against the free RAM and never swaps.** Before it builds, it estimates the peak of the `rustc` jobs that run together
  (`backend_codegen.rustc_peak_estimate_mb`, from the chunk sizes and the codegen units; the constants and where they come from are in
  that function's comment) and keeps it within 85 % of the RAM available: the smaller of `MemAvailable` and what a cgroup memory limit still
  leaves. If the wanted `jobs` (default 3) at the wanted units do not fit, it lowers the jobs first (down to 1), then the units (halved each
  time), then the chunk size (halved, down to 4,000 nodes), and prints ONE line on stderr, `RAM: building with N jobs / M units`, with what it
  changed and the MB free when it changed anything. If one job on a 4,000-node chunk does not fit, the build is refused in one line
  (`backend_codegen.NotEnoughRam`, a `ValueError`: exit 3), and `--backend best` falls back to the flat format with a notice. `options["ram_mb"]`
  or the environment variable `QF_RAM_MB` replaces the figure read from the machine (`--opt ram_mb=N` in `compare` and `choose`). The estimate
  is a model calibrated on builds of small chunks (up to 69,000 nodes) and set to cover the 850 MB per rustc that was read on 150,000-node chunks
  of a 4.5-million-node forest; a 150,000-node chunk at 16 units has not been measured. It is not a measurement of your forest: when a build is
  killed for memory anyway, give a lower `ram_mb`.
- **A library is refused on a CPU that lacks its features.** The build writes the CPU features it used into the library
  (`QFTARGET1|<cpu>|<arch>|<features>|`, found by reading the file), and `CodegenForest` compares them with the features every logical CPU of the
  machine reports in `/proc/cpuinfo` BEFORE the library is loaded (a check inside a native library could be folded to true by the compiler it
  was built with): one line naming the missing features and `--native`, `backend_codegen.TargetCpuMismatch` (a `ValueError`: `check` exits 3, `optimize --verify`
  reports the build as not compiled). A default build records the `x86-64-v2` features and is checked the same way; a native one is the case the check is for. Features whose
  `/proc/cpuinfo` name the check does not know are not checked; libraries built before the record load as they always did. To build on one machine
  and run on another (`compare --stage build` / `--stage measure`, a library sent to a server) keep the default, or name the oldest CPU you ship to.
- **A profile-guided build must have used its profile for its hot functions.** rustc applies a profile that does not match the code without a word: a
  function whose control flow changed just loses its counts. The hot functions are the fewest that together hold 90 % of the counts of the profile
  (`llvm-profdata show --all-functions --counts`; in practice the generated trees). After the profile-use build, `codegen-pgo` fails with `BuildFailed`
  (one line, exit 3, nothing written) when the profile holds no counts, when one of its hottest generated functions is not in the code (a profile of
  another model), or when rustc/LLVM reported that a hot function lost its profile (a hash mismatch, or no data under the name the profile knows it by,
  as when a codegen unit is numbered differently) or printed a warning the check cannot read. LLVM's warnings about any other function are cold: code
  that never ran in the profile run, or ran so rarely that it holds none of the 90 % (an error path, a chunk's entry, a tree the profile run did not
  reach). They are counted (`info["profile_cold_unmatched"]`, named in the note) and never fatal. A successful build reports `profiled_functions` (the
  generated functions that have profile data), `profile_hot_functions` and says `profile on N functions` in its note. Pinned by
  `tests/test_codegen_target.py::ProfileGateTest`: a cold miss passes, a hot miss fails.
- **A profile-guided build is tuned for the rows it was profiled on, so they must be enough and must look like the rows you will predict.** The
  profile run uses the rows of `--data` (`optimize`: the selection rows; at most 20,000, a seeded random pick; held-out rows are only for the
  speed test). The build refuses fewer than 2,048 rows and profiles the rows in two disjoint halves (a seeded shuffle, so ordered rows are
  not split into early and late), apart, then merges them: counts add, so the profile used is the same as from one run. `llvm-profdata overlap`
  of the two halves must be at least 82 % at the edge (branch counter) level (the function-level overlap, the share of the counts each function holds, is reported next to it and does not gate: it was 94.9 % or more in every run measured and 97.4 % or more from 100 rows, because every tree function runs once per row block). When the rows are too few or the
  halves disagree, `best` (and `optimize`'s default) builds `codegen-block` and its notice gives the row count or the two figures and says to give
  `--data` with more, representative rows; `--backend codegen-pgo` fails with the same line (`ProfileUnreliable`, a `BuildFailed`: exit 3, nothing
  written). A good build reports `profile_rows`, `profile_overlap` and `profile_function_overlap` (`optimize` prints them in `report.md`). The
  check cannot know whether the rows look like production: halves of rows from the wrong place agree with each other. Give rows from the
  distribution the model will be asked about (the numbers behind the limits are in the next paragraph and in LIMITS.md).
  What the profile run executes: the generated code has one function per tree and row block (`b<k>`, called by one entry function per chunk, from
  one closure in the driver); `predict`, `predict_margin` (a base margin) and `predict_f64` (float64 answers) all go through that same closure,
  and a call with several threads only runs the same `chunk` on a slice of the rows per thread, so the tree functions, which hold nearly all of
  the time, are the same whatever the call and the thread count; a last block with fewer rows than the block size runs the same functions on
  fewer rows. What differs is the driver around them: `run`/`chunk` exist once per answer type (float32, float64), the base margin has its own
  branch in `chunk`, and the thread dispatch runs only on the threaded path. A function that never ran in the profile is tuned for size, so a
  second, small pass runs `PROFILE_COVER_FACTOR` (2.5) blocks of the rows on two threads, with float64 answers and with a base margin
  (`profile_run(..., "cover")`); its counts are merged into the profile and are not part of the halves comparison. Pinned by
  `test_codegen_target.BestBackendTest` (floor, halves, notice, report) and `ProfileGateTest` (the calls of the two passes).
- `codegen-pgo` needs `llvm-profdata` from the LLVM major version of `rustc` (`rustc -vV` prints it): rustup's `llvm-tools` component, or `apt install llvm`.
  The tool is looked for in this order and never taken from another LLVM version, because rustc silently loses or rejects the profile of another
  one: first the `llvm-tools` copy of rustc's own toolchain (`$(rustc --print sysroot)/lib/rustlib/<host>/bin`; with rustc on `PATH` that is rustup's
  proxy, so the folder rustc was found in says nothing about it), then `llvm-profdata` on `PATH` and the `/usr/lib/llvm-*/bin` ones of a distro, each
  only if its own `--version` reports rustc's LLVM major version (not the folder's name). With none, `best` builds `codegen-block` and the notice
  names both versions (`no llvm-profdata of rustc's LLVM 23 (found only LLVM 18 at /usr/bin/llvm-profdata)`), and `--backend codegen-pgo` refuses
  with the same line before any build (`ProfdataMissing`, exit 3). Pinned by `tests/test_profdata.py` and `test_codegen_target.BestBackendTest`. It also
  RUNS the instrumented code it builds, so a `--target-cpu` this CPU lacks features of is refused before the first build (`BuildFailed`, one line,
  exit 3): use `--backend codegen-block`, which is not run while it builds, or profile on a CPU that has them. The
  profile run uses real rows; the speed test uses other, held-out rows. The harness reports three rates per variant: held-out real rows
  (the one to trust), the profile rows, and rows built from the thresholds.
- Generated code holds a tree at most 1,000 levels deep (one nested if/else per tree; rustc's parser overflows past about 1,300). A deeper
  tree is refused in one line before rustc runs, `--backend auto` leaves generated code out, and the flat and wave backends hold it.
- Compile time and the memory of `rustc` grow with the node count; the generated code is split into chunks to bound the memory of each
  `rustc` process (`--opt chunk_nodes=N jobs=N`). Keep chunks at about 50,000 nodes or more: each chunk is one call per row block and
  chunks are linked without inlining across them (nothing is lost: the driver calls a chunk through a table once per block of rows, so there is
  no call to inline). `opt-level` 3 builds longer than 2 did (the `fast` shape compiles each chunk's units in parallel). For the 208,649-node pruned forest of `bench/bench_e2e.py` the `codegen-pgo` build took
  156 s on an Intel Core i7-8550U in the `fast` shape with three `rustc` jobs, the largest peaking at 608 MB (`LIMITS.md`, compile cost).
- A backend is a class in a file: `--backend path/to/file.py:ClassName` loads it (the interface is the docstring of `scripts/backends.py`).
  That executes the file; use it on trusted input only.
- `scripts/compare.py` prints the side-by-side table by hand: `--stage build` on a build machine and `--stage measure` on the target
  (`--opt block=1024 chunk_nodes=N jobs=N codegen_units=N`). Each build records a signature of the model it was built from, and
  `--stage measure` refuses a work folder built from another model. `--random 0` skips the rows built on the thresholds.
- Recompiling over a file that a process has loaded is safe: the compilers write a new file and rename it over the old one, so the running
  process keeps the old forest until it reloads. A process that opens the same `.so` path again gets the already-loaded library; restart it
  to pick up a new one. `FlatForest.close()` refuses new predictions and waits for the ones other threads are running before it frees the
  forest, so the old forest of a hot swap can be closed while requests are still in flight.

What every published speed ratio is against (`bench/`, `scripts/compare.py`, `check --speed`): the source library's own `predict` over a fixed
batch of 20,000 rows, tiled or cut from your sample, with the library and the compiled forest given the SAME number of threads (every usable
core by default, `--threads all`, or `--threads N`; `bench/README-NUMBERS.md` says which counts, 8 and 1, stand next to each ratio). For xgboost
the library side is `Booster.inplace_predict` with that `nthread`, without building a `DMatrix` in the timed call; for LightGBM `num_threads`; for
YDF `predict(num_threads=...)`. The report names the library, and the keys `library_threads` and `compiled_threads` name the counts; `--threads 1`
gives the one-thread figures.

## `check`: the equality harness

```
python3 scripts/quickforest check MODEL ARTIFACT [--data SELECTION.npz] [--random 10000] [--decision argmax]
    [--speed] [--margin] [--f64] [--absolute] [--threads N|all] [--json OUT.json] [--backend NAME]
```

ARTIFACT is loaded as `backends.load_artifact` does: the backend named in `ARTIFACT.choice.json` (written by `compile` with the default
`best` backend, `--backend auto` and `choose.py`), else by suffix (`.so` is generated code, anything else the flat file). `--backend NAME` forces one. The report path of
`--json` (and of `compare --json` / `--md`) is checked before the run: a folder the user cannot write, or an existing target that is not a
regular file (`/dev/null`), is refused in one line (exit 1); a write that fails anyway after the run is one line naming the verdict and the
path, exit 1.

It compares the compiled forest with the source library on your rows, on `--random` rows built to hit the split thresholds exactly
(equal, one float32 step above and below) and NaN, and on up to four times as many `paths` rows that reach split nodes and fall
on their threshold, a step below it and missing. The nodes are sampled with a budget of 4/3 of `--random` (13,333 at the default 10,000; 3/4 of that with a categorical split): every tree is asked for at least one node, but the quotas round up, so a forest of several thousand trees whose quotas add up to more than the budget keeps a random subset of the picks and a few of its trees (91 of 7,500 in one case measured) have no path row; raise `--random` to reach every tree. It exits 1 unless the maximum difference is <= 1e-6 (relative to `max(1, |value|)`) and
every unambiguous decision is identical; for xgboost the difference must be 0 (see the exceptions in the accuracy table of LIMITS.md). `--absolute` judges the bar on the plain difference (a log-scale output of 12 then has to match to 1e-6
absolute). `--margin` also passes a base margin to both sides. `--f64` judges the float64 answers. `--speed` adds the throughput
against the library (`--threads N|all`, default all, sets the thread count of the comparison predictions; the speed line times the library and the compiled forest on the same N threads, all usable cores by default, and reports `library_threads` and `compiled_threads`), the load time and the extra memory. The differential test (`tests/differential`) runs the same comparisons on thousands
of random models with a tolerance per library.

## Linear leaves and the base margin

**Linear leaves** (LightGBM `linear_tree=True`): a leaf is `leaf_const + sum coef_k * x[f_k]`, in double, in the file's order. If any
feature of that leaf's linear model is NaN, the tree returns the plain `leaf_value` instead, as the library's predictor does (pinned by
`tests/test_linear.py::LinearLeavesTest.test_nan_in_a_leaf_feature_gives_the_plain_leaf_value_and_nothing_else_does`). Only NaN counts: 0.0 and
the zero bucket are numbers. LightGBM refuses `zero_as_missing` with linear trees at training. The flat file for such a forest is format 6
(`--f64-leaves` also writes it, for the leaves float32 would round); every other forest keeps the layout it had before linear leaves existed,
format 4 or 5 (`tests/test_linear.py::OldForestsStayTheSameTest`). Supported by `flat`, `codegen`, `codegen-block` and `codegen-pgo` (the leaf becomes
inline Rust with the same multiply-add order and no fused operations); `wave` and `wave-avx2` declare it unsupported and `--backend auto`
skips them. `prune` refuses linear forests; compile them whole (`compile`, or `optimize --keep 100% --compile`). Constants and
coefficients are stored exactly in double.

**Base margin** (the output of a base model that a boosted forest corrects, or an offset): every backend takes
`predict(x, threads=None, base_margin=None)`. `base_margin` is (rows, groups), or (rows,) for one group, of numbers ADDED to the raw score of
every model of the forest (every fold) before the objective's transform, in float64; non-finite or wrongly shaped input is refused. `x` and
`base_margin` must be real and unmasked: a complex array or a masked (`np.ma`) array is refused with a `ValueError`; fill the masked values
first (`np.ma.filled`). The
reference is the library's own predictor given the same numbers: LightGBM and YDF with the raw score plus the margin and then the
objective's transform; xgboost with a DMatrix `base_margin`, which REPLACES `base_score`, so the model's own base margin is added to it
(`tests/test_linear.py::BaseMarginTest.test_a_margin_is_added_to_xgboost_not_swapped_for_its_base_score`). C ABI:
`tfl_predict_margin(m, x, n_rows, out, threads, engine, base_margin_f64_or_null)` for the flat file and
`tfl_cg_predict_margin(x, n_rows, out, threads, base_margin)` for generated code. `check --margin` uses the data file's `base_margin` array
when it has one, else a seeded random one. The base model itself lives in the caller, not in the forest. Target transforms (a ratio, a value per unit of area,
an index) are likewise the caller's: apply them before training and after prediction.

A LightGBM binary model trained with `sigmoid:s` has `s` folded into its leaves; the margin stays in LightGBM's raw-score units (the answer
is sigma(s * (raw + margin)), as for a model trained with `init_score`), so every evaluator multiplies the margin by `s` (`forest.margin_scale`; generated code exports
`tfl_cg_margin_scale`; the flat file stores it as described in [The flat file](#the-flat-file)). A file whose models have DIFFERENT `s` refuses a base margin with a ValueError (one margin cannot mean
the same to each); without a margin it evaluates as usual. A random-forest model's margin is the averaged, scaled one (`average_output`).

## Float64 output and exact sums

`predict(x, threads=None, base_margin=None, dtype=np.float64)` on every backend returns the answer without the final rounding to float32
(`dtype=np.float32` is the default). C ABI: `tfl_predict_f64(m, x, n_rows, out_f64, threads, engine, base_margin_f64_or_null)` and, in
generated code, `tfl_cg_predict_f64(...)`. On a
double-sum file the float32 answer is the float64 one rounded, bit for bit (`tests/differential`, check `round`). `check --f64` judges the
float64 answers; with `--absolute` the 1e-6 bar is the plain difference of two doubles.

Two opt-in compile options, forwarded by `optimize` and by `--backend auto` (in `compare` and `choose` they are `--opt f64_leaves=true`).
The default files stay byte-identical when they are not used.

- `--f64-leaves` (LightGBM; no effect on xgboost or YDF boosted models, which have no leaves to widen): stores the leaves that float32 would
  round in double. The float64 answer is then bit-identical to `Booster.predict` (`tests/test_f64.py::LightGbmF64Test`, and the
  differential test: 0.0 for identity objectives, 1e-12 with a transform such as sigmoid, softmax or exp).
- `--float32-sums` (YDF only; refused for the other libraries): adds the leaves in float32 in YDF's own order (the initial prediction first
  for trees of at most 64 leaves, last for bigger trees), reproducing `model.predict` bit for bit for regression
  (`tests/test_f64.py::YdfF64Test`). Classification agrees to within 2.4e-7 (boosted trees) or 5e-7 (random forest), because the float32 transform can differ by an ulp or three and a forest divides its leaves before the float32 sum. A base margin
  is added late, in double. Use it for a YDF member under an absolute gate.

Where float32 rounding still enters, per library:

- xgboost: its own arithmetic is float32 (the softmax normaliser is summed in double, as xgboost's `common::Softmax` does), so the float64 answer is its float32 result widened; with a base margin xgboost's float32 order
  decides, and the difference grows with the number of trees (`tests/differential`, bound `xgb+bm`).
- LightGBM: the text model holds double leaves but the default file stores float32 ones. The rounding is up to half an ulp of a leaf per tree,
  so a target with a large scale or offset (leaves of about 1000 adding up to about 1) can break a 1e-6 bar: standardise the target or use
  `--f64-leaves`. The differential test judges the default file against the library run on the same float32-rounded leaves (exact, key `lgb`), which
  proves the whole gap to the library's own answer is that rounding.
- YDF: its own predictor sums in float32, so double sums differ from it by that summation noise (key `ydf-dbl`); `--float32-sums` removes it.

## Using a compiled forest from Python

```python
import sys

sys.path.insert(0, "scripts")
from evaluator import FlatForest  # imports numpy only; no xgboost, LightGBM or ydf needed

forest = FlatForest("OUT/model.flat")  # engine=0: the walker; the library is found next to the file
y = forest.predict(x)  # (rows, features) float32 -> (rows,) or (rows, groups); all usable cores (threads=1 for one)
y64 = forest.predict(x, dtype=np.float64)  # no final rounding to float32 (import numpy as np)
forest.close()
```

`FlatForest` loads the files of the `flat`, `wave` and `wave-avx2` backends. The default `best` backend writes generated code (a `.so`, plus
`OUT.choice.json` naming its backend): load whatever `compile` or `optimize` wrote with `backends.load_artifact(path)`, which returns a predictor with the
same `predict(x, threads=None, base_margin=None, dtype=...)`, `n_features`, `n_groups`, ... (`from backends import load_artifact` after the same `sys.path.insert`).
It needs numpy only, and refuses a library built for a CPU this machine lacks (see [`compile`: backends](#compile-backends)).

`x` is (rows, features), float32, finite or NaN. `forest.n_features`, `n_groups`, `n_folds`, `n_trees` and `n_nodes` describe the file.
A file compiled with `auto` or with a generated-code backend loads with `backends.load_artifact(path)`, which reads `path.choice.json`
(it can name only built-in backends and never runs a file). The library is searched in `TREEFLAT_LIB`, then next to the model file, then in
`bin/`; `evaluator.find_lib` names the machine when none fits. Copies of one build share one loaded instance, so loading a thousand compiled
model folders in one process is safe (`tests/test_differential_pins.py::LibraryCopiesTest`); a process that loads a thousand different
generated-code libraries is bound by the 1,024 thread-local keys a process has.

### Threads

`predict(x, threads=None, ...)` on every backend (`FlatForest`, the wave engines, `CodegenForest`), `check --threads N|all` and
`optimize --threads N|all` take the same value. The default is `None` / `all`: **every CPU the process may run on**
(`len(os.sched_getaffinity(0))` where it exists, else `os.cpu_count()`, lowered to the container's CPU quota when the cgroup has one:
`cpu.max` of cgroup v2 or `cpu.cfs_quota_us` of v1, rounded up; at least 1, at most 64, the cap the library applies as well).
Below `evaluator.PARALLEL_MIN_ROWS` rows the default uses one thread, because starting and joining threads then costs more than the walk
saves. A whole number forces that many threads (clamped to 1..64) whatever the row count; `threads=1` is one thread. Anything else is a
`ValueError`. The answers are bit-identical for any thread count (`tests/test_pipeline.py`, and the differential's `threads` key).
The library splits the rows into chunks of whole 1,024-row blocks, so a call never uses more threads than it has blocks.
The `check --speed` ratio times the library and the compiled forest on the same number of threads (every usable core by default; `--threads 1` gives the one-thread figure; see LIMITS.md, Speed). A custom backend whose `predict` has
`threads=1` as its default keeps working: `check` passes no thread count unless a number was given.

The row cutoff is `evaluator.PARALLEL_MIN_ROWS = 8192`, **measured** (2026-10-22, under the earlier build defaults: `opt-level` 2 and the baseline CPU for the `codegen-block` rows, the prebuilt library of the `flat` rows is unchanged; not re-measured with the release defaults; `predict` on 1 thread against every core, median of many calls,
rows 1 to 100,000; ratio above 1 = all cores faster, 1.00 = the library stayed on one thread):

tiny = 3 trees of depth 3; small = 30 trees of depth 4 (about 900 nodes); large = 200 trees of depth 8 (xgboost, 20 features, synthetic).

A quiet 2-core laptop (Celeron N4020, 2 threads, nothing else running during the measurement), time of one thread divided by time of both cores:

| forest, backend | 100 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 100,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny, flat | 1.00 | 1.00 | 0.99 | 0.69 | 1.50 | 1.64 | 1.80 |
| tiny, codegen-block | 1.00 | 0.45 | 0.59 | 1.09 | 1.37 | 1.53 | 1.75 |
| small, flat | 0.99 | 1.00 | 1.68 | 1.85 | 1.85 | 1.87 | 1.86 |
| small, codegen-block | 1.01 | 1.22 | 1.47 | 1.65 | 1.73 | 1.82 | 1.87 |
| large, flat | 1.00 | 1.00 | 1.91 | 1.57 | 1.55 | 1.50 | 1.50 |
| large, codegen-block | 1.01 | 1.49 | 1.53 | 1.56 | 1.57 | 1.55 | 1.53 |

A shared 8-thread laptop (i7-8550U, other jobs running, load about 11: indicative only, not a published figure), one thread divided by 8 threads:

| forest, backend | 100 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 100,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| tiny, flat | 1.00 | 1.31 | 0.97 | 1.06 | 1.44 | 1.78 | 1.67 |
| tiny, codegen-block | 1.00 | 0.51 | 0.86 | 1.06 | 1.21 | 1.28 | 1.79 |
| small, flat | 1.00 | 1.03 | 1.83 | 1.97 | 2.56 | 2.76 | 3.15 |
| small, codegen-block | 1.03 | 1.31 | 2.15 | 3.06 | 3.91 | 4.67 | 5.43 |

Reading: starting a thread costs about 0.1 to 0.2 ms, so a tiny forest loses below roughly 4,096 rows (0.69x on the 2-core laptop at 4,096), while
a bigger one gains from about 2,000 rows; from 8,192 rows the all-cores call was faster in every measured case (1.2x to 5.4x on these
forests, close to the core count on a large batch). The cutoff is the conservative end of that range and is not tuned further. These are
cutoff measurements, not speed figures of the release; the one-thread ratios of the README are unchanged.

## The flat file

One little-endian binary file. A 64-byte header (magic `TREEFLT1`, format, counts of features, groups, folds, trees and nodes, the
objective, the fold-averaging rule, flags, and a 64-bit FNV-1a checksum of every byte of the file, the eight bytes of the checksum itself read as zero), the fold and tree records, then
the nodes: 8 bytes each, holding the float32 threshold (the leaf value at a leaf), the distance to the right child (the left child is the
next node, trees are laid out depth first), the feature and the default direction for NaN. The file is read through a memory map, so
loading costs a page-table update rather than a parse (`bench/README-NUMBERS.md` section 1 gives the load times). Formats: 4 for a plain
float32 forest (what an xgboost forest produces), 5 when the forest needs double sums, extended nodes (zero-as-missing, categories) or an
identity objective with several groups, 6 when it has linear leaves or, with `--f64-leaves`, leaves that float32 would round. (Formats 1 to 3
were written before the checksum covered the header; such a file is refused with a line that says to compile the model again.) The exact layout is the comment at the top of `rust/src/lib.rs`. A categorical node looks up `trunc(x)`: NaN, a value below 0 after truncation toward zero (so -0.5 is category 0, which can go left) and an absent category go right, as in LightGBM.

The margin scale `s` of a LightGBM binary model (above) is not used by the walker; the library reports it (`tfl_margin_scale`, which
`FlatForest.margin_scale` calls) from the file image it has verified, so Python never reads the header itself. In a format 4 or 5
file it is a double in header bytes 56..64 (0 means 1, NaN means several different scales). A format 6 file uses those bytes for the linear-leaf
counts, so the scale is the constant of one extra linear-leaf record at the end of the table that no leaf refers to (its fallback field holds a fixed
marker bit pattern, `evaluator.margin_scale_marker()`; a constant of -1 means several different scales).

The loader checks the header, the checksum, the record ranges, the feature indices and the child distances before it reads a node. The
checksum covers the whole file, the header included, so no single flipped bit of the 64 header bytes loads
(`tests/test_robust.py::FlatHeaderIsCheckedTest`). `tests/fuzz` and the Rust mutation test damage every field to check that, and a file
that loads counts as clean only when it answers bit for bit like the undamaged one. A file is never edited in place: a memory-mapped file that another
process truncates gives SIGBUS on the next access, and the compilers replace by rename.

## Structure export

```
python3 scripts/quickforest structure MODEL OUT.npz
```

```python
import sys

sys.path.insert(0, "scripts")
import structure

a = structure.export("model.ubj")  # dict of numpy arrays; structure.save(model, "out.npz") also writes the archive
```

A model's trees as plain arrays, read by the same readers as every other command; no row is run through the model. `MODEL` is ONE xgboost,
LightGBM or YDF model, pruned or not (a folder of folds is refused: export one fold at a time). The archive is `.npz` (numpy loads it with
`allow_pickle=False`, and a float32 survives it exactly; JSON would round-trip the thresholds through text); any other suffix is refused.

| array | dtype, shape | meaning |
|---|---|---|
| `format`, `kind`, `task` | text, 0-d | `xgboost` / `lightgbm` / `ydf`; the output transform (`identity`, `sigmoid`, `softmax`, `exp`); `classification` / `regression` |
| `n_features`, `n_groups`, `margin_scale` | int64, int64, float64; 0-d | inputs; output groups; the factor a base margin you add is multiplied by (LightGBM `sigmoid:s`; 1.0 for every other model, a LightGBM random forest included, whose averaged score is not scaled) |
| `base_margin` | float32 (groups,) | the margin every row starts from |
| `tree_group`, `source_tree` | int32 (trees,) | the group the tree adds to; the 0-based index of the library tree it comes from (the numbers of `kept_indices`; it differs from the tree's own number only for a YDF random-forest classification tree, which is one tree per class here) |
| `node_offset` | int64 (trees + 1,) | the nodes of tree `t` are `node_offset[t]:node_offset[t + 1]` of the arrays below |
| `left`, `right` | int32 (nodes,) | child ids inside the tree (0 is the root), -1 at a leaf |
| `feature` | int32 (nodes,) | the feature a split reads, -1 at a leaf |
| `threshold` | float32 (nodes,) | the split threshold, or the leaf value at a leaf, exactly as QuickForest reads it (LightGBM's `x <= t` is already the float32 `x < threshold`) |
| `default_left` | bool (nodes,) | where NaN goes; False at a leaf |
| `zero_missing` | bool (nodes,) | the node also sends `\|x\| <= float32(1e-35)` (1.0000000180025095e-35) the `default_left` way (LightGBM zero-as-missing); False at a leaf |
| `leaf_index` | int32 (nodes,) | the leaf's ordinal among its tree's leaves in node order, -1 at an inner node |

A walk: a row goes left when `x[feature] < threshold` (NaN the way `default_left` says; at a `zero_missing` node also a value with
`|x| <= float32(1e-35)`), and the margin of group `g` is `base_margin[g]` plus the leaf values the row reaches in the trees whose `tree_group` is `g`,
added in tree order (float32 sums for xgboost, double for LightGBM and YDF). Nodes xgboost dropped after growing them are not in the
arrays. A model with categorical splits or linear leaves cannot be written this way and is refused in one line, exit 1.
`tests/test_structure.py` walks the arrays in numpy against each library's margin (identical for xgboost, within `check.TOL` otherwise)
on random rows and on rows that land on, just below and just above thresholds.

## Rebuilding and testing

The Rust source declares rustc and cargo 1.88 as its minimum (`rust-version` in `Cargo.toml`; it has been built with 1.93 and 1.99); an older one stops with cargo's own message. The
generated code of the `codegen` backends is plainer and may build with older compilers too (`codegen-block` was run with rustc 1.75.0 on 2026-10-22, before the build defaults of 1.0; not tried with them).

```
RUST_DIR=<folder with rustup/ and cargo/> TMPDIR=<scratch> rust/build.sh      # fmt, clippy -D warnings, tests, build, copy to bin/
QUICK=1 rust/build.sh                                                          # build only (skips fmt, clippy and tests)
python3 -m unittest discover -s tests -p "test_*.py"                           # synthetic data only
python3 tests/differential/run_differential.py --mode quick                    # 36 cases
python3 tests/fuzz/fuzz_loader.py --mode quick                                 # about 3,300 cases
python3 tests/diag_pages.py                                                     # a report, not a test: which pages of a model file stay mapped
```

`tests/test_input_matrix.py` is the table of refusals: every command against every kind of damaged model, data array, artifact, output path and
numeric option must end in the exit code of the table above and ONE line on stderr, with no traceback, signal or hang. A new command, option or
input adds its row to `ENTRY_POINTS`, `DAMAGES` or `OPTIONS` there.

`RUST_DIR` is optional: without it `cargo` is taken from `PATH`. The YDF tests skip without `ydf`, and the generated-code ones without
`rustc`. `rust/build.sh` builds for the baseline instruction set of the machine's architecture and names the result
`bin/libtreeflat-$(uname -m).so`. The shipped x86-64 library comes from this script: rebuilt with `rustc` 1.99.0 in another folder it is the same file
byte for byte (same SHA-256); another `rustc` version gives a different file with the same exported functions. `rust/rust-toolchain.toml` pins 1.99.0 for rustup users, and CI rebuilds the library and reports whether its SHA-256 equals the committed one (a warning, not a failure: a different linker on the runner can change the bytes). The READMEs of `tests/differential` and `tests/fuzz` describe what each case is, what is checked and
the tolerance table.
