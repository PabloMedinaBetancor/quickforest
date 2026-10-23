# Fuzz tests

`fuzz_loader.py` feeds damaged and hostile input to every file and every entry point of QuickForest and checks that each one ends in a
clean Python exception with a message. Never a segfault, a hang, an out-of-bounds read, an internal error leaking out, or a wrong answer.

    python3 -s tests/fuzz/fuzz_loader.py --mode quick     # ~3,300 cases (fewer without ydf), 1-2 minutes on a quiet machine, runs in CI
    python3 -s tests/fuzz/fuzz_loader.py --mode full      # ~123k cases, 8 minutes on an 8-thread laptop (20 on a shared 4-thread one)
    python3 -s tests/fuzz/fuzz_loader.py --replay flat:xgb_multi:child:1234    # one case: target:base:kind:seed

Use a Python with numpy (and xgboost / lightgbm for the `xgb` / `lgb` targets). `--out DIR` chooses where findings and the replayed
file go, `--only flat,lgb` limits the targets, `--seed0 N` draws a different set. Exit code 1 when any case is not clean. `findings.txt` has one line per non-clean case, and each line is a replay id.

## Targets

| target | what is attacked |
|---|---|
| `flat` | the flat forest file, the only file the Rust evaluator reads: truncation at every kind of length, extension, bit flips (checksum left wrong, and fixed so the structure checks are reached; single bits of the header, which the checksum covers), wrong magic / format / flags, huge and zero counts in every header field, child distances and feature indices out of range, NaN thresholds, tree and fold records out of order, damaged category and linear-leaf tables, 2-5 mutations stacked. A file that loads is also predicted with every engine, 1 and 3 threads, float32 and float64, with and without a base margin; the engines must agree bit for bit, and the answers must be bit-identical to the undamaged file's (a mutation whose checksum was repaired is the exception: it loads a different forest, which is what repairing it means, and is counted apart as `LOADED a different forest`). |
| `lgb` | a LightGBM text model: truncation, removed fields, huge or negative counts, child indices out of range or cyclic, non-numeric numbers, NUL bytes. |
| `xgb` | an xgboost JSON model: the same, on the JSON (cycles, two parents, out-of-range children). |
| `npz` | the selection / data file (`load_data`): truncated archive, missing X, wrong dtypes, pickled objects. |
| `choice` | `ARTIFACT.choice.json`: invalid JSON, unknown backend, wrong types, a path to a Python file (must never be run). |
| `api` | misuse of the Python API: wrong shapes and dtypes, closed forest, bad `threads` / `engine` / margin, odd paths (FIFO, directory, empty file, NUL byte), concurrent predict from several Python threads. |

The Rust side has its own mutation test (`cargo test`, `damaged_files_are_rejected_or_predict_inside_their_own_memory`): 9 fixture
files in `rust/tests/data/*.flat`, 1,500 mutations each by default (`TREEFLAT_MUTATIONS=N`), run against the library directly, in the
debug profile where integer overflow and out-of-bounds panics. It applies the same oracle: a mutant that loads must answer bit for bit like the original, unless its checksum was repaired, and it counts those apart.

The nine files in `rust/tests/data/` hold nothing but small models trained on random numbers: they are the flat files that `build_bases` in `fuzz_loader.py`
makes (xgboost, LightGBM and YDF models with fixed seeds, compiled with the flat backend). `fuzz_loader.py --mode quick --out DIR` writes that set to
`DIR/work/bases/`; the Rust test uses nine of the files (`xgb_multi`, `xgb_bin`, `xgb_folds`, `xgb_deep`, `lgb_cat`, `lgb_cat_f64`, `lgb_linear`, `lgb_multi`, `ydf_f32`).

## What "clean" means

Each target runs in a child process that handles a batch of cases. A child that dies by a signal, or does not answer within 30 s, is
recorded as CRASH / HANG on the case it was on, and a new child continues with the next case.

- Clean: `ValueError`, `RuntimeError` (including `EngineRefused`), `OSError`, `EOFError`, or a library's own error class (`XGBoostError`,
  `LightGBMError`, `BadZipFile`, `YdfError`) with a non-empty message; or the file loads, all engines give the same bits, and the answers are bit-identical to the undamaged file's (or its checksum was repaired by the mutation: `LOADED a different forest`, expected, counted apart).
- Not clean: CRASH (signal), HANG, `KeyError`, `IndexError`, `TypeError`, `AttributeError`, `OverflowError`, `SystemExit`, `MemoryError`,
  an empty message, or DIVERGE (a file that loads but whose engines disagree, or a changed file that loads with the original checksum).

## What it does not cover

- A file truncated by ANOTHER process while it is memory-mapped makes the kernel send SIGBUS on the next page touched. This is how mmap works
  and cannot be caught in-process; do not shrink or overwrite a `.flat` in place while a service holds it (the compilers replace by rename).
- A damaged text model that QF's parser accepts can still abort or segfault xgboost or LightGBM when the library is asked to load it
  (a QF command that checks against the library does that in its own process). The fuzzer loads each such file with the library in a
  process of its own and counts a library crash apart, as `SOURCE LIBRARY CRASH`, in `library-crashes.txt`; it is not a failure of the run.
- A generated `.so` is code: `dlopen` runs its initialisers. Only load artifacts you built. The fuzz targets damage data files, not code.
- `get('file.py')` and `--backend file.py:Class` execute the named Python file by design (trusted input); a `choice.json` can name only
  built-in backends.
- `CodegenForest.predict` is not fuzzed (only `FlatForest` is); its inputs are shape-checked by the same Python wrapper.
- Seeds are fixed per mode, so a green run is a repeatable fact, not a probability of safety.

## Pinning tests

Every bug the fuzzer found has a unit test in `tests/test_robust.py` that fails on the old code (listed in `../../CHANGELOG.md`).
After a fix, add the test first, then re-run the fuzzer with the same replay id.

## Full run (seed0 1)

123,063 cases in 458 s on 2026-10-23, on the release code (the 1.0 build defaults), on an Intel Core i7-8550U (4 cores, 8 threads): flat 86,240, LightGBM text 18,000,
xgboost JSON 18,000, npz 500, choice 280, api 43. Result: 0 CRASH, 0 HANG, 0 UNCLEAN, 0 DIVERGE. Of the flat files, 74,655 were refused with a `ValueError` that names the file and the reason, 8,169 loaded and
gave bit-identical answers on all three engines, 1 and 3 threads, and 3,416 had their checksum repaired by the mutation and loaded a different forest (counted apart).
Of the text files, 79 (75 LightGBM, 4 xgboost) that QF's parser had accepted made the model's own library fail when asked to load them: 54 aborted
(SIGABRT), 4 segfaulted (SIGSEGV) and 21 needed more memory than the probe's address-space ceiling (3,072 MB above what the libraries hold after import). All 79 are counted apart
as `SOURCE LIBRARY CRASH` (see `library-crashes.txt`), none is a verdict on QuickForest, and none is a failure.
