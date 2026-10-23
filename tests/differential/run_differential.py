#!/usr/bin/env python3
"""Differential test at scale: thousands of small random models, each compiled with every public backend and compared with the library's own
predict.

    run_differential.py --mode quick            # about a minute, for CI (a fixed 36-case plan)
    run_differential.py --mode full             # thousands of models (default 3000), resumable
    run_differential.py --mode full --cases 5000 --workers 2 --seed0 7
    run_differential.py --case 1234 --seed0 1   # replay ONE case (prints every check, keeps the files)

What a case is (all of it is drawn from numpy's default_rng([seed0, case]), so a case number plus seed0 is a complete bug report):
  * a format  : xgboost (every third case), LightGBM, YDF
  * a model   : regression / binary / multiclass (and, per library, poisson, gamma, tweedie, quantile, ranking, ...), boosted or random
                forest, 1..3 k-fold members (averaged as output or as margin), 1 to 150+ trees, stumps to deep trees (depth 12, 255 leaves),
                NaN in the data (0 / 5 / 30 %), constant and collinear columns, LightGBM categorical splits, zero_as_missing, linear leaves,
                YDF winner-take-all and averaged forests
  * rows      : normal rows, NaN, exact zeros, the zero bucket (1e-36), +-1e6 and (when no linear leaf) +-3e38, categories unseen in training,
                and rows built from the model's own thresholds (exactly, +-1 float32 step)
  * compiled  : flat walker (engine 0), wave (engine 1) and wave-avx2 (engine 2) where the file can hold them, codegen-block and
                codegen-pgo on a share of the cases (rustc is the cost: --codegen-every / --pgo-every), with the opt-in --f64-leaves
                (LightGBM) and --float32-sums (YDF); float32 and float64 answers; no margin and a random base margin; 1 and 3 threads
  * checks    : (a) compiled vs the library, against the tolerance TABLE below; (b) every backend bit-identical to the walker;
                (c) threads 1 == threads 3; (d) float32 answer == float64 answer rounded; (e) the engines that cannot hold a file
                refuse it with EngineRefused / a one-line ValueError, never a crash.
Output (in --out, default tests/differential/out): results.jsonl (one line per finished case; the resume state), failures.txt (ONE line
per failed check, with the case number, the options and the error against the limit), summary.md (also printed). Exit code 1 if any
check failed or a case crashed.

Needs numpy, xgboost, lightgbm and ydf (any missing library skips its cases and says so), rustc for the generated-code backends
(skipped with a note when absent). Synthetic data only. See README.md in this folder for the tolerance table and how to read a failure."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

ULP32 = 2.0**-24  # half an ulp of a float32, relative
BM_MAX = 3.0  # the base margins the harness draws are uniform(-BM_MAX, BM_MAX)
YDF_F32S_CLS_GBT = 2.4e-7  # YDF --float32-sums, boosted classification: 4 ulp of a probability, measured 1.8e-7 (see the ydf-f32s-cls row)
# the same for a random forest: its leaves are divided by the number of trees BEFORE the float32 sum, measured 3.9e-7
YDF_F32S_CLS_RF = 5e-7
FUSING_MACHINES = ("aarch64", "arm64")  # LightGBM's wheel for these fuses the linear leaf's multiply-add (see the lgb-lin row)
FUSED_REL = 1e-12  # on those machines a linear-leaf answer may differ by this times the magnitude of its terms (see `fused_slack`)


# --------------------------------------------------------------------------------------------- the tolerance table
# One place, printed in summary.md and README.md. Error = |compiled - library| / max(1, |library|) (so a probability or a log-scale target
# is judged absolutely and a big value relatively), maximum over rows and outputs. "exact" means == 0.0.
TABLE = [
    # (key, what, float32 answer, float64 answer, where the claim comes from)
    (
        "xgb",
        "xgboost regression / binary / multi:softprob, no margin (folds averaged as output or as margin)",
        "exact (0)",
        "exact (0)",
        "docs/REFERENCE.md: xgboost's own arithmetic is float32 (the softmax normaliser is summed in double, as xgboost's common::Softmax does), the f64 answer is its float32 result widened",
    ),
    (
        "xgb+bm",
        "xgboost, base margin",
        "max(1e-6, S)",
        "max(1e-6, S)",
        "docs/REFERENCE.md: with a base margin xgboost's float32 order decides. S = N x 2^-24 x (B + init + 3) x 1.01, the bound of a sequential float32 sum of N trees (measured 1.2e-6 on an 80-tree model with answers near 100)",
    ),
    (
        "lgb",
        "LightGBM, default file (float32 leaves), identity objective",
        "6.1e-8",
        "exact (0); 1e-12 with a margin",
        "docs/REFERENCE.md: the default file stores each leaf rounded to float32 after the factors the compile folds in (sigmoid s, rf 1/iterations). The reference is the library's own Booster on a copy of the model whose leaves are folded and rounded the same way (`lgb_float32_leaf_files`): float32 of its own leaves, summed in double in tree order, so the answer is bit-identical to it, like --f64-leaves; f32 = half an ulp of the answer; the margin is added in a different order than numpy's raw + margin",
    ),
    (
        "lgb-tr",
        "LightGBM, default file, sigmoid / softmax / exp",
        "6.1e-8",
        "1e-12",
        "same, but the transform is numpy's exp against the evaluator's (an ulp of a double)",
    ),
    (
        "lgb-f64l",
        "LightGBM --f64-leaves, identity objective",
        "6.1e-8",
        "exact (0); 1e-12 with a margin",
        "docs/REFERENCE.md: bit-identical to Booster.predict; f32 = half an ulp of the answer; the margin is added in a different order than numpy's raw + margin",
    ),
    (
        "lgb-f64l-tr",
        "LightGBM --f64-leaves, sigmoid / softmax / exp",
        "6.1e-8",
        "1e-12",
        "same, but the transform is numpy's exp against the evaluator's (an ulp of a double)",
    ),
    (
        "lgb-lin",
        "LightGBM linear leaves, identity objective",
        "6.1e-8",
        "exact (0); 1e-12 with a margin; on aarch64 / arm64 also 1e-12 x (|answer| + sum of the terms' magnitudes) / max(1, |answer|) per row",
        "docs/REFERENCE.md: constants and coefficients stored in f64, 0.0 against Booster.predict. A plain (non-linear) tree inside a linear forest, e.g. the one-leaf tree of a constant target, keeps its leaf in float32 without --f64-leaves: the reference rounds that tree's leaves the same way (as for the key lgb). Averaging folds or adding a base margin is exact only to 8 x 2^-52 x S per row, S = the sum of |every tree's value| (the largest partial sum: a 1e6 input times a coefficient). On aarch64 / arm64 a row also gets an allowance of 1e-12 x (|answer| + P) / max(1, |answer|), P = the sum over the trees of |constant| + the sum of |coefficient x input| of the leaf the row reaches (a platform rule that scales with the terms, not a looser bar elsewhere): LightGBM's aarch64 wheel fuses const + coeff x x into one multiply-add (GCC's default there), so the library's own answer differs by one rounding of the largest partial sum (about 1e-16 of it) from the evaluator's, which never fuses, and from LightGBM's x86 build; with 1e6 inputs and an answer near zero that is far above a fixed 1e-12",
    ),
    ("lgb-lin-tr", "LightGBM linear leaves, sigmoid / softmax / exp", "6.1e-8", "1e-12", "same, with the transform"),
    (
        "ydf-dbl",
        "YDF, double sums (default): the documented float32-summation class",
        "C + 2.6e-7",
        "C + 2e-7",
        "C = N x 2^-24 x (|init| + sum of max abs leaf) x 1.01: bound of YDF's own sequential float32 sum; reported separately from the exact rows",
    ),
    (
        "ydf-f32s-reg",
        "YDF --float32-sums, regression",
        "exact (0) without margin; 6.1e-8 with margin",
        "exact (0), one fold; 6.1e-8 for several folds",
        "docs/REFERENCE.md: reproduces model.predict bit for bit; the reference rounds to float32 where the margin or the fold average is added",
    ),
    (
        "ydf-f32s-cls",
        "YDF --float32-sums, classification (and Poisson)",
        "boosted 2.4e-7, random forest 5e-7",
        "boosted 2.4e-7, random forest 5e-7",
        "docs/REFERENCE.md: the float32 sums are YDF's own bit for bit, the probability is not: YDF's float32 transform differs by an ulp or three (boosted trees: measured 1.8e-7); a random forest's leaves are divided by the number of trees before the float32 sum here and after it in YDF (measured 3.9e-7 at 30 trees)",
    ),
    (
        "backend",
        "any backend against the flat walker",
        "exact (0)",
        "exact (0)",
        "docs/REFERENCE.md: all backends bit-identical to the walker",
    ),
    ("threads", "3 threads against 1 thread", "exact (0)", "exact (0)", "rows are independent"),
    (
        "round",
        "float32 answer against float64 answer rounded to float32",
        "exact (0)",
        "-",
        "every file: the f32 answer is the f64 one rounded",
    ),
    (
        "written",
        "a pruned model file (real-world provenance cases) reloaded by its own library and predicted through the default path "
        "(xgboost Booster.predict and the scikit-learn wrapper, LightGBM Booster(model_file).predict) against the compiled pruned forest",
        "xgboost exact (0); LightGBM as lgb-f64l",
        "xgboost exact (0); LightGBM as lgb-f64l",
        "docs/REFERENCE.md: prune writes a normal model file of the same format; a user reads it back with the library's own predict",
    ),
    (
        "refusal",
        "wave engines on a file they cannot hold",
        "EngineRefused",
        "EngineRefused",
        "docs/REFERENCE.md: raises EngineRefused, never a crash or a wrong answer",
    ),
]


def table_markdown() -> str:
    out = ["| key | what | float32 answer | float64 answer | claim |", "|---|---|---|---|---|"]
    out += [f"| {k} | {w} | {a} | {b} | {c} |" for k, w, a, b, c in TABLE]
    return "\n".join(out)


FLT_MAX = float(np.finfo(np.float32).max)


def saturated(ref, got) -> np.ndarray:
    """Where the library's answer is beyond FLT_MAX in magnitude (or infinite) and the compiled answer is +-inf or +-FLT_MAX with the library's sign. Float32's
    range ends at FLT_MAX, so there a double library's 4e38 and a float32 inf or FLT_MAX cannot agree to the bit, and this is as close as
    they can be. Any other answer there (0, NaN, the wrong sign, a number far off) is a wrong answer and keeps its error. float64 arrays."""
    with np.errstate(all="ignore"):
        edge = np.isinf(got) | (np.abs(got) == FLT_MAX)
        return (np.abs(ref) > FLT_MAX) & edge & (np.sign(got) == np.sign(ref))


def rel_err(ref, got) -> float:
    """max |got - ref| / max(1, |ref|); NaN against NaN is equal, NaN against a number is infinite; beyond float32's range see `saturated`.
    ref is float64-ised."""
    ref = np.asarray(ref, np.float64).reshape(len(ref), -1)
    got = np.asarray(got, np.float64).reshape(len(ref), -1)
    same = (ref == got) | (np.isnan(ref) & np.isnan(got))
    with np.errstate(all="ignore"):
        d = np.where(same, 0.0, np.abs(got - ref) / np.maximum(1.0, np.abs(ref)))
    d = np.where(np.isnan(d), np.inf, d)
    d = np.where(saturated(ref, got), 0.0, d)
    return float(d.max()) if d.size else 0.0


def same_bits(a, b) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def forest_scale(forest) -> tuple[float, int, float]:
    """(B, N, init): B = largest, over models and output groups, of the sum over that group's trees of max |leaf|; N = most trees of one
    group in one model; init = largest |initial prediction|. The inputs of the proven float32 bounds below."""
    B, N, init = 0.0, 0, 0.0
    for m in forest.models:
        per_group: dict[int, list] = {}
        for t in m.trees:
            leaves = (t.leaf64 if t.leaf64 is not None else t.value.astype(np.float64))[t.left == -1]
            per_group.setdefault(int(t.group), []).append(float(np.abs(leaves).max()) if len(leaves) else 0.0)
        for v in per_group.values():
            B, N = max(B, sum(v)), max(N, len(v))
        init = max(init, float(np.abs(np.asarray(m.base, np.float64)).max()) if len(m.base) else 0.0)
    return B, N, init


def lgb_float32_leaf_files(files, out_dir: Path) -> list[Path]:
    """Copies of LightGBM text models whose plain leaves are what the default compile stores: each leaf times the factor the compile folds
    into it (a binary `sigmoid:s` is a factor s, an rf model `average_output` a factor 1/iterations: `scale / (iterations if rf else 1)`,
    one double multiplication) and then rounded to float32, one value at a time, to nearest. The copy then says what is already folded
    in: `sigmoid:1` and no `average_output`. The library's own Booster on these copies is the exact oracle of the default file: float32
    of its own leaves, summed in double in tree order, the objective's transform and the folds' average done by the library's own
    arithmetic. No bound is needed, the comparison is exact (see `bar`). A folds set with a linear tree (constants, coefficients and
    fallback stay in double) is the exception: only the plain leaves are rounded, and the headers are left as they are.
    `lgb_oracle_folds_factor` says which of the two a file got."""
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    any_linear = any(lgb_has_linear_tree(f) for f in files)  # one folds set, one rule: the base margin is scaled the same way for all
    for f in files:
        blocks = re.split(r"(?m)^(?=Tree=\d+$)", Path(f).read_text())
        blocks[0] = re.sub(r"(?m)^tree_sizes=.*\n", "", blocks[0])  # the byte size of each tree block: the rounded numbers change them
        factor = 1.0
        if not any_linear:
            head = blocks[0]
            rf = re.search(r"(?m)^average_output$", head) is not None
            groups = int(re.search(r"(?m)^num_tree_per_iteration=(\d+)$", head).group(1))
            scale = re.search(r"(?m)^objective=binary sigmoid:(\S+)$", head)
            factor = (float(scale.group(1)) if scale else 1.0) / ((len(blocks) - 1) // groups if rf else 1)
            head = re.sub(r"(?m)^average_output\n", "", head)
            blocks[0] = re.sub(r"(?m)^(objective=binary sigmoid:)\S+$", r"\g<1>1", head)

        def rounded(m, factor=factor):
            return "leaf_value=" + " ".join(repr(float(np.float32(float(v) * factor))) for v in m.group(1).split())

        for k, b in enumerate(blocks):
            if k and not re.search(r"(?m)^is_linear=1$", b):
                blocks[k] = re.sub(r"(?m)^leaf_value=(.*)$", rounded, b)
        made.append(out_dir / Path(f).name)
        made[-1].write_text("".join(blocks))
    return made


def lgb_has_linear_tree(path) -> bool:
    return re.search(r"(?m)^is_linear=1$", Path(path).read_text()) is not None


def lgb_oracle_folds_factor(files, forest) -> float:
    """The factor a base margin must carry for the oracle files of `lgb_float32_leaf_files`: the factor the library would have applied
    itself (the forest's `margin_scale`: s of a binary `sigmoid:s`, 1 for rf) is now folded into the leaves, so the margin is scaled here
    instead. 1 for a file with a linear tree, whose copy keeps its header."""
    return 1.0 if any(lgb_has_linear_tree(f) for f in files) else forest.margin_scale


def libm_expf(a: np.ndarray) -> np.ndarray:
    """The C library's `expf`, which is what xgboost calls (numpy's float32 exp is its own SIMD code and differs by an ulp for some inputs)."""
    import ctypes
    import ctypes.util

    lib = ctypes.CDLL(ctypes.util.find_library("m"))
    lib.expf.restype, lib.expf.argtypes = ctypes.c_float, [ctypes.c_float]
    return np.vectorize(lambda v: lib.expf(float(v)), otypes=[np.float32])(a)


def xgb_margin_average_reference(files, x: np.ndarray, kind: str) -> np.ndarray:
    """xgboost folds averaged as margin, without a base margin, in xgboost's own float32 arithmetic: the library's margin of each fold,
    summed in fold order and divided by their number, then xgboost's transform (`expf` of the C library; the sigmoid with the clamp of
    `common::Sigmoid`; the softmax normaliser summed in double and rounded to float32 once, as `common::Softmax` does). Exact, so the key `xgb` applies. (`qf_common.transform` is numpy's
    float32 arithmetic.)"""
    import qf_formats

    parts = [qf_formats.open_source(f).predict(x, margin=True) for f in files]
    m = np.mean(parts, axis=0, dtype=np.float32).astype(np.float32)
    one = np.float32(1)
    if kind == "softmax":
        e = libm_expf((m - m.max(-1, keepdims=True)).astype(np.float32))
        return (e / np.cumsum(e.astype(np.float64), axis=-1)[..., -1:].astype(np.float32)).astype(np.float32)
    if kind == "sigmoid":  # `common::Sigmoid`: the exponent is clamped at 88.7f (NaN stays NaN) and 1e-16f is added
        x = np.minimum(-m, np.float32(88.7)).astype(np.float32)
        return (one / (libm_expf(x) + one + np.float32(1e-16))).astype(np.float32)
    return m


def threshold_rows(forest, nf: int, cap: int = 24000) -> np.ndarray:
    """Rows that HUG every split of the last tree of each model: for each inner node, a row that reaches the node (the splits above it are
    satisfied) with the node's feature at its threshold, one float32 step below, one step above and NaN (a categorical node: a member, the
    edges of the set, a value outside it, NaN). `check.random_rows` hugs a random threshold of the whole forest, so a given node of the last
    tree was hit by luck; a defect planted in that tree (a swapped child, a flipped default direction, a threshold one step off) is seen
    only when a row reaches its node and falls on the right side of it. `cap` rows per model at most (nodes are then taken evenly)."""
    inf, f32 = np.float32(np.inf), np.float32
    rows = []
    for m in forest.models:
        if not m.trees:
            continue
        t = m.trees[-1]
        mine: list[np.ndarray] = []
        stack = [(0, np.full(nf, -inf, f32), np.full(nf, inf, f32), np.full(nf, np.nan, f32))]
        while stack:
            node, lo, hi, fixed = stack.pop()
            if t.left[node] == -1:
                continue
            # a point of every feature's open interval [lo, hi) (the categories a split fixed stay as they are)
            with np.errstate(all="ignore"):
                mid = (lo.astype(np.float64) + hi.astype(np.float64)) / 2
                v = np.where(
                    np.isfinite(lo) & np.isfinite(hi),
                    mid,
                    np.where(
                        np.isfinite(hi), hi - np.maximum(1, np.abs(hi)), np.where(np.isfinite(lo), lo + np.maximum(1, np.abs(lo)), 0.5)
                    ),
                )
            v = v.astype(f32)
            v = np.where(v < hi, v, np.nextafter(hi, -inf)).astype(f32)
            v = np.where(v >= lo, v, lo).astype(f32)
            base = np.where(np.isnan(fixed), v, fixed).astype(f32)
            f = int(t.feature[node])
            cats = (t.cat or {}).get(node)
            if cats is not None:
                c = np.asarray(cats, np.float64)
                cands = [f32(x) for x in (c[0], c[-1], c.max() + 1, c.max() + 0.5, -1.0)] if len(c) else [f32(0)]
                inside = [f32(c[0])] if len(c) else [f32(0)]
                outside = [f32(c.max() + 1) if len(c) else f32(1)]
            else:
                th = f32(t.value[node])
                cands = [th, np.nextafter(th, -inf), np.nextafter(th, inf)]
            for x in [*cands, f32(np.nan)]:
                r = base.copy()
                r[f] = x
                mine.append(r)
            if cats is not None:
                fx_l, fx_r = fixed.copy(), fixed.copy()
                fx_l[f], fx_r[f] = inside[0], outside[0]
                stack.append((int(t.left[node]), lo, hi, fx_l))
                stack.append((int(t.right[node]), lo, hi, fx_r))
            else:
                hi_l, lo_r = hi.copy(), lo.copy()
                hi_l[f] = min(hi_l[f], th)
                lo_r[f] = max(lo_r[f], th)
                stack.append((int(t.left[node]), lo, hi_l, fixed))
                stack.append((int(t.right[node]), lo_r, hi, fixed))
        if len(mine) > cap:
            mine = [mine[i] for i in np.linspace(0, len(mine) - 1, cap).astype(int)]
        rows += mine
    return np.array(rows, f32).reshape(-1, nf)


# --------------------------------------------------------------------------------------------- model generators
# What a recorded result belongs to. A case number alone does not name a case: the same number under another --seed0, or under a changed
# generator (this file), is another model. The resume state keeps only the records of THIS seed0 and THIS generator.
GENERATOR = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]


def stamp(seed0: int) -> dict:
    return {"seed0": seed0, "gen": GENERATOR}


class Skip(Exception):
    """A case that cannot be built in this environment (a library missing); counted, not failed."""


WIDE_CATEGORIES = 45  # categories 32 and above need a second 32-bit word in LightGBM's category bitset


def gen_matrix(rng, n, nf, cat_cols=(), categories=None):
    x = (rng.normal(size=(n, nf)) * rng.choice([0.1, 1, 30])).astype(np.float32)
    if rng.random() < 0.4:
        x[:, 0] = 3.0  # a constant column
    if rng.random() < 0.4 and nf > 2:
        x[:, 1] = x[:, 2] * 2  # collinear columns
    for c in cat_cols:
        drawn = int(rng.choice([3, 6, 12]))
        x[:, c] = rng.integers(0, categories or drawn, n)
    x[rng.random(x.shape) < rng.choice([0, 0.05, 0.3])] = np.nan
    return x


def target(rng, x, kind):
    z = np.nan_to_num(x)
    nf = x.shape[1]
    y = z[:, 0] * 0.3 + 2 * z[:, min(3, nf - 1)] + (z[:, min(2, nf - 1)] > 0) * 3 + rng.normal(size=len(x)) * 0.3
    if rng.random() < 0.03:
        y = np.full(len(x), 1.5)  # no split is worth making: trees of a single leaf
    if kind == "binary":
        y = (y > np.median(y)).astype(float)
        if rng.random() < 0.05:
            y[:] = 0.0
    elif kind == "multiclass":
        y = np.digitize(y, np.quantile(y, [0.33, 0.66])) if np.ptp(y) > 0 else np.zeros(len(y))
        y = np.minimum(y, 2).astype(int)
        if len(np.unique(y)) < 3:  # a constant target: keep all three classes so every fold of a k-fold set has the same schema
            y[:3] = [0, 1, 2]
    return y


def tree_count(rng, hi=150, shallow=False):
    """Few trees (1-5) a third of the time, many (up to 400 when the trees are shallow) an eighth of it."""
    return int(rng.choice([1, 2, 3, 5, 10, 30, 80, 400 if shallow else hi]))


# How people really train. The generator otherwise calls each library's native API on a bare matrix, which is not how most models are made.
# One case in six of a library (see `provenance`) is trained the other way:
#   flavor 0: through pandas. xgboost: a DMatrix built from a DataFrame stores the feature names and types in the file. LightGBM: a column of
#             the `category` dtype (the model file then ends with a pandas_categorical line).
#   flavor 1: a scikit-learn wrapper (XGBClassifier / XGBRegressor / LGBMClassifier / LGBMRegressor) fitted on a DataFrame with early stopping on
#             a validation tail: the xgboost file carries best_iteration / best_score and the rounds past the best; the wrapper saves its own meta.
# Such a case also writes a pruned copy of every model file, reloads it with the model's own library and predicts through the DEFAULT path
# (`written_files_check`). Without pandas (or, for flavor 1, scikit-learn) the case is skipped and the summary counts it with the reason.
def provenance(case: int):
    """None, or the flavor of the case: cases 15-17 and 33-35 of every 36 (k = case // 3, one library per case % 3), so quick mode has both."""
    k = case // 3
    return (k // 6) % 2 if k % 6 == 5 else None


def need_provenance_libraries(flavor: int) -> None:
    try:
        import pandas  # noqa: F401
    except ImportError as e:
        raise Skip(f"real-world provenance cases need pandas ({e})") from e
    if flavor == 1:
        try:
            import sklearn  # noqa: F401
        except ImportError as e:
            raise Skip(f"scikit-learn wrapper cases need scikit-learn ({e})") from e


def feature_names(nf: int) -> list[str]:
    """Names like a DataFrame brings: a space and a non-ASCII letter in one of them."""
    return [f"feat {i}" + ("\u00e9" if i == 0 else "") for i in range(nf)]


def train_xgb_real(flavor, p, rounds, rf, obj, x, y, names):
    """(model with save_model, "classifier" | "regressor" | None): the common ways to train an xgboost model that the native call hides."""
    import pandas as pd
    import xgboost as xgb

    frame = pd.DataFrame(x, columns=names)
    classes = np.unique(y)
    classifier = obj in ("binary:logistic", "multi:softprob") and len(classes) > 1 and np.array_equal(classes, np.arange(len(classes)))
    if flavor == 0 or rf or (obj == "multi:softprob" and not classifier):
        b = xgb.train(p, xgb.DMatrix(frame, label=y), rounds)
        assert b.feature_names == names
        return b, None
    q = dict(p)
    q.update(learning_rate=q.pop("eta"), n_jobs=q.pop("nthread"), random_state=q.pop("seed"))
    q.pop("num_class", None)  # the classifier counts the classes itself
    cut = int(len(x) * 0.75)
    cls = xgb.XGBClassifier if classifier else xgb.XGBRegressor
    ycol = y.astype(int) if classifier else y
    m = cls(n_estimators=rounds, early_stopping_rounds=5, **q)
    m.fit(frame[:cut], ycol[:cut], eval_set=[(frame[cut:], ycol[cut:])], verbose=False)
    return m, "classifier" if classifier else "regressor"


def train_lgb_real(flavor, p, ntree, obj, boost, x, y, group, cat, names):
    """A trained LightGBM Booster, from a DataFrame whose categorical columns have the `category` dtype (flavor 0), or from a scikit-learn
    wrapper with early stopping (flavor 1; the wrapper cuts its booster to the best rounds, as LightGBM does)."""
    import lightgbm as lgb
    import pandas as pd

    frame = pd.DataFrame(x, columns=names)
    for c in cat:
        codes = np.nan_to_num(x[:, c], nan=-1).astype(np.int64)  # NaN is the missing category (-1)
        frame[names[c]] = pd.Categorical.from_codes(codes, categories=[f"c{j}" for j in range(max(1, int(codes.max()) + 1))])
    classes = np.unique(y)
    if flavor == 0 or obj == "lambdarank":
        return lgb.train(p, lgb.Dataset(frame, y, categorical_feature="auto", group=group), ntree)
    q = {k: v for k, v in p.items() if k not in ("boosting", "seed", "num_threads", "verbose", "num_class")}
    q.update(boosting_type=boost, random_state=p["seed"], n_jobs=p["num_threads"], verbosity=p["verbose"])
    classifier = (obj == "binary" and len(classes) == 2) or (obj == "multiclass" and len(classes) >= 3)
    cut = int(len(x) * 0.75)
    m = (lgb.LGBMClassifier if classifier else lgb.LGBMRegressor)(n_estimators=ntree, **q)
    stop = [lgb.early_stopping(5, verbose=False)] if boost in ("gbdt", "goss") else []  # dart and rf do not early-stop
    m.fit(frame[:cut], y[:cut], eval_set=[(frame[cut:], y[cut:])] if stop else None, callbacks=stop, categorical_feature="auto")
    return m.booster_


def make_xgb(rng, d: Path, case: int = 0):
    import xgboost as xgb

    obj = str(rng.choice(["reg:squarederror", "reg:squarederror", "binary:logistic", "reg:logistic", "multi:softprob", "multi:softprob"]))
    kind = {"reg:squarederror": "regression", "binary:logistic": "binary", "reg:logistic": "binary", "multi:softprob": "multiclass"}[obj]
    nfold = int(rng.choice([1, 1, 2, 3]))
    nf = int(rng.integers(3, 14))
    rf = bool(rng.random() < 0.25)
    depth = int(rng.choice([1, 2, 3, 6, 10, 12]))
    grow = str(rng.choice(["depthwise", "lossguide"]))
    n = int(rng.integers(120, 700))
    ncls = int(rng.integers(3, 6))
    ntree = tree_count(rng, 150 if depth <= 6 else 60, shallow=depth <= 3)
    # the tree method and gamma come from a stream of their own (a jumped copy of this case's generator), so the draws above and below keep
    # giving the models they always gave. gamma > 0 makes the exact method keep its dropped splits in the model's arrays.
    side = np.random.Generator(rng.bit_generator.jumped())
    method = str(side.choice(["hist", "hist", "hist", "exact", "approx"]))
    gamma = float(side.choice([0.0, 0.3, 1.0, 4.0] if method != "hist" else [0.0, 0.0, 0.0, 0.5]))
    desc = f"xgb {obj} {'rf' if rf else 'gbtree'} depth={depth} {grow} trees={ntree} nf={nf} folds={nfold} {method} gamma={gamma:g}"
    prov = provenance(case)
    names, wrappers = None, []
    if prov is not None:
        need_provenance_libraries(prov)
        names = feature_names(nf)
        desc += f" provenance={'dataframe' if prov == 0 else 'sklearn-early-stopping'}"
    for k in range(nfold):
        x = gen_matrix(rng, n, nf)
        y = target(rng, x, "multiclass" if obj == "multi:softprob" else kind)
        if obj == "multi:softprob":
            y = np.minimum(y, ncls - 1)
        elif kind == "regression":
            y = y + float(rng.choice([0, 100]))
        p = {
            "objective": obj,
            "max_depth": depth,
            "eta": float(rng.choice([0.1, 0.3, 1.0])),
            "nthread": 1,
            "verbosity": 0,
            "seed": int(rng.integers(0, 10**6)),
            "tree_method": method,
            "gamma": gamma,
            "grow_policy": grow,
            "max_bin": int(rng.choice([16, 256])),
        }
        if grow == "lossguide":
            p["max_leaves"] = int(rng.choice([4, 12, 64]))
        if obj == "multi:softprob":
            p["num_class"] = ncls
        if rng.random() < 0.3 and obj != "multi:softprob":
            p["base_score"] = float(rng.normal()) if obj == "reg:squarederror" else float(rng.choice([0.2, 0.5, 0.8]))
        rounds = ntree
        if rf:
            p.update(num_parallel_tree=min(ntree, 40), subsample=0.7, eta=1.0)
            p["colsample_bytree" if method == "exact" else "colsample_bynode"] = 0.7  # the exact method has no per-node column sampling
            rounds = 1
        if prov is None:
            b = xgb.train(p, xgb.DMatrix(x, label=y), rounds)
        else:
            b, wrapper = train_xgb_real(prov, p, rounds, rf, obj, x, y, names)
            wrappers.append(wrapper)
        b.save_model(str(d / f"fold{k}.{rng.choice(['ubj', 'json'])}"))
    return desc, "xgboost", kind, {"nf": nf, "cat": [], "folds": nfold, "provenance": prov, "names": names, "wrappers": wrappers}


def make_lgb(rng, d: Path, case: int = 0):
    import lightgbm as lgb

    obj = str(
        rng.choice(
            [
                "regression",
                "regression",
                "binary",
                "multiclass",
                "poisson",
                "regression_l1",
                "huber",
                "gamma",
                "tweedie",
                "cross_entropy",
                "quantile",
                "fair",
                "mape",
                "lambdarank",
            ]
        )
    )
    boost = str(rng.choice(["gbdt", "gbdt", "gbdt", "dart", "goss", "rf"]))
    lin = (case // 3) % 3 == 0 and obj not in ("regression_l1", "quantile", "mape", "lambdarank")
    if obj == "lambdarank" and boost == "rf":
        boost = "gbdt"
    if boost == "rf" and obj in ("lambdarank",):
        boost = "gbdt"
    nf = int(rng.integers(3, 12))
    nfold = int(rng.choice([1, 1, 2, 3]))
    leaves = int(rng.choice([2, 3, 7, 15, 31, 255]))
    ntree = tree_count(rng, 120 if leaves <= 31 else 40, shallow=leaves <= 7)
    kind = {"binary": "binary", "cross_entropy": "binary", "multiclass": "multiclass"}.get(obj, "regression")
    ncat = int(rng.choice([0, 0, 1, 2]))
    wide = case % 7 == 4  # a share of the cases has a categorical column of WIDE_CATEGORIES values, whose split sets span several words
    ncat = max(ncat, 1) if wide else ncat
    cat = list(range(nf - ncat, nf))
    n = int(rng.integers(150, 1500 if leaves > 31 else 700))
    zero_missing = bool(rng.random() < 0.15) and not lin
    desc = f"lgb {obj}/{boost} lin={lin} leaves={leaves} trees={ntree} cat={len(cat)}{'x' + str(WIDE_CATEGORIES) if wide else ''} zam={zero_missing} nf={nf} folds={nfold}"
    prov = provenance(case)
    names = None
    if prov is not None:
        need_provenance_libraries(prov)
        names = feature_names(nf)
        desc += f" provenance={'dataframe' if prov == 0 else 'sklearn-early-stopping'}"
    for k in range(nfold):
        x = gen_matrix(rng, n, nf, cat, WIDE_CATEGORIES if wide else None)
        y = target(rng, x, kind)
        if wide:  # the target depends on the category, so that the trees split on it with sets of many categories
            y = y + np.nan_to_num(x[:, cat[0]]) % 7
            if kind != "regression":
                y = (y > np.median(y)).astype(int) if kind == "binary" else np.digitize(y, np.quantile(y, [0.33, 0.66]))
        p = dict(
            objective=obj,
            boosting=boost,
            linear_tree=lin,
            linear_lambda=float(rng.choice([0, 0.1, 5])),
            num_leaves=leaves,
            learning_rate=0.1,
            verbose=-1,
            num_threads=1,
            deterministic=True,
            min_data_in_leaf=int(rng.choice([2, 5, 20])),
            max_depth=int(rng.choice([-1, 2, 4, 12])),
            seed=int(rng.integers(0, 10**6)),
            use_missing=bool(rng.random() < 0.85),
            zero_as_missing=zero_missing,
            max_cat_to_onehot=int(rng.choice([2, 4, 8])),
            min_data_per_group=3,
            cat_smooth=1.0,
        )
        group = None
        if boost == "rf":
            p.update(bagging_freq=1, bagging_fraction=0.7, feature_fraction=0.8)
        if boost == "dart":
            p.update(drop_rate=0.3)
        if obj in ("binary", "cross_entropy"):
            if rng.random() < 0.3 and obj == "binary":
                p["sigmoid"] = 1.7
        elif obj == "multiclass":
            p["num_class"] = 3
        elif obj in ("poisson", "tweedie"):
            y = rng.poisson(np.exp(0.2 * np.clip(np.nan_to_num(x[:, 0]), -3, 3)))
        elif obj == "gamma":
            y = np.exp(np.clip(y, -3, 3) * 0.3) + 0.1
        elif obj == "mape":
            y = np.abs(y) + 1.0
        elif obj == "quantile":
            p["alpha"] = 0.8
        elif obj == "lambdarank":
            y = np.digitize(y, np.quantile(y, [0.5, 0.8])).astype(int)
            group = [10] * (n // 10)
            x, y = x[: 10 * (n // 10)], y[: 10 * (n // 10)]
        if prov is None:
            ds = lgb.Dataset(x, y, categorical_feature=cat if cat else "auto", group=group)
            booster = lgb.train(p, ds, ntree)
        else:
            booster = train_lgb_real(prov, p, ntree, obj, boost, x, y, group, cat, names)
        booster.save_model(str(d / f"fold{k}.txt"))
    meta = {"nf": nf, "cat": cat, "lin": lin, "obj": obj, "folds": nfold, "rf": boost == "rf", "wide": wide, "provenance": prov}
    return desc, "lightgbm", kind, {**meta, "names": names}


def make_ydf(rng, d: Path, case: int = 0):
    import ydf

    rf = bool(rng.random() < 0.4)
    kind = str(rng.choice(["regression", "regression", "binary", "multiclass"]))
    nfold = int(rng.choice([1, 1, 2]))
    nf = int(rng.integers(3, 10))
    depth = int(rng.choice([2, 3, 5, 8, 12, 16]))
    ntree = tree_count(rng, 100 if depth <= 5 else 30, shallow=depth <= 3)
    wta = bool(rng.random() < 0.5)
    loss = None
    if not rf:
        loss = {
            "regression": str(rng.choice(["SQUARED_ERROR", "SQUARED_ERROR", "MEAN_AVERAGE_ERROR", "POISSON"])),
            "binary": "BINOMIAL_LOG_LIKELIHOOD",
            "multiclass": "MULTINOMIAL_LOG_LIKELIHOOD",
        }[kind]
    n = int(rng.integers(150, 800))
    desc = f"ydf {'rf' if rf else 'gbt'} {kind} {loss or ('wta' if wta else 'avg')} depth={depth} trees={ntree} nf={nf} folds={nfold}"
    for k in range(nfold):
        x = gen_matrix(rng, n, nf)
        y = target(rng, x, kind)
        if kind == "regression":
            y = y + float(rng.choice([0, 12, 20]))
            if loss == "POISSON":
                y = rng.poisson(np.exp(0.2 * np.clip(np.nan_to_num(x[:, 0]), -3, 3))).astype(float)
        extra = {} if rf else dict(shrinkage=float(rng.choice([0.1, 0.02])), loss=loss)
        if rf and kind != "regression":
            extra["winner_take_all"] = wta
        task = ydf.Task.REGRESSION if kind == "regression" else ydf.Task.CLASSIFICATION
        lrn = ydf.RandomForestLearner if rf else ydf.GradientBoostedTreesLearner
        ycol = y if kind == "regression" else y.astype(int)
        if kind != "regression" and len(np.unique(ycol)) < 2:
            ycol[:2] = [0, 1]
        model = lrn(
            label="y", task=task, num_trees=ntree, max_depth=depth, random_seed=int(rng.integers(1, 10**6)), num_threads=1, **extra
        ).train({**{f"f{i}": x[:, i] for i in range(nf)}, "y": ycol}, verbose=0)
        model.save(str(d / f"fold{k}"))
    return desc, "ydf", kind, {"nf": nf, "cat": [], "rf": rf, "folds": nfold}


MAKERS = {"xgboost": make_xgb, "lightgbm": make_lgb, "ydf": make_ydf}


def make_rows(rng, forest, nf, cat, lin, n_random=250, wide=False):
    import check

    hug = check.random_rows(forest, 200, int(rng.integers(0, 10**6)))
    base = (rng.normal(size=(n_random, nf)) * rng.choice([1, 3, 20])).astype(np.float32)
    base[rng.random(base.shape) < 0.1] = np.nan
    base[rng.random(base.shape) < 0.05] = 0.0
    base[rng.random(base.shape) < 0.03] = 1e-36
    base[rng.random(base.shape) < 0.02] = np.float32(1e6) * rng.choice([-1, 1])
    if not lin:
        base[rng.random(base.shape) < 0.01] = np.float32(3.0e38) * rng.choice([-1, 1])
    values = [0, 1, 2, 3, 4, 7, 11, 12, -1, 2.5, np.nan]
    if wide:
        values += [31, 32, 33, 44, 45, 63, 64, 200]
    for c in cat:
        base[:, c] = rng.choice(np.array(values, np.float32), n_random)
    nan_row = np.full((1, nf), np.nan, np.float32)
    zero_row = np.zeros((1, nf), np.float32)
    lim = np.float32(1e6) if lin else np.float32(3.0e38)
    last = threshold_rows(forest, nf)
    return np.clip(np.concatenate([hug, base, nan_row, zero_row, last]), -lim, lim).astype(np.float32)


# --------------------------------------------------------------------------------------------- checking one model set
def written_files_check(chk, fmt, files, d: Path, xt, meta) -> None:
    """What QuickForest writes is what a user reads: the pruned copy of every model file is reloaded by its OWN library and predicted through
    the DEFAULT path a user calls (xgboost: `Booster.predict` without an iteration range, and the scikit-learn wrapper's `load_model` +
    `predict` when the model came from one; LightGBM: `Booster(model_file).predict`), then compared with the compiled pruned forest."""
    import compile
    import evaluator
    import prune
    import qf_common as qf

    np.savez(d / "wf.npz", X=xt)
    for i, f in enumerate(files):
        out = d / "wf-pruned" / f.stem
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                prune.run(f, d / "wf.npz", out, "50%")
        except SystemExit as e:  # e.g. linear-leaf trees cannot be pruned, by design
            chk.notes.append(f"{f.name}: not pruned ({str(e.code)[:80]}), the written-file check is skipped")
            continue
        pruned = out / f.name
        art = d / "wf-art" / f"{f.stem}.flat"
        art.parent.mkdir(exist_ok=True)
        opts = {"f64_leaves": True} if fmt == "lightgbm" else {}
        compile.compile_model(pruned, art, backend="flat", **opts)
        forest = qf.load_forest([pruned])
        transform = forest.kind in ("sigmoid", "softmax", "exp")
        fl = evaluator.FlatForest(art)
        dt = "f64" if fmt == "lightgbm" else "f32"
        want = fl.predict(xt, 1, dtype=np.float64) if dt == "f64" else fl.predict(xt, 1)
        fl.close()
        _, lim = bar(fmt, meta, opts, "", transform, dt, False, forest_scale(forest))
        names = meta.get("names")
        if fmt == "xgboost":
            import pandas as pd
            import xgboost as xgb

            b = xgb.Booster()
            b.load_model(str(pruned))
            got = {"Booster.predict": b.predict(xgb.DMatrix(pd.DataFrame(xt, columns=names) if names else xt))}
            wrapper = (meta.get("wrappers") or [None] * len(files))[i]  # fold k is files[k]: the folder is read in natural order
            if wrapper:
                m = (xgb.XGBClassifier if wrapper == "classifier" else xgb.XGBRegressor)()
                m.load_model(str(pruned))
                frame = pd.DataFrame(xt, columns=names)
                w = m.predict_proba(frame) if wrapper == "classifier" else m.predict(frame)
                got["scikit-learn wrapper"] = w[:, 1] if w.ndim == 2 and w.shape[1] == 2 and want.ndim == 1 else w
        else:
            import lightgbm as lgb

            got = {"Booster(model_file).predict": lgb.Booster(model_file=str(pruned)).predict(xt)}
        for how, answer in got.items():
            chk.compare("written", f"pruned file via {how}", fmt, f.name, want, answer, lim)


def margin_slack(forest, xt, bm, use_bm):
    """Per-row allowance for LightGBM's float64 answers when a margin is added to the leaves (base margin) or the models are averaged: the
    library and the evaluator add the same terms in another order, so they agree only to a few ulp of the LARGEST PARTIAL SUM, which on
    a row with a 1e6 input times a linear coefficient is far above the answer (case 28: terms of 5e4 cancelling to a margin of 25 give an
    exp() 1.2e-12 away). 8 x 2^-52 x S, S = the sum of |every tree's value| (+ |base| + |base margin|) of a row, a bound on any partial
    sum, averaged over the models. A single model without a base margin gets no allowance: there the claim is exact (0.0)."""
    import qf_common as qf

    total = np.zeros(len(xt))
    for m in forest.models:
        s = np.abs(m.base.astype(np.float64)).max() + np.zeros(len(xt))
        if use_bm:
            s = s + np.abs(bm).max(axis=1)
        for t in m.trees:
            s = s + np.abs(qf.tree_values(t, xt).astype(np.float64))
        total += s
    return 8 * 2.0**-52 * total / len(forest.models)


def fusing_machine() -> bool:
    return platform.machine().lower() in FUSING_MACHINES


def term_scale(forest, xt):
    """Per row, the magnitude of the terms that make up a LightGBM answer: |base| plus, for every tree, |value| of the leaf the row reaches
    (a linear leaf: |constant| + the sum of |coefficient x input|, or |fallback| when an input is NaN), averaged over the models. The
    size of the largest partial sum, where a fused multiply-add rounds once and a separate multiply and add round twice."""
    import qf_common as qf

    x = np.ascontiguousarray(xt, np.float32)
    total = np.zeros(len(x))
    for m in forest.models:
        s = np.abs(m.base.astype(np.float64)).max() + np.zeros(len(x))
        for t in m.trees:
            pos = qf.leaf_positions(t, x)
            term = np.abs(t.value[pos].astype(np.float64))
            for node, (const, fallback, feats, coefs) in (t.linear or {}).items():
                sel = np.flatnonzero(pos == node)
                if len(sel):
                    xs = x[sel][:, feats].astype(np.float64)
                    with np.errstate(all="ignore"):
                        mag = abs(const) + np.abs(np.asarray(coefs, np.float64) * xs).sum(axis=1)
                    term[sel] = np.where(np.isnan(xs).any(axis=1), abs(fallback), mag)
            s = s + term
        total += s
    return total / len(forest.models)


def fused_slack(forest, xt, ref):
    """aarch64 / arm64 only (None elsewhere): the per-row allowance, in the units of `rel_err`, for LightGBM linear leaves. The library's
    wheel fuses `const + coef * x`, the evaluator rounds twice; they differ by about one rounding of the largest partial sum, which is
    the magnitude of the terms, not of the answer (inputs near 1e6 with an answer near zero). |got - ref| <= FUSED_REL x (|ref| + P), P =
    `term_scale`, divided by max(1, |ref|) as the error is; the smallest over a row's output groups; 0 where the reference is not finite."""
    if not fusing_machine():
        return None
    r = np.abs(np.asarray(ref, np.float64).reshape(len(xt), -1))
    with np.errstate(all="ignore"):
        allow = FUSED_REL * (r + term_scale(forest, xt)[:, None]) / np.maximum(1.0, r)
    return np.where(np.isfinite(allow), allow, 0.0).min(axis=1)


class Checker:
    def __init__(self, case: int, desc: str, work: Path, stats: dict, codegen: str, verbose=False):
        self.case, self.desc, self.work, self.stats = case, desc, work, stats
        self.codegen = codegen  # "none" | "block" | "pgo" (pgo builds block too)
        self.failures: list[str] = []
        self.notes: list[str] = []
        self.checks = 0
        self.verbose = verbose
        self.engine_ran = {1: 0, 2: 0}
        self.codegen_built = 0
        self.inject = bool(os.environ.get("QF_DIFF_INJECT"))  # --selftest: corrupt every compiled answer, the run must then FAIL

    def stat(self, key):
        return self.stats.setdefault(key, new_stat())

    def fail(self, check, backend, opts, got, limit):
        self.failures.append(f"FAIL case={self.case} [{self.desc}] check={check!r} backend={backend} {opts} got={got} limit={limit}")

    def compare(self, key, label, backend, opts, ref, got, limit, slack=None):
        """Record the observed error under `key`; fail when above `limit`."""
        self.checks += 1
        if self.inject:
            got = np.array(got, np.float64)
            got.flat[0] += 3e-6 * max(1.0, abs(got.flat[0]))
        e = rel_err(ref, got)
        s = self.stat(key)
        s["n"] += 1
        s["max"] = max(s["max"], e)
        s["over_1e-6"] += int(e > 1e-6)
        if self.verbose:
            print(f"  {key:12s} {label:32s} {backend:16s} err={e:.3e} limit={limit:.3e}  {opts}")
        if slack is not None:  # per-row allowance: the conditioning of a margin sum (see `margin_slack`), on top of the table's limit
            r = np.asarray(ref, np.float64).reshape(len(ref), -1)
            g = np.asarray(got, np.float64).reshape(len(ref), -1)
            with np.errstate(all="ignore"):
                d = np.where((r == g) | (np.isnan(r) & np.isnan(g)), 0.0, np.abs(g - r) / np.maximum(1.0, np.abs(r)))
            ok = bool(np.all((d <= limit + np.asarray(slack, np.float64)[:, None]) | saturated(r, g)))
        else:
            ok = e <= limit
        if not ok:
            s["fail"] += 1
            self.fail(label, backend, opts, f"{e:.3e}", f"{limit:.3e}")

    def identical(self, key, label, backend, opts, got, want):
        """Bit-for-bit equality of each pair; `key` is the table row the count goes under."""
        self.checks += 1
        s = self.stat(key)
        s["n"] += 1
        if self.inject:
            got = [
                np.nextafter(np.asarray(g, np.float32), np.float32(np.inf)) if np.asarray(g).dtype == np.float32 else g + 1e-12 for g in got
            ]
        ok = all(same_bits(g, w) for g, w in zip(got, want, strict=True))
        if self.verbose:
            print(f"  {key:12s} {label:32s} {backend:16s} {'identical' if ok else 'DIFFERENT'}  {opts}")
        if not ok:
            s["fail"] += 1
            e = max(rel_err(w, g) for g, w in zip(got, want, strict=True))
            s["max"] = max(s["max"], e)
            self.fail(label, backend, opts, f"{e:.3e}", "0")

    def refusal(self, holds, name, opts, what, clean=True):
        """A wavefront engine met a file: it must answer if the file is one it holds, refuse with EngineRefused if not."""
        self.checks += 1
        s = self.stat("refusal")
        s["n"] += 1
        if holds:
            s["fail"] += 1
            self.fail("an engine refused a file it holds", name, opts, what, "answers")
        elif not clean or what == "answered":
            s["fail"] += 1
            self.fail("an engine met a file it cannot hold", name, opts, what, "EngineRefused")


def bar(fmt, meta, opts, kind, transform, dtype, bm, scale, avg="output"):
    """(table key, limit) for a compiled answer against the library. See TABLE."""
    B, N, init = scale
    f32 = dtype == "f32"
    folds = meta.get("folds", 1)
    if fmt == "xgboost":
        if bm:
            return "xgb+bm", max(1e-6, max(N, 1) * ULP32 * (B + init + BM_MAX) * 1.01)
        return "xgb", 0.0
    if fmt == "lightgbm":
        key = "lgb-lin" if meta.get("lin") else "lgb-f64l" if opts.get("f64_leaves") else "lgb"
        key += "-tr" if transform else ""
        if f32:
            return key, 6.1e-8
        return key, 1e-12 if transform or bm or meta.get("rf") else 0.0  # the aarch64 allowance is per row: `fused_slack`
    if opts.get("float32_sums"):
        if kind != "regression" or transform:
            return "ydf-f32s-cls", YDF_F32S_CLS_RF if meta.get("rf") else YDF_F32S_CLS_GBT
        return "ydf-f32s-reg", (0.0 if folds == 1 and (not f32 or not bm) else 6.1e-8)
    c = max(N, 1) * ULP32 * (B + init) * 1.01  # sequential float32 sum of N terms: |err| <= (N-1) u sum|terms|
    return "ydf-dbl", c + 2e-7 + 1e-12


def engine_holds(art: Path) -> bool:
    """Whether the wavefront engines may run this file: its header flags (u32 at byte 48 from the double format on) hold only the late-margin bit."""
    import backend_flat

    b = art.read_bytes()[:64]
    fmt = struct.unpack_from("<I", b, 8)[0]
    flags = struct.unpack_from("<I", b, 48)[0] if fmt >= backend_flat.FORMAT_DOUBLE else 0
    return flags & ~8 == 0 and fmt <= backend_flat.FORMAT_LINEAR


def new_stat():
    return {"n": 0, "max": 0.0, "over_1e-6": 0, "fail": 0}


def run_case(case: int, seed0: int, work: Path, stats: dict, codegen: str, verbose=False, forced_fmt=None, on_trained=None):
    rng = np.random.default_rng([seed0, case])
    fmt = forced_fmt or ("xgboost", "lightgbm", "ydf")[case % 3]
    d = work / f"c{case}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    try:
        desc, fmt, kind, meta = MAKERS[fmt](rng, d, case)
    except ImportError as e:
        raise Skip(str(e)) from e
    if on_trained:
        on_trained()
    import backends
    import compile
    import evaluator
    import qf_common as qf
    import qf_formats

    files = qf_formats.model_files(d)
    chk = Checker(case, desc, d, stats, codegen, verbose=verbose)
    try:
        from backend_codegen import toolchain_bin

        cg_ok = (toolchain_bin() / "rustc").exists()
    except Exception:
        cg_ok = False
    if codegen != "none" and not cg_ok:
        chk.notes.append("no rustc: generated-code backends skipped")
    pgo_ok = False
    if cg_ok:
        try:
            from backend_codegen import llvm_profdata

            llvm_profdata()
            pgo_ok = True
        except Exception:  # a distro rustc without the llvm-tools component: codegen-block still runs
            pass
    if codegen == "pgo" and cg_ok and not pgo_ok:
        chk.notes.append("no llvm-profdata: codegen-pgo skipped")
    for avg in ("output", "margin") if len(files) > 1 else ("output",):
        variants = [{}]
        if fmt == "lightgbm":
            variants.append({"f64_leaves": True})
        if fmt == "ydf":
            variants.append({"float32_sums": True})
        for opts in variants:
            forest = qf.load_forest(files, avg, bool(opts.get("float32_sums")))
            xt = make_rows(rng, forest, meta["nf"], meta["cat"], bool(meta.get("lin")), wide=bool(meta.get("wide")))
            if case % 5 == 2:
                # past 2,048 rows, so that three threads each get a block of their own (a thread takes 1,024 rows at a time): below that
                # the check "3 threads equal 1 thread", with or without a base margin, compares one block with itself
                pad = np.random.default_rng([seed0, case, 1]).integers(0, len(xt), 2100 - len(xt) + 1)
                xt = np.concatenate([xt, xt[pad]])
            bm = rng.uniform(-BM_MAX, BM_MAX, (len(xt), forest.groups))
            bm[rng.random(bm.shape) < 0.5] = 0.0
            tag = f"avg={avg} opts={opts or '{}'}"
            transform = forest.kind in ("sigmoid", "softmax", "exp")
            scale = forest_scale(forest)
            oracle = files
            # the default file holds float32 leaves: the library on those leaves is exact
            if fmt == "lightgbm" and not opts.get("f64_leaves"):
                oracle = lgb_float32_leaf_files(files, d / "oracle32")
            # folds with different LightGBM sigmoid scales cannot take one base margin (the evaluator refuses it, `margin_scale` is NaN)
            margins = (False, True) if forest.margin_scale == forest.margin_scale else (False,)
            oracle_bm = bm * (lgb_oracle_folds_factor(files, forest) if oracle is not files else 1.0) if margins[-1] else bm
            refs = {u: qf.reference_predict(oracle, xt, avg, oracle_bm if u else None) for u in margins}
            if fmt == "xgboost" and avg == "margin" and len(files) > 1:
                refs[False] = xgb_margin_average_reference(files, xt, forest.kind)
            stem = f"a-{avg}-{'-'.join(opts) or 'def'}"
            ad = d / "art"
            ad.mkdir(exist_ok=True)
            art = ad / f"{stem}.flat"
            backends.get("flat").compile(forest, art, opts)
            fl = evaluator.FlatForest(art)
            walker = {}
            for use_bm in margins:
                kw = {"base_margin": bm} if use_bm else {}
                o32 = fl.predict(xt, 1, **kw)
                o64 = fl.predict(xt, 1, dtype=np.float64, **kw)
                walker[use_bm] = (o32, o64)
                chk.identical(
                    "threads",
                    "3 threads == 1 thread",
                    "flat",
                    f"{tag} bm={use_bm}",
                    (fl.predict(xt, 3, **kw), fl.predict(xt, 3, dtype=np.float64, **kw)),
                    (o32, o64),
                )
                chk.identical("round", "f32 answer == f64 answer rounded", "flat", f"{tag} bm={use_bm}", (o32,), (o64.astype(np.float32),))
                for dt, out in (("f32", o32), ("f64", o64)):
                    key, lim = bar(fmt, meta, opts, kind, transform, dt, use_bm, scale, avg)
                    slack = None
                    if dt == "f64" and fmt == "lightgbm" and (len(files) > 1 or use_bm):
                        slack = margin_slack(forest, xt, bm, use_bm)
                    if dt == "f64" and key.startswith("lgb-lin"):
                        fused = fused_slack(forest, xt, refs[use_bm])
                        if fused is not None:
                            slack = fused if slack is None else slack + fused
                    chk.compare(key, f"{dt} answer vs the library", "flat", f"{tag} bm={use_bm}", refs[use_bm], out, lim, slack)
            # the wavefront engines run on the same file when it can hold them, and refuse it cleanly when not
            for eng, name in ((1, "wave"), (2, "wave-avx2")):
                holds = engine_holds(art)
                fe = evaluator.FlatForest(art, engine=eng)
                for use_bm in margins:
                    kw = {"base_margin": bm} if use_bm else {}
                    try:
                        w = (fe.predict(xt, 1, **kw), fe.predict(xt, 3, dtype=np.float64, **kw))
                    except evaluator.EngineRefused:
                        chk.refusal(holds, name, f"{tag} bm={use_bm}", "EngineRefused")
                        continue
                    except Exception as e:
                        chk.refusal(holds, name, f"{tag} bm={use_bm}", f"{type(e).__name__}: {str(e)[:80]}", clean=False)
                        continue
                    chk.engine_ran[eng] += int(fe.engine_used == eng)
                    chk.identical(
                        "backend",
                        f"{name} bit-identical to the walker",
                        f"{name}(used {fe.engine_used})",
                        f"{tag} bm={use_bm}",
                        w,
                        walker[use_bm],
                    )
                fe.close()
            fl.close()
            # compile refuses the wave backends on what they cannot hold: a one-line ValueError, not a traceback or a bad file
            for name in ("wave", "wave-avx2"):
                if forest.requires:
                    chk.checks += 1
                    st = chk.stat("refusal")
                    st["n"] += 1
                    try:
                        compile.compile_model(
                            d if len(files) > 1 else files[0],
                            ad / f"{stem}-{name}.flat",
                            backend=name,
                            average=avg,
                            **({"float32_sums": True} if opts.get("float32_sums") else {}),
                        )
                        st["fail"] += 1
                        chk.fail("compile accepted a backend that cannot hold the model", name, tag, "compiled", "ValueError")
                    except ValueError as e:
                        if "\n" in str(e).strip():
                            st["fail"] += 1
                            chk.fail("refusal is not one line", name, tag, str(e)[:60], "one line")
                    except Exception as e:
                        st["fail"] += 1
                        chk.fail("compile raised something other than ValueError", name, tag, f"{type(e).__name__}", "ValueError")
            # generated code, on the share of cases the plan picks
            if codegen != "none" and cg_ok:
                for bk in ["codegen-block"] + (["codegen-pgo"] if codegen == "pgo" and pgo_ok else []):
                    so = ad / f"{stem}-{bk}.so"
                    o = {**opts, "jobs": 1}
                    if bk == "codegen-pgo":
                        # a profile-guided build refuses fewer rows than backend_codegen.PROFILE_MIN_ROWS: the same real rows, repeated
                        from backend_codegen import PROFILE_MIN_ROWS

                        prof = xt[:200]
                        np.save(ad / "pgo_rows.npy", np.tile(prof, (-(-int(1.25 * PROFILE_MIN_ROWS) // len(prof)), 1)))
                        o["pgo_rows"] = str(ad / "pgo_rows.npy")
                    try:
                        backends.get(bk).compile(forest, so, o)
                        cg = backends.get(bk).load(so)
                    except Exception as e:
                        chk.checks += 1
                        chk.fail("compile and load", bk, tag, f"{type(e).__name__}: {str(e)[:100]}", "builds")
                        continue
                    for use_bm in margins:
                        kw = {"base_margin": bm} if use_bm else {}
                        got = (cg.predict(xt, 1, **kw), cg.predict(xt, 1, dtype=np.float64, **kw))
                        chk.identical("backend", f"{bk} bit-identical to the walker", bk, f"{tag} bm={use_bm}", got, walker[use_bm])
                    with contextlib.suppress(Exception):
                        cg.close()
                    chk.codegen_built += 1
    if meta.get("provenance") is not None and fmt in ("xgboost", "lightgbm"):
        written_files_check(chk, fmt, files, d, xt, meta)
    return chk


# --------------------------------------------------------------------------------------------- driver
def plan_codegen(case: int, mode: str, codegen_every: int, pgo_every: int) -> str:
    """Which generated-code backends this case also builds: one case in `every`, the last of each run of `every` cases, moved back by 0, 1
    or 2 places in turn. The format of a case is `case % 3`, so a plain `case % every == every - 1` picked only YDF cases whenever `every` was
    a multiple of 3 (6, 12, 18, 60: all the defaults); the shift visits the three formats and keeps the share at one in `every`."""

    def due(every: int) -> bool:
        if not every:
            return False
        run, place = divmod(case, every)
        return place == every - 1 - min(run % 3, every - 1)

    if due(pgo_every):
        return "pgo"
    if due(codegen_every):
        return "block"
    return "none"


def worker(args) -> int:
    out = Path(args.out)
    work = Path(args.work) if args.work else out / "work" / f"w{args.worker_id}"
    work.mkdir(parents=True, exist_ok=True)
    state = out / f"worker-{args.worker_id}.jsonl"
    cases = [int(c) for c in args.case_list.split(",")] if args.case_list else []
    stats: dict = {}
    for case in cases:
        with state.open("a") as f:
            f.write(json.dumps({"start": case}) + "\n")
        t0 = time.time()
        rec = {"case": case}
        try:
            chk = run_case(
                case,
                args.seed0,
                work,
                stats,
                plan_codegen(case, args.mode, args.codegen_every, args.pgo_every),
                verbose=args.verbose,
                on_trained=lambda case=case: state.open("a").write(json.dumps({"trained": case, **stamp(args.seed0)}) + "\n"),
            )
            rec.update(ok=not chk.failures, checks=chk.checks, failures=chk.failures, desc=chk.desc, engines=chk.engine_ran)
            if chk.failures and args.keep_failures:
                dest = out / "failures" / f"case-{case}"
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(work / f"c{case}", dest, ignore=shutil.ignore_patterns("*-src"))
        except Skip as e:
            rec.update(ok=True, skipped=str(e)[:100], checks=0, failures=[])
        except Exception as e:
            rec.update(
                ok=False,
                checks=0,
                failures=[
                    f"FAIL case={case} EXCEPTION {type(e).__name__}: {str(e)[:200]} | {traceback.format_exc().splitlines()[-3][:150]}"
                ],
            )
        rec["seconds"] = round(time.time() - t0, 2)
        rec["stats"] = stats
        rec.update(stamp(args.seed0))
        shutil.rmtree(work / f"c{case}", ignore_errors=True)
        with state.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        stats = {}
    return 0


def records(out: Path, seed0: int):
    """The records of the workers in `out` that belong to this seed0 and this generator (an older run's lines are another run's models)."""
    for wf in sorted(out.glob("worker-*.jsonl")):
        for line in wf.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and {k: r.get(k) for k in stamp(seed0)} == stamp(seed0):
                yield r


def trained(out: Path, case: int, seed0: int) -> bool:
    """Whether a worker recorded that the model of this case was trained (so a later crash is in QuickForest, not in the training library)."""
    return any(r.get("trained") == case for r in records(out, seed0))


def collect(out: Path, seed0: int):
    done = {}
    for r in records(out, seed0):
        if "start" not in r and "trained" not in r:
            done[r["case"]] = r
    return done


def merge_stats(done: dict) -> dict:
    tot: dict = {}
    for r in done.values():
        for k, s in r.get("stats", {}).items():
            t = tot.setdefault(k, {"n": 0, "max": 0.0, "over_1e-6": 0, "fail": 0})
            t["n"] += s["n"]
            t["max"] = max(t["max"], s["max"])
            t["over_1e-6"] += s["over_1e-6"]
            t["fail"] += s["fail"]
    return tot


def summary(done: dict, args, seconds: float, crashes: list[int]) -> tuple[str, int]:
    tot = merge_stats(done)
    fails = [ln for r in done.values() for ln in r.get("failures", [])]
    by_fmt = {"xgboost": [0, 0], "lightgbm": [0, 0], "ydf": [0, 0]}
    for c, r in done.items():
        f = ("xgboost", "lightgbm", "ydf")[c % 3] if not args.format else args.format
        by_fmt[f][0] += 1
        by_fmt[f][1] += 0 if r.get("ok") else 1
    skipped = sum(1 for r in done.values() if r.get("skipped") and not r.get("library_crash"))
    reasons: dict[str, int] = {}
    for r in done.values():
        if r.get("skipped") and not r.get("library_crash"):
            reasons[r["skipped"]] = reasons.get(r["skipped"], 0) + 1
    lib_crash = sum(1 for r in done.values() if r.get("library_crash"))
    eng = {1: 0, 2: 0}
    for r in done.values():
        for k, v in (r.get("engines") or {}).items():
            eng[int(k)] += v
    checks = sum(r.get("checks", 0) for r in done.values())
    lines = [
        f"# Differential test, mode {args.mode}, seed0 {args.seed0}",
        "",
        f"cases finished: {len(done)}  (xgboost {by_fmt['xgboost'][0]}, LightGBM {by_fmt['lightgbm'][0]}, YDF {by_fmt['ydf'][0]}; skipped for a missing library: {skipped}; the training library itself crashed: {lib_crash})",
        *(["skipped, by reason: " + "; ".join(f"{n} x {why}" for why, n in sorted(reasons.items()))] if reasons else []),
        f"checks run: {checks}   failed checks: {len(fails)}   cases that crashed the process: {len(crashes)}   wall time: {seconds:.0f} s",
        f"wave engines that really ran (cases): scalar wave {eng[1]}, AVX2 gathers {eng[2]}",
        "",
        "## Observed error per table key (max over all rows and checks; 'over 1e-6' counts checks)",
        "",
        "| key | checks | max error | checks over 1e-6 | failed |",
        "|---|---|---|---|---|",
    ]
    for k, _w, *_ in TABLE:
        if k in tot:
            s = tot[k]
            lines.append(f"| {k} | {s['n']} | {s['max']:.3e} | {s['over_1e-6']} | {s['fail']} |")
    lines += ["", "## Tolerance table (limit per format; error = |compiled - library| / max(1, |library|))", "", table_markdown(), ""]
    if fails or crashes:
        lines.append("## Failures")
        lines += [f"- {ln}" for ln in fails[:200]]
        lines += [f"- CRASH case={c}: the worker process died (signal/exit) while running it" for c in crashes]
    else:
        lines.append("No failure.")
    return "\n".join(lines), (1 if fails or crashes else 0)


def selftest(args) -> int:
    """The harness must be able to fail: corrupt the answers on purpose and demand that every kind of check notices."""
    out = Path(args.out) / "selftest"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    env = {**os.environ, "QF_DIFF_INJECT": "1", "TREEFLAT_FORCE_GATHER": "1"}
    cmd = [
        sys.executable,
        "-s",
        str(Path(__file__).resolve()),
        "--mode",
        "quick",
        "--cases",
        "6",
        "--workers",
        "1",
        "--out",
        str(out),
        "--seed0",
        "1",
        "--codegen-every",
        "2",
        "--pgo-every",
        "0",
    ]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    text = r.stdout
    need = ["vs the library", "bit-identical to the walker", "3 threads == 1 thread", "f32 answer == f64 answer rounded"]
    missing = [n for n in need if n not in text]
    ok = r.returncode == 1 and not missing
    print(
        f"selftest: exit code {r.returncode} (1 expected), kinds of check that did not fail: {missing or 'none'} -> {'OK' if ok else 'THE HARNESS IS BLIND'}"
    )
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--cases", type=int, default=None, help="number of cases (default: quick 36, full 3000)")
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--work", default=None, help="scratch folder (default OUT/work); never /tmp unless you say so")
    ap.add_argument("--codegen-every", type=int, default=None, help="one case in N also builds codegen-block (0 = never)")
    ap.add_argument("--pgo-every", type=int, default=None, help="one case in N also builds codegen-pgo (0 = never)")
    ap.add_argument("--format", choices=["xgboost", "lightgbm", "ydf"], default=None, help="only this library (with --case)")
    ap.add_argument("--case", type=int, default=None, help="replay one case, verbose")
    ap.add_argument("--fresh", action="store_true", help="forget earlier results in --out")
    ap.add_argument("--keep-failures", action="store_true", default=True)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="corrupt every compared answer on purpose (6 cases): the run must fail on every kind of check",
    )
    ap.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--case-list", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.selftest:
        return selftest(args)
    # run the AVX2 engine even where Downfall mitigation makes it slower (this is a correctness test)
    os.environ.setdefault("TREEFLAT_FORCE_GATHER", "1")
    quick = args.mode == "quick"
    if args.cases is None:
        args.cases = 36 if quick else 3000
    if args.codegen_every is None:
        args.codegen_every = 6 if quick else 12
    if args.pgo_every is None:
        args.pgo_every = 18 if quick else 60
    out = Path(args.out)
    if args.worker_id is not None:
        out.mkdir(parents=True, exist_ok=True)
        return worker(args)
    if args.case is not None:
        args.verbose = True
        out.mkdir(parents=True, exist_ok=True)
        work = Path(args.work) if args.work else out / "work" / "replay"
        work.mkdir(parents=True, exist_ok=True)
        stats: dict = {}
        cg = plan_codegen(args.case, args.mode, args.codegen_every, args.pgo_every)
        print(f"replaying case {args.case} seed0 {args.seed0} codegen={cg}")
        chk = run_case(args.case, args.seed0, work, stats, cg, verbose=True, forced_fmt=args.format)
        print(f"{chk.desc}: {chk.checks} checks, {len(chk.failures)} failed")
        for ln in chk.failures:
            print(ln)
        return 1 if chk.failures else 0
    if args.fresh and out.exists():
        for p in out.glob("worker-*.jsonl"):
            p.unlink()
        (out / "crashes.txt").unlink(missing_ok=True)
        shutil.rmtree(out / "failures", ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = collect(out, args.seed0)
    todo = [c for c in range(args.cases) if c not in done]
    crashes: list[int] = []
    print(f"mode {args.mode}: {len(done)} cases already done, {len(todo)} to run, {args.workers} worker(s), seed0 {args.seed0}", flush=True)
    env = {**os.environ, "TREEFLAT_FORCE_GATHER": "1", "PYTHONHASHSEED": "0"}
    # the parent splits the cases round-robin; a worker that dies is restarted after the case it died on is recorded as a crash
    queues = [todo[i :: args.workers] for i in range(args.workers)]
    procs = {}
    wid_next = 0

    def spawn(queue):
        nonlocal wid_next
        wid = wid_next
        wid_next += 1
        cmd = [
            sys.executable,
            "-s",
            str(Path(__file__).resolve()),
            "--mode",
            args.mode,
            "--seed0",
            str(args.seed0),
            "--out",
            str(out),
            "--worker-id",
            str(wid),
            "--case-list",
            ",".join(map(str, queue)),
            "--codegen-every",
            str(args.codegen_every),
            "--pgo-every",
            str(args.pgo_every),
        ]
        if args.work:
            cmd += ["--work", str(Path(args.work) / f"w{wid}")]
        if args.format:
            cmd += ["--format", args.format]
        return subprocess.Popen(cmd, env=env, stdout=(out / f"worker-{wid}.log").open("w"), stderr=subprocess.STDOUT), wid

    for q in queues:
        if q:
            p, wid = spawn(q)
            procs[wid] = (p, q)
    last_report = time.time()
    while procs:
        time.sleep(2)
        for wid, (p, q) in list(procs.items()):
            if p.poll() is None:
                continue
            del procs[wid]
            d = collect(out, args.seed0)
            remaining = [c for c in q if c not in d]
            if remaining:  # the worker died: its first unfinished case is the culprit
                culprit = remaining[0]
                if not trained(
                    out, culprit, args.seed0
                ):  # the process died inside the library that TRAINS the model, before any QuickForest code ran
                    note = f"the training library crashed (worker exit {p.returncode}) before the model existed; not a QuickForest case"
                    print(f"SKIP case={culprit}: {note}; continuing with {len(remaining) - 1} more", flush=True)
                    with (out / "worker-skips.jsonl").open("a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "case": culprit,
                                    "ok": True,
                                    "skipped": note,
                                    "checks": 0,
                                    "failures": [],
                                    "library_crash": True,
                                    **stamp(args.seed0),
                                }
                            )
                            + "\n"
                        )
                else:
                    crashes.append(culprit)
                    print(f"CRASH case={culprit} (worker exit {p.returncode}); continuing with {len(remaining) - 1} more", flush=True)
                    with (out / "crashes.txt").open("a") as f:
                        f.write(f"case={culprit} seed0={args.seed0} exit={p.returncode} gen={GENERATOR}\n")
                rest = remaining[1:]
                if rest:
                    np2, wid2 = spawn(rest)
                    procs[wid2] = (np2, rest)
        if time.time() - last_report > 120:
            print(f"  ... {len(collect(out, args.seed0))}/{args.cases} cases, {time.time() - t0:.0f} s", flush=True)
            last_report = time.time()
    done = collect(out, args.seed0)
    if (out / "crashes.txt").exists():
        listed = [
            dict(kv.split("=", 1) for kv in ln.split()) for ln in (out / "crashes.txt").read_text().splitlines() if ln.startswith("case=")
        ]
        # this run's crashes only (its seed0 and generator), and not a case that a later try finished
        crashes = sorted({int(c["case"]) for c in listed if c.get("seed0") == str(args.seed0) and c.get("gen") == GENERATOR})
        crashes = [c for c in crashes if c < args.cases and c not in done]
    text, code = summary({c: r for c, r in done.items() if c < args.cases}, args, time.time() - t0, crashes)
    (out / "summary.md").write_text(text + "\n")
    (out / "failures.txt").write_text("\n".join(ln for r in done.values() for ln in r.get("failures", [])) + "\n")
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
