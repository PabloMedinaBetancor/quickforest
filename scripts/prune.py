#!/usr/bin/env python3
"""Prune the FOREST, not each tree: keep the few trees that, added one at a time, best preserve the full ensemble's DECISIONS.

    prune.py MODEL(file|dir) DATA.npz OUT_DIR [--keep 10%|30|auto [--keep-ladder 5%,10%,25%,50%] [--min-agreement 95]] [--decision argmax|threshold:T|rule.py[:func]] [--method greedy|orientation]

MODEL is an xgboost / LightGBM (text) / YDF (folder) model, or a directory of them (the folds of a k-fold ensemble: each fold is
pruned on its own). Regression models can use the regression bar: --regression log|raw (median 0.5 %, none above 2 %, error no worse).
DATA.npz holds X plus optional fold / group / slice / eligible arrays (see docs/REFERENCE.md). Writes the pruned models in the same
format and names to OUT_DIR, plus report.json (its `kept_indices` lists, per model file, the kept trees as indices into the original
model). Selection rows and held-out rows are always different rows. An xgboost model that the in-memory path would not fit in
--memory-mb (default 1500) is read in a stream instead (`select_stream.py`, REFERENCE.md): the same trees are kept, and the memory does not
grow with the number of trees.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from fractions import Fraction
from functools import partial
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qf_common as qf
import qf_formats
import select_stream


def select_greedy(L, spec, k, rule, mean_scale, p_full, d_full, w=None):
    """The k trees per group that `greedy_order` picks first, sorted."""
    return [np.array(sorted(x)) for x in greedy_order(L, spec, k, rule, mean_scale, p_full, d_full, w)]


def outputs_of(m, kind):
    """The outputs of margins `m` for a kind that treats every output on its own (sigmoid, exp, identity)."""
    if kind == "sigmoid":
        return 1 / (1 + np.exp(-m))
    if kind == "exp":
        return np.exp(np.clip(m, -80, 80))
    return m


def best_in_block(L, spec, g, s_new, sums_g, eo, den_o, rule, d_full, p_full, w, reg, var, ch):
    """The best candidate of the trees `ch` (group g) at this greedy step -> (tree, score)."""
    mc = spec.base[g] + s_new * (sums_g[None, :] + L[ch])
    if spec.kind == "softmax":
        ec = np.exp(np.clip(mc, -80, 80))
        p = eo[None] / (ec + den_o[None])[..., None]
        p[..., g] = ec / (ec + den_o[None])
    elif eo is None:  # one output
        p = outputs_of(mc, spec.kind)[..., None]
    else:  # several independent outputs (xgboost one tree per output): group g's candidates next to the other groups' current outputs
        q = outputs_of(mc, spec.kind)
        p = np.repeat(eo[None].astype(q.dtype, copy=False), len(ch), 0)
        p[..., g] = q
    mse = (((p - p_full[None]) * w[None]) ** 2).mean((1, 2))
    if reg:
        score = -mse / var
    else:
        score = (rule(p) == d_full[None]).sum(1) - 0.5 * np.minimum(mse, 1.0)
    i = int(np.argmax(score))
    return int(ch[i]), float(score[i])


def no_pick() -> ValueError:
    """Every candidate scored NaN (the outputs on the selection rows overflow float32 or are not finite): no tree can be called the best."""
    return ValueError(
        "greedy selection: the model's outputs on the selection rows are not finite or overflow float32 (a regression target of about 1e19 "
        "or more); scale the target down and retrain, or prune with the model's outputs in a smaller range"
    )


def greedy_order(L, spec, k, rule, mean_scale, p_full, d_full, w=None, threads=1):
    """Round-robin over the groups; each step adds the tree of that group whose addition most raises decision agreement with the
    full model on the selection rows (ties and plateaus broken by the squared error of the outputs). Every remaining candidate is
    scored at once from running sums: no step re-predicts. Returns, per group, the first k picks IN THE ORDER they were made: a step
    looks only at the picks before it, so the first j of them are exactly the selection `greedy_order(..., j, ...)` makes
    (`--keep auto` runs this once to its largest rung and reads every smaller rung off the front of the order).
    `L` is the (trees, rows) leaf matrix, or any object that hands out the rows of the trees asked for (`select_stream.LeafCodes`). The
    candidates are scored in blocks, on `threads` threads: the block a pick comes from, and so the pick, does not depend on how many."""
    g_n, n = spec.groups, L.shape[1]
    gidx = [np.where(spec.tree_info == g)[0] for g in range(g_n)]
    total = [len(ix) for ix in gidx]
    kept = [[] for _ in range(g_n)]
    sums = np.zeros((g_n, n), np.float32)
    remaining = [ix.copy() for ix in gidx]
    base = spec.base
    reg = spec.task == "regression"
    w = np.ones(n, np.float32) if w is None else w.astype(np.float32)  # regression: weights that turn errors into relative ones
    w = w.reshape(n, -1)  # (rows, 1) or (rows, groups): a model of several regression outputs weighs every column on its own
    var = float(((p_full * w) ** 2).mean() if reg and w.any() else p_full.var()) or 1.0
    for step in range(g_n * k):
        g = step % g_n
        s_new = total[g] / (len(kept[g]) + 1) if mean_scale else 1.0
        mo = np.empty((n, g_n), np.float32)
        for o in range(g_n):
            s_o = total[o] / len(kept[o]) if mean_scale and kept[o] else 1.0
            mo[:, o] = base[o] + s_o * sums[o]
        eo = den_o = None
        if spec.kind == "softmax":
            eo = np.exp(np.clip(mo, -80, 80))
            den_o = eo.sum(1) - eo[:, g]
        elif g_n > 1:  # independent outputs: the others as they stand now, for the candidate's column to sit next to
            eo = outputs_of(mo, spec.kind)
        cand = remaining[g]
        step_rows = candidate_block(n, g_n)
        score_one = partial(best_in_block, L, spec, g, s_new, sums[g], eo, den_o, rule, d_full, p_full, w, reg, var)
        best, best_score = None, -np.inf
        blocks = (cand[a : a + step_rows] for a in range(0, len(cand), step_rows))
        for tree, score in select_stream.ordered_map(score_one, blocks, threads):
            if score > best_score:
                best, best_score = tree, score
        if best is None:
            raise no_pick()
        remaining[g] = remaining[g][remaining[g] != best]
        kept[g].append(best)
        sums[g] += L[best]
    return kept


def select_orientation(L, spec, k, margin_full):
    """The k trees per group that `orientation_order` ranks first, sorted."""
    return [np.sort(x) for x in orientation_order(L, spec, k, margin_full)]


def orientation_order(L, spec, k, margin_full):
    """Keep, per group, the k trees whose output points most in the direction of the full group margin (cosine), best first. Cheap, and
    much weaker than greedy on the held-out decisions; use it only when greedy is too slow."""
    kept = []
    for g in range(spec.groups):
        ix = np.where(spec.tree_info == g)[0]
        y = margin_full[:, g] - margin_full[:, g].mean()
        block = len(ix) if isinstance(L, np.ndarray) else candidate_block(L.shape[1], 1)  # a streamed forest: a block of trees at a time
        cos = np.concatenate([orientation_cosine(L[ix[a : a + block]], y) for a in range(0, len(ix), block)])
        kept.append(ix[np.argsort(-cos)[:k]])
    return kept


def orientation_cosine(rows, y):
    x = rows.astype(np.float64)
    x -= x.mean(1, keepdims=True)
    return (x @ y) / (np.sqrt((x**2).sum(1)) * np.sqrt((y**2).sum()) + 1e-12)


def group_margin(lm, spec):
    """(rows, groups) float64 margin of the full forest on the selection rows: base + the leaves of each group added tree by tree. A
    streamed forest (`select_stream.LeafCodes`) has added them already, in the same order."""
    if not isinstance(lm, np.ndarray):
        return spec.base[None, :] + lm.margin
    if spec.vector_width:  # (trees, rows, W): every tree adds its whole leaf vector
        return spec.base[None, :] + lm.sum(0, dtype=np.float64)
    cols = []
    for g in range(spec.groups):
        mine = spec.tree_info == g
        cols.append(spec.base[g] + (lm if mine.all() else lm[mine]).sum(0, dtype=np.float64))  # no copy of the matrix for a single group
    return np.stack(cols, 1)


BLOCK_NUMBERS = 131_072  # numbers in the (candidates, rows, groups) array a greedy step scores at once


def candidate_block(n_rows: int, groups: int) -> int:
    """How many candidate trees a greedy step scores at once: its array stays near BLOCK_NUMBERS numbers, small enough for the processor's
    cache (measured on an i3 and an i7: 16 and 25 % faster than a quarter of a million, twice as fast as four million). Any block size picks
    the same trees."""
    return max(1, BLOCK_NUMBERS // (n_rows * groups))


def select_greedy_units(L, spec, unit_of, k, rule, p_full, d_full):
    """The k units that `greedy_units_order` picks first, as sorted forest trees per group."""
    return [np.sort(x) for x in greedy_units_order(L, spec, unit_of, k, rule, p_full, d_full)]


def greedy_units_order(L, spec, unit_of, k, rule, p_full, d_full):
    """Greedy selection when a model's unit (a YDF classification tree) holds one forest tree per output group and must be kept
    whole: each step adds the unit whose leaf vector most raises decision agreement; kept[g] are the forest trees of group g, in the order their units were picked."""
    n_u, g_n, n = int(unit_of.max()) + 1, spec.groups, L.shape[1]
    ids = np.empty((n_u, g_n), np.int64)
    ids[unit_of, spec.tree_info] = np.arange(len(unit_of))
    sums = np.zeros((n, g_n), np.float32)
    remaining, kept = np.arange(n_u), []
    step_rows = candidate_block(n, g_n)
    for step in range(k):
        s_new = n_u / (step + 1)
        best, best_score = None, -np.inf
        for a in range(0, len(remaining), step_rows):
            ch = remaining[a : a + step_rows]
            p = spec.base[None, None] + s_new * (sums[None] + L[ids[ch]].transpose(0, 2, 1))
            mse = ((p - p_full[None]) ** 2).mean((1, 2))
            score = (rule(p) == d_full[None]).sum(1) - 0.5 * np.minimum(mse, 1.0)
            i = int(np.argmax(score))
            if score[i] > best_score:
                best, best_score = int(ch[i]), float(score[i])
        if best is None:
            raise no_pick()
        remaining = remaining[remaining != best]
        kept.append(best)
        sums += L[ids[best]].T
    return [ids[kept, g] for g in range(g_n)]


def greedy_vector_order(L, spec, k, rule, mean_scale, p_full, d_full, w=None):
    """Greedy selection for a forest of VECTOR trees (`spec.vector_width` = W): a vector tree is one unit that adds its whole leaf row to
    all W output groups, so each step adds the tree whose addition most raises the decisions' agreement with the full model (a regression
    model: lowers the weighted squared error, per entry). `L` hands out the (trees, rows, W) float32 leaf rows: `select_stream.LeafCodes`
    (a small code per tree and row plus each tree's leaf table, as `vector_leaves` builds it) or a plain array; `p_full` (rows, W), `d_full`
    the full model's decisions `rule(p_full)` ((rows,) or (rows, K)), `w` (rows, W) the weights that turn regression errors into relative
    ones (None = ones). A mean-type model (random forest) is re-normalised by the number of kept trees. Returns, per group, the SAME
    list of tree numbers in the order they were picked, so `kept[g]` and `--keep N` keep their meaning: N vector trees."""
    (t_n, n), g_n = L.shape[:2], spec.groups
    reg = spec.task == "regression"
    w = np.ones((n, g_n), np.float32) if w is None else w.astype(np.float32).reshape(n, g_n)
    var = float(((p_full * w) ** 2).mean() if reg and w.any() else p_full.var()) or 1.0
    sums = np.zeros((n, g_n), np.float32)
    remaining, picked = np.arange(t_n), []
    step_rows = candidate_block(n, g_n)
    for step in range(min(k, t_n)):
        s_new = t_n / (step + 1) if mean_scale else 1.0
        best, best_score = None, -np.inf
        for a in range(0, len(remaining), step_rows):
            ch = remaining[a : a + step_rows]
            p = qf.transform(spec.base[None, None] + s_new * (sums[None] + L[ch]), spec.kind)
            mse = (((p - p_full[None]) * w[None]) ** 2).mean((1, 2))
            if reg:
                score = -mse / var
            else:
                score = (rule(p) == d_full[None]).reshape(len(ch), -1).sum(1) - 0.5 * np.minimum(mse, 1.0)
            i = int(np.argmax(score))
            if score[i] > best_score:
                best, best_score = int(ch[i]), float(score[i])
        if best is None:
            raise no_pick()
        remaining = remaining[remaining != best]
        picked.append(best)
        sums += L[best]
    return [np.array(picked, np.int64) for _ in range(g_n)]


def vector_leaves(src, rows) -> select_stream.LeafCodes:
    """The leaf provider of a vector forest as pruning reads it: the source's `leaf_codes(rows)` ((trees, rows) leaf ordinals) and
    `leaf_vectors()` (one (leaves, W) table per tree) held as `LeafCodes`, so the (trees, rows, W) array of floats never exists; `margin`
    is the float64 sum of the leaf rows, tree by tree in order. A source that only has `leaf_matrix` hands out the array itself."""
    if not hasattr(src, "leaf_codes"):
        return src.leaf_matrix(rows)
    codes, tables = src.leaf_codes(rows), src.leaf_vectors()
    off = np.zeros(len(tables) + 1, np.int64)
    off[1:] = np.cumsum([len(t) for t in tables])
    lv = np.concatenate(tables, 0).astype(np.float32, copy=False)
    margin = np.zeros((len(rows), lv.shape[1]), np.float64)
    for i in range(len(tables)):
        margin += lv[off[i] + codes[i].astype(np.int64)]
    return select_stream.LeafCodes(codes, lv, off[:-1], margin, None)


def all_trees(src) -> list[np.ndarray]:
    """Every tree of the model, per output group (a vector forest: the same list for every group, since a tree feeds them all)."""
    spec = src.spec
    if spec.vector_width:
        return [np.arange(len(spec.nodes)) for _ in range(spec.groups)]
    return [np.flatnonzero(spec.tree_info == g) for g in range(spec.groups)]


def kept_indices(src, kept) -> dict:
    """Which trees of the ORIGINAL model a pruned copy keeps, as 0-based indices into the library's own tree list (xgboost `trees`, the
    LightGBM `Tree=` blocks, the YDF trees): `per_group` holds one sorted list per output group, `flat` one sorted list of all of them,
    `original_trees` the length of the library's list. `kept[g]` are tree numbers of the neutral forest (`src.model_ir()`); they are
    mapped to the library's numbers through `src.unit_of` where one library tree is several forest trees (a YDF random-forest
    classification tree is one forest tree per class: its index then appears in every group's list)."""
    unit_of = getattr(src, "unit_of", None)
    n_library = len(src.spec.nodes) if unit_of is None else int(np.max(unit_of)) + 1
    per_group = []
    for g in kept:
        ix = np.asarray(g, dtype=np.int64)
        per_group.append(sorted({int(i) for i in (ix if unit_of is None else np.asarray(unit_of)[ix])}))
    return {"original_trees": n_library, "per_group": per_group, "flat": sorted(set().union(*per_group))}


def library_counts(src, keep_ix) -> dict:
    """Trees and nodes of the model as its library counts them, before and after pruning. `keep_ix` are forest trees (the same unit once
    per group it feeds). Where one library tree is several forest trees (`src.unit_of`: a YDF random-forest classification tree is one
    forest tree per class, all with the same nodes), the library's tree is counted once and its nodes once."""
    nodes, unit_of = src.spec.nodes, getattr(src, "unit_of", None)
    if unit_of is None:
        return {"trees": len(nodes), "kept_trees": len(keep_ix), "nodes_before": int(nodes.sum()), "nodes_after": int(nodes[keep_ix].sum())}
    unit_of = np.asarray(unit_of)
    units, first = np.unique(unit_of, return_index=True)  # the first forest tree of each unit stands for all of them
    kept = np.unique(unit_of[keep_ix])
    return {
        "trees": len(units),
        "kept_trees": len(kept),
        "nodes_before": int(nodes[first].sum()),
        "nodes_after": int(nodes[first[np.searchsorted(units, kept)]].sum()),
    }


def meets_agreement(held: dict, bar: float) -> bool:
    """Whether the held-out agreement reaches the bar, judged on the row counts: the reported `pct` is rounded to 3 decimals."""
    # the bar as the decimal the user typed (repr of the float), compared exactly: `bar * rows` in floating point can land one ulp above an exact tie
    return held["rows"] > 0 and 100 * held["same"] >= Fraction(repr(float(bar))) * held["rows"]


def agreement(rule, p_a, p_b):
    """How often two answers decide alike. With one decision per row that is the figure of 1.1. With K > 1 outputs (decisions
    (rows, K)) `same`/`pct` are the WORST output's (every output must reach the bar), `same_per_output` / `agreement_pct_per_output`
    list all of them, and `all_outputs_pct` counts the rows where every output agrees; a disagreement is a row where any output differs."""
    da, db = rule(p_a), rule(p_b)
    same = da == db
    if same.ndim == 1:
        out = {"rows": len(same), "same": int(same.sum()), "pct": round(100 * float(same.mean()), 3) if len(same) else None}
        out["mean_abs_output_diff"] = round(float(np.abs(p_a - p_b).mean()), 6) if len(same) else None
        return out, np.where(~same)[0], da, db
    n = len(same)
    per = same.sum(0).astype(np.int64)
    worst = int(per.min()) if len(per) else 0
    out = {"rows": n, "same": worst, "pct": round(100 * worst / n, 3) if n else None}
    out["same_per_output"] = [int(v) for v in per]
    out["agreement_pct_per_output"] = [round(100 * int(v) / n, 3) if n else None for v in per]
    out["all_outputs_pct"] = round(100 * float(same.all(1).mean()), 3) if n else None
    out["mean_abs_output_diff"] = round(float(np.abs(p_a - p_b).mean()), 6) if n else None
    return out, np.where(~same.all(1))[0], da, db


def pooled_pct(figures: list[dict]) -> float:
    """One agreement percent for the rows of several `agreement` figures together (the models of a folder). With K > 1 outputs it is the worst
    output's over ALL the rows: the sum of each model's own worst output is not that (model A can be worst on output 0, model B on output 1,
    and no output is worst everywhere), it understates the pooled figure."""
    rows = sum(a["rows"] for a in figures)
    if all("same_per_output" in a for a in figures) and figures:
        same = min(sum(a["same_per_output"][k] for a in figures) for k in range(len(figures[0]["same_per_output"])))
    else:
        same = sum(a["same"] for a in figures)
    return round(100 * same / max(1, rows), 3)


def parse_keep(text: str, per_group: int) -> int:
    """ "10%" is a share of the trees of each group; a plain number is a count per group. At least one tree is always kept, a
    request for more than a group holds keeps them all, and zero, a negative, a count below 1 (`0.1` is not 10 %) or a non-number
    is refused (never silently one tree)."""
    t = str(text).strip()
    try:
        v = float(t[:-1]) / 100 * per_group if t.endswith("%") else float(t)
    except ValueError:
        raise SystemExit(f"--keep {text!r}: give a share like 10% or a count like 30") from None
    if not v > 0:
        raise SystemExit(f"--keep {text!r}: keep at least one tree (a share like 10% or a count like 30)")
    if not t.endswith("%") and v < 1:  # 0.1 is meant as 10 %, and would otherwise keep ONE tree per group without a word
        raise SystemExit(f"--keep {text!r}: a count is at least 1 tree per group; a share needs the percent sign (10%)")
    if not math.isfinite(v):  # "inf%": more than a group holds
        return per_group
    # the rule is "nearest whole tree, a half to the even one" on the EXACT product: 7 % of 150 is 10.5 -> 10, where the float product
    # 0.07 * 150 = 10.500000000000002 rounded to 11
    exact = _exact_number(t[:-1]) / 100 * per_group if t.endswith("%") else _exact_number(t)
    return min(per_group, max(1, round(min(exact, per_group))))


def _exact_number(text: str) -> Fraction:
    """The number a `--keep` text writes, as an exact fraction (`float` has already accepted the text; a spelling Fraction does not read, such
    as `1_0` on an older Python, falls back to the float's own value)."""
    try:
        return Fraction(text.strip())
    except ValueError:
        return Fraction(float(text))


def keeps_all(text: str) -> bool:
    """True for a SHARE of 100 % or more (a plain 100 is a count of trees, not a share)."""
    t = str(text).strip()
    try:
        return t.endswith("%") and float(t[:-1]) >= 100
    except ValueError:
        return False


def is_auto(text) -> bool:
    """True for `--keep auto`: the smallest rung of a ladder of shares that passes the bar of the run."""
    return str(text).strip().lower() == "auto"


AUTO_LADDER = ("5%", "10%", "25%", "50%")


def check_keep(keep, keep_ladder=None, min_agreement=95.0):
    """Refuses, before any file is read, a `--keep-ladder` without `--keep auto`, a ladder entry that is not a share below 100 % or a
    count of at least 1, and a repeated entry. Returns the ladder (`AUTO_LADDER` by default) for `--keep auto`, else None.
    `keep_ladder` is a list or a comma-separated text: `5%,10%,25%,50%` or counts such as `10,50,200`."""
    if not is_auto(keep):
        if keep_ladder is not None:
            raise SystemExit("--keep-ladder only goes with --keep auto")
        return None
    if not math.isfinite(min_agreement):
        raise SystemExit(f"--min-agreement {min_agreement}: give a finite percent")
    if keep_ladder is None:
        return AUTO_LADDER
    entries = keep_ladder.split(",") if isinstance(keep_ladder, str) else list(keep_ladder)
    ladder = tuple(str(e).strip() for e in entries)
    if not ladder or any(not e for e in ladder):
        raise SystemExit(f"--keep-ladder {keep_ladder!r}: give shares or counts separated by commas, like 5%,10%,25%,50%")
    for e in ladder:
        if keeps_all(e):  # the whole forest is what auto falls back to when no rung passes
            raise SystemExit(f"--keep-ladder {e}: 100% or more is the whole forest, which `--keep auto` already falls back to")
        try:
            finite = math.isfinite(float(e[:-1] if e.endswith("%") else e))
        except ValueError:
            finite = False
        if not finite:
            raise SystemExit(f"--keep-ladder {e!r}: give a share like 10% or a count like 30")
        parse_keep(e, 1000)  # a zero, a negative or a count below 1 is refused as in --keep
    if len(set(ladder)) != len(ladder):
        raise SystemExit(f"--keep-ladder {keep_ladder!r}: an entry is repeated")
    return ladder


def auto_report(rungs, chosen, whole, ladder, min_agreement, regression) -> dict:
    """The `auto` block of report.json: every rung tried (smallest first) with its trees, nodes and held-out figures, and the one chosen."""

    def row(r):
        t = [x["model"] for x in r["picks"]]
        h = r["heldout"]
        out = {
            "keep": r["labels"][0],
            "also": r["labels"][1:],  # other ladder entries that round to the same trees
            "kept_per_group": [m["kept_per_group"] for m in t],
            "kept_trees": sum(m["kept_trees"] for m in t),
            "nodes": sum(m["nodes_after"] for m in t),
            "heldout_rows": h["rows"],
            "heldout_agreement_pct": h["pct"],
            "pass": r.get("passes", True),
        }
        if "agreement_pct_per_output" in h:
            out["agreement_pct_per_output"] = h["agreement_pct_per_output"]
        if r["regression"] is not None:
            g = r["regression"]
            out["regression"] = {
                k: g[k] for k in ("median_pct", "p99_pct", "max_pct", "mae_full", "mae_pruned", "error_checked", "ok") if k in g
            }
            out["regression"]["failures"] = g["failures"]
            if "per_output" in g:
                out["regression"]["per_output"] = g["per_output"]
        return out

    bar = regression.describe() if regression else {"min_agreement_pct": min_agreement}
    return {
        "ladder": list(ladder),
        "bar": bar,
        "rungs": [row(r) for r in rungs],
        "chosen": {
            "keep": "100% (whole forest)" if whole else chosen["labels"][0],
            "whole_forest": whole,
            "reason": "no rung of the ladder passed the bar: the whole forest is kept, exact (as --keep 100%)"
            if whole
            else "the smallest rung that passed the bar",
            **{k: v for k, v in row(chosen).items() if k not in ("keep", "also", "pass")},
        },
    }


def auto_lines(r: dict) -> list[str]:
    """The rungs of an `auto` report as text lines (for the console and report.md)."""
    a = r.get("auto")
    if not a:
        return []
    lines = ["keep auto: rungs tried, smallest first (the bar: " + json.dumps(a["bar"]) + ")"]
    for g in a["rungs"]:
        if "regression" in g:
            q = g["regression"]
            figure = f"median {q['median_pct']} %, max {q['max_pct']} %"
            failing = [f"output {o['output']}" for o in q.get("per_output", []) if not o["ok"]]  # K > 1: say which
        else:
            figure = f"held-out agreement {g['heldout_agreement_pct']} % on {g['heldout_rows']:,} rows"
            per = g.get("agreement_pct_per_output")
            failing = [f"output {j}" for j, v in enumerate(per or []) if v is None or v < a["bar"].get("min_agreement_pct", 0)]
            if per:
                figure += f", per output {per}"
                if not g["pass"] and not failing:  # the rounded figure of the worst output sits on the bar: name it anyway
                    failing = [f"output {j}" for j, v in enumerate(per) if v == min(per)]
        if failing:
            figure += f" (below the bar: {', '.join(failing)})"
        lines.append(
            f"  {g['keep']:>5}: {g['kept_per_group']} per group, {g['kept_trees']:,} trees, {g['nodes']:,} nodes, {figure}: "
            + ("PASS" if g["pass"] else "fail")
        )
    c = a["chosen"]
    lines.append(f"  chosen: {c['keep']} ({c['reason']})")
    return lines


def is_model_path(model) -> bool:
    """A path (str / PathLike) names a model file or folder; anything else is an estimator object handed over from Python."""
    return isinstance(model, (str, bytes, os.PathLike))


SKLEARN_TYPES = ("DecisionTreeRegressor", "DecisionTreeClassifier", "RandomForestRegressor", "RandomForestClassifier")
SKLEARN_TYPES += ("ExtraTreesRegressor", "ExtraTreesClassifier")


def export_estimator(est, folder: Path, data_path) -> tuple[Path, dict]:
    """A fitted sklearn estimator -> `folder/model.qfsk` (verified against the estimator by `qf_sklearn.export_sklearn`, which refuses to
    write on any difference) -> (the path, the export's result: `verified_rows`, `max_diff`, `normalised`). One-line `UsageError` for an
    object that is not a supported fitted estimator, or when sklearn cannot be imported."""
    kind = type(est).__name__
    if not type(est).__module__.startswith("sklearn"):
        raise qf.UsageError(f"model: {type(est).__module__}.{kind} is neither a model path nor a fitted sklearn estimator")
    if kind not in SKLEARN_TYPES:
        raise qf.UsageError(
            f"sklearn {kind}: not supported; DecisionTree/RandomForest/ExtraTrees Regressor/Classifier only (see LIMITS for what is refused)"
        )
    try:
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted
    except ImportError:
        raise qf.UsageError(
            "model: a sklearn estimator was given but sklearn cannot be imported; install scikit-learn 1.3 or newer"
        ) from None
    try:
        check_is_fitted(est)
    except NotFittedError:
        raise qf.UsageError(f"sklearn {kind} is not fitted: call fit first") from None
    import qf_sklearn

    folder.mkdir(parents=True, exist_ok=True)
    rows = None if data_path is None else qf.load_data_cli(data_path)["X"][:20_000]  # real rows next to the probes the export builds
    path = folder / "model.qfsk"
    return path, dict(qf_sklearn.export_sklearn(est, path, X=rows))


def check_truth(y, n_out: int, regression, task: str, name: str) -> None:
    """The truth `y` must have one column per output of the model: shape (rows,) or (rows, 1) for one output, (rows, K) for K. Refused
    in one line otherwise, when it is used. A K-target regression model without the regression bar does not read it, so only there a
    different width goes unchecked."""
    if y is None or (task == "regression" and not regression and n_out > 1):
        return
    try:
        qf.require_truth_columns(y, n_out, name)
    except ValueError as e:
        raise SystemExit(str(e)) from None


def open_model(path, n_rows, memory_mb, scratch, threads=None):
    """The model as a source: the library's own reader, or, when that would need more than `memory_mb` for this many selection rows (an
    xgboost file with hundreds of thousands of trees), a `select_stream.StreamSource` that never holds the forest."""
    if select_stream.wanted(path, n_rows, memory_mb):
        select_stream.refuse_vector_leaves(path)  # before the scratch folder exists: a refused model leaves nothing behind
        folder = tempfile.mkdtemp(prefix=".qf-scratch-", dir=scratch)
        with qf.refusing_damaged(path, "model", qf_formats.ModelFileError):
            return select_stream.StreamSource(
                path, scratch=Path(folder), memory_mb=memory_mb, threads=threads or select_stream.auto_threads()
            )
    return qf_formats.open_source(path)


def prune_threads(text: str) -> int:
    """argparse type for `--threads`: a whole number, or `all` for every core."""
    if text.strip().lower() == "all":
        return os.cpu_count() or 1
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a whole number or 'all'") from None


def check_options(
    *,
    sub=1.0,
    max_sel_rows=20000,
    eval_frac=0.5,
    tol=None,
    seed=15,
    memory_mb=select_stream.DEFAULT_MEMORY_MB,
    max_held_rows=None,
) -> None:
    """The numbers that steer the selection, refused before any file is read (`optimize` calls it too, though it prunes nothing when it
    keeps 100 %)."""
    for flag, ok in (
        ("--sub (a share of the selection rows, in (0, 1])", 0 < sub <= 1),
        ("--max-sel-rows (at least 1)", max_sel_rows >= 1),
        ("--eval-frac (a share, in (0, 1))", 0 < eval_frac < 1),
        ("--tol (a bin width above 0)", tol is None or tol > 0),
        ("--seed (not negative)", seed >= 0),
        ("--memory-mb (at least 1)", memory_mb >= 1),
        ("--max-held-rows (at least 1)", max_held_rows is None or max_held_rows >= 1),
    ):
        if not ok:  # a count that selects no row (--sub 0 or -1, --max-sel-rows 0) is an error, not one silent row
            raise SystemExit(f"{flag}: out of range")


REPORT_FILES = ("report.json", "heldout_probs.npz")  # the files `run` writes into OUT_DIR besides the pruned models


def run(
    model,
    data_path,
    out_dir,
    keep="10%",
    *,
    decision="argmax",
    method="greedy",
    scale="auto",
    sub=1.0,
    seed=15,
    eval_frac=0.5,
    max_sel_rows=20000,
    tol=None,
    cases=50,
    save_probs=False,
    regression=None,
    average="output",
    min_agreement=95.0,
    keep_ladder=None,
    memory_mb=select_stream.DEFAULT_MEMORY_MB,
    scratch=None,
    max_held_rows=None,
    threads=None,
) -> dict:
    t_start = time.perf_counter()
    estimator = None if is_model_path(model) else model
    if estimator is not None:  # a fitted sklearn estimator (Python only): exported and verified first, then everything runs on the file
        model, exported = export_estimator(estimator, Path(out_dir) / "source", data_path)
    threads = None if threads is None else max(1, min(int(threads), 64))  # as everywhere: a count below 1 is one thread, and 64 at most
    check_options(
        sub=sub,
        max_sel_rows=max_sel_rows,
        eval_frac=eval_frac,
        tol=tol,
        seed=seed,
        memory_mb=memory_mb,
        max_held_rows=max_held_rows,
    )
    ladder = check_keep(keep, keep_ladder, min_agreement)  # None unless keep is "auto"
    auto = ladder is not None
    files = qf_formats.model_files(model)
    data = qf.load_data_cli(data_path)
    x = data["X"]
    sl = qf.slices(data, eval_frac, seed)
    fold = data.get("fold") if len(files) > 1 else None
    if fold is not None and not set(np.unique(fold[sl >= 0]).tolist()) <= set(range(len(files))):
        raise SystemExit("the fold array must hold values 0..n_models-1 (one fold file per model)")
    out = Path(out_dir)
    for f in files:
        if (
            f.name.casefold() in REPORT_FILES
        ):  # the pruned model is written under its own name next to the report (case-folded: NTFS, macOS)
            raise SystemExit(
                f"{f}: a model file called {f.name} would be overwritten by the report written next to it (OUT_DIR/{f.name}); rename it"
            )
        if os.path.realpath(out / f.name) == os.path.realpath(f):  # realpath: a link loop is not an error here
            raise SystemExit(f"{out} holds the model {f.name} itself: pruning would overwrite it; choose another OUT_DIR")
    if out.is_dir():  # the pruned folds of an earlier run with more models would be compiled together with these
        stale = sorted(q.name for q in out.iterdir() if q.name not in {f.name for f in files} and qf_formats.is_model_file(q))
        if stale:
            raise SystemExit(
                f"{out} already holds model files this run does not write ({', '.join(stale[:5])}): they would be compiled with the new "
                "ones as extra folds; choose an empty OUT_DIR or delete them"
            )
    if out.exists() and not out.is_dir():
        raise SystemExit(f"{out} is a file, not a folder: choose a new OUT_DIR")
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise SystemExit(f"{out}: cannot create the output folder ({e.strerror})") from None
    rng = np.random.default_rng(seed)
    report = {
        "params": {"keep": keep, "decision": decision, "method": method, "scale": scale, "sub": sub, "seed": seed},
        "models": [],
        "kept_indices": {},
    }
    if auto:
        report["params"].update(keep_ladder=list(ladder), min_agreement=min_agreement)
    if regression:
        report["params"]["regression"] = regression.describe()
    if estimator is not None:
        report["model"], report["library_diff"], report["export"] = str(files[0]), exported["max_diff"], exported
    labels = ladder if auto else (keep,)
    rung_root = out / "_rungs"  # --keep auto: every rung's pruned models, until the smallest passing one is chosen and moved up
    if auto and rung_root.exists():
        raise SystemExit(f"{rung_root} exists: --keep auto uses that name for its working folder; choose another OUT_DIR")
    margin_avg = average == "margin" and fold is None and len(files) > 1
    held_cap = max_held_rows or select_stream.HELD_ROWS
    shared_held = None  # without a fold array every model answers the SAME held-out rows (they are averaged row by row): drawn once per run
    cap_run = False  # a run with any streamed model caps the held-out rows of all its models alike
    if fold is None:
        n_sel = min(max_sel_rows, int((sl == 0).sum()))
        cap_run = any(select_stream.wanted(f, n_sel, memory_mb) for f in files)
    rule = None
    head_key = None  # what the first model of a folder looks like: the others must agree (they are averaged or pooled row by row)
    per_file = []  # per model file: the full model's held-out answers, and what each tried k per group gave
    try:
        for j, f in enumerate(files):
            t0 = time.perf_counter()
            in_fold = (fold == j) if fold is not None else np.ones(len(x), bool)
            src = open_model(f, min(max_sel_rows, int((in_fold & (sl == 0)).sum())), memory_mb, scratch or out, threads)
            streamed = isinstance(src, select_stream.StreamSource)
            spec = src.spec
            qf.check_width(x, spec.n_features, data_path)
            if getattr(src, "linear", False):
                raise SystemExit(
                    f"{f.name}: linear-leaf trees (linear_tree=true) cannot be pruned (a leaf is a function of the row, not a table entry); "
                    "QuickForest only compiles them: use compile, or optimize --keep 100% --compile"
                )
            parse_s = time.perf_counter() - t0
            mine = (fold == j) if fold is not None else np.ones(len(x), bool)
            s_rows, h_rows = np.where(mine & (sl == 0))[0], np.where(mine & (sl == 1))[0]
            if len(s_rows) == 0 or len(h_rows) == 0:
                raise SystemExit(f"model {f.name}: needs selection rows and held-out rows (got {len(s_rows)} and {len(h_rows)})")
            if regression and spec.task != "regression":
                raise SystemExit(f"{f.name}: the regression bar needs a regression model (this one is {spec.task})")
            vec = (
                spec.vector_width > 0
            )  # vector trees: one selection unit adds a whole leaf row (tree_info carries no group: never read it)
            segs = qf.decision_segments(spec.groups, spec.kind, spec.vector_width, getattr(src, "segments", None))
            n_out = qf.truth_outputs(spec.groups, spec.kind, spec.task, spec.vector_width, getattr(src, "segments", None))
            key = (
                spec.n_features,
                spec.groups,
                spec.kind,
                spec.task,
                bool(vec),
                n_out,
                None if segs is None else tuple(np.ravel(segs).tolist()),
            )
            if head_key not in (None, key):
                raise SystemExit(f"{f.name}: the models of the folder disagree on features, outputs, classes or objective")
            head_key = key
            check_truth(data.get("y"), n_out, regression, spec.task, f.name)
            if vec and method != "greedy":
                raise SystemExit(f"{f.name}: --method {method} is not available for multi-output (vector) trees; use greedy")
            pick = s_rows
            if sub < 1.0:
                pick = np.sort(rng.choice(pick, max(1, int(sub * len(pick))), replace=False))
            if len(pick) > max_sel_rows:
                pick = np.sort(rng.choice(pick, max_sel_rows, replace=False))
            if streamed:  # a forest too big for xgboost to load answers only the rows its walk covers
                s_rows = pick
            if fold is not None:  # each model has its own fold's rows, concatenated: the cap is per model and only a streamed one needs it
                if streamed and len(h_rows) > held_cap:
                    h_rows = np.sort(rng.choice(h_rows, held_cap, replace=False))
            elif cap_run and len(h_rows) > held_cap:
                if shared_held is None:
                    shared_held = np.sort(rng.choice(h_rows, held_cap, replace=False))
                h_rows = shared_held
            if src.fmt != "xgboost" and scale != "auto":
                raise SystemExit("--scale applies to xgboost only (LightGBM and YDF average their random forests themselves)")
            mean_scale = spec.rf if scale == "auto" else scale == "mean"
            units = not vec and src.unit_of is not None and spec.groups > 1 and len(set(src.unit_of.tolist())) < len(src.unit_of)
            per_group = len(spec.nodes) if vec else int(np.bincount(spec.tree_info, minlength=spec.groups).min())
            single = getattr(src, "single_tree", False) or str(getattr(src, "fmt", "")).startswith("sklearn")
            if per_group == 1 and single and not auto and not keeps_all(keep):
                raise SystemExit(f"{f.name}: a single decision tree cannot be pruned (it has no trees to leave out); use --keep 100%")
            ks = {lab: parse_keep(lab, per_group) for lab in labels}
            k_max = max(ks.values())
            t1 = time.perf_counter()
            if vec:
                lm = vector_leaves(src, x[pick])
            else:
                lm = src.leaf_matrix(x[pick], extra=x[h_rows]) if streamed else src.leaf_matrix(x[pick])
            t_pre = time.perf_counter() - t1
            margin = group_margin(lm, spec)
            p_full = qf.transform(margin, spec.kind)
            scale_info = None
            if streamed:
                scale_info = {
                    "mode": "stream",
                    "memory_mb": memory_mb,
                    "leaf_codes_bytes": lm.code_bytes,
                    "leaf_codes_on_disk": lm.on_disk,
                    "selection_rows": len(pick),
                    "heldout_rows_used": len(h_rows),
                    "scan_seconds": src.timings["scan"],
                }
                held_full = qf.transform(spec.base[None, :] + lm.extra_margin, spec.kind)
            rule = qf.make_rule(decision, spec.groups, spec.kind, p_full, tol, spec.task, segs)
            t2 = time.perf_counter()
            # one selection to the largest k: every smaller k is the front of the same order (auto reads all its rungs off it)
            if method == "greedy" and vec:
                w = None if not (regression and regression.space == "raw") else 1 / np.maximum(np.abs(p_full), 1e-9)
                order = greedy_vector_order(lm, spec, k_max, rule, mean_scale, p_full, rule(p_full), w)
            elif method == "greedy" and units:
                order = greedy_units_order(lm, spec, src.unit_of, k_max, rule, p_full, rule(p_full))
            elif method == "greedy":
                w = None if not (regression and regression.space == "raw") else 1 / np.maximum(np.abs(p_full), 1e-9)
                order = greedy_order(
                    lm,
                    spec,
                    k_max,
                    rule,
                    mean_scale,
                    p_full,
                    rule(p_full),
                    w,
                    threads=(threads or select_stream.auto_threads()) if streamed else (threads or 1),
                )
            elif units:
                raise SystemExit("--method orientation needs one tree per output group; use greedy for this model")
            else:
                order = orientation_order(lm, spec, k_max, margin)
            t_pick = time.perf_counter() - t2
            if streamed:
                lm.close()
            del lm
            # the honest figures come from the real pruned model, predicted by the model's own library
            pf_s, pf_h = (p_full, held_full) if streamed else (src.predict(x[s_rows]), src.predict(x[h_rows]))
            tries = {}
            for k in sorted(set(ks.values())):
                kept = [np.array(sorted(g[:k])) for g in order]
                dest = (rung_root / f"k{k}" if auto else out) / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.save_pruned(kept, mean_scale, dest)
                pruned = qf_formats.open_source(dest)
                keep_ix = kept[0] if vec else np.concatenate(kept)  # a vector tree feeds every group: counted once
                pp_s, pp_h = pruned.predict(x[s_rows]), pruned.predict(x[h_rows])
                a_sel, _, _, _ = agreement(rule, pf_s, pp_s)
                a_held, _, _, _ = agreement(rule, pf_h, pp_h)
                counts = library_counts(src, keep_ix)
                tries[k] = {
                    "pp_h": pp_h,
                    "mp_h": pruned.predict(x[h_rows], margin=True) if margin_avg else None,
                    "kept_indices": kept_indices(src, kept),
                    "model": {
                        "file": f.name,
                        "format": src.fmt,
                        "rf": spec.rf,
                        "trees": counts["trees"],
                        "kept_trees": counts["kept_trees"],
                        "kept_per_group": k,
                        "nodes_before": counts["nodes_before"],
                        "nodes_after": counts["nodes_after"],
                        "bytes_before": src.size_bytes,
                        "bytes_after": pruned.size_bytes,
                        "selection_rows": len(pick),
                        "selection_agreement": a_sel,
                        "heldout_agreement": a_held,
                        "seconds": {"parse": round(parse_s, 2), "leaf_matrix": round(t_pre, 2), "pick": round(t_pick, 2)},
                        **({"scale": scale_info} if scale_info else {}),
                        **({"notes": src.notes} if getattr(src, "notes", None) else {}),
                    },
                }
            if auto:  # the fallback of --keep auto, built from the full model itself: exact by construction
                a_s, a_h = agreement(rule, pf_s, pf_s)[0], agreement(rule, pf_h, pf_h)[0]
                every = all_trees(src)
                counts = library_counts(src, np.arange(len(spec.nodes)))
                tries["whole"] = {
                    "pp_h": pf_h,
                    "mp_h": src.predict(x[h_rows], margin=True) if margin_avg else None,
                    "kept_indices": kept_indices(src, every),
                    "model": {
                        "file": f.name,
                        "format": src.fmt,
                        "rf": spec.rf,
                        "trees": counts["trees"],
                        "kept_trees": counts["trees"],
                        "kept_per_group": per_group,
                        "nodes_before": counts["nodes_before"],
                        "nodes_after": counts["nodes_before"],
                        "bytes_before": src.size_bytes,
                        "bytes_after": src.size_bytes,
                        "selection_rows": len(pick),
                        "selection_agreement": a_s,
                        "heldout_agreement": a_h,
                        "seconds": {"parse": round(parse_s, 2), "leaf_matrix": round(t_pre, 2), "pick": round(t_pick, 2)},
                        **({"scale": scale_info} if scale_info else {}),
                        **({"notes": src.notes} if getattr(src, "notes", None) else {}),
                    },
                }
            per_file.append(
                {
                    "file": f,
                    "ks": ks,
                    "tries": tries,
                    "pf_h": pf_h,
                    "mf_h": src.predict(x[h_rows], margin=True) if margin_avg else None,
                    "h_rows": h_rows,
                }
            )
            if streamed:
                src.close()

        def pooled(picks) -> dict:
            """The ensemble's held-out figures for one choice of k per file (`picks`: one `tries` entry per file). Out-of-fold rows keep
            their own fold's answer; without folds the models are averaged."""
            if fold is not None:
                rows = np.concatenate([e["h_rows"] for e in per_file])
                pf, pp = np.concatenate([e["pf_h"] for e in per_file]), np.concatenate([t["pp_h"] for t in picks])
            else:
                rows = per_file[0]["h_rows"]
                if any(not np.array_equal(e["h_rows"], rows) for e in per_file):  # the models are averaged row by row
                    raise AssertionError("the models of a folder answered different held-out rows")
                pf, pp = np.mean([e["pf_h"] for e in per_file], 0), np.mean([t["pp_h"] for t in picks], 0)
                if margin_avg:  # the deployed ensemble answers the transform of the mean margin
                    pf = qf.transform(np.mean([e["mf_h"] for e in per_file], 0), spec.kind)
                    pp = qf.transform(np.mean([t["mp_h"] for t in picks], 0), spec.kind)
            ens, bad, df, dp = agreement(rule, pf, pp)
            ens["mode"] = "out-of-fold" if fold is not None else "mean of the models" + (" (margin)" if margin_avg else "")
            sel = [t["model"]["selection_agreement"] for t in picks]
            ens["selection_pct"] = pooled_pct(sel)
            # each model's own agreement on its held-out rows, pooled: the figure the selection one is compared with (several models
            # without a fold array are averaged into one ensemble figure, `pct`, which is another quantity)
            own = [t["model"]["heldout_agreement"] for t in picks]
            ens["models_pct"] = pooled_pct(own)
            y = data.get("y")
            if y is not None and not regression and spec.task != "regression" and df.shape == y[rows].shape:
                # the truth is given: how often the full and the pruned forest hit it (agreement with the full forest is not accuracy)
                axis = 0 if df.ndim == 2 else None  # several outputs: the figure is the WORST output's, all of them listed
                full_acc, pruned_acc = (df == y[rows]).mean(axis), (dp == y[rows]).mean(axis)
                ens["accuracy"] = {
                    "rows": len(rows),
                    "full_pct": round(100 * float(np.min(full_acc)), 3),
                    "pruned_pct": round(100 * float(np.min(pruned_acc)), 3),
                }
                if axis == 0:
                    ens["accuracy"]["full_pct_per_output"] = [round(100 * float(v), 3) for v in full_acc]
                    ens["accuracy"]["pruned_pct_per_output"] = [round(100 * float(v), 3) for v in pruned_acc]
            reg = None
            if regression:
                reg = regression.check(pf, pp, None if y is None else y[rows].astype(np.float64))
            return {"heldout": ens, "bad": bad, "df": df, "dp": dp, "rows": rows, "pf": pf, "pp": pp, "regression": reg, "picks": picks}

        # the rungs: one per distinct choice of k per file (two ladder entries that round to the same trees are one rung), smallest first
        rung_labels = {}
        for lab in labels:
            rung_labels.setdefault(tuple(e["ks"][lab] for e in per_file), []).append(lab)
        rungs = []
        for key in sorted(rung_labels, key=lambda t: (sum(t), t)):
            r = pooled([e["tries"][key[i]] for i, e in enumerate(per_file)])
            r.update(key=key, labels=rung_labels[key])
            r["passes"] = bool(r["regression"]["ok"]) if regression else meets_agreement(r["heldout"], min_agreement)
            rungs.append(r)
        if auto:
            chosen = next((r for r in rungs if r["passes"]), None)
            whole = chosen is None
            if whole:
                chosen = pooled([e["tries"]["whole"] for e in per_file])
                chosen["labels"], chosen["key"] = ["100%"], tuple(e["ks"][labels[0]] for e in per_file)
        else:
            chosen, whole = rungs[0], False
        if auto:  # the chosen rung's models take the place of the plain run's output
            for i, e in enumerate(per_file):
                dest, f = out / e["file"].name, e["file"]
                if dest.is_dir():
                    shutil.rmtree(dest)
                elif dest.exists():
                    dest.unlink()
                if whole:
                    shutil.copytree(f, dest) if f.is_dir() else shutil.copy2(f, dest)
                else:
                    shutil.move(str(rung_root / f"k{chosen['key'][i]}" / f.name), str(dest))
    finally:
        if auto:
            shutil.rmtree(rung_root, ignore_errors=True)
    picks = chosen["picks"]
    report["models"] = [t["model"] for t in picks]
    report["kept_indices"] = {t["model"]["file"]: t["kept_indices"] for t in picks}
    report["heldout"] = chosen["heldout"]
    rows, pf, pp, bad, df, dp = (chosen[k] for k in ("rows", "pf", "pp", "bad", "df", "dp"))
    if regression:
        report["regression"] = chosen["regression"]
    if regression and "per_output" in chosen["regression"]:
        report["per_output"] = chosen["regression"][
            "per_output"
        ]  # K > 1 outputs: one dict each; the top-level figures are the worst output's
    elif "agreement_pct_per_output" in chosen["heldout"]:
        report["per_output"] = [{"output": j, "agreement_pct": a} for j, a in enumerate(chosen["heldout"]["agreement_pct_per_output"])]
    if auto:
        report["auto"] = auto_report(rungs, chosen, whole, ladder, min_agreement, regression)
    report["disagreements"] = [
        {
            "row": int(rows[i]),
            "full": df[i].tolist(),
            "pruned": dp[i].tolist(),
            "p_full": np.round(pf[i], 4).tolist(),
            "p_pruned": np.round(pp[i], 4).tolist(),
        }
        for i in bad[:cases]
    ]
    nb = sum(m["nodes_before"] for m in report["models"])
    na = sum(m["nodes_after"] for m in report["models"])
    report["totals"] = {
        "nodes_before": nb,
        "nodes_after": na,
        "node_ratio": round(nb / max(na, 1), 2),
        "bytes_before": sum(m["bytes_before"] for m in report["models"]),
        "bytes_after": sum(m["bytes_after"] for m in report["models"]),
        "seconds": round(time.perf_counter() - t_start, 1),
    }
    if save_probs:
        npz = io.BytesIO()
        np.savez(npz, rows=rows, p_full=pf, p_pruned=pp)
        qf.replace_file(out / "heldout_probs.npz", data=npz.getvalue())
    qf.replace_file(out / "report.json", data=json.dumps(report, indent=1).encode())
    return report


def summary(r: dict) -> str:
    text = _summary(r)
    if r.get("auto"):
        c = r["auto"]["chosen"]
        text = f"keep auto -> {c['keep']}" + (": " + c["reason"] + "; " if c["whole_forest"] else ": ") + text
    streamed = [m["scale"] for m in r["models"] if m.get("scale")]
    if streamed:  # a model too big to load: say how its figures were made
        sc = streamed[0]
        where = "in a file" if sc["leaf_codes_on_disk"] else "in memory"
        text += (
            f" [read in a stream: {sc['selection_rows']:,} selection rows, {sc['heldout_rows_used']:,} held-out rows answered by the walk, "
            f"leaf codes {sc['leaf_codes_bytes'] / 1e6:,.0f} MB {where}]"
        )
    return text


def _summary(r: dict) -> str:
    h, t = r["heldout"], r["totals"]
    if "regression" in r:
        g = r["regression"]
        err = f"; MAE full {g['mae_full']} pruned {g['mae_pruned']}" if g["error_checked"] else "; truth y absent: error bar NOT checked"
        return (
            f"pruned {len(r['models'])} model(s): nodes {t['nodes_before']:,} -> {t['nodes_after']:,} ({t['node_ratio']}x), "
            f"bytes {t['bytes_before']:,} -> {t['bytes_after']:,}; held-out |pruned - full| on {g['rows']:,} rows: median "
            f"{g['median_pct']} %, "
            f"p99 {g['p99_pct']} %, max {g['max_pct']} %{err}{' (the worst of ' + str(len(g['per_output'])) + ' outputs)' if 'per_output' in g else ''}; bars {'PASS' if g['ok'] else 'FAIL (' + '; '.join(g['failures']) + ')'}; "
            f"{t['seconds']} s"
        )
    per = (
        f" (worst of {len(h['agreement_pct_per_output'])} outputs {h['agreement_pct_per_output']}, all agree {h['all_outputs_pct']} %)"
        if "agreement_pct_per_output" in h
        else ""
    )
    return (
        f"pruned {len(r['models'])} model(s): nodes {t['nodes_before']:,} -> {t['nodes_after']:,} ({t['node_ratio']}x), "
        f"bytes {t['bytes_before']:,} -> {t['bytes_after']:,}; held-out agreement {h['pct']} %{per} on {h['rows']:,} rows "
        f"({h['rows'] - h['same']} disagree; selection rows {h['selection_pct']} %, so overfit "
        f"{h['selection_pct'] - h.get('models_pct', h['pct']):+.2f} points); {t['seconds']} s"
    )


def main():
    qf.odd_names_are_printable()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument(
        "--keep",
        default="10%",
        help=(
            "a share of the trees of each class (10%%) or a count of trees per class (30), rounded to the nearest whole tree, at least 1; "
            "auto: the smallest rung of --keep-ladder that passes the bar"
        ),
    )
    ap.add_argument("--keep-ladder", help="--keep auto: the rungs to try, smallest first, comma separated (default 5%%,10%%,25%%,50%%)")
    ap.add_argument(
        "--min-agreement", type=qf.finite_float, default=95.0, help="--keep auto: held-out agreement bar, percent (not used by other keeps)"
    )
    ap.add_argument("--decision", default="argmax", help="argmax | threshold:T | rule.py[:func]")
    ap.add_argument("--method", choices=("greedy", "orientation"), default="greedy")
    ap.add_argument("--scale", choices=("auto", "mean", "none"), default="auto", help="mean: leaves x trees/kept (random forests)")
    ap.add_argument("--sub", type=float, default=1.0, help="fraction of the selection rows used (the seed varies it)")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--eval-frac", type=float, default=0.5, help="held-out share when the data has no slice or group")
    ap.add_argument("--max-sel-rows", type=int, default=20000)
    ap.add_argument(
        "--memory-mb",
        type=int,
        default=select_stream.DEFAULT_MEMORY_MB,
        help="memory the selection may use, MB: an xgboost model that the in-memory path would not fit in it is read in a stream (see LIMITS.md)",
    )
    ap.add_argument("--scratch", help="where a streamed run keeps its working files (default: inside OUT_DIR; deleted at the end)")
    ap.add_argument(
        "--threads",
        type=prune_threads,
        help="threads for the selection (and the walk of a streamed run): a number, or all; default 1, and 2 for a streamed run",
    )
    ap.add_argument(
        "--max-held-rows",
        type=int,
        help="at most this many held-out rows are answered by the full forest in a run that streams any model (default 20000); a folder without a fold array "
        "then judges every model on the same rows, so its in-memory models are cut to them too; a run with no streamed model uses all",
    )
    ap.add_argument("--tol", type=qf.finite_float, help="regression: agreement bin width (default 0.1 sd)")
    ap.add_argument("--cases", type=qf.count_arg(0), default=50, help="disagreement cases listed in the report")
    ap.add_argument(
        "--average", choices=("output", "margin"), default="output", help="how several models without a fold array combine (as compile)"
    )
    ap.add_argument("--save-probs", action="store_true", help="also write heldout_probs.npz (full and pruned outputs)")
    ap.add_argument(
        "--regression",
        choices=("log", "raw"),
        help="regression bar instead of class decisions: the model outputs log(target) or the target itself",
    )
    ap.add_argument("--med-tol", type=qf.finite_float, default=0.5, help="regression: median |pruned - full| bar, percent")
    ap.add_argument("--max-tol", type=qf.finite_float, default=2.0, help="regression: no row above this, percent")
    ap.add_argument(
        "--error-slack", type=qf.finite_float, default=0.0, help="regression: allowed relative worsening of the held-out MAE vs truth y"
    )
    a = ap.parse_args()
    reg = qf.RegBar(a.regression, a.med_tol, a.max_tol, a.error_slack) if a.regression else None
    try:
        r = run(
            a.model,
            a.data,
            a.out,
            a.keep,
            decision=a.decision,
            method=a.method,
            scale=a.scale,
            sub=a.sub,
            seed=a.seed,
            eval_frac=a.eval_frac,
            max_sel_rows=a.max_sel_rows,
            tol=a.tol,
            cases=a.cases,
            save_probs=a.save_probs,
            regression=reg,
            average=a.average,
            min_agreement=a.min_agreement,
            keep_ladder=a.keep_ladder,
            memory_mb=a.memory_mb,
            scratch=a.scratch,
            max_held_rows=a.max_held_rows,
            threads=a.threads,
        )
    except (ValueError, OSError) as e:  # a rule, a path or a value that cannot be used: one line
        raise SystemExit(f"prune.py: {qf.one_line(e)}") from None
    print(summary(r))
    for line in auto_lines(r):
        print(line)


if __name__ == "__main__":
    main()
