#!/usr/bin/env python3
"""Equality harness: the compiled forest against xgboost / LightGBM / YDF on the caller's rows AND on random rows built to hit the split
thresholds exactly (equal, one float32 step above and below) and NaN, and on rows that reach nodes of every tree and fall on their
threshold (set `paths`). Fails (exit 1) unless the max difference is <= 1e-6 (relative to max(1, |value|)) and every unambiguous decision
is identical. For xgboost without `--margin` the bar is the claim itself: the answers must be identical to the bit (difference 0).

    check.py MODEL(file|dir) ARTIFACT [--data DATA.npz] [--random 10000] [--decision argmax] [--speed] [--margin] [--f64] [--backend NAME]

The ARTIFACT is loaded as `backends.load_artifact` does: the backend named in `ARTIFACT.choice.json` (what `compile` with its default backend,
`compile --backend auto` and `choose.py` write), else by suffix (`.so` generated code, otherwise the flat file). `--backend NAME` forces one.

`--absolute` judges the 1e-6 bar on the plain difference (a log-scale target of ~12 then has to match to 1e-6 absolute, i.e. ~8e-8 relative; the
float32 answer's own rounding is up to 4.8e-7 there and 9.5e-7 above 16). `--margin` also passes a base margin (rows, groups) to both sides:
the data file's `base_margin` array when it has one, else a seeded random one (uniform in -3..3); it is ADDED to each model's raw score
before the objective's transform, and the library gets the same numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backend_codegen
import backends
import evaluator
import qf_common as qf
import qf_formats

TOL = 1e-6


MAX_RANDOM_VALUES = 100_000_000  # rows x features of one random test set (400 MB of float32)
VERIFY_ROWS = 10_000  # random rows `optimize --verify` tests with, as many as `check --random` defaults to
VERIFY_MIN_ROWS = 1_000  # fewer random rows are no test (reached above 100,000 features)


def verify_rows(n_features: int) -> int:
    """How many random rows `optimize --verify` can test with: 10,000, fewer when 10,000 x features would pass the one test set's limit
    (`MAX_RANDOM_VALUES`; a model at the 32,767-feature limit of LIMITS.md gets 3,051). Below `VERIFY_MIN_ROWS` (a model of more than 100,000
    features, which `compile` refuses anyway) the check is impossible and says so."""
    rows = min(VERIFY_ROWS, MAX_RANDOM_VALUES // max(1, n_features))
    if rows < VERIFY_MIN_ROWS:
        raise ValueError(
            f"--verify: a model of {n_features:,} features cannot be tested in one set of at most {MAX_RANDOM_VALUES:,} values "
            f"(at least {VERIFY_MIN_ROWS:,} random rows are needed); verify with `check --random N` on a smaller sample, or leave out --verify"
        )
    return rows


def random_rows(forest: qf.Forest, n: int, seed: int) -> np.ndarray:
    """Rows whose cells are, half the time, one of the model's own thresholds for that feature (exactly, or one float32 step away)."""
    rng = np.random.default_rng(seed)
    nf = forest.n_features
    if n * nf > MAX_RANDOM_VALUES:  # a count that would be allocated, filled and swapped for ever (or killed) is a refused argument
        raise ValueError(f"{n:,} random rows of {nf:,} features are {n * nf:,} values; one test set holds at most {MAX_RANDOM_VALUES:,}")
    feats, vals = [], []
    for m in forest.models:
        for t in m.trees:
            inner = t.left != -1
            num = inner.copy()
            for k, c in (t.cat or {}).items():  # a categorical node's "threshold" is its category set: hug the members and the edges
                num[k] = False
                # values just outside the set, and the negatives that truncate toward zero to -1 (-1.0, -1.5, -1.9): they go right, also at an
                # empty set, and a generated `x as i64 < 0` test once sent them left
                extra = [c.max() + 1, c.max() + 0.5] if len(c) else [0.0, 0.5]
                feats.append(np.full(len(c) + len(extra) + 3, t.feature[k]))
                vals.append(np.concatenate([c, extra, [-1.0, -1.5, -1.9]]).astype(np.float32))
            feats.append(t.feature[num])
            vals.append(t.value[num])
    f, v = np.concatenate(feats), np.concatenate(vals)
    order = np.lexsort((v, f))
    f, v = f[order], v[order]
    cnt = np.bincount(f, minlength=nf)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    lo = np.full(nf, -1.0, np.float32)
    hi = np.full(nf, 1.0, np.float32)
    has = cnt > 0
    lo[has] = np.minimum.reduceat(v, off[has]) - 1
    hi[has] = np.maximum.reduceat(v, off[has]) + 1
    x = (lo + (hi - lo) * rng.random((n, nf), dtype=np.float32)).astype(np.float32)
    if has.any():  # a forest of leaf-only trees has no thresholds to hug
        pick = (rng.random((n, nf)) < 0.5) & has
        idx = off + (rng.random((n, nf)) * cnt).astype(np.int64)
        idx = np.minimum(idx, len(v) - 1)
        t = v[idx]
        step = rng.integers(0, 3, (n, nf))
        t = np.where(step == 1, np.nextafter(t, np.float32(np.inf)), np.where(step == 2, np.nextafter(t, np.float32(-np.inf)), t))
        x = np.where(pick, t, x).astype(np.float32)
    x[rng.random((n, nf)) < 0.1] = np.nan
    big = np.float32(np.finfo(np.float32).max)
    return np.clip(x, -big, big)  # +-inf (an infinite threshold) is not an input the formats agree on; NaN stays


def path_rows(forest: qf.Forest, n: int, seed: int) -> np.ndarray:
    """About `n` rows that REACH split nodes of every tree and fall on their threshold: for a sampled node, a row whose path from the root
    satisfies every split above it and whose feature is at the node's threshold, one float32 step below it, and NaN (a categorical node:
    a member of the set, a value outside it, a negative fraction, NaN). A defect in a node (a threshold one step off, a flipped default direction, swapped
    children) changes an answer only for such a row; `random_rows` hugs a random threshold of the whole forest, so a given node was hit
    by luck. The sampled nodes are limited to n/3 (n/4 with a categorical split); `run` asks for n = 4 x `--random` rows, so 4/3 of
    `--random` nodes (13,333 at the default 10,000), and to as many as keep the path rows within `MAX_RANDOM_VALUES` values (a model of
    32,767 features gets 1,017 nodes, 3,051 rows). Every tree is asked for at least one node and the rest is shared in proportion to the trees' sizes;
    the quotas round up, so when they add up to more than the budget (a forest of several thousand trees) a random subset of the picks is kept and a few
    trees (91 of 7,500 in one case measured) have no path row, only the `random` rows hugging their thresholds: raise `--random` to cover every tree."""
    rng = np.random.default_rng(seed)
    nf, inf, f32 = forest.n_features, np.float32(np.inf), np.float32
    trees = [t for m in forest.models for t in m.trees]
    inner = [np.flatnonzero(t.left != -1) for t in trees]
    total = sum(len(i) for i in inner)
    if total == 0 or n < 3:
        return np.zeros((0, nf), np.float32)
    per_node = 4 if any(t.cat for t in trees) else 3  # a categorical node gets a fourth row: a negative fraction
    # the base rows and the path rows are each held to one test set's limit, as `verify_rows` holds the random rows: a wide forest gets fewer
    # path rows, never a refusal after the pruning and the compile
    budget = max(min(n // per_node, MAX_RANDOM_VALUES // (per_node * max(1, nf))), 1)
    picks = []
    for ti, ids in enumerate(inner):
        if len(ids):
            quota = min(len(ids), max(1, round(budget * len(ids) / total)))
            picks += [(ti, int(k)) for k in rng.choice(ids, quota, replace=False)]
    if len(picks) > budget:
        picks = [picks[i] for i in sorted(rng.choice(len(picks), budget, replace=False))]
    base = random_rows(forest, len(picks), seed + 1)
    parents, rows = {}, []
    for j, (ti, k) in enumerate(picks):
        t = trees[ti]
        if ti not in parents:
            up = np.full((len(t.left), 2), -1, np.int64)  # the parent and which child this is (0 = left, 1 = right)
            inn = np.flatnonzero(t.left != -1)
            up[t.left[inn]] = np.stack([inn, np.zeros(len(inn), np.int64)], 1)
            up[t.right[inn]] = np.stack([inn, np.ones(len(inn), np.int64)], 1)
            parents[ti] = up
        up, chain, node = parents[ti], [], k
        while up[node][0] != -1:
            chain.append((int(up[node][0]), int(up[node][1])))
            node = int(up[node][0])
        lo, hi, fixed = {}, {}, {}
        for a, side in reversed(chain):  # from the root down: the interval each feature must lie in, the category a split fixed
            f, cats = int(t.feature[a]), (t.cat or {}).get(a)
            if cats is not None:
                c = np.asarray(cats, np.float64)
                member, outside = (c[0], c.max() + 1) if len(c) else (0.0, 1.0)
                fixed[f] = f32(outside if side else member)
            elif side == 0:
                hi[f] = min(hi.get(f, inf), f32(t.value[a]))
            else:
                lo[f] = max(lo.get(f, -inf), f32(t.value[a]))
        r0 = base[j].copy()
        for f in {*lo, *hi}:
            a, b = lo.get(f, -inf), hi.get(f, inf)
            v = r0[f]
            if np.isnan(v) or not (a <= v < b):
                with np.errstate(all="ignore"):
                    if np.isfinite(a) and np.isfinite(b):
                        v = f32((np.float64(a) + np.float64(b)) / 2)
                    elif np.isfinite(b):
                        v = f32(b - max(f32(1), abs(b)))
                    elif np.isfinite(a):
                        v = f32(a + max(f32(1), abs(a)))
                    else:
                        v = f32(0.5)
                v = np.nextafter(b, -inf) if v >= b else a if v < a else v
            r0[f] = v
        for f, v in fixed.items():
            r0[f] = v
        f, cats = int(t.feature[k]), (t.cat or {}).get(k)
        if cats is not None:
            c = np.asarray(cats, np.float64)
            cands = [f32(c[0]) if len(c) else f32(0), f32(c.max() + 1) if len(c) else f32(1), f32(-1.5), f32(np.nan)]
        else:
            th = f32(t.value[k])
            cands = [th, np.nextafter(th, -inf), f32(np.nan)]
        for v in cands:
            r = r0.copy()
            r[f] = v
            rows.append(r)
    big = f32(np.finfo(np.float32).max)
    return np.clip(np.array(rows, f32).reshape(-1, nf), -big, big)  # NaN stays


def ambiguous(decision: str, p: np.ndarray, groups: int) -> np.ndarray:
    """Rows whose decision sits within float noise of a boundary: a difference there is not an error of the evaluator."""
    if decision == "argmax" and groups > 1:
        top = np.sort(p, axis=1)
        return (top[:, -1] - top[:, -2]) <= 2 * TOL
    if decision == "argmax" or decision.startswith("threshold:"):
        t = 0.5 if decision == "argmax" else float(decision.split(":", 1)[1])
        return np.abs(p[:, 0] - t) <= 2 * TOL
    return np.zeros(len(p), bool)


def bit_exact(files, average: str, forest, margin: bool = False) -> bool:
    """Whether the compiled forest is judged at difference 0 against the library: xgboost's answers are claimed identical to the bit
    (docs/REFERENCE.md), except with a base margin (xgboost adds it in its own float32 order) and with several folds averaged as a margin
    through a sigmoid / softmax (the reference applies numpy's float32 transform, xgboost the C library's)."""
    return (
        qf_formats.detect(files[0]) == "xgboost"
        and not margin
        and not (len(files) > 1 and average == "margin" and forest.kind != "identity")
    )


def compare(name, ref, got, decision, rule, groups, absolute=False, exact=False):
    ref, got = ref.reshape(len(ref), -1), got.reshape(len(got), -1)
    scale = 1.0 if absolute else np.maximum(1.0, np.abs(ref))  # relative to max(1, |value|) unless the caller wants an absolute bar
    err = float((np.abs(ref - got) / scale).max()) if len(ref) else 0.0
    out = {"set": name, "rows": len(ref), "max_err": err}
    if rule is not None:
        skip = ambiguous(decision, ref, groups)
        diff = (rule(ref) != rule(got)) & ~skip
        out.update(decision_mismatches=int(diff.sum()), ambiguous_skipped=int(skip.sum()))
    else:
        out.update(decision_mismatches=0, ambiguous_skipped=0)
    out["ok"] = err <= (0.0 if exact else TOL) and out["decision_mismatches"] == 0
    return out


def rss_mb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6


class ArtifactLoader:
    """What `check` loads with when no `--backend` is named: the artifact says which backend it is (`backends.load_artifact`)."""

    @staticmethod
    def load(artifact, **options):
        return backends.load_artifact(artifact, **options)


def bench(files, artifact, backend, x, threads, f64=False):
    """The model's own library against the compiled forest on the SAME number of threads (fixed batch, `qf_formats.library_rate`); load time
    and extra memory of the compiled one. `threads=None` (the default of `check`) is every usable core (`evaluator.usable_cpus`); a whole
    number is that many (clamped to 1..64; 1 gives the one-thread figures). The report says how many threads both sides ran on."""
    threads = evaluator.usable_cpus() if threads is None else evaluator.clamp_threads(threads)
    srcs = [qf_formats.open_source(f) for f in files]
    lib_rate = qf_formats.library_rate(srcs, x, threads=threads)
    reps = max(1, 20000 // len(x))
    xs = np.tile(x, (reps, 1))
    r0, t0 = rss_mb(), time.perf_counter()
    pred = backend.load(Path(artifact))
    load_s, extra = time.perf_counter() - t0, rss_mb() - r0
    t = time.perf_counter()
    pred.predict(xs, threads)
    t_flat = time.perf_counter() - t
    extra_f64 = {}
    if f64:  # float32 and float64 answers on the same rows, interleaved passes, medians (the ratio inside one run is the safe figure)
        t32, t64 = [], []
        for _ in range(5):
            t = time.perf_counter()
            pred.predict(xs, threads)
            t32.append(time.perf_counter() - t)
            t = time.perf_counter()
            pred.predict(xs, threads, dtype=np.float64)
            t64.append(time.perf_counter() - t)
        a, b = float(np.median(t32)), float(np.median(t64))
        extra_f64 = {
            "f32_output_rows_per_s": round(len(xs) / a),
            "f64_output_rows_per_s": round(len(xs) / b),
            "f64_over_f32_time": round(b / a, 3),
        }
    return {
        **extra_f64,
        "rows": len(xs),
        "library": srcs[0].fmt,
        "baseline": f"{srcs[0].fmt}, {threads} thread(s), {qf_formats.BASELINE_ROWS:,}-row batches",
        "library_threads": threads,
        "compiled_threads": threads,
        "library_rows_per_s": round(lib_rate),  # the model's own library (xgboost, LightGBM or YDF)
        "compiled_rows_per_s": round(len(xs) / t_flat),
        "speedup": round(len(xs) / t_flat / lib_rate, 2),
        "load_seconds": round(load_s, 3),
        "extra_rss_mb": round(extra, 1),
    }


def run(
    model,
    artifact,
    backend=None,
    data=None,
    *,
    n_random=10000,
    seed=15,
    decision="argmax",
    average="output",
    threads=None,
    speed=False,
    max_rows=200000,
    margin=False,
    absolute=False,
    f64=False,
) -> dict:
    """`backend`: a backend name, or None to load the artifact as `backends.load_artifact` does (its `.choice.json`, else its suffix).
    `f64`: judge the compiled forest's float64 answers (`predict(..., dtype=np.float64)`) instead of the float32 ones; with `absolute`
    the 1e-6 bar is then on the plain difference of two doubles (LightGBM / YDF: the library's own double answer)."""
    if n_random < 0:
        raise ValueError(f"--random {n_random}: the number of random rows cannot be negative")
    if not data and not n_random:
        raise ValueError("nothing to test: give --data, or leave --random above 0")
    files = qf_formats.model_files(model)
    be = ArtifactLoader if backend is None else backends.get(backend)
    pred = be.load(Path(artifact))
    forest = qf.load_forest(files, average)
    qf.require_rule(decision, forest)
    if margin and not (np.isfinite(forest.margin_scale) and forest.margin_scale > 0):
        raise ValueError(
            "--margin: the models have different sigmoid scales (LightGBM sigmoid:s), so one base margin means different things to each"
        )
    sets, margins = [], {}
    if data:
        d = qf.load_data_cli(data)
        qf.check_width(d["X"], forest.n_features, data)
        sets.append(("data", d["X"][:max_rows]))
        if margin and "base_margin" in d:
            margins["data"] = np.asarray(d["base_margin"], np.float64).reshape(len(d["X"]), -1)[:max_rows]
    if n_random:
        sets.append(("random", random_rows(forest, n_random, seed)))
        paths = path_rows(forest, 4 * n_random, seed + 2)
        if len(paths):
            sets.append(("paths", paths))
    # xgboost's answers are claimed identical to the bit (docs/REFERENCE.md), so a difference of any size fails, as the differential's key
    # `xgb` does. Not judged that way: a base margin (xgboost adds it in its own float32 order), and several folds averaged as a margin
    # through a sigmoid / softmax (the reference applies numpy's float32 transform, xgboost the C library's).
    exact = bit_exact(files, average, forest, margin)
    results = []
    for name, x in sets:
        if margin and name not in margins:
            margins[name] = np.random.default_rng(seed + 1).uniform(-3, 3, (len(x), forest.groups))
        bm = margins.get(name)
        ref = qf.reference_predict(files, x, average, bm)
        rule = qf.make_rule(decision, forest.groups, forest.kind, None, None, forest.task) if forest.task == "classification" else None
        kw = ({"base_margin": bm} if bm is not None else {}) | ({"dtype": np.float64} if f64 else {})
        got = pred.predict(x, *(() if threads is None else (threads,)), **kw)  # None: the backend's own default (all cores)
        results.append({**compare(name, ref, got, decision, rule, forest.groups, absolute, exact), "base_margin": bm is not None})
    out = {
        "ok": all(r["ok"] for r in results),
        "bar": ("identical to the bit" if exact else f"max {'absolute' if absolute else 'relative'} err <= {TOL}")
        + " and identical decisions"
        + (" (float64 output)" if f64 else ""),
        "sets": results,
    }
    if speed and sets:
        # real rows when the caller gave them: the threshold-hugging random rows walk unlikely, deep paths and run about twice as slow
        basis, rows = next(((n, x) for n, x in sets if n == "data"), sets[-1])
        out["speed"] = {
            **bench(files, artifact, be, rows[:2000], threads, f64),
            "rows_from": "real rows" if basis == "data" else "random rows",
        }
    return out


def main():
    qf.odd_names_are_printable()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("artifact")
    ap.add_argument("--data")
    ap.add_argument("--random", type=qf.count_arg(0), default=10000)
    ap.add_argument("--decision", default="argmax")
    ap.add_argument("--average", choices=("output", "margin"), default="output")
    ap.add_argument("--backend", help="force a backend (flat, wave, codegen-block, ...); default: the one the artifact says it is")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument(
        "--threads",
        type=qf.threads_arg,
        default=None,
        metavar="N|all",
        help="threads of the compiled forest's predict: all (default: every usable core on a large batch) or a whole number; "
        "--speed times the library and the compiled forest on that many (all: every usable core; 1 for the one-thread figures)",
    )
    ap.add_argument("--speed", action="store_true")
    ap.add_argument("--margin", action="store_true", help="also pass a base margin to both sides (see the top of this file)")
    ap.add_argument(
        "--f64",
        action="store_true",
        help="judge the float64 answers (predict(dtype=np.float64)); with --speed also times them against float32",
    )
    ap.add_argument("--absolute", action="store_true", help="the 1e-6 bar on |library - compiled| itself, not relative to max(1, |value|)")
    ap.add_argument("--json")
    a = ap.parse_args()
    try:
        if a.json:
            qf.output_file(a.json, "--json")
        r = run(
            a.model,
            a.artifact,
            a.backend,
            a.data,
            n_random=a.random,
            seed=a.seed,
            decision=a.decision,
            average=a.average,
            threads=a.threads,
            speed=a.speed,
            margin=a.margin,
            absolute=a.absolute,
            f64=a.f64,
        )
    except (evaluator.EngineRefused, backend_codegen.TargetCpuMismatch) as e:
        # wave engines on a walker-only file, or generated code built for a CPU this machine lacks (loading it would crash): the reason, exit 3
        print(f"check.py: {qf.one_line(e)}", file=sys.stderr)
        sys.exit(3)
    except (ValueError, OSError) as e:  # an argument or a file that cannot be used: one line
        raise SystemExit(f"check.py: {qf.one_line(e)}") from None
    for s in r["sets"]:
        print(
            f"{s['set']}{' + base margin' if s['base_margin'] else ''}: {s['rows']:,} rows, max err {s['max_err']:.2e}, decision "
            f"mismatches {s['decision_mismatches']} (ambiguous skipped {s['ambiguous_skipped']})"
        )
    if "speed" in r:
        print("speed:", json.dumps(r["speed"]))
    print("PASS" if r["ok"] else "FAIL")
    if a.json:
        try:
            qf.replace_file(a.json, data=json.dumps(r, indent=1).encode())
        except OSError as e:  # the folder was checked before the run; this is a write that failed since (a full disk, a folder removed)
            raise SystemExit(
                f"check.py: {'PASS' if r['ok'] else 'FAIL'}, but --json {a.json} could not be written: {qf.one_line(e)}"
            ) from None
    sys.exit(0 if r["ok"] else 1)


if __name__ == "__main__":
    main()
