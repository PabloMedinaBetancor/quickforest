#!/usr/bin/env python3
"""One command to a ready forest: prune -> held-out agreement -> compile -> equality test. Refuses to emit a model that fails a bar.

    quickforest optimize MODEL --data DATA.npz --keep 10%|50%|30|auto|100% --compile --verify [--out DIR] [--regression log|raw]
                         [--backend best|flat|codegen-pgo|...] [--native | --target-cpu x86-64-v3|...] [--shape fast|top]

--keep is the size you choose: a share (`50%`, `25%`, `10%`), a count per class (`30`), `100%` for no pruning (compile the whole forest, exact),
or `auto`: try the rungs of --keep-ladder (default 5%,10%,25%,50%), smallest first, keep the SMALLEST that passes the bar of the run, and, if
none does, compile the whole forest (exact, exit code 0, the reason in the report). With a count or share you chose, `--accept-any` keeps
it whatever the held-out figures say (they are reported; the bar does not stop the run).

MODEL: xgboost, LightGBM (text) or YDF (folder) model, or a folder of them (k-fold). --regression replaces the class-agreement bar by
the regression bar: median |pruned - full| <= 0.5 %, none above 2 %, held-out error vs the data's `y` no worse.

--compile builds the fastest artifact this machine can make (`--backend best`, the default): generated code profiled on --data (`codegen-pgo`) for
the portable x86-64-v2 CPU (`generic` on arm); the flat format only where rustc is missing, too little RAM is free or the model is too big for generated
code (the report says why; without an llvm-profdata of rustc's LLVM version the build is `codegen-block`). --native builds for the CPU of this machine instead: the build only runs on that CPU type, and it measured no faster than the portable build on our workloads (a library
built for features the CPU lacks is refused when loaded). Generated code is built in the `fast` shape (opt 3, no link-time optimisation, 16 codegen units,
the profile-guided pass in the same shape) unless --shape top asks for one codegen unit everywhere (chunks and driver) and no LTO across chunks; the build plans its parallel jobs against the free RAM.

Bars: held-out agreement >= --min-agreement (set it from your own real-case bar), and with --verify the compiled forest equal to
the model's own library (<= 1e-6, identical decisions). On a failed bar nothing but report.md / report.json is left in DIR and the
exit code is 2. If the backend cannot hold the model (see docs/REFERENCE.md) pruned/ is kept, nothing is compiled, and the exit code is 3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backend_codegen
import backends
import check
import compile
import numpy as np
import prune
import qf_common as qf
import qf_formats

FLOAT32_SUMS_YDF_ONLY = (
    "float32 sums reproduce YDF's own predictor and apply to YDF models only (xgboost already sums in float32, LightGBM in double)"
)


def relocate_paths(obj, old: Path, new: Path | None):
    """`obj` with every path inside the folder `old` moved to the same place inside `new`, or to None when that folder was deleted: the report
    is written after the work folder moved (or went), and must name only files that exist."""
    if isinstance(obj, dict):
        return {k: relocate_paths(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relocate_paths(v, old, new) for v in obj]
    if isinstance(obj, str) and obj:
        p = Path(obj)
        if p == old or old in p.parents:
            return None if new is None else str(new / p.relative_to(old))
    return obj


def default_out(model) -> str:
    """`MODEL-optimized` next to the model, by the name the model really has (`.` and `..` are folders with a name too)."""
    real = Path(os.path.realpath(model))
    if not real.stem:
        raise ValueError(f"{model}: no folder name to build the default --out from: give --out")
    return str(real.parent / f"{real.stem}-optimized")


def optimize(
    model,
    data,
    out,
    keep="10%",
    compile_=False,
    verify=False,
    *,
    decision="argmax",
    method="greedy",
    min_agreement=95.0,
    backend="best",
    average="output",
    target_cpu=None,
    shape=None,
    seed=15,
    sub=1.0,
    threads=None,
    regression=None,
    f64_leaves=False,
    float32_sums=False,
    keep_ladder=None,
    accept_any=False,
) -> dict:
    if not math.isfinite(min_agreement):
        raise ValueError(f"min_agreement must be a finite number, got {min_agreement}")
    prune.check_options(sub=sub, seed=seed)
    prune.check_keep(keep, keep_ladder, min_agreement)  # a bad --keep-ladder is found now, before any work
    if accept_any and prune.is_auto(keep):
        raise ValueError(
            "--accept-any takes the size you chose without the bar; --keep auto chooses the size by the bar: use one or the other"
        )
    if (f64_leaves or float32_sums) and not compile_:
        raise ValueError("--f64-leaves and --float32-sums shape the compiled forest: add --compile")
    if float32_sums and any(qf_formats.detect(f) != "ydf" for f in qf_formats.model_files(model)):
        # compile refuses this at once; here it would have pruned first and only then found that nothing can be compiled
        raise ValueError(FLOAT32_SUMS_YDF_ONLY)
    if verify:
        width = qf.data_width(data)  # --verify cannot test a model this wide: found now, not after pruning and compiling it
        if width is not None:
            check.verify_rows(width)
    if backend not in ("auto", "best"):
        backends.get(backend)  # a mistyped --backend is found now, before any work, with or without --compile
    if target_cpu and not compile_:
        raise ValueError("--native and --target-cpu shape the compiled forest: add --compile")
    backend_codegen.resolve_target_cpu({"target_cpu": target_cpu})  # a bad CPU name is found now, not after the pruning
    backend_codegen.resolve_shape({"shape": shape})  # and a bad shape
    if shape and not compile_:
        raise ValueError("--shape shapes the compiled forest: add --compile")
    out = Path(out)
    if out.exists() and not out.is_dir():
        raise SystemExit(f"{out} is a file, not a folder: choose a new --out")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: choose a new --out")
    work, existed = out / "_work", out.exists()
    try:
        work.mkdir(parents=True)
    except OSError as e:
        raise SystemExit(f"{out}: cannot create the output folder ({e.strerror})") from None
    try:
        t0 = time.perf_counter()
        rep, failures, kept = {"model": str(model), "keep": keep, "stages": {}}, [], None
        if prune.keeps_all(keep):
            pruned = work / "pruned"
            pruned.mkdir()
            kept = {}
            # the data and the regression bar are validated as in every pruned run, though nothing is selected from them
            rows = qf.load_data_cli(data)["X"]
            # the same checks as every pruned run: a stray file or a mixed folder is refused, not copied
            for f in qf_formats.model_files(model):
                src = qf_formats.open_source(f)
                qf.check_width(rows, src.spec.n_features, data)
                if src.spec.task == "classification":  # the rule is only built here to refuse one that cannot be used
                    qf.make_rule(decision, src.spec.groups, src.spec.kind, None, None, src.spec.task)
                if regression and src.spec.task != "regression":
                    raise SystemExit(f"{f.name}: the regression bar needs a regression model (this one is {src.spec.task})")
                kept[f.name] = prune.kept_indices(src, [np.flatnonzero(src.spec.tree_info == g) for g in range(src.spec.groups)])
                if f.is_dir():
                    shutil.copytree(f, pruned / f.name)
                else:
                    shutil.copy2(f, pruned / f.name)
            rep["stages"]["prune"] = "skipped (keep 100%)"
        else:
            pr = prune.run(
                model,
                data,
                work / "pruned",
                keep,
                decision=decision,
                method=method,
                sub=sub,
                seed=seed,
                regression=regression,
                average=average,
                min_agreement=min_agreement,
                keep_ladder=keep_ladder,
            )
            pruned = work / "pruned"
            kept = pr.get("kept_indices")
            rep["stages"]["prune"] = {
                "summary": prune.summary(pr),
                **pr["totals"],
                "heldout_pct": pr["heldout"]["pct"],
                "heldout_rows": pr["heldout"]["rows"],
                "disagreements": pr["heldout"]["rows"] - pr["heldout"]["same"],
                "cases": pr["disagreements"],  # the first 50: a refused run deletes pruned/report.json, which is where prune keeps them
            }
            whole = False
            if "auto" in pr:  # --keep auto: every rung tried and the one chosen, next to the usual figures
                rep["stages"]["prune"]["auto"] = pr["auto"]
                whole = pr["auto"]["chosen"]["whole_forest"]
            if "accuracy" in pr["heldout"]:
                rep["stages"]["prune"]["accuracy"] = pr["heldout"]["accuracy"]
            bar_failures = []
            if regression:
                rep["stages"]["prune"]["regression"] = pr["regression"]
                bar_failures = [f"regression bar: {m}" for m in pr["regression"]["failures"]]
            elif not whole and not prune.meets_agreement(pr["heldout"], min_agreement):  # the whole forest agrees with itself, exactly
                h = pr["heldout"]
                bar_failures.append(f"held-out agreement {100 * h['same'] / max(1, h['rows']):.4f} % < bar {min_agreement} %")
            if accept_any:  # the size was asked for: the figures are reported, and the bar does not stop the run
                rep["accept_any"] = {"waived": bar_failures, "min_agreement": None if regression else min_agreement}
            else:
                failures += bar_failures
        artifact, notice = None, None
        if compile_ and not failures:
            try:
                chosen = backend == "best"
                if chosen:  # named before the build: the artifact's suffix says how it loads
                    backend, notice = compile.resolve_best(pruned, average, float32_sums)
                artifact = work / ("model.bin" if backend == "auto" else f"model{backends.get(backend).suffix}")
                rep["stages"]["compile"] = compile.compile_model(
                    pruned,
                    artifact,
                    backend,
                    average,
                    data=data,
                    decision=decision,
                    f64_leaves=f64_leaves,
                    float32_sums=float32_sums,
                    target_cpu=target_cpu,
                    shape=shape,
                    pgo_fallback=chosen,  # profile rows too few or unlike one another: codegen-block, with a notice, as `best` does
                )
                if notice:  # the choice's notice first, then what the build itself said (its fallback)
                    built = rep["stages"]["compile"].get("notice")
                    rep["stages"]["compile"]["notice"] = f"{notice}; {built}" if built else notice
                backend = rep["stages"]["compile"]["backend"]  # "auto" has chosen one
            except (
                ValueError,
                backend_codegen.ToolchainMissing,
            ) as e:  # not a failed bar: the pruned models are still good, only this backend cannot be built
                rep["compile_unsupported"] = qf.one_line(e)
            if verify and "compile" in rep["stages"]:
                n_features = qf_formats.open_source(qf_formats.model_files(pruned)[0]).spec.n_features
                try:
                    v = check.run(
                        pruned,
                        artifact,
                        backend,
                        data,
                        n_random=check.verify_rows(n_features),
                        seed=seed,
                        decision=decision,
                        average=average,
                        threads=threads,
                        speed=True,
                        f64=bool(f64_leaves or float32_sums),
                    )
                except (
                    backend_codegen.TargetCpuMismatch
                ) as e:  # built for a CPU this machine lacks: not a failed bar, the pruned models stay
                    rep["compile_unsupported"] = qf.one_line(e)
                    rep["stages"].pop("compile")
                    artifact.unlink(missing_ok=True)
                    Path(f"{artifact}.choice.json").unlink(missing_ok=True)
                    v = None
                if v is not None:
                    rep["stages"]["verify"] = v
                    if not v["ok"]:
                        failures.append(
                            "compiled forest differs from the source library: "
                            + "; ".join(
                                f"{s['set']} err {s['max_err']:.1e} / {s['decision_mismatches']} decisions"
                                for s in v["sets"]
                                if not s["ok"]
                            )
                        )
        rep["compile_unsupported"] = rep.get("compile_unsupported")
        # pruned/ is emitted only if no bar failed: then the report names the trees of the original model it holds
        if not failures and kept is not None:
            rep["kept_indices"] = kept
        rep["ok"], rep["failures"], rep["seconds"] = not failures, failures, round(time.perf_counter() - t0, 1)
        if failures:
            shutil.rmtree(work)
        else:
            for item in work.iterdir():
                shutil.move(str(item), out / item.name)
            work.rmdir()
        rep.update(
            relocate_paths(rep, work, None if failures else out)
        )  # the artifact was built in _work/: name where it is now, if at all
        qf.replace_file(out / "report.json", data=json.dumps(rep, indent=1).encode())
        qf.replace_file(out / "report.md", data=qf.utf8(report_md(rep, out)))
        return rep

    except BaseException:  # a refused argument or a crash must not leave a folder that blocks the next try in the same --out
        shutil.rmtree(work, ignore_errors=True)
        if not existed:
            out.rmdir()
        raise


def report_md(rep: dict, out: Path) -> str:
    lines = [f"# optimize: {'OK' if rep['ok'] else 'REFUSED'}", f"model {rep['model']}, keep {rep['keep']}, {rep['seconds']} s"]
    p = rep["stages"].get("prune")
    if isinstance(p, dict):
        lines.append(f"- prune: {p['summary']}")
        if "auto" in p:
            lines += ["", *[f"  {ln}" for ln in prune.auto_lines({"auto": p["auto"]})], ""]
        if p.get("cases") and "regression" not in p:
            shown = p["cases"][:10]
            lines.append(
                f"- disagreeing held-out rows (first {len(shown)} of {p['disagreements']}): "
                + "; ".join(f"row {c['row']}: {c['full']} -> {c['pruned']}" for c in shown)
            )
        if "regression" in p and not p["regression"]["error_checked"]:
            lines.append("- WARNING: no truth `y` in the data, so the held-out error bar was NOT checked")
    elif p:
        lines.append(f"- prune: {p}")
    if isinstance(p, dict) and "accuracy" in p:
        a = p["accuracy"]
        lines.append(
            f"- held-out accuracy against the truth y ({a['rows']:,} rows): full forest {a['full_pct']} %, pruned {a['pruned_pct']} %"
        )
    if "accept_any" in rep:
        w = rep["accept_any"]["waived"]
        lines.append(
            "- ACCEPTED WITHOUT THE BAR (--accept-any): "
            + ("; ".join(w) if w else "the bar would have passed")
            + (
                f" (the bar was --min-agreement {rep['accept_any']['min_agreement']})"
                if w and rep["accept_any"]["min_agreement"] is not None
                else ""
            )
        )
    c = rep["stages"].get("compile")
    if c:
        target = (
            f", built for {c['target_cpu']} CPU" + (" (runs only on this CPU type)" if c["target_cpu"] == "native" else "")
            if "target_cpu" in c
            else ""
        )
        target += f", {c['shape']} shape" if "shape" in c else ""
        lines.append(f"- compile ({c['backend']}{target}): {c['bytes'] / 1e6:.2f} MB, {c['trees']:,} trees, {c['nodes']:,} nodes")
        if "profile_rows" in c:
            lines.append(
                f"- profile: {c['profile_rows']:,} rows; two halves of them agree on {c['profile_overlap']:.1f} % of the edge counts "
                f"and {c['profile_function_overlap']:.1f} % of the function counts"
            )
        if c.get("notice"):
            lines.append(f"- note: {c['notice']}")
        if c.get("table_md"):
            lines.append(f"\nBackends timed on held-out rows ({c['basis']}):\n\n{c['table_md']}")
    v = rep["stages"].get("verify")
    if v:
        lines.append(
            "- verify: "
            + ("PASS" if v["ok"] else "FAIL")
            + ", "
            + ", ".join(
                f"{s['set']} {s['rows']:,} rows max err {s['max_err']:.1e}, decisions differ {s['decision_mismatches']}" for s in v["sets"]
            )
        )
        if "speed" in v:
            s = v["speed"]
            lines.append(
                f"- speed ({s.get('library_threads', 1)} thread(s), {s['rows_from']}): {s.get('library', 'xgboost')} {s['library_rows_per_s']:,} rows/s, compiled "
                f"{s['compiled_rows_per_s']:,} rows/s ({s['speedup']}x); load {s['load_seconds']} s, extra RSS {s['extra_rss_mb']} MB"
            )
    lines += [f"- FAILED: {f}" for f in rep["failures"]]
    if rep["compile_unsupported"]:
        lines.append(f"- NOT COMPILED: {rep['compile_unsupported']}")
    lines.append(
        "Emitted: "
        + ("pruned/ and the compiled artifact" if rep["ok"] and "compile" in rep["stages"] else "pruned/" if rep["ok"] else "nothing")
    )
    return "\n".join(lines) + "\n"


def main(argv=None):
    qf.odd_names_are_printable()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", required=True)
    ap.add_argument(
        "--keep",
        default="10%",
        help=(
            "a share of the trees of each class (10%%) or a count of trees per class (30), rounded to the nearest whole tree, at least 1; "
            "100%% = no pruning; auto: the smallest rung of --keep-ladder that passes the bar"
        ),
    )
    ap.add_argument(
        "--accept-any",
        action="store_true",
        help="take the --keep size you chose whatever the held-out figures say (best-N from a huge forest): they are reported, the bar does "
        "not stop the run. Not with --keep auto",
    )
    ap.add_argument("--keep-ladder", help="--keep auto: the rungs to try, smallest first, comma separated (default 5%%,10%%,25%%,50%%)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--decision", default="argmax")
    ap.add_argument("--method", choices=("greedy", "orientation"), default="greedy")
    ap.add_argument("--min-agreement", type=qf.finite_float, default=95.0, help="held-out agreement bar, percent")
    ap.add_argument(
        "--backend",
        default="best",
        help="best (default: generated code, profile-guided on --data), flat, wave, codegen-pgo, ... or auto: time the candidates "
        "and keep the fastest exact one",
    )
    ap.add_argument(
        "--target-cpu",
        help="the CPU generated code is built for: x86-64-v2 (default, portable; generic on arm), x86-64-v3, x86-64, native, ...; "
        "a library is refused at load time on a CPU that lacks its features",
    )
    ap.add_argument(
        "--native",
        action="store_true",
        help="build for the CPU of THIS machine (--target-cpu native): the build only runs on that CPU type; measured no faster than the portable build on our workloads",
    )
    ap.add_argument(
        "--shape",
        choices=tuple(backend_codegen.SHAPES),
        help="how generated code is built: fast (default: opt 3, no link-time optimisation, 16 codegen units; the profile-guided pass in the same shape) "
        "or top (one codegen unit everywhere, chunks and driver, and no LTO across chunks; slower to build)",
    )
    ap.add_argument("--average", choices=("output", "margin"), default="output")
    ap.add_argument("--seed", type=int, default=15)
    ap.add_argument("--sub", type=float, default=1.0)
    ap.add_argument(
        "--threads",
        type=qf.threads_arg,
        default=None,
        metavar="N|all",
        help="threads of the compiled forest's predict in --verify, and of the speed line (the library and the compiled forest on the same number): "
        "all (default: every usable core) or a whole number",
    )
    ap.add_argument(
        "--f64-leaves",
        action="store_true",
        help="compile with double leaves where float32 would round them (as compile --f64-leaves); --verify then tests the float64 answer",
    )
    ap.add_argument(
        "--float32-sums",
        action="store_true",
        help=(
            "YDF only: compile with float32 sums as YDF's own predictor (as compile --float32-sums); --verify then tests the float64 answer"
        ),
    )
    ap.add_argument("--regression", choices=("log", "raw"), help="regression bar: the model outputs log(target) or the target itself")
    ap.add_argument("--med-tol", type=qf.finite_float, default=0.5, help="regression: median |pruned - full| bar, percent")
    ap.add_argument("--max-tol", type=qf.finite_float, default=2.0, help="regression: no row above this, percent")
    ap.add_argument(
        "--error-slack", type=qf.finite_float, default=0.0, help="regression: allowed relative worsening of the held-out MAE vs y"
    )
    a = ap.parse_args(argv)
    reg = qf.RegBar("log" if a.regression == "log" else "raw", a.med_tol, a.max_tol, a.error_slack) if a.regression else None
    if a.verify and not a.compile:
        ap.error("--verify tests the compiled forest: add --compile")
    try:
        out = a.out or default_out(a.model)
        rep = optimize(
            a.model,
            a.data,
            out,
            a.keep,
            a.compile,
            a.verify,
            decision=a.decision,
            method=a.method,
            min_agreement=a.min_agreement,
            backend=a.backend,
            target_cpu=backend_codegen.cpu_option(a.native, a.target_cpu),
            shape=a.shape,
            average=a.average,
            seed=a.seed,
            sub=a.sub,
            threads=a.threads,
            regression=reg,
            f64_leaves=a.f64_leaves,
            float32_sums=a.float32_sums,
            keep_ladder=a.keep_ladder,
            accept_any=a.accept_any,
        )
    except (ValueError, OSError) as e:  # an argument or a file that cannot be used: one line
        raise SystemExit(f"optimize.py: {qf.one_line(e)}") from None
    print((Path(out) / "report.md").read_text(), end="")
    sys.exit(2 if not rep["ok"] else 3 if rep["compile_unsupported"] else 0)


if __name__ == "__main__":
    main()
