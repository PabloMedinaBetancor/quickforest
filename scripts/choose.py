#!/usr/bin/env python3
"""Pick the fastest EXACT backend for one model on THIS machine and write its artifact: `compile.py --backend auto` calls this.

    choose.py MODEL(file|dir) OUT --data DATA.npz [--decision argmax] [--average output|margin] [--cpu N] [--only a,b] [--keep-work]

Steps: parse the model; list the candidates that can hold it (`supports`) and run here (the generated-code ones need rustc; the
AVX2 one only where the CPU's gathers are fast, see `engine_ok`); compile them in a work folder next to OUT; run each in its own
process against the library's own answers (compare.run_table); keep the fastest one that is exact, judged on HELD-OUT real rows
(rows that neither the exactness test nor the profile-guided build saw). The winner is copied to OUT (+ the evaluator library for the
flat family) and `OUT.choice.json` remembers which backend loads it: `backends.load_artifact(OUT)`. The table is printed.
A profile-guided build takes minutes on a pruned model and far longer on a very large one; pass `--only flat,wave` to skip it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backend_codegen
import backends
import compare
import evaluator
import qf_common as qf
import qf_formats

# Largest model, in nodes, that the generated-code backends are offered for (rustc's time and memory grow with the node count).
MAX_CODEGEN_NODES = 6_000_000


def can_run(name: str) -> bool:
    """Is this backend usable on this machine (a toolchain for the generated code, a non-mitigated gather unit for AVX2)?"""
    if name.startswith("codegen"):
        try:
            backend_codegen.toolchain_bin()
            if name == "codegen-pgo":
                backend_codegen.llvm_profdata()
        except FileNotFoundError:
            return False
    if name == "wave-avx2":
        lib = evaluator.load_lib(evaluator.find_lib())
        return bool(lib.tfl_engine_ok(2))
    return True


def candidates(forest, only=None) -> tuple[list[str], dict[str, str]]:
    """(names to build, {name: why not}) for this model and machine. `codegen` (a row at a time) is never offered: it lost everywhere."""
    names, skipped = [], {}
    for name, be in backends.BACKENDS.items():
        if only and name not in only:
            continue
        if name == "codegen":
            continue
        missing = forest.requires - set(getattr(be, "supports", ()))
        if missing:
            skipped[name] = f"needs {sorted(missing)}"
        elif name.startswith("codegen") and forest.n_nodes > MAX_CODEGEN_NODES:
            skipped[name] = f"{forest.n_nodes:,} nodes is above the {MAX_CODEGEN_NODES:,} the generated code was built for"
        elif name.startswith("codegen") and (depth := backend_codegen.too_deep(forest)) is not None:
            skipped[name] = f"a tree is {depth:,} levels deep, above the {backend_codegen.MAX_TREE_DEPTH:,} generated code can hold"
        elif not can_run(name):
            skipped[name] = (
                "no rustc, or no llvm-profdata of its LLVM version, here"
                if name.startswith("codegen")
                else "gathers are slow or absent on this CPU"
            )
        else:
            names.append(name)
    return names, skipped


def choose(
    model,
    out,
    data,
    *,
    decision="argmax",
    average="output",
    cpu=None,
    only=None,
    keep_work=False,
    compile_options=None,
    log=print,
) -> dict:
    out = Path(out)
    files = qf_formats.model_files(model)
    qf.require_utf8_path(out, "the artifact's")
    qf.refuse_to_overwrite(out, files)
    qf.require_cpu(cpu)
    forest = qf.load_forest(files, average, bool((compile_options or {}).get("float32_sums")))
    qf.require_rule(decision, forest)
    unknown = [n for n in only or () if n not in backends.BACKENDS]
    if unknown:
        raise ValueError(f"--only {','.join(unknown)}: not a backend; one of {sorted(backends.BACKENDS)}")
    names, skipped = candidates(forest, only)
    if not names:
        raise ValueError(f"no backend can hold this model here: {skipped}")
    for n, why in skipped.items():
        log(f"skipped {n}: {why}")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".choose-", dir=out.parent))
    try:
        table = compare.run_table(
            model, names, data, work, decision=decision, average=average, cpu=cpu, small=(), compile_options=compile_options, log=log
        )
        best = compare.best_exact(table)
        if best is None:
            raise ValueError(
                "no candidate was exact on the test rows: " + "; ".join(f"{r['backend']}: {r.get('status')}" for r in table["rows"])
            )
        art = work / Path(best["artifact"]).name
        qf.replace_file(out, src=art)
        if art.suffix == ".flat":
            qf.copy_unless_same(evaluator.find_lib(art), out.parent / evaluator.LIB_NAME)
        info = {
            "backend": best["backend"],
            "rows_per_s": best["rows_per_s"],
            "speedup_vs_library": best["speedup"],
            "basis": "held-out real rows",
            "cpu": table["cpu"],
            "artifact": str(out),
            "bytes": out.stat().st_size,
            "nodes": forest.n_nodes,
            "trees": forest.n_trees,
            "skipped": skipped,
            "table_md": compare.table_md(table),
        }
        qf.replace_file(
            f"{out}.choice.json",
            data=json.dumps({k: info[k] for k in ("backend", "rows_per_s", "speedup_vs_library", "basis", "cpu")}, indent=1).encode(),
        )
        return info
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)


def main():
    qf.odd_names_are_printable()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument(
        "--data", required=True, help="real rows (npz as for prune/check); half of what is timed and what a PGO build learns from"
    )
    ap.add_argument("--decision", default="argmax")
    ap.add_argument("--average", choices=("output", "margin"), default="output")
    ap.add_argument("--cpu", type=int)
    ap.add_argument("--only", help="comma list: restrict the candidates")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--opt", action="append", default=[], help="compile option key=value for the generated-code backends")
    a = ap.parse_args()
    try:
        opts = qf.parse_options(a.opt)
        info = choose(
            a.model,
            a.out,
            a.data,
            decision=a.decision,
            average=a.average,
            cpu=a.cpu,
            only=a.only.split(",") if a.only is not None else None,
            keep_work=a.keep_work,
            compile_options=opts,
        )
    except (
        ValueError,
        OSError,
    ) as e:  # no backend can hold the model, none was exact, an option it cannot take or a path it cannot use: a message and exit 3
        print(f"choose.py: {qf.one_line(e)}", file=sys.stderr)
        sys.exit(3)
    print(info["table_md"])
    print(f"winner: {info['backend']} ({info['rows_per_s']:,} rows/s held-out, {info['speedup_vs_library']}x the library) -> {a.out}")


if __name__ == "__main__":
    main()
