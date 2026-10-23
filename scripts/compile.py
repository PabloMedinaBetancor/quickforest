#!/usr/bin/env python3
"""Compile an xgboost / LightGBM / YDF model (pruned or not; a file, or a directory of fold files averaged) to ONE artifact.

    compile.py MODEL(file|dir) OUT [--average output|margin] [--backend best|flat|auto|...] [--data DATA.npz --decision argmax]
               [--native | --target-cpu x86-64-v2|x86-64-v3|x86-64|...] [--shape fast|top]

The default backend, `best`, is the fastest build this machine can make: generated code (rustc builds it at opt-level 3, no link-time optimisation, 16 codegen units
per chunk: `--shape fast`, or `--shape top` = one codegen unit everywhere and no LTO across chunks; the jobs and units are planned against the free RAM), profile-guided
(`codegen-pgo`) when --data gives representative rows, else `codegen-block` with one notice line (also when no llvm-profdata of rustc's LLVM version is found); the flat format only when rustc is
missing, too little RAM is free or the model is beyond what generated code holds (a notice says why). OUT is a `.so` (or a flat file) and `OUT.choice.json` names the
loader: load it with `backends.load_artifact(OUT)`. The generated code is built for the portable x86-64-v2 CPU (`generic` on arm); `--native` (or `--target-cpu CPU`) builds for
this machine's CPU instead: the build only runs on that CPU type, and a library built for features the CPU lacks is refused at load time. `--backend flat` writes packed nodes plus the prebuilt evaluator library
next to it (load it with evaluator.FlatForest). The average rule says how folds combine: "output" = mean of each fold's probabilities,
"margin" = transform of the mean margin.
`--backend auto` (needs --data) builds every backend that can hold the model on this machine, times them on held-out real rows and
keeps the fastest exact one (choose.py); load the result with `backends.load_artifact(OUT)`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import backend_codegen
import backends
import qf_common as qf
import qf_formats

PROFILE_ROWS = 20000  # rows of the data file a profile-guided build runs its profile pass over


def profile_rows(data, path: Path, n_features: int | None = None) -> None:
    """Writes the rows of the data file that a profile-guided build runs over (at most PROFILE_ROWS, a seeded random pick) to `path` as an
    .npy. `n_features`: the model's, which the rows must match."""
    x = qf.load_data(data)["X"]
    if n_features is not None:
        qf.check_width(x, n_features, data)
    if len(x) > PROFILE_ROWS:
        x = x[np.sort(np.random.default_rng(0).choice(len(x), PROFILE_ROWS, replace=False))]
    np.save(path, np.ascontiguousarray(x, dtype=np.float32))


def best_backend(forest: qf.Forest, have_data: bool) -> tuple[str, str | None]:
    """(backend, notice) of the `best` default: generated code with a profile (`codegen-pgo`) when representative rows are given, `codegen-block`
    without them, the flat format only where generated code cannot be built (the notice names why). The notice is None when nothing was given up."""
    try:
        backend_codegen.toolchain_bin()
    except backend_codegen.ToolchainMissing:
        return "flat", (
            "rustc not found: compiled to the flat format, which is slower than generated code; install rustc (apt install rustc, or rustup) "
            "or set RUST_DIR for the fast build"
        )
    import choose

    if forest.n_nodes > choose.MAX_CODEGEN_NODES:
        return (
            "flat",
            f"{forest.n_nodes:,} nodes is above the {choose.MAX_CODEGEN_NODES:,} generated code was built for: compiled to the flat format",
        )
    if (depth := backend_codegen.too_deep(forest)) is not None:
        return (
            "flat",
            f"a tree is {depth:,} levels deep, above the {backend_codegen.MAX_TREE_DEPTH:,} generated code holds: compiled to the flat format",
        )
    free = backend_codegen.available_ram_mb({})
    if free is not None and not backend_codegen.plan_build([backend_codegen.MIN_CHUNK_NODES], 1, 1, free):
        return (
            "flat",
            f"only {free:,.0f} MB of RAM are free, too little to build generated code without swapping: compiled to the flat format",
        )
    if not have_data:
        return "codegen-block", (
            "no --data: the profile-guided build (codegen-pgo, usually faster) needs representative rows; built codegen-block. "
            "Give --data with real rows to get it"
        )
    try:
        backend_codegen.llvm_profdata()
    except backend_codegen.ToolchainMissing as e:
        return (
            "codegen-block",
            f"{getattr(e, 'why', 'llvm-profdata not found')}: built codegen-block, without the profile-guided build; "
            f"{backend_codegen.PROFDATA_HINT}",
        )
    return "codegen-pgo", None


def resolve_best(model, average: str = "output", float32_sums: bool = False, have_data: bool = True) -> tuple[str, str | None]:
    """`best_backend` for a model on disk (`optimize` names its artifact after the backend, so it asks before compiling)."""
    return best_backend(qf.load_forest(qf_formats.model_files(model), average, float32_sums), have_data)


def compile_model(model, out, backend: str = "best", average: str = "output", **options) -> dict:
    out = Path(out)
    qf.require_utf8_path(out, "the artifact's")
    qf.refuse_to_overwrite(out, qf_formats.model_files(model))
    if out.is_dir():
        raise ValueError(f"{out} is a folder: give the file name of the artifact")
    if not out.parent.is_dir():
        raise ValueError(f"{out.parent}: no such folder for the artifact")
    if not os.access(out.parent, os.W_OK | os.X_OK):  # the artifact is written as a new file in its folder: found now, not after the build
        raise ValueError(f"{out.parent}: the folder is not writable, the artifact cannot be written there")
    requested = backend
    if backend == "auto":
        import choose

        if not options.get("data"):
            raise ValueError("--backend auto times the candidates on real rows: give --data")
        t = time.perf_counter()
        info = choose.choose(
            model,
            out,
            options["data"],
            decision=options.get("decision", "argmax"),
            average=average,
            cpu=options.get("cpu"),
            compile_options={
                **{k: bool(options.get(k)) for k in ("f64_leaves", "float32_sums")},
                **({"target_cpu": options["target_cpu"]} if options.get("target_cpu") else {}),
                **({"shape": options["shape"]} if options.get("shape") else {}),
            },
        )
        info.update(average=average, compile_seconds=round(time.perf_counter() - t, 1))
        return info
    be = None if backend == "best" else backends.get(backend)  # a mistyped backend is found before the model is read
    backend_codegen.resolve_shape(options)  # a mistyped --shape is found before the model is read
    if options.get("target_cpu") and backend in ("flat", "wave", "wave-avx2"):
        raise ValueError(f"--native and --target-cpu choose the CPU generated code is built for: {backend} is not generated code")
    if options.get("shape") and backend in ("flat", "wave", "wave-avx2"):
        raise ValueError(f"--shape chooses how generated code is built: {backend} is not generated code")
    t0 = time.perf_counter()
    forest = qf.load_forest(qf_formats.model_files(model), average, bool(options.get("float32_sums")))
    qf.require_rule(options.get("decision", "argmax"), forest)
    notice = None
    if backend == "best":
        backend, notice = best_backend(forest, bool(options.get("data")) or options.get("pgo_rows") is not None)
        be = backends.get(backend)
    missing = forest.requires - set(getattr(be, "supports", ()))
    if missing:
        raise ValueError(
            f"backend {be.name} cannot compile this model: it needs {sorted(missing)} (qf_common.Forest.requires). "
            "Use a backend that holds them (flat, or codegen-block; docs/REFERENCE.md, 'compile: backends') or the library's own predictor."
        )
    t1 = time.perf_counter()
    # a profile-guided build learns which branches are taken from real rows; when `best` chose it (here, or `optimize`, which names the
    # artifact before the build and says so with pgo_fallback) and the rows are too few or too unlike one another to be the workload, it builds
    # codegen-block and says why in the notice. Asked for by name, it fails with that line.
    fallback = requested == "best" or bool(options.pop("pgo_fallback", False))
    try:
        if getattr(be, "pgo", False) and options.get("pgo_rows") is None:
            if not options.get("data"):
                raise ValueError(f"--backend {be.name} runs a profile pass over real rows: give --data")
            with tempfile.TemporaryDirectory() as tmp:
                profile_rows(options["data"], Path(tmp) / "profile.npy", forest.n_features)
                info = be.compile(forest, out, {**options, "pgo_rows": str(Path(tmp) / "profile.npy")})
        else:
            info = be.compile(forest, out, options)
    except backend_codegen.ProfileUnreliable as e:
        if not fallback:
            raise
        be = backends.get("codegen-block")
        notice = f"{notice}; {e}; built codegen-block" if notice else f"{e}; built codegen-block"
        info = be.compile(forest, out, {k: v for k, v in options.items() if k != "pgo_rows"})
    choice = Path(f"{out}.choice.json")
    choice.unlink(missing_ok=True)  # an earlier `--backend auto` build of this path chose a loader for ITS file
    if requested == "best":  # the loader comes from what was built, not from the suffix the output happens to have
        qf.replace_file(choice, data=json.dumps({"backend": be.name}).encode())
    info.update(backend=be.name, average=average, parse_seconds=round(t1 - t0, 1), compile_seconds=round(time.perf_counter() - t1, 1))
    if requested == "best" and out.suffix in (".flat", ".so") and out.suffix != be.suffix:
        # the loader comes from what was built, so the file still loads by `load_artifact`; a name that says the other kind misleads a reader by suffix
        said = (
            f"{out.name} holds generated code, not the flat format its name says; give --backend flat for the flat format"
            if be.suffix == ".so"
            else f"{out.name} holds the flat format, not the generated code its name says; give --backend codegen-block for generated code"
        )
        notice = f"{notice}; {said}" if notice else said
    if notice:
        info["notice"] = notice
    return info


def main():
    qf.odd_names_are_printable()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--average", choices=("output", "margin"), default="output")
    ap.add_argument(
        "--backend",
        default="best",
        help="best (default: generated code, profile-guided with --data), flat, wave, wave-avx2, codegen-block, codegen-pgo, auto, or path.py:Class",
    )
    ap.add_argument(
        "--data", help="real rows for the profile-guided build (the default with --data, and codegen-pgo) and for --backend auto"
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
    ap.add_argument("--decision", default="argmax")
    ap.add_argument(
        "--f64-leaves",
        action="store_true",
        help="store LightGBM / YDF leaves in double where float32 would round them (format 6 records / double literals in generated code): "
        "with predict(dtype=np.float64) the answer then carries no float32 rounding at all; the default keeps files byte-identical",
    )
    ap.add_argument(
        "--float32-sums",
        action="store_true",
        help="YDF only: add the leaves in float32 in tree order from the initial prediction, as YDF's own predictor does "
        "(its answer is then "
        "reproduced bit for bit; the default double sums differ from it by YDF's own float32 rounding noise)",
    )
    ap.add_argument("--no-lib", action="store_true", help="do not copy the evaluator library next to the artifact")
    a = ap.parse_args()
    try:
        info = compile_model(
            a.model,
            a.out,
            a.backend,
            a.average,
            no_lib=a.no_lib,
            target_cpu=backend_codegen.cpu_option(a.native, a.target_cpu),
            shape=a.shape,
            data=a.data,
            decision=a.decision,
            f64_leaves=a.f64_leaves,
            float32_sums=a.float32_sums,
        )
    except (
        ValueError,
        OSError,
        backend_codegen.ToolchainMissing,
    ) as e:  # a refusal (this backend or option cannot hold the model, a path cannot be used, or rustc is not installed): a message, exit 3
        print(f"compile.py: {qf.one_line(e)}", file=sys.stderr)
        sys.exit(3)
    parsed = f"parse {info['parse_seconds']} s, " if "parse_seconds" in info else ""  # --backend auto parses inside each candidate's build
    cpu = f", target {info['target_cpu']}" if "target_cpu" in info else ""
    cpu += f", shape {info['shape']}" if "shape" in info else ""
    if info.get("notice"):
        print(f"compile.py: note: {info['notice']}", file=sys.stderr)
    print(
        f"{a.out}: {info['bytes'] / 1e6:.2f} MB, {info['trees']:,} trees, {info['nodes']:,} nodes ({info['backend']}{cpu}, average "
        f"{a.average}); {parsed}compile {info['compile_seconds']} s"
    )


if __name__ == "__main__":
    main()
