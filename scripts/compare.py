#!/usr/bin/env python3
"""Every compile backend on one model, side by side: exactness against the source library (xgboost, LightGBM or YDF: real rows, rows built
on the thresholds, identical decisions), bit-identity with the baseline walker, batch throughput against the library's own predict (both
on the same number of threads: every usable core by default, N with --threads N, 1 for the one-thread figures), compile time, artifact size, load time and memory. Each backend runs its load + test + timing in its own process, so a crash or
a leak in one cannot disturb another, and its memory is its own.

    compare.py MODEL(file|dir) --data DATA.npz [--backends flat,wave,wave-avx2,codegen,...|all] [--decision argmax] [--cpu N] [--threads N|all]
                [--work DIR] [--md OUT.md] [--json OUT.json] [--eq-rows 20000] [--random 10000] [--speed-rows 20000]

`compile --backend auto` calls `run_table` for every candidate that can hold the model and keeps the fastest exact one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import backends
import check
import evaluator
import qf_common as qf
import qf_formats

BASELINE = "flat"


def hwm_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


# ------------------------------------------------------------------------------------------------------------ worker
def worker(spec_path: str) -> None:
    """Runs in its own process: load the artifact, answer the test sets, time a batch. Writes the result as JSON."""
    spec = json.loads(Path(spec_path).read_text())
    be = backends.get(spec["backend"])
    r0, t0 = check.rss_mb(), time.perf_counter()
    pred = be.load(Path(spec["artifact"]))
    res = {"load_seconds": time.perf_counter() - t0, "engine_used": getattr(pred, "engine_used", None)}
    outs = {}
    for name, path in spec["sets"].items():
        x = np.load(path)
        t = time.perf_counter()
        outs[name] = pred.predict(x, spec["threads"])
        res[f"{name}_seconds"] = time.perf_counter() - t
        save_atomic(Path(spec["out_dir"]) / f"{spec['tag']}-{name}.npy", outs[name])
    res["rss_load_mb"] = check.rss_mb() - r0
    xs = np.load(spec["speed"])
    n = len(xs)
    t = time.perf_counter()
    pred.predict(xs[: min(n, 64)], spec["threads"])  # warm-up and a first timing
    probe = (time.perf_counter() - t) / min(n, 64)
    n = int(min(n, max(64, spec["budget"] / 4 / max(probe, 1e-9))))  # about budget/4 seconds a pass
    xs = np.ascontiguousarray(xs[:n])
    best, total = 1e30, 0.0
    for _ in range(int(spec["reps"])):
        t = time.perf_counter()
        pred.predict(xs, spec["threads"])
        dt = time.perf_counter() - t
        best, total = min(best, dt), total + dt
        if total > spec["budget"]:
            break
    res.update(speed_rows=n, rows_per_s=n / best, rss_peak_mb=hwm_mb() - r0, rss_after_mb=check.rss_mb() - r0)
    # the same forest on the rows a profile-guided build learned from, and on random rows built from the thresholds: a layout tuned on
    # the profile rows must not look fast only there. `rows_per_s` (held-out real rows, never seen by the build) is the figure to trust.
    for key, path in (spec.get("extra") or {}).items():
        xe = np.ascontiguousarray(np.load(path)[:n])
        be_, tot = 1e30, 0.0
        for _ in range(int(spec["reps"])):
            t = time.perf_counter()
            pred.predict(xe, spec["threads"])
            dt = time.perf_counter() - t
            be_, tot = min(be_, dt), tot + dt
            if tot > spec["budget"] / 2:
                break
        res[f"rows_per_s_{key}"] = len(xe) / be_
    qf.replace_file(spec["result"], data=json.dumps(res).encode())


# ------------------------------------------------------------------------------------------------------------ parent
def library_speed(files, xs, reps=2, threads=1) -> tuple[float, str]:
    """(rows/s on `threads` threads, library name) of the library that wrote the model (xgboost, LightGBM or YDF), on a fixed batch
    (`qf_formats.library_rate`), check.bench's way."""
    srcs = [qf_formats.open_source(f) for f in files]
    return qf_formats.library_rate(srcs, xs, reps, threads), srcs[0].fmt


def save_atomic(path: Path, a) -> None:
    """np.save through a temporary name: several builds can share one work folder without reading half a file."""
    qf.replace_via_temp(path, lambda tmp: np.save(tmp, a))


def cached(path: Path, make):
    if path.is_file():
        return np.load(path)
    a = make()
    save_atomic(path, a)
    return a


def model_fingerprint(files) -> str:
    """What a model IS: the names and the hash of the bytes of its files (a YDF model is a folder: every file in it). Size and mtime are
    not enough: a folder's mtime and size do not move when a file inside is rewritten, and a retrain of the same size restored with
    `rsync -a` keeps both. Hashing is about 0.4 s per 150 MB, against minutes of every step that follows."""
    h = hashlib.sha1()
    for f in map(Path, files):
        inner = sorted(q for q in f.rglob("*") if q.is_file()) if f.is_dir() else [f]
        for q in inner:
            d = hashlib.sha1()
            with open(q, "rb") as fh:
                while chunk := fh.read(1 << 20):
                    d.update(chunk)
            h.update(qf.utf8(f"{f.name}/{q.relative_to(f) if f.is_dir() else ''}:{q.stat().st_size}:{d.hexdigest()}\n"))
    return h.hexdigest()


def data_fingerprint(path) -> str:
    """What a data file IS: the hash of its bytes. Its name and size are not enough: `np.savez` of another array of the same shape writes a
    file of the same name and size, and the rows measured would be those of the file it replaced."""
    d = hashlib.sha1()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            d.update(chunk)
    return d.hexdigest()


def build_signature(files, average) -> str:
    """Which model and which averaging a compiled artifact belongs to: a build and a measure of different ones must not meet."""
    return f"{model_fingerprint(files)}:{average}"


def prepare(model, data, work, *, average, eq_rows, n_random, speed_rows, seed, log, decision="argmax"):
    """Test sets, the library's answers for them (cached), the rows to time and the rows a profile-guided build learns from."""
    if not Path(data).is_file():  # before anything is written: a missing data file is a refused argument, not a traceback
        raise ValueError(f"{data}: no such data file")
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    files = qf_formats.model_files(model)
    t0 = time.perf_counter()
    forest = qf.load_forest(files, average)
    qf.require_rule(decision, forest)
    log(f"parsed {forest.n_trees:,} trees, {forest.n_nodes:,} nodes in {time.perf_counter() - t0:.1f} s")
    stamp = model_fingerprint(files)  # a re-pruned or retrained model of the same name is another model
    sig = json.dumps([Path(model).name, stamp, data_fingerprint(data), average, eq_rows, n_random, speed_rows, seed])
    if (work / "prepared.json").is_file() and (work / "prepared.json").read_text() == sig:
        return files, forest  # the sets, the library's answers and the profile rows of an identical earlier call are all there
    for stale in work.glob("ref-*.npy"):  # answers of another model (or other sets): the cache is keyed by set name and length only
        stale.unlink()
    x_all = qf.load_data(data)["X"]
    qf.check_width(x_all, forest.n_features, data)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.permutation(len(x_all))[: min(eq_rows, len(x_all))])
    sets = {"data": np.ascontiguousarray(x_all[pick])}
    if n_random:
        sets["random"] = check.random_rows(forest, n_random, seed)
    for k, x in sets.items():
        cached(work / f"ref-{k}-{len(x)}.npy", lambda x=x: qf.reference_predict(files, x, average))
        save_atomic(work / f"x-{k}.npy", x)
    small_n = max(200, eq_rows // 40)
    for k in sets:
        save_atomic(work / f"x-{k}-small.npy", sets[k][:small_n])
    perm = rng.permutation(len(x_all))
    n_speed = min(speed_rows, max(1, len(x_all) // 2))  # a small data file keeps half of its rows for the profile
    speed_x = np.ascontiguousarray(x_all[np.sort(perm[:n_speed])])
    save_atomic(work / "x-speed.npy", speed_x)
    # profile-guided builds learn from other rows than the ones they are timed on (and, when the data allows, than the exactness sets)
    spare = np.setdiff1d(perm[n_speed:], pick)
    profile = spare[:20000] if len(spare) else perm[n_speed:][:20000]
    save_atomic(work / "x-pgo.npy", np.ascontiguousarray(x_all[profile]))
    qf.replace_file(work / "sets.json", data=json.dumps({k: len(x) for k, x in sets.items()}).encode())
    qf.replace_file(work / "prepared.json", data=qf.utf8(sig))
    return files, forest


def build_all(forest, names, work, *, compile_options=None, log=print, model_sig=None) -> list[dict]:
    """Compiles every backend into `work` (needs rustc for the generated-code ones); the result goes to build-NAME.json, with the
    signature of the model it was built from (`measure_all` refuses artifacts of another model)."""
    work = Path(work)
    rows = []
    for name in names:
        be = backends.get(name)
        art = work / f"{name}{be.suffix}"
        row = {"backend": name, "artifact": str(art), "model_sig": model_sig}
        try:
            t = time.perf_counter()
            info = be.compile(forest, art, {"pgo_rows": str(work / "x-pgo.npy"), **(compile_options or {})})
            row["compile_seconds"] = round(time.perf_counter() - t, 1)
            row["artifact_mb"] = round(Path(art).stat().st_size / 1e6, 2)
            row["lib_mb"] = round(evaluator.find_lib(art).stat().st_size / 1e6, 2) if be.suffix == ".flat" else 0.0
            row.update({k: v for k, v in info.items() if k in ("rustc_peak_mb", "note", "src_mb")})
        except Exception as e:
            row["status"] = f"compile failed: {str(e)[:160]}"
            log(f"{name}: {row['status']}")
        rows.append(row)
        # one file each: parallel builds share the folder
        qf.replace_file(work / f"build-{name}.json", data=json.dumps(row, indent=1).encode())
    return rows


def measure_all(
    files, forest, work, *, decision="argmax", average="output", cpu=None, small=(), budget=6.0, log=print, names=None, threads=1
) -> dict:
    """Runs each built artifact (in its own process, on this machine) against the library's cached answers and times it. Refuses artifacts
    that were built from another model (or averaged another way): they would be judged against this model's answers (NOT EXACT).
    `threads`: the thread count of the timing for the library and the compiled forest alike; None is every usable core of THIS machine."""
    threads = evaluator.usable_cpus() if threads is None else evaluator.clamp_threads(threads)
    work = Path(work)
    order = [n for n in (names or backends.BACKENDS) if (work / f"build-{n}.json").is_file()]
    built = [json.loads((work / f"build-{n}.json").read_text()) for n in order]
    want = build_signature(files, average)
    for row in built:
        if row.get("model_sig") is None:
            log(f"{row['backend']}: build-{row['backend']}.json has no model signature: cannot tell which model it is")
        elif row["model_sig"] != want:
            raise SystemExit(
                f"{work}: {row['backend']} was built from another model (or another --average) than {Path(files[0]).name}"
                f"{' and its folds' if len(files) > 1 else ''}; build again with --stage build, or measure with the model it was built from"
            )
    sets = json.loads((work / "sets.json").read_text())
    refs = {k: np.load(work / f"ref-{k}-{n}.npy") for k, n in sets.items()}
    exact = check.bit_exact(files, average, forest)
    speed_x = np.load(work / "x-speed.npy")
    t = time.perf_counter()
    lib_rate, library = library_speed(files, speed_x, threads=threads)
    log(f"{library} {threads} thread(s): {lib_rate:,.0f} rows/s ({time.perf_counter() - t:.0f} s)")
    # as check.py
    rule = qf.make_rule(decision, forest.groups, forest.kind, None, None, forest.task) if forest.task == "classification" else None
    rows, flat_out = [], {}
    for row in built:
        name = row["backend"]
        if "status" in row:
            rows.append(row)
            continue
        use_small = name in small
        spec = {
            "backend": name,
            "artifact": str(work / Path(row["artifact"]).name),  # a measure stage may run where the work folder was copied to
            "sets": {k: str(work / f"x-{k}{'-small' if use_small else ''}.npy") for k in sets},
            "speed": str(work / "x-speed.npy"),
            "extra": {"profile": str(work / "x-pgo.npy"), **({"random": str(work / "x-random.npy")} if "random" in sets else {})},
            "threads": threads,
            "budget": budget,
            "reps": 5,
            "out_dir": str(work),
            "tag": name,
            "result": str(work / f"res-{name}.json"),
        }
        qf.replace_file(work / f"spec-{name}.json", data=json.dumps(spec).encode())
        cmd = [sys.executable, "-I", str(HERE / "compare.py"), "--worker", str(work / f"spec-{name}.json")]
        if cpu is not None:
            cmd = ["taskset", "-c", str(cpu), *cmd]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if p.returncode != 0:
            row["status"] = f"run failed (exit {p.returncode}): {(p.stderr or p.stdout).strip()[-200:]}"
            rows.append(row)
            log(f"{name}: {row['status']}")
            continue
        res = json.loads((work / f"res-{name}.json").read_text())
        ok, worst, mism, bit = True, 0.0, 0, True
        for k in sets:
            got = np.load(work / f"{name}-{k}.npy")
            c = check.compare(k, refs[k][: len(got)], got, decision, rule, forest.groups, exact=exact)
            ok &= c["ok"]
            worst, mism = max(worst, c["max_err"]), mism + c["decision_mismatches"]
            row[f"rows_{k}"] = c["rows"]
            if name == BASELINE:
                flat_out[k] = got
            bit &= bool(k in flat_out and np.array_equal(got, flat_out[k][: len(got)]))
        row.update(
            status="ok" if ok else "NOT EXACT",
            bitwise_vs_flat=bit if flat_out else None,
            max_err=worst,
            decision_mismatches=mism,
            load_seconds=round(res["load_seconds"], 3),
            rss_load_mb=round(res["rss_load_mb"], 1),
            rss_peak_mb=round(res["rss_peak_mb"], 1),
            rows_per_s=round(res["rows_per_s"]),
            speedup=round(res["rows_per_s"] / lib_rate, 2),
            engine_used=res["engine_used"],
            rows_per_s_profile=round(res.get("rows_per_s_profile", 0)),
            rows_per_s_random=round(res.get("rows_per_s_random", 0)),
        )
        rows.append(row)
        log(
            f"{name}: {row['status']}, {row['rows_per_s']:,} rows/s ({row['speedup']}x), max err {worst:.1e}, bitwise vs flat "
            f"{row['bitwise_vs_flat']}"
        )
    return {
        "trees": forest.n_trees,
        "nodes": forest.n_nodes,
        "features": forest.n_features,
        "library": library,
        "threads": threads,  # both the library and the compiled forest were timed on this many threads
        "library_rows_per_s": round(lib_rate),  # the model's own library (xgboost, LightGBM or YDF)
        "cpu": cpu_name(),
        "rows": rows,
    }


def run_table(
    model,
    names,
    data,
    work,
    *,
    decision="argmax",
    average="output",
    cpu=None,
    eq_rows=20000,
    n_random=10000,
    speed_rows=20000,
    seed=15,
    budget=6.0,
    small=(),
    compile_options=None,
    stage="all",
    log=print,
    threads=1,
) -> dict:
    """`threads`: the thread count of the timing, for the library and for the compiled forest alike (1: the speed baseline).
    `stage`: "build" (compile every backend; needs rustc), "measure" (run what was built, on this machine), or "all".
    `small`: backends too slow for the big test sets: they get `eq_rows // 40` real and random rows."""
    qf.require_utf8_path(work, "the work folder's")
    qf.require_cpu(cpu)
    files, forest = prepare(
        model, data, work, average=average, eq_rows=eq_rows, n_random=n_random, speed_rows=speed_rows, seed=seed, log=log, decision=decision
    )
    if (compile_options or {}).get("float32_sums"):
        forest = qf.load_forest(files, average, True)
    if stage in ("all", "build"):
        build_all(forest, names, work, compile_options=compile_options, log=log, model_sig=build_signature(files, average))
    if stage == "build":
        return {}
    table = measure_all(
        files, forest, work, decision=decision, average=average, cpu=cpu, small=small, budget=budget, log=log, names=names, threads=threads
    )
    table["model"] = str(model)
    return table


def cpu_name(cpuinfo="/proc/cpuinfo") -> str:
    """The CPU as the kernel names it: `model name` (x86), `Model name` (some aarch64 kernels), `Hardware` (Android and boards; as
    bench/common.py reads it), else the interpreter's own guess."""
    try:
        for ln in Path(cpuinfo).read_text().splitlines():
            if ln.startswith(("model name", "Model name", "Hardware")):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def best_exact(table: dict) -> dict | None:
    ok = [r for r in table["rows"] if r.get("status") == "ok"]
    return max(ok, key=lambda r: r["rows_per_s"]) if ok else None


def table_md(table: dict) -> str:
    lib = table.get("library", "xgboost")
    head = (
        f"{table['model']}: {table['trees']:,} trees, {table['nodes']:,} nodes, {table['features']} features. {table['cpu']}. "
        f"{lib} {table.get('threads', 1)} thread(s), {qf_formats.BASELINE_ROWS:,}-row batches: {table['library_rows_per_s']:,} rows/s.\n\n"
        f"| variant | status | rows/s held-out | x {lib}, {table.get('threads', 1)} thread(s) | rows/s profile rows | rows/s random rows | max err | bitwise = flat "
        f"| compile s | size MB | load s | RSS MB (load / peak) |\n"
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|\n"
    )
    lines = []
    for r in table["rows"]:
        if "rows_per_s" not in r:
            lines.append(f"| {r['backend']} | {r['status']} | | | | | | | {r.get('compile_seconds', '')} | | | |")
            continue
        size = r["artifact_mb"] + r.get("lib_mb", 0)
        tag = r["backend"] + (f" [{r['engine_used']}]" if r["engine_used"] not in (None, 0) else "")
        lines.append(
            f"| {tag} | {r['status']} | {r['rows_per_s']:,} | {r['speedup']} | {r['rows_per_s_profile']:,} | {r['rows_per_s_random']:,} | "
            f"{r['max_err']:.1e} | {r['bitwise_vs_flat']} | "
            f"{r['compile_seconds']} | {size:.2f} | {r['load_seconds']} | {r['rss_load_mb']} / {r['rss_peak_mb']} |"
        )
    return head + "\n".join(lines) + "\n"


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2])
    qf.odd_names_are_printable()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--data", required=True)
    ap.add_argument("--backends", default="all")
    ap.add_argument("--decision", default="argmax")
    ap.add_argument("--average", choices=("output", "margin"), default="output")
    ap.add_argument("--cpu", type=int, help="pin each run to this logical CPU (taskset)")
    ap.add_argument(
        "--threads",
        type=qf.threads_arg,
        default=None,
        metavar="N|all",
        help="threads of the timing, for the library and the compiled forest alike: all (default: every usable core) or a whole number "
        "(1 gives the one-thread figures)",
    )
    ap.add_argument("--work", required=True)
    ap.add_argument("--md")
    ap.add_argument("--json")
    ap.add_argument("--eq-rows", type=qf.count_arg(1), default=20000)
    ap.add_argument("--random", type=qf.count_arg(0), default=10000)
    ap.add_argument("--speed-rows", type=qf.count_arg(1), default=20000)
    ap.add_argument("--small", default="", help="comma list of backends to test on fewer rows (slow ones)")
    ap.add_argument("--stage", choices=("all", "build", "measure"), default="all", help="build on one machine, measure on another")
    ap.add_argument(
        "--opt",
        action="append",
        default=[],
        help="compile option key=value (block, chunk_nodes, jobs, opt_level, codegen_units, keep_src, shape, target_cpu, ram_mb, f64_leaves), repeatable",
    )
    a = ap.parse_args()
    names = list(backends.BACKENDS)
    if a.backends != "all":
        names = a.backends.split(",")
    try:
        for flag, path in (("--md", a.md), ("--json", a.json)):
            if path:
                qf.output_file(path, flag)
        t = run_table(
            a.model,
            names,
            a.data,
            a.work,
            decision=a.decision,
            average=a.average,
            cpu=a.cpu,
            eq_rows=a.eq_rows,
            n_random=a.random,
            speed_rows=a.speed_rows,
            small=a.small.split(","),
            compile_options=qf.parse_options(a.opt),
            stage=a.stage,
            threads=a.threads,
        )
    except (ValueError, OSError) as e:  # an argument or a path that cannot be used: one line
        raise SystemExit(f"compare.py: {qf.one_line(e)}") from None
    if a.stage == "build":
        return
    md = table_md(t)
    print(md)
    for flag, path, data in (("--md", a.md, lambda: qf.utf8(md)), ("--json", a.json, lambda: json.dumps(t, indent=1).encode())):
        if path:
            try:
                qf.replace_file(path, data=data())
            except OSError as e:  # the folder was checked before the run; this is a write that failed since (a full disk, a folder removed)
                raise SystemExit(
                    f"compare.py: the table is printed above, but {flag} {path} could not be written: {qf.one_line(e)}"
                ) from None


if __name__ == "__main__":
    main()
