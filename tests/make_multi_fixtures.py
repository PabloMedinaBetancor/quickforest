"""Generator of the QF 1.2 multi-output fixtures (tests/fixtures/multi/). Owner: W0-FIX.

    python3 -s tests/make_multi_fixtures.py --group sklearn|xgboost|ydf|k1 [--out DIR] [--qf11 DIR]

Groups
  sklearn   the sklearn models of DISENO 5.5 (RF regressor K = 1, 2, 5, 16; ET K = 3; DT K = 2; RF classifier K = 2 with class sets (3, 2)
            and K = 1 with 4 classes; a NaN-trained RF), each as <name>.joblib (test-only: QF itself never reads a pickle) plus
            <name>.npz with the rows and the library's own predictions (`predict`, and `proba_<k>` for classifiers).
  xgboost   multi_output_tree models: reg:squarederror K = 2, 5; binary:logistic multilabel K = 2; multi:softprob 3 classes; K = 1 under
            multi_output_tree. <name>.ubj + <name>.npz (`margin`, `predict` from booster.predict).
  ydf       the bench ydf-rf model (3 classes) and a binary RF, as YDF folders + <name>.npz (`predict` = model.predict).
  k1        the 1.1 byte-identity files and the K = 1 prune report pins. MUST be built with --qf11 = a throwaway export of tag v1.1
            (git archive v1.1 arboles-eficientes), never from 1.2 code: it trains the K = 1 models with the libraries and compiles/prunes
            them with that tree's scripts. Writes k1_*.flat (flat files), k1_*.sha256 and prune_pin_*.json.

Everything is seeded and trained on ONE thread, so a rerun on the same library versions gives the same files. Library versions go in
manifest.json. Models and files over 3 MB are listed in .gitignore by `--group` runs (regenerate them; README.md has their sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bench"))
import common as c

F = 42  # features, as the bench full size
ROWS_TRAIN, ROWS_PRED = 2000, 2000


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(p).read_bytes())
    return h.hexdigest()


def sha256_dir(p: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(p).rglob("*")):
        if f.is_file():
            h.update(str(f.relative_to(p)).encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def features(rows: int, seed: int, nan: float = 0.0) -> np.ndarray:
    """Float32 features: normals, a block of small integers (ties on thresholds), optionally NaNs."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, F)).astype(np.float32)
    x[:, F // 2 :: 5] = rng.integers(0, 12, size=x[:, F // 2 :: 5].shape).astype(np.float32)
    if nan:
        x[rng.random(x.shape) < nan] = np.nan
    return x


def targets(x: np.ndarray, k: int, seed: int) -> np.ndarray:
    """(rows, k) float64 regression targets: non-linear in a few features, different per output, with noise."""
    rng = np.random.default_rng(seed)
    z = np.nan_to_num(x)
    w = rng.normal(size=(F, k))
    t = np.tanh(z @ w * 0.3) * 3 + np.sin(z[:, :1] * z[:, 1:2]) + 0.5 * (z[:, 2:3] > 0.3) * z[:, 3:4]
    return (t + rng.normal(size=t.shape) * 0.3) * np.arange(1, k + 1) ** 0.5 + np.arange(k)  # outputs in different scales


def class_targets(x: np.ndarray, n_classes: list[int], seed: int) -> np.ndarray:
    """(rows, K) integer labels, output k has n_classes[k] classes."""
    rng = np.random.default_rng(seed)
    z = np.nan_to_num(x)
    cols = []
    for ck in n_classes:
        s = np.tanh(z @ rng.normal(size=(F, ck)) * 0.3) + rng.normal(size=(len(x), ck)) * 0.4
        cols.append(s.argmax(1))
    return np.stack(cols, 1).astype(np.int64)


# ---------------------------------------------------------------------------------------------------------------------- sklearn
def make_sklearn(out: Path, manifest: dict) -> None:
    import joblib
    import sklearn
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    manifest["sklearn"] = sklearn.__version__
    rf = dict(n_estimators=300, max_depth=8, n_jobs=1, random_state=11)
    specs = []
    for k in (1, 2, 5, 16):
        specs.append((f"rf_reg_k{k}", RandomForestRegressor(**rf), "reg", k, 0.0))
    specs.append(("et_reg_k3", ExtraTreesRegressor(**rf), "reg", 3, 0.0))
    specs.append(("dt_reg_k2", DecisionTreeRegressor(max_depth=8, random_state=11), "reg", 2, 0.0))
    specs.append(("rf_clf_k2_c3_2", RandomForestClassifier(**rf), "clf", [3, 2], 0.0))
    specs.append(("rf_clf_k1_c4", RandomForestClassifier(**rf), "clf", [4], 0.0))
    specs.append(("rf_reg_k2_nan", RandomForestRegressor(**rf), "reg", 2, 0.08))
    for name, est, kind, k, nan in specs:
        xt, xp = features(ROWS_TRAIN, 100, nan), features(ROWS_PRED, 101, nan)
        if kind == "reg":
            yt = targets(xt, k, 200)
            est.fit(xt, yt if k > 1 else yt[:, 0])
            pred = {"predict": est.predict(xp)}
        else:
            yt = class_targets(xt, k, 200)
            est.fit(xt, yt if len(k) > 1 else yt[:, 0])
            pr = est.predict_proba(xp)
            pr = pr if isinstance(pr, list) else [pr]
            pred = {"predict": est.predict(xp)} | {f"proba_{i}": p for i, p in enumerate(pr)}
        joblib.dump(est, out / f"{name}.joblib", compress=3)
        np.savez_compressed(out / f"{name}.npz", x=xp, **pred)
        print("sklearn", name, flush=True)
    # a model of the wrong kind for the refusal tests (GradientBoosting is refused by the exporter)
    from sklearn.ensemble import GradientBoostingRegressor

    xt = features(500, 100)
    joblib.dump(GradientBoostingRegressor(n_estimators=10, random_state=1).fit(xt, targets(xt, 1, 200)[:, 0]), out / "refused_gbr.joblib")


# --------------------------------------------------------------------------------------------------------------------- xgboost
def xgb_params(**kw) -> dict:
    return {"tree_method": "hist", "nthread": 1, "verbosity": 0, "seed": 1, "max_depth": 4, "eta": 0.3, **kw}


def train_xgb_multi(x, y, params, rounds, path: Path, xp: np.ndarray) -> None:
    import xgboost as xgb

    b = xgb.train(params, xgb.DMatrix(x, label=y), rounds)
    b.save_model(str(path))
    b = xgb.Booster(model_file=str(path))  # predictions of the file as saved
    d = xgb.DMatrix(xp)
    np.savez_compressed(path.with_suffix(".npz"), x=xp, margin=b.predict(d, output_margin=True), predict=b.predict(d))


def make_xgboost(out: Path, manifest: dict) -> None:
    import xgboost as xgb

    manifest["xgboost"] = xgb.__version__
    xt, xp = features(ROWS_TRAIN, 100, 0.04), features(ROWS_PRED, 101, 0.04)
    mot = dict(multi_strategy="multi_output_tree")
    for k in (2, 5):
        train_xgb_multi(xt, targets(xt, k, 200), xgb_params(objective="reg:squarederror", **mot), 20, out / f"xgb_vec_reg_k{k}.ubj", xp)
    ylab = (targets(xt, 2, 200) > np.median(targets(xt, 2, 200), 0)).astype(np.float32)  # multilabel: 2 independent binary columns
    train_xgb_multi(xt, ylab, xgb_params(objective="binary:logistic", **mot), 20, out / "xgb_vec_multilabel_k2.ubj", xp)
    train_xgb_multi(
        xt,
        class_targets(xt, [3], 200)[:, 0],
        xgb_params(objective="multi:softprob", num_class=3, **mot),
        20,
        out / "xgb_vec_softprob_c3.ubj",
        xp,
    )
    train_xgb_multi(xt, targets(xt, 1, 200)[:, 0], xgb_params(objective="reg:squarederror", **mot), 20, out / "xgb_mot_k1.ubj", xp)
    # same K = 5 data under the default strategy, for the "unchanged" K scalar trees path
    train_xgb_multi(
        xt,
        targets(xt, 5, 200),
        xgb_params(objective="reg:squarederror", multi_strategy="one_output_per_tree"),
        20,
        out / "xgb_oopt_reg_k5.ubj",
        xp,
    )
    print("xgboost done", flush=True)


# ------------------------------------------------------------------------------------------------------------------------- ydf
def make_ydf(out: Path, manifest: dict) -> None:
    import ydf

    manifest["ydf"] = ydf.__version__
    x, y = c.make_data(6000, F, 3, 7)  # the bench ydf-rf data (bench_compile full shape: 42 features, depth 8)
    xp = x[:ROWS_PRED]
    for name, classes, trees, depth, yy in (("ydf_rf_c3", 3, 300, 8, y), ("ydf_rf_binary", 2, 100, 6, (y > 0).astype(np.int64))):
        folder = out / name
        c.train_ydf(x, yy, folder, classes, trees, depth, 7, forest=True)
        m = ydf.load_model(str(folder))
        np.savez_compressed(out / f"{name}.npz", x=xp, predict=m.predict({f"f{i}": xp[:, i] for i in range(F)}))
        print("ydf", name, flush=True)


# -------------------------------------------------------------------------------------------------------------------- 1.1 files
def strip_volatile(o):
    """Report minus what changes between runs: timings and absolute paths."""
    if isinstance(o, dict):
        return {
            k: strip_volatile(v) for k, v in o.items() if not (k == "seconds" or k.endswith("_seconds") or k in ("started", "version_time"))
        }
    if isinstance(o, list):
        return [strip_volatile(v) for v in o]
    return o


def make_k1(out: Path, manifest: dict, qf11: Path) -> None:
    """Trains the K = 1 models, compiles them to flat files and prunes them with the 1.1 tree `qf11`; stores the bytes and the reports."""
    import xgboost as xgb

    scripts = qf11 / "scripts"
    assert (scripts / "compile.py").exists() and not (scripts / "qf_sklearn.py").exists(), "--qf11 must be an export of tag v1.1"
    manifest["qf11_tree"] = "git archive v1.1 arboles-eficientes"
    manifest["qf11_commit"] = subprocess.run(
        [os.environ.get("G", "git"), "-C", str(HERE.parents[2]), "rev-parse", "v1.1"], capture_output=True, text=True
    ).stdout.strip()
    work = out / "_k1_work"
    work.mkdir(exist_ok=True)
    xt = features(ROWS_TRAIN, 100, 0.04)
    yt = targets(xt, 1, 200)[:, 0]
    # (a) K = 1 xgboost under multi_output_tree (size_leaf_vector 1) and the same data under one_output_per_tree
    models = {}
    for tag, strat in (("mot", "multi_output_tree"), ("oopt", "one_output_per_tree")):
        p = work / f"xgb_{tag}_k1.ubj"
        xgb.train(xgb_params(objective="reg:squarederror", multi_strategy=strat), xgb.DMatrix(xt, label=yt), 20).save_model(str(p))
        models[f"k1_xgb_{tag}"] = p
    # (b) binary YDF random forest
    xb, yb = c.make_data(6000, F, 3, 7)
    fb = work / "ydf_rf_binary"
    c.train_ydf(xb, (yb > 0).astype(np.int64), fb, 2, 100, 6, 7, forest=True)
    models["k1_ydf_rf_binary"] = fb
    # (c) multi-class YDF RF as 1.1 compiles it (K copies of each tree): the bit-for-bit reference of the vector path
    fm = work / "ydf_rf_c3"
    c.train_ydf(xb, yb, fm, 3, 300, 8, 7, forest=True)
    models["k1_ydf_rf_c3_replicated"] = fm
    data = work / "data.npz"
    c.save_selection(data, xt)
    for name, model in models.items():
        dest = out / f"{name}.flat"
        r = subprocess.run(
            [sys.executable, "-s", str(scripts / "compile.py"), str(model), str(dest), "--backend", "flat"], capture_output=True, text=True
        )
        if r.returncode:
            raise SystemExit(f"{name}: 1.1 compile failed: {r.stdout[-400:]} {r.stderr[-400:]}")
        (out / "libtreeflat-x86_64.so").unlink(missing_ok=True)  # the evaluator library the flat compile writes beside its output
        (out / f"{name}.sha256").write_text(sha256(dest) + "\n")
        print("k1", name, sha256(dest)[:16], flush=True)
    # the model files the compiles came from, so a test can compile them with 1.2 code (small ones only)
    for name in ("k1_xgb_mot", "k1_xgb_oopt"):
        (out / f"{name}_model.ubj").write_bytes(models[name].read_bytes())
    # 1.1's K = 1 prune reports: regression (raw and log) on the xgboost model, classification on the binary RF
    sel = work / "prune_data.npz"
    np.savez(sel, X=xt[:1500], y=yt[:1500].astype(np.float64))
    for tag, extra, model in (
        ("reg_raw", ["--regression", "raw", "--keep", "30%"], models["k1_xgb_oopt"]),
        ("reg_log", ["--regression", "log", "--keep", "30%"], models["k1_xgb_oopt"]),
        ("clf_binary_rf", ["--keep", "30%", "--decision", "argmax"], models["k1_ydf_rf_binary"]),
    ):
        dst = work / f"prune_{tag}"
        d = sel if tag.startswith("reg") else work / "prune_data_clf.npz"
        if tag == "clf_binary_rf":
            np.savez(d, X=xb[:3000])
        r = subprocess.run(
            [sys.executable, "-s", str(scripts / "prune.py"), str(model), str(d), str(dst), *extra], capture_output=True, text=True
        )
        if r.returncode or not (dst / "report.json").exists():
            raise SystemExit(f"prune pin {tag} failed: {r.stdout[-400:]} {r.stderr[-400:]}")
        rep = strip_volatile(json.loads((dst / "report.json").read_text()))
        (out / f"prune_pin_{tag}.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
        print("prune pin", tag, flush=True)


def write_manifest(out: Path, manifest: dict) -> None:
    p = out / "manifest.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(manifest)
    old["files"] = {
        f.name: sha256(f) if f.is_file() else sha256_dir(f)
        for f in sorted(out.iterdir())
        if f.name not in ("manifest.json", "_k1_work", ".gitignore", "README.md")
    }
    p.write_text(json.dumps(old, indent=1, sort_keys=True) + "\n")


def write_gitignore(out: Path, limit: int = 3_000_000) -> None:
    big = []
    for f in sorted(out.iterdir()):
        size = f.stat().st_size if f.is_file() else sum(g.stat().st_size for g in f.rglob("*") if g.is_file())
        if size > limit and f.name not in ("manifest.json",):
            big.append(f.name + ("/" if f.is_dir() else ""))
    (out / ".gitignore").write_text("_k1_work/\n" + "".join(b + "\n" for b in big))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--group", required=True, choices=("sklearn", "xgboost", "ydf", "k1"))
    ap.add_argument("--out", default=str(HERE / "fixtures" / "multi"))
    ap.add_argument("--qf11", help="group k1: a throwaway export of tag v1.1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    manifest: dict = {"python": sys.version.split()[0], "numpy": np.__version__}
    if a.group == "sklearn":
        make_sklearn(out, manifest)
    elif a.group == "xgboost":
        make_xgboost(out, manifest)
    elif a.group == "ydf":
        make_ydf(out, manifest)
    else:
        if not a.qf11:
            raise SystemExit("group k1 needs --qf11 DIR (an export of tag v1.1)")
        make_k1(out, manifest, Path(a.qf11).resolve())
    write_gitignore(out)
    write_manifest(out, manifest)


if __name__ == "__main__":
    main()
