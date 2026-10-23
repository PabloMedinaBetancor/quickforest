"""Writes the small models of tests/fixtures and the SHA-256 of the flat file the compiler makes of each (expected.json).
expected.json was made before the linear leaves and the base margin existed: test_linear.py checks that today's compiler still writes the
same bytes for those models (today xgboost writes format 4 and LightGBM with categorical / zero-as-missing nodes format 5).
Run: python3 -s make_fixtures.py   (needs a Python with xgboost and lightgbm; do NOT rerun it with a changed compiler)."""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import backend_flat
import qf_common as qf

rng = np.random.default_rng(20)
n, f = 1500, 7
X = rng.normal(size=(n, f)).astype(np.float32)
X[:, 3] = rng.integers(0, 6, n)
X[rng.random((n, f)) < 0.06] = np.nan
Xz = np.where(rng.random((n, f)) < 0.12, 0.0, X).astype(np.float32)
y = np.nan_to_num(X[:, 0]) * 2 + np.nan_to_num(X[:, 1]) ** 2 + (X[:, 3] == 2) + rng.normal(size=n) * 0.2
yc = (np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 2]) * 0.5 > 0).astype(int) + (np.nan_to_num(X[:, 1]) > 1)
np.save(HERE / "x.npy", X[:300])

b = xgb.train({"objective": "reg:squarederror", "max_depth": 4, "eta": 0.2, "nthread": 1, "seed": 1}, xgb.DMatrix(X, y), 30)
b.save_model(str(HERE / "xgb_reg.json"))
b = xgb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 3, "eta": 0.3, "nthread": 1, "seed": 1}, xgb.DMatrix(X, yc), 12)
b.save_model(str(HERE / "xgb_multi.ubj"))
p = dict(
    objective="regression",
    num_leaves=9,
    learning_rate=0.1,
    verbose=-1,
    num_threads=1,
    min_data_in_leaf=20,
    zero_as_missing=True,
    deterministic=True,
)
m = lgb.train(p, lgb.Dataset(Xz, y, categorical_feature=[3]), 25)
m.save_model(str(HERE / "lgb_ext.txt"))
p = dict(objective="regression", num_leaves=7, learning_rate=0.1, verbose=-1, num_threads=1, min_data_in_leaf=20, deterministic=True)
m = lgb.train(p, lgb.Dataset(X, y), 20)
m.save_model(str(HERE / "lgb_plain.txt"))

exp = {}
with tempfile.TemporaryDirectory() as td:
    for name in ("xgb_reg.json", "xgb_multi.ubj", "lgb_ext.txt", "lgb_plain.txt"):
        fr = qf.load_forest([HERE / name], "output")
        blob = backend_flat.pack(fr, lambda bb: 0)  # the checksum is irrelevant to the comparison below: hash the body
        exp[name] = hashlib.sha256(blob).hexdigest()
(HERE / "expected.json").write_text(json.dumps(exp, indent=1))
print(exp)
