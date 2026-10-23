#!/usr/bin/env python3
"""A model's structure as plain numpy arrays, read without running any prediction on it.

    structure.py MODEL OUT.npz

MODEL is ONE xgboost / LightGBM (text) / YDF (folder) model, pruned or not (a folder of k-fold files is refused: export one fold at a
time). OUT.npz is a numpy archive (`numpy.load(OUT.npz)`; no pickle) with the arrays that docs/REFERENCE.md lists. The model is read by
the same readers as every other command (`qf_formats`, `qf_common.ModelIR`): nothing is evaluated and no float is computed on it.

From Python: `structure.export(model)` returns the arrays as a dict, `structure.save(model, out)` writes the archive and returns the
dict. The semantics of a walk over the arrays (a row goes left when `x[feature] < threshold`, NaN the way `default_left` says, and so on)
are in the "Structure export" section of docs/REFERENCE.md.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import qf_common as qf
import qf_formats


def export(model) -> dict[str, np.ndarray]:
    """The structure of ONE model (a file, or a YDF folder) as arrays, in the order the readers hold its trees.

    Scalars (0-d arrays): `format` ("xgboost", "lightgbm", "ydf"), `kind` of the output transform ("identity", "sigmoid", "softmax", "exp"),
    `task` ("classification", "regression"), `n_features`, `n_groups`, `margin_scale` (LightGBM `sigmoid:s`: 1.0 for every other model).
    `base_margin`: float32 (n_groups,), the margin every row starts from.

    Per tree (n_trees = len(tree_group)): `tree_group` (int32, the output group the tree adds to), `source_tree` (int32, the 0-based index of
    the library's tree it comes from: the same as its own number except for a YDF random-forest classification tree, which is one tree per
    class here, so that `kept_indices` of `prune` count library trees) and `node_offset` (int64, n_trees + 1 entries: the nodes of tree t
    are `node_offset[t]:node_offset[t + 1]` of the per-node arrays).

    Per node, concatenated over the trees (node ids inside a tree start at 0 = the root; child ids are inside the same tree): `left`,
    `right` (int32, -1 at a leaf), `feature` (int32, -1 at a leaf), `threshold` (float32: the split threshold, or the leaf value at a
    leaf, exactly as QF reads it), `default_left` (bool, where NaN goes; False at a leaf), `zero_missing` (bool: the node also sends
    |x| <= 1e-35 the `default_left` way; LightGBM only), `leaf_index` (int32: the leaf's ordinal among the leaves of its tree in node
    order, -1 at an inner node).

    A model with categorical splits or linear leaves cannot be written as these arrays and is refused with a ValueError."""
    files = qf_formats.model_files(model)
    if len(files) != 1:
        raise ValueError(f"{model}: a folder of {len(files)} models; export one model file at a time")
    path = files[0]
    src = qf_formats.open_source(path)
    ir = src.model_ir()
    for t in ir.trees:
        if t.cat:
            raise ValueError(f"{path.name}: categorical splits cannot be exported as plain arrays (a split would need a set of categories)")
        if t.linear:
            raise ValueError(f"{path.name}: linear leaves cannot be exported as plain arrays (a leaf is a function of the row)")
    unit_of = getattr(src, "unit_of", None)
    n = len(ir.trees)
    sizes = np.array([len(t.left) for t in ir.trees], dtype=np.int64)
    leaf = np.concatenate([t.left == -1 for t in ir.trees])
    leaf_index = np.concatenate([np.where(t.left == -1, np.cumsum(t.left == -1) - 1, -1) for t in ir.trees])
    out: dict[str, np.ndarray] = {
        "format": np.array(src.fmt),
        "kind": np.array(src.spec.kind),
        "task": np.array(src.spec.task),
        "n_features": np.array(src.spec.n_features, np.int64),
        "n_groups": np.array(src.spec.groups, np.int64),
        "margin_scale": np.array(ir.margin_scale, np.float64),
        "base_margin": np.asarray(ir.base, np.float32).copy(),
        "tree_group": np.array([t.group for t in ir.trees], np.int32),
        "source_tree": np.arange(n, dtype=np.int32) if unit_of is None else np.asarray(unit_of, np.int32),
        "node_offset": np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
        "left": np.concatenate([t.left for t in ir.trees]).astype(np.int32),
        "right": np.concatenate([t.right for t in ir.trees]).astype(np.int32),
        "feature": np.where(leaf, -1, np.concatenate([t.feature for t in ir.trees])).astype(np.int32),
        "threshold": np.concatenate([t.value for t in ir.trees]).astype(np.float32),
        "default_left": np.concatenate([t.default_left for t in ir.trees]) & ~leaf,
        "zero_missing": np.concatenate([np.zeros(len(t.left), bool) if t.zero_missing is None else t.zero_missing for t in ir.trees])
        & ~leaf,
        "leaf_index": leaf_index.astype(np.int32),
    }
    return out


def save(model, out) -> dict[str, np.ndarray]:
    """`export(model)` written to `out`, a `.npz` file: a new file renamed over the name, never written through a link that sits there."""
    out = Path(out)
    if out.suffix != ".npz":
        raise ValueError(f"{out}: the output is a numpy archive; give a name ending in .npz")
    qf.output_file(out, "OUT")
    files = qf_formats.model_files(model)
    if any(os.path.realpath(f) == os.path.realpath(out) for f in files):  # realpath: a link loop is not an error here
        raise ValueError(f"{out}: that is the model file itself; the archive would overwrite it. Give another file name")
    arrays = export(model)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    qf.replace_file(out, data=buf.getvalue())
    return arrays


def summary(a: dict[str, np.ndarray], out) -> str:
    return (
        f"structure of a {a['format']} model: {len(a['tree_group']):,} trees, {len(a['left']):,} nodes, {int(a['n_groups'])} group(s), "
        f"{int(a['n_features'])} features, output {a['kind']}; written to {out}"
    )


def main():
    qf.odd_names_are_printable()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("out", help="a .npz file")
    a = ap.parse_args()
    try:
        arrays = save(a.model, a.out)
    except (ValueError, OSError) as e:  # a model or a path that cannot be used: one line
        raise SystemExit(f"structure.py: {qf.one_line(e)}") from None
    print(summary(arrays, a.out))


if __name__ == "__main__":
    main()
