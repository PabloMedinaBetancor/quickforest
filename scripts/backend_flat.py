"""Default compile backend: the forest as one flat file of packed 8-byte nodes (format in rust/src/lib.rs), evaluated by the
Rust cdylib. Compile needs the evaluator's library only for the checksum."""

from __future__ import annotations

import struct
from pathlib import Path

import evaluator
import numpy as np
import qf_common as qf
from qf_common import AVERAGE, FEATURES, OBJECTIVE, Forest, Tree

FLAG_SUM64, FLAG_EXT, FLAG_LINEAR, FLAG_LATE = 1, 2, 4, 8
# The format numbers (header byte 8; rust/src/lib.rs). 1 to 3 were the layout whose checksum left the header out: the loader refuses them.
FORMAT_XGB, FORMAT_DOUBLE, FORMAT_LINEAR = 4, 5, 6
CHECKSUM_AT = 40  # the checksum is that of the whole file with these eight bytes read as zero
REC_DTYPE = np.dtype([("c", "<f8"), ("fb", "<f8"), ("off", "<u4"), ("cnt", "<u4")])  # one linear leaf: 24 bytes
COEF_DTYPE = np.dtype([("f", "<u4"), ("pad", "<u4"), ("c", "<f8")])  # one coefficient: 16 bytes
MAX_CATEGORY = 1 << 24


def needs_ext(forest: Forest) -> bool:
    return any(t.cat or (t.zero_missing is not None and t.zero_missing.any()) for m in forest.models for t in m.trees)


def inexact_leaves(tree: Tree) -> np.ndarray:
    """Per node: a leaf whose double value is not the float32 the default compile stores (an f64-leaf compile keeps it in double)."""
    if tree.leaf64 is None:
        return np.zeros(len(tree.left), bool)
    v = tree.leaf64
    return (tree.left == -1) & (v != v.astype(np.float32).astype(np.float64))


def needs_linear(forest: Forest, exact: bool = False) -> bool:
    """The linear format (records after the node table): linear leaves, or, with `exact` (f64 leaves), leaves that float32 would round."""
    return any(t.linear or (exact and inexact_leaves(t).any()) for m in forest.models for t in m.trees)


def flatten(
    tree: Tree, n_features: int, ext: bool = False, cats: list | None = None, lin: tuple | None = None, exact: bool = False
) -> np.ndarray:
    """The tree's nodes depth first (left child = next node) as packed u64 words. `ext`: extended nodes (13-bit feature, bit 61
    zero-as-missing, bit 62 categorical); the category sets of this tree are appended to the list `cats` of u32 words. `lin`: the lists
    (records, coefficients) that the linear leaves of this tree are appended to; such a leaf is bit 63 plus the index of its record.
    `exact`: a plain leaf that float32 would round becomes a record without coefficients (constant = fallback = the double value)."""
    left, right = tree.left.tolist(), tree.right.tolist()
    order, stack = [], [0]
    while stack:
        i = stack.pop()
        order.append(i)
        if left[i] != -1:
            stack.append(right[i])
            stack.append(left[i])
    order = np.asarray(order)
    n = len(order)
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n)
    internal = tree.left[order] != -1
    here = np.arange(n)
    dist = np.where(internal, pos[np.maximum(tree.right[order], 0)] - here, 0)
    if not np.array_equal(pos[tree.left[order][internal]], here[internal] + 1):
        raise ValueError("unexpected tree layout")
    feat = np.where(internal, tree.feature[order], 0).astype(np.int64)
    max_feat = 0x2000 if ext else 0x8000
    if internal.any() and (dist[internal].min() < 2 or dist.max() > 0xFFFF or feat.max() >= min(n_features, max_feat)):
        raise ValueError(
            "a tree is too large for the node format (right child more than 65,535 nodes away), or a feature index is out of range "
            f"(at most {(0x2000 if ext else MAX_FEATURES):,} features{' with categorical or zero-as-missing nodes' if ext else ''})"
        )
    bits = tree.value[order].view(np.uint32).astype(np.uint64)
    default_left = (tree.default_left[order] & internal).astype(np.uint64)
    word = bits | (dist.astype(np.uint64) << 32) | (feat.astype(np.uint64) << 48) | (default_left << 63)
    if ext:
        if tree.zero_missing is not None:
            word |= (tree.zero_missing[order] & internal).astype(np.uint64) << 61
        for k, c in (tree.cat or {}).items():
            c = np.asarray(c, dtype=np.int64)
            if len(c) and (c.min() < 0 or c.max() >= MAX_CATEGORY):
                raise ValueError(f"a categorical split uses category values outside 0..{MAX_CATEGORY - 1:,}")
            words = np.zeros(int(c.max()) // 32 + 1 if len(c) else 0, dtype=np.uint64)
            np.bitwise_or.at(words, c // 32, np.uint64(1) << (c % 32).astype(np.uint64))
            i = int(pos[k])
            word[i] = (word[i] & ~np.uint64(0xFFFFFFFF)) | np.uint64(len(cats))
            word[i] |= np.uint64(1) << np.uint64(62)
            cats.extend([len(words), *map(int, words)])
    if tree.linear:
        recs, coefs = lin
        for k, (const, fallback, feats, cf) in tree.linear.items():
            if not (np.isfinite(const) and np.isfinite(fallback) and np.isfinite(cf).all()):
                raise ValueError("a linear leaf has a non-finite constant or coefficient")
            if len(feats) and (feats.min() < 0 or feats.max() >= n_features):
                raise ValueError("a linear leaf uses a feature outside the model")
            word[int(pos[k])] = (np.uint64(1) << np.uint64(63)) | np.uint64(len(recs))
            recs.append((const, fallback, len(coefs), len(feats)))
            coefs.extend((int(f), 0, float(c)) for f, c in zip(feats, cf, strict=True))
    elif exact:
        recs, coefs = lin
        for i in np.flatnonzero(inexact_leaves(tree)[order]):
            v = float(tree.leaf64[order[i]])
            if not np.isfinite(v):
                raise ValueError("a leaf value is not a number")
            word[i] = (np.uint64(1) << np.uint64(63)) | np.uint64(len(recs))
            recs.append((v, v, len(coefs), 0))
    return word


MAX_FEATURES = 0x7FFF  # the evaluator's loader refuses a header with more
MAX_GROUPS = 1024  # the same for output groups (classes): rust/src/lib.rs MAX_GROUPS


def pack(forest: Forest, checksum, exact: bool = False) -> bytes:
    """The flat file. `exact` (the f64 leaves): the leaves of a LightGBM / YDF forest that float32 cannot hold are stored in double (linear-format
    records without coefficients); a forest whose leaves are all float32 numbers (xgboost, a YDF boosted model) packs byte-identically."""
    if forest.n_features > MAX_FEATURES:
        raise ValueError(f"the flat format holds at most {MAX_FEATURES:,} features (this model has {forest.n_features:,})")
    if forest.groups > MAX_GROUPS:
        raise ValueError(f"the flat format holds at most {MAX_GROUPS:,} output groups (classes); this model has {forest.groups:,}")
    exact = exact and forest.sum64  # float32 sums add float32 leaves: nothing to keep in double
    ext, linear = needs_ext(forest), needs_linear(forest, exact)
    if linear and not forest.sum64:
        raise ValueError("linear leaves need double sums (a LightGBM forest)")
    flags = (
        (FLAG_SUM64 if forest.sum64 else 0)
        | (FLAG_EXT if ext else 0)
        | (FLAG_LINEAR if linear else 0)
        | (FLAG_LATE if forest.late_margin and not forest.sum64 else 0)
    )
    cats: list = []
    lin: tuple = ([], [])
    folds, trees, words, count = [], [], [], 0
    for m in forest.models:
        folds.append((len(trees), len(m.trees)))
        for t in m.trees:
            w = flatten(t, forest.n_features, ext, cats, lin, exact)
            trees.append((count, t.group))
            words.append(w)
            count += len(w)
    scale = forest.margin_scale
    # see rust/src/lib.rs `margin_scale`: a record no leaf refers to, after all the others (-1: several different scales)
    if linear and scale != 1.0:
        lin[0].append((scale if scale == scale else -1.0, evaluator.margin_scale_marker(), 0, 0))
    body = b"".join(struct.pack("<II", a, b) for a, b in folds)
    base = np.concatenate([m.base for m in forest.models]).astype("<f4").tobytes()
    body += base + bytes(-len(base) % 8)
    body += b"".join(struct.pack("<II", r, g) for r, g in trees)
    body += np.concatenate(words).astype("<u8").tobytes()
    table = np.asarray(cats, dtype="<u4").tobytes()
    body += table + bytes(-len(table) % 8)
    if linear:
        body += np.array(lin[0], REC_DTYPE).tobytes() + np.array(lin[1], COEF_DTYPE).tobytes()
    # the xgboost format whenever the forest is expressible in it; the double format for double sums, exp, identity with groups, extensions;
    # the linear format for linear or exact leaves
    v2 = bool(flags) or forest.kind == "exp" or (forest.kind == "identity" and forest.groups > 1)
    head = b"TREEFLT1" + struct.pack(
        "<8I",
        FORMAT_LINEAR if linear else FORMAT_DOUBLE if v2 else FORMAT_XGB,
        forest.n_features,
        forest.groups,
        len(folds),
        len(trees),
        count,
        OBJECTIVE[forest.kind],
        AVERAGE[forest.average],
    )
    if linear:
        tail = struct.pack("<II", len(lin[0]), len(lin[1]))
    else:
        tail = struct.pack("<d", 0.0 if scale == 1.0 else scale)  # bytes 56..64: the margin scale (0 = 1, NaN: several different scales)
    tail = struct.pack("<II", flags, len(cats)) + tail
    assert len(head) == CHECKSUM_AT
    # the checksum covers the whole file, its own eight bytes read as zero: a damaged header is refused like a damaged body
    return head + struct.pack("<Q", checksum(head + bytes(8) + tail + body)) + tail + body


class FlatBackend:
    name = "flat"
    suffix = ".flat"
    # everything Forest.requires may list; the engines that read the same file but cannot run some of it set their own smaller set
    supports = FEATURES

    def compile(self, forest: Forest, out: Path, options: dict) -> dict:
        lib = evaluator.find_lib()
        blob = pack(forest, lambda b: evaluator.fnv1a(b, lib), bool(options.get("f64_leaves")))
        qf.replace_file(out, data=blob)
        if not options.get("no_lib"):
            qf.copy_unless_same(lib, out.parent / evaluator.LIB_NAME)  # a new inode: a process that has the old library mapped keeps it
        return {"artifact": str(out), "bytes": len(blob), "nodes": forest.n_nodes, "trees": forest.n_trees}

    def load(self, artifact: Path, **options):
        return evaluator.FlatForest(artifact)
