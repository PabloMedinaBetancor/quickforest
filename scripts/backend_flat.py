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
FLAG_VECTOR, FLAG_VEC64 = 16, 32  # format 7 only: vector leaves; the vector table holds float64
VECTOR_GROUP_U32 = 0xFFFF_FFFF  # the group field of a vector tree's record
# The format numbers (header byte 8; rust/src/lib.rs). 1 to 3 were the layout whose checksum left the header out: the loader refuses them.
FORMAT_XGB, FORMAT_DOUBLE, FORMAT_LINEAR, FORMAT_VECTOR = 4, 5, 6, 7
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
    tree: Tree,
    n_features: int,
    ext: bool = False,
    cats: list | None = None,
    lin: tuple | None = None,
    exact: bool = False,
    vec: list | None = None,
) -> np.ndarray:
    """The tree's nodes depth first (left child = next node) as packed u64 words. `ext`: extended nodes (13-bit feature, bit 61
    zero-as-missing, bit 62 categorical); the category sets of this tree are appended to the list `cats` of u32 words. `lin`: the lists
    (records, coefficients) that the linear leaves of this tree are appended to; such a leaf is bit 63 plus the index of its record.
    `exact`: a plain leaf that float32 would round becomes a record without coefficients (constant = fallback = the double value).
    `vec`: for a vector tree, the list the (leaves, groups) array of its leaf rows is appended to (float64 rows when `exact`, else float32), in the depth-first order of the
    leaf words: a leaf is the number of its row, the k-th vector leaf of the file holding row k (the loader checks exactly that)."""
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
    if tree.vector is not None:
        at = np.flatnonzero(~internal)
        word[at] = np.uint64(sum(map(len, vec))) + np.arange(len(at), dtype=np.uint64)
        rows = tree.vector64 if exact and tree.vector64 is not None else tree.vector
        vec.append(rows[order[at]])
    elif tree.linear:
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


def output_segments(forest: Forest) -> list[int] | None:
    """The output segments the file of a vector forest carries (its reader takes the number of outputs and their kind from them): the forest's
    own; else one width-1 segment per column for a regression with several outputs (identity) and for independent sigmoids (xgboost
    multilabel); else none, which is ONE output (random-forest class shares, a softmax)."""
    if forest.segments is not None:
        return forest.segments
    if forest.groups >= 2 and (forest.kind == "sigmoid" or (forest.kind == "identity" and forest.task != "classification")):
        return list(range(forest.groups + 1))
    return None


def scalar_segments(forest: Forest) -> list[int] | None:
    """The segment table of a forest of SCALAR trees: K width-1 regression columns for a one-tree-per-output identity forest (xgboost
    multi-target), so that the flat file names K outputs as the generated code does (`backend_codegen.file_segments`); none otherwise.
    A sigmoid over several groups needs vector trees: refused here in one line, not as the loader's bare "counts out of range"."""
    if forest.groups >= 2 and forest.kind == "sigmoid":
        raise ValueError(qf.MULTILABEL_REFUSAL.format(objective="a sigmoid objective", groups=forest.groups))
    if forest.groups >= 2 and forest.kind == "identity" and forest.task != "classification":
        if forest.margin_scale != 1.0 or any(m.margin_scale != 1.0 for m in forest.models):
            return None  # format 7 has no margin scale field: the table cannot share the file with one
        return list(range(forest.groups + 1))
    return None


def pack(forest: Forest, checksum, exact: bool = False) -> bytes:
    """The flat file. `exact` (the f64 leaves): the leaves of a LightGBM / YDF forest that float32 cannot hold are stored in double (linear-format
    records without coefficients); a forest whose leaves are all float32 numbers (xgboost, a YDF boosted model) packs byte-identically."""
    if forest.n_features > MAX_FEATURES:
        raise ValueError(f"the flat format holds at most {MAX_FEATURES:,} features (this model has {forest.n_features:,})")
    if forest.groups > MAX_GROUPS:
        raise ValueError(f"the flat format holds at most {MAX_GROUPS:,} output groups (classes); this model has {forest.groups:,}")
    exact = exact and forest.sum64  # float32 sums add float32 leaves: nothing to keep in double
    vector = any(t.vector is not None for m in forest.models for t in m.trees)
    segments = output_segments(forest) if vector else scalar_segments(forest)
    if vector:
        qf.check_vector(forest)  # one kind of tree, W == groups >= 2, finite rows, valid segments, margin scale 1
        if needs_linear(forest, exact):
            raise ValueError("linear or f64 scalar leaves cannot share a file with vector leaves")
    ext, linear = needs_ext(forest), needs_linear(forest, exact)
    if linear and not vector:
        segments = None  # linear / f64 records are format 6, which has no segment table: the forest keeps the pre-1.2 layout (one output)
    if linear and not vector:
        segments = None  # linear / f64 records are format 6, which has no segment table: the forest keeps the pre-1.2 layout (one output)
    if linear and not forest.sum64:
        raise ValueError("linear leaves need double sums (a LightGBM forest)")
    vec64 = vector and exact and any(not _is_float32(t.vector64) for m in forest.models for t in m.trees if t.vector64 is not None)
    flags = (
        (FLAG_SUM64 if forest.sum64 else 0)
        | (FLAG_EXT if ext else 0)
        | (FLAG_LINEAR if linear else 0)
        | (FLAG_LATE if forest.late_margin and not forest.sum64 else 0)
        | (FLAG_VECTOR if vector else 0)
        | (FLAG_VEC64 if vec64 else 0)
    )
    cats: list = []
    lin: tuple = ([], [])
    vec: list = []
    folds, trees, words, count = [], [], [], 0
    for m in forest.models:
        folds.append((len(trees), len(m.trees)))
        for t in m.trees:
            w = flatten(t, forest.n_features, ext, cats, lin, exact if not vector else vec64, vec)
            trees.append((count, VECTOR_GROUP_U32 if t.vector is not None else t.group))
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
    n_vec = sum(map(len, vec))
    if vector:
        if n_vec > 0xFFFF_FFFF:
            raise ValueError(f"the flat format holds at most {0xFFFF_FFFF:,} vector leaves (this model has {n_vec:,})")
        rows = np.concatenate(vec).astype("<f8" if vec64 else "<f4").tobytes()
        body += rows + bytes(-len(rows) % 8)
    if segments is not None:
        seg = np.asarray(segments, dtype="<u4").tobytes()
        body += seg + bytes(-len(seg) % 8)
    # the xgboost format whenever the forest is expressible in it; the double format for double sums, exp, identity with groups, extensions;
    # the linear format for linear or exact leaves; the vector format for vector leaves
    v2 = bool(flags) or forest.kind == "exp" or (forest.kind == "identity" and forest.groups > 1)
    head = b"TREEFLT1" + struct.pack(
        "<8I",
        FORMAT_VECTOR if vector or segments is not None else FORMAT_LINEAR if linear else FORMAT_DOUBLE if v2 else FORMAT_XGB,
        forest.n_features,
        forest.groups,
        len(folds),
        len(trees),
        count,
        OBJECTIVE[forest.kind],
        AVERAGE[forest.average],
    )
    if vector or segments is not None:  # format 7: a scalar forest with a segment table has no vector leaves (n_vec 0) and no vector flag
        tail = struct.pack("<II", n_vec, 0 if segments is None else len(segments) - 1)
    elif linear:
        tail = struct.pack("<II", len(lin[0]), len(lin[1]))
    else:
        tail = struct.pack("<d", 0.0 if scale == 1.0 else scale)  # bytes 56..64: the margin scale (0 = 1, NaN: several different scales)
    tail = struct.pack("<II", flags, len(cats)) + tail
    assert len(head) == CHECKSUM_AT
    # the checksum covers the whole file, its own eight bytes read as zero: a damaged header is refused like a damaged body
    return head + struct.pack("<Q", checksum(head + bytes(8) + tail + body)) + tail + body


def _is_float32(a: np.ndarray) -> bool:
    return bool(np.array_equal(a, a.astype(np.float32).astype(np.float64)))


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
