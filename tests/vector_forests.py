"""Vector forests for the tests of QuickForest 1.2: `make_vector_forest` builds a random one, `explode` turns it into the scalar forest that
must answer exactly the same, bit for bit, on every engine (each vector tree is W scalar trees in groups 0..W-1, same structure, same order).
`explode` is the reference of every vector test (DISENO A2, A6 V1)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import qf_common as qf

MAX_CATEGORY = 40


def random_shape(rng: np.random.Generator, depth: int, split_chance: float = 0.85):
    """(left, right) int32 arrays of a random binary tree of at most `depth` levels (a root that never splits is allowed only at depth 0),
    children numbered after their parent, breadth first."""
    left, right, level = [-1], [-1], [0]
    i = 0
    while i < len(left):
        if level[i] < depth and (level[i] == 0 or rng.random() <= split_chance):
            left[i], right[i] = len(left), len(left) + 1
            left += [-1, -1]
            right += [-1, -1]
            level += [level[i] + 1] * 2
        i += 1
    return np.array(left, np.int32), np.array(right, np.int32)


def make_vector_tree(rng, width: int, depth: int, n_features: int, ext: bool, table64: bool) -> qf.Tree:
    left, right = random_shape(rng, depth)
    n = len(left)
    leaf = left == -1
    feature = np.where(leaf, 0, rng.integers(0, n_features, n)).astype(np.int32)
    value = np.where(leaf, 0, rng.normal(0, 1, n)).astype(np.float32)
    default_left = rng.random(n) < 0.5
    zero_missing, cat = None, None
    if ext:
        zero_missing = (~leaf) & (rng.random(n) < 0.3)
        cat = {}
        for i in np.flatnonzero((~leaf) & (rng.random(n) < 0.25)):
            cat[int(i)] = np.unique(rng.integers(0, MAX_CATEGORY, rng.integers(1, 6))).astype(np.int64)
            zero_missing[i] = False
    vector = np.zeros((n, width), np.float32)
    vector[leaf] = rng.normal(0, 1, (int(leaf.sum()), width)).astype(np.float32)
    vector64 = None
    if table64:
        vector64 = np.zeros((n, width), np.float64)
        vector64[leaf] = rng.normal(0, 1, (int(leaf.sum()), width))
        vector = vector64.astype(np.float32)
    return qf.Tree(qf.VECTOR_GROUP, left, right, feature, value, default_left, zero_missing, cat, vector=vector, vector64=vector64)


def make_vector_forest(
    seed: int = 0,
    *,
    width: int = 5,
    depth: int = 4,
    trees: int = 6,
    folds: int = 1,
    n_features: int = 6,
    kind: str = "identity",
    average: str = "output",
    sum64: bool = False,
    table64: bool = False,
    ext: bool = False,
    segments: list[int] | None = None,
) -> qf.Forest:
    """A random forest of vector trees: `folds` models of `trees` trees each, W = `width` groups. `ext` adds categorical and zero-as-missing
    nodes (they need `sum64`, as the library adapters that produce them do); `table64` gives every tree a `vector64` (the float32 `vector`
    is its rounding)."""
    rng = np.random.default_rng(seed)
    models = []
    for _ in range(folds):
        trees_ = [make_vector_tree(rng, width, depth, n_features, ext, table64) for _ in range(trees)]
        models.append(qf.ModelIR(rng.normal(0, 0.2, width).astype(np.float32), trees_))
    forest = qf.Forest(n_features, width, kind, average, models, "auto", sum64 or ext or table64, segments=segments)
    qf.check_vector(forest)
    return forest


def explode(forest: qf.Forest) -> qf.Forest:
    """The scalar forest that answers exactly as `forest` does: each vector tree becomes W scalar trees, tree by tree with groups 0..W-1, in
    its place (so every group sums its trees in the same order); a scalar tree is kept. Segments are dropped (they change no prediction)."""
    models = []
    for m in forest.models:
        trees = []
        for t in m.trees:
            if t.vector is None:
                trees.append(t)
                continue
            for g in range(forest.groups):
                value = np.where(t.left == -1, t.vector[:, g], t.value)  # thresholds stay at the splits
                leaf64 = None if t.vector64 is None else t.vector64[:, g].copy()
                trees.append(qf.Tree(g, t.left, t.right, t.feature, value, t.default_left, t.zero_missing, t.cat, None, leaf64))
        models.append(qf.ModelIR(m.base, trees, m.margin_scale))
    out = qf.Forest(
        forest.n_features,
        forest.groups,
        forest.kind,
        forest.average,
        models,
        forest.task,
        forest.sum64,
        forest.late_margin,
        forest.margin_scale,
    )
    qf.check_vector(out)
    return out
