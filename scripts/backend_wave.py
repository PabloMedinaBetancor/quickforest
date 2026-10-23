"""Backends `wave` and `wave-avx2`: the same flat file as `flat`, evaluated by the wavefront engines of rust/src/wave.rs (all rows of a block walk one
tree together, level by level, from a compact list of active (row, node) items, so no lane idles and no branch depends on the data).
`wave` is scalar (any CPU); `wave-avx2` uses AVX2 gathers when the CPU has them and runs the scalar engine otherwise
(`predictor.engine_used` says which). Same results as `flat`, bit for bit."""

from __future__ import annotations

from pathlib import Path

import evaluator
from backend_flat import FlatBackend


class WaveBackend(FlatBackend):
    name = "wave"
    engine = 1
    supports = frozenset()  # xgboost semantics only (float32 sums); the walker (`flat`) runs the LightGBM / YDF extensions

    def load(self, artifact: Path, **options):
        return evaluator.FlatForest(artifact, engine=self.engine)


class WaveAvx2Backend(WaveBackend):
    name = "wave-avx2"
    engine = 2
