"""Rows for a profile-guided build in the tests: the real rows of a fixture, repeated to a bit more than the profile floor (a build refuses
fewer rows than `backend_codegen.PROFILE_MIN_ROWS`; the halves of a repeated set agree, so the overlap check passes)."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import backend_codegen as bc


def tiled(x, factor: float = 1.25) -> np.ndarray:
    x = np.asarray(x)
    n = math.ceil(bc.PROFILE_MIN_ROWS * factor)
    return np.ascontiguousarray(np.tile(x, (math.ceil(n / len(x)), 1)))
