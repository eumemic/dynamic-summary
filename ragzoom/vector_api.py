"""Vector API: canonical vector type for retrieval and ranking.

Core logic consumes this open Vector type (normalized float32) irrespective of
backend. VectorIndex implementations translate native formats to this type.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

MetaDict = dict[str, str | int | float | bool | None]


@dataclass(slots=True)
class Vector:
    """Canonical, normalized vector with minimal metadata.

    Invariants:
    - vec is float32 and L2-normalized (||v|| ~= 1)
    - dim equals vec.shape[0]
    - scores in core use dot product, which equals cosine for normalized vecs
    """

    id: str
    vec: NDArray[np.float32]
    meta: MetaDict
    model_id: str
    dim: int

    def __post_init__(self) -> None:
        if self.vec.dtype != np.float32:
            # Copy to ensure float32 invariant
            self.vec = np.asarray(self.vec, dtype=np.float32)
        # Normalize if needed (tolerate tiny numeric drift)
        n = float(np.linalg.norm(self.vec))
        if n == 0.0:
            raise ValueError("Vector has zero norm; cannot normalize")
        # If not already near 1.0, normalize in-place copy
        if not (0.999 <= n <= 1.001):
            self.vec = (self.vec / n).astype(np.float32, copy=False)
        # Dimensional safety
        self.dim = int(self.vec.shape[0])


def ensure_normalized(
    v: NDArray[np.float64] | NDArray[np.float32] | list[float],
) -> NDArray[np.float32]:
    """Return a float32, L2-normalized copy of the input vector."""
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n == 0.0:
        raise ValueError("Cannot normalize zero vector")
    return (arr / n).astype(np.float32, copy=False)


def dot_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute dot product similarity in [0, 1] for normalized vectors."""
    # Clip to [0,1] to guard tiny numeric negatives from rounding
    return float(np.clip(np.dot(a, b), 0.0, 1.0))


def vectors_matrix(vectors: list[Vector]) -> NDArray[np.float32]:
    """Stack vectors into a contiguous 2D float32 matrix (n x d)."""
    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    d = vectors[0].dim
    for v in vectors:
        if v.dim != d:
            raise ValueError("Vectors have mismatched dimensions in set")
    mat = np.vstack([v.vec for v in vectors]).astype(np.float32, copy=False)
    return mat
