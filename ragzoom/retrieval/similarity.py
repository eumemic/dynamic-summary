"""Generic similarity utilities operating on canonical Vector types.

These functions centralize the math used by retrieval (relevance, pairwise
similarities) and enforce basic safety invariants (dimensions/models match).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.vector_api import Vector, ensure_normalized, vectors_matrix


def relevance_scores(
    query_embedding: NDArray[np.float32] | list[float],
    candidates: list[Vector],
) -> list[float]:
    """Compute relevance scores dot(q, v) for each candidate.

    Args:
        query_embedding: raw query vector; will be normalized to float32
        candidates: list of canonical Vectors (already normalized)
    Returns:
        List of floats in [0,1]
    """
    if not candidates:
        return []
    q = ensure_normalized(query_embedding)
    mat = vectors_matrix(candidates)
    sims = mat @ q
    sims = np.clip(sims, 0.0, 1.0)
    return [float(x) for x in sims]


def pairwise_similarities(vectors: list[Vector]) -> NDArray[np.float32]:
    """Compute pairwise dot similarities for a small set of vectors.

    Returns an (n x n) float32 matrix with diagonal ~1.0.
    """
    if not vectors:
        return cast(NDArray[np.float32], np.zeros((0, 0), dtype=np.float32))
    mat = vectors_matrix(vectors)
    sims = (mat @ mat.T).astype(np.float32, copy=False)
    sims = cast(NDArray[np.float32], sims)
    # Clamp tiny numerical drift
    np.clip(sims, 0.0, 1.0, out=sims)
    return sims
