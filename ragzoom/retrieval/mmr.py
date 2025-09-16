"""Generic MMR selection operating on canonical Vectors.

Implements greedy Maximal Marginal Relevance:
  argmax_v (lambda * rel(v) - (1-lambda) * max_{s in selected} sim(v, s))

Relies on generic similarity utilities; does not depend on backend types.
"""

from __future__ import annotations

import numpy as np

from ragzoom.retrieval.similarity import pairwise_similarities, relevance_scores
from ragzoom.vector_api import Vector, ensure_normalized


def select_diverse(
    query_embedding: list[float],
    candidates: list[Vector],
    k: int,
    lambda_param: float,
) -> list[str]:
    """Select k diverse IDs from candidates using MMR.

    Args:
        query_embedding: Raw query embedding (will be normalized)
        candidates: Candidate vectors (normalized)
        k: Number of selections
        lambda_param: Balance relevance vs. diversity in [0,1]

    Returns:
        List of selected candidate ids
    """
    if k <= 0 or not candidates:
        return []

    # Normalize query once and compute relevance vector
    qn = ensure_normalized(query_embedding)
    rel = np.asarray(relevance_scores(qn, candidates), dtype=np.float32)

    # Precompute pairwise similarities among candidates
    pw = pairwise_similarities(candidates)
    n = len(candidates)
    selected_mask = np.zeros(n, dtype=bool)
    selected: list[int] = []

    # First pick: max relevance
    first = int(np.argmax(rel))
    selected.append(first)
    selected_mask[first] = True

    if k == 1 or n == 1:
        return [candidates[first].id]

    # Greedy picks
    while len(selected) < min(k, n):
        un = np.where(~selected_mask)[0]
        if un.size == 0:
            break
        # For each unselected, compute max similarity to already selected set
        if selected:
            sel_idx = np.asarray(selected, dtype=int)
            max_sim = np.max(pw[np.ix_(un, sel_idx)], axis=1)
        else:
            max_sim = np.zeros(un.size)
        mmr = lambda_param * rel[un] - (1.0 - lambda_param) * max_sim
        pick = un[int(np.argmax(mmr))]
        selected.append(int(pick))
        selected_mask[pick] = True

    return [candidates[i].id for i in selected]
