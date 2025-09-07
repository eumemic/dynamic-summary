"""Protocol for vector search backends.

Defines the minimal surface that retrieval needs for candidate search and MMR.
Keeping this small allows us to plug in pgvector, FAISS/numpy, or other
implementations without disturbing the rest of the system.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


class VectorSearchMetadata(Protocol):
    """Metadata associated with search results.

    Implementations should at minimum provide:
    - span_start: int
    - span_end: int
    - parent_id: str
    - document_id: str
    - is_leaf: int (0/1)
    """

    span_start: int
    span_end: int
    parent_id: str
    document_id: str
    is_leaf: int


@runtime_checkable
class VectorIndex(Protocol):
    """Pluggable vector index contract used by retrieval.

    Two core capabilities are required:
    - similarity search with a simple filter (typically by document_id)
    - MMR re-ranking for diversity
    """

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[tuple[str, float, VectorSearchMetadata]]:
        """Return top-N similar node IDs with similarity scores and metadata.

        Implementations may return cosine similarity in [0, 1] or another
        monotonic similarity measure. Retrieval treats higher as better.
        """

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, VectorSearchMetadata]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Select k diverse candidates using MMR.

        Implementations may delegate to a shared helper; reproducibility is
        preferred to black-box stochastic behavior for testability.
        """
