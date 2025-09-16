"""Service for computing node relevance scores."""

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragzoom.contracts.vector_index import VectorIndex
    from ragzoom.document_store import DocumentStore

logger = logging.getLogger(__name__)


class ScoringService:
    """Computes relevance scores for nodes in the coverage map."""

    def __init__(self, store: "DocumentStore", vector_index: "VectorIndex"):
        """Initialize scoring service.

        Args:
            store: DocumentStore instance for node retrieval
        """
        self.store = store
        self.vector_index = vector_index
        self.logger = logger

    def compute_scores(
        self,
        query_embedding: list[float],
        coverage_map: dict[str, bool],
        candidates: list[tuple[str, float, dict[str, str | int | float | bool | None]]],
    ) -> dict[str, float]:
        """Compute similarity scores for all nodes in coverage map.

        Args:
            query_embedding: Query embedding vector
            coverage_map: Map of node IDs in coverage
            candidates: Initial candidate nodes with pre-computed similarities

        Returns:
            Dictionary mapping node IDs to similarity scores
        """
        scores = {}

        for node_id, similarity, _ in candidates:
            if node_id in coverage_map:
                scores[node_id] = similarity

        nodes_needing_scores = set(coverage_map.keys()) - set(scores.keys())
        if nodes_needing_scores:
            self._compute_remaining_scores(
                query_embedding, nodes_needing_scores, scores
            )

        return scores

    def _compute_remaining_scores(
        self,
        query_embedding: list[float],
        node_ids: set[str],
        scores: dict[str, float],
    ) -> None:
        """Compute similarities for nodes not in initial candidates.

        Args:
            query_embedding: Query embedding vector
            node_ids: Set of node IDs needing scores
            scores: Dictionary to update with computed scores
        """
        if not node_ids:
            return

        # Score all requested nodes via VectorIndex
        try:
            from ragzoom.vector_api import ensure_normalized

            vecs = self.vector_index.get_vectors(list(node_ids))
            qn = ensure_normalized(query_embedding)
            for v in vecs:
                scores[v.id] = float(max(0.0, min(1.0, float(qn @ v.vec))))
            # Any IDs not returned by vector index get 0.0 score
            for node_id in node_ids:
                scores.setdefault(node_id, 0.0)
        except Exception as e:
            self.logger.warning(f"Failed to score via VectorIndex: {e}")
            for node_id in node_ids:
                scores[node_id] = 0.0

    @staticmethod
    def _compute_cosine_similarities_batch(
        query_vec: NDArray[np.float64], embeddings_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute cosine similarities between query and multiple embeddings.

        Args:
            query_vec: Query embedding vector (1D)
            embeddings_matrix: Matrix of embeddings (2D: num_embeddings x embedding_dim)

        Returns:
            Array of cosine similarities in range [0, 1]
        """
        # Normalize query vector
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return np.zeros(embeddings_matrix.shape[0])

        # Normalize embedding vectors
        embedding_norms = np.linalg.norm(embeddings_matrix, axis=1)

        # Handle zero-norm embeddings
        zero_norm_mask = embedding_norms == 0
        embedding_norms = np.where(zero_norm_mask, 1.0, embedding_norms)

        # Compute similarities
        similarities = np.dot(embeddings_matrix, query_vec) / (
            query_norm * embedding_norms
        )

        # Set zero-norm embeddings to 0 similarity
        similarities = np.where(zero_norm_mask, 0.0, similarities)

        # Clip to [0, 1] range
        return np.clip(similarities, 0.0, 1.0)
