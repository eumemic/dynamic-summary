"""Service for computing node relevance scores."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ragzoom.store import StoreManager

logger = logging.getLogger(__name__)


class ScoringService:
    """Computes relevance scores for nodes in the coverage map."""

    def __init__(self, store: "StoreManager"):
        """Initialize scoring service.

        Args:
            store: Store instance for node retrieval
        """
        self.store = store

    def compute_scores(
        self,
        query_embedding: list[float],
        coverage_map: dict[str, bool],
        candidates: list[tuple[str, float, dict[str, Any]]],
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

        # Batch fetch all nodes in a single operation
        nodes = self.store.nodes.get_nodes(list(node_ids))

        # Prepare for vectorized computation
        query_vec = np.array(query_embedding)
        valid_embeddings = []
        valid_node_ids = []

        # Collect valid embeddings
        for node in nodes:
            if node.embedding is not None:
                valid_embeddings.append(node.embedding)
                valid_node_ids.append(node.id)
            else:
                scores[node.id] = 0.0

        # Handle missing nodes
        loaded_node_ids = {node.id for node in nodes}
        for node_id in node_ids:
            if node_id not in loaded_node_ids:
                scores[node_id] = 0.0

        # Vectorized similarity computation for all valid embeddings
        if valid_embeddings:
            try:
                embeddings_matrix = np.array(valid_embeddings)
                similarities = self._compute_cosine_similarities_batch(
                    query_vec, embeddings_matrix
                )

                # Update scores dictionary
                for node_id, similarity in zip(valid_node_ids, similarities):
                    scores[node_id] = similarity

            except Exception as e:
                logger.warning(f"Failed to compute batch similarities: {e}")
                # Fallback to individual computation
                for node_id in valid_node_ids:
                    scores[node_id] = 0.0

    @staticmethod
    def _compute_cosine_similarities_batch(
        query_vec: np.ndarray, embeddings_matrix: np.ndarray
    ) -> np.ndarray:
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
