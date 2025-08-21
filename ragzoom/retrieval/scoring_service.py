"""Service for computing node relevance scores."""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ragzoom.store import StoreManager, TreeNode

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
        query_vec = np.array(query_embedding)

        for node_id in node_ids:
            node: TreeNode | None = self.store.nodes.get_node(node_id)
            if node is not None and node.embedding is not None:
                try:
                    similarity = self._compute_cosine_similarity(
                        query_vec, np.array(node.embedding)
                    )
                    scores[node_id] = similarity
                except Exception as e:
                    logger.warning(
                        f"Failed to compute embedding similarity for node {node_id}: {e}"
                    )
                    scores[node_id] = 0.0

    @staticmethod
    def _compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity in range [0, 1]
        """
        similarity = float(
            np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )
        return max(0.0, min(1.0, similarity))
