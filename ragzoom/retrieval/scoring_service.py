"""Service for computing node relevance scores."""

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragzoom.document_store import DocumentStore

logger = logging.getLogger(__name__)


class ScoringService:
    """Computes relevance scores for nodes in the coverage map."""

    def __init__(self, store: "DocumentStore"):
        """Initialize scoring service.

        Args:
            store: DocumentStore instance for node retrieval
        """
        self.store = store
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

        # Try to use node embeddings first (for backward compatibility with existing stores)
        nodes = self.store.nodes.get_nodes(list(node_ids))

        # Prepare for vectorized computation
        query_vec = np.array(query_embedding)
        valid_embeddings = []
        valid_node_ids = []
        nodes_without_embeddings = set()

        # Collect valid embeddings from nodes that have them
        for node in nodes:
            embedding = getattr(node, "embedding", None)
            if embedding is not None:
                valid_embeddings.append(embedding)
                valid_node_ids.append(node.id)
            else:
                nodes_without_embeddings.add(node.id)

        # Handle missing nodes
        loaded_node_ids = {node.id for node in nodes}
        for node_id in node_ids:
            if node_id not in loaded_node_ids:
                nodes_without_embeddings.add(node_id)

        # For nodes without embeddings, use search service fallback
        if nodes_without_embeddings:
            try:
                # Get total node count for this document to ensure we get all similarities
                doc_nodes = self.store.nodes.get_all()
                total_nodes = len(doc_nodes)

                # Search all vectors in this document via the vector index
                search_results = self.store.search.similar(
                    query_embedding, n_results=total_nodes
                )

                # Build score map from search results
                score_map = {
                    node_id: score for (node_id, score, _meta) in search_results
                }

                # Assign scores for nodes without embeddings
                for node_id in nodes_without_embeddings:
                    scores[node_id] = float(score_map.get(node_id, 0.0))

            except Exception as e:
                self.logger.warning(f"Failed to get scores from search service: {e}")
                # Fallback: assign zero scores
                for node_id in nodes_without_embeddings:
                    scores[node_id] = 0.0

        # Vectorized similarity computation for nodes with valid embeddings
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
                self.logger.warning(f"Failed to compute batch similarities: {e}")
                # Fallback to individual computation
                for node_id in valid_node_ids:
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
