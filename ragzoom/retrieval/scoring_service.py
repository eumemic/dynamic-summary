"""Service for computing node relevance scores."""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ragzoom.vector_api import dot_similarity, ensure_normalized, unpack_embedding

if TYPE_CHECKING:
    from ragzoom.contracts.tree_node import TreeNode
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
        coverage_map: Mapping[str, bool],
        candidates: list[tuple[str, float, dict[str, str | int | float | bool | None]]],
        nodes: Mapping[str, "TreeNode"] | None = None,
    ) -> dict[str, float]:
        """Compute scores using pre-indexed embeddings on nodes.

        All nodes with embeddings get scored via dot product with query.
        Nodes without embeddings (leaves not yet indexed) get 0.0.

        Args:
            query_embedding: Query embedding vector
            coverage_map: Map of node IDs in coverage
            candidates: Initial candidate nodes with pre-computed similarities
            nodes: Tree nodes with embeddings

        Returns:
            Dictionary mapping node IDs to similarity scores
        """
        scores: dict[str, float] = {}
        qn = ensure_normalized(query_embedding)

        if not nodes:
            # Fallback: use pre-computed candidate scores only
            for node_id, score, _ in candidates:
                if node_id in coverage_map:
                    scores[node_id] = score
            return scores

        # Score all nodes in coverage using their pre-indexed embeddings
        for node_id in coverage_map:
            node = nodes.get(node_id)
            if node is None:
                continue

            embedding = node.embedding
            if embedding is not None:
                node_vec = unpack_embedding(embedding)
                scores[node_id] = dot_similarity(node_vec, qn)
            else:
                # Leaf not yet indexed - assign 0.0
                scores[node_id] = 0.0

        return scores
