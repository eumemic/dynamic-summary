"""Service for computing node relevance scores."""

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ragzoom.vector_api import ensure_normalized

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
        """Compute scores using leaf similarity + bottom-up propagation.

        For leaf nodes (seeds and their siblings), scores come from embeddings.
        For inner nodes, scores are propagated bottom-up as avg(child_scores).

        Args:
            query_embedding: Query embedding vector
            coverage_map: Map of node IDs in coverage
            candidates: Initial candidate nodes with pre-computed similarities
            nodes: Tree nodes for propagation (required for bottom-up scoring)

        Returns:
            Dictionary mapping node IDs to similarity scores
        """
        scores: dict[str, float] = {}
        qn = ensure_normalized(query_embedding)

        # 1. Seeds: use pre-computed scores from candidates
        seed_ids: set[str] = set()
        for node_id, score, _ in candidates:
            if node_id in coverage_map:
                scores[node_id] = score
                seed_ids.add(node_id)

        # 2. Seed siblings: fetch embeddings
        if nodes:
            sibling_ids = self._get_sibling_ids(seed_ids, nodes, coverage_map)
            if sibling_ids:
                sibling_vecs = self.vector_index.get_vectors(list(sibling_ids))
                for v in sibling_vecs:
                    scores[v.id] = float(max(0.0, min(1.0, float(qn @ v.vec))))

        # 3. Propagate scores bottom-up for inner nodes
        if nodes:
            self._propagate_scores_bottom_up(scores, nodes, coverage_map)

        return scores

    def _get_sibling_ids(
        self,
        seed_ids: set[str],
        nodes: Mapping[str, "TreeNode"],
        coverage_map: Mapping[str, bool],
    ) -> set[str]:
        """Find siblings of seeds that need real scores.

        For each seed, find its sibling (via parent) if the sibling is in coverage.
        These siblings need embedding-based scores since they're at the same level
        as seeds and may participate in tiling decisions.
        """
        sibling_ids: set[str] = set()
        for seed_id in seed_ids:
            seed = nodes.get(seed_id)
            if not seed:
                continue
            parent_id = seed.parent_id
            if not parent_id:
                continue
            parent = nodes.get(parent_id)
            if not parent:
                continue
            # Find sibling
            for child_id in (parent.left_child_id, parent.right_child_id):
                if child_id and child_id != seed_id and child_id in coverage_map:
                    sibling_ids.add(child_id)
        return sibling_ids - seed_ids  # Exclude seeds already scored

    def _propagate_scores_bottom_up(
        self,
        scores: dict[str, float],
        nodes: Mapping[str, "TreeNode"],
        coverage_map: Mapping[str, bool],
    ) -> None:
        """Propagate scores from children to parents using avg().

        Process nodes by height (bottom-up) so children are scored before parents.
        Inner nodes get avg(child_scores). Nodes with no scored children get 0.0.
        """
        # Get max height in coverage
        max_height = max(
            (n.height for n in nodes.values() if n.id in coverage_map),
            default=0,
        )

        # Process height by height, bottom-up
        for height in range(1, max_height + 1):
            for node_id in coverage_map:
                node = nodes.get(node_id)
                if not node or node.height != height:
                    continue
                # Note: intentionally overwrite any existing score (ancestor-of-seed case)

                # Collect children scores
                child_scores: list[float] = []
                for child_id in (node.left_child_id, node.right_child_id):
                    if child_id and child_id in scores:
                        child_scores.append(scores[child_id])

                # Propagate as average of children (empty = 0.0)
                scores[node_id] = (
                    sum(child_scores) / len(child_scores) if child_scores else 0.0
                )

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
