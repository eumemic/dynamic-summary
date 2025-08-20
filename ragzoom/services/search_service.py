"""Search service for vector similarity search and MMR using pgvector."""

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import select

from ragzoom.models import TreeNode
from ragzoom.storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SearchService:
    """Service for vector search and MMR operations using pgvector."""

    def __init__(self, database_manager: DatabaseManager):
        """Initialize search service.

        Args:
            database_manager: Database manager for DB operations
        """
        self.db_manager = database_manager
        self.SessionLocal = database_manager.SessionLocal

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar nodes using pgvector cosine distance.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional filter conditions (not used in pgvector implementation)

        Returns:
            List of (id, similarity, metadata) tuples where similarity is in [0, 1]
        """
        query_array = list(map(float, query_embedding))

        with self.SessionLocal() as session:
            stmt = (
                select(
                    TreeNode.id,
                    TreeNode.embedding.cosine_distance(query_array).label("distance"),
                    TreeNode.span_start,
                    TreeNode.span_end,
                    TreeNode.parent_id,
                    TreeNode.document_id,
                    TreeNode.left_child_id,
                    TreeNode.right_child_id,
                )
                .order_by(TreeNode.embedding.cosine_distance(query_array))
                .limit(n_results)
            )

            # Apply document filter if specified
            if where and "document_id" in where:
                stmt = stmt.where(TreeNode.document_id == where["document_id"])

            rows = session.execute(stmt).all()

        output = []
        for row in rows:
            distance = float(row.distance)
            # Convert cosine distance to similarity score
            # Cosine distance ranges from 0 (identical) to 2 (opposite)
            # We map this to similarity: 1 (identical) to 0 (opposite)
            similarity = 1.0 - (distance / 2.0)
            similarity = max(0.0, min(1.0, similarity))

            metadata = {
                "span_start": row.span_start,
                "span_end": row.span_end,
                "parent_id": row.parent_id or "",
                "document_id": row.document_id or "",
                "is_leaf": (
                    1
                    if (row.left_child_id is None and row.right_child_id is None)
                    else 0
                ),
            }
            output.append((row.id, similarity, metadata))

        return output

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, dict[str, Any]]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Apply MMR (Maximal Marginal Relevance) to get diverse results.

        Args:
            query_embedding: Query embedding vector
            candidates: List of (id, similarity, metadata) candidate tuples
            lambda_param: Balance between relevance and diversity (0-1)
            k: Number of results to select

        Returns:
            List of selected node IDs
        """
        if not candidates or k <= 0:
            return []

        # Get embeddings for all candidates
        candidate_ids = [c[0] for c in candidates]

        with self.SessionLocal() as session:
            rows = session.execute(
                select(TreeNode.id, TreeNode.embedding).where(
                    TreeNode.id.in_(candidate_ids)
                )
            ).all()

        id_to_embedding = {
            row.id: np.array(row.embedding, dtype=np.float32) for row in rows
        }

        # Build candidate embeddings array in order
        cand_embs = np.array([id_to_embedding[cid] for cid in candidate_ids])
        query_emb = np.array(query_embedding)

        # Vectorized similarity computation
        query_sims = np.dot(cand_embs, query_emb)

        # MMR iterative selection with optimized operations
        selected_mask = np.zeros(len(candidates), dtype="bool")
        selected_indices = []

        # Select first item (highest relevance)
        first_idx = np.argmax(query_sims)
        selected_indices.append(first_idx)
        selected_mask[first_idx] = True

        # Pre-compute pairwise similarities for efficiency
        if len(candidates) > 1:
            pairwise_sims = np.dot(cand_embs, cand_embs.T)

        # Select remaining items
        for _ in range(1, min(k, len(candidates))):
            # Vectorized MMR computation for all unselected
            unselected_mask = ~selected_mask
            if not np.any(unselected_mask):
                break

            # Relevance scores for unselected
            relevances = query_sims[unselected_mask]

            # Max similarity to any selected item (vectorized)
            max_sims = (
                np.max(pairwise_sims[np.ix_(unselected_mask, selected_mask)], axis=1)
                if np.any(selected_mask)
                else np.zeros(int(np.sum(unselected_mask)))
            )

            # MMR scores
            mmr_scores = lambda_param * relevances - (1 - lambda_param) * max_sims

            # Get index in unselected subset
            best_unselected_idx = np.argmax(mmr_scores)

            # Convert to original index
            unselected_indices = np.where(unselected_mask)[0]
            best_idx = unselected_indices[best_unselected_idx]

            selected_indices.append(best_idx)
            selected_mask[best_idx] = True

        # Return selected node IDs
        return [candidates[i][0] for i in selected_indices]
