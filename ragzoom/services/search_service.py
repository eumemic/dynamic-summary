"""Service for vector similarity search and MMR diversity."""

import logging
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class SearchService:
    """Service for vector search operations and MMR diversity."""

    def __init__(self, database_manager: DatabaseManager):
        """Initialize search service.

        Args:
            database_manager: Database manager for ChromaDB operations
        """
        self.db_manager = database_manager

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar nodes using Chroma.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional filter conditions

        Returns:
            List of (id, similarity, metadata) tuples where similarity is in [0, 1]
        """
        # Convert to numpy array if needed
        query_array = np.array(query_embedding, dtype=np.float32)
        results = self.db_manager.collection.query(
            query_embeddings=cast(Any, [query_array]),
            n_results=n_results,
            where=where,
        )

        # Return list of (id, similarity, metadata) tuples
        output = []
        ids = results.get("ids")
        distances = results.get("distances")
        metadatas = results.get("metadatas")

        if ids and distances and metadatas and len(ids) > 0:
            for i in range(len(ids[0])):
                # Convert cosine distance to similarity
                # Cosine distance ranges from 0 to 2, where 0 is identical
                # Similarity = 1 - (distance / 2) to map to [0, 1]
                distance = float(distances[0][i])
                similarity = 1.0 - (distance / 2.0)
                # Ensure similarity is in valid range [0, 1]
                similarity = max(0.0, min(1.0, similarity))

                output.append(
                    (
                        ids[0][i],
                        similarity,
                        (
                            dict(metadatas[0][i])
                            if isinstance(metadatas[0][i], dict)
                            else {}
                        ),
                    )
                )

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
            candidates: List of (id, similarity, metadata) tuples
            lambda_param: Balance between relevance and diversity (0-1)
            k: Number of results to select

        Returns:
            List of selected node IDs in MMR order
        """
        if not candidates or k <= 0:
            return []

        # Get embeddings for all candidates
        candidate_ids = [c[0] for c in candidates]

        # Batch retrieve embeddings
        results = self.db_manager.collection.get(
            ids=candidate_ids, include=["embeddings"]
        )

        # Create ID to embedding mapping for O(1) lookup
        embeddings = results.get("embeddings")
        ids = results.get("ids")
        if embeddings is None or ids is None:
            return []

        id_to_embedding = {ids[i]: np.array(embeddings[i]) for i in range(len(ids))}

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
