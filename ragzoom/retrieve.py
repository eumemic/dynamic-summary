"""Retrieval logic with MMR diversity for RagZoom."""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from openai import OpenAI

from ragzoom.config import QueryConfig
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from ragzoom.store import Store, TreeNode

if TYPE_CHECKING:
    from ragzoom.index import TreeBuilder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    node_ids: list[str]
    scores: dict[str, float]
    coverage_map: dict[str, bool]
    tiling: list[str] | None = None  # List of node IDs in the tiling
    nodes: dict[str, "TreeNode"] | None = (
        None  # Pre-loaded nodes to avoid redundant loading
    )


class Retriever:
    """Handles retrieval and MMR diversity for query processing."""

    def __init__(
        self,
        query_config: QueryConfig,
        store: Store,
        api_key: str = "",
        tree_builder: Optional["TreeBuilder"] = None,
    ):
        """Initialize retriever.

        Args:
            query_config: Query configuration
            store: Store instance
            api_key: OpenAI API key (if not provided, reads from OPENAI_API_KEY env)
            tree_builder: Optional TreeBuilder instance
        """
        self.query_config = query_config
        self.store = store

        # Get API key from parameter or environment
        import os

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key required for Retriever")

        self.client = OpenAI(api_key=api_key)
        self.dp_generator = DynamicTilingGenerator(query_config)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for query text."""
        try:
            response = self.client.embeddings.create(
                model=self.query_config.embedding_model,
                input=query,
                # Let OpenAI API determine dimensions - no need for hardcoded values
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise

    async def retrieve_async(
        self,
        query: str,
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
    ) -> RetrievalResult:
        """Async retrieval method with MMR diversity.

        Args:
            query: Query text to search for
            num_seeds: Number of seed nodes to retrieve
            budget_tokens: Token budget for the final summary
            document_id: Optional document ID to filter by

        Supports three modes:
        1. Budget only: Calculate conservative num_seeds to guarantee no overflow
        2. Budget + num_seeds: Use num_seeds but drop nodes if needed for budget
        3. num_seeds only: Just use num_seeds, no budget enforcement
        """
        # Determine which mode we're in
        if budget_tokens is not None and num_seeds is None:
            # Mode 1: Budget only - calculate conservative num_seeds
            num_seeds = self._calculate_conservative_num_seeds(
                budget_tokens, document_id
            )
            logger.info(
                f"Budget-only mode: calculated conservative num_seeds={num_seeds} for budget={budget_tokens}"
            )
        elif budget_tokens is not None and num_seeds is not None:
            # Mode 2: Budget + num_seeds - will enforce both constraints
            logger.info(f"Mixed mode: num_seeds={num_seeds}, budget={budget_tokens}")
        elif num_seeds is None:
            # Mode 3: num_seeds only (using default)
            # Use a reasonable default chunk size for calculation
            default_chunk_size = 256
            num_seeds = self.query_config.budget_tokens // default_chunk_size
            logger.info(f"num_seeds-only mode: using num_seeds={num_seeds}")

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Step 1: Initial retrieval (2 * num_seeds candidates)
        k_candidates = int(num_seeds * self.query_config.mmr_k_multiplier)

        # Filter by document_id if provided
        where_filter = {"document_id": document_id} if document_id else None
        candidates = self.store.search_similar(
            query_embedding, k_candidates, where=where_filter
        )

        # Step 2: Apply MMR to get diverse num_seeds results
        selected_ids = self.store.compute_mmr_diverse_results(
            query_embedding, candidates, self.query_config.mmr_lambda, num_seeds
        )

        # Step 3: Build coverage map (selected + ancestors)
        coverage_map = self._build_coverage_map(selected_ids)

        # Step 4: Apply pinned nodes
        pinned_nodes = self.store.get_pinned_nodes(self.store.PIN_DEPTH_MAX)
        for node in pinned_nodes:
            coverage_map[node.id] = True

        # Build scores map - compute similarity for ALL nodes in coverage map
        scores = {}

        # First, add scores for the candidate nodes (already have similarities)
        for node_id, similarity, _ in candidates:
            if node_id in coverage_map:
                scores[node_id] = similarity

        # Then, compute similarities for all other nodes in coverage map
        nodes_needing_scores = set(coverage_map.keys()) - set(scores.keys())
        if nodes_needing_scores:
            # Get embeddings and compute similarities for ancestors
            for node_id in nodes_needing_scores:
                ancestor_node: TreeNode | None = self.store.get_node(node_id)
                if ancestor_node is not None:
                    # Get node's embedding from Chroma
                    try:
                        result = self.store.collection.get(
                            ids=[node_id], include=["embeddings"]
                        )
                        embeddings = result.get("embeddings")
                        if embeddings is not None and len(embeddings) > 0:
                            node_embedding = embeddings[0]
                            # Compute cosine similarity
                            import numpy as np

                            query_vec = np.array(query_embedding)
                            node_vec = np.array(node_embedding)
                            # Cosine similarity = dot product of normalized vectors
                            similarity = float(
                                np.dot(query_vec, node_vec)
                                / (np.linalg.norm(query_vec) * np.linalg.norm(node_vec))
                            )
                            scores[node_id] = max(0.0, min(1.0, similarity))
                    except Exception as e:
                        logger.warning(
                            f"Failed to get embedding for node {node_id}: {e}"
                        )
                        scores[node_id] = 0.0

        # Handle empty coverage map case
        if not coverage_map:
            # No nodes selected, return empty result
            return RetrievalResult(
                node_ids=selected_ids,
                scores=scores,
                coverage_map=coverage_map,
                tiling=[],
                nodes={},
            )

        # Load all nodes in coverage map to avoid redundant loading later
        # Use batch loading for efficiency
        nodes: dict[str, TreeNode] = {}
        node_ids_to_load = list(coverage_map.keys())
        if node_ids_to_load:
            loaded_nodes = self.store.get_nodes(node_ids_to_load)
            for node in loaded_nodes:
                nodes[node.id] = node

        # Find the root node in the coverage map
        root_id = None
        for node_id, node in nodes.items():
            # Check if this node has no parent in the coverage map
            if node.parent_id is None or node.parent_id not in nodes:
                root_id = node_id
                break

        if not root_id:
            # No root found - this should never happen as coverage map should include all ancestors
            raise ValueError(
                f"No root node found in coverage map. Coverage map has {len(nodes)} nodes but none have no parent in the map."
            )

        # Step 5: Extract tiling using DP algorithm
        final_budget = (
            budget_tokens
            if budget_tokens is not None
            else self.query_config.budget_tokens
        )
        dp_result = self.dp_generator.find_optimal_tiling(
            final_budget, scores, nodes, root_id
        )

        return RetrievalResult(
            node_ids=selected_ids,
            scores=scores,
            coverage_map=coverage_map,  # Use the original coverage map
            tiling=dp_result.tiling.node_ids,
            nodes=nodes,
        )

    # jscpd:ignore-start
    def retrieve(
        self,
        query: str,
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
    ) -> RetrievalResult:
        """Synchronous wrapper for retrieve_async.

        Args:
            query: Query text to search for
            num_seeds: Number of seed nodes to retrieve
            budget_tokens: Token budget for the final summary
            document_id: Optional document ID to filter by

        Creates a new event loop if needed to run the async version.
        For async contexts, use retrieve_async directly.
        """
        # jscpd:ignore-end
        return asyncio.run(
            self.retrieve_async(query, num_seeds, budget_tokens, document_id)
        )

    def _build_coverage_map(self, selected_ids: list[str]) -> dict[str, bool]:
        """Build a coverage map including selected nodes, their ancestors, and all required siblings to maintain the coverage property."""
        if not selected_ids:
            return {}

        # Mark selected nodes as covered and update access
        coverage_map = {node_id: True for node_id in selected_ids}
        for node_id in selected_ids:
            self.store.update_node_access(node_id)

        # Add all ancestors
        ancestors = self.store.get_ancestors(selected_ids)
        for ancestor in ancestors:
            coverage_map[ancestor.id] = True

        # Iteratively ensure coverage: if a child is in the coverage set, include its sibling (if exists) so parent span equals union of children spans
        while True:
            nodes_in_coverage = self.store.get_nodes(list(coverage_map.keys()))
            new_nodes_added = False
            for node in nodes_in_coverage:
                # If node is in coverage and is an internal node in the main tree, ensure both children are present
                left = node.left_child_id
                right = node.right_child_id
                if left or right:
                    # If a child is present in the coverage set, include its sibling if it exists
                    # This maintains the coverage property
                    if left and left in coverage_map:
                        # Left child is in coverage
                        if right and right not in coverage_map:
                            # Include right sibling if it exists
                            coverage_map[right] = True
                            new_nodes_added = True
                    elif right and right in coverage_map:
                        # Right child is in coverage
                        if left and left not in coverage_map:
                            # Include left sibling (must exist in left-balanced tree)
                            coverage_map[left] = True
                            new_nodes_added = True
            if not new_nodes_added:
                break
        return coverage_map

    def _calculate_conservative_num_seeds(
        self, budget_tokens: int, document_id: str | None = None
    ) -> int:
        """Calculate conservative num_seeds that is more grounded in document reality."""

        # Get all nodes for the document to calculate an average cost.
        all_nodes = self.store.get_all_nodes_for_document(document_id)
        if not all_nodes:
            # Use a reasonable constant fallback with warning
            logger.warning(
                f"No nodes found for document {document_id}, using default estimate"
            )
            return max(1, budget_tokens // 200)  # 200 tokens per node estimate

        # Calculate actual average from real nodes
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        total_tokens = sum(len(tokenizer.encode(node.text)) for node in all_nodes)
        average_tokens_per_node = total_tokens / len(all_nodes)

        if average_tokens_per_node == 0:
            average_tokens_per_node = 200  # Fallback constant

        # Add safety buffer and calculate
        safe_average_cost = average_tokens_per_node * 1.25
        conservative_num_seeds = max(1, int(budget_tokens // safe_average_cost))

        return conservative_num_seeds
