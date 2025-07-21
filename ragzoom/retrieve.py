"""Retrieval logic with MMR diversity for RagZoom."""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from openai import OpenAI
from openai._types import NOT_GIVEN

from ragzoom.config import RagZoomConfig
from ragzoom.dynamic_frontier import DynamicFrontierGenerator, Segment, SegmentInfo
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
    frontier_segments: Optional[list["Segment"]] = None
    segment_infos: Optional[list["SegmentInfo"]] = None
    nodes: Optional[dict[str, "TreeNode"]] = (
        None  # Pre-loaded nodes to avoid redundant loading
    )


class Retriever:
    """Handles retrieval and MMR diversity for query processing."""

    def __init__(
        self,
        config: RagZoomConfig,
        store: Store,
        tree_builder: Optional["TreeBuilder"] = None,
    ):
        """Initialize retriever."""
        self.config = config
        self.store = store
        self.tree_builder = tree_builder
        self.client = OpenAI(api_key=config.openai_api_key)
        self.dp_generator = DynamicFrontierGenerator(config, store)

        # Per-request cache to avoid double refresh
        self._refreshed_node_ids: set[str] = set()

    def _get_query_embedding(self, query: str) -> list[float]:
        """Get embedding for query text."""
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=query,
                dimensions=(
                    self.config.embedding_dimensions
                    if self.config.embedding_dimensions is not None
                    else NOT_GIVEN
                ),
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise

    async def retrieve_async(
        self,
        query: str,
        n_max: Optional[int] = None,
        budget_tokens: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Async retrieval method with MMR diversity and dirty node refresh.

        Supports three modes:
        1. Budget only: Calculate conservative n_max to guarantee no overflow
        2. Budget + n_max: Use n_max but drop nodes if needed for budget
        3. n_max only: Just use n_max, no budget enforcement
        """
        # Refresh dirty nodes before retrieval
        await self._refresh_dirty_nodes_async()

        # Continue with existing logic...
        # Determine which mode we're in
        if budget_tokens is not None and n_max is None:
            # Mode 1: Budget only - calculate conservative n_max
            n_max = self._calculate_conservative_n_max(budget_tokens, document_id)
            logger.info(
                f"Budget-only mode: calculated conservative n_max={n_max} for budget={budget_tokens}"
            )
        elif budget_tokens is not None and n_max is not None:
            # Mode 2: Budget + n_max - will enforce both constraints
            logger.info(f"Mixed mode: n_max={n_max}, budget={budget_tokens}")
        elif n_max is None:
            # Mode 3: n_max only (using default)
            n_max = self.config.n_max
            logger.info(f"n_max-only mode: using n_max={n_max}")

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Step 1: Initial retrieval (2 * n_max candidates)
        k_candidates = int(n_max * self.config.mmr_k_multiplier)

        # Filter by document_id if provided
        where_filter = {"document_id": document_id} if document_id else None
        candidates = self.store.search_similar(
            query_embedding, k_candidates, where=where_filter
        )

        # Step 2: Apply MMR to get diverse n_max results
        selected_ids = self.store.compute_mmr_diverse_results(
            query_embedding, candidates, self.config.mmr_lambda, n_max
        )

        # Step 3: Build coverage map (selected + ancestors)
        coverage_map = self._build_coverage_map(selected_ids)

        # Step 4: Apply pinned nodes
        pinned_nodes = self.store.get_pinned_nodes(self.config.pin_depth_max)
        for node in pinned_nodes:
            coverage_map[node.id] = True

        # Build scores map - only include nodes in coverage map to ensure
        # DP algorithm can only use nodes from the coverage tree
        scores = {
            cand[0]: 1.0 - cand[1] for cand in candidates if cand[0] in coverage_map
        }  # Convert distance to similarity

        # Step 5: Extract frontier using DP algorithm
        final_budget = (
            budget_tokens if budget_tokens is not None else self.config.budget_tokens
        )
        dp_result = self.dp_generator.find_optimal_frontier(
            final_budget, scores, document_id, coverage_map
        )

        # Load all nodes in coverage map to avoid redundant loading later
        nodes: dict[str, TreeNode] = {}
        for node_id in coverage_map:
            maybe_node = self.store.get_node(node_id)
            if maybe_node is not None:
                nodes[node_id] = maybe_node

        return RetrievalResult(
            node_ids=selected_ids,
            scores=scores,
            coverage_map=coverage_map,  # Use the original coverage map
            frontier_segments=dp_result.segments,
            segment_infos=dp_result.segment_infos,
            nodes=nodes,
        )

    def retrieve(
        self,
        query: str,
        n_max: Optional[int] = None,
        budget_tokens: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Synchronous wrapper for retrieve_async.

        Creates a new event loop if needed to run the async version.
        For async contexts, use retrieve_async directly.
        """
        return asyncio.run(
            self.retrieve_async(query, n_max, budget_tokens, document_id)
        )

    async def _refresh_dirty_nodes_async(self, limit: Optional[int] = None) -> None:
        """Refresh dirty nodes by re-summarizing them asynchronously."""
        if not self.tree_builder:
            logger.warning(
                "No TreeBuilder available for refresh, skipping dirty node refresh"
            )
            return

        dirty_nodes = self.store.get_dirty_nodes()
        if not dirty_nodes:
            return

        # Filter out already-refreshed nodes and apply limit
        effective_limit = (
            limit if limit is not None else self.config.dirty_refresh_limit
        )
        nodes_to_refresh = []
        for node in dirty_nodes:
            if node.id not in self._refreshed_node_ids and not self.store.is_leaf_node(
                node.id
            ):
                nodes_to_refresh.append(node.id)
                if len(nodes_to_refresh) >= effective_limit:
                    break

        if not nodes_to_refresh:
            return

        logger.info(f"Refreshing {len(nodes_to_refresh)} dirty nodes")

        try:
            refreshed_count = await self.tree_builder.refresh_nodes_async(
                nodes_to_refresh
            )
            # Update cache to prevent re-refresh in same request
            self._refreshed_node_ids.update(nodes_to_refresh[:refreshed_count])
            logger.info(f"Successfully refreshed {refreshed_count} nodes")
        except Exception as e:
            logger.error(f"Error during async refresh: {e}")

    def _build_coverage_map(self, selected_ids: list[str]) -> dict[str, bool]:
        """Build coverage map including selected nodes and their ancestors."""
        coverage_map = {}

        # Mark selected nodes as covered
        for node_id in selected_ids:
            coverage_map[node_id] = True
            self.store.update_node_access(node_id)

        # Get and mark ancestors
        ancestors = self.store.get_ancestors(selected_ids)
        for ancestor in ancestors:
            coverage_map[ancestor.id] = True

        return coverage_map

    def _calculate_conservative_n_max(
        self, budget_tokens: int, document_id: Optional[str] = None
    ) -> int:
        """Calculate conservative n_max that is more grounded in document reality."""

        # Get all nodes for the document to calculate an average cost.
        # This is more realistic than a hardcoded multiplier.
        all_nodes = self.store.get_all_nodes_for_document(document_id)
        if not all_nodes:
            # Fallback to old logic if no nodes are found
            leaf_tokens = (
                self.config.leaf_tokens if self.config.leaf_tokens > 0 else 256
            )
            return max(1, budget_tokens // leaf_tokens)

        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")

        total_tokens = sum(len(tokenizer.encode(node.text)) for node in all_nodes)
        average_tokens_per_node = total_tokens / len(all_nodes)

        if average_tokens_per_node == 0:
            average_tokens_per_node = self.config.leaf_tokens

        # Add a small safety buffer (e.g., 25%) to the average to be safe
        safe_average_cost = average_tokens_per_node * 1.25

        conservative_n_max = max(1, int(budget_tokens // safe_average_cost))

        return conservative_n_max
