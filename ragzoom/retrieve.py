"""Retrieval logic with MMR diversity for RagZoom."""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from openai import OpenAI
from openai._types import NOT_GIVEN

from ragzoom.config import RagZoomConfig
from ragzoom.dynamic_frontier import DynamicFrontierGenerator, SummarySegment
from ragzoom.store import Store

if TYPE_CHECKING:
    from ragzoom.index import TreeBuilder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    node_ids: list[str]
    scores: dict[str, float]
    coverage_map: dict[str, bool]
    frontier_nodes: list[str]
    frontier_segments: Optional[list["SummarySegment"]] = None


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

        # Track access history for freshness scoring
        self.access_history: dict[str, tuple[float, int]] = (
            {}
        )  # node_id -> (similarity, turns_ago)
        self.current_turn = 0

        # Per-request cache to avoid double refresh
        self._refreshed_node_ids: set[str] = set()

        self._memo_cache: dict[
            tuple[Optional[str], int], tuple[list[SummarySegment], float]
        ] = {}

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

        # Build scores map
        scores = {
            cand[0]: 1.0 - cand[1] for cand in candidates
        }  # Convert distance to similarity

        # Step 5: Extract frontier using DP algorithm
        final_budget = (
            budget_tokens if budget_tokens is not None else self.config.budget_tokens
        )
        frontier_segments = self.dp_generator.find_optimal_frontier(
            final_budget, scores, document_id
        )
        frontier_nodes = list(set(seg.node_id for seg in frontier_segments))

        # Update access history
        self._update_access_history(selected_ids, candidates)

        return RetrievalResult(
            node_ids=selected_ids,
            scores=scores,
            coverage_map=coverage_map,
            frontier_nodes=frontier_nodes,
            frontier_segments=frontier_segments,
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
        """
        try:
            # Try to get existing event loop
            asyncio.get_running_loop()
            # We're already in an async context, can't use asyncio.run
            logger.warning(
                "retrieve() called from async context; use retrieve_async() instead"
            )
            # Fall back to sync-only refresh (skip dirty node refresh)
            return self._retrieve_sync_only(query, n_max, budget_tokens, document_id)
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self.retrieve_async(query, n_max, budget_tokens, document_id)
            )

    def _retrieve_sync_only(
        self,
        query: str,
        n_max: Optional[int] = None,
        budget_tokens: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Synchronous retrieval without dirty node refresh.

        Used as fallback when called from async context.
        """
        # Same logic as retrieve_async but without the await
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

        # Build scores map
        scores = {
            cand[0]: 1.0 - cand[1] for cand in candidates
        }  # Convert distance to similarity

        # Step 5: Extract frontier using DP algorithm
        final_budget = (
            budget_tokens if budget_tokens is not None else self.config.budget_tokens
        )
        frontier_segments = self.dp_generator.find_optimal_frontier(
            final_budget, scores, document_id
        )
        frontier_nodes = list(set(seg.node_id for seg in frontier_segments))

        # Update access history
        self._update_access_history(selected_ids, candidates)

        return RetrievalResult(
            node_ids=selected_ids,
            scores=scores,
            coverage_map=coverage_map,
            frontier_nodes=frontier_nodes,
            frontier_segments=frontier_segments,
        )

    async def _refresh_dirty_nodes_async(self, limit: int = 10) -> None:
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
        nodes_to_refresh = []
        for node in dirty_nodes:
            if node.id not in self._refreshed_node_ids and node.depth > 0:
                nodes_to_refresh.append(node.id)
                if len(nodes_to_refresh) >= limit:
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

    def _update_access_history(
        self, selected_ids: list[str], candidates: list[tuple[str, float, dict]]
    ) -> None:
        """Update access history for freshness scoring."""
        self.current_turn += 1

        # Update selected nodes
        for node_id in selected_ids:
            # Find similarity score
            sim_score = 1.0
            for cand in candidates:
                if cand[0] == node_id:
                    sim_score = 1.0 - cand[1]  # Convert distance to similarity
                    break

            self.access_history[node_id] = (sim_score, 0)

        # Age existing entries
        for node_id in list(self.access_history.keys()):
            if node_id not in selected_ids:
                sim, turns_ago = self.access_history[node_id]
                self.access_history[node_id] = (sim, turns_ago + 1)

                # Remove old entries based on TTL
                if self.config.ttl_turns > 0 and turns_ago + 1 > self.config.ttl_turns:
                    del self.access_history[node_id]

    def get_priority_scores(self) -> dict[str, float]:
        """Calculate priority scores for sliding queue eviction."""
        priority_scores = {}

        # Guard against div-by-zero if decay is 0
        decay = self.config.freshness_decay
        if decay <= 0 or decay > 1:
            decay = 0.9  # Safe default
            logger.warning(
                f"Invalid freshness_decay {self.config.freshness_decay}, using 0.9"
            )

        for node_id, (similarity, turns_ago) in self.access_history.items():
            # Priority = similarity * decay^turns_ago
            priority = similarity * (decay**turns_ago)
            # Clamp to [0, 1] range (similarity might be > 1 from Chroma)
            priority = max(0.0, min(1.0, priority))
            priority_scores[node_id] = priority

        return priority_scores

    async def retrieve_with_eviction_async(
        self,
        query: str,
        token_budget: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Async retrieve with sliding queue eviction to fit token budget."""
        if token_budget is None:
            token_budget = self.config.budget_tokens

        # Initial retrieval
        result = await self.retrieve_async(query, document_id=document_id)

        # Calculate token usage
        total_tokens = 0
        node_tokens = {}

        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")

        for node_id in result.frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                tokens = len(tokenizer.encode(node.text))
                node_tokens[node_id] = tokens
                total_tokens += tokens

        # If within budget, return as-is
        if total_tokens <= token_budget:
            return result

        # Need eviction - get priority scores
        priority_scores = self.get_priority_scores()

        # Sort frontier nodes by priority (lowest first for eviction)
        frontier_with_priority = []
        for node_id in result.frontier_nodes:
            priority = priority_scores.get(node_id, 0.0)
            frontier_with_priority.append((priority, node_id))

        frontier_with_priority.sort()

        # Evict nodes until within budget
        evicted = set()
        for priority, node_id in frontier_with_priority:
            if total_tokens <= token_budget:
                break

            if node_id in node_tokens:
                total_tokens -= node_tokens[node_id]
                evicted.add(node_id)

        # Update frontier
        result.frontier_nodes = [
            nid for nid in result.frontier_nodes if nid not in evicted
        ]

        return result

    def retrieve_with_eviction(
        self,
        query: str,
        token_budget: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Sync wrapper for retrieve_with_eviction_async."""
        try:
            # Try to get existing event loop
            asyncio.get_running_loop()
            # We're already in an async context
            logger.warning(
                "retrieve_with_eviction() called from async context; use retrieve_with_eviction_async() instead"
            )
            # Fall back to regular retrieve (without dirty refresh)
            return self._retrieve_with_eviction_sync_only(
                query, token_budget, document_id
            )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(
                self.retrieve_with_eviction_async(query, token_budget, document_id)
            )

    def _retrieve_with_eviction_sync_only(
        self,
        query: str,
        token_budget: Optional[int] = None,
        document_id: Optional[str] = None,
    ) -> RetrievalResult:
        """Sync-only version of retrieve_with_eviction without dirty refresh."""
        if token_budget is None:
            token_budget = self.config.budget_tokens

        # Initial retrieval (without dirty refresh)
        result = self._retrieve_sync_only(query, document_id=document_id)

        # Calculate token usage
        total_tokens = 0
        node_tokens = {}

        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")

        for node_id in result.frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                tokens = len(tokenizer.encode(node.text))
                node_tokens[node_id] = tokens
                total_tokens += tokens

        # If within budget, return as-is
        if total_tokens <= token_budget:
            return result

        # Need eviction - get priority scores
        priority_scores = self.get_priority_scores()

        # Sort frontier nodes by priority (lowest first for eviction)
        frontier_with_priority = []
        for node_id in result.frontier_nodes:
            priority = priority_scores.get(node_id, 0.0)
            frontier_with_priority.append((priority, node_id))

        frontier_with_priority.sort()

        # Evict nodes until within budget
        evicted = set()
        for priority, node_id in frontier_with_priority:
            if total_tokens <= token_budget:
                break

            if node_id in node_tokens:
                total_tokens -= node_tokens[node_id]
                evicted.add(node_id)

        # Update frontier
        result.frontier_nodes = [
            nid for nid in result.frontier_nodes if nid not in evicted
        ]

        return result

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
