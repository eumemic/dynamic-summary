"""Retrieval logic with MMR diversity for RagZoom."""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from openai import OpenAI

from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from ragzoom.retrieval import (
    BudgetPlanner,
    CoverageBuilder,
    EmbeddingService,
    ScoringService,
)
from ragzoom.retrieval.telemetry_decorator import TelemetryCollector
from ragzoom.store import Store, TreeNode
from ragzoom.telemetry_query import QueryTelemetry

if TYPE_CHECKING:
    from ragzoom.index import TreeBuilder

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    node_ids: list[str]
    scores: dict[str, float]
    coverage_map: dict[str, bool]
    tiling: list[str] | None = None
    nodes: dict[str, "TreeNode"] | None = None


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

        import os

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key required for Retriever")

        self.client = OpenAI(api_key=api_key)
        self.dp_generator = DynamicTilingGenerator(query_config)

        # Initialize services
        self.embedding_service = EmbeddingService(
            self.client, store, query_config.embedding_model
        )
        self.coverage_builder = CoverageBuilder(store)
        self.scoring_service = ScoringService(store)

        # Get default chunk size from IndexConfig for budget planning
        index_config = IndexConfig.load()
        self.budget_planner = BudgetPlanner(store, index_config.target_chunk_tokens)

    def _record_telemetry_phase(self, phase_name: str) -> None:
        """Record telemetry phase timing if collector is available."""
        if hasattr(self, "_telemetry_collector"):
            self._telemetry_collector.end_phase(phase_name)
            self._telemetry_collector.start_phase()

    def _record_telemetry_metric(self, metric_name: str, value) -> None:
        """Record telemetry metric if collector is available."""
        if hasattr(self, "_telemetry_collector"):
            self._telemetry_collector.record_metric(metric_name, value)

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
        # Start telemetry if collector is available
        if hasattr(self, "_telemetry_collector"):
            self._telemetry_collector.start_phase()

        # Determine which mode we're in
        if budget_tokens is not None and num_seeds is None:
            num_seeds = self.budget_planner.calculate_conservative_num_seeds(
                budget_tokens, document_id
            )
            logger.info(
                f"Budget-only mode: calculated conservative num_seeds={num_seeds} for budget={budget_tokens}"
            )
        elif budget_tokens is not None and num_seeds is not None:
            logger.info(f"Mixed mode: num_seeds={num_seeds}, budget={budget_tokens}")
        elif num_seeds is None:
            num_seeds = self.budget_planner.calculate_conservative_num_seeds(
                self.query_config.budget_tokens, document_id
            )
            logger.info(
                f"Default mode: calculated conservative num_seeds={num_seeds} from budget={self.query_config.budget_tokens}"
            )

        self._record_telemetry_metric("seeds_requested", num_seeds)

        # Phase 1: Get query embedding
        query_embedding = self.embedding_service.get_query_embedding(query, document_id)
        self._record_telemetry_phase("embedding")
        self._record_telemetry_metric(
            "embedding_model", self.query_config.embedding_model
        )

        # Phase 2: Initial retrieval
        k_candidates = int(num_seeds * self.query_config.mmr_k_multiplier)
        where_filter = {"document_id": document_id} if document_id else None
        candidates = self.store.search_similar(
            query_embedding, k_candidates, where=where_filter
        )
        self._record_telemetry_phase("search")
        self._record_telemetry_metric("candidates_retrieved", len(candidates))

        # Phase 3: Apply MMR
        selected_ids = self.store.compute_mmr_diverse_results(
            query_embedding, candidates, self.query_config.mmr_lambda, num_seeds
        )
        self._record_telemetry_phase("mmr")
        self._record_telemetry_metric("seeds_found", len(selected_ids))

        # Phase 4: Build coverage map
        coverage_map = self.coverage_builder.build_complete_coverage_map(selected_ids)
        self._record_telemetry_phase("coverage_map")
        self._record_telemetry_metric("coverage_size", len(coverage_map))

        # Phase 5: Build scores map
        scores = self.scoring_service.compute_scores(
            query_embedding, coverage_map, candidates
        )
        self._record_telemetry_phase("scoring")

        # Handle empty coverage map case
        if not coverage_map:
            return RetrievalResult(
                node_ids=selected_ids,
                scores=scores,
                coverage_map=coverage_map,
                tiling=[],
                nodes={},
            )

        # Load all nodes in coverage map
        nodes: dict[str, TreeNode] = {}
        node_ids_to_load = list(coverage_map.keys())
        if node_ids_to_load:
            loaded_nodes = self.store.get_nodes(node_ids_to_load)
            for node in loaded_nodes:
                nodes[node.id] = node

        # Find the root node
        root_id = None
        for node_id, node in nodes.items():
            if node.parent_id is None or node.parent_id not in nodes:
                root_id = node_id
                break

        if not root_id:
            raise ValueError(
                f"No root node found in coverage map. Coverage map has {len(nodes)} nodes but none have no parent in the map."
            )

        # Phase 6: Extract tiling using DP algorithm
        final_budget = (
            budget_tokens
            if budget_tokens is not None
            else self.query_config.budget_tokens
        )
        dp_result = self.dp_generator.find_optimal_tiling(
            final_budget, scores, nodes, root_id
        )
        self._record_telemetry_phase("dp")
        self._record_telemetry_metric("tiling_size", len(dp_result.tiling.node_ids))

        # Calculate output tokens
        output_tokens = sum(
            nodes[node_id].token_count
            for node_id in dp_result.tiling.node_ids
            if node_id in nodes
        )
        self._record_telemetry_metric("output_tokens", output_tokens)

        return RetrievalResult(
            node_ids=selected_ids,
            scores=scores,
            coverage_map=coverage_map,
            tiling=dp_result.tiling.node_ids,
            nodes=nodes,
        )

    # jscpd:ignore-start - Sync wrapper for async method (legitimate duplication pattern)
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

    async def retrieve_with_telemetry(
        self,
        query: str,
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
    ) -> tuple[RetrievalResult, QueryTelemetry]:
        """Async retrieval with detailed telemetry collection.

        Args:
            query: Query text to search for
            num_seeds: Number of seed nodes to retrieve
            budget_tokens: Token budget for the final summary
            document_id: Optional document ID to filter by

        Returns:
            Tuple of (RetrievalResult, QueryTelemetry) with detailed timing info
        """
        collector = TelemetryCollector()
        self._telemetry_collector = collector

        collector.start_query(query, num_seeds, budget_tokens, document_id)

        try:
            result = await self.retrieve_async(
                query, num_seeds, budget_tokens, document_id
            )
            telemetry = collector.finalize()
            if telemetry is None:
                raise RuntimeError("Telemetry collection failed")
            return result, telemetry
        finally:
            if hasattr(self, "_telemetry_collector"):
                delattr(self, "_telemetry_collector")
