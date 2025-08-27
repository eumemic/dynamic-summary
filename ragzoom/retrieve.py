"""Retrieval logic with MMR diversity for RagZoom."""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from openai import OpenAI

from ragzoom.config import IndexConfig, QueryConfig, SecretStr
from ragzoom.document_store import DocumentStore
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from ragzoom.retrieval import (
    BudgetPlanner,
    CoverageBuilder,
    EmbeddingService,
    ScoringService,
)
from ragzoom.retrieval.telemetry_collector import TelemetryCollector
from ragzoom.store import StoreManager, TreeNode
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
        store: StoreManager | DocumentStore,
        api_key: str | SecretStr = "",
        tree_builder: Optional["TreeBuilder"] = None,
        use_async_dp: bool = False,
        min_nodes_for_parallel: int = 10,
    ):
        """Initialize retriever.

        Args:
            query_config: Query configuration
            store: StoreManager instance
            api_key: OpenAI API key as SecretStr or string (if not provided, reads from OPENAI_API_KEY env)
            tree_builder: Optional TreeBuilder instance
            use_async_dp: Whether to use async DP generator for parallelization
            min_nodes_for_parallel: Minimum nodes in subtree to enable parallelization
        """
        self.query_config = query_config
        self.store = store
        self.use_async_dp = use_async_dp

        # Type annotation for async_dp_generator
        from ragzoom.dynamic_tiling import AsyncDynamicTilingGenerator

        self.async_dp_generator: AsyncDynamicTilingGenerator | None

        # Get API key from parameter or environment
        from ragzoom.config import ensure_secret_str

        actual_key = ensure_secret_str(api_key, "Retriever")

        self.client = OpenAI(api_key=actual_key)
        self.dp_generator = DynamicTilingGenerator(query_config)

        # Initialize async generator if requested
        if use_async_dp:
            self.async_dp_generator = AsyncDynamicTilingGenerator(
                query_config, min_nodes_for_parallel
            )
        else:
            self.async_dp_generator = None

        # Initialize services
        self.embedding_service = EmbeddingService(
            self.client, store, query_config.embedding_model
        )
        self.coverage_builder = CoverageBuilder(store)
        self.scoring_service = ScoringService(store)

        # Get default chunk size from IndexConfig for budget planning
        index_config = IndexConfig.load()
        self.budget_planner = BudgetPlanner(store, index_config.target_chunk_tokens)

    async def retrieve_async(
        self,
        query: str,
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
        telemetry_collector: TelemetryCollector | None = None,
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
        # Start telemetry if collector is provided
        if telemetry_collector:
            telemetry_collector.start_phase()

        # Determine effective document scope
        effective_doc_id = (
            getattr(self.store, "document_id", None)
            if isinstance(self.store, DocumentStore)
            else document_id
        )

        # Determine which mode we're in
        if budget_tokens is not None and num_seeds is None:
            num_seeds = self.budget_planner.calculate_conservative_num_seeds(
                budget_tokens, effective_doc_id
            )
            logger.info(
                f"Budget-only mode: calculated conservative num_seeds={num_seeds} for budget={budget_tokens}"
            )
        elif budget_tokens is not None and num_seeds is not None:
            logger.info(f"Mixed mode: num_seeds={num_seeds}, budget={budget_tokens}")
        elif num_seeds is None:
            num_seeds = self.budget_planner.calculate_conservative_num_seeds(
                self.query_config.budget_tokens, effective_doc_id
            )
            logger.info(
                f"Default mode: calculated conservative num_seeds={num_seeds} from budget={self.query_config.budget_tokens}"
            )

        if telemetry_collector:
            telemetry_collector.record_metric("seeds_requested", num_seeds)

        # Phase 1: Get query embedding
        query_embedding = self.embedding_service.get_query_embedding(
            query, effective_doc_id
        )
        if telemetry_collector:
            telemetry_collector.end_phase("embedding")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric(
                "embedding_model", self.query_config.embedding_model
            )

        # Phase 2: Initial retrieval
        k_candidates = int(num_seeds * self.query_config.mmr_k_multiplier)
        where_filter = {"document_id": effective_doc_id} if effective_doc_id else None
        candidates = self.store.search.search_similar(
            query_embedding, k_candidates, where=where_filter
        )
        if telemetry_collector:
            telemetry_collector.end_phase("search")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric("candidates_retrieved", len(candidates))

        # Phase 3: Apply MMR
        selected_ids = self.store.search.compute_mmr_diverse_results(
            query_embedding, candidates, self.query_config.mmr_lambda, num_seeds
        )
        if telemetry_collector:
            telemetry_collector.end_phase("mmr")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric("seeds_found", len(selected_ids))

        # Phase 4: Build coverage map with document scoping when available
        if isinstance(self.store, DocumentStore) and effective_doc_id:
            coverage_map = CoverageBuilder(self.store).build_complete_coverage_map(
                selected_ids
            )
        elif effective_doc_id:
            # self.store is StoreManager in this branch
            doc_store = self.store.for_document(effective_doc_id)  # type: ignore[union-attr]
            coverage_map = CoverageBuilder(doc_store).build_complete_coverage_map(
                selected_ids
            )
        else:
            coverage_map = self.coverage_builder.build_complete_coverage_map(
                selected_ids
            )
        if telemetry_collector:
            telemetry_collector.end_phase("coverage_map")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric("coverage_size", len(coverage_map))

        # Phase 5: Build scores map
        scores = self.scoring_service.compute_scores(
            query_embedding, coverage_map, candidates
        )
        if telemetry_collector:
            telemetry_collector.end_phase("scoring")
            telemetry_collector.start_phase()

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
            loaded_nodes = self.store.nodes.get_nodes(node_ids_to_load)
            for node in loaded_nodes:
                nodes[node.id] = node

        # Find the root node
        root_id = None
        for node_id, node in nodes.items():
            if node.is_root() or node.parent_id not in nodes:
                root_id = node_id
                break

        if not root_id:
            raise ValueError(
                f"Failed to find root node in coverage map with {len(nodes)} nodes. "
                f"This indicates the coverage tree is incomplete - all ancestors should be included "
                f"up to the document root. Selected node IDs: {selected_ids[:5]}{'...' if len(selected_ids) > 5 else ''}"
            )

        # Phase 6: Extract tiling using DP algorithm
        final_budget = (
            budget_tokens
            if budget_tokens is not None
            else self.query_config.budget_tokens
        )

        # Use async DP generator if available, otherwise use sync version
        if self.async_dp_generator is not None:
            dp_result = await self.async_dp_generator.find_optimal_tiling(
                final_budget, scores, nodes, root_id
            )
        else:
            dp_result = self.dp_generator.find_optimal_tiling(
                final_budget, scores, nodes, root_id
            )
        if telemetry_collector:
            telemetry_collector.end_phase("dp")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric(
                "tiling_size", len(dp_result.tiling.node_ids)
            )

        # Calculate output tokens
        output_tokens = sum(
            nodes[node_id].token_count
            for node_id in dp_result.tiling.node_ids
            if node_id in nodes
        )
        if telemetry_collector:
            telemetry_collector.record_metric("output_tokens", output_tokens)

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
        collector.start_query(query, num_seeds, budget_tokens, document_id)

        result = await self.retrieve_async(
            query, num_seeds, budget_tokens, document_id, collector
        )
        telemetry = collector.finalize()
        if telemetry is None:
            raise RuntimeError("Telemetry collection failed")
        return result, telemetry
