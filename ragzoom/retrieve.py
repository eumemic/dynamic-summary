"""Retrieval logic with MMR diversity for RagZoom."""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragzoom.config import QueryConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_filter import (
    DocumentIdFilter,
    SpanEndLtFilter,
    SpanOverlapsFilter,
    VectorFilter,
)
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from ragzoom.error_handling import handle_graceful_error
from ragzoom.greedy_tiling import GreedyTilingGenerator
from ragzoom.retrieval import (
    BudgetPlanner,
    CoverageBuilder,
    EmbeddingService,
    ScoringService,
)
from ragzoom.retrieval.telemetry_collector import TelemetryCollector
from ragzoom.telemetry_query import QueryTelemetry

if TYPE_CHECKING:
    from ragzoom.contracts.vector_index import VectorIndex
    from ragzoom.document_store import DocumentStore
    from ragzoom.vector_api import Vector

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""

    node_ids: list[str]
    scores: dict[str, float]
    coverage_map: dict[str, bool]
    tiling: list[str] | None = None
    nodes: dict[str, "TreeNode"] | None = None
    seed_count: int = 0
    verbatim_count: int = 0
    actual_start: int = 0
    actual_end: int | None = None


class Retriever:
    """Handles retrieval and MMR diversity for query processing."""

    def __init__(
        self,
        query_config: QueryConfig,
        document_store: "DocumentStore",
        embedding_service: EmbeddingService,
        budget_planner: BudgetPlanner,
        vector_index: "VectorIndex",
        use_async_dp: bool = False,
        min_nodes_for_parallel: int = 10,
    ):
        """Initialize retriever.

        Args:
            query_config: Query configuration
            document_store: DocumentStore instance for document-scoped operations
            embedding_service: Service for generating query embeddings
            budget_planner: Service for calculating conservative seed counts
            use_async_dp: Whether to use async DP generator for parallelization
            min_nodes_for_parallel: Minimum nodes in subtree to enable parallelization
        """
        self.query_config = query_config
        self.document_store = document_store
        self.embedding_service = embedding_service
        self.budget_planner = budget_planner
        self.use_async_dp = use_async_dp
        # Backend-agnostic vector index
        self.vector_index = vector_index

        # Type annotation for async_dp_generator
        from ragzoom.dynamic_tiling import AsyncDynamicTilingGenerator

        self.async_dp_generator: AsyncDynamicTilingGenerator | None

        self.dp_generator = DynamicTilingGenerator(query_config)
        self.greedy_generator = GreedyTilingGenerator(query_config)

        # Initialize async generator if requested
        if use_async_dp:
            self.async_dp_generator = AsyncDynamicTilingGenerator(
                query_config, min_nodes_for_parallel
            )
        else:
            self.async_dp_generator = None

    # ------------------------------------------------------------------
    # Helper methods for shared retrieval logic
    # ------------------------------------------------------------------

    def _filter_stale_vectors(self, raw_candidates: list["Vector"]) -> list["Vector"]:
        """Filter out vectors that don't exist in storage to preserve invariants."""
        cand_ids = [v.id for v in raw_candidates]
        existing_nodes = self.document_store.nodes.get_nodes(cand_ids)
        existing_ids = {n.id for n in existing_nodes}
        return [v for v in raw_candidates if v.id in existing_ids]

    def _build_candidates_for_scoring(
        self,
        query_embedding: list[float],
        vec_candidates: list["Vector"],
    ) -> list[tuple[str, float, dict[str, str | int | float | bool | None]]]:
        """Build legacy-shaped candidates list for scoring service compatibility."""
        from typing import cast as _cast

        from ragzoom.retrieval import similarity as sim

        rels = sim.relevance_scores(query_embedding, vec_candidates)
        candidates: list[
            tuple[str, float, dict[str, str | int | float | bool | None]]
        ] = []
        for i, v in enumerate(vec_candidates):
            md = v.meta
            candidates.append(
                (
                    v.id,
                    float(rels[i]),
                    {
                        "span_start": _cast(int, md["span_start"]),
                        "span_end": _cast(int, md["span_end"]),
                        "parent_id": _cast(str, md["parent_id"]),
                        "document_id": _cast(str, md["document_id"]),
                        "is_leaf": _cast(int, md["is_leaf"]),
                    },
                )
            )
        return candidates

    def _load_coverage_nodes(
        self,
        coverage_map: dict[str, bool],
        base_nodes: dict[str, TreeNode],
    ) -> dict[str, TreeNode]:
        """Load all nodes in coverage map, supplementing base_nodes as needed."""
        nodes: dict[str, TreeNode] = dict(base_nodes)
        missing_ids = [nid for nid in coverage_map.keys() if nid not in nodes]
        if missing_ids:
            loaded_nodes = self.document_store.nodes.get_nodes(missing_ids)
            for node in loaded_nodes:
                nodes[node.id] = node
        return nodes

    def _find_coverage_root_ids(self, nodes: dict[str, TreeNode]) -> list[str]:
        """Find root node IDs in the coverage map (nodes without parents in coverage)."""
        root_ids: list[str] = []
        for node_id, node_p in nodes.items():
            parent_id = node_p.parent_id
            is_root = getattr(node_p, "is_root", lambda: parent_id is None)()
            if is_root or parent_id not in nodes:
                root_ids.append(node_id)
        return root_ids

    # ------------------------------------------------------------------
    # Main retrieval methods
    # ------------------------------------------------------------------

    async def retrieve_async(
        self,
        query: str,
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
        recent_verbatim_budget: int | None = None,
        telemetry_collector: TelemetryCollector | None = None,
        span_start: int = 0,
        span_end: int | None = None,
    ) -> RetrievalResult:
        """Async retrieval method with MMR diversity.

        Args:
            query: Query text to search for
            num_seeds: Number of seed nodes to retrieve
            budget_tokens: Token budget for the final summary
            document_id: Optional document ID to filter by
            recent_verbatim_budget: Token budget for recent leaves to include verbatim
            span_start: Start of document window (character position, default 0)
            span_end: End of document window (default: document end)

        Supports three modes:
        1. Budget only: Calculate conservative num_seeds to guarantee no overflow
        2. Budget + num_seeds: Use num_seeds but drop nodes if needed for budget
        3. num_seeds only: Just use num_seeds, no budget enforcement

        When span_start/span_end are specified, the tiling covers exactly the
        minimal span [actual_start, actual_end) that contains the requested window.
        """
        # Start telemetry if collector is provided
        if telemetry_collector:
            telemetry_collector.start_phase()

        # Determine effective document scope
        effective_doc_id = (
            getattr(self.document_store, "document_id", None) or document_id
        )

        # Compute window bounds if windowed query
        from ragzoom.retrieval.coverage_builder import WindowBounds

        window_bounds: WindowBounds | None = None
        actual_start = 0
        actual_end: int | None = None

        # Get document span end for defaults and validation
        nodes_wrapper = getattr(self.document_store, "nodes", None)
        repo = getattr(nodes_wrapper, "_repo", None) if nodes_wrapper else None
        doc_span_end: int | None = None
        if repo is not None and effective_doc_id:
            doc_span_end = repo.get_document_span_end(effective_doc_id)

        # Resolve span_end default
        resolved_span_end = span_end if span_end is not None else doc_span_end

        # Check if we have a windowed query (not default full document)
        # Note: isinstance checks guard against mocked repos in tests
        is_windowed = span_start > 0 or (
            isinstance(resolved_span_end, int)
            and isinstance(doc_span_end, int)
            and resolved_span_end < doc_span_end
        )

        if is_windowed and effective_doc_id and resolved_span_end is not None:
            # Validate window bounds
            if span_start >= resolved_span_end:
                raise ValueError(
                    f"span_start ({span_start}) must be less than span_end ({resolved_span_end})"
                )
            if doc_span_end is not None and span_start >= doc_span_end:
                raise ValueError(
                    f"span_start ({span_start}) exceeds document length ({doc_span_end})"
                )

            # Compute window bounds
            doc_coverage_builder = CoverageBuilder(self.document_store)
            window_bounds = doc_coverage_builder.compute_window_bounds(
                span_start, resolved_span_end, effective_doc_id
            )
            actual_start = window_bounds.actual_start
            actual_end = window_bounds.actual_end
        else:
            # Full document query - actual_start remains 0
            actual_end = doc_span_end

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

        # Pre-select verbatim leaves to establish horizon for seed filtering
        # Seeds should only come from the non-verbatim region (before the horizon)
        pinned_ids: set[str] = set()
        verbatim_horizon: int | None = None
        if recent_verbatim_budget and recent_verbatim_budget > 0:
            # For windowed queries, count back from actual_end instead of document end
            if (
                is_windowed
                and actual_end is not None
                and repo is not None
                and effective_doc_id
            ):
                verbatim_leaves = repo.get_recent_leaves_within_budget_before(
                    effective_doc_id, recent_verbatim_budget, actual_end
                )
            else:
                verbatim_leaves = (
                    self.document_store.nodes.get_recent_leaves_within_budget(
                        recent_verbatim_budget
                    )
                )
            pinned_ids = {leaf.id for leaf in verbatim_leaves}
            if verbatim_leaves:
                # Horizon = start of verbatim region (earliest span_start among verbatim)
                # Leaves are returned sorted by span_start, so first one is the horizon
                verbatim_horizon = verbatim_leaves[0].span_start

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

        # Phase 2: Initial retrieval (always via VectorIndex v2)
        # Filter seeds to only come from before verbatim horizon
        k_candidates = int(num_seeds * self.query_config.mmr_k_multiplier)
        from ragzoom.retrieval import mmr

        filters: list[VectorFilter] = []
        if effective_doc_id:
            filters.append(DocumentIdFilter(effective_doc_id))
        if verbatim_horizon is not None:
            filters.append(SpanEndLtFilter(verbatim_horizon))
        if is_windowed and actual_end is not None:
            filters.append(SpanOverlapsFilter(actual_start, actual_end))
        raw_candidates = self.vector_index.search_similar(
            query_embedding,
            k_candidates,
            filters if filters else None,
        )
        # Filter out stale vectors that don't exist in storage to preserve invariants
        vec_candidates = self._filter_stale_vectors(raw_candidates)
        if telemetry_collector:
            telemetry_collector.record_metric(
                "candidates_filtered", len(vec_candidates)
            )
        if telemetry_collector:
            telemetry_collector.end_phase("search")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric(
                "candidates_retrieved", len(vec_candidates)
            )

        # Phase 3: Apply MMR selection using generic math over canonical vectors
        selected_ids = mmr.select_diverse(
            query_embedding,
            vec_candidates,
            num_seeds,
            self.query_config.mmr_lambda,
        )
        seed_count = len(selected_ids)
        verbatim_count = len(pinned_ids) if pinned_ids else 0

        # Add verbatim leaves to selected_ids so coverage includes them
        # (verbatim leaves were pre-selected earlier to establish the horizon)
        if pinned_ids:
            selected_ids_set = set(selected_ids)
            for leaf_id in pinned_ids:
                if leaf_id not in selected_ids_set:
                    selected_ids.append(leaf_id)

        # Build legacy-shaped candidates for scoring service compatibility
        candidates = self._build_candidates_for_scoring(query_embedding, vec_candidates)
        if telemetry_collector:
            telemetry_collector.end_phase("mmr")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric("seeds_found", len(selected_ids))

        # Phase 4: Build coverage map
        # Use document-scoped coverage builder
        seed_meta_all = {v.id: v.meta for v in vec_candidates}
        seed_metadata = {
            node_id: seed_meta_all[node_id]
            for node_id in selected_ids
            if node_id in seed_meta_all
        }
        doc_coverage_builder = CoverageBuilder(self.document_store)
        if window_bounds is not None:
            coverage_result = doc_coverage_builder.build_windowed_coverage(
                selected_ids,
                window_bounds,
                seed_metadata=seed_metadata,
                pinned_ids=pinned_ids,
            )
        else:
            coverage_result = doc_coverage_builder.build_complete_coverage(
                selected_ids, seed_metadata=seed_metadata, pinned_ids=pinned_ids
            )
        coverage_map = coverage_result.coverage_map
        if telemetry_collector:
            telemetry_collector.end_phase("coverage_map")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric("coverage_size", len(coverage_map))

        # Phase 5: Build scores map
        # Use document-scoped scoring service
        doc_scoring_service = ScoringService(self.document_store, self.vector_index)
        scores = doc_scoring_service.compute_scores(
            query_embedding, coverage_map, candidates, nodes=coverage_result.nodes
        )
        if telemetry_collector:
            telemetry_collector.end_phase("scoring")
            telemetry_collector.start_phase()

        # Load all nodes in coverage map
        nodes = self._load_coverage_nodes(coverage_map, coverage_result.nodes)

        # Find all root nodes (supporting forests)
        root_ids = self._find_coverage_root_ids(nodes)

        if not root_ids:
            raise ValueError(
                f"Failed to find root nodes in coverage map with {len(nodes)} nodes. "
                f"This indicates the coverage tree is incomplete - all ancestors should be included "
                f"up to the document root. Selected node IDs: {selected_ids[:5]}{'...' if len(selected_ids) > 5 else ''}"
            )

        root_ids.sort(
            key=lambda node_id: (
                int(getattr(nodes[node_id], "span_start", 0)),
                node_id,
            )
        )

        # Sanity: drop any selected ids that aren't present in the document store
        selected_ids = [nid for nid in selected_ids if nid in nodes]

        # Phase 6: Extract tiling using DP/greedy algorithm
        # Combined budget = base + verbatim (pinned nodes are in coverage, can't be rolled up)
        base_budget = (
            budget_tokens
            if budget_tokens is not None
            else self.query_config.budget_tokens
        )
        final_budget = base_budget + (recent_verbatim_budget or 0)

        # Choose tiling strategy
        tiling_strategy = self.query_config.tiling_strategy
        if tiling_strategy == "dp":
            import warnings

            warnings.warn(
                "tiling_strategy='dp' is deprecated and will be removed in a future version. "
                "The greedy algorithm produces better results in practice because it doesn't "
                "rely on heuristic budget allocation. Use tiling_strategy='greedy' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if tiling_strategy == "greedy":
            dp_result = self.greedy_generator.find_optimal_tiling_over_roots(
                root_ids, final_budget, scores, nodes
            )
        elif self.async_dp_generator is not None:
            dp_result = await self.async_dp_generator.find_optimal_tiling_over_roots(
                root_ids, final_budget, scores, nodes, pinned_ids=pinned_ids
            )
        else:
            dp_result = self.dp_generator.find_optimal_tiling_over_roots(
                root_ids, final_budget, scores, nodes, pinned_ids=pinned_ids
            )
        if telemetry_collector:
            telemetry_collector.end_phase("dp")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric(
                "tiling_size", len(dp_result.tiling.node_ids)
            )

        tiling_ids = list(dp_result.tiling.node_ids)

        # Ensure all tiling nodes are present in the preloaded set
        missing_in_nodes = [nid for nid in tiling_ids if nid not in nodes]
        if missing_in_nodes:
            try:
                loaded_more = self.document_store.nodes.get_nodes(missing_in_nodes)
                for n in loaded_more:
                    nodes[n.id] = n
            except Exception as exc:
                handle_graceful_error(
                    exc, "Failed to load missing tiling nodes", default=None
                )

        # Calculate output tokens (after best-effort completion)
        output_tokens = sum(
            nodes[node_id].token_count for node_id in tiling_ids if node_id in nodes
        )
        if telemetry_collector:
            telemetry_collector.record_metric("output_tokens", output_tokens)

        return RetrievalResult(
            node_ids=selected_ids,
            scores=scores,
            coverage_map=coverage_map,
            tiling=tiling_ids,
            nodes=nodes,
            seed_count=seed_count,
            verbatim_count=verbatim_count,
            actual_start=actual_start,
            actual_end=actual_end,
        )

    # jscpd:ignore-start - Sync wrapper for async method (legitimate duplication pattern)
    def retrieve(
        self,
        query: str,
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
        recent_verbatim_budget: int | None = None,
        span_start: int = 0,
        span_end: int | None = None,
    ) -> RetrievalResult:
        """Synchronous wrapper for retrieve_async.

        Args:
            query: Query text to search for
            num_seeds: Number of seed nodes to retrieve
            budget_tokens: Token budget for the final summary
            document_id: Optional document ID to filter by
            recent_verbatim_budget: Token budget for recent leaves to include verbatim
            span_start: Start of document window (character position, default 0)
            span_end: End of document window (default: document end)

        Creates a new event loop if needed to run the async version.
        For async contexts, use retrieve_async directly.
        """
        # jscpd:ignore-end
        return asyncio.run(
            self.retrieve_async(
                query,
                num_seeds,
                budget_tokens,
                document_id,
                recent_verbatim_budget,
                span_start=span_start,
                span_end=span_end,
            )
        )

    async def retrieve_for_context(
        self,
        query_text: str,
        span_end_limit: int,
        budget_tokens: int,
        document_id: str | None = None,
        recent_verbatim_token_budget: int = 0,
    ) -> str:
        """Retrieve and assemble context for nodes before span_end_limit.

        Used during indexing to build preceding_context for nodes. Queries the
        existing tree using the node's own text as the query, returning assembled
        text from the semantically relevant preceding content.

        Args:
            query_text: The text to use as query (typically the node's own text)
            span_end_limit: Only include nodes where span_end < this value
            budget_tokens: Token budget for the dynamic summary portion
            document_id: Optional document ID to filter by
            recent_verbatim_token_budget: Token budget for verbatim leaves from
                the unfinished region (derived from last_eligible_span_start - frontier).
                Currently unused; reserved for future enhancement to include
                recent verbatim leaf content alongside the dynamic summary.

        Returns:
            Assembled text from the tiled retrieval result, or empty string if
            no relevant preceding content exists.
        """
        # Note: recent_verbatim_token_budget is reserved for future use.
        # The current implementation uses only budget_tokens for the dynamic summary.
        _ = recent_verbatim_token_budget  # Suppress unused warning
        # Determine effective document scope
        effective_doc_id = (
            getattr(self.document_store, "document_id", None) or document_id
        )

        # Calculate conservative num_seeds for the budget
        num_seeds = self.budget_planner.calculate_conservative_num_seeds(
            budget_tokens, effective_doc_id
        )
        if num_seeds == 0:
            return ""

        # Get query embedding from the node's text (async for concurrency)
        query_embedding = await self.embedding_service.get_query_embedding_async(
            query_text, effective_doc_id
        )

        # Build filters: document scope + span_end < limit
        filters: list[VectorFilter] = [SpanEndLtFilter(span_end_limit)]
        if effective_doc_id:
            filters.append(DocumentIdFilter(effective_doc_id))

        # Search for candidates from preceding tree only
        k_candidates = int(num_seeds * self.query_config.mmr_k_multiplier)
        raw_candidates = self.vector_index.search_similar(
            query_embedding,
            k_candidates,
            filters,
        )

        if not raw_candidates:
            return ""

        # Filter out stale vectors
        vec_candidates = self._filter_stale_vectors(raw_candidates)

        if not vec_candidates:
            return ""

        # Apply MMR selection
        from ragzoom.retrieval import mmr

        selected_ids = mmr.select_diverse(
            query_embedding,
            vec_candidates,
            num_seeds,
            self.query_config.mmr_lambda,
        )

        if not selected_ids:
            return ""

        # Build coverage map
        seed_meta_all = {v.id: v.meta for v in vec_candidates}
        seed_metadata = {
            node_id: seed_meta_all[node_id]
            for node_id in selected_ids
            if node_id in seed_meta_all
        }
        from ragzoom.retrieval import CoverageBuilder

        doc_coverage_builder = CoverageBuilder(self.document_store)
        coverage_result = doc_coverage_builder.build_complete_coverage(
            selected_ids, seed_metadata=seed_metadata
        )
        coverage_map = coverage_result.coverage_map

        # Build scores map
        candidates = self._build_candidates_for_scoring(query_embedding, vec_candidates)
        doc_scoring_service = ScoringService(self.document_store, self.vector_index)
        scores = doc_scoring_service.compute_scores(
            query_embedding, coverage_map, candidates, nodes=coverage_result.nodes
        )

        # Load all nodes in coverage map
        nodes = self._load_coverage_nodes(coverage_map, coverage_result.nodes)

        # Find root nodes
        root_ids = self._find_coverage_root_ids(nodes)

        if not root_ids:
            return ""

        root_ids.sort(
            key=lambda node_id: (
                int(getattr(nodes[node_id], "span_start", 0)),
                node_id,
            )
        )

        # Generate tiling using greedy algorithm
        dp_result = self.greedy_generator.find_optimal_tiling_over_roots(
            root_ids, budget_tokens, scores, nodes
        )
        tiling_ids = list(dp_result.tiling.node_ids)

        if not tiling_ids:
            return ""

        # Ensure all tiling nodes are loaded
        missing_in_nodes = [nid for nid in tiling_ids if nid not in nodes]
        if missing_in_nodes:
            loaded_more = self.document_store.nodes.get_nodes(missing_in_nodes)
            for n in loaded_more:
                nodes[n.id] = n

        # Assemble text from tiling
        texts = [
            nodes[nid].text for nid in tiling_ids if nid in nodes and nodes[nid].text
        ]
        return "\n\n".join(texts)

    async def retrieve_with_telemetry(
        self,
        query: str,
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
        recent_verbatim_budget: int | None = None,
    ) -> tuple[RetrievalResult, QueryTelemetry]:
        """Async retrieval with detailed telemetry collection.

        Args:
            query: Query text to search for
            num_seeds: Number of seed nodes to retrieve
            budget_tokens: Token budget for the final summary
            document_id: Optional document ID to filter by
            recent_verbatim_budget: Token budget for recent leaves to include verbatim

        Returns:
            Tuple of (RetrievalResult, QueryTelemetry) with detailed timing info
        """
        collector = TelemetryCollector()
        collector.start_query(query, num_seeds, budget_tokens, document_id)

        result = await self.retrieve_async(
            query,
            num_seeds,
            budget_tokens,
            document_id,
            recent_verbatim_budget,
            telemetry_collector=collector,
        )
        telemetry = collector.finalize()
        if telemetry is None:
            raise RuntimeError("Telemetry collection failed")
        return result, telemetry
