"""Retrieval logic with MMR diversity for RagZoom."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ragzoom.config import QueryConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_filter import (
    DocumentIdFilter,
    SpanEndLtFilter,
    SpanOverlapsFilter,
    VectorFilter,
)
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
from ragzoom.vector_api import Vector

if TYPE_CHECKING:
    from ragzoom.contracts.vector_index import VectorIndex
    from ragzoom.document_store import DocumentStore
    from ragzoom.vector_api import Vector

logger = logging.getLogger(__name__)


def _parse_query_timestamp(iso_string: str) -> float:
    """Parse an ISO 8601 timestamp string to Unix timestamp for queries.

    Args:
        iso_string: ISO 8601 formatted string with timezone info.

    Returns:
        Unix timestamp as float seconds since epoch.

    Raises:
        ValueError: If the string is invalid or lacks timezone info.
    """
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 timestamp format: {iso_string}") from e

    if dt.tzinfo is None:
        raise ValueError(
            f"Timestamp must include timezone info (e.g., 'Z' or '+00:00'): {iso_string}"
        )

    return dt.timestamp()


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
    ):
        """Initialize retriever.

        Args:
            query_config: Query configuration
            document_store: DocumentStore instance for document-scoped operations
            embedding_service: Service for generating query embeddings
            budget_planner: Service for calculating conservative seed counts
            vector_index: Vector index for similarity search
        """
        self.query_config = query_config
        self.document_store = document_store
        self.embedding_service = embedding_service
        self.budget_planner = budget_planner
        self.vector_index = vector_index
        self.tiling_generator = GreedyTilingGenerator(query_config)

    # ------------------------------------------------------------------
    # Helper methods for shared retrieval logic
    # ------------------------------------------------------------------

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
        query: str = "",
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
        recent_verbatim_budget: int | None = None,
        telemetry_collector: TelemetryCollector | None = None,
        span_start: int = 0,
        span_end: int | None = None,
        query_embedding: list[float] | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
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
            query_embedding: Pre-computed query embedding. If provided, skips the
                embedding API call. Used during indexing for inner nodes where the
                parent embedding (avg of children) is already available.
            time_start: Start of time window (ISO 8601 with timezone). For temporal
                documents only. Will be mapped to span_start via leaf lookup.
            time_end: End of time window (ISO 8601 with timezone). For temporal
                documents only. Will be mapped to span_end via leaf lookup.

        Supports three modes:
        1. Budget only: Calculate conservative num_seeds to guarantee no overflow
        2. Budget + num_seeds: Use num_seeds but drop nodes if needed for budget
        3. num_seeds only: Just use num_seeds, no budget enforcement

        When span_start/span_end are specified, the tiling covers exactly the
        minimal span [actual_start, actual_end) that contains the requested window.

        When time_start/time_end are specified on a temporal document, these are
        converted to span bounds via leaf lookup before retrieval.
        """
        # Start telemetry if collector is provided
        if telemetry_collector:
            telemetry_collector.start_phase()

        # Determine effective document scope
        effective_doc_id = (
            getattr(self.document_store, "document_id", None) or document_id
        )

        # Get repository reference for queries
        nodes_wrapper = getattr(self.document_store, "nodes", None)
        repo = getattr(nodes_wrapper, "_repo", None) if nodes_wrapper else None
        doc_repo = getattr(self.document_store, "_doc_repo", None)

        # Time→span mapping: convert time window to span window
        if time_start is not None or time_end is not None:
            if not effective_doc_id:
                raise ValueError("document_id is required for time-windowed queries")
            if repo is None or doc_repo is None:
                raise ValueError("Repository not available for time-windowed queries")

            # Validate document is temporal
            is_temporal = doc_repo.get_document_is_temporal(effective_doc_id)
            if not is_temporal:
                raise ValueError(
                    f"Time-windowed queries require a temporal document. "
                    f"Document '{effective_doc_id}' is non-temporal. "
                    f"Index with timestamps to enable time queries."
                )

            # Parse timestamps
            time_start_unix = _parse_query_timestamp(time_start) if time_start else None
            time_end_unix = _parse_query_timestamp(time_end) if time_end else None

            # Validate time_end >= time_start if both provided
            if (
                time_start_unix is not None
                and time_end_unix is not None
                and time_end_unix < time_start_unix
            ):
                raise ValueError(
                    f"time_end ({time_end}) must be >= time_start ({time_start})"
                )

            # Map time to span via leaf lookup
            if time_start_unix is not None:
                leaf_start = repo.get_leaf_at_time_position(
                    effective_doc_id, time_start_unix, "start"
                )
                if leaf_start is None:
                    raise ValueError(
                        f"No leaf found at time_start={time_start}. "
                        f"The requested time may be outside the document's time range."
                    )
                span_start = leaf_start.span_start

            if time_end_unix is not None:
                leaf_end = repo.get_leaf_at_time_position(
                    effective_doc_id, time_end_unix, "end"
                )
                if leaf_end is None:
                    raise ValueError(
                        f"No leaf found at time_end={time_end}. "
                        f"The requested time may be outside the document's time range."
                    )
                span_end = leaf_end.span_end

        # Compute window bounds if windowed query
        from ragzoom.retrieval.coverage_builder import WindowBounds

        window_bounds: WindowBounds | None = None
        actual_start = 0
        actual_end: int | None = None

        # Get document span end for defaults and validation
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

            # Compute window bounds for windowed query
            doc_coverage_builder = CoverageBuilder(self.document_store)
            window_bounds = doc_coverage_builder.compute_window_bounds(
                span_start, resolved_span_end, effective_doc_id
            )
            actual_start = window_bounds.actual_start
            actual_end = window_bounds.actual_end
        elif effective_doc_id and doc_span_end is not None:
            # Full document query - compute window bounds for entire document
            doc_coverage_builder = CoverageBuilder(self.document_store)
            window_bounds = doc_coverage_builder.compute_window_bounds(
                0, doc_span_end, effective_doc_id
            )
            actual_start = window_bounds.actual_start
            actual_end = window_bounds.actual_end
        else:
            actual_end = doc_span_end

        # Validate: at least one of num_seeds or budget_tokens must be provided
        effective_budget = budget_tokens or self.query_config.budget_tokens
        if num_seeds is None and effective_budget is None:
            raise ValueError(
                "At least one of num_seeds or budget_tokens must be specified"
            )

        # Determine which mode we're in
        if effective_budget is not None and num_seeds is None:
            num_seeds = self.budget_planner.calculate_conservative_num_seeds(
                effective_budget, effective_doc_id
            )
            logger.debug(
                f"Budget-only mode: calculated conservative num_seeds={num_seeds} "
                f"for budget={effective_budget}"
            )
        elif effective_budget is not None and num_seeds is not None:
            logger.debug(
                f"Mixed mode: num_seeds={num_seeds}, budget={effective_budget}"
            )
        else:
            # Seeds-only mode: num_seeds provided but no budget
            logger.debug(
                f"Seeds-only mode: num_seeds={num_seeds}, no budget constraint"
            )

        # At this point num_seeds is guaranteed to be set
        assert num_seeds is not None

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
                # Filter out leaves that start before the window
                if actual_start is not None:
                    verbatim_leaves = [
                        leaf
                        for leaf in verbatim_leaves
                        if leaf.span_start >= actual_start
                    ]
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

        # Phase 1: Get query embedding (skip if num_seeds=0 or pre-computed embedding)
        from ragzoom.retrieval import mmr

        # Use a separate local variable to avoid redefinition of parameter
        effective_query_embedding: list[float]
        if num_seeds == 0:
            # num_seeds=0: skip embedding and vector search, produce minimal summary
            effective_query_embedding = []
            selected_ids: list[str] = []
            vec_candidates: list[Vector] = []
            if telemetry_collector:
                telemetry_collector.end_phase("embedding")
                telemetry_collector.start_phase()
                telemetry_collector.record_metric("candidates_retrieved", 0)
        else:
            # Get embedding: either pre-computed or computed on demand
            if query_embedding is not None:
                # Pre-computed embedding provided (e.g., during indexing)
                effective_query_embedding = query_embedding
            else:
                effective_query_embedding = (
                    await self.embedding_service.get_query_embedding_async(
                        query, effective_doc_id
                    )
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
            filters: list[VectorFilter] = []
            if effective_doc_id:
                filters.append(DocumentIdFilter(effective_doc_id))
            if verbatim_horizon is not None:
                filters.append(SpanEndLtFilter(verbatim_horizon))
            if is_windowed and actual_end is not None:
                filters.append(SpanOverlapsFilter(actual_start, actual_end))
            raw_candidates = self.vector_index.search_similar(
                effective_query_embedding,
                k_candidates,
                filters if filters else None,
            )
            # Stale vectors (from deleted nodes) are handled by coverage builder
            # which checks coord_version in metadata and falls back to DB if needed
            vec_candidates = raw_candidates
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
                effective_query_embedding,
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
        candidates = self._build_candidates_for_scoring(
            effective_query_embedding, vec_candidates
        )
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
        if window_bounds is None:
            raise ValueError("Cannot build coverage: window_bounds not computed")
        coverage_result = doc_coverage_builder.build_windowed_coverage(
            selected_ids,
            window_bounds,
            seed_metadata=seed_metadata,
            pinned_ids=pinned_ids,
        )
        coverage_map = coverage_result.coverage_map
        if telemetry_collector:
            telemetry_collector.end_phase("coverage_map")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric("coverage_size", len(coverage_map))

        # Phase 5: Build scores map
        # Use document-scoped scoring service (skip if no query embedding)
        if len(effective_query_embedding) > 0:
            doc_scoring_service = ScoringService(self.document_store, self.vector_index)
            scores = doc_scoring_service.compute_scores(
                effective_query_embedding,
                coverage_map,
                candidates,
                nodes=coverage_result.nodes,
            )
        else:
            # num_seeds=0: no query, use default scores (0.0 for all nodes)
            scores = {node_id: 0.0 for node_id in coverage_map}
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

        # Phase 6: Extract tiling using greedy algorithm
        # If budget is specified, add verbatim budget; otherwise tiling returns full frontier
        if effective_budget is not None:
            final_budget: int | None = effective_budget + (recent_verbatim_budget or 0)
        else:
            final_budget = None

        # Generate tiling
        tiling_result = self.tiling_generator.find_optimal_tiling_over_roots(
            root_ids, final_budget, scores, nodes
        )
        if telemetry_collector:
            telemetry_collector.end_phase("tiling")
            telemetry_collector.start_phase()
            telemetry_collector.record_metric(
                "tiling_size", len(tiling_result.tiling.node_ids)
            )

        tiling_ids = list(tiling_result.tiling.node_ids)

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
        query: str = "",
        num_seeds: int | None = None,
        budget_tokens: int | None = None,
        document_id: str | None = None,
        recent_verbatim_budget: int | None = None,
        span_start: int = 0,
        span_end: int | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
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
            time_start: Start of time window (ISO 8601 with timezone)
            time_end: End of time window (ISO 8601 with timezone)

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
                time_start=time_start,
                time_end=time_end,
            )
        )

    async def retrieve_for_context(
        self,
        query_text: str,
        span_end_limit: int,
        budget_tokens: int,
        document_id: str | None = None,
        recent_verbatim_token_budget: int = 0,
        query_embedding: list[float] | None = None,
        num_seeds: int | None = None,
    ) -> RetrievalResult:
        """Retrieve tiling of nodes covering [0, span_end_limit).

        Used during indexing to build preceding_context for nodes. Queries the
        existing tree using the node's own text as the query, returning the
        tiling nodes that cover the preceding content.

        Args:
            query_text: The text to use as query (typically the node's own text)
            span_end_limit: Only include nodes where span_end <= this value
            budget_tokens: Token budget for the dynamic summary portion
            document_id: Optional document ID to filter by
            recent_verbatim_token_budget: Token budget for verbatim leaves
            query_embedding: Pre-computed query embedding. If provided, skips the
                embedding API call. Used for inner nodes where the parent embedding
                (avg of children) is already available.
            num_seeds: Number of seed nodes for retrieval. If None, auto-calculated
                from budget.

        Returns:
            RetrievalResult with tiling node IDs in result.tiling and nodes in result.nodes.
        """
        # Nothing to retrieve if span_end_limit is 0 (at document start)
        if span_end_limit <= 0:
            return RetrievalResult(
                node_ids=[],
                scores={},
                coverage_map={},
                tiling=[],
                nodes={},
            )

        return await self.retrieve_async(
            query=query_text,
            num_seeds=num_seeds,
            budget_tokens=budget_tokens,
            document_id=document_id,
            recent_verbatim_budget=recent_verbatim_token_budget,
            span_start=0,
            span_end=span_end_limit,
            query_embedding=query_embedding,
        )

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
