"""Query service for RagZoom query processing."""

import logging
from dataclasses import dataclass
from typing import NamedTuple

from ragzoom.assemble import Assembler
from ragzoom.config import OperationalConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.query_log import QueryLog
from ragzoom.retrieve import RetrievalResult, Retriever
from ragzoom.vector_factory import create_vector_index

logger = logging.getLogger(__name__)


class _QueryComponents(NamedTuple):
    """Components needed for query execution."""

    retriever: Retriever
    assembler: Assembler
    budget: int | None


@dataclass
class QueryResult:
    """Result from query execution."""

    summary: str
    token_count: int
    nodes_retrieved: int
    tiling_size: int
    query_id: str
    seed_count: int = 0
    verbatim_count: int = 0
    actual_start: int = 0
    actual_end: int | None = None


@dataclass
class TilingNode:
    """A node in the tiling with metadata for display."""

    node_id: str
    text: str
    time_start: float | None
    time_end: float | None
    height: int
    is_seed: bool


@dataclass
class QueryResultWithTiling:
    """Query result including tiling nodes for time-aware formatting."""

    result: QueryResult
    tiling: list[TilingNode]


class QueryService:
    """Service for query processing operations."""

    def __init__(
        self,
        store: StorageBackend,
        query_config: QueryConfig,
        operational_config: OperationalConfig,
        query_logger: QueryLog,
    ):
        """Initialize query service.

        Args:
            store: Store instance for data access
            query_config: Configuration for queries
            operational_config: Operational configuration
        """
        self.store = store
        self.query_config = query_config
        self.operational_config = operational_config
        self.query_logger = query_logger

    def _create_query_components(
        self,
        document_id: str,
        token_budget: int | None,
    ) -> _QueryComponents:
        """Create fresh retrieval components for a query.

        Components are created per-query to ensure thread safety and
        document isolation.
        """
        budget = token_budget or self.query_config.budget_tokens

        from openai import OpenAI

        from ragzoom.config import IndexConfig
        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService

        client = OpenAI(
            api_key=self.operational_config.openai_api_key.get_secret_value()
        )
        document_store = self.store.for_document(document_id)
        embedding_service = EmbeddingService(
            client, document_store, self.query_config.embedding_model
        )
        index_cfg = IndexConfig.load()
        # For retrieval operations, use target_embedding_context_tokens as fallback
        # when target_chunk_tokens is None (client-managed chunking mode)
        chunk_tokens = (
            index_cfg.target_chunk_tokens
            if index_cfg.target_chunk_tokens is not None
            else index_cfg.target_embedding_context_tokens
        )
        budget_planner = BudgetPlanner(document_store, chunk_tokens)
        vector_index = create_vector_index(
            self.operational_config.vector_backend,
            self.operational_config.database_url,
            self.query_config.embedding_model,
        )
        retriever = Retriever(
            self.query_config,
            document_store,
            embedding_service,
            budget_planner,
            vector_index,
        )
        assembler = Assembler(document_store)

        return _QueryComponents(retriever=retriever, assembler=assembler, budget=budget)

    def _build_query_result(
        self,
        query_text: str,
        document_id: str,
        retrieval_result: RetrievalResult,
        assembler: Assembler,
        budget: int | None,
        num_seeds: int | None,
    ) -> QueryResult:
        """Build QueryResult from retrieval result."""
        summary = assembler.assemble(retrieval_result)
        token_count = assembler.get_token_count(summary)
        query_id = self._record_query(
            query_text, document_id, budget, num_seeds, retrieval_result
        )
        return QueryResult(
            summary=summary,
            token_count=token_count,
            nodes_retrieved=len(retrieval_result.node_ids),
            tiling_size=len(retrieval_result.tiling) if retrieval_result.tiling else 0,
            query_id=query_id,
            seed_count=retrieval_result.seed_count,
            verbatim_count=retrieval_result.verbatim_count,
            actual_start=retrieval_result.actual_start,
            actual_end=retrieval_result.actual_end,
        )

    # jscpd:ignore-start - sync/async method pair (legitimate duplication pattern)
    def execute_query(
        self,
        query_text: str,
        document_id: str,
        num_seeds: int | None = None,
        token_budget: int | None = None,
        recent_verbatim_budget: int | None = None,
        span_start: int = 0,
        span_end: int | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> QueryResult:
        """Execute a query and return assembled result.

        Args:
            query_text: Query text
            document_id: Document ID to query within
            num_seeds: Optional override for number of seed nodes
            token_budget: Optional override for token budget
            recent_verbatim_budget: Token budget for recent leaves to include verbatim
            span_start: Start of document window (character position, default: 0)
            span_end: End of document window (default: document end)
            time_start: Optional ISO 8601 start time for temporal filtering
            time_end: Optional ISO 8601 end time for temporal filtering

        Returns:
            QueryResult with summary and statistics
        """
        components = self._create_query_components(document_id, token_budget)

        retrieval_result = components.retriever.retrieve(
            query_text,
            budget_tokens=components.budget,
            document_id=document_id,
            num_seeds=num_seeds,
            recent_verbatim_budget=recent_verbatim_budget,
            span_start=span_start,
            span_end=span_end,
            time_start=time_start,
            time_end=time_end,
        )

        return self._build_query_result(
            query_text,
            document_id,
            retrieval_result,
            components.assembler,
            components.budget,
            num_seeds,
        )

    async def execute_query_async(
        self,
        query_text: str,
        document_id: str,
        num_seeds: int | None = None,
        token_budget: int | None = None,
        recent_verbatim_budget: int | None = None,
        span_start: int = 0,
        span_end: int | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> QueryResult:
        """Execute a query asynchronously.

        Args:
            query_text: Query text
            document_id: Document ID to query within
            num_seeds: Optional override for number of seed nodes
            token_budget: Optional override for token budget
            recent_verbatim_budget: Token budget for recent leaves to include verbatim
            span_start: Start of document window (character position, default: 0)
            span_end: End of document window (default: document end)
            time_start: Optional ISO 8601 start time for temporal filtering
            time_end: Optional ISO 8601 end time for temporal filtering

        Returns:
            QueryResult with summary and statistics
        """
        components = self._create_query_components(document_id, token_budget)

        retrieval_result = await components.retriever.retrieve_async(
            query_text,
            num_seeds=num_seeds,
            budget_tokens=components.budget,
            document_id=document_id,
            recent_verbatim_budget=recent_verbatim_budget,
            span_start=span_start,
            span_end=span_end,
            time_start=time_start,
            time_end=time_end,
        )

        return self._build_query_result(
            query_text,
            document_id,
            retrieval_result,
            components.assembler,
            components.budget,
            num_seeds,
        )

    # jscpd:ignore-end

    async def execute_query_with_tiling_async(
        self,
        query_text: str,
        document_id: str,
        num_seeds: int | None = None,
        token_budget: int | None = None,
        recent_verbatim_budget: int | None = None,
        span_start: int = 0,
        span_end: int | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> QueryResultWithTiling:
        """Execute a query and return result with tiling nodes.

        Like execute_query_async but preserves the tiling nodes for
        downstream formatting (e.g., displaying timestamps per tile).

        Args:
            query_text: Query text
            document_id: Document ID to query within
            num_seeds: Optional override for number of seed nodes
            token_budget: Optional override for token budget
            recent_verbatim_budget: Token budget for recent leaves to include verbatim
            span_start: Start of document window (character position, default: 0)
            span_end: End of document window (default: document end)
            time_start: Optional ISO 8601 start time for temporal filtering
            time_end: Optional ISO 8601 end time for temporal filtering

        Returns:
            QueryResultWithTiling containing both the query result and tiling nodes
        """
        components = self._create_query_components(document_id, token_budget)

        retrieval_result = await components.retriever.retrieve_async(
            query_text,
            num_seeds=num_seeds,
            budget_tokens=components.budget,
            document_id=document_id,
            recent_verbatim_budget=recent_verbatim_budget,
            span_start=span_start,
            span_end=span_end,
            time_start=time_start,
            time_end=time_end,
        )

        query_result = self._build_query_result(
            query_text,
            document_id,
            retrieval_result,
            components.assembler,
            components.budget,
            num_seeds,
        )

        # Build tiling nodes with metadata
        tiling_nodes: list[TilingNode] = []
        tiling_ids = retrieval_result.tiling or []
        nodes = retrieval_result.nodes or {}
        seed_ids = set(retrieval_result.node_ids)

        for node_id in tiling_ids:
            node = nodes.get(node_id)
            if node is None:
                continue
            tiling_nodes.append(
                TilingNode(
                    node_id=node_id,
                    text=node.text,
                    time_start=node.time_start,
                    time_end=node.time_end,
                    height=node.height,
                    is_seed=node_id in seed_ids,
                )
            )

        return QueryResultWithTiling(result=query_result, tiling=tiling_nodes)

    def update_config(
        self,
        budget_tokens: int | None = None,
        mmr_lambda: float | None = None,
    ) -> None:
        """Update query configuration dynamically.

        Args:
            budget_tokens: New token budget
            mmr_lambda: New MMR lambda parameter
        """
        # Update config if any parameters changed
        if budget_tokens is not None or mmr_lambda is not None:
            self.query_config = self.query_config.replace(
                budget_tokens=budget_tokens,
                mmr_lambda=mmr_lambda,
            )
            # Note: Retriever is now created per-request with latest config

    def _record_query(
        self,
        query_text: str,
        document_id: str,
        budget_tokens: int | None,
        num_seeds: int | None,
        retrieval_result: RetrievalResult,
    ) -> str:
        """Persist the query for later inspection."""

        tiling = retrieval_result.tiling
        if tiling is None:
            raise ValueError("Retrieval result missing tiling; cannot log query")

        seeds = set(retrieval_result.node_ids)
        return self.query_logger.record_query(
            document_id=document_id,
            query_text=query_text,
            budget_tokens=budget_tokens,
            num_seeds=num_seeds,
            tiling_ids=tiling,
            scores=retrieval_result.scores,
            seed_ids=seeds,
        )
