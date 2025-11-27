"""Query service for RagZoom query processing."""

import logging
from dataclasses import dataclass

from ragzoom.assemble import Assembler
from ragzoom.config import OperationalConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.query_log import QueryLog
from ragzoom.retrieve import RetrievalResult, Retriever
from ragzoom.vector_factory import create_vector_index

logger = logging.getLogger(__name__)


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
        # Note: Retriever and Assembler are now created per-request with DocumentStore

    # jscpd:ignore-start - Legitimate sync/async pattern duplication
    def execute_query(
        self,
        query_text: str,
        document_id: str,
        num_seeds: int | None = None,
        token_budget: int | None = None,
        recent_verbatim_budget: int | None = None,
    ) -> QueryResult:
        """Execute a query and return assembled result.

        Args:
            query_text: Query text
            document_id: Document ID to query within
            num_seeds: Optional override for number of seed nodes
            token_budget: Optional override for token budget
            recent_verbatim_budget: Token budget for recent leaves to include verbatim

        Returns:
            QueryResult with summary and statistics
        """
        # Use provided budget or config default
        budget = token_budget or self.query_config.budget_tokens

        # Create document-scoped store and components
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
        budget_planner = BudgetPlanner(document_store, index_cfg.target_chunk_tokens)
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

        # Retrieve relevant nodes
        retrieval_result = retriever.retrieve(
            query_text,
            budget_tokens=budget,
            document_id=document_id,
            num_seeds=num_seeds,
            recent_verbatim_budget=recent_verbatim_budget,
        )

        # Assemble summary
        summary = assembler.assemble(retrieval_result)
        token_count = assembler.get_token_count(summary)

        query_id = self._record_query(
            query_text,
            document_id,
            budget,
            num_seeds,
            retrieval_result,
        )

        return QueryResult(
            summary=summary,
            token_count=token_count,
            nodes_retrieved=len(retrieval_result.node_ids),
            tiling_size=len(retrieval_result.tiling) if retrieval_result.tiling else 0,
            query_id=query_id,
            seed_count=retrieval_result.seed_count,
            verbatim_count=retrieval_result.verbatim_count,
        )

    # jscpd:ignore-end

    # jscpd:ignore-start - Legitimate sync/async pattern duplication
    async def execute_query_async(
        self,
        query_text: str,
        document_id: str,
        num_seeds: int | None = None,
        token_budget: int | None = None,
        recent_verbatim_budget: int | None = None,
    ) -> QueryResult:
        """Execute a query asynchronously.

        Args:
            query_text: Query text
            document_id: Document ID to query within
            num_seeds: Optional override for number of seed nodes
            token_budget: Optional override for token budget
            recent_verbatim_budget: Token budget for recent leaves to include verbatim

        Returns:
            QueryResult with summary and statistics
        """
        # Use provided budget or config default
        budget = token_budget or self.query_config.budget_tokens

        # Create document-scoped store and components
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
        budget_planner = BudgetPlanner(document_store, index_cfg.target_chunk_tokens)
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

        # Retrieve relevant nodes
        retrieval_result = await retriever.retrieve_async(
            query_text,
            num_seeds=num_seeds,
            budget_tokens=budget,
            document_id=document_id,
            recent_verbatim_budget=recent_verbatim_budget,
        )

        # Assemble summary
        summary = assembler.assemble(retrieval_result)
        token_count = assembler.get_token_count(summary)

        query_id = self._record_query(
            query_text,
            document_id,
            budget,
            num_seeds,
            retrieval_result,
        )

        return QueryResult(
            summary=summary,
            token_count=token_count,
            nodes_retrieved=len(retrieval_result.node_ids),
            tiling_size=len(retrieval_result.tiling) if retrieval_result.tiling else 0,
            query_id=query_id,
            seed_count=retrieval_result.seed_count,
            verbatim_count=retrieval_result.verbatim_count,
        )

    # jscpd:ignore-end

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
        budget_tokens: int,
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
