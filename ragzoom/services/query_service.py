"""Query service for RagZoom query processing."""

import logging
from dataclasses import dataclass

from ragzoom.assemble import Assembler
from ragzoom.config import OperationalConfig, QueryConfig
from ragzoom.retrieve import Retriever
from ragzoom.store import Store

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result from query execution."""

    summary: str
    token_count: int
    nodes_retrieved: int
    tiling_size: int


class QueryService:
    """Service for query processing operations."""

    def __init__(
        self,
        store: Store,
        query_config: QueryConfig,
        operational_config: OperationalConfig,
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
        self.retriever = Retriever(
            query_config,
            store,
            api_key=operational_config.openai_api_key,
        )
        self.assembler = Assembler(store)

    # jscpd:ignore-start - Legitimate sync/async pattern duplication
    def execute_query(
        self,
        query_text: str,
        document_id: str,
        num_seeds: int | None = None,
        token_budget: int | None = None,
    ) -> QueryResult:
        """Execute a query and return assembled result.

        Args:
            query_text: Query text
            document_id: Document ID to query within
            num_seeds: Optional override for number of seed nodes
            token_budget: Optional override for token budget

        Returns:
            QueryResult with summary and statistics
        """
        # Use provided budget or config default
        budget = token_budget or self.query_config.budget_tokens

        # Retrieve relevant nodes
        retrieval_result = self.retriever.retrieve(
            query_text,
            budget_tokens=budget,
            document_id=document_id,
            num_seeds=num_seeds,
        )

        # Assemble summary
        summary = self.assembler.assemble(retrieval_result)
        token_count = self.assembler.get_token_count(summary)

        return QueryResult(
            summary=summary,
            token_count=token_count,
            nodes_retrieved=len(retrieval_result.node_ids),
            tiling_size=len(retrieval_result.tiling) if retrieval_result.tiling else 0,
        )

    # jscpd:ignore-end

    # jscpd:ignore-start - Legitimate sync/async pattern duplication
    async def execute_query_async(
        self,
        query_text: str,
        document_id: str,
        num_seeds: int | None = None,
        token_budget: int | None = None,
    ) -> QueryResult:
        """Execute a query asynchronously.

        Args:
            query_text: Query text
            document_id: Document ID to query within
            num_seeds: Optional override for number of seed nodes
            token_budget: Optional override for token budget

        Returns:
            QueryResult with summary and statistics
        """
        # Use provided budget or config default
        budget = token_budget or self.query_config.budget_tokens

        # Retrieve relevant nodes
        retrieval_result = await self.retriever.retrieve_async(
            query_text,
            num_seeds,
            budget,
            document_id=document_id,
        )

        # Assemble summary
        summary = self.assembler.assemble(retrieval_result)
        token_count = self.assembler.get_token_count(summary)

        return QueryResult(
            summary=summary,
            token_count=token_count,
            nodes_retrieved=len(retrieval_result.node_ids),
            tiling_size=len(retrieval_result.tiling) if retrieval_result.tiling else 0,
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
        # Build update dictionary
        updates: dict[str, int | float] = {}
        if budget_tokens is not None:
            updates["budget_tokens"] = budget_tokens
        if mmr_lambda is not None:
            updates["mmr_lambda"] = mmr_lambda

        if updates:
            # Update config
            self.query_config = self.query_config.replace(**updates)

            # Recreate retriever with new config
            self.retriever = Retriever(
                self.query_config,
                self.store,
                api_key=self.operational_config.openai_api_key,
            )
