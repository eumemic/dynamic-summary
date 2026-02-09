"""In-process query execution for server-side consumers.

Extracts the core retrieval + assembly pipeline from
``RetrievalServicer.ExecuteQuery`` so that the search agent can call it
directly without going through gRPC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from openai import OpenAI

from ragzoom.assemble import Assembler
from ragzoom.client.grpc_client import (
    ExecuteQueryOutput,
    NodeSummary,
    RetrievalView,
)
from ragzoom.document_store import DocumentStore
from ragzoom.retrieval.budget_planner import BudgetPlanner
from ragzoom.retrieval.embedding_service import EmbeddingService
from ragzoom.retrieve import Retriever
from ragzoom.services.query_service import QueryResult
from ragzoom.vector_factory import create_vector_index

if TYPE_CHECKING:
    from ragzoom.server.state import ServerState


def _unix_to_iso8601(ts: float) -> str:
    """Convert Unix timestamp (float seconds) to ISO 8601 string with Z suffix."""
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def build_retriever(
    state: ServerState,
    *,
    document_id: str,
    embedding_model: str | None = None,
) -> tuple[Retriever, DocumentStore]:
    """Build a Retriever and DocumentStore for the given document.

    Shared by both the gRPC servicer and the in-process query executor.
    """
    resolved_embedding = embedding_model or state.query_config.embedding_model
    document_store = state.store.for_document(document_id)
    client = OpenAI(
        api_key=state.operational_config.openai_api_key.get_secret_value(),
        timeout=state.operational_config.openai_timeout,
    )
    embedding_service = EmbeddingService(client, document_store, resolved_embedding)
    # For retrieval operations, use target_embedding_tokens as fallback
    # when target_chunk_tokens is None (client-managed chunking mode)
    chunk_tokens = (
        state.index_config.target_chunk_tokens
        if state.index_config.target_chunk_tokens is not None
        else state.index_config.target_embedding_tokens
    )
    budget_planner = BudgetPlanner(document_store, chunk_tokens)
    vector_index = create_vector_index(
        state.operational_config.vector_backend,
        state.operational_config.database_url,
        resolved_embedding,
    )
    query_config = state.query_config
    if resolved_embedding != state.query_config.embedding_model:
        query_config = query_config.replace(embedding_model=resolved_embedding)
    retriever = Retriever(
        query_config,
        document_store,
        embedding_service,
        budget_planner,
        vector_index,
    )
    return retriever, document_store


async def execute_query_internal(
    state: ServerState,
    *,
    document_id: str,
    query: str,
    budget_tokens: int,
    time_start: str | None = None,
    time_end: str | None = None,
) -> ExecuteQueryOutput:
    """Execute a retrieval query in-process, returning the same output
    type that the gRPC client produces.

    This is the canonical entry point for server-side query execution.
    The search agent calls it directly to avoid gRPC overhead.

    Args:
        state: Shared server state (store, configs, services).
        document_id: Document to query.
        query: Semantic search query text.
        budget_tokens: Maximum tokens in the assembled output.
        time_start: Optional ISO 8601 lower bound for temporal filtering.
        time_end: Optional ISO 8601 upper bound for temporal filtering.

    Returns:
        ``ExecuteQueryOutput`` with assembled summary, retrieval view,
        and empty visualization/validation fields (those are debug-only
        concerns handled by the gRPC servicer layer).
    """
    retriever, document_store = build_retriever(state, document_id=document_id)

    retrieval_result = await retriever.retrieve_async(
        query,
        budget_tokens=budget_tokens,
        document_id=document_id,
        time_start=time_start,
        time_end=time_end,
    )

    assembler = Assembler(document_store)
    summary_text = assembler.assemble(retrieval_result)
    token_count = assembler.get_token_count(summary_text)

    # Build the same RetrievalView that the gRPC client returns.
    # TreeNode stores time_start/time_end as Unix float; NodeSummary uses ISO 8601 strings.
    nodes_payload: dict[str, NodeSummary] = {}
    for node_id in retrieval_result.tiling or []:
        node = (retrieval_result.nodes or {}).get(node_id)
        if node is None:
            continue
        ts_start = node.time_start
        ts_end = node.time_end
        nodes_payload[node_id] = NodeSummary(
            node_id=node_id,
            text=node.text,
            token_count=node.token_count,
            span_start=node.span_start,
            span_end=node.span_end,
            parent_id=node.parent_id or "",
            left_child_id=node.left_child_id or "",
            right_child_id=node.right_child_id or "",
            height=node.height,
            time_start=_unix_to_iso8601(ts_start) if ts_start is not None else None,
            time_end=_unix_to_iso8601(ts_end) if ts_end is not None else None,
        )

    retrieval_view = RetrievalView(
        selected_ids=list(retrieval_result.node_ids),
        tiling_ids=list(retrieval_result.tiling or []),
        scores=dict(retrieval_result.scores),
        coverage_map=dict(retrieval_result.coverage_map or {}),
        nodes=nodes_payload,
    )

    query_result = QueryResult(
        summary=summary_text,
        token_count=token_count,
        nodes_retrieved=len(retrieval_result.node_ids),
        tiling_size=len(retrieval_result.tiling or []),
        query_id="",
        seed_count=retrieval_result.seed_count,
        verbatim_count=retrieval_result.verbatim_count,
        actual_start=retrieval_result.actual_start,
        actual_end=retrieval_result.actual_end,
    )

    return ExecuteQueryOutput(
        query_result=query_result,
        retrieval=retrieval_view,
        visualization="",
        validation_warning="",
    )
