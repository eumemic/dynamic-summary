"""Output formatting utilities for CLI commands.

Provides functions to convert internal response objects to various
output formats (JSON, text, etc.) for CLI consumption.
"""

from __future__ import annotations

from typing import TypedDict

from ragzoom.client.grpc_client import ExecuteQueryOutput


class TilingNodeDict(TypedDict):
    """JSON schema for a tiling node."""

    node_id: str
    text: str
    span_start: int
    span_end: int
    time_start: str | None
    time_end: str | None
    height: int
    is_seed: bool
    token_count: int


class ActualSpanDict(TypedDict):
    """JSON schema for actual span bounds."""

    start: int
    end: int | None


class QueryJsonOutput(TypedDict):
    """JSON schema for query output per specs/json-output-mode.md."""

    summary: str
    token_count: int
    seed_count: int
    tiling_size: int
    actual_span: ActualSpanDict
    tiling: list[TilingNodeDict]
    query: str
    document_id: str


def build_json_output(
    response: ExecuteQueryOutput,
    query_text: str,
    document_id: str,
) -> QueryJsonOutput:
    """Convert query response to JSON-serializable dictionary.

    Builds output matching the schema in specs/json-output-mode.md.
    Tiling order is preserved from response.retrieval.tiling_ids.

    Args:
        response: The query execution output from gRPC client.
        query_text: Original query string.
        document_id: Document that was queried.

    Returns:
        Dictionary ready for JSON serialization with fields:
        - summary, token_count, seed_count, tiling_size
        - actual_span (start/end)
        - tiling (ordered list of node details)
        - query, document_id
    """
    query_result = response.query_result
    retrieval = response.retrieval

    tiling: list[TilingNodeDict] = []
    for node_id in retrieval.tiling_ids:
        node = retrieval.nodes.get(node_id)
        if node is None:
            continue
        tiling.append(
            TilingNodeDict(
                node_id=node_id,
                text=node.text,
                span_start=node.span_start,
                span_end=node.span_end,
                time_start=node.time_start,
                time_end=node.time_end,
                height=node.height,
                is_seed=node_id in retrieval.selected_ids,
                token_count=node.token_count,
            )
        )

    return QueryJsonOutput(
        summary=query_result.summary,
        token_count=query_result.token_count,
        seed_count=query_result.seed_count,
        tiling_size=query_result.tiling_size,
        actual_span=ActualSpanDict(
            start=query_result.actual_start,
            end=query_result.actual_end,
        ),
        tiling=tiling,
        query=query_text,
        document_id=document_id,
    )
