"""Output formatting utilities for CLI commands.

Provides functions to convert internal response objects to various
output formats (JSON, text, etc.) for CLI consumption.
"""

from __future__ import annotations

from typing import TypedDict

from ragzoom.client.grpc_client import ExecuteQueryOutput, RetrievalView
from ragzoom.exceptions import (
    ConfigurationError,
    DatabaseError,
    DocumentNotFoundError,
    LLMError,
    NodeNotFoundError,
    ResourceError,
    ValidationError,
)


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


def format_tiling_spans(response: ExecuteQueryOutput) -> str:
    """Format query response as text with temporal Span markers.

    Produces a variable-resolution summary where each tiling node is annotated
    with its time range and summarization height. Height=0 nodes are verbatim
    transcript (returned as-is); higher nodes are wrapped in <Span> tags with
    time_start/time_end attributes so consumers can zoom into specific periods.

    This is the canonical text format for RagZoom query results — used by the
    MCP recall tool, the benchmarking harness, and the CLI.

    Args:
        response: The query execution output from gRPC client.

    Returns:
        Formatted text with Span markers, or a fallback message if the
        tiling is empty.
    """
    return _format_retrieval_spans(response.retrieval)


def _format_retrieval_spans(retrieval: RetrievalView) -> str:
    """Format a RetrievalView's tiling as text with resolution markers."""
    nodes = []
    for node_id in retrieval.tiling_ids:
        node = retrieval.nodes.get(node_id)
        if node and node.text:
            nodes.append(node)

    if not nodes:
        return "No results found for this query."

    is_temporal = any(n.time_start is not None for n in nodes)
    max_height = max(n.height for n in nodes)

    lines: list[str] = []
    lines.append("<Explanation>")

    if is_temporal:
        first_start = nodes[0].time_start or "?"
        last_end = nodes[-1].time_end or "?"
        lines.append(
            "This is a variable-resolution summary covering "
            f"{first_start} to {last_end}."
        )
    else:
        first_span = nodes[0].span_start
        last_span = nodes[-1].span_end
        lines.append(
            "This is a variable-resolution summary covering "
            f"characters {first_span:,}–{last_span:,}."
        )

    lines.append(
        "Each span's height indicates summarization level: "
        "height=0 is verbatim text, higher values are "
        "increasingly compressed."
    )
    if max_height > 0:
        if is_temporal:
            lines.append(
                "To zoom in, invoke recall() with time_start/time_end "
                "to constrain the time range."
            )
        else:
            lines.append(
                "To get more detail, invoke recall() with a larger " "token_budget."
            )
    lines.append("</Explanation>")
    lines.append("")

    for node in nodes:
        if node.height == 0:
            lines.append(node.text)
            lines.append("")
        elif is_temporal:
            start = node.time_start or "?"
            end = node.time_end or "?"
            verbatim = node.token_count * (2**node.height)
            lines.append(
                f'<Span time_start="{start}" time_end="{end}" '
                f"height={node.height} "
                f"tokens={node.token_count} verbatim_tokens={verbatim}>"
            )
            lines.append(node.text)
            lines.append("</Span>")
            lines.append("")
        else:
            verbatim = node.token_count * (2**node.height)
            lines.append(
                f"<Span span_start={node.span_start} "
                f"span_end={node.span_end} height={node.height} "
                f"tokens={node.token_count} verbatim_tokens={verbatim}>"
            )
            lines.append(node.text)
            lines.append("</Span>")
            lines.append("")

    return "\n".join(lines).rstrip()


class ErrorJsonOutput(TypedDict):
    """JSON schema for error output per specs/json-output-mode.md."""

    error: str
    code: str


def build_json_error(error_message: str, error_code: str) -> ErrorJsonOutput:
    """Build a JSON error response.

    Args:
        error_message: Human-readable error description.
        error_code: Machine-readable error code (e.g., NOT_FOUND, VALIDATION_ERROR).

    Returns:
        Dictionary with 'error' and 'code' fields for JSON serialization.
    """
    return ErrorJsonOutput(error=error_message, code=error_code)


def build_json_error_from_exception(exc: Exception) -> ErrorJsonOutput:
    """Build a JSON error response from an exception.

    Maps exception types to appropriate error codes:
    - DocumentNotFoundError, NodeNotFoundError -> NOT_FOUND
    - ValidationError -> VALIDATION_ERROR
    - LLMError -> LLM_ERROR
    - ConfigurationError -> CONFIGURATION_ERROR
    - DatabaseError -> DATABASE_ERROR
    - ResourceError -> RESOURCE_ERROR
    - Other exceptions -> INTERNAL_ERROR

    Args:
        exc: The exception to convert.

    Returns:
        Dictionary with 'error' and 'code' fields for JSON serialization.
    """
    error_message = str(exc)

    if isinstance(exc, DocumentNotFoundError | NodeNotFoundError):
        return build_json_error(error_message, "NOT_FOUND")
    elif isinstance(exc, ValidationError):
        return build_json_error(error_message, "VALIDATION_ERROR")
    elif isinstance(exc, LLMError):
        return build_json_error(error_message, "LLM_ERROR")
    elif isinstance(exc, ConfigurationError):
        return build_json_error(error_message, "CONFIGURATION_ERROR")
    elif isinstance(exc, DatabaseError):
        return build_json_error(error_message, "DATABASE_ERROR")
    elif isinstance(exc, ResourceError):
        return build_json_error(error_message, "RESOURCE_ERROR")
    else:
        return build_json_error(error_message, "INTERNAL_ERROR")
