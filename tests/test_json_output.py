"""Tests for JSON output mode build_json_output() function.

Tests verify the function correctly converts query response data
to the JSON schema specified in specs/json-output-mode.md.
"""

from __future__ import annotations

from ragzoom.client.grpc_client import (
    ExecuteQueryOutput,
    NodeSummary,
    RetrievalView,
)
from ragzoom.exceptions import (
    ConfigurationError,
    DatabaseError,
    DocumentNotFoundError,
    LLMError,
    ResourceError,
    ValidationError,
)
from ragzoom.output_formatters import (
    build_json_error,
    build_json_error_from_exception,
    build_json_output,
    format_tiling_spans,
)
from ragzoom.services.query_service import QueryResult


def _make_node(
    node_id: str,
    *,
    text: str = "Node text",
    token_count: int = 50,
    span_start: int = 0,
    span_end: int = 100,
    height: int = 0,
    time_start: str | None = None,
    time_end: str | None = None,
) -> NodeSummary:
    """Helper to create NodeSummary with defaults."""
    return NodeSummary(
        node_id=node_id,
        text=text,
        token_count=token_count,
        span_start=span_start,
        span_end=span_end,
        parent_id="",
        left_child_id="",
        right_child_id="",
        height=height,
        time_start=time_start,
        time_end=time_end,
    )


def _make_response(
    *,
    summary: str = "Test summary",
    token_count: int = 100,
    seed_count: int = 1,
    tiling_size: int = 2,
    actual_start: int = 0,
    actual_end: int | None = 1000,
    tiling_ids: list[str] | None = None,
    selected_ids: list[str] | None = None,
    nodes: dict[str, NodeSummary] | None = None,
) -> ExecuteQueryOutput:
    """Helper to create ExecuteQueryOutput with defaults."""
    if tiling_ids is None:
        tiling_ids = []
    if selected_ids is None:
        selected_ids = []
    if nodes is None:
        nodes = {}

    query_result = QueryResult(
        summary=summary,
        token_count=token_count,
        nodes_retrieved=len(selected_ids),
        tiling_size=tiling_size,
        query_id="test-query-id",
        seed_count=seed_count,
        verbatim_count=0,
        actual_start=actual_start,
        actual_end=actual_end,
    )

    retrieval = RetrievalView(
        selected_ids=selected_ids,
        tiling_ids=tiling_ids,
        scores={},
        coverage_map={},
        nodes=nodes,
    )

    return ExecuteQueryOutput(
        query_result=query_result,
        retrieval=retrieval,
        visualization="",
        validation_warning="",
    )


class TestBuildJsonOutput:
    """Tests for build_json_output function."""

    def test_basic_schema_fields(self) -> None:
        """Output contains all required top-level fields."""
        response = _make_response(
            summary="Test summary text",
            token_count=150,
            seed_count=2,
            tiling_size=3,
            actual_start=100,
            actual_end=5000,
        )

        result = build_json_output(
            response=response,
            query_text="test query",
            document_id="doc.txt",
        )

        assert result["summary"] == "Test summary text"
        assert result["token_count"] == 150
        assert result["seed_count"] == 2
        assert result["tiling_size"] == 3
        assert result["actual_span"] == {"start": 100, "end": 5000}
        assert result["query"] == "test query"
        assert result["document_id"] == "doc.txt"
        assert "tiling" in result
        assert isinstance(result["tiling"], list)

    def test_tiling_order_preserved(self) -> None:
        """Tiling nodes appear in same order as tiling_ids."""
        node1 = _make_node("node-1", span_start=0, span_end=100)
        node2 = _make_node("node-2", span_start=100, span_end=200)
        node3 = _make_node("node-3", span_start=200, span_end=300)

        response = _make_response(
            tiling_ids=["node-2", "node-1", "node-3"],  # Specific order
            nodes={"node-1": node1, "node-2": node2, "node-3": node3},
        )

        result = build_json_output(
            response=response,
            query_text="query",
            document_id="doc.txt",
        )

        tiling = result["tiling"]
        assert len(tiling) == 3
        assert tiling[0]["node_id"] == "node-2"
        assert tiling[1]["node_id"] == "node-1"
        assert tiling[2]["node_id"] == "node-3"

    def test_tiling_node_fields(self) -> None:
        """Each tiling node has all required fields."""
        node = _make_node(
            "test-node",
            text="Node summary text",
            token_count=75,
            span_start=500,
            span_end=1000,
            height=2,
        )

        response = _make_response(
            tiling_ids=["test-node"],
            selected_ids=["test-node"],  # This node is a seed
            nodes={"test-node": node},
        )

        result = build_json_output(
            response=response,
            query_text="query",
            document_id="doc.txt",
        )

        tiling_node = result["tiling"][0]
        assert tiling_node["node_id"] == "test-node"
        assert tiling_node["text"] == "Node summary text"
        assert tiling_node["span_start"] == 500
        assert tiling_node["span_end"] == 1000
        assert tiling_node["height"] == 2
        assert tiling_node["is_seed"] is True
        assert tiling_node["token_count"] == 75

    def test_is_seed_false_for_non_seeds(self) -> None:
        """Non-seed nodes have is_seed=False."""
        node = _make_node("non-seed-node")

        response = _make_response(
            tiling_ids=["non-seed-node"],
            selected_ids=[],  # Not in seeds
            nodes={"non-seed-node": node},
        )

        result = build_json_output(
            response=response,
            query_text="query",
            document_id="doc.txt",
        )

        assert result["tiling"][0]["is_seed"] is False

    def test_temporal_fields_present(self) -> None:
        """Temporal fields included when document has timestamps."""
        node = _make_node(
            "temporal-node",
            time_start="2024-01-21T10:00:00Z",
            time_end="2024-01-21T10:30:00Z",
        )

        response = _make_response(
            tiling_ids=["temporal-node"],
            nodes={"temporal-node": node},
        )

        result = build_json_output(
            response=response,
            query_text="query",
            document_id="doc.txt",
        )

        tiling_node = result["tiling"][0]
        assert tiling_node["time_start"] == "2024-01-21T10:00:00Z"
        assert tiling_node["time_end"] == "2024-01-21T10:30:00Z"

    def test_temporal_fields_null_for_non_temporal(self) -> None:
        """Temporal fields are null for non-temporal documents."""
        node = _make_node("non-temporal-node")  # No time_start/time_end

        response = _make_response(
            tiling_ids=["non-temporal-node"],
            nodes={"non-temporal-node": node},
        )

        result = build_json_output(
            response=response,
            query_text="query",
            document_id="doc.txt",
        )

        tiling_node = result["tiling"][0]
        assert tiling_node["time_start"] is None
        assert tiling_node["time_end"] is None

    def test_empty_tiling(self) -> None:
        """Empty tiling produces empty list, not error."""
        response = _make_response(
            tiling_ids=[],
            nodes={},
        )

        result = build_json_output(
            response=response,
            query_text="query",
            document_id="doc.txt",
        )

        assert result["tiling"] == []

    def test_actual_span_with_none_end(self) -> None:
        """actual_span.end can be null (unbounded queries)."""
        response = _make_response(
            actual_start=0,
            actual_end=None,
        )

        result = build_json_output(
            response=response,
            query_text="query",
            document_id="doc.txt",
        )

        assert result["actual_span"]["start"] == 0
        assert result["actual_span"]["end"] is None

    def test_skips_missing_nodes(self) -> None:
        """Tiling IDs without corresponding nodes are skipped."""
        node = _make_node("existing-node")

        response = _make_response(
            tiling_ids=["existing-node", "missing-node"],
            nodes={"existing-node": node},  # missing-node not in dict
        )

        result = build_json_output(
            response=response,
            query_text="query",
            document_id="doc.txt",
        )

        # Should only have the existing node
        assert len(result["tiling"]) == 1
        assert result["tiling"][0]["node_id"] == "existing-node"


class TestBuildJsonError:
    """Tests for build_json_error function."""

    def test_basic_error_response(self) -> None:
        """Error response has error and code fields."""

        result = build_json_error("Document not found", "NOT_FOUND")

        assert result["error"] == "Document not found"
        assert result["code"] == "NOT_FOUND"

    def test_error_response_keys(self) -> None:
        """Error response contains only expected keys."""

        result = build_json_error("Some error", "ERROR_CODE")

        assert set(result.keys()) == {"error", "code"}

    def test_error_from_document_not_found_exception(self) -> None:
        """DocumentNotFoundError maps to NOT_FOUND code."""
        exc = DocumentNotFoundError("test-doc")
        result = build_json_error_from_exception(exc)

        assert result["code"] == "NOT_FOUND"
        assert "test-doc" in result["error"]

    def test_error_from_validation_exception(self) -> None:
        """ValidationError maps to VALIDATION_ERROR code."""
        exc = ValidationError("field", "value", "bad value")
        result = build_json_error_from_exception(exc)

        assert result["code"] == "VALIDATION_ERROR"

    def test_error_from_llm_exception(self) -> None:
        """LLMError maps to LLM_ERROR code."""
        exc = LLMError("embedding", "test-model", "API rate limit exceeded")
        result = build_json_error_from_exception(exc)

        assert result["code"] == "LLM_ERROR"

    def test_error_from_configuration_exception(self) -> None:
        """ConfigurationError maps to CONFIGURATION_ERROR code."""
        exc = ConfigurationError("api_key", "string", None)
        result = build_json_error_from_exception(exc)

        assert result["code"] == "CONFIGURATION_ERROR"

    def test_error_from_database_exception(self) -> None:
        """DatabaseError maps to DATABASE_ERROR code."""
        exc = DatabaseError("query", "connection failed")
        result = build_json_error_from_exception(exc)

        assert result["code"] == "DATABASE_ERROR"

    def test_error_from_resource_exception(self) -> None:
        """ResourceError maps to RESOURCE_ERROR code."""
        exc = ResourceError("memory", "allocation", "out of memory")
        result = build_json_error_from_exception(exc)

        assert result["code"] == "RESOURCE_ERROR"

    def test_error_from_generic_exception(self) -> None:
        """Generic exceptions map to INTERNAL_ERROR code."""

        exc = RuntimeError("unexpected error")
        result = build_json_error_from_exception(exc)

        assert result["code"] == "INTERNAL_ERROR"
        assert "unexpected error" in result["error"]


class TestFormatTilingSpans:
    """Tests for format_tiling_spans function."""

    def test_empty_tiling_returns_fallback(self) -> None:
        """Empty tiling produces a human-readable fallback message."""
        response = _make_response(tiling_ids=[], nodes={})
        result = format_tiling_spans(response)
        assert result == "No results found for this query."

    def test_verbatim_node_no_span_wrapper(self) -> None:
        """Height=0 nodes appear as plain text without Span tags."""
        node = _make_node(
            "leaf",
            text="Hello, world!",
            height=0,
            time_start="2024-01-01T10:00:00Z",
            time_end="2024-01-01T10:00:00Z",
        )
        response = _make_response(
            tiling_ids=["leaf"],
            nodes={"leaf": node},
        )
        result = format_tiling_spans(response)
        assert "Hello, world!" in result
        assert "<Span" not in result

    def test_summary_node_wrapped_in_span(self) -> None:
        """Height>0 nodes are wrapped in Span tags with time attributes."""
        node = _make_node(
            "summary",
            text="Discussion about authentication.",
            height=2,
            time_start="2024-01-01T09:00:00Z",
            time_end="2024-01-01T12:00:00Z",
        )
        response = _make_response(
            tiling_ids=["summary"],
            nodes={"summary": node},
        )
        result = format_tiling_spans(response)
        assert '<Span time_start="2024-01-01T09:00:00Z"' in result
        assert 'time_end="2024-01-01T12:00:00Z"' in result
        assert "height=2" in result
        assert "Discussion about authentication." in result
        assert "</Span>" in result

    def test_explanation_header_present(self) -> None:
        """Output includes an Explanation header describing the format."""
        node = _make_node(
            "n1",
            text="Content",
            height=0,
            time_start="2024-01-01T10:00:00Z",
            time_end="2024-01-01T10:05:00Z",
        )
        response = _make_response(
            tiling_ids=["n1"],
            nodes={"n1": node},
        )
        result = format_tiling_spans(response)
        assert "<Explanation>" in result
        assert "</Explanation>" in result
        assert "variable-resolution summary" in result

    def test_zoom_hint_when_summaries_present(self) -> None:
        """Explanation includes zoom hint when max_height > 0."""
        node = _make_node(
            "s1",
            text="Summary",
            height=1,
            time_start="2024-01-01T09:00:00Z",
            time_end="2024-01-01T12:00:00Z",
        )
        response = _make_response(
            tiling_ids=["s1"],
            nodes={"s1": node},
        )
        result = format_tiling_spans(response)
        assert "zoom in" in result.lower() or "time_start/time_end" in result

    def test_no_zoom_hint_when_all_verbatim(self) -> None:
        """Explanation omits zoom hint when all nodes are height=0."""
        node = _make_node(
            "leaf",
            text="Verbatim",
            height=0,
            time_start="2024-01-01T10:00:00Z",
            time_end="2024-01-01T10:00:00Z",
        )
        response = _make_response(
            tiling_ids=["leaf"],
            nodes={"leaf": node},
        )
        result = format_tiling_spans(response)
        assert "To zoom in" not in result

    def test_mixed_heights_ordered(self) -> None:
        """Mixed verbatim + summary nodes preserve tiling order."""
        summary = _make_node(
            "s1",
            text="Summary of morning.",
            height=2,
            time_start="2024-01-01T09:00:00Z",
            time_end="2024-01-01T12:00:00Z",
        )
        leaf = _make_node(
            "leaf1",
            text="Exact message.",
            height=0,
            time_start="2024-01-01T12:00:00Z",
            time_end="2024-01-01T12:00:00Z",
        )
        response = _make_response(
            tiling_ids=["s1", "leaf1"],
            nodes={"s1": summary, "leaf1": leaf},
        )
        result = format_tiling_spans(response)
        # Summary comes before leaf in output
        assert result.index("Summary of morning.") < result.index("Exact message.")

    def test_skips_empty_text_nodes(self) -> None:
        """Nodes with empty text are skipped."""
        empty = _make_node("empty", text="", height=0)
        good = _make_node(
            "good",
            text="Real content",
            height=0,
            time_start="2024-01-01T10:00:00Z",
            time_end="2024-01-01T10:00:00Z",
        )
        response = _make_response(
            tiling_ids=["empty", "good"],
            nodes={"empty": empty, "good": good},
        )
        result = format_tiling_spans(response)
        assert "Real content" in result

    def test_non_temporal_summary_has_span_offsets(self) -> None:
        """Non-temporal summary nodes use span_start/span_end instead of time."""
        node = _make_node(
            "s1", text="Chapter overview.", height=2, span_start=0, span_end=5000
        )
        response = _make_response(
            tiling_ids=["s1"],
            nodes={"s1": node},
        )
        result = format_tiling_spans(response)
        assert "span_start=0" in result
        assert "span_end=5000" in result
        assert "height=2" in result
        assert "time_start" not in result
        assert "time_end" not in result

    def test_non_temporal_explanation_shows_character_range(self) -> None:
        """Non-temporal explanation shows character range, not time range."""
        node = _make_node(
            "s1", text="Summary text.", height=1, span_start=0, span_end=12000
        )
        response = _make_response(
            tiling_ids=["s1"],
            nodes={"s1": node},
        )
        result = format_tiling_spans(response)
        assert "characters" in result
        assert "time_start/time_end" not in result
        assert "token_budget" in result

    def test_non_temporal_verbatim_no_wrapper(self) -> None:
        """Non-temporal height=0 nodes appear as plain text."""
        node = _make_node("leaf", text="Raw paragraph.", height=0)
        response = _make_response(
            tiling_ids=["leaf"],
            nodes={"leaf": node},
        )
        result = format_tiling_spans(response)
        assert "Raw paragraph." in result
        assert "<Span" not in result

    def test_temporal_span_includes_token_estimates(self) -> None:
        """Temporal height>0 Span tags include tokens and verbatim_tokens."""
        node = _make_node(
            "s1",
            text="Morning discussion.",
            height=3,
            token_count=200,
            time_start="2024-01-01T09:00:00Z",
            time_end="2024-01-01T12:00:00Z",
        )
        response = _make_response(
            tiling_ids=["s1"],
            nodes={"s1": node},
        )
        result = format_tiling_spans(response)
        assert "tokens=200" in result
        # 200 * 2^3 = 1600
        assert "verbatim_tokens=1600" in result

    def test_non_temporal_span_includes_token_estimates(self) -> None:
        """Non-temporal height>0 Span tags include tokens and verbatim_tokens."""
        node = _make_node(
            "s1",
            text="Chapter overview.",
            height=2,
            token_count=100,
            span_start=0,
            span_end=5000,
        )
        response = _make_response(
            tiling_ids=["s1"],
            nodes={"s1": node},
        )
        result = format_tiling_spans(response)
        assert "tokens=100" in result
        # 100 * 2^2 = 400
        assert "verbatim_tokens=400" in result

    def test_height_1_verbatim_tokens_is_double(self) -> None:
        """Height=1 node has verbatim_tokens = tokens * 2."""
        node = _make_node(
            "s1",
            text="Brief summary.",
            height=1,
            token_count=50,
            time_start="2024-01-01T10:00:00Z",
            time_end="2024-01-01T11:00:00Z",
        )
        response = _make_response(
            tiling_ids=["s1"],
            nodes={"s1": node},
        )
        result = format_tiling_spans(response)
        assert "tokens=50" in result
        # 50 * 2^1 = 100
        assert "verbatim_tokens=100" in result
