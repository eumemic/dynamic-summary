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
