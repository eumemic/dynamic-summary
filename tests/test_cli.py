"""Tests for the RagZoom CLI in the gRPC-backed architecture."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ragzoom.cli import cli
from ragzoom.client.grpc_client import (
    ClearedDocumentResult,
    DocumentStatusView,
    ExecuteQueryOutput,
    NodeSummary,
    RetrievalView,
    WorkerRunSnapshot,
)
from ragzoom.exceptions import InvalidOperationError
from ragzoom.services.document_service import DocumentInfo, SystemStatus
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.services.query_service import QueryResult


class CliMocks(TypedDict):
    store: MagicMock
    document_service: MagicMock
    grpc_client: MagicMock
    vector_index: MagicMock


@pytest.fixture
def runner() -> CliRunner:
    """Provide a Click test runner."""
    return CliRunner()


@pytest.fixture
def cli_mocks() -> Iterator[CliMocks]:
    """Patch expensive dependencies and provide reusable fakes."""
    with (
        patch("ragzoom.cli.create_store_with_docker") as mock_create_store,
        patch("ragzoom.cli.DocumentService") as mock_document_service,
        patch("ragzoom.cli.GrpcRagzoomClient") as mock_grpc_client_cls,
        patch("ragzoom.vector_factory.create_vector_index") as mock_create_vector_index,
    ):
        # Storage backend impersonation used by several commands
        store = MagicMock(name="store")
        document_row = SimpleNamespace(
            id="doc-123",
            embedding_model="text-embedding-3-small",
            file_path="/path/to/file.txt",
            indexed_at=datetime(2023, 1, 1),
        )
        store.list_documents.return_value = [document_row]
        store.get_document_by_id.return_value = document_row
        doc_store = MagicMock(name="doc_store")
        doc_store.nodes.get_leaves.return_value = [MagicMock() for _ in range(5)]
        store.for_document.return_value = doc_store
        mock_create_store.return_value = store

        # Document service behaviour
        document_service = MagicMock(name="document_service")
        document_service.get_system_status.return_value = SystemStatus(
            total_nodes=10,
            leaf_nodes=5,
            tree_depth=3,
            pinned_nodes=0,
        )
        document_service.list_documents.return_value = [
            DocumentInfo(
                document_id="doc-123",
                file_path="/path/to/file.txt",
                indexed_at=datetime(2023, 1, 1),
                node_count=15,
            )
        ]
        document_service.clear_document.return_value = 10
        document_service.clear_all_documents.return_value = 50
        mock_document_service.return_value = document_service

        # Vector index stubbed for clear command side effects
        vector_index = MagicMock(name="vector_index")
        mock_create_vector_index.return_value = vector_index

        # gRPC client facade
        index_result = IndexingResult(
            document_id="doc-123",
            chunks_created=5,
            tree_depth=3,
            mutated_nodes=5,
            resummarized_nodes=2,
            new_leaves=5,
            telemetry=None,
        )
        node_summary = NodeSummary(
            node_id="n1",
            text="Leaf text",
            token_count=42,
            span_start=0,
            span_end=42,
            parent_id="",
            left_child_id="",
            right_child_id="",
            height=0,
        )
        retrieval_view = RetrievalView(
            selected_ids=["n1"],
            tiling_ids=["n1"],
            scores={"n1": 1.0},
            coverage_map={"n1": True},
            nodes={"n1": node_summary},
        )
        execute_output = ExecuteQueryOutput(
            query_result=QueryResult(
                summary="This is a summary of the content.",
                token_count=50,
                nodes_retrieved=1,
                tiling_size=1,
                query_id="",
            ),
            retrieval=retrieval_view,
            visualization="<viz>",
            validation_warning="",
        )
        grpc_client = MagicMock(name="grpc_client")
        grpc_client.__enter__.return_value = grpc_client
        grpc_client.__exit__.return_value = None
        grpc_client.append_text.return_value = index_result
        grpc_client.execute_query.return_value = execute_output
        grpc_client.get_document_status.return_value = DocumentStatusView(
            document_id="doc-123",
            leaf_count=5,
            tree_depth=4,
            has_pending_work=False,
        )
        grpc_client.iter_worker_snapshots.return_value = [
            WorkerRunSnapshot(
                message="workers drained",
                idle=True,
                queue_depth=0,
                inflight=0,
                documents={},
            )
        ]
        grpc_client.clear_document.return_value = ClearedDocumentResult(
            document_id="doc-123",
            deleted_nodes=10,
            document_existed=True,
        )
        grpc_client.clear_all_documents.return_value = [
            ClearedDocumentResult(
                document_id="doc-123",
                deleted_nodes=10,
                document_existed=True,
            )
        ]
        mock_grpc_client_cls.return_value = grpc_client

        yield CliMocks(
            store=store,
            document_service=document_service,
            grpc_client=grpc_client,
            vector_index=vector_index,
        )


@pytest.fixture
def api_key() -> Iterator[None]:
    """Ensure an API key is visible to the CLI while the test runs."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        yield


def _write_temp_file(tmp_path: Path, name: str, content: str) -> Path:
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_cli_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "RagZoom: Incremental, hierarchical RAG memory system." in result.output


def test_status_command(runner: CliRunner, cli_mocks: CliMocks, api_key: None) -> None:
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "SYSTEM STATUS" in result.output
    cli_mocks["document_service"].get_system_status.assert_called_once()


def test_index_command_with_file(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None, tmp_path: Path
) -> None:
    file_path = _write_temp_file(tmp_path, "doc.txt", "Test content")
    result = runner.invoke(cli, ["index", str(file_path)])
    assert result.exit_code == 0
    assert "Document indexed successfully" in result.output
    cli_mocks["grpc_client"].append_text.assert_called_once()
    cli_mocks["grpc_client"].get_document_status.assert_called_once_with("doc.txt")
    assert "Tree height: 4" in result.output


def test_index_command_without_awaiting_workers(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None, tmp_path: Path
) -> None:
    file_path = _write_temp_file(tmp_path, "doc.txt", "Test content")
    result = runner.invoke(cli, ["index", str(file_path), "--no-await-workers"])

    assert result.exit_code == 0
    assert "Leaf ingestion queued" in result.output
    cli_mocks["grpc_client"].iter_worker_snapshots.assert_not_called()
    cli_mocks["grpc_client"].get_document_status.assert_not_called()


def test_index_command_rejects_telemetry_without_await(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None, tmp_path: Path
) -> None:
    file_path = _write_temp_file(tmp_path, "doc.txt", "Test content")
    result = runner.invoke(
        cli,
        [
            "index",
            str(file_path),
            "--telemetry",
            "metrics.json",
            "--no-await-workers",
        ],
    )

    assert result.exit_code != 0
    assert "--telemetry cannot be combined with --no-await-workers" in result.output


def test_index_command_with_document_id(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None, tmp_path: Path
) -> None:
    file_path = _write_temp_file(tmp_path, "doc.txt", "Test content")
    result = runner.invoke(
        cli,
        ["index", str(file_path), "--document-id", "my-doc"],
    )
    assert result.exit_code == 0
    call = cli_mocks["grpc_client"].append_text.call_args
    assert call.kwargs["document_id"] == "my-doc"
    cli_mocks["grpc_client"].get_document_status.assert_called_once_with("my-doc")


def test_index_append_requires_document_id(
    runner: CliRunner, api_key: None, tmp_path: Path
) -> None:
    file_path = _write_temp_file(tmp_path, "doc.txt", "Append me")
    result = runner.invoke(cli, ["index", str(file_path), "--append"])
    assert result.exit_code != 0
    assert "--document-id is required" in result.output


def test_index_append_invokes_append(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None, tmp_path: Path
) -> None:
    file_path = _write_temp_file(tmp_path, "doc.txt", "Append me")
    result = runner.invoke(
        cli,
        [
            "index",
            str(file_path),
            "--document-id",
            "append-doc",
            "--append",
        ],
    )
    assert result.exit_code == 0
    cli_mocks["grpc_client"].append_text.assert_called_once()
    assert not cli_mocks["grpc_client"].index_document.called


def test_index_with_summary_system_prompt(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None, tmp_path: Path
) -> None:
    """Test that --summary-system-prompt is passed to the gRPC client."""
    custom_prompt = "You are a legal document summarizer. Output ONLY compressed text."
    file_path = _write_temp_file(tmp_path, "doc.txt", "Test content for indexing")
    result = runner.invoke(
        cli,
        [
            "index",
            str(file_path),
            "--summary-system-prompt",
            custom_prompt,
        ],
    )
    assert result.exit_code == 0
    call = cli_mocks["grpc_client"].append_text.call_args
    assert call.kwargs["summary_system_prompt"] == custom_prompt


def test_query_command(runner: CliRunner, cli_mocks: CliMocks, api_key: None) -> None:
    result = runner.invoke(cli, ["query", "Tell me about cats", "-d", "doc-123"])
    assert result.exit_code == 0
    assert "This is a summary of the content." in result.output
    cli_mocks["grpc_client"].execute_query.assert_called_once()


def test_query_with_options(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    result = runner.invoke(
        cli,
        [
            "query",
            "Tell me about cats",
            "-d",
            "doc-123",
            "--num-seeds",
            "5",
            "--token-budget",
            "1000",
        ],
    )
    assert result.exit_code == 0
    call = cli_mocks["grpc_client"].execute_query.call_args
    assert call.kwargs["num_seeds"] == 5
    assert call.kwargs["budget_tokens"] == 1000


def test_query_with_time_window(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that --time-start and --time-end CLI options are passed to the gRPC client."""
    result = runner.invoke(
        cli,
        [
            "query",
            "Tell me about cats",
            "-d",
            "doc-123",
            "--time-start",
            "2024-01-21T14:00:00Z",
            "--time-end",
            "2024-01-21T15:00:00Z",
        ],
    )
    assert result.exit_code == 0
    call = cli_mocks["grpc_client"].execute_query.call_args
    assert call.kwargs["time_start"] == "2024-01-21T14:00:00Z"
    assert call.kwargs["time_end"] == "2024-01-21T15:00:00Z"


def test_query_time_window_defaults_to_none(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that time window parameters default to None when not provided."""
    result = runner.invoke(
        cli,
        [
            "query",
            "Tell me about cats",
            "-d",
            "doc-123",
        ],
    )
    assert result.exit_code == 0
    call = cli_mocks["grpc_client"].execute_query.call_args
    assert call.kwargs["time_start"] is None
    assert call.kwargs["time_end"] is None


def test_pin_command(runner: CliRunner, cli_mocks: CliMocks, api_key: None) -> None:
    result = runner.invoke(cli, ["pin", "node-123"])
    assert result.exit_code == 0
    cli_mocks["document_service"].pin_node.assert_called_once_with("node-123")


def test_pin_command_failure(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    cli_mocks["document_service"].pin_node.side_effect = InvalidOperationError(
        "pin_node", "Node is already pinned"
    )
    result = runner.invoke(cli, ["pin", "node-999"])
    assert result.exit_code == 1
    assert "Failed to pin node node-999" in result.output


def test_documents_command(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    result = runner.invoke(cli, ["documents"])
    assert result.exit_code == 0
    assert "Document ID: doc-123" in result.output
    assert "Leaf nodes: 5" in result.output
    cli_mocks["document_service"].list_documents.assert_called_once()


def test_clear_specific_document(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    result = runner.invoke(
        cli,
        ["clear", "--document-id", "doc-123", "--confirm"],
    )
    assert result.exit_code == 0
    cli_mocks["grpc_client"].clear_document.assert_called_once_with("doc-123")


def test_clear_all_data(runner: CliRunner, cli_mocks: CliMocks, api_key: None) -> None:
    result = runner.invoke(cli, ["clear", "--confirm"])
    assert result.exit_code == 0
    cli_mocks["grpc_client"].clear_all_documents.assert_called_once()


def test_serve_command(runner: CliRunner, api_key: None) -> None:
    with patch("uvicorn.run") as mock_uvicorn:
        result = runner.invoke(cli, ["serve", "--port", "8080"])
        assert result.exit_code == 0
        mock_uvicorn.assert_called_once_with(
            "ragzoom.api:app", host="127.0.0.1", port=8080, reload=False
        )


def test_status_succeeds_without_api_key(runner: CliRunner) -> None:
    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
        result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0


def test_set_session_pid_command(runner: CliRunner) -> None:
    """Test that set-session-pid CLI command calls the underlying function."""
    with patch("ragzoom.cli.set_session_pid") as mock_set_pid:
        result = runner.invoke(cli, ["set-session-pid", "my-session-123", "42"])
        assert result.exit_code == 0
        mock_set_pid.assert_called_once_with("my-session-123", 42)
        assert "my-session-123" in result.output
        assert "42" in result.output


def test_query_json_output(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that --json flag outputs valid JSON with expected schema."""

    # Set up a more realistic response with tiling nodes
    node1 = NodeSummary(
        node_id="node-1",
        text="First chunk of content",
        token_count=25,
        span_start=0,
        span_end=100,
        parent_id="",
        left_child_id="",
        right_child_id="",
        height=0,
        time_start="2024-01-21T10:00:00Z",
        time_end="2024-01-21T10:15:00Z",
    )
    node2 = NodeSummary(
        node_id="node-2",
        text="Second chunk of content",
        token_count=25,
        span_start=100,
        span_end=200,
        parent_id="",
        left_child_id="",
        right_child_id="",
        height=0,
        time_start="2024-01-21T10:15:00Z",
        time_end="2024-01-21T10:30:00Z",
    )
    retrieval_view = RetrievalView(
        selected_ids=["node-1"],
        tiling_ids=["node-1", "node-2"],
        scores={"node-1": 0.9, "node-2": 0.7},
        coverage_map={"node-1": True, "node-2": True},
        nodes={"node-1": node1, "node-2": node2},
    )
    execute_output = ExecuteQueryOutput(
        query_result=QueryResult(
            summary="Summary of the content about cats.",
            token_count=50,
            nodes_retrieved=2,
            tiling_size=2,
            query_id="test-query",
            seed_count=1,
            verbatim_count=0,
            actual_start=0,
            actual_end=200,
        ),
        retrieval=retrieval_view,
        visualization="",
        validation_warning="",
    )
    cli_mocks["grpc_client"].execute_query.return_value = execute_output

    result = runner.invoke(
        cli,
        ["query", "Tell me about cats", "-d", "doc-123", "--json"],
    )
    assert result.exit_code == 0

    # Parse output as JSON
    output = json.loads(result.output)

    # Verify top-level schema fields
    assert output["summary"] == "Summary of the content about cats."
    assert output["token_count"] == 50
    assert output["seed_count"] == 1
    assert output["tiling_size"] == 2
    assert output["query"] == "Tell me about cats"
    assert output["document_id"] == "doc-123"
    assert output["actual_span"] == {"start": 0, "end": 200}

    # Verify tiling structure
    assert len(output["tiling"]) == 2
    assert output["tiling"][0]["node_id"] == "node-1"
    assert output["tiling"][0]["is_seed"] is True
    assert output["tiling"][0]["time_start"] == "2024-01-21T10:00:00Z"
    assert output["tiling"][1]["node_id"] == "node-2"
    assert output["tiling"][1]["is_seed"] is False


def test_query_json_output_suppresses_debug_visualization(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that --json with --debug still outputs only JSON (no visualization)."""

    result = runner.invoke(
        cli,
        ["query", "Tell me about cats", "-d", "doc-123", "--json", "--debug"],
    )
    assert result.exit_code == 0

    # Output should be valid JSON (no extra text from debug visualization)
    output = json.loads(result.output)
    assert "summary" in output
    assert "tiling" in output


def test_query_json_error_response(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that --json flag outputs JSON error format when exception occurs."""
    from ragzoom.exceptions import DocumentNotFoundError

    # Make the gRPC client raise an exception
    cli_mocks["grpc_client"].execute_query.side_effect = DocumentNotFoundError(
        "test-doc"
    )

    result = runner.invoke(
        cli,
        ["query", "Tell me about cats", "-d", "test-doc", "--json"],
    )

    # Should exit with error code
    assert result.exit_code == 1

    # Output should be valid JSON with error schema
    output = json.loads(result.output)
    assert "error" in output
    assert "code" in output
    assert output["code"] == "NOT_FOUND"
    assert "test-doc" in output["error"]


def test_query_no_bm25_flag(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that --no-bm25 flag disables BM25 hybrid search.

    Spec: specs/bm25-hybrid-search.md § CLI Flag
    Success: `ragzoom query --no-bm25 "test"` disables BM25 search
    """
    # Run query with --no-bm25 flag
    result = runner.invoke(
        cli,
        ["query", "Tell me about cats", "-d", "doc-123", "--no-bm25"],
    )

    # Command should succeed
    assert result.exit_code == 0

    # Verify the gRPC client was called
    grpc_client = cli_mocks["grpc_client"]
    grpc_client.execute_query.assert_called_once()

    # Note: The use_bm25 flag is stored in QueryConfig but is not yet
    # passed to execute_query (that's Phase 21: Retriever Integration).
    # This test verifies the CLI flag is accepted without error.
