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
    DocumentWorkStatus,
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
        patch("ragzoom.cli.ensure_server_running", return_value="127.0.0.1:50051"),
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
        grpc_client.get_document_work_status.return_value = DocumentWorkStatus(
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
    cli_mocks["grpc_client"].get_document_work_status.assert_called_once_with("doc.txt")
    assert "Tree height: 4" in result.output


def test_index_command_without_awaiting_workers(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None, tmp_path: Path
) -> None:
    file_path = _write_temp_file(tmp_path, "doc.txt", "Test content")
    result = runner.invoke(cli, ["index", str(file_path), "--no-await-workers"])

    assert result.exit_code == 0
    assert "Leaf ingestion queued" in result.output
    cli_mocks["grpc_client"].iter_worker_snapshots.assert_not_called()
    cli_mocks["grpc_client"].get_document_work_status.assert_not_called()


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
    cli_mocks["grpc_client"].get_document_work_status.assert_called_once_with("my-doc")


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


def test_index_with_summarization_guidance(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None, tmp_path: Path
) -> None:
    """Test that --summarization-guidance is passed to the gRPC client."""
    custom_guidance = "This is medical documentation. Preserve all medication names."
    file_path = _write_temp_file(tmp_path, "doc.txt", "Test content for indexing")
    result = runner.invoke(
        cli,
        [
            "index",
            str(file_path),
            "--summarization-guidance",
            custom_guidance,
        ],
    )
    assert result.exit_code == 0
    call = cli_mocks["grpc_client"].append_text.call_args
    assert call.kwargs["summarization_guidance"] == custom_guidance


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


def test_server_start_daemon_flag(runner: CliRunner) -> None:
    """Test that --daemon flag triggers daemonization.

    Spec: specs/daemon-lifecycle.md § CLI Commands > ragzoom server start
    Success: `ragzoom server start --daemon` runs in background
    """
    with (
        patch("ragzoom.cli.run_server") as mock_run_server,
        patch("ragzoom.cli.daemonize") as mock_daemonize,
        patch("ragzoom.cli.write_port_file") as mock_write_port,
        patch("ragzoom.cli.install_shutdown_handlers") as mock_handlers,
    ):
        result = runner.invoke(cli, ["server", "start", "--daemon"])

        # Command should succeed
        assert result.exit_code == 0

        # Daemonize should be called (forks to background)
        mock_daemonize.assert_called_once()

        # Port file should be written
        mock_write_port.assert_called_once_with(50051)

        # Signal handlers should be installed for graceful shutdown
        mock_handlers.assert_called_once()

        # Server should be started
        mock_run_server.assert_called_once()


def test_server_start_daemon_flag_with_custom_port(runner: CliRunner) -> None:
    """Test that --daemon with --port writes correct port to port file."""
    with (
        patch("ragzoom.cli.run_server") as mock_run_server,
        patch("ragzoom.cli.daemonize") as mock_daemonize,
        patch("ragzoom.cli.write_port_file") as mock_write_port,
        patch("ragzoom.cli.install_shutdown_handlers"),
    ):
        result = runner.invoke(cli, ["server", "start", "--daemon", "--port", "50052"])

        assert result.exit_code == 0
        mock_daemonize.assert_called_once()
        mock_write_port.assert_called_once_with(50052)
        mock_run_server.assert_called_once()

        # Verify port passed to run_server
        call_args = mock_run_server.call_args
        assert call_args[0][0].port == 50052


def test_server_start_without_daemon_flag(runner: CliRunner) -> None:
    """Test that without --daemon, daemonize is NOT called (foreground mode)."""
    with (
        patch("ragzoom.cli.run_server") as mock_run_server,
        patch("ragzoom.cli.daemonize") as mock_daemonize,
        patch("ragzoom.cli.write_port_file") as mock_write_port,
    ):
        # Note: Without --daemon, the server runs in foreground (current behavior)
        # We don't actually run it in tests, so we mock run_server to prevent it
        result = runner.invoke(cli, ["server", "start"])

        # Should succeed
        assert result.exit_code == 0

        # Daemonize should NOT be called in foreground mode
        mock_daemonize.assert_not_called()

        # Port file should NOT be written in foreground mode
        mock_write_port.assert_not_called()

        # Server should still be started
        mock_run_server.assert_called_once()


def test_server_stop_command(runner: CliRunner) -> None:
    """Test that `server stop` sends SIGTERM and cleans up state.

    Spec: specs/daemon-lifecycle.md § CLI Commands > ragzoom server stop
    Success: Sends SIGTERM, waits for graceful shutdown, cleans up state files
    """
    with (
        patch("ragzoom.cli.read_pid_file", return_value=12345),
        patch(
            "ragzoom.cli.is_pid_stale", side_effect=[False, False, True]
        ) as mock_stale,
        patch("ragzoom.cli.os.kill") as mock_kill,
        patch("ragzoom.cli.cleanup_stale_state") as mock_cleanup,
        patch("ragzoom.cli.time.sleep"),
    ):
        result = runner.invoke(cli, ["server", "stop"])

        # Command should succeed
        assert result.exit_code == 0

        # Should send SIGTERM to the process
        import signal

        mock_kill.assert_called_once_with(12345, signal.SIGTERM)

        # Should check if process is dead (stale) repeatedly
        assert mock_stale.call_count >= 2

        # Should clean up state files
        mock_cleanup.assert_called_once()

        # Should print success message
        assert "Stopped" in result.output or "stopped" in result.output


def test_server_stop_no_daemon_running(runner: CliRunner) -> None:
    """Test that `server stop` handles no daemon running gracefully."""
    with patch("ragzoom.cli.read_pid_file", return_value=None):
        result = runner.invoke(cli, ["server", "stop"])

        # Command should still succeed (no-op)
        assert result.exit_code == 0

        # Should indicate no daemon is running
        assert (
            "not running" in result.output.lower()
            or "no daemon" in result.output.lower()
        )


def test_server_stop_stale_pid(runner: CliRunner) -> None:
    """Test that `server stop` handles stale PID (process already dead)."""
    with (
        patch("ragzoom.cli.read_pid_file", return_value=12345),
        patch("ragzoom.cli.is_pid_stale", return_value=True),
        patch("ragzoom.cli.cleanup_stale_state") as mock_cleanup,
    ):
        result = runner.invoke(cli, ["server", "stop"])

        # Command should succeed (cleanup only)
        assert result.exit_code == 0

        # Should clean up stale state files
        mock_cleanup.assert_called_once()

        # Should indicate process was already dead
        assert (
            "not running" in result.output.lower() or "stale" in result.output.lower()
        )


def test_server_stop_timeout(runner: CliRunner) -> None:
    """Test that `server stop` handles timeout waiting for process to die."""
    with (
        patch("ragzoom.cli.read_pid_file", return_value=12345),
        patch("ragzoom.cli.is_pid_stale", return_value=False),  # Never dies
        patch("ragzoom.cli.os.kill") as mock_kill,
        patch("ragzoom.cli.cleanup_stale_state") as mock_cleanup,
        patch("ragzoom.cli.time.sleep"),
        patch(
            "ragzoom.cli.time.monotonic",
            side_effect=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        ),
    ):
        result = runner.invoke(cli, ["server", "stop"])

        # Should send SIGTERM
        import signal

        mock_kill.assert_called_with(12345, signal.SIGTERM)

        # Should still clean up state files even on timeout
        mock_cleanup.assert_called_once()

        # Should indicate timeout
        assert "timeout" in result.output.lower() or "force" in result.output.lower()


def test_server_status_command(runner: CliRunner) -> None:
    """Test that `server status` shows running daemon info.

    Spec: specs/daemon-lifecycle.md § CLI Commands > ragzoom server status
    Success: Shows "Running: PID X, port Y, uptime Z" when daemon is running
    """
    with (
        patch("ragzoom.cli.read_pid_file", return_value=12345),
        patch("ragzoom.cli.read_port_file", return_value=50051),
        patch("ragzoom.cli.is_pid_stale", return_value=False),
        patch("ragzoom.cli.get_process_uptime", return_value="2h 15m"),
    ):
        result = runner.invoke(cli, ["server", "status"])

        # Command should succeed
        assert result.exit_code == 0

        # Should show PID
        assert "12345" in result.output

        # Should show port
        assert "50051" in result.output

        # Should show uptime
        assert "2h 15m" in result.output

        # Should indicate running
        assert "running" in result.output.lower()


def test_server_status_not_running(runner: CliRunner) -> None:
    """Test that `server status` shows 'Not running' when no daemon."""
    with patch("ragzoom.cli.read_pid_file", return_value=None):
        result = runner.invoke(cli, ["server", "status"])

        # Command should succeed
        assert result.exit_code == 0

        # Should indicate not running
        assert "not running" in result.output.lower()


def test_server_status_stale_pid(runner: CliRunner) -> None:
    """Test that `server status` detects stale PID (process died)."""
    with (
        patch("ragzoom.cli.read_pid_file", return_value=12345),
        patch("ragzoom.cli.is_pid_stale", return_value=True),
    ):
        result = runner.invoke(cli, ["server", "status"])

        # Command should succeed
        assert result.exit_code == 0

        # Should indicate not running (stale PID means effectively not running)
        assert "not running" in result.output.lower()


def test_server_status_no_port_file(runner: CliRunner) -> None:
    """Test that `server status` handles missing port file gracefully."""
    with (
        patch("ragzoom.cli.read_pid_file", return_value=12345),
        patch("ragzoom.cli.read_port_file", return_value=None),
        patch("ragzoom.cli.is_pid_stale", return_value=False),
        patch("ragzoom.cli.get_process_uptime", return_value="5m"),
    ):
        result = runner.invoke(cli, ["server", "status"])

        # Command should succeed
        assert result.exit_code == 0

        # Should show PID
        assert "12345" in result.output

        # Should indicate running (port unknown is OK)
        assert "running" in result.output.lower()


def test_server_logs_command(runner: CliRunner, tmp_path: Path) -> None:
    """Test that `server logs` shows daemon log contents.

    Spec: specs/daemon-lifecycle.md § CLI Commands > ragzoom server logs
    Success: Shows daemon.log contents (default: 50 lines)
    """
    # Create a test log file with 60 lines
    log_file = tmp_path / "daemon.log"
    log_lines = [f"Log line {i}\n" for i in range(60)]
    log_file.write_text("".join(log_lines))

    with patch("ragzoom.cli.get_log_file_path", return_value=log_file):
        result = runner.invoke(cli, ["server", "logs"])

        # Command should succeed
        assert result.exit_code == 0

        # Should show the last 50 lines (default)
        # Lines 10-59 should be present (0-indexed: lines 10 through 59)
        assert "Log line 10" in result.output
        assert "Log line 59" in result.output

        # Line 9 should NOT be present (it's outside the default 50)
        assert "Log line 9" not in result.output


def test_server_logs_with_n_flag(runner: CliRunner, tmp_path: Path) -> None:
    """Test that `server logs -n` limits the number of lines."""
    # Create a test log file
    log_file = tmp_path / "daemon.log"
    log_lines = [f"Log line {i}\n" for i in range(20)]
    log_file.write_text("".join(log_lines))

    with patch("ragzoom.cli.get_log_file_path", return_value=log_file):
        result = runner.invoke(cli, ["server", "logs", "-n", "5"])

        # Command should succeed
        assert result.exit_code == 0

        # Should show only last 5 lines (lines 15-19)
        assert "Log line 15" in result.output
        assert "Log line 19" in result.output

        # Line 14 should NOT be present
        assert "Log line 14" not in result.output


def test_server_logs_no_log_file(runner: CliRunner, tmp_path: Path) -> None:
    """Test that `server logs` handles missing log file gracefully."""
    # Point to a non-existent log file
    log_file = tmp_path / "daemon.log"

    with patch("ragzoom.cli.get_log_file_path", return_value=log_file):
        result = runner.invoke(cli, ["server", "logs"])

        # Command should succeed but indicate no logs
        assert result.exit_code == 0
        assert (
            "no log file" in result.output.lower()
            or "not found" in result.output.lower()
        )


def test_server_logs_empty_file(runner: CliRunner, tmp_path: Path) -> None:
    """Test that `server logs` handles empty log file gracefully."""
    # Create an empty log file
    log_file = tmp_path / "daemon.log"
    log_file.write_text("")

    with patch("ragzoom.cli.get_log_file_path", return_value=log_file):
        result = runner.invoke(cli, ["server", "logs"])

        # Command should succeed
        assert result.exit_code == 0
        # Output should be empty or indicate no logs
        # (no assertion on specific message - empty output is valid)


def test_document_status_human_format(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that document-status outputs human-readable format by default."""
    cli_mocks["grpc_client"].get_document_status.return_value = DocumentStatusView(
        document_id="session-abc123",
        exists=True,
        is_temporal=True,
        leaf_count=100,
        node_count=142,
        complete_forest_size=197,
        completion_pct=72.1,
        time_start="2026-01-25T22:47:42Z",
        time_end="2026-01-26T07:04:15Z",
    )

    result = runner.invoke(cli, ["document-status", "session-abc123"])

    assert result.exit_code == 0
    assert "Document: session-abc123" in result.output
    assert "Type: temporal" in result.output
    assert "Leaves: 100" in result.output
    assert "Nodes: 142 / 197 (72.1% complete)" in result.output
    assert "Time range: 2026-01-25T22:47:42Z to 2026-01-26T07:04:15Z" in result.output
    cli_mocks["grpc_client"].get_document_status.assert_called_once_with(
        "session-abc123"
    )


def test_document_status_json_format(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that document-status --json outputs JSON format."""
    cli_mocks["grpc_client"].get_document_status.return_value = DocumentStatusView(
        document_id="session-abc123",
        exists=True,
        is_temporal=True,
        leaf_count=100,
        node_count=142,
        complete_forest_size=197,
        completion_pct=72.1,
        time_start="2026-01-25T22:47:42Z",
        time_end="2026-01-26T07:04:15Z",
    )

    result = runner.invoke(cli, ["document-status", "session-abc123", "--json"])

    assert result.exit_code == 0
    output = json.loads(result.output)
    assert output["document_id"] == "session-abc123"
    assert output["exists"] is True
    assert output["is_temporal"] is True
    assert output["leaf_count"] == 100
    assert output["node_count"] == 142
    assert output["complete_forest_size"] == 197
    assert output["completion_pct"] == 72.1
    assert output["time_start"] == "2026-01-25T22:47:42Z"
    assert output["time_end"] == "2026-01-26T07:04:15Z"


def test_document_status_nonexistent_document(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that document-status handles non-existent documents."""
    cli_mocks["grpc_client"].get_document_status.return_value = DocumentStatusView(
        document_id="unknown-doc",
        exists=False,
        is_temporal=False,
        leaf_count=0,
        node_count=0,
        complete_forest_size=0,
        completion_pct=0.0,
        time_start=None,
        time_end=None,
    )

    result = runner.invoke(cli, ["document-status", "unknown-doc"])

    assert result.exit_code == 0
    assert "Document: unknown-doc" in result.output
    assert "Status: does not exist" in result.output
    # Should NOT show type, leaves, nodes, or time range for non-existent docs
    assert "Type:" not in result.output
    assert "Leaves:" not in result.output


def test_document_status_non_temporal_document(
    runner: CliRunner, cli_mocks: CliMocks, api_key: None
) -> None:
    """Test that document-status handles non-temporal documents correctly."""
    cli_mocks["grpc_client"].get_document_status.return_value = DocumentStatusView(
        document_id="regular-doc",
        exists=True,
        is_temporal=False,
        leaf_count=50,
        node_count=75,
        complete_forest_size=97,
        completion_pct=77.3,
        time_start=None,
        time_end=None,
    )

    result = runner.invoke(cli, ["document-status", "regular-doc"])

    assert result.exit_code == 0
    assert "Document: regular-doc" in result.output
    assert "Type: non-temporal" in result.output
    assert "Leaves: 50" in result.output
    assert "Nodes: 75 / 97 (77.3% complete)" in result.output
    # Should NOT show time range for non-temporal documents
    assert "Time range:" not in result.output
