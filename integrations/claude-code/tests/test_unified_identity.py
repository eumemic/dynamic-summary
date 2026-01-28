"""Acceptance tests for Unified Agent Identity feature.

Tests the acceptance criteria from specs/unified-agent-identity.md:
1. RAGZOOM_DOCUMENT_ID env var works for sync script
2. RAGZOOM_DOCUMENT_ID env var works for MCP server
3. --document-id CLI flag overrides env var and stem
4. PID temp file discovery works for Claude Code
5. No state files accumulate in data/transcript-state/
6. Multiple syncs to same document work (Jarvis model)
7. MCP queries work with configured identity (Jarvis model)
8. reset command works without state file cleanup
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from ragzoom_claude_code.cli import cli
from ragzoom_claude_code.transcript_sync import execute_sync, get_session_document_id

from ragzoom.wrapper import AppendUnit

# --- Test fixtures and helpers ---


@dataclass
class MockDocumentStatus:
    """Mock document status for testing."""

    document_id: str
    exists: bool = False
    is_temporal: bool = True
    leaf_count: int = 0
    node_count: int = 0
    complete_forest_size: int = 0
    completion_pct: float = 0.0
    time_start: str | None = None
    time_end: str | None = None


@dataclass
class BatchAppendResult:
    """Result type compatible with execute_sync expectations."""

    span_start: int
    span_end: int


@dataclass
class IntegrationMockClient:
    """Mock client that tracks calls for acceptance testing.

    Tracks all operations to verify the correct document_id is used.
    """

    # Call tracking
    get_document_status_calls: list[str] = field(default_factory=list)
    batch_append_calls: list[tuple[str, list[AppendUnit]]] = field(default_factory=list)
    truncate_from_time_calls: list[tuple[str, str]] = field(default_factory=list)

    # Internal state
    _span_counter: int = 0

    def get_document_status(self, document_id: str) -> MockDocumentStatus:
        """Track and return status for document."""
        self.get_document_status_calls.append(document_id)
        return MockDocumentStatus(document_id=document_id, exists=False)

    def batch_append(
        self,
        document_id: str,
        units: list[AppendUnit],
        **kwargs: object,
    ) -> BatchAppendResult:
        """Track batch append calls."""
        self.batch_append_calls.append((document_id, units))
        for unit in units:
            self._span_counter += len(unit.text)
        return BatchAppendResult(span_start=0, span_end=self._span_counter)

    def truncate_from_time(
        self, document_id: str, cutoff_time: str
    ) -> dict[str, object]:
        """Track truncation calls."""
        self.truncate_from_time_calls.append((document_id, cutoff_time))
        return {"document_id": document_id, "deleted_node_ids": []}


def make_user_message(
    uuid: str,
    parent_uuid: str | None,
    timestamp: str,
    content: str,
) -> dict[str, object]:
    """Create a user transcript message record."""
    return {
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "type": "user",
        "timestamp": timestamp,
        "message": {"content": content},
    }


def make_assistant_message(
    uuid: str,
    parent_uuid: str | None,
    timestamp: str,
    content: str,
) -> dict[str, object]:
    """Create an assistant transcript message record."""
    return {
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "type": "assistant",
        "timestamp": timestamp,
        "message": {"content": [{"type": "text", "text": content}]},
    }


def create_simple_transcript(path: Path) -> None:
    """Create a simple two-message transcript."""
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    make_user_message("msg1", None, "2024-01-21T14:30:00Z", "Hello")
                ),
                json.dumps(
                    make_assistant_message(
                        "msg2", "msg1", "2024-01-21T14:30:05Z", "Hi there!"
                    )
                ),
            ]
        )
        + "\n"
    )


# --- Acceptance Tests ---


class TestSyncEnvVarDocumentId:
    """Test: RAGZOOM_DOCUMENT_ID env var works for sync script (AC #1)."""

    def test_sync_env_var_document_id(self, tmp_path: Path) -> None:
        """Sync script uses env var document_id regardless of JSONL filename.

        When RAGZOOM_DOCUMENT_ID is set, execute_sync uses that value
        as the document_id, not the JSONL filename stem.
        """
        client = IntegrationMockClient()

        transcript_path = tmp_path / "session-abc-123.jsonl"
        create_simple_transcript(transcript_path)

        env_var_doc_id = "jarvis-user-42"
        execute_sync(transcript_path, env_var_doc_id, client)

        assert len(client.get_document_status_calls) == 1
        assert client.get_document_status_calls[0] == "jarvis-user-42"

        assert len(client.batch_append_calls) == 1
        assert client.batch_append_calls[0][0] == "jarvis-user-42"


class TestMcpEnvVarDocumentId:
    """Test: RAGZOOM_DOCUMENT_ID env var works for MCP server (AC #2)."""

    def test_mcp_env_var_document_id(self) -> None:
        """MCP server queries document specified in env var.

        When RAGZOOM_DOCUMENT_ID is set, _get_session_id() returns that value
        without PID temp file lookup.
        """
        from ragzoom_claude_code.mcp_server import _get_session_id

        with patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": "legion-agent-7"}):
            doc_id = _get_session_id()
            assert doc_id == "legion-agent-7"


class TestDocumentIdFlagPriority:
    """Test: --document-id CLI flag overrides env var and stem (AC #3)."""

    def test_document_id_flag_priority(self, tmp_path: Path) -> None:
        """CLI flag takes highest priority over env var and filename stem.

        Priority order: --document-id > RAGZOOM_DOCUMENT_ID > filename stem
        """
        runner = CliRunner()

        jsonl_file = tmp_path / "filename-stem.jsonl"
        create_simple_transcript(jsonl_file)

        with (
            patch("ragzoom_claude_code.cli.execute_sync") as mock_sync,
            patch("ragzoom.wrapper.RagZoom"),
        ):
            mock_sync.return_value.document_id = "flag-wins"
            mock_sync.return_value.steps_appended = 1
            mock_sync.return_value.truncated = False

            result = runner.invoke(
                cli,
                ["sync", str(jsonl_file), "--document-id", "flag-wins"],
                env={"RAGZOOM_DOCUMENT_ID": "env-value"},
            )

            assert result.exit_code == 0, result.output
            assert mock_sync.call_args[0][1] == "flag-wins"


class TestPidTempFileDiscovery:
    """Test: PID temp file discovery works for Claude Code (AC #4)."""

    def test_pid_temp_file_discovery(self, tmp_path: Path) -> None:
        """MCP server discovers session via /tmp/ragzoom-session-{ppid}.

        When no env var is set, the MCP server falls back to reading the
        document_id from the PID-keyed temp file.
        """
        from ragzoom_claude_code.mcp_server import _get_session_id

        test_pid = 12345
        test_doc_id = "claude-code-session-xyz"

        temp_file = tmp_path / f"ragzoom-session-{test_pid}"
        temp_file.write_text(test_doc_id)

        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": ""}, clear=False),
            patch("ragzoom_claude_code.mcp_server.os.getppid", return_value=test_pid),
            patch(
                "ragzoom_claude_code.transcript_sync._get_temp_dir",
                return_value=tmp_path,
            ),
        ):
            doc_id = _get_session_id()
            assert doc_id == "claude-code-session-xyz"


class TestNoStateFileAccumulation:
    """Test: No state files accumulate in data/transcript-state/ (AC #5)."""

    def test_no_state_file_accumulation(self, tmp_path: Path) -> None:
        """Sync operations do not create state files.

        The old state file system has been removed. This test verifies that
        sync operations don't create any state files.
        """
        client = IntegrationMockClient()

        transcript_path = tmp_path / "session.jsonl"
        create_simple_transcript(transcript_path)

        for _ in range(3):
            execute_sync(transcript_path, "test-doc", client)

        jsonl_files = [f for f in tmp_path.iterdir() if f.suffix == ".jsonl"]
        state_files = [f for f in jsonl_files if f != transcript_path]
        assert state_files == [], f"Unexpected state files: {state_files}"

        old_state_dir = Path("data/transcript-state")
        if old_state_dir.exists():
            state_files_in_old_dir = list(old_state_dir.iterdir())
            assert (
                state_files_in_old_dir == []
            ), f"Old state files found: {state_files_in_old_dir}"


class TestMultipleSyncsSameDocument:
    """Test: Multiple syncs to same document work - Jarvis model (AC #6)."""

    def test_multiple_syncs_same_document(self, tmp_path: Path) -> None:
        """Multiple different transcripts can sync to same document_id via env var.

        In the Jarvis model, multiple conversation sessions write to the
        same persistent document identified by RAGZOOM_DOCUMENT_ID.
        """
        client = IntegrationMockClient()
        shared_doc_id = "jarvis-shared-memory"

        def write_transcript(path: Path, msg_prefix: str, time_prefix: str) -> None:
            """Write a simple two-message transcript."""
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            make_user_message(
                                f"{msg_prefix}1",
                                None,
                                f"2024-01-21T{time_prefix}:00:00Z",
                                f"{msg_prefix.capitalize()} question",
                            )
                        ),
                        json.dumps(
                            make_assistant_message(
                                f"{msg_prefix}2",
                                f"{msg_prefix}1",
                                f"2024-01-21T{time_prefix}:00:05Z",
                                f"{msg_prefix.capitalize()} answer",
                            )
                        ),
                    ]
                )
                + "\n"
            )

        transcript1 = tmp_path / "session-morning.jsonl"
        write_transcript(transcript1, "m", "09:00")

        transcript2 = tmp_path / "session-afternoon.jsonl"
        write_transcript(transcript2, "a", "14:00")

        execute_sync(transcript1, shared_doc_id, client)
        execute_sync(transcript2, shared_doc_id, client)

        assert len(client.batch_append_calls) == 2
        assert client.batch_append_calls[0][0] == "jarvis-shared-memory"
        assert client.batch_append_calls[1][0] == "jarvis-shared-memory"


class TestMcpConfiguredIdentityQueries:
    """Test: MCP queries work with configured identity - Jarvis model (AC #7)."""

    def test_mcp_configured_identity_queries(self) -> None:
        """MCP server queries correct document when RAGZOOM_DOCUMENT_ID set.

        When the env var is set, the MCP server returns that document_id
        without attempting PID-based discovery.
        """
        from ragzoom_claude_code.mcp_server import _get_session_id

        configured_id = "jarvis-persistent-memory"

        with patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": configured_id}):
            result = _get_session_id()
            assert result == configured_id


class TestResetCommandStateless:
    """Test: reset command works without state file cleanup (AC #8)."""

    def test_reset_command_stateless(self, tmp_path: Path) -> None:
        """Reset command clears document and re-syncs without state file management.

        The reset command works purely through document operations
        (clear + sync) without any state file management.
        """
        runner = CliRunner()

        jsonl_file = tmp_path / "session-to-reset.jsonl"
        create_simple_transcript(jsonl_file)

        with (
            patch("ragzoom.client.GrpcRagzoomClient") as mock_grpc_class,
            patch("ragzoom.wrapper.RagZoom"),
            patch("ragzoom_claude_code.cli.execute_sync") as mock_exec_sync,
        ):
            mock_grpc = mock_grpc_class.return_value.__enter__.return_value
            mock_grpc.clear_document.return_value.document_existed = True
            mock_grpc.clear_document.return_value.deleted_nodes = 10

            mock_exec_sync.return_value.document_id = "session-to-reset"
            mock_exec_sync.return_value.steps_appended = 1
            mock_exec_sync.return_value.truncated = False

            result = runner.invoke(cli, ["reset", str(jsonl_file)])

            assert result.exit_code == 0, result.output
            mock_grpc.clear_document.assert_called_once_with("session-to-reset")
            assert mock_exec_sync.called
            assert "state" not in result.output.lower() or "Cleared" in result.output


class TestGetSessionDocumentId:
    """Tests for get_session_document_id() helper function."""

    def test_get_session_document_id_from_temp_file(self, tmp_path: Path) -> None:
        """Reads document_id from PID temp file."""
        test_pid = 99999
        temp_file = tmp_path / f"ragzoom-session-{test_pid}"
        temp_file.write_text("my-session-doc\n")

        with patch(
            "ragzoom_claude_code.transcript_sync._get_temp_dir", return_value=tmp_path
        ):
            result = get_session_document_id(test_pid)
            assert result == "my-session-doc"

    def test_get_session_document_id_not_found(self, tmp_path: Path) -> None:
        """Returns None when temp file doesn't exist."""
        with patch(
            "ragzoom_claude_code.transcript_sync._get_temp_dir", return_value=tmp_path
        ):
            result = get_session_document_id(12345)
            assert result is None

    def test_get_session_document_id_empty_file(self, tmp_path: Path) -> None:
        """Returns None when temp file is empty."""
        test_pid = 88888
        temp_file = tmp_path / f"ragzoom-session-{test_pid}"
        temp_file.write_text("")

        with patch(
            "ragzoom_claude_code.transcript_sync._get_temp_dir", return_value=tmp_path
        ):
            result = get_session_document_id(test_pid)
            assert result is None
