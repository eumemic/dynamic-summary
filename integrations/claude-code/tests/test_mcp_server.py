"""Tests for MCP server identity resolution."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ragzoom_claude_code.mcp_server import _get_session_id


class TestMcpServerEnvVarIdentity:
    """Tests for RAGZOOM_DOCUMENT_ID env var support in MCP server."""

    def test_env_var_takes_priority(self, tmp_path: Path) -> None:
        """RAGZOOM_DOCUMENT_ID env var is used when set."""
        # Patch environment to set RAGZOOM_DOCUMENT_ID
        with patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": "jarvis-user-123"}):
            result = _get_session_id()

        assert result == "jarvis-user-123"

    def test_env_var_returns_string_not_tuple(self) -> None:
        """When env var is set, return type is str not tuple."""
        with patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": "configured-doc"}):
            result = _get_session_id()

        # Should be a string, not a tuple
        assert isinstance(result, str)
        assert result == "configured-doc"

    def test_empty_env_var_not_used(self, tmp_path: Path) -> None:
        """Empty RAGZOOM_DOCUMENT_ID is treated as not set."""
        # Empty string should not be used as document ID
        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": ""}),
            pytest.raises(ValueError, match="No session found"),
        ):
            _get_session_id()

    def test_env_var_skips_pid_temp_file_lookup(self, tmp_path: Path) -> None:
        """When env var is set, PID temp file is not consulted."""
        # This verifies get_session_document_id() is not called when env var is set
        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": "skip-temp-file-lookup"}),
            patch(
                "ragzoom_claude_code.mcp_server.get_session_document_id"
            ) as mock_get_session,
        ):
            result = _get_session_id()

        assert result == "skip-temp-file-lookup"
        mock_get_session.assert_not_called()


class TestMcpServerPidTempFileDiscovery:
    """Tests for PID temp file discovery in MCP server."""

    def test_pid_temp_file_discovery(self, tmp_path: Path) -> None:
        """MCP server discovers session via PID temp file when env var not set.

        This is a unit test that verifies the MCP server calls get_session_document_id
        with the correct parent PID.
        """
        expected_doc_id = "discovered-session-abc123"
        parent_pid = 12345

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.getppid", return_value=parent_pid),
            patch(
                "ragzoom_claude_code.mcp_server.get_session_document_id"
            ) as mock_get_session,
        ):
            mock_get_session.return_value = expected_doc_id
            result = _get_session_id()

        assert result == expected_doc_id
        mock_get_session.assert_called_once_with(parent_pid)

    def test_pid_temp_file_discovery_end_to_end(self, tmp_path: Path) -> None:
        """End-to-end test: creates temp file, verifies MCP server discovers session.

        This test actually writes to a temp file and verifies the full discovery flow
        without mocking get_session_document_id(). Per acceptance criteria #4:
        "PID temp file discovery works for Claude Code sessions"
        """
        expected_doc_id = "e2e-discovered-session-xyz789"
        parent_pid = 54321

        # Write temp file simulating what SessionStart hook does
        temp_file = tmp_path / f"ragzoom-session-{parent_pid}"
        temp_file.write_text(expected_doc_id)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.getppid", return_value=parent_pid),
            # Patch _get_temp_dir in transcript_sync to use our test directory
            patch(
                "ragzoom_claude_code.transcript_sync._get_temp_dir",
                return_value=tmp_path,
            ),
        ):
            result = _get_session_id()

        assert result == expected_doc_id

    def test_pid_temp_file_not_found_raises_error(self) -> None:
        """Error raised when PID temp file doesn't exist and no env var."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("os.getppid", return_value=99999),
            patch(
                "ragzoom_claude_code.mcp_server.get_session_document_id",
                return_value=None,
            ),
            pytest.raises(ValueError, match="No session found for PID 99999"),
        ):
            _get_session_id()


class TestRecallToolSearch:
    """Tests for recall tool using the agentic search endpoint."""

    def test_recall_tool_calls_execute_search(self) -> None:
        """Recall tool passes question to execute_search and returns answer."""
        expected_doc_id = "test-session-doc"

        mock_result = MagicMock()
        mock_result.answer = "The auth bug was in the JWT validation."

        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": expected_doc_id}),
            patch(
                "ragzoom_claude_code.mcp_server.execute_search",
                return_value=mock_result,
            ) as mock_search,
        ):
            from ragzoom_claude_code.mcp_server import recall

            result = recall(query="What was the auth bug?")

            mock_search.assert_called_once_with(
                question="What was the auth bug?",
                document_id=expected_doc_id,
                server_address="localhost:50051",
            )
            assert result == "The auth bug was in the JWT validation."
