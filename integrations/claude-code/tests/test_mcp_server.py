"""Tests for MCP server identity resolution and timestamp handling."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ragzoom_claude_code.mcp_server import (
    _MEMORY_PERSONA_GUIDANCE,
    _ensure_timezone,
    _get_session_id,
)


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


class TestEnsureTimezone:
    """Tests for _ensure_timezone timestamp normalization."""

    def test_none_passes_through(self) -> None:
        assert _ensure_timezone(None) is None

    def test_naive_timestamp_gets_utc(self) -> None:
        result = _ensure_timezone("2024-01-15T10:00:00")
        assert result == "2024-01-15T10:00:00+00:00"

    def test_utc_z_suffix_preserved(self) -> None:
        result = _ensure_timezone("2024-01-15T10:00:00Z")
        assert result is not None
        assert result.endswith("+00:00") or result.endswith("Z")

    def test_explicit_offset_preserved(self) -> None:
        result = _ensure_timezone("2024-01-15T10:00:00+05:30")
        assert result is not None
        assert "+05:30" in result

    def test_date_only_gets_utc(self) -> None:
        result = _ensure_timezone("2024-01-15")
        assert result is not None
        assert "+00:00" in result


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
                time_start=None,
                time_end=None,
                server_address="localhost:50051",
                search_guidance=_MEMORY_PERSONA_GUIDANCE,
                session_id=None,
            )
            assert "The auth bug was in the JWT validation." in result

    def test_recall_tool_forwards_time_constraints(self) -> None:
        """Recall tool normalizes and forwards time_start/time_end."""
        expected_doc_id = "test-session-doc"

        mock_result = MagicMock()
        mock_result.answer = "Found it in the morning session."

        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": expected_doc_id}),
            patch(
                "ragzoom_claude_code.mcp_server.execute_search",
                return_value=mock_result,
            ) as mock_search,
        ):
            from ragzoom_claude_code.mcp_server import recall

            result = recall(
                query="What happened?",
                time_start="2024-01-15T10:00:00",
                time_end="2024-01-15T12:00:00",
            )

            # Bare timestamps get UTC via _ensure_timezone
            mock_search.assert_called_once_with(
                question="What happened?",
                document_id=expected_doc_id,
                time_start="2024-01-15T10:00:00+00:00",
                time_end="2024-01-15T12:00:00+00:00",
                server_address="localhost:50051",
                search_guidance=_MEMORY_PERSONA_GUIDANCE,
                session_id=None,
            )
            assert "Found it in the morning session." in result

    def test_recall_returns_session_id_in_response(self) -> None:
        """Recall response includes session_id for follow-up queries."""
        mock_result = MagicMock()
        mock_result.answer = "You were debugging the auth flow."
        mock_result.session_id = "search-session-abc123"

        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": "test-doc"}),
            patch(
                "ragzoom_claude_code.mcp_server.execute_search",
                return_value=mock_result,
            ),
        ):
            from ragzoom_claude_code.mcp_server import recall

            result = recall(query="What was I debugging?")

        assert "You were debugging the auth flow." in result
        assert "search-session-abc123" in result

    def test_recall_forwards_session_id_for_continuation(self) -> None:
        """Passing session_id resumes an existing search agent session."""
        mock_result = MagicMock()
        mock_result.answer = "Specifically, the JWT expiry check."
        mock_result.session_id = "search-session-abc123"

        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": "test-doc"}),
            patch(
                "ragzoom_claude_code.mcp_server.execute_search",
                return_value=mock_result,
            ) as mock_search,
        ):
            from ragzoom_claude_code.mcp_server import recall

            recall(
                query="Which part of auth specifically?",
                session_id="search-session-abc123",
            )

            mock_search.assert_called_once_with(
                question="Which part of auth specifically?",
                document_id="test-doc",
                time_start=None,
                time_end=None,
                server_address="localhost:50051",
                search_guidance=_MEMORY_PERSONA_GUIDANCE,
                session_id="search-session-abc123",
            )

    def test_recall_no_session_id_when_server_returns_none(self) -> None:
        """Response omits session metadata when server returns no session_id."""
        mock_result = MagicMock()
        mock_result.answer = "No context found."
        mock_result.session_id = None

        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": "test-doc"}),
            patch(
                "ragzoom_claude_code.mcp_server.execute_search",
                return_value=mock_result,
            ),
        ):
            from ragzoom_claude_code.mcp_server import recall

            result = recall(query="Something obscure?")

        assert result == "No context found."
        assert "Session:" not in result
