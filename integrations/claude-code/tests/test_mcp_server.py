"""Tests for MCP server identity resolution."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

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

    def test_env_var_skips_state_file_lookup(self, tmp_path: Path) -> None:
        """When env var is set, state files are not consulted."""
        # This test verifies that _get_state_dir() is not called when env var is set
        with (
            patch.dict(os.environ, {"RAGZOOM_DOCUMENT_ID": "skip-state-lookup"}),
            patch(
                "ragzoom_claude_code.mcp_server._get_state_dir"
            ) as mock_get_state_dir,
        ):
            result = _get_session_id()

        assert result == "skip-state-lookup"
        mock_get_state_dir.assert_not_called()
