"""Tests for PID temp file identity discovery."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest
from ragzoom_claude_code.transcript_sync import get_session_document_id


@pytest.fixture
def temp_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Patch _get_temp_dir to use tmp_path for all tests."""
    with patch(
        "ragzoom_claude_code.transcript_sync._get_temp_dir",
        return_value=tmp_path,
    ):
        yield tmp_path


class TestGetSessionDocumentId:
    """Tests for get_session_document_id() function."""

    def test_reads_document_id_from_temp_file(self, temp_dir: Path) -> None:
        """Returns document_id from /tmp/ragzoom-session-{pid} when file exists."""
        pid = 12345
        temp_file = temp_dir / f"ragzoom-session-{pid}"
        temp_file.write_text("my-document-id\n")

        result = get_session_document_id(pid)

        assert result == "my-document-id"

    def test_strips_whitespace_from_document_id(self, temp_dir: Path) -> None:
        """Document ID is stripped of leading/trailing whitespace."""
        pid = 54321
        temp_file = temp_dir / f"ragzoom-session-{pid}"
        temp_file.write_text("  document-with-spaces  \n\n")

        result = get_session_document_id(pid)

        assert result == "document-with-spaces"

    def test_returns_none_when_file_not_found(self, temp_dir: Path) -> None:
        """Returns None when temp file doesn't exist."""
        pid = 99999  # No file created for this PID

        result = get_session_document_id(pid)

        assert result is None

    def test_returns_none_for_empty_file(self, temp_dir: Path) -> None:
        """Returns None when temp file exists but is empty."""
        pid = 11111
        temp_file = temp_dir / f"ragzoom-session-{pid}"
        temp_file.write_text("")

        result = get_session_document_id(pid)

        assert result is None

    def test_returns_none_for_whitespace_only_file(self, temp_dir: Path) -> None:
        """Returns None when temp file contains only whitespace."""
        pid = 22222
        temp_file = temp_dir / f"ragzoom-session-{pid}"
        temp_file.write_text("   \n\n  ")

        result = get_session_document_id(pid)

        assert result is None
