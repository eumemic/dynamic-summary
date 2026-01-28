"""Tests for CLI command options."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from ragzoom_claude_code.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    """CLI runner for invoking commands."""
    return CliRunner()


@pytest.fixture
def mock_sync_result() -> MagicMock:
    """Mock result from execute_sync."""
    result = MagicMock()
    result.document_id = "test-doc"
    result.steps_appended = 5
    result.truncated = False
    return result


@pytest.fixture
def mock_sync_env(mock_sync_result: MagicMock) -> Generator[MagicMock, None, None]:
    """Patch execute_sync and RagZoom client for sync command tests."""
    with (
        patch(
            "ragzoom_claude_code.cli.execute_sync", return_value=mock_sync_result
        ) as mock_sync,
        patch("ragzoom.wrapper.RagZoom"),
    ):
        yield mock_sync


class TestSyncDocumentIdOption:
    """Tests for --document-id option on sync command."""

    def test_document_id_flag_overrides_filename(
        self, runner: CliRunner, mock_sync_env: MagicMock, tmp_path: Path
    ) -> None:
        """The --document-id flag overrides the filename stem."""
        jsonl_file = tmp_path / "session-abc.jsonl"
        jsonl_file.write_text('{"type": "test"}\n')

        result = runner.invoke(
            cli, ["sync", str(jsonl_file), "--document-id", "custom-doc-id"]
        )

        assert result.exit_code == 0, result.output
        assert mock_sync_env.call_args[0][1] == "custom-doc-id"

    def test_short_flag_works(
        self, runner: CliRunner, mock_sync_env: MagicMock, tmp_path: Path
    ) -> None:
        """The -d short flag works for document-id."""
        jsonl_file = tmp_path / "session-abc.jsonl"
        jsonl_file.write_text('{"type": "test"}\n')

        result = runner.invoke(cli, ["sync", str(jsonl_file), "-d", "short-flag-doc"])

        assert result.exit_code == 0, result.output
        assert mock_sync_env.call_args[0][1] == "short-flag-doc"

    def test_env_var_used_when_no_flag(
        self, runner: CliRunner, mock_sync_env: MagicMock, tmp_path: Path
    ) -> None:
        """RAGZOOM_DOCUMENT_ID env var is used when no flag provided."""
        jsonl_file = tmp_path / "session-abc.jsonl"
        jsonl_file.write_text('{"type": "test"}\n')

        result = runner.invoke(
            cli, ["sync", str(jsonl_file)], env={"RAGZOOM_DOCUMENT_ID": "env-var-doc"}
        )

        assert result.exit_code == 0, result.output
        assert mock_sync_env.call_args[0][1] == "env-var-doc"

    def test_flag_takes_priority_over_env_var(
        self, runner: CliRunner, mock_sync_env: MagicMock, tmp_path: Path
    ) -> None:
        """The --document-id flag takes priority over env var."""
        jsonl_file = tmp_path / "session-abc.jsonl"
        jsonl_file.write_text('{"type": "test"}\n')

        result = runner.invoke(
            cli,
            ["sync", str(jsonl_file), "--document-id", "flag-wins"],
            env={"RAGZOOM_DOCUMENT_ID": "env-var-doc"},
        )

        assert result.exit_code == 0, result.output
        assert mock_sync_env.call_args[0][1] == "flag-wins"

    def test_defaults_to_filename_stem(
        self, runner: CliRunner, mock_sync_env: MagicMock, tmp_path: Path
    ) -> None:
        """Document ID defaults to JSONL filename stem when no override."""
        jsonl_file = tmp_path / "my-session-id.jsonl"
        jsonl_file.write_text('{"type": "test"}\n')

        result = runner.invoke(cli, ["sync", str(jsonl_file)])

        assert result.exit_code == 0, result.output
        assert mock_sync_env.call_args[0][1] == "my-session-id"

    def test_priority_order(
        self, runner: CliRunner, mock_sync_env: MagicMock, tmp_path: Path
    ) -> None:
        """Full priority order: flag > env var > filename stem."""
        jsonl_file = tmp_path / "filename-stem.jsonl"
        jsonl_file.write_text('{"type": "test"}\n')

        # Flag wins over env var and stem
        result = runner.invoke(
            cli,
            ["sync", str(jsonl_file), "-d", "flag-value"],
            env={"RAGZOOM_DOCUMENT_ID": "env-value"},
        )
        assert result.exit_code == 0, result.output
        assert mock_sync_env.call_args[0][1] == "flag-value"

        # Env var wins over stem (no flag)
        result = runner.invoke(
            cli, ["sync", str(jsonl_file)], env={"RAGZOOM_DOCUMENT_ID": "env-value"}
        )
        assert result.exit_code == 0, result.output
        assert mock_sync_env.call_args[0][1] == "env-value"

        # Stem is used when no flag or env var
        result = runner.invoke(cli, ["sync", str(jsonl_file)], env={})
        assert result.exit_code == 0, result.output
        assert mock_sync_env.call_args[0][1] == "filename-stem"
