"""Tests for config persistence in foreground server mode."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ragzoom.cli import cli


def test_config_persisted_without_daemon_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Config should be persisted even when --daemon is not used."""
    # Set up isolated state dir
    monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))

    config_file = tmp_path / "my-config.json"
    config_file.write_text('{"target_chunk_tokens": null}')

    # Mock run_server to prevent actual server start
    with patch("ragzoom.cli.run_server"):
        runner = CliRunner()
        result = runner.invoke(cli, ["server", "start", "--config", str(config_file)])

    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Config should be persisted
    from ragzoom.daemon import read_config_file

    persisted = read_config_file()
    assert persisted is not None, "Config was not persisted for foreground server"
    assert persisted.get("target_chunk_tokens") is None
