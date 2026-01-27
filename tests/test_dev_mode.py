"""Tests for dev/prod mode separation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from ragzoom.cli import (
    DEV_PORT,
    PRODUCTION_PORT,
    _get_default_port,
    _is_dev_invocation,
)
from ragzoom.daemon import (
    DaemonStartError,
    ensure_server_running,
    get_daemon_state_dir,
)
from ragzoom.daemon import (
    _is_dev_invocation as daemon_is_dev_invocation,
)


class TestDevInvocationDetection:
    """Test detection of dev vs production invocation."""

    def test_module_invocation_is_dev(self) -> None:
        """Invoking via python -m ragzoom.cli is detected as dev mode."""
        with patch("sys.argv", ["/path/to/ragzoom/cli.py", "server", "start"]):
            assert _is_dev_invocation() is True
            assert daemon_is_dev_invocation() is True

    def test_entry_point_is_production(self) -> None:
        """Invoking via ragzoom entry point is detected as production mode."""
        with patch("sys.argv", ["/usr/local/bin/ragzoom", "server", "start"]):
            assert _is_dev_invocation() is False
            assert daemon_is_dev_invocation() is False

    def test_bare_ragzoom_is_production(self) -> None:
        """Bare 'ragzoom' command (no path) is production mode."""
        with patch("sys.argv", ["ragzoom", "server", "start"]):
            assert _is_dev_invocation() is False
            assert daemon_is_dev_invocation() is False

    def test_windows_module_path_is_dev(self) -> None:
        """Windows-style module path is detected as dev mode."""
        with patch("sys.argv", ["C:\\code\\ragzoom\\cli.py", "server", "start"]):
            assert _is_dev_invocation() is True
            assert daemon_is_dev_invocation() is True

    def test_empty_argv_is_production(self) -> None:
        """Empty sys.argv defaults to production mode."""
        with patch("sys.argv", []):
            assert _is_dev_invocation() is False
            assert daemon_is_dev_invocation() is False


class TestDefaultPort:
    """Test default port selection based on invocation mode."""

    def test_dev_mode_uses_dev_port(self) -> None:
        """Dev mode uses DEV_PORT (50052)."""
        with patch("sys.argv", ["/path/to/ragzoom/cli.py", "server", "start"]):
            assert _get_default_port() == DEV_PORT
            assert _get_default_port() == 50052

    def test_production_mode_uses_production_port(self) -> None:
        """Production mode uses PRODUCTION_PORT (50051)."""
        with patch("sys.argv", ["ragzoom", "server", "start"]):
            assert _get_default_port() == PRODUCTION_PORT
            assert _get_default_port() == 50051


class TestStateDirSeparation:
    """Test state directory selection based on invocation mode."""

    def test_dev_mode_uses_dev_state_dir(self) -> None:
        """Dev mode uses ragzoom-dev state directory."""
        with (
            patch("sys.argv", ["/path/to/ragzoom/cli.py", "server", "start"]),
            patch.dict(os.environ, {}, clear=True),
        ):
            state_dir = get_daemon_state_dir()
            assert "ragzoom-dev" in str(state_dir)

    def test_production_mode_uses_production_state_dir(self) -> None:
        """Production mode uses ragzoom state directory."""
        with (
            patch("sys.argv", ["ragzoom", "server", "start"]),
            patch.dict(os.environ, {}, clear=True),
        ):
            state_dir = get_daemon_state_dir()
            assert str(state_dir).endswith("ragzoom")
            assert "ragzoom-dev" not in str(state_dir)

    def test_env_override_takes_precedence(self, tmp_path: Path) -> None:
        """RAGZOOM_STATE_DIR overrides dev/prod defaults."""
        custom_dir = str(tmp_path / "custom-state")
        with (
            patch("sys.argv", ["/path/to/ragzoom/cli.py", "server", "start"]),
            patch.dict(os.environ, {"RAGZOOM_STATE_DIR": custom_dir}),
        ):
            state_dir = get_daemon_state_dir()
            assert str(state_dir) == custom_dir


class TestEnsureServerRunningDevMode:
    """Test ensure_server_running behavior in dev mode."""

    def test_dev_mode_fails_fast_when_server_not_running(self, tmp_path: Path) -> None:
        """In dev mode, ensure_server_running fails fast instead of auto-starting."""
        with (
            patch("sys.argv", ["/path/to/ragzoom/cli.py", "index", "file.txt"]),
            patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}),
            patch("ragzoom.daemon.is_server_healthy", return_value=False),
        ):
            with pytest.raises(DaemonStartError) as exc_info:
                ensure_server_running()

            # Error message should guide user to start server manually
            assert "Dev server is not running" in str(exc_info.value)
            assert "python -m ragzoom.cli server start" in str(exc_info.value)

    def test_dev_mode_returns_address_when_server_healthy(self, tmp_path: Path) -> None:
        """In dev mode, returns address if server is already healthy."""
        port_file = tmp_path / "daemon.port"
        port_file.write_text("50052\n")

        with (
            patch("sys.argv", ["/path/to/ragzoom/cli.py", "index", "file.txt"]),
            patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}),
            patch("ragzoom.daemon.is_server_healthy", return_value=True),
        ):
            address = ensure_server_running()
            assert address == "127.0.0.1:50052"


class TestPortConstants:
    """Test port constant values."""

    def test_production_port_value(self) -> None:
        """PRODUCTION_PORT is 50051."""
        assert PRODUCTION_PORT == 50051

    def test_dev_port_value(self) -> None:
        """DEV_PORT is 50052."""
        assert DEV_PORT == 50052

    def test_ports_are_different(self) -> None:
        """Dev and production ports must be different."""
        assert DEV_PORT != PRODUCTION_PORT
