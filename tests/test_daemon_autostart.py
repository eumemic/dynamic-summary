"""Tests for daemon auto-start functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragzoom.daemon import DaemonStartError, ensure_server_running


class TestEnsureServerRunning:
    """Tests for ensure_server_running() function."""

    def test_returns_address_when_server_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When server is already healthy, returns address without starting."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))

        with (
            patch("ragzoom.daemon.is_server_healthy", return_value=True),
            patch("ragzoom.daemon.get_server_address", return_value="127.0.0.1:50051"),
            patch("ragzoom.daemon.cleanup_stale_state") as mock_cleanup,
        ):
            address = ensure_server_running()

            assert address == "127.0.0.1:50051"
            # Should NOT have tried to start/cleanup
            mock_cleanup.assert_not_called()

    def test_starts_daemon_when_unhealthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When server is unhealthy, starts daemon and returns address."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))

        # Track health check calls - first unhealthy, then healthy after start
        health_calls = [False, True]

        def mock_is_healthy() -> bool:
            return health_calls.pop(0)

        with (
            # Force production mode for auto-start test
            patch("ragzoom.daemon._is_dev_invocation", return_value=False),
            patch("ragzoom.daemon.is_server_healthy", side_effect=mock_is_healthy),
            patch("ragzoom.daemon.cleanup_stale_state") as mock_cleanup,
            patch("ragzoom.daemon.start_daemon") as mock_start,
            patch("ragzoom.daemon.get_server_address", return_value="127.0.0.1:50051"),
        ):
            address = ensure_server_running()

            assert address == "127.0.0.1:50051"
            mock_cleanup.assert_called_once()
            mock_start.assert_called_once()

    def test_raises_on_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When daemon never becomes healthy, raises DaemonStartError."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))

        with (
            # Force production mode for auto-start test
            patch("ragzoom.daemon._is_dev_invocation", return_value=False),
            patch("ragzoom.daemon.is_server_healthy", return_value=False),
            patch("ragzoom.daemon.cleanup_stale_state"),
            patch("ragzoom.daemon.start_daemon"),
            patch("ragzoom.daemon.get_server_address", return_value=None),
            pytest.raises(DaemonStartError, match="timed out"),
        ):
            # Use very short timeout for test
            ensure_server_running(timeout=0.1)

    def test_returns_address_on_successful_start(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After successful start, returns the server address."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))

        # Simulate: first check unhealthy, then healthy after start
        health_sequence = iter([False, True])

        with (
            # Force production mode for auto-start test
            patch("ragzoom.daemon._is_dev_invocation", return_value=False),
            patch("ragzoom.daemon.is_server_healthy", side_effect=health_sequence),
            patch("ragzoom.daemon.cleanup_stale_state"),
            patch("ragzoom.daemon.start_daemon"),
            patch("ragzoom.daemon.get_server_address", return_value="127.0.0.1:50055"),
        ):
            address = ensure_server_running()
            assert address == "127.0.0.1:50055"


class TestStartDaemon:
    """Tests for start_daemon() helper function."""

    def _get_popen_cmd(self, mock_popen: MagicMock) -> list[str]:
        """Extract the command list from a mocked Popen call."""
        call_args = mock_popen.call_args
        assert call_args is not None
        if call_args.args:
            cmd = call_args.args[0]
        else:
            cmd = call_args.kwargs.get("args")
        assert cmd is not None
        assert isinstance(cmd, list)
        return cmd

    def test_spawns_subprocess_with_daemon_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """start_daemon() launches ragzoom server start --daemon."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))

        from ragzoom.daemon import start_daemon

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            start_daemon()

            mock_popen.assert_called_once()
            cmd = self._get_popen_cmd(mock_popen)
            assert "server" in cmd
            assert "start" in cmd
            assert "--daemon" in cmd

    def test_uses_default_port(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """start_daemon() uses default port 50051."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))

        from ragzoom.daemon import start_daemon

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            start_daemon()

            cmd = self._get_popen_cmd(mock_popen)
            # Should include --port with default value
            assert "--port" in cmd

    def test_accepts_custom_port(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """start_daemon() accepts custom port parameter."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))

        from ragzoom.daemon import start_daemon

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            start_daemon(port=50099)

            cmd = self._get_popen_cmd(mock_popen)
            # Should include the custom port
            assert "--port" in cmd
            port_idx = cmd.index("--port")
            assert cmd[port_idx + 1] == "50099"


# NOTE: TestCliAutoStartTriggers was removed because CLI no longer triggers
# daemon auto-start. The CLI now uses _resolve_server_address() which performs
# a TCP connectivity check and fails fast if the server is unreachable.
# See tests/test_cli.py for the new behavior tests:
# - test_no_autostart_function_exists
# - test_resolve_server_address_fails_fast
# - test_server_unreachable_error_message
