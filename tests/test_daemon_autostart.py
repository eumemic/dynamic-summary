"""Tests for daemon auto-start functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ragzoom.client.grpc_client import (
    ClearedDocumentResult,
    ExecuteQueryOutput,
    RetrievalView,
)
from ragzoom.daemon import DaemonStartError, ensure_server_running
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.services.query_service import QueryResult


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


class TestCliAutoStartTriggers:
    """Tests for CLI commands triggering daemon auto-start."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Provide a Click test runner."""
        return CliRunner()

    def test_query_autostarts_daemon(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Query command triggers auto-start when using default server address."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))
        # Clear any server address env var so default is used
        monkeypatch.delenv("RAGZOOM_SERVER_ADDRESS", raising=False)

        from ragzoom.cli import cli

        with (
            patch("ragzoom.cli.ensure_server_running") as mock_ensure,
            patch("ragzoom.cli.GrpcRagzoomClient") as mock_client_cls,
        ):
            mock_ensure.return_value = "127.0.0.1:50051"
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_client.execute_query.return_value = ExecuteQueryOutput(
                query_result=QueryResult(
                    summary="Test summary",
                    token_count=10,
                    nodes_retrieved=1,
                    tiling_size=1,
                    query_id="test-query-id",
                ),
                retrieval=RetrievalView(
                    tiling_ids=[],
                    nodes={},
                    selected_ids=[],
                    scores={},
                    coverage_map={},
                ),
                visualization="",
                validation_warning="",
            )

            result = runner.invoke(cli, ["query", "test query", "-d", "doc-123"])

            # ensure_server_running should be called when no explicit address
            mock_ensure.assert_called_once()
            assert result.exit_code == 0

    def test_query_skips_autostart_with_explicit_address(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Query command does NOT auto-start when explicit server address given."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))
        monkeypatch.delenv("RAGZOOM_SERVER_ADDRESS", raising=False)

        from ragzoom.cli import cli

        with (
            patch("ragzoom.cli.ensure_server_running") as mock_ensure,
            patch("ragzoom.cli.GrpcRagzoomClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_client.execute_query.return_value = ExecuteQueryOutput(
                query_result=QueryResult(
                    summary="Test summary",
                    token_count=10,
                    nodes_retrieved=1,
                    tiling_size=1,
                    query_id="test-query-id",
                ),
                retrieval=RetrievalView(
                    tiling_ids=[],
                    nodes={},
                    selected_ids=[],
                    scores={},
                    coverage_map={},
                ),
                visualization="",
                validation_warning="",
            )

            runner.invoke(
                cli,
                [
                    "query",
                    "test query",
                    "-d",
                    "doc-123",
                    "--server-address",
                    "192.168.1.100:50051",
                ],
            )

            # ensure_server_running should NOT be called with explicit address
            mock_ensure.assert_not_called()
            # GrpcRagzoomClient should be called with the explicit address
            mock_client_cls.assert_called_once_with("192.168.1.100:50051")

    def test_index_autostarts_daemon(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Index command triggers auto-start when using default server address."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))
        monkeypatch.delenv("RAGZOOM_SERVER_ADDRESS", raising=False)

        from ragzoom.cli import cli

        # Create a test file to index
        test_file = tmp_path / "test.txt"
        test_file.write_text("Test content for indexing")

        with (
            patch("ragzoom.cli.ensure_server_running") as mock_ensure,
            patch("ragzoom.cli.GrpcRagzoomClient") as mock_client_cls,
        ):
            mock_ensure.return_value = "127.0.0.1:50051"
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_client.append_text.return_value = IndexingResult(
                document_id="test.txt",
                chunks_created=1,
                tree_depth=1,
                mutated_nodes=1,
                resummarized_nodes=0,
                new_leaves=1,
            )

            runner.invoke(cli, ["index", str(test_file)])

            # ensure_server_running should be called when no explicit address
            mock_ensure.assert_called_once()

    def test_clear_autostarts_daemon(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Clear command triggers auto-start when using default server address."""
        monkeypatch.setenv("RAGZOOM_STATE_DIR", str(tmp_path))
        monkeypatch.delenv("RAGZOOM_SERVER_ADDRESS", raising=False)

        from ragzoom.cli import cli

        with (
            patch("ragzoom.cli.ensure_server_running") as mock_ensure,
            patch("ragzoom.cli.GrpcRagzoomClient") as mock_client_cls,
        ):
            mock_ensure.return_value = "127.0.0.1:50051"
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)

            mock_client.clear_document.return_value = ClearedDocumentResult(
                document_id="doc-123",
                document_existed=True,
                deleted_nodes=10,
            )

            runner.invoke(cli, ["clear", "-d", "doc-123", "--confirm"])

            # ensure_server_running should be called when no explicit address
            mock_ensure.assert_called_once()
