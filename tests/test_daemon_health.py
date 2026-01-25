"""Tests for daemon health check functionality."""

import signal
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import grpc
import pytest

from ragzoom.daemon import (
    get_server_address,
    grpc_health_check,
    is_server_healthy,
)


def make_rpc_error(status_code: object) -> object:
    """Create a grpc.RpcError with a configurable status code for testing."""
    error = grpc.RpcError()
    error.code = MagicMock(return_value=status_code)  # type: ignore[method-assign,unused-ignore]
    return error


class TestGetServerAddress:
    """Tests for get_server_address() function."""

    def test_returns_address_from_port_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When port file exists, returns localhost:port."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))

            port_file = state_dir / "daemon.port"
            port_file.write_text("50055\n")

            address = get_server_address()
            assert address == "127.0.0.1:50055"

    def test_returns_none_when_no_port_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When port file doesn't exist, returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            address = get_server_address()
            assert address is None


class TestGrpcHealthCheck:
    """Tests for grpc_health_check() function."""

    def test_returns_true_when_server_responds(self) -> None:
        """When gRPC server is reachable and responds, returns True."""
        mock_channel = MagicMock()
        mock_stub = MagicMock()
        mock_stub.GetDocument = MagicMock(return_value=MagicMock())

        with patch("grpc.insecure_channel", return_value=mock_channel):
            with patch(
                "ragzoom.rpc.dynamic_summary_pb2_grpc.WorkerServiceStub",
                return_value=mock_stub,
            ):
                result = grpc_health_check("127.0.0.1:50051", timeout=1.0)

        assert result is True

    def test_returns_true_when_not_found_error(self) -> None:
        """When gRPC returns NOT_FOUND, server is healthy (responding)."""
        mock_channel = MagicMock()
        mock_stub = MagicMock()

        # Create a mock RpcError with NOT_FOUND status
        error = make_rpc_error(grpc.StatusCode.NOT_FOUND)
        mock_stub.GetDocument = MagicMock(side_effect=error)

        with patch("grpc.insecure_channel", return_value=mock_channel):
            with patch(
                "ragzoom.rpc.dynamic_summary_pb2_grpc.WorkerServiceStub",
                return_value=mock_stub,
            ):
                result = grpc_health_check("127.0.0.1:50051", timeout=1.0)

        assert result is True

    def test_returns_false_when_unavailable(self) -> None:
        """When gRPC returns UNAVAILABLE, returns False."""
        mock_channel = MagicMock()
        mock_stub = MagicMock()

        error = make_rpc_error(grpc.StatusCode.UNAVAILABLE)
        mock_stub.GetDocument = MagicMock(side_effect=error)

        with patch("grpc.insecure_channel", return_value=mock_channel):
            with patch(
                "ragzoom.rpc.dynamic_summary_pb2_grpc.WorkerServiceStub",
                return_value=mock_stub,
            ):
                result = grpc_health_check("127.0.0.1:50051", timeout=1.0)

        assert result is False

    def test_returns_false_when_deadline_exceeded(self) -> None:
        """When gRPC call times out, returns False."""
        mock_channel = MagicMock()
        mock_stub = MagicMock()

        error = make_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)
        mock_stub.GetDocument = MagicMock(side_effect=error)

        with patch("grpc.insecure_channel", return_value=mock_channel):
            with patch(
                "ragzoom.rpc.dynamic_summary_pb2_grpc.WorkerServiceStub",
                return_value=mock_stub,
            ):
                result = grpc_health_check("127.0.0.1:50051", timeout=0.1)

        assert result is False

    def test_returns_false_on_generic_exception(self) -> None:
        """When any exception occurs, returns False instead of raising."""
        mock_channel = MagicMock()
        mock_stub = MagicMock()
        mock_stub.GetDocument = MagicMock(side_effect=Exception("Generic error"))

        with patch("grpc.insecure_channel", return_value=mock_channel):
            with patch(
                "ragzoom.rpc.dynamic_summary_pb2_grpc.WorkerServiceStub",
                return_value=mock_stub,
            ):
                result = grpc_health_check("127.0.0.1:50051", timeout=1.0)

        assert result is False


class TestIsServerHealthy:
    """Tests for is_server_healthy() function."""

    def test_returns_false_when_no_pid_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When PID file doesn't exist, returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            result = is_server_healthy()
            assert result is False

    def test_returns_false_when_stale_pid(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When PID file exists but process is not running, returns False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # Write a stale PID (use very high PID unlikely to exist)
            pid_file = state_dir / "daemon.pid"
            pid_file.write_text("999999999\n")

            result = is_server_healthy()
            assert result is False

    def test_returns_false_when_process_running_but_grpc_unresponsive(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When process is running but gRPC doesn't respond, returns False."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # Write current process PID (definitely running)
            pid_file = state_dir / "daemon.pid"
            pid_file.write_text(f"{os.getpid()}\n")

            # Write a port file
            port_file = state_dir / "daemon.port"
            port_file.write_text("50099\n")

            # Mock grpc_health_check to return False
            with patch("ragzoom.daemon.grpc_health_check", return_value=False):
                result = is_server_healthy()

            assert result is False

    def test_returns_true_when_process_running_and_grpc_healthy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When process is running AND gRPC responds, returns True."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # Write current process PID (definitely running)
            pid_file = state_dir / "daemon.pid"
            pid_file.write_text(f"{os.getpid()}\n")

            # Write a port file
            port_file = state_dir / "daemon.port"
            port_file.write_text("50051\n")

            # Mock grpc_health_check to return True
            with patch("ragzoom.daemon.grpc_health_check", return_value=True):
                result = is_server_healthy()

            assert result is True

    def test_returns_false_when_no_port_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When process is running but no port file, returns False."""
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # Write current process PID (definitely running)
            pid_file = state_dir / "daemon.pid"
            pid_file.write_text(f"{os.getpid()}\n")

            # Don't write port file

            result = is_server_healthy()
            assert result is False


class TestCrashRecovery:
    """Tests for crash recovery functions."""

    def test_cleanup_stale_state_removes_pid_and_port_files(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cleanup_stale_state() removes PID and port files."""
        from ragzoom.daemon import cleanup_stale_state

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # Create state files
            pid_file = state_dir / "daemon.pid"
            port_file = state_dir / "daemon.port"
            pid_file.write_text("12345\n")
            port_file.write_text("50051\n")

            cleanup_stale_state()

            assert not pid_file.exists()
            assert not port_file.exists()

    def test_cleanup_stale_state_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cleanup_stale_state() doesn't raise when files don't exist."""
        from ragzoom.daemon import cleanup_stale_state

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # No files exist - should not raise
            cleanup_stale_state()

    def test_kill_stale_process_sends_sigterm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kill_stale_process() sends SIGTERM to the process."""
        from ragzoom.daemon import kill_stale_process

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # Write a PID file with a mock PID
            pid_file = state_dir / "daemon.pid"
            pid_file.write_text("12345\n")

            # Mock os.kill to verify SIGTERM is sent
            with patch("os.kill") as mock_kill:
                # Make is_pid_stale return False (process exists)
                with patch("ragzoom.daemon.is_pid_stale", return_value=False):
                    kill_stale_process()
                    mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_kill_stale_process_no_op_when_no_pid_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kill_stale_process() does nothing when no PID file exists."""
        from ragzoom.daemon import kill_stale_process

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # No PID file exists - should not raise
            with patch("os.kill") as mock_kill:
                kill_stale_process()
                mock_kill.assert_not_called()

    def test_kill_stale_process_no_op_when_process_already_gone(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kill_stale_process() does nothing when process is already dead."""
        from ragzoom.daemon import kill_stale_process

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            # Write a PID file
            pid_file = state_dir / "daemon.pid"
            pid_file.write_text("12345\n")

            # Process is already gone (stale PID)
            with patch("os.kill") as mock_kill:
                with patch("ragzoom.daemon.is_pid_stale", return_value=True):
                    kill_stale_process()
                    mock_kill.assert_not_called()

    def test_kill_stale_process_handles_eperm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kill_stale_process() handles EPERM gracefully (process owned by root)."""
        import errno

        from ragzoom.daemon import kill_stale_process

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            pid_file = state_dir / "daemon.pid"
            pid_file.write_text("1\n")  # PID 1 (init) we don't own

            with patch("os.kill") as mock_kill:
                mock_kill.side_effect = OSError(errno.EPERM, "Operation not permitted")
                with patch("ragzoom.daemon.is_pid_stale", return_value=False):
                    # Should not raise
                    kill_stale_process()

    def test_kill_stale_process_handles_esrch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """kill_stale_process() handles ESRCH gracefully (process disappeared)."""
        import errno

        from ragzoom.daemon import kill_stale_process

        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)
            monkeypatch.setenv("RAGZOOM_STATE_DIR", str(state_dir))
            state_dir.mkdir(parents=True, exist_ok=True)

            pid_file = state_dir / "daemon.pid"
            pid_file.write_text("12345\n")

            with patch("os.kill") as mock_kill:
                # Process existed when is_pid_stale checked, but died before kill
                mock_kill.side_effect = OSError(errno.ESRCH, "No such process")
                with patch("ragzoom.daemon.is_pid_stale", return_value=False):
                    # Should not raise
                    kill_stale_process()
