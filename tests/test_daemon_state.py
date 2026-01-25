"""Tests for daemon state directory management."""

import os
from pathlib import Path
from unittest.mock import patch

from ragzoom.daemon import get_daemon_state_dir


class TestStateDirectoryXdgCompliant:
    """Test XDG-compliant state directory resolution."""

    def test_default_state_directory(self) -> None:
        """Default state directory is ~/.local/state/ragzoom/."""
        # Clear environment variable to test default behavior
        with patch.dict(os.environ, {}, clear=True):
            # Patch expanduser to return a known value
            with patch.object(Path, "expanduser") as mock_expand:
                mock_expand.return_value = Path("/home/testuser/.local/state/ragzoom")
                state_dir = get_daemon_state_dir()
                assert state_dir == Path("/home/testuser/.local/state/ragzoom")

    def test_environment_override(self) -> None:
        """RAGZOOM_STATE_DIR overrides default location."""
        custom_dir = "/custom/state/dir"
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": custom_dir}):
            state_dir = get_daemon_state_dir()
            assert state_dir == Path(custom_dir)

    def test_environment_override_expands_tilde(self) -> None:
        """RAGZOOM_STATE_DIR expands ~ in path."""
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": "~/my-ragzoom-state"}):
            state_dir = get_daemon_state_dir()
            # Should expand ~ to home directory
            assert not str(state_dir).startswith("~")
            assert "my-ragzoom-state" in str(state_dir)

    def test_state_directory_is_absolute(self) -> None:
        """State directory is always an absolute path."""
        with patch.dict(os.environ, {}, clear=True):
            state_dir = get_daemon_state_dir()
            assert state_dir.is_absolute()

    def test_state_directory_with_relative_override(self) -> None:
        """Relative RAGZOOM_STATE_DIR is converted to absolute."""
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": "relative/path"}):
            state_dir = get_daemon_state_dir()
            assert state_dir.is_absolute()
            assert str(state_dir).endswith("relative/path")


class TestStateDirectoryCreation:
    """Test state directory creation behavior."""

    def test_ensure_state_dir_creates_directory(self, tmp_path: Path) -> None:
        """ensure_daemon_state_dir() creates directory if it doesn't exist."""
        from ragzoom.daemon import ensure_daemon_state_dir

        test_dir = tmp_path / "ragzoom-state"
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(test_dir)}):
            result = ensure_daemon_state_dir()
            assert result == test_dir
            assert test_dir.exists()
            assert test_dir.is_dir()

    def test_ensure_state_dir_idempotent(self, tmp_path: Path) -> None:
        """ensure_daemon_state_dir() is idempotent."""
        from ragzoom.daemon import ensure_daemon_state_dir

        test_dir = tmp_path / "ragzoom-state"
        test_dir.mkdir(parents=True)

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(test_dir)}):
            result = ensure_daemon_state_dir()
            assert result == test_dir
            assert test_dir.exists()

    def test_ensure_state_dir_creates_parent_directories(self, tmp_path: Path) -> None:
        """ensure_daemon_state_dir() creates parent directories as needed."""
        from ragzoom.daemon import ensure_daemon_state_dir

        test_dir = tmp_path / "deep" / "nested" / "ragzoom-state"
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(test_dir)}):
            result = ensure_daemon_state_dir()
            assert result == test_dir
            assert test_dir.exists()


class TestPidFileManagement:
    """Test PID file read/write/cleanup operations."""

    def test_write_pid_file(self, tmp_path: Path) -> None:
        """write_pid_file() creates daemon.pid with correct content."""
        from ragzoom.daemon import write_pid_file

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            write_pid_file(12345)
            pid_file = tmp_path / "daemon.pid"
            assert pid_file.exists()
            assert pid_file.read_text().strip() == "12345"

    def test_read_pid_file_returns_pid(self, tmp_path: Path) -> None:
        """read_pid_file() returns PID when file exists."""
        from ragzoom.daemon import read_pid_file

        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("54321\n")

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            result = read_pid_file()
            assert result == 54321

    def test_read_pid_file_returns_none_when_missing(self, tmp_path: Path) -> None:
        """read_pid_file() returns None when file doesn't exist."""
        from ragzoom.daemon import read_pid_file

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            result = read_pid_file()
            assert result is None

    def test_read_pid_file_returns_none_for_invalid_content(
        self, tmp_path: Path
    ) -> None:
        """read_pid_file() returns None for non-integer content."""
        from ragzoom.daemon import read_pid_file

        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("not-a-pid\n")

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            result = read_pid_file()
            assert result is None

    def test_remove_pid_file_deletes_file(self, tmp_path: Path) -> None:
        """remove_pid_file() deletes the PID file."""
        from ragzoom.daemon import remove_pid_file

        pid_file = tmp_path / "daemon.pid"
        pid_file.write_text("12345\n")

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            remove_pid_file()
            assert not pid_file.exists()

    def test_remove_pid_file_idempotent(self, tmp_path: Path) -> None:
        """remove_pid_file() succeeds even if file doesn't exist."""
        from ragzoom.daemon import remove_pid_file

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            # Should not raise
            remove_pid_file()

    def test_is_pid_stale_returns_false_for_running_process(self) -> None:
        """is_pid_stale() returns False for our own PID (known running)."""
        from ragzoom.daemon import is_pid_stale

        # Our own process is definitely running
        result = is_pid_stale(os.getpid())
        assert result is False

    def test_is_pid_stale_returns_true_for_nonexistent_process(self) -> None:
        """is_pid_stale() returns True for a PID that doesn't exist."""
        from ragzoom.daemon import is_pid_stale

        # PID 99999999 is unlikely to exist
        result = is_pid_stale(99999999)
        assert result is True

    def test_write_pid_file_creates_state_directory(self, tmp_path: Path) -> None:
        """write_pid_file() creates state directory if needed."""
        from ragzoom.daemon import write_pid_file

        state_dir = tmp_path / "nested" / "state"
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(state_dir)}):
            write_pid_file(12345)
            assert state_dir.exists()
            assert (state_dir / "daemon.pid").exists()


class TestPortFileManagement:
    """Test port file read/write/cleanup operations."""

    def test_write_port_file(self, tmp_path: Path) -> None:
        """write_port_file() creates daemon.port with correct content."""
        from ragzoom.daemon import write_port_file

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            write_port_file(50051)
            port_file = tmp_path / "daemon.port"
            assert port_file.exists()
            assert port_file.read_text().strip() == "50051"

    def test_read_port_file_returns_port(self, tmp_path: Path) -> None:
        """read_port_file() returns port when file exists."""
        from ragzoom.daemon import read_port_file

        port_file = tmp_path / "daemon.port"
        port_file.write_text("50052\n")

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            result = read_port_file()
            assert result == 50052

    def test_read_port_file_returns_none_when_missing(self, tmp_path: Path) -> None:
        """read_port_file() returns None when file doesn't exist."""
        from ragzoom.daemon import read_port_file

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            result = read_port_file()
            assert result is None

    def test_read_port_file_returns_none_for_invalid_content(
        self, tmp_path: Path
    ) -> None:
        """read_port_file() returns None for non-integer content."""
        from ragzoom.daemon import read_port_file

        port_file = tmp_path / "daemon.port"
        port_file.write_text("not-a-port\n")

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            result = read_port_file()
            assert result is None

    def test_remove_port_file_deletes_file(self, tmp_path: Path) -> None:
        """remove_port_file() deletes the port file."""
        from ragzoom.daemon import remove_port_file

        port_file = tmp_path / "daemon.port"
        port_file.write_text("50051\n")

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            remove_port_file()
            assert not port_file.exists()

    def test_remove_port_file_idempotent(self, tmp_path: Path) -> None:
        """remove_port_file() succeeds even if file doesn't exist."""
        from ragzoom.daemon import remove_port_file

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            # Should not raise
            remove_port_file()

    def test_write_port_file_creates_state_directory(self, tmp_path: Path) -> None:
        """write_port_file() creates state directory if needed."""
        from ragzoom.daemon import write_port_file

        state_dir = tmp_path / "nested" / "state"
        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(state_dir)}):
            write_port_file(50051)
            assert state_dir.exists()
            assert (state_dir / "daemon.port").exists()

    def test_get_port_file_path(self, tmp_path: Path) -> None:
        """get_port_file_path() returns correct path."""
        from ragzoom.daemon import get_port_file_path

        with patch.dict(os.environ, {"RAGZOOM_STATE_DIR": str(tmp_path)}):
            result = get_port_file_path()
            assert result == tmp_path / "daemon.port"
