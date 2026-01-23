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
