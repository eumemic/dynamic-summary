"""Daemon lifecycle management for RagZoom gRPC server.

Provides XDG-compliant state directory management, PID file handling,
health checks, and auto-start capabilities.
"""

import os
from pathlib import Path


def get_daemon_state_dir() -> Path:
    """Get the daemon state directory path.

    Uses XDG Base Directory Specification:
    - Default: ~/.local/state/ragzoom/
    - Override: RAGZOOM_STATE_DIR environment variable

    Returns:
        Absolute path to the state directory (may not exist yet).
    """
    env_override = os.environ.get("RAGZOOM_STATE_DIR")
    if env_override:
        # Expand ~ and convert to absolute path
        path = Path(env_override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    # XDG default: ~/.local/state/ragzoom/
    return Path("~/.local/state/ragzoom").expanduser()


def ensure_daemon_state_dir() -> Path:
    """Ensure the daemon state directory exists.

    Creates the directory and any parent directories if needed.

    Returns:
        Absolute path to the state directory (guaranteed to exist).
    """
    state_dir = get_daemon_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir
