"""Daemon lifecycle management for RagZoom gRPC server.

Provides XDG-compliant state directory management, PID file handling,
health checks, and auto-start capabilities.
"""

import errno
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from types import FrameType

# Type alias for daemon config values (primitives only - JSON-serializable)
ConfigValue = str | int | float | bool | None

PID_FILENAME = "daemon.pid"
PORT_FILENAME = "daemon.port"
LOG_FILENAME = "daemon.log"
CONFIG_FILENAME = "daemon.config.json"

# Config directory (shared across dev/prod)
PRODUCTION_CONFIG_DIR = "~/.config/ragzoom"

# Dev/Prod separation
PRODUCTION_STATE_DIR = "~/.local/state/ragzoom"
DEV_STATE_DIR = "~/.local/state/ragzoom-dev"
PRODUCTION_PORT = 50051
DEV_PORT = 50052


def _is_dev_invocation() -> bool:
    """Detect if invoked via 'python -m ragzoom.cli' vs 'ragzoom' entry point.

    Returns True for module invocation (development), False for entry point (production).
    This enables automatic dev/prod separation without explicit flags.
    """
    argv0 = sys.argv[0] if sys.argv else ""
    # Module invocation: argv[0] ends with .py or contains ragzoom/cli.py path
    return (
        argv0.endswith(".py") or "ragzoom/cli.py" in argv0 or "ragzoom\\cli.py" in argv0
    )


def _resolve_path(path_str: str) -> Path:
    """Resolve a path string to an absolute Path.

    Expands ~ and converts relative paths to absolute.

    Args:
        path_str: Path string to resolve.

    Returns:
        Absolute path.
    """
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def get_config_dir() -> Path:
    """XDG config directory for ragzoom settings.

    Returns ~/.config/ragzoom/ (or $XDG_CONFIG_HOME/ragzoom/ if set).
    Does NOT vary by dev/prod - config is shared.
    """
    xdg_config = os.environ.get("XDG_CONFIG_HOME", "~/.config")
    return Path(xdg_config).expanduser() / "ragzoom"


def get_daemon_state_dir() -> Path:
    """Get the daemon state directory path.

    Uses XDG Base Directory Specification with dev/prod separation:
    - Production: ~/.local/state/ragzoom/
    - Development: ~/.local/state/ragzoom-dev/
    - Override: RAGZOOM_STATE_DIR environment variable (always takes precedence)

    Returns:
        Absolute path to the state directory (may not exist yet).
    """
    env_override = os.environ.get("RAGZOOM_STATE_DIR")
    if env_override:
        return _resolve_path(env_override)

    # Use dev or prod state directory based on invocation mode
    if _is_dev_invocation():
        return Path(DEV_STATE_DIR).expanduser()
    return Path(PRODUCTION_STATE_DIR).expanduser()


def ensure_daemon_state_dir() -> Path:
    """Ensure the daemon state directory exists.

    Creates the directory and any parent directories if needed.

    Returns:
        Absolute path to the state directory (guaranteed to exist).
    """
    state_dir = get_daemon_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir


def get_pid_file_path() -> Path:
    """Get the path to the PID file.

    Returns:
        Path to daemon.pid in the state directory.
    """
    return get_daemon_state_dir() / PID_FILENAME


def write_pid_file(pid: int) -> None:
    """Write the daemon PID to the PID file.

    Creates the state directory if it doesn't exist.

    Args:
        pid: Process ID to write.
    """
    state_dir = ensure_daemon_state_dir()
    pid_file = state_dir / PID_FILENAME
    pid_file.write_text(f"{pid}\n")


def read_pid_file() -> int | None:
    """Read the daemon PID from the PID file.

    Returns:
        The PID as an integer, or None if:
        - The file doesn't exist
        - The file contents are not a valid integer
    """
    pid_file = get_pid_file_path()
    if not pid_file.exists():
        return None

    try:
        content = pid_file.read_text().strip()
        return int(content)
    except (ValueError, OSError):
        return None


def remove_pid_file() -> None:
    """Remove the PID file if it exists.

    This is idempotent - does not raise if the file doesn't exist.
    """
    pid_file = get_pid_file_path()
    try:
        pid_file.unlink()
    except FileNotFoundError:
        pass


def is_pid_stale(pid: int) -> bool:
    """Check if a PID refers to a non-existent process.

    Uses the "kill 0" trick: sending signal 0 checks if a process
    exists without actually sending a signal.

    Args:
        pid: Process ID to check.

    Returns:
        True if the process does NOT exist (stale PID).
        False if the process exists.
    """
    try:
        os.kill(pid, 0)
        return False  # Process exists
    except OSError as e:
        if e.errno == errno.ESRCH:
            # ESRCH = No such process
            return True
        if e.errno == errno.EPERM:
            # EPERM = Process exists but we don't have permission to signal it
            return False
        # Other errors - assume stale to be safe
        return True


def get_port_file_path() -> Path:
    """Get the path to the port file.

    Returns:
        Path to daemon.port in the state directory.
    """
    return get_daemon_state_dir() / PORT_FILENAME


def write_port_file(port: int) -> None:
    """Write the daemon port to the port file.

    Creates the state directory if it doesn't exist.

    Args:
        port: Port number to write.
    """
    state_dir = ensure_daemon_state_dir()
    port_file = state_dir / PORT_FILENAME
    port_file.write_text(f"{port}\n")


def read_port_file() -> int | None:
    """Read the daemon port from the port file.

    Returns:
        The port as an integer, or None if:
        - The file doesn't exist
        - The file contents are not a valid integer
    """
    port_file = get_port_file_path()
    if not port_file.exists():
        return None

    try:
        content = port_file.read_text().strip()
        return int(content)
    except (ValueError, OSError):
        return None


def remove_port_file() -> None:
    """Remove the port file if it exists.

    This is idempotent - does not raise if the file doesn't exist.
    """
    port_file = get_port_file_path()
    try:
        port_file.unlink()
    except FileNotFoundError:
        pass


def get_log_file_path() -> Path:
    """Get the default path to the daemon log file.

    Returns:
        Path to daemon.log in the state directory.
    """
    return get_daemon_state_dir() / LOG_FILENAME


def get_config_file_path() -> Path:
    """Get the path to the daemon config file.

    Uses RAGZOOM_DAEMON_CONFIG environment variable if set,
    otherwise uses daemon.config.json in the state directory.

    Returns:
        Path to daemon.config.json or custom location.
    """
    env_override = os.environ.get("RAGZOOM_DAEMON_CONFIG")
    if env_override:
        return _resolve_path(env_override)

    return get_daemon_state_dir() / CONFIG_FILENAME


def write_config_file(config: dict[str, ConfigValue]) -> None:
    """Write daemon configuration to the config file.

    Creates the parent directory if it doesn't exist.
    Only persists fields that affect daemon behavior.

    Args:
        config: Configuration dict with daemon settings.
               Supported keys:
               - target_chunk_tokens: int | None
               - summarization_guidance: str | None
               - database_url: str | None
    """
    # Only persist daemon-relevant fields
    persistent_fields = [
        "target_chunk_tokens",
        "summarization_guidance",
        "database_url",
    ]
    filtered_config = {k: v for k, v in config.items() if k in persistent_fields}

    # Skip if nothing to persist
    if not filtered_config:
        return

    config_file = get_config_file_path()
    # Ensure parent directory exists (handles both default and custom paths)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(filtered_config, indent=2) + "\n")
    # Restrict permissions since config may contain database credentials
    config_file.chmod(0o600)


def read_config_file() -> dict[str, ConfigValue] | None:
    """Read daemon configuration from the config file.

    Returns:
        Configuration dict, or None if:
        - The file doesn't exist
        - The file contents are not valid JSON
    """
    config_file = get_config_file_path()
    if not config_file.exists():
        return None

    try:
        content = config_file.read_text()
        data = json.loads(content)
        if not isinstance(data, dict):
            return None
        return data
    except (json.JSONDecodeError, OSError):
        return None


def remove_config_file() -> None:
    """Remove the config file if it exists.

    This is idempotent - does not raise if the file doesn't exist.
    """
    config_file = get_config_file_path()
    try:
        config_file.unlink()
    except FileNotFoundError:
        pass


def daemonize(log_file: Path | None = None, ready_fd: int | None = None) -> None:
    """Fork the current process into a background daemon.

    Implements the standard Unix double-fork daemonization pattern:
    1. First fork: Parent exits, child continues
    2. setsid(): Child becomes session leader
    3. Second fork: Session leader exits, grandchild continues
       (prevents acquiring controlling terminal)
    4. Redirect stdin to /dev/null, stdout/stderr to log file
    5. Write daemon PID to state file
    6. Signal readiness via ready_fd (if provided)

    Args:
        log_file: Path to redirect stdout/stderr. If None, uses default
                  location in state directory. Parent directories will
                  be created if they don't exist.
        ready_fd: If provided, write b"R" to this file descriptor after
                  daemonization completes, then close it. Used for
                  synchronization in tests to detect daemon readiness
                  without sleep-based polling.

    Note:
        This function only returns in the final daemon process.
        Parent processes call sys.exit(0).
    """

    if log_file is None:
        log_file = get_log_file_path()

    # Flush buffered output before forking to prevent duplication
    sys.stdout.flush()
    sys.stderr.flush()

    # First fork - parent exits, child continues
    pid = os.fork()
    if pid > 0:
        # Parent - exit cleanly
        sys.exit(0)

    # Child - become session leader
    os.setsid()

    # Second fork - prevents acquiring controlling terminal
    pid = os.fork()
    if pid > 0:
        # Intermediate process - exit
        sys.exit(0)

    # Grandchild (daemon) continues from here

    # Ensure log file directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Redirect stdin to /dev/null
    devnull_fd = os.open("/dev/null", os.O_RDONLY)
    os.dup2(devnull_fd, sys.stdin.fileno())
    os.close(devnull_fd)

    # Redirect stdout and stderr to log file
    log_fd = os.open(str(log_file), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)

    # Write daemon PID to state file
    write_pid_file(os.getpid())

    # Signal readiness to parent process if ready_fd was provided
    if ready_fd is not None:
        os.write(ready_fd, b"R")
        os.close(ready_fd)


def install_shutdown_handlers(
    cleanup_callback: Callable[[], None] | None = None,
) -> None:
    """Install signal handlers for graceful daemon shutdown.

    Handles SIGTERM and SIGINT by:
    1. Calling the optional cleanup callback
    2. Removing state files (PID and port files)
    3. Exiting with status 0

    Args:
        cleanup_callback: Optional function to call before cleanup.
                         Use this to stop servers, finish in-flight work, etc.
    """

    def shutdown_handler(signum: int, frame: FrameType | None) -> None:
        """Handle shutdown signals."""
        # Call cleanup callback if provided
        if cleanup_callback is not None:
            cleanup_callback()

        # Remove state files
        remove_pid_file()
        remove_port_file()

        # Exit cleanly
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)


def get_server_address() -> str | None:
    """Get the server address from the port file.

    Returns:
        Server address in "host:port" format, or None if:
        - Port file doesn't exist
        - Port file contents are invalid
    """
    port = read_port_file()
    if port is None:
        return None
    return f"127.0.0.1:{port}"


def grpc_health_check(address: str, timeout: float = 2.0) -> bool:
    """Check if a gRPC server is responsive at the given address.

    Performs a lightweight gRPC call to verify the server is accepting
    connections and responding to requests.

    Args:
        address: Server address in "host:port" format.
        timeout: Maximum time to wait for response in seconds.

    Returns:
        True if server responds successfully, False otherwise.
    """
    import grpc

    from ragzoom.rpc import dynamic_summary_pb2 as pb2
    from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc

    try:
        # Create a channel and make a lightweight call
        channel = grpc.insecure_channel(address)
        stub = pb2_grpc.WorkerServiceStub(channel)

        # Use GetDocument with empty ID - fast, read-only, minimal side effects
        request = pb2.GetDocumentRequest(document_id="")
        # The call will fail with NOT_FOUND but that proves the server is alive
        try:
            stub.GetDocument(request, timeout=timeout)
        except grpc.RpcError as rpc_error:
            # NOT_FOUND or INVALID_ARGUMENT means server is responding - healthy!
            # (Empty document_id triggers INVALID_ARGUMENT validation)
            code = rpc_error.code()
            if code in (grpc.StatusCode.NOT_FOUND, grpc.StatusCode.INVALID_ARGUMENT):
                return True
            # UNAVAILABLE or DEADLINE_EXCEEDED means server is not healthy
            return False
        finally:
            channel.close()

        # If we got here without exception, server is healthy
        return True
    except Exception:
        # Any other exception means unhealthy
        return False


def is_server_healthy() -> bool:
    """Check if the daemon is running and responsive.

    Performs a two-phase health check:
    1. Verify PID file exists and process is running
    2. Verify gRPC server responds to requests

    Both checks must pass for the server to be considered healthy.

    Returns:
        True only if process is running AND gRPC responds.
    """
    # Phase 1: Check PID file
    pid = read_pid_file()
    if pid is None:
        return False

    # Phase 2: Check if process is still running
    if is_pid_stale(pid):
        return False

    # Phase 3: Check if we have a port file
    address = get_server_address()
    if address is None:
        return False

    # Phase 4: Check if gRPC is responsive
    return grpc_health_check(address)


def cleanup_stale_state() -> None:
    """Remove stale state files (PID and port files).

    This is safe to call even if files don't exist - it's idempotent.
    Use this after killing a stale process or when recovering from a crash.
    """
    remove_pid_file()
    remove_port_file()


def kill_stale_process() -> None:
    """Kill the stale daemon process if one exists.

    Reads the PID from the PID file and sends SIGTERM to the process.
    This is a no-op if:
    - No PID file exists
    - The PID file contains an invalid value
    - The process is already dead (stale PID)

    Handles permission errors gracefully (e.g., if the process is
    owned by root or another user).
    """
    pid = read_pid_file()
    if pid is None:
        return

    # Check if process is already dead
    if is_pid_stale(pid):
        return

    # Try to kill the process
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        # ESRCH: Process died between our check and the kill
        # EPERM: Process owned by someone else (shouldn't happen for our daemon)
        # Both are acceptable outcomes - the process is gone or we can't touch it
        if e.errno in (errno.ESRCH, errno.EPERM):
            return
        raise


class DaemonStartError(Exception):
    """Raised when the daemon fails to start or become healthy."""


DEFAULT_STARTUP_TIMEOUT = 30.0
HEALTH_CHECK_INTERVAL = 0.2


def start_daemon(port: int = PRODUCTION_PORT, config_path: Path | None = None) -> None:
    """Start the daemon as a background subprocess.

    Spawns a new process running `ragzoom server start --daemon`.
    This function returns immediately after spawning; it does NOT
    wait for the server to become healthy.

    IMPORTANT: Uses the `ragzoom` console script (not `python -m ragzoom.cli`)
    to ensure the daemon runs in production mode and writes state files to
    the production state directory (~/.local/state/ragzoom/).

    Args:
        port: Port number for the daemon to listen on.
        config_path: Optional path to config file. If provided, the daemon
                     will be started with `--config <path>` to use persisted
                     settings (e.g., target_chunk_tokens, summarization_guidance).

    Raises:
        FileNotFoundError: If the `ragzoom` command is not found in PATH.
    """
    # Find the ragzoom console script - must use this (not python -m ragzoom.cli)
    # to ensure daemon runs in production mode with correct state directory
    ragzoom_cmd = shutil.which("ragzoom")
    if ragzoom_cmd is None:
        raise FileNotFoundError(
            "Cannot find 'ragzoom' command in PATH. "
            "Ensure ragzoom is installed: pip install /path/to/dynamic-summary"
        )

    # Build command to start daemon
    cmd = [
        ragzoom_cmd,
        "server",
        "start",
        "--daemon",
        "--port",
        str(port),
    ]

    # Add config path if provided
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])

    # Spawn subprocess and detach - we don't wait for it
    # The daemon itself handles forking and writing PID/port files
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def wait_for_healthy(timeout: float = DEFAULT_STARTUP_TIMEOUT) -> bool:
    """Wait for the daemon to become healthy.

    Polls is_server_healthy() until it returns True or timeout is reached.

    Args:
        timeout: Maximum time to wait in seconds.

    Returns:
        True if server became healthy, False if timeout.
    """

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_server_healthy():
            return True
        time.sleep(HEALTH_CHECK_INTERVAL)
    return False


def get_process_uptime(pid: int) -> str:
    """Get human-readable uptime for a process.

    Uses psutil to get the process start time and calculates the
    duration from then to now.

    Args:
        pid: Process ID to check.

    Returns:
        Human-readable uptime string like "2h 15m" or "5m" or "30s".
        Returns "unknown" if the process doesn't exist or uptime can't be determined.
    """
    import psutil

    try:
        process = psutil.Process(pid)
        create_time = process.create_time()
        uptime_seconds = int(time.time() - create_time)

        # Handle clock skew or other anomalies
        if uptime_seconds < 0:
            return "unknown"

        # Less than a minute
        if uptime_seconds < 60:
            return f"{uptime_seconds}s"

        # Less than an hour
        if uptime_seconds < 3600:
            minutes = uptime_seconds // 60
            return f"{minutes}m"

        # An hour or more
        hours = uptime_seconds // 3600
        minutes = (uptime_seconds % 3600) // 60
        if minutes > 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h"
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return "unknown"


def ensure_server_running(timeout: float = DEFAULT_STARTUP_TIMEOUT) -> str:
    """Ensure the daemon is running and return its address.

    This is the main entry point for auto-start functionality:
    1. If server is already healthy, returns its address immediately
    2. Otherwise, cleans up stale state and starts a fresh daemon
    3. Waits for the daemon to become healthy
    4. Returns the server address

    In dev mode (invoked via `python -m ragzoom.cli`), auto-start is disabled
    to avoid confusion. The function will fail fast with a helpful error message
    directing the user to manually start the dev server.

    If a persisted config file exists (from a previous `--config` invocation),
    it will be passed to the daemon to restore settings like target_chunk_tokens
    and summarization_guidance.

    Args:
        timeout: Maximum time to wait for server to become healthy.

    Returns:
        Server address in "host:port" format.

    Raises:
        DaemonStartError: If server fails to start or become healthy within timeout,
                          or if in dev mode and server is not running.
    """
    is_dev = _is_dev_invocation()

    # Fast path: server already running and healthy
    if is_server_healthy():
        address = get_server_address()
        if address is not None:
            return address

    # In dev mode, fail fast instead of auto-starting to avoid confusion
    if is_dev:
        raise DaemonStartError(
            "Dev server is not running. "
            "Start it manually with: python -m ragzoom.cli server start\n"
            f"Dev server uses port {DEV_PORT} and state dir {DEV_STATE_DIR}"
        )

    # Server not running or unhealthy - start fresh
    cleanup_stale_state()

    # Check for persisted config from previous daemon start
    config_path: Path | None = None
    if read_config_file() is not None:
        config_path = get_config_file_path()

    start_daemon(config_path=config_path)

    # Wait for server to become healthy
    if not wait_for_healthy(timeout):
        raise DaemonStartError(
            f"Daemon failed to start: timed out after {timeout}s waiting for healthy state"
        )

    address = get_server_address()
    if address is None:
        raise DaemonStartError("Daemon started but port file not found")

    return address
