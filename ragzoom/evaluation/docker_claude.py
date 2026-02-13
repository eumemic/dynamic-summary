"""Docker container manager for a bare Claude CLI environment.

Runs the native Claude CLI binary in a container with no ~/.claude/
directory, eliminating cold-start overhead from plugins, MCP servers,
and skills.  The SDK communicates with the containerised CLI via a
wrapper script that uses ``docker exec -i`` for stdin/stdout passthrough.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_NAME = "ragzoom-claude-cli"
_CONTAINER_NAME = "ragzoom-claude-cli"
_DOCKERFILE_DIR = Path(__file__).resolve().parents[2] / "docker" / "claude-cli"
_WRAPPER_SCRIPT = _DOCKERFILE_DIR / "claude-docker-wrapper.sh"


class DockerClaudeContainer:
    """Lifecycle manager for the bare Claude CLI Docker container.

    Mirrors the ``DockerPostgres`` pattern: ensure_image → ensure_running →
    verify_cli → stop.  The SDK calls the wrapper script as ``cli_path``,
    which forwards into this container via ``docker exec -i``.
    """

    def __init__(self, session_base: Path) -> None:
        """Initialise with the host path to bind-mount into the container.

        Args:
            session_base: Directory under which SDK session dirs live.
                          Mounted at the same path inside the container so
                          ``XDG_DATA_HOME`` values work without translation.
        """
        self._session_base = session_base
        self._session_base.mkdir(parents=True, exist_ok=True)

    @property
    def wrapper_path(self) -> Path:
        """Path to the wrapper script the SDK should use as ``cli_path``."""
        return _WRAPPER_SCRIPT

    # ------------------------------------------------------------------
    # Image management
    # ------------------------------------------------------------------

    def _image_exists(self) -> bool:
        try:
            result = subprocess.run(
                ["docker", "images", "-q", _IMAGE_NAME],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return bool(result.stdout.strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def ensure_image(self) -> None:
        """Build the Docker image if it doesn't already exist."""
        if self._image_exists():
            logger.debug("Docker image %s already exists", _IMAGE_NAME)
            return
        logger.info("Building Docker image %s …", _IMAGE_NAME)
        subprocess.run(
            ["docker", "build", "-t", _IMAGE_NAME, str(_DOCKERFILE_DIR)],
            check=True,
            timeout=300,
        )
        logger.info("Docker image %s built successfully", _IMAGE_NAME)

    # ------------------------------------------------------------------
    # Container management
    # ------------------------------------------------------------------

    def _container_listed(self, *, include_stopped: bool = False) -> bool:
        """Check if the container appears in ``docker ps``.

        Args:
            include_stopped: If True, also list stopped containers (``-a``).
        """
        cmd = [
            "docker",
            "ps",
            *(("-a",) if include_stopped else ()),
            "--filter",
            f"name=^/{_CONTAINER_NAME}$",
            "--format",
            "{{.Names}}",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return _CONTAINER_NAME in result.stdout.strip().split("\n")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def ensure_running(self) -> None:
        """Ensure the container is running, creating it if necessary."""
        self.ensure_image()

        if self._container_listed():
            logger.debug("Container %s is already running", _CONTAINER_NAME)
            return

        if self._container_listed(include_stopped=True):
            logger.info("Starting existing container %s", _CONTAINER_NAME)
            subprocess.run(
                ["docker", "start", _CONTAINER_NAME],
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            logger.info("Creating container %s", _CONTAINER_NAME)
            session_str = str(self._session_base)
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    _CONTAINER_NAME,
                    "-v",
                    f"{session_str}:{session_str}",
                    _IMAGE_NAME,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.info("Container %s created and running", _CONTAINER_NAME)

        self.verify_cli()

    def verify_cli(self) -> None:
        """Verify the Claude CLI is reachable inside the container."""
        result = subprocess.run(
            ["docker", "exec", _CONTAINER_NAME, "claude", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Claude CLI not working inside container: {result.stderr.strip()}"
            )
        logger.info("Claude CLI verified: %s", result.stdout.strip())

    def stop(self) -> None:
        """Stop and remove the container."""
        try:
            subprocess.run(
                ["docker", "rm", "-f", _CONTAINER_NAME],
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.info("Container %s removed", _CONTAINER_NAME)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning("Failed to remove container %s", _CONTAINER_NAME)
