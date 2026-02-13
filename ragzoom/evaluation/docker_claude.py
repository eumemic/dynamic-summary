"""Docker container management for bare Claude CLI environments.

Runs the native Claude CLI binary in containers with no ~/.claude/
directory, eliminating cold-start overhead from plugins, MCP servers,
and skills.  The SDK communicates with the containerised CLI via a
wrapper script that uses ``docker exec -i`` for stdin/stdout passthrough.

For concurrent benchmarks, ``DockerClaudePool`` spins up N isolated
containers (one per concurrency slot) so SDK instances don't contend.
"""

from __future__ import annotations

import logging
import stat
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_IMAGE_NAME = "ragzoom-claude-cli"
_DEFAULT_CONTAINER_NAME = "ragzoom-claude-cli"
_DOCKERFILE_DIR = Path(__file__).resolve().parents[2] / "docker" / "claude-cli"
_WRAPPER_SCRIPT = _DOCKERFILE_DIR / "claude-docker-wrapper.sh"


class DockerClaudeContainer:
    """Lifecycle manager for a bare Claude CLI Docker container.

    Mirrors the ``DockerPostgres`` pattern: ensure_image → ensure_running →
    verify_cli → stop.  The SDK calls the wrapper script as ``cli_path``,
    which forwards into this container via ``docker exec -i``.
    """

    def __init__(
        self,
        session_base: Path,
        container_name: str = _DEFAULT_CONTAINER_NAME,
    ) -> None:
        """Initialise with the host path to bind-mount into the container.

        Args:
            session_base: Directory under which SDK session dirs live.
                          Mounted at the same path inside the container so
                          ``XDG_DATA_HOME`` values work without translation.
            container_name: Docker container name (default: ``ragzoom-claude-cli``).
        """
        self._session_base = session_base
        self._session_base.mkdir(parents=True, exist_ok=True)
        self._container_name = container_name

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
            f"name=^/{self._container_name}$",
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
            return self._container_name in result.stdout.strip().split("\n")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

    def ensure_running(self) -> None:
        """Ensure the container is running, creating it if necessary."""
        self.ensure_image()

        if self._container_listed():
            logger.debug("Container %s is already running", self._container_name)
            return

        if self._container_listed(include_stopped=True):
            logger.info("Starting existing container %s", self._container_name)
            subprocess.run(
                ["docker", "start", self._container_name],
                check=True,
                capture_output=True,
                timeout=30,
            )
        else:
            logger.info("Creating container %s", self._container_name)
            session_str = str(self._session_base)
            subprocess.run(
                [
                    "docker",
                    "run",
                    "-d",
                    "--name",
                    self._container_name,
                    "-v",
                    f"{session_str}:{session_str}",
                    _IMAGE_NAME,
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.info("Container %s created and running", self._container_name)

        self.verify_cli()

    def verify_cli(self) -> None:
        """Verify the Claude CLI is reachable inside the container."""
        result = subprocess.run(
            ["docker", "exec", self._container_name, "claude", "--version"],
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
                ["docker", "rm", "-f", self._container_name],
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.info("Container %s removed", self._container_name)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning("Failed to remove container %s", self._container_name)


class DockerClaudePool:
    """Pool of N Docker containers for concurrent Claude SDK invocations.

    Each pool slot gets its own container and trampoline wrapper script,
    eliminating resource contention when multiple SDK instances run in
    parallel.
    """

    def __init__(self, session_base: Path, pool_size: int) -> None:
        self._containers = [
            DockerClaudeContainer(session_base, f"ragzoom-claude-cli-{i}")
            for i in range(pool_size)
        ]
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None

    @property
    def size(self) -> int:
        return len(self._containers)

    def ensure_running(self) -> None:
        """Build the image once, start all containers, generate trampolines."""
        # Build image via the first container (shared image name).
        self._containers[0].ensure_image()

        for container in self._containers:
            container.ensure_running()

        self._tmp_dir = tempfile.TemporaryDirectory(prefix="ragzoom-pool-")
        tmp_path = Path(self._tmp_dir.name)
        wrapper_abs = str(_WRAPPER_SCRIPT.resolve())

        for i, container in enumerate(self._containers):
            trampoline = tmp_path / f"claude-slot-{i}.sh"
            trampoline.write_text(
                f"#!/usr/bin/env bash\n"
                f'export RAGZOOM_CLAUDE_CONTAINER="{container._container_name}"\n'
                f'exec "{wrapper_abs}" "$@"\n'
            )
            trampoline.chmod(trampoline.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

        logger.info("Docker pool: %d containers running", len(self._containers))

    def cli_path(self, slot: int) -> Path:
        """Return the trampoline wrapper path for a given pool slot."""
        if self._tmp_dir is None:
            raise RuntimeError("Pool not started — call ensure_running() first")
        return Path(self._tmp_dir.name) / f"claude-slot-{slot}.sh"

    def stop(self) -> None:
        """Stop all containers and clean up temp dir."""
        for container in self._containers:
            container.stop()
        if self._tmp_dir is not None:
            self._tmp_dir.cleanup()
            self._tmp_dir = None
        logger.info("Docker pool stopped")
