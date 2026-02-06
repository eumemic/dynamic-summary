"""Isolated RagZoom server lifecycle manager for benchmarks."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from ragzoom.daemon import grpc_health_check

logger = logging.getLogger(__name__)

BENCHMARK_PORT = 50053
HEALTH_CHECK_INTERVAL = 0.5


@dataclass
class BenchmarkServerConfig:
    """Configuration for an isolated benchmark server."""

    port: int = BENCHMARK_PORT
    state_dir: Path | None = field(default=None)
    startup_timeout: float = 60.0


class BenchmarkServerError(Exception):
    """Raised when the benchmark server fails to start or stop."""


class BenchmarkServerManager:
    """Async context manager that runs an isolated RagZoom server.

    Spawns a development-mode server on a dedicated port with a temporary
    state directory, ensuring complete isolation from dev and production
    servers.
    """

    def __init__(self, config: BenchmarkServerConfig | None = None) -> None:
        self._config = config or BenchmarkServerConfig()
        self._process: subprocess.Popen[bytes] | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._state_dir: Path | None = None

    @property
    def address(self) -> str:
        """Server address in host:port format."""
        return f"127.0.0.1:{self._config.port}"

    async def __aenter__(self) -> BenchmarkServerManager:
        self._start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self._stop()

    def _start(self) -> None:
        """Start the isolated server subprocess and wait for it to be healthy."""
        # Set up state directory
        if self._config.state_dir is not None:
            self._state_dir = self._config.state_dir
            self._state_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._temp_dir = tempfile.TemporaryDirectory(prefix="ragzoom-bench-")
            self._state_dir = Path(self._temp_dir.name)

        logger.info(
            "Starting benchmark server on port %d (state: %s)",
            self._config.port,
            self._state_dir,
        )

        env = os.environ.copy()
        env["RAGZOOM_STATE_DIR"] = str(self._state_dir)
        env["RAGZOOM_DATA_DIR"] = str(self._state_dir)  # SQLite DB path

        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ragzoom.cli",
                "server",
                "start",
                "--port",
                str(self._config.port),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._wait_for_healthy()
        logger.info("Benchmark server healthy at %s", self.address)

    def _wait_for_healthy(self) -> None:
        """Poll until the server responds to gRPC health checks."""
        assert self._process is not None
        deadline = time.monotonic() + self._config.startup_timeout

        while time.monotonic() < deadline:
            # Check if process died
            if self._process.poll() is not None:
                stdout = self._process.stdout.read() if self._process.stdout else b""
                stderr = self._process.stderr.read() if self._process.stderr else b""
                raise BenchmarkServerError(
                    f"Benchmark server exited with code {self._process.returncode}.\n"
                    f"stdout: {stdout.decode(errors='replace')}\n"
                    f"stderr: {stderr.decode(errors='replace')}"
                )

            if grpc_health_check(self.address, timeout=1.0):
                return

            time.sleep(HEALTH_CHECK_INTERVAL)

        # Timed out — kill and report
        self._kill()
        raise BenchmarkServerError(
            f"Benchmark server did not become healthy within "
            f"{self._config.startup_timeout}s"
        )

    def _stop(self) -> None:
        """Gracefully stop the server and clean up state."""
        if self._process is not None and self._process.poll() is None:
            logger.info("Stopping benchmark server (pid %d)", self._process.pid)
            self._process.send_signal(signal.SIGTERM)
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("Server did not exit gracefully, sending SIGKILL")
                self._kill()
        self._process = None

        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _kill(self) -> None:
        """Force-kill the server process."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=5.0)
