"""Persistent benchmark server lifecycle manager.

The benchmark server lives on port 50053 with state in /tmp/ragzoom-bench-state.
It is started fresh (killing any existing instance) when ingesting, and reused
as-is for --skip-ingest follow-up runs.  It is never cleaned up automatically —
the user can always do more follow-ups.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from ragzoom.daemon import grpc_health_check

logger = logging.getLogger(__name__)

BENCHMARK_PORT = 50053
BENCHMARK_STATE_DIR = Path("/tmp/ragzoom-bench-state")
HEALTH_CHECK_INTERVAL = 0.5
STARTUP_TIMEOUT = 60.0


class BenchmarkServerError(Exception):
    """Raised when the benchmark server fails to start or is not running."""


class BenchmarkServerManager:
    """Manages a persistent, isolated RagZoom server for benchmarks.

    Two modes of operation:

    - ``start_fresh()``: Kill any existing benchmark server, wipe the state
      directory, and start a new server.  Use this when ingesting data.
    - ``verify_running()``: Check that a benchmark server is already healthy.
      Use this with ``--skip-ingest`` to reuse an existing index.

    The server is never stopped automatically — it stays alive for follow-up
    ``--skip-ingest`` runs.
    """

    def __init__(self, port: int = BENCHMARK_PORT) -> None:
        self._port = port
        self._state_dir = BENCHMARK_STATE_DIR

    @property
    def address(self) -> str:
        """Server address in host:port format."""
        return f"127.0.0.1:{self._port}"

    def start_fresh(self) -> str:
        """Kill existing server, wipe state, start a new server.

        Returns the server address (host:port).
        """
        self._kill_existing()
        self._clean_state()
        self._start()
        return self.address

    def verify_running(self) -> str:
        """Verify that a benchmark server is already healthy.

        Returns the server address (host:port).

        Raises ``BenchmarkServerError`` if no server is running.
        """
        if not grpc_health_check(self.address, timeout=2.0):
            raise BenchmarkServerError(
                f"No benchmark server running on {self.address}. "
                "Run without --skip-ingest first to start one."
            )
        logger.info("Benchmark server healthy at %s", self.address)
        return self.address

    def _kill_existing(self) -> None:
        """Kill any process listening on the benchmark port."""
        result = subprocess.run(
            ["lsof", "-ti", f":{self._port}"],
            capture_output=True,
            text=True,
        )
        pids = result.stdout.strip()
        if pids:
            for pid in pids.splitlines():
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info("Killed existing benchmark server (pid %s)", pid)
                except (ProcessLookupError, ValueError):
                    pass
            # Brief pause for port release
            time.sleep(0.5)

    def _clean_state(self) -> None:
        """Remove and recreate the state directory."""
        if self._state_dir.exists():
            shutil.rmtree(self._state_dir)
            logger.info("Cleaned state dir: %s", self._state_dir)
        self._state_dir.mkdir(parents=True)

    def _start(self) -> None:
        """Start the server subprocess and wait for it to be healthy."""
        logger.info(
            "Starting benchmark server on port %d (state: %s)",
            self._port,
            self._state_dir,
        )

        env = os.environ.copy()
        env["RAGZOOM_STATE_DIR"] = str(self._state_dir)
        env["RAGZOOM_DATA_DIR"] = str(self._state_dir)

        log_path = self._state_dir / "server.log"
        log_file: io.TextIOWrapper = log_path.open("w")

        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ragzoom.cli",
                "server",
                "start",
                "--port",
                str(self._port),
            ],
            env=env,
            stdout=log_file,
            stderr=log_file,
        )

        self._wait_for_healthy(process)
        logger.info(
            "Benchmark server healthy at %s (pid %d)", self.address, process.pid
        )

    def _wait_for_healthy(self, process: subprocess.Popen[bytes]) -> None:
        """Poll until the server responds to gRPC health checks."""
        deadline = time.monotonic() + STARTUP_TIMEOUT

        while time.monotonic() < deadline:
            if process.poll() is not None:
                log_path = self._state_dir / "server.log"
                log_tail = log_path.read_text()[-2000:] if log_path.exists() else ""
                raise BenchmarkServerError(
                    f"Benchmark server exited with code {process.returncode}.\n"
                    f"server.log (last 2000 chars):\n{log_tail}"
                )

            if grpc_health_check(self.address, timeout=1.0):
                return

            time.sleep(HEALTH_CHECK_INTERVAL)

        process.kill()
        process.wait(timeout=5.0)
        raise BenchmarkServerError(
            f"Benchmark server did not become healthy within {STARTUP_TIMEOUT}s"
        )
