"""Cross-process document lock tests for SQLite backend."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend

LOCK_HELPER = textwrap.dedent(
    """
    import os
    import pathlib
    import time

    from ragzoom.backends.sqlite_backend import SQLiteStorageBackend

    db_path = pathlib.Path(os.environ["RZ_SQLITE_PATH"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    acquired = pathlib.Path(os.environ["RZ_LOCK_ACQUIRED"])
    release = pathlib.Path(os.environ["RZ_LOCK_RELEASE"])
    behavior = os.environ.get("RZ_LOCK_BEHAVIOR", "wait")

    backend = SQLiteStorageBackend("sqlite:///" + db_path.as_posix())
    with backend.lock_document("doc-lock-test"):
        acquired.write_text("1")
        if behavior == "wait":
            while not release.exists():
                time.sleep(0.001)
        else:
            while True:
                time.sleep(0.1)
    """
)


def _py(code: str) -> tuple[str, ...]:
    return (sys.executable, "-c", code)


def _spawn_lock_holder(
    env: dict[str, str],
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(_py(LOCK_HELPER), env=env)


@pytest.mark.slow_threshold(5.0)
@pytest.mark.xdist_group("doc-lock")
def test_cross_process_document_lock(tmp_path: Path) -> None:
    """Process 1 holds the lock; process 2 must fail until release."""

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = data_dir / "sqlite.db"
    acquired_path = tmp_path / "acquired.signal"
    release_path = tmp_path / "release.signal"

    env = dict(os.environ)
    env.update(
        {
            "RZ_SQLITE_PATH": str(sqlite_path),
            "RZ_LOCK_ACQUIRED": str(acquired_path),
            "RZ_LOCK_RELEASE": str(release_path),
            "RZ_LOCK_BEHAVIOR": "wait",
        }
    )

    holder = _spawn_lock_holder(env)
    try:
        deadline = time.time() + 5.0
        while not acquired_path.exists() and time.time() < deadline:
            time.sleep(0.005)
        assert acquired_path.exists(), "Lock holder did not signal acquisition"

        backend = SQLiteStorageBackend("sqlite:///" + sqlite_path.as_posix())
        with pytest.raises(RuntimeError, match="currently being modified"):
            with backend.lock_document("doc-lock-test"):
                pass

        release_path.write_text("1")
    finally:
        holder.wait(timeout=10)


@pytest.mark.slow_threshold(5.0)
@pytest.mark.xdist_group("doc-lock")
def test_lock_released_after_abrupt_exit(tmp_path: Path) -> None:
    """Lock must be released when the owning process is terminated."""

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = data_dir / "sqlite.db"
    acquired_path = tmp_path / "acquired.signal"
    release_path = tmp_path / "release.signal"

    env = dict(os.environ)
    env.update(
        {
            "RZ_SQLITE_PATH": str(sqlite_path),
            "RZ_LOCK_ACQUIRED": str(acquired_path),
            "RZ_LOCK_RELEASE": str(release_path),
            "RZ_LOCK_BEHAVIOR": "spin",
        }
    )

    holder = _spawn_lock_holder(env)
    try:
        deadline = time.time() + 5.0
        while not acquired_path.exists() and time.time() < deadline:
            time.sleep(0.005)
        assert acquired_path.exists(), "Lock holder did not signal acquisition"

        holder.terminate()
        holder.wait(timeout=10)

        backend = SQLiteStorageBackend("sqlite:///" + sqlite_path.as_posix())
        with backend.lock_document("doc-lock-test"):
            pass
    finally:
        if holder.poll() is None:  # pragma: no cover - defensive cleanup
            holder.kill()
