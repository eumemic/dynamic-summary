"""Cross-process document lock tests for SQLite backend.

Verifies that two separate processes cannot acquire the same document lock
simultaneously when using the SQLiteStorageBackend.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_cross_process_document_lock(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """Verify cross-process lock using push-style signaling instead of sleeps.

    Process 1 acquires the lock, signals readiness, then waits for a release
    signal from the parent. Process 2 immediately attempts to acquire the same
    lock and must fail. Parent then signals release and waits for P1 to exit.
    """
    # Use a temp data dir and python vector backend (no chroma dependency)
    monkeypatch.setenv("RAGZOOM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAGZOOM_VECTOR_BACKEND", "python")

    # Ensure data dir exists for sqlite and create sync paths
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    acquired_path = tmp_path / "acquired.signal"
    release_path = tmp_path / "release.signal"
    monkeypatch.setenv("RZ_LOCK_ACQUIRED", str(acquired_path))
    monkeypatch.setenv("RZ_LOCK_RELEASE", str(release_path))

    # Process 1: acquire lock, signal, then wait for release signal
    code1 = """
import os, time, pathlib
import os as _os
base = os.environ['RAGZOOM_DATA_DIR']
sig_acq = pathlib.Path(os.environ['RZ_LOCK_ACQUIRED'])
sig_rel = pathlib.Path(os.environ['RZ_LOCK_RELEASE'])
# Compute backend-equivalent lock path: <base>/data/.ragzoom/locks/<doc>.lock
lock_path = pathlib.Path(base) / 'data' / '.ragzoom' / 'locks' / 'doc-lock-test.lock'
lock_path.parent.mkdir(parents=True, exist_ok=True)
fd = None
try:
    fd = _os.open(lock_path.as_posix(), _os.O_CREAT | _os.O_EXCL | _os.O_WRONLY)
    _os.write(fd, f"pid={_os.getpid()}\\n".encode('utf-8'))
    # Signal to parent that lock is acquired
    sig_acq.write_text('1')
    # Wait until parent signals release
    while not sig_rel.exists():
        time.sleep(0.001)
finally:
    try:
        if fd is not None:
            _os.close(fd)
    finally:
        try:
            _os.unlink(lock_path.as_posix())
        except FileNotFoundError:
            pass
        """

    p1 = subprocess.Popen(_py(code1), env=dict(os.environ))
    try:
        # Wait until P1 signals lock acquisition (up to ~2s)
        deadline = time.time() + 2.0
        while not acquired_path.exists() and time.time() < deadline:
            time.sleep(0.005)
        assert acquired_path.exists(), "P1 did not acquire lock in time"

        # Attempt to acquire the same lock in the parent process via backend; expect failure
        from ragzoom.backends.sqlite_backend import (
            SQLiteStorageBackend,  # local import to keep child fast
        )

        url = f"sqlite:///{tmp_path}/data/sqlite.db"
        b = SQLiteStorageBackend(url)
        failed = False
        try:
            with b.lock_document("doc-lock-test"):
                pass
        except Exception:
            failed = True
        assert (
            failed
        ), "Expected lock acquisition to fail in parent while child holds lock"

        # Signal P1 to release and exit
        release_path.write_text("1")
    finally:
        p1.wait(timeout=10)
