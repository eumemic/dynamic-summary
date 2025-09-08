"""Cross-process document lock tests for SQLite backend.

Verifies that two separate processes cannot acquire the same document lock
simultaneously when using the SQLiteStorageBackend.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_cross_process_document_lock(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Use a temp data dir and python vector backend (no chroma dependency)
    monkeypatch.setenv("RAGZOOM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAGZOOM_VECTOR_BACKEND", "python")

    # Ensure data dir exists for sqlite
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    # Process 1: acquire lock and hold for 2 seconds
    code1 = textwrap.dedent(
        """
        import os, time
        from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
        base = os.environ['RAGZOOM_DATA_DIR']
        url = f"sqlite:///{base}/data/sqlite.db"
        b = SQLiteStorageBackend(url)
        with b.lock_document('doc-lock-test'):
            time.sleep(2.0)
        """
    )

    p1 = subprocess.Popen(_py(code1), env=dict(os.environ))
    try:
        # Give process 1 a moment to acquire the lock
        time.sleep(0.4)

        # Process 2: attempt to acquire the same lock; expect failure
        code2 = textwrap.dedent(
            """
            import os, sys
            from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
            base = os.environ['RAGZOOM_DATA_DIR']
            url = f"sqlite:///{base}/data/sqlite.db"
            b = SQLiteStorageBackend(url)
            try:
                with b.lock_document('doc-lock-test'):
                    pass
                # Acquired lock unexpectedly
                print('ACQUIRED', flush=True)
                sys.exit(0)
            except Exception as e:
                print('LOCKED', flush=True)
                sys.exit(2)
            """
        )
        p2 = subprocess.run(
            _py(code2), env=dict(os.environ), capture_output=True, text=True
        )

        # We expect the second process to fail to acquire lock and print LOCKED
        assert (
            p2.returncode != 0
        ), f"unexpected returncode {p2.returncode}, stdout={p2.stdout}, stderr={p2.stderr}"
        assert "LOCKED" in p2.stdout, f"expected LOCKED in stdout, got: {p2.stdout}"
    finally:
        p1.wait(timeout=10)
