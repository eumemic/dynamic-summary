"""Lightweight file-based document lock for file-backed deployments.

This avoids external dependencies and works cross-platform by relying on the
atomicity of O_CREAT|O_EXCL for creating a lock file. It is intended for
short-lived, non-blocking acquisition in CLI workflows.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType


@dataclass
class FileDocumentLock(AbstractContextManager[None]):
    path: Path
    _fd: int | None = None

    def __enter__(self) -> None:  # noqa: D401 - trivial
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Attempt atomic create; fail if exists
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            self._fd = os.open(self.path.as_posix(), flags)
            # Optionally write owner metadata for diagnostics
            info = f"pid={os.getpid()}\n"
            os.write(self._fd, info.encode("utf-8"))
        except FileExistsError as e:
            raise RuntimeError(
                f"Document is currently being modified (lock: {self.path})"
            ) from e
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if self._fd is not None:
                os.close(self._fd)
        finally:
            self._fd = None
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


def document_lock_path(base_dir: Path, document_id: str | None) -> Path:
    """Compute lock file path for a document within base_dir.

    base_dir should be a writable application directory (e.g., data dir).
    """
    doc = document_id or "_global"
    return base_dir.joinpath(".ragzoom", "locks", f"{doc}.lock")
