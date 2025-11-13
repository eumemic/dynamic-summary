"""Lightweight file-based document lock for file-backed deployments.

Uses OS-level file locks so the operating system releases the lock if a process
crashes or exits unexpectedly. This prevents stale lock files from blocking
future indexing runs while remaining dependency-free and cross-platform.
"""

from __future__ import annotations

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, cast

_OPEN_FLAGS = os.O_CREAT | os.O_RDWR

if os.name == "nt":  # pragma: no cover - exercised in Windows CI
    import msvcrt as _msvcrt

    class _MsvcrtProto(Protocol):
        LK_NBLCK: int
        LK_UNLCK: int
        O_BINARY: int

        def locking(self, fd: int, mode: int, nbytes: int) -> None: ...

    msvcrt = cast("_MsvcrtProto", _msvcrt)

    _LOCK_SIZE = 1

    def _acquire_os_lock(fd: int) -> None:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, _LOCK_SIZE)
        except OSError as exc:  # Raised when the lock is already held
            raise BlockingIOError from exc

    def _release_os_lock(fd: int) -> None:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, _LOCK_SIZE)

    _OPEN_FLAGS |= getattr(os, "O_BINARY", 0)
else:  # pragma: no cover - exercised in Linux/macOS CI
    import fcntl

    def _acquire_os_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release_os_lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


_LOCK_ERROR = "Document is currently being modified (lock: {path})"


@dataclass
class FileDocumentLock(AbstractContextManager[None]):
    path: Path
    _fd: int | None = None
    _locked: bool = False

    def __enter__(self) -> None:  # noqa: D401 - trivial
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path.as_posix(), _OPEN_FLAGS, 0o600)
        self._fd = fd
        try:
            _acquire_os_lock(fd)
            self._locked = True
        except BlockingIOError as exc:
            self._cleanup_fd()
            raise RuntimeError(_LOCK_ERROR.format(path=self.path)) from exc
        except OSError:
            self._cleanup_fd()
            raise

        # Record owner metadata for diagnostics
        os.ftruncate(fd, 0)
        info = f"pid={os.getpid()}\n"
        os.write(fd, info.encode("utf-8"))
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if self._fd is not None and self._locked:
                try:
                    _release_os_lock(self._fd)
                finally:
                    self._locked = False
        finally:
            self._cleanup_fd()
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass

    def _cleanup_fd(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            finally:
                self._fd = None
                self._locked = False


def document_lock_path(base_dir: Path, document_id: str | None) -> Path:
    """Compute lock file path for a document within base_dir.

    base_dir should be a writable application directory (e.g., data dir).
    """
    doc = document_id or "_global"
    return base_dir.joinpath(".ragzoom", "locks", f"{doc}.lock")
