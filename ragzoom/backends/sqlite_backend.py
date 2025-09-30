"""SQLite StorageBackend with in-memory and file-backed modes.

Embeddings are managed by a separate VectorIndex (e.g., PythonVectorIndex).
This backend focuses on providing a real database for tests and development
without any server dependency.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import cast

from ragzoom.backends.sqlite_db import SqliteDatabaseManager
from ragzoom.backends.sqlite_repositories import (
    SqliteDocumentRepository,
    SqliteNodeRepository,
)
from ragzoom.contracts.node_repository import NodeRepository as NodeRepositoryProtocol
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.models import Document
from ragzoom.services.cache_manager import CacheManager
from ragzoom.services.tree_navigator import TreeNavigator
from ragzoom.utils.locks import FileDocumentLock, document_lock_path


class _InProcessDocLock(AbstractContextManager[None]):
    def __init__(self, lock: threading.Lock) -> None:
        self._l = lock

    def __enter__(self) -> None:  # noqa: D401 - trivial
        if not self._l.acquire(blocking=False):
            raise RuntimeError("Document is currently being modified")
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        try:
            self._l.release()
        except RuntimeError:
            # Already released
            pass


class SQLiteStorageBackend(StorageBackend):
    def __init__(
        self,
        url: str = "sqlite:///:memory:",
        *,
        vector_backend: str = "python",
        vector_persist_dir: str | None = None,
    ) -> None:
        self.db = SqliteDatabaseManager(url)
        # Cache for nodes (align with existing StoreManager default)
        self.cache: CacheManager[TreeNode] = CacheManager(1000)
        # Per-document in-process locks
        self._locks: dict[str | None, threading.Lock] = {}
        self._lock_base_dir: Path | None = None
        if url.startswith("sqlite:") and ":memory:" in url:
            self._lock_base_dir = Path(tempfile.mkdtemp(prefix="ragzoom-lock-"))

        # No embedded VectorIndex; use independent VectorIndex via factory where needed

        # Repositories
        self.node_repo = SqliteNodeRepository(self.db, self.cache)
        self.doc_repo = SqliteDocumentRepository(self.db)
        # Tree navigation uses repository path operations
        self.tree_nav = TreeNavigator(cast(NodeRepositoryProtocol, self.node_repo))
        # Vector search is handled by independent VectorIndex; no search shim here

    # Removed internal vector index; backends do not manage vector storage

    def _get_lock(self, doc_id: str | None) -> threading.Lock:
        lock = self._locks.get(doc_id)
        if lock is None:
            lock = threading.Lock()
            self._locks[doc_id] = lock
        return lock

    # jscpd:ignore-start Adapter mirrors StoreManager.for_document for interface compatibility
    def for_document(self, document_id: str | None) -> DocumentStore:
        # Compose a DocumentStore with SQLite-backed repositories
        return DocumentStore(
            document_id=document_id,
            node_repo=self.node_repo,
            tree_navigator=self.tree_nav,
            doc_repo=self.doc_repo,
        )

    # jscpd:ignore-end

    def lock_document(self, document_id: str | None) -> AbstractContextManager[None]:
        # Combine cross-process file lock with in-process thread lock
        url = str(self.db.url)
        if self._lock_base_dir is not None:
            base_dir = self._lock_base_dir
        elif url.startswith("sqlite:") and ":memory:" not in url:
            path_part = url.split("sqlite:///")[-1]
            base_dir = Path(path_part).parent
        else:
            from ragzoom.worktree_utils import get_default_sqlite_path

            base_dir = get_default_sqlite_path(None).parent

        file_lock = FileDocumentLock(document_lock_path(base_dir, document_id))
        thread_lock = _InProcessDocLock(self._get_lock(document_id))

        class _CombinedLock(AbstractContextManager[None]):
            def __enter__(self) -> None:  # noqa: D401 - trivial
                file_lock.__enter__()
                thread_lock.__enter__()
                return None

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                try:
                    thread_lock.__exit__(exc_type, exc, tb)
                finally:
                    file_lock.__exit__(exc_type, exc, tb)

        return _CombinedLock()

    def list_documents(self) -> list[Document]:
        # Delegate to repository
        try:
            docs = self.doc_repo.list_documents()
            return docs  # type: ignore[return-value]
        except Exception:
            return []

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        embedding_model: str,
        summary_model: str,
    ) -> DocumentStore:
        self.doc_repo.add_document(
            document_id,
            file_path,
            embedding_model,
            summary_model,
        )
        return self.for_document(document_id)

    def clear_document(self, document_id: str) -> int:
        return self.doc_repo.clear_document(document_id)

    def get_document_by_id(self, document_id: str) -> Document | None:
        row = self.doc_repo.get_document_by_id(document_id)
        return row  # type: ignore[return-value]

    def get_document_by_path(self, file_path: str) -> Document | None:
        row = self.doc_repo.get_document_by_path(file_path)
        return row  # type: ignore[return-value]

    def close(self) -> None:
        self.db.close()
        if self._lock_base_dir is not None:
            shutil.rmtree(self._lock_base_dir, ignore_errors=True)


# (search adapter removed; VectorIndex is used directly where needed)
