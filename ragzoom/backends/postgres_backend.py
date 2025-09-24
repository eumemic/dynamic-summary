"""PostgreSQL StorageBackend implementation using pgvector.

This backend uses DatabaseManager + repositories directly and exposes the
StorageBackend protocol. It avoids leaking the legacy StoreManager and keeps a
uniform API across backends.
"""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager
from types import TracebackType

from sqlalchemy import text
from sqlalchemy.engine import Connection, Engine

from ragzoom.config import OperationalConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.models import Document, PostgresTreeNode
from ragzoom.repositories.document_repository import (
    DocumentRepository as PostgresDocumentRepository,
)
from ragzoom.repositories.postgres_node_repository import PostgresNodeRepository
from ragzoom.services.cache_manager import CacheManager
from ragzoom.services.tree_navigator import TreeNavigator
from ragzoom.storage.database_manager import DatabaseManager


class _NoOpLock(AbstractContextManager[None]):
    """A no-op context manager used as a placeholder write lock."""

    def __enter__(self) -> None:  # noqa: D401 - trivial
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


class _AdvisoryLock(AbstractContextManager[None]):
    """PostgreSQL advisory lock scoped to a document."""

    def __init__(self, engine: Engine, key1: int, key2: int) -> None:
        self._engine = engine
        self._key1 = key1
        self._key2 = key2
        self._conn: Connection | None = None

    def __enter__(self) -> None:  # noqa: D401 - trivial
        self._conn = self._engine.connect()
        self._conn.execute(
            text("SELECT pg_advisory_lock(:k1, :k2)"),
            {"k1": self._key1, "k2": self._key2},
        )
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            if self._conn is not None:
                self._conn.execute(
                    text("SELECT pg_advisory_unlock(:k1, :k2)"),
                    {"k1": self._key1, "k2": self._key2},
                )
        finally:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _hash_document_lock(document_id: str) -> tuple[int, int]:
    digest = hashlib.sha256(document_id.encode("utf-8")).digest()
    key1 = int.from_bytes(digest[:8], "big", signed=False)
    key2 = int.from_bytes(digest[8:16], "big", signed=False)

    def _to_signed(value: int) -> int:
        if value >= 2**63:
            return value - 2**64
        return value

    return _to_signed(key1), _to_signed(key2)


class PostgresStorageBackend(StorageBackend):
    """PostgreSQL-backed StorageBackend using repositories and services."""

    DEFAULT_CACHE_SIZE = 1000

    def __init__(
        self, config: OperationalConfig, embedding_model: str = "text-embedding-3-small"
    ) -> None:
        self.config = config
        # Initialize core components
        self.db_manager = DatabaseManager(config, embedding_model)
        self.cache_manager = CacheManager[PostgresTreeNode](
            config.cache_size or self.DEFAULT_CACHE_SIZE
        )
        self.node_repo = PostgresNodeRepository(self.db_manager, self.cache_manager)
        self.doc_repo = PostgresDocumentRepository(self.db_manager, self.cache_manager)
        self.tree_navigator = TreeNavigator(self.node_repo)

    # Document-scoped API
    def for_document(self, doc_id: str | None) -> DocumentStore:
        return DocumentStore(
            document_id=doc_id,
            node_repo=self.node_repo,
            tree_navigator=self.tree_navigator,
            doc_repo=self.doc_repo,
        )

    # Locking (no-op; can be replaced with advisory locks)
    def lock_document(self, document_id: str | None) -> AbstractContextManager[None]:
        if not document_id:
            return _NoOpLock()
        key1, key2 = _hash_document_lock(document_id)
        return _AdvisoryLock(self.db_manager.engine, key1, key2)

    # Multi-document API
    # jscpd:ignore-start - delegation mirrors repository API for compatibility
    def list_documents(self) -> list[Document]:
        return self.doc_repo.list_documents()

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        embedding_model: str,
        summary_model: str,
    ) -> DocumentStore:
        self.doc_repo.add_document(
            document_id,
            file_path,
            content_hash,
            embedding_model,
            summary_model,
        )
        # jscpd:ignore-end
        return self.for_document(document_id)

    def clear_document(self, document_id: str) -> int:
        return self.doc_repo.clear_document(document_id)

    def get_document_by_id(self, document_id: str) -> Document | None:
        return self.doc_repo.get_document_by_id(document_id)

    def get_document_by_path(self, file_path: str) -> Document | None:
        return self.doc_repo.get_document_by_path(file_path)

    def close(self) -> None:
        self.db_manager.close()
