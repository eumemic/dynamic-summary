"""PostgreSQL StorageBackend implementation using pgvector.

This backend uses DatabaseManager + repositories directly and exposes the
StorageBackend protocol. It avoids leaking the legacy StoreManager and keeps a
uniform API across backends.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType

from ragzoom.config import OperationalConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.models import Document, PostgresTreeNode
from ragzoom.repositories.document_repository import DocumentRepository
from ragzoom.repositories.node_repository import NodeRepository
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
        self.node_repo = NodeRepository(self.db_manager, self.cache_manager)
        self.doc_repo = DocumentRepository(self.db_manager, self.cache_manager)
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
        return _NoOpLock()

    # Multi-document API
    # jscpd:ignore-start - delegation mirrors repository API for compatibility
    def list_documents(self) -> list[Document]:
        return self.doc_repo.list_documents()

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
        summary_model: str,
    ) -> DocumentStore:
        self.doc_repo.add_document(
            document_id,
            file_path,
            content_hash,
            chunk_count,
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
