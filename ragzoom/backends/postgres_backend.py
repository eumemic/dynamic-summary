"""Adapter that exposes the existing PostgreSQL StoreManager as a StorageBackend.

This is a minimal wrapper to enable progressive adoption of the pluggable
storage interface without refactoring existing call sites. The write lock is a
no-op for now; advisory locks can be added in a follow-up without changing the
engine contract.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType

from ragzoom.document_store import DocumentStore
from ragzoom.interfaces.storage_backend import StorageBackend
from ragzoom.models import Document
from ragzoom.store import StoreManager


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
    """StorageBackend adapter for the existing StoreManager (PostgreSQL)."""

    def __init__(self, store_manager: StoreManager) -> None:
        self._store = store_manager

    # Document-scoped API
    def for_document(self, doc_id: str | None) -> DocumentStore:
        return self._store.for_document(doc_id)

    # Locking
    def lock_document(self, document_id: str | None) -> AbstractContextManager[None]:
        # Placeholder: use advisory locks in a follow-up change
        return _NoOpLock()

    # Multi-document API
    def list_documents(self) -> list[Document]:
        return self._store.list_documents()

    # jscpd:ignore-start Adapter method forwards to underlying StoreManager.
    # The structure mirrors the original API by design and is considered a
    # legitimate duplication for interface compatibility.
    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
        summary_model: str,
    ) -> DocumentStore:
        return self._store.add_document(
            document_id,
            file_path,
            content_hash,
            chunk_count,
            embedding_model,
            summary_model,
        )

    # jscpd:ignore-end

    def clear_document(self, document_id: str) -> int:
        return self._store.clear_document(document_id)

    def get_document_by_id(self, document_id: str) -> Document | None:
        return self._store.get_document_by_id(document_id)

    def get_document_by_path(self, file_path: str) -> Document | None:
        return self._store.get_document_by_path(file_path)

    def close(self) -> None:
        self._store.close()
