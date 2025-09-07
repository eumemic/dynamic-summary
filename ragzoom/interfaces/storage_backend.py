"""Protocol for pluggable storage backends.

This abstracts the SQL/document store so the engine can operate against
SQLite, PostgreSQL, or an in-memory/testing implementation without being
coupled to a concrete database. The surface mirrors the existing StoreManager
/ DocumentStore capabilities and adds a write-lock context manager used by the
engine to serialize document mutations.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from ragzoom.document_store import DocumentStore
from ragzoom.models import Document


@runtime_checkable
class StorageBackend(Protocol):
    """Pluggable storage contract.

    Backends must provide document-scoped stores and a write lock. The rest of
    the surface mirrors the StoreManager API that callers use today, allowing
    an adapter to wrap the current PostgreSQL implementation without invasive
    refactors.
    """

    # Document-scoped store factory
    def for_document(self, doc_id: str | None) -> DocumentStore: ...

    # Write lock for engine-level serialization
    def lock_document(self, document_id: str | None) -> AbstractContextManager[None]:
        """Acquire a write lock for a document.

        Implementations should provide a non-blocking lock that is held for the
        duration of the context manager. If the lock cannot be acquired, the
        implementation should raise an exception immediately rather than block.
        """

    # Multi-document operations
    def list_documents(self) -> list[Document]:
        """List all documents in the system."""

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
        summary_model: str,
    ) -> DocumentStore:
        """Create a document metadata record and return its scoped store."""

    def clear_document(self, document_id: str) -> int:
        """Delete all nodes for a document and remove the metadata record."""

    # Convenience accessors (optional usage by callers)
    def get_document_by_id(self, document_id: str) -> Document | None:
        """Fetch a document by ID."""

    def get_document_by_path(self, file_path: str) -> Document | None:
        """Fetch a document by file path, if any."""

    # Introspection/utility
    def close(self) -> None:
        """Release resources/connections held by the backend."""
