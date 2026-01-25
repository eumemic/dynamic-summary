"""Protocol for pluggable storage backends."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from sqlalchemy.engine import Engine

from ragzoom.document_store import DocumentStore
from ragzoom.models import Document

if TYPE_CHECKING:
    from ragzoom.server.lease import IndexerLease, LeaseConfig


@runtime_checkable
class StorageBackend(Protocol):
    @property
    def engine(self) -> Engine:
        """Return the SQLAlchemy engine for this backend.

        Used for migrations and other database-level operations.
        """
        ...

    def for_document(self, doc_id: str | None) -> DocumentStore: ...

    def lock_document(self, doc_id: str | None) -> AbstractContextManager[None]: ...

    def create_lease(self, config: LeaseConfig | None = None) -> IndexerLease:
        """Create a global indexer lease for single-writer coordination.

        Args:
            config: Optional lease configuration. Uses defaults if not provided.

        The lease ensures only one IndexingEngine can write to the database
        at a time, preventing corruption during deployments where multiple
        server instances may briefly run simultaneously.
        """
        ...

    def list_documents(self) -> list[Document]: ...

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        embedding_model: str,
        summary_model: str,
        summarization_guidance: str | None = None,
    ) -> DocumentStore: ...

    def clear_document(self, document_id: str) -> int: ...

    def delete_nodes_from_span(
        self, document_id: str, span_start: int
    ) -> list[str]: ...

    def get_document_by_id(self, document_id: str) -> Document | None: ...

    def get_document_by_path(self, file_path: str) -> Document | None: ...

    def close(self) -> None: ...
