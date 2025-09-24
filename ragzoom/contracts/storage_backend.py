"""Protocol for pluggable storage backends."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Protocol, runtime_checkable

from ragzoom.document_store import DocumentStore
from ragzoom.models import Document


@runtime_checkable
class StorageBackend(Protocol):
    def for_document(self, doc_id: str | None) -> DocumentStore: ...

    def lock_document(self, doc_id: str | None) -> AbstractContextManager[None]: ...

    def list_documents(self) -> list[Document]: ...

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        embedding_model: str,
        summary_model: str,
    ) -> DocumentStore: ...

    def clear_document(self, document_id: str) -> int: ...

    def get_document_by_id(self, document_id: str) -> Document | None: ...

    def get_document_by_path(self, file_path: str) -> Document | None: ...

    def close(self) -> None: ...
