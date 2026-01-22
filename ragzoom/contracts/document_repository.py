"""Protocol for document repository used by backends and DocumentStore."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

try:  # types only
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from sqlalchemy.orm import Session
except Exception:  # pragma: no cover
    pass


@runtime_checkable
class DocumentRepository(Protocol):
    def list_documents(self) -> Sequence[object]: ...
    def clear_document(
        self, document_id: str, *, session: Session | None = None
    ) -> int: ...
    def get_document_by_id(self, document_id: str) -> object | None: ...
    def get_document_embedding_model(self, document_id: str) -> str | None: ...
    def get_document_is_temporal(self, document_id: str) -> bool | None: ...
    def set_document_is_temporal(
        self, document_id: str, *, is_temporal: bool
    ) -> None: ...
