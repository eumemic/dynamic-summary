"""SQLite StorageBackend with in-memory and file-backed modes.

Embeddings are managed by a separate VectorIndex (e.g., PythonVectorIndex).
This backend focuses on providing a real database for tests and development
without any server dependency.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.backends.python_vector_index import PythonVectorIndex
from ragzoom.backends.sqlite_db import SqliteDatabaseManager
from ragzoom.backends.sqlite_repositories import (
    SqliteDocumentRepository,
    SqliteNodeRepository,
)
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorSearchMetadata
from ragzoom.document_store import DocumentStore
from ragzoom.models import Document, TreeNode
from ragzoom.repositories.node_repository import NodeRepository
from ragzoom.services.cache_manager import CacheManager
from ragzoom.services.tree_navigator import TreeNavigator


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
    def __init__(self, url: str = "sqlite:///:memory:") -> None:
        self.db = SqliteDatabaseManager(url)
        # Cache for nodes (align with existing StoreManager default)
        self.cache: CacheManager[TreeNode] = CacheManager(1000)
        # Per-document in-process locks
        self._locks: dict[str | None, threading.Lock] = {}

        # Vector index for this backend (in-memory for tests; file-backed via path)
        # For file-backed deployments, pass a directory to persist
        self.vector_index = PythonVectorIndex()

        # Repositories
        self.node_repo = SqliteNodeRepository(self.db, self.cache)
        self.doc_repo = SqliteDocumentRepository(self.db)
        # Tree navigation uses repository path operations
        self.tree_nav = TreeNavigator(cast(NodeRepository, self.node_repo))
        # SearchService-compatible shim over PythonVectorIndex will be provided later
        # For now, keep the interface by composing a thin adapter
        self.search_service = _VectorIndexSearchAdapter(self.vector_index)

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
            node_repo=self.node_repo,  # type: ignore[arg-type]
            search_service=self.search_service,  # type: ignore[arg-type]
            tree_navigator=self.tree_nav,  # type: ignore[arg-type]
            doc_repo=self.doc_repo,  # type: ignore[arg-type]
        )

    # jscpd:ignore-end

    def lock_document(self, document_id: str | None) -> AbstractContextManager[None]:
        return _InProcessDocLock(self._get_lock(document_id))

    def list_documents(self) -> list[Document]:
        # Delegate to repository
        try:
            docs = self.doc_repo.list_documents()  # type: ignore[attr-defined]
            return docs  # type: ignore[return-value]
        except Exception:
            return []

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


class _VectorIndexSearchAdapter:
    """Thin adapter exposing SearchService-like methods over PythonVectorIndex.

    Only the methods used by DocumentStore wrappers are implemented.
    """

    def __init__(self, index: PythonVectorIndex) -> None:
        # We don't hold a DatabaseManager here; this class only exposes compatible methods
        self._index = index

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, str | int | float] | None = None,
    ) -> list[tuple[str, float, dict[str, str | int | float | bool | None]]]:
        # Convert to NodeMetadataDict shape expected downstream
        results = self._index.search_similar(
            query_embedding,
            n_results,
            cast(dict[str, str | int | float | bool | None] | None, where),
        )
        out: list[tuple[str, float, dict[str, str | int | float | bool | None]]] = []
        for node_id, score, meta in results:
            out.append(
                (
                    node_id,
                    score,
                    {
                        "span_start": meta.span_start,
                        "span_end": meta.span_end,
                        "parent_id": meta.parent_id,
                        "document_id": meta.document_id,
                        "is_leaf": meta.is_leaf,
                    },
                )
            )
        return out

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, dict[str, str | int | float | bool | None]]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        conv: list[tuple[str, float, VectorSearchMetadata]] = cast(
            list[tuple[str, float, VectorSearchMetadata]],
            [(nid, sc, cast(VectorSearchMetadata, md)) for (nid, sc, md) in candidates],
        )
        return self._index.compute_mmr_diverse_results(
            query_embedding, conv, lambda_param, k
        )

    # Optional upsert API used during indexing when embeddings are not stored in SQL
    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        self._index.upsert(items)
