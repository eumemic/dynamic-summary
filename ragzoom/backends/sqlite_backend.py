"""SQLite StorageBackend with in-memory and file-backed modes.

Embeddings are managed by a separate VectorIndex (e.g., PythonVectorIndex).
This backend focuses on providing a real database for tests and development
without any server dependency.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
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
from ragzoom.utils.locks import FileDocumentLock, document_lock_path
from ragzoom.worktree_utils import (
    DEFAULT_VECTOR_DIR_NAME,
    get_default_vector_dir,
)


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

        # Vector index for this backend
        self.vector_index = self._make_vector_index(vector_backend, vector_persist_dir)

        # Repositories
        self.node_repo = SqliteNodeRepository(self.db, self.cache)
        self.doc_repo = SqliteDocumentRepository(self.db)
        # Tree navigation uses repository path operations
        self.tree_nav = TreeNavigator(cast(NodeRepository, self.node_repo))
        # SearchService-compatible shim over configured VectorIndex is provided here
        # For now, keep the interface by composing a thin adapter
        self.search_service = _VectorIndexSearchAdapter(self.vector_index)

    def _make_vector_index(
        self, backend: str, persist_dir: str | None
    ) -> PythonVectorIndex:
        # Resolve persistence dir default based on sqlite database path
        if persist_dir is None:
            try:
                url = str(self.db.url)
            except Exception:
                url = ""
            # Extract file path from sqlite:/// URL
            if url.startswith("sqlite:") and ":memory:" not in url:
                # naive parse: sqlite:////abs or sqlite:///rel
                path_part = url.split("sqlite:///")[-1]
                import os

                db_dir = os.path.dirname(path_part)
                persist_dir = os.path.join(db_dir, DEFAULT_VECTOR_DIR_NAME)

        if backend == "chroma":
            from ragzoom.backends.chroma_vector_index import ChromaVectorIndex

            # Chroma requires a directory path
            if persist_dir is None:
                base = str(get_default_vector_dir(None))
            else:
                base = persist_dir
            try:
                import os

                os.makedirs(base, exist_ok=True)
            except Exception:
                pass
            return ChromaVectorIndex(base)  # type: ignore[return-value]

        # Default to PythonVectorIndex (optionally persistent)
        return PythonVectorIndex(persist_dir)

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
            tree_navigator=self.tree_nav,
            doc_repo=self.doc_repo,  # type: ignore[arg-type]
        )

    # jscpd:ignore-end

    def lock_document(self, document_id: str | None) -> AbstractContextManager[None]:
        # Combine cross-process file lock with in-process thread lock
        url = str(self.db.url)
        if url.startswith("sqlite:") and ":memory:" not in url:
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
            if isinstance(meta, dict):
                md: dict[str, str | int | float | bool | None] = {
                    "span_start": int(meta.get("span_start", 0)),
                    "span_end": int(meta.get("span_end", 0)),
                    "parent_id": str(meta.get("parent_id", "")),
                    "document_id": str(meta.get("document_id", "")),
                    "is_leaf": int(meta.get("is_leaf", 0)),
                }
            else:
                # Assume VectorSearchMetadata with attributes
                md = {
                    "span_start": int(getattr(meta, "span_start", 0)),
                    "span_end": int(getattr(meta, "span_end", 0)),
                    "parent_id": str(getattr(meta, "parent_id", "")),
                    "document_id": str(getattr(meta, "document_id", "")),
                    "is_leaf": int(getattr(meta, "is_leaf", 0)),
                }
            out.append((node_id, score, md))
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
