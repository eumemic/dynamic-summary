"""Local Store adapter that exposes StoreInterface over SQLiteStorageBackend.

This provides a StoreManager-like surface backed by the in-memory/file-backed
SQLite backend so existing tests and services that expect a Store can run
without PostgreSQL.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.models import Document, TreeNode


# jscpd:ignore-start Adapter intentionally mirrors StoreManager for compatibility
class LocalStoreAdapter:
    """Adapter exposing Store-like API over SQLiteStorageBackend."""

    PIN_DEPTH_MAX = 2

    def __init__(self, backend: SQLiteStorageBackend):
        self._backend = backend

        # Expose repositories/services similar to StoreManager
        self.node_repo = backend.node_repo
        self.doc_repo = backend.doc_repo
        self.search_service = backend.search_service
        self.tree_navigator = backend.tree_nav

        # Session factory and caches for compatibility
        self.SessionLocal = backend.db.SessionLocal
        self.node_cache = backend.cache.cache
        self.cache_order = backend.cache.cache_order

        self._active_transaction = False

    # --- Document-scoped store ---
    def for_document(self, document_id: str | None) -> DocumentStore:
        return self._backend.for_document(document_id)

    # --- Multi-document management ---
    def list_documents(self) -> list[Document]:
        # Minimal implementation; tests rarely depend on this
        docs: list[Document] = []
        with self.SessionLocal() as session:  # type: ignore[attr-defined]
            from ragzoom.backends.sqlite_db import SqliteDocument

            rows = session.query(SqliteDocument).all()
            for r in rows:
                session.expunge(r)
                docs.append(r)  # type: ignore[arg-type]
        return docs

    def get_document_by_path(self, file_path: str) -> Document | None:
        return self.doc_repo.get_document_by_path(file_path)  # type: ignore[return-value]

    def get_document_by_id(self, document_id: str) -> Document | None:
        return self.doc_repo.get_document_by_id(document_id)  # type: ignore[return-value]

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

    def clear_document(self, document_id: str, *, session: object | None = None) -> int:
        # session is accepted for compatibility but ignored by SQLite adapter unless it is a SQLAlchemy session
        try:
            # Best-effort: pass through if this looks like a session
            return self.doc_repo.clear_document(document_id, session=session)  # type: ignore[arg-type]
        except TypeError:
            return self.doc_repo.clear_document(document_id)

    def delete_document_nodes(
        self, document_id: str, *, session: None = None
    ) -> int:  # noqa: ARG002
        # Nodes only, leave document record in place
        return self.doc_repo.delete_document_nodes(document_id)  # type: ignore[attr-defined]

    # --- Node operations (compatibility layer) ---
    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: (
            list[float] | NDArray[np.float64]
        ),  # noqa: ARG002 - embeddings not stored in SQLite repo
        span_start: int,
        span_end: int,
        parent_id: str | None = None,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
        document_id: str | None = None,
        token_count: int = 0,
        height: int = 0,
        is_left_child: bool | None = None,
    ) -> TreeNode:
        # Route via doc-scoped repository
        doc_store = self.for_document(document_id)
        created = doc_store.nodes.add_batch(
            [
                {
                    "node_id": node_id,
                    "text": text,
                    "span_start": span_start,
                    "span_end": span_end,
                    "parent_id": parent_id,
                    "left_child_id": left_child_id,
                    "right_child_id": right_child_id,
                    "document_id": document_id,
                    "token_count": token_count,
                    "height": height,
                }
            ]
        )
        return created[0]

    def add_nodes_batch(self, nodes_data: list[dict[str, object]]) -> list[TreeNode]:
        if not nodes_data:
            return []
        # Group by document to preserve semantics
        by_doc: dict[str | None, list[dict[str, object]]] = {}
        for d in nodes_data:
            key = cast(str | None, d.get("document_id"))
            by_doc.setdefault(key, []).append(d)
        out: list[TreeNode] = []
        for doc_id, group in by_doc.items():
            ds = self.for_document(doc_id)
            out.extend(
                ds.nodes.add_batch(
                    cast(
                        list[
                            dict[
                                str,
                                str
                                | int
                                | float
                                | bool
                                | list[float]
                                | NDArray[np.float64]
                                | None,
                            ]
                        ],
                        group,
                    )
                )
            )
        return out

    def update_parent_references_batch(self, updates: list[tuple[str, str]]) -> None:
        self.node_repo.update_parent_references_batch(updates)

    def get_node(self, node_id: str) -> TreeNode | None:
        return self.node_repo.get_node(node_id)

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        return self.node_repo.get_nodes(node_ids)

    def update_node_access(self, node_id: str) -> None:
        self.node_repo.update_node_access(node_id)

    def get_leaf_nodes(self) -> list[TreeNode]:
        return self.node_repo.get_leaf_nodes()

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        return self.node_repo.get_all_nodes_for_document(document_id)

    def get_all_nodes_for_document_paginated(
        self, document_id: str | None, *, page_size: int = 1000
    ) -> list[list[TreeNode]]:
        return self.node_repo.get_all_nodes_for_document_paginated(
            document_id, page_size=page_size
        )

    # --- Search operations ---
    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[tuple[str, float, dict[str, str | int | float | bool | None]]]:
        return self.search_service.search_similar(
            query_embedding,
            n_results,
            cast(dict[str, str | int | float] | None, where),
        )  # type: ignore[return-value]

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, dict[str, object]]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        return self.search_service.compute_mmr_diverse_results(  # type: ignore[return-value]
            query_embedding,
            cast(
                list[tuple[str, float, dict[str, str | int | float | bool | None]]],
                candidates,
            ),
            lambda_param,
            k,
        )

    # --- Tree navigation ---
    def get_children(self, node_id: str) -> tuple[TreeNode | None, TreeNode | None]:
        return self.tree_navigator.get_children(node_id)

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        return self.tree_navigator.get_ancestors(node_ids)

    def get_root_node(self) -> TreeNode | None:
        return self.tree_navigator.get_root_node()

    def get_root_node_for_document(self, document_id: str | None) -> TreeNode | None:
        return self.tree_navigator.get_root_node_for_document(document_id)

    def get_node_depth(self, node_id: str) -> int:
        return self.tree_navigator.get_node_depth(node_id)

    def is_leaf_node(self, node_id: str) -> bool:
        return self.tree_navigator.is_leaf_node(node_id)

    def is_root_node(self, node_id: str) -> bool:
        return self.tree_navigator.is_root_node(node_id)

    # --- System-wide pinning ---
    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        return self.node_repo.get_pinned_nodes(depth_max)

    def pin_node(self, node_id: str) -> None:
        # Enforce depth policy similarly to StoreManager
        depth = self.tree_navigator.get_node_depth(node_id)
        if depth > self.PIN_DEPTH_MAX:
            from ragzoom.exceptions import InvalidOperationError

            raise InvalidOperationError(
                "pin_node",
                f"Node {node_id} is at depth {depth}, which exceeds maximum pin depth {self.PIN_DEPTH_MAX}",
            )
        self.node_repo.pin_node(node_id)

    # --- Lifecycle ---
    @contextmanager
    def transaction(self) -> Generator[object, None, None]:
        if self._active_transaction:
            raise RuntimeError(
                "Nested transactions are not supported. Please use the same session for all operations within a transaction."
            )
        self._active_transaction = True
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._active_transaction = False
            session.close()

    def close(self) -> None:
        self._backend.close()

    @staticmethod
    def compute_content_hash(content: str) -> str:
        import hashlib

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Backward-compat convenience properties
    @property
    def nodes(self) -> object:  # noqa: D401 - simple alias
        return self.node_repo

    @property
    def documents(self) -> object:  # noqa: D401 - simple alias
        return self.doc_repo

    @property
    def search(self) -> object:  # noqa: D401 - simple alias
        return self.search_service

    @property
    def tree(self) -> object:  # noqa: D401 - simple alias
        return self.tree_navigator


# jscpd:ignore-end
