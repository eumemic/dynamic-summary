"""SQLite repositories implementing the minimal surface used by DocumentStore.

These mirror the signatures used by DocumentStore without relying on the
PostgreSQL/pgvector models, enabling an in-memory/file-backed SQLite backend
for tests and development.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from ragzoom.backends.sqlite_db import (
    SqliteDatabaseManager,
    SqliteDocument,
    SqliteTreeNode,
)
from ragzoom.models import TreeNode  # For type hints only
from ragzoom.services.cache_manager import CacheManager


class SqliteNodeRepository:
    def __init__(self, db: SqliteDatabaseManager, cache: CacheManager[TreeNode]):
        self.db = db
        self.cache_manager = cache
        self.SessionLocal = db.SessionLocal

    # --- Create/Update ---
    def add_nodes_batch(
        self, nodes_data: list[dict[str, object]], *, session: Session | None = None
    ) -> list[TreeNode]:
        if not nodes_data:
            return []
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            created = []
            for data in nodes_data:
                node = SqliteTreeNode(
                    id=str(data["node_id"]),
                    parent_id=cast(str | None, data.get("parent_id")),
                    left_child_id=cast(str | None, data.get("left_child_id")),
                    right_child_id=cast(str | None, data.get("right_child_id")),
                    span_start=cast(int, data["span_start"]),
                    span_end=cast(int, data["span_end"]),
                    text=cast(str, data["text"]),
                    token_count=cast(int, data.get("token_count", 0)),
                    document_id=cast(str | None, data.get("document_id")),
                    preceding_neighbor_id=cast(
                        str | None, data.get("preceding_neighbor_id")
                    ),
                    following_neighbor_id=cast(
                        str | None, data.get("following_neighbor_id")
                    ),
                    height=cast(int, data.get("height", 0)),
                    path=cast(str, data.get("path", "")),
                )
                session.add(node)
                created.append(node)
            if own_session:
                session.commit()

            # Ensure attributes are loaded and detach before returning
            for n in created:
                try:
                    session.refresh(n)
                    session.expunge(n)
                except Exception:
                    # Best effort; tests mainly need scalar attributes
                    pass

            # Cache invalidation is minimal here; callers rarely read immediately
            return created  # type: ignore[return-value]
        finally:
            if own_session:
                session.close()

    # jscpd:ignore-start - Small wrapper mirrors NodeRepository signature for tests
    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: list[float] | NDArray[np.float64],
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
        created = self.add_nodes_batch(
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

    # jscpd:ignore-end

    def update_parent_references_batch(
        self, updates: list[tuple[str, str]], *, session: Session | None = None
    ) -> None:
        if not updates:
            return
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            for node_id, parent_id in updates:
                session.execute(
                    update(SqliteTreeNode)
                    .where(SqliteTreeNode.id == node_id)
                    .values(parent_id=parent_id)
                )
            if own_session:
                session.commit()
        finally:
            if own_session:
                session.close()

    # --- Read ---
    def get_node(self, node_id: str) -> TreeNode | None:
        with self.SessionLocal() as session:
            row = session.get(SqliteTreeNode, node_id)
            if row:
                try:
                    session.expunge(row)
                except Exception:
                    pass
            return row  # type: ignore[return-value]

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        if not node_ids:
            return []
        with self.SessionLocal() as session:
            rows = (
                session.execute(
                    select(SqliteTreeNode).where(SqliteTreeNode.id.in_(node_ids))
                )
                .scalars()
                .all()
            )
            for r in rows:
                try:
                    session.expunge(r)
                except Exception:
                    pass
            return rows  # type: ignore[return-value]

    def get_nodes_by_paths(self, paths: list[str]) -> list[TreeNode]:
        if not paths:
            return []
        with self.SessionLocal() as session:
            rows = (
                session.execute(
                    select(SqliteTreeNode).where(SqliteTreeNode.path.in_(paths))
                )
                .scalars()
                .all()
            )
            # jscpd:ignore-start - Detach loop repeats across helpers by design
            for r in rows:
                try:
                    session.expunge(r)
                except Exception:
                    pass
            # jscpd:ignore-end
            return rows  # type: ignore[return-value]

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        with self.SessionLocal() as session:
            if document_id:
                rows = (
                    session.execute(
                        select(SqliteTreeNode).where(
                            SqliteTreeNode.document_id == document_id
                        )
                    )
                    .scalars()
                    .all()
                )
            else:
                rows = session.execute(select(SqliteTreeNode)).scalars().all()
            return rows  # type: ignore[return-value]

    def get_all_nodes_for_document_paginated(
        self, document_id: str | None, *, page_size: int = 1000
    ) -> list[list[TreeNode]]:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        with self.SessionLocal() as session:
            if document_id:
                total_rows = (
                    session.execute(
                        select(SqliteTreeNode).where(
                            SqliteTreeNode.document_id == document_id
                        )
                    )
                    .scalars()
                    .all()
                )
            else:
                total_rows = session.execute(select(SqliteTreeNode)).scalars().all()
            batches: list[list[TreeNode]] = []
            for i in range(0, len(total_rows), page_size):
                batches.append(
                    cast(list[TreeNode], list(total_rows[i : i + page_size]))
                )
            return batches

    def get_leaf_nodes(self) -> list[TreeNode]:
        with self.SessionLocal() as session:
            rows = (
                session.execute(
                    select(SqliteTreeNode).where(
                        SqliteTreeNode.left_child_id.is_(None),
                        SqliteTreeNode.right_child_id.is_(None),
                    )
                )
                .scalars()
                .all()
            )
            return rows  # type: ignore[return-value]

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        with self.SessionLocal() as session:
            stmt = select(SqliteTreeNode).where(SqliteTreeNode.is_pinned == 1)
            rows = session.execute(stmt).scalars().all()
            if depth_max is None:
                return rows  # type: ignore[return-value]
            return [r for r in rows if len(r.path) <= depth_max]  # type: ignore[misc]

    # --- Mutations ---
    def pin_node(self, node_id: str) -> None:
        with self.SessionLocal() as session:
            session.execute(
                update(SqliteTreeNode)
                .where(SqliteTreeNode.id == node_id)
                .values(is_pinned=1)
            )
            session.commit()

    def update_node_access(self, node_id: str) -> None:
        with self.SessionLocal() as session:
            row = session.get(SqliteTreeNode, node_id)
            if row:
                row.access_count = (row.access_count or 0) + 1
                session.add(row)
                session.commit()


class SqliteDocumentRepository:
    def __init__(self, db: SqliteDatabaseManager):
        self.db = db
        self.SessionLocal = db.SessionLocal

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
        summary_model: str,
        *,
        session: Session | None = None,
    ) -> None:
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            doc = SqliteDocument(
                id=document_id,
                file_path=file_path,
                content_hash=content_hash,
                chunk_count=chunk_count,
                embedding_model=embedding_model,
                summary_model=summary_model,
            )
            session.add(doc)
            if own_session:
                session.commit()
        finally:
            if own_session:
                session.close()

    def clear_document(
        self, document_id: str, *, session: Session | None = None
    ) -> int:
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            # Delete nodes then document
            del_nodes = session.execute(
                delete(SqliteTreeNode).where(SqliteTreeNode.document_id == document_id)
            )
            session.execute(
                delete(SqliteDocument).where(SqliteDocument.id == document_id)
            )
            if own_session:
                session.commit()
            return int(del_nodes.rowcount or 0)
        finally:
            if own_session:
                session.close()

    # Provide nodes-only deletion for compatibility with tests expecting
    # StoreInterface.delete_document_nodes semantics
    def delete_document_nodes(
        self, document_id: str, *, session: Session | None = None
    ) -> int:  # noqa: D401
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            del_nodes = session.execute(
                delete(SqliteTreeNode).where(SqliteTreeNode.document_id == document_id)
            )
            if own_session:
                session.commit()
            return int(del_nodes.rowcount or 0)
        finally:
            if own_session:
                session.close()

    def get_document_by_id(self, document_id: str) -> SqliteDocument | None:
        with self.SessionLocal() as session:
            return session.get(SqliteDocument, document_id)

    def get_document_by_path(self, file_path: str) -> SqliteDocument | None:
        with self.SessionLocal() as session:
            row = (
                session.execute(
                    select(SqliteDocument).where(SqliteDocument.file_path == file_path)
                )
                .scalars()
                .first()
            )
            return row

    def get_document_embedding_model(self, document_id: str) -> str | None:
        return self.db.get_document_embedding_model(document_id)

    def list_documents(self) -> list[SqliteDocument]:
        with self.SessionLocal() as session:
            rows = session.query(SqliteDocument).all()
            return rows
