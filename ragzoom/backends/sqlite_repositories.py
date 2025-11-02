"""SQLite repositories implementing the minimal surface used by DocumentStore.

These mirror the signatures used by DocumentStore without relying on the
PostgreSQL/pgvector models, enabling an in-memory/file-backed SQLite backend
for tests and development.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import (
    case,
    delete,
    func,
    insert,
    literal_column,
    or_,
    select,
    tuple_,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ragzoom.backends.sqlite_db import (
    SqliteDatabaseManager,
    SqliteDocument,
    SQLiteTreeNode,
)
from ragzoom.contracts.tree_node import TreeNode  # For type hints only
from ragzoom.services.cache_manager import CacheManager


def _detach_rows(session: Session, rows: Sequence[SQLiteTreeNode]) -> list[TreeNode]:
    """Detach ORM instances and return typed list for callers.

    Centralizes a repeated pattern used by read helpers to avoid retaining
    session-bound instances beyond the query scope.
    """
    for r in rows:
        try:
            session.expunge(r)
        except Exception:
            pass
    return cast(list[TreeNode], list(rows))


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
            # Build payload once to avoid duplication
            payload = [
                {
                    "id": str(data["node_id"]),
                    "parent_id": cast(str | None, data.get("parent_id")),
                    "left_child_id": cast(str | None, data.get("left_child_id")),
                    "right_child_id": cast(str | None, data.get("right_child_id")),
                    "span_start": cast(int, data["span_start"]),
                    "span_end": cast(int, data["span_end"]),
                    "text": cast(str, data["text"]),
                    "token_count": cast(int, data.get("token_count", 0)),
                    "document_id": cast(str | None, data.get("document_id")),
                    "preceding_neighbor_id": cast(
                        str | None, data.get("preceding_neighbor_id")
                    ),
                    "following_neighbor_id": cast(
                        str | None, data.get("following_neighbor_id")
                    ),
                    "height": cast(int, data.get("height", 0)),
                    "level_index": cast(int, data.get("level_index", 0)),
                }
                for data in nodes_data
            ]

            session.execute(insert(SQLiteTreeNode), payload)
            if own_session:
                session.commit()

            # For very large batches, callers typically don't need returned rows
            if len(nodes_data) >= 1000:
                return []

            # Fetch and detach inserted rows so callers receive ORM-like objects
            ids = [str(data["node_id"]) for data in nodes_data]
            rows = (
                session.execute(
                    select(SQLiteTreeNode).where(SQLiteTreeNode.id.in_(ids))
                )
                .scalars()
                .all()
            )
            return _detach_rows(session, rows)
        finally:
            if own_session:
                session.close()

    # jscpd:ignore-start - SQLite implementation parallels Postgres version for parity
    def upsert_nodes_batch(
        self, nodes_data: list[dict[str, object]], *, session: Session | None = None
    ) -> list[TreeNode]:
        if not nodes_data:
            return []
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            node_ids: list[str] = []
            for raw in nodes_data:
                node_id = str(raw["node_id"])
                node_ids.append(node_id)
                stmt = sqlite_insert(SQLiteTreeNode).values(
                    id=node_id,
                    parent_id=cast(str | None, raw.get("parent_id")),
                    left_child_id=cast(str | None, raw.get("left_child_id")),
                    right_child_id=cast(str | None, raw.get("right_child_id")),
                    span_start=cast(int, raw.get("span_start", 0)),
                    span_end=cast(int, raw.get("span_end", 0)),
                    text=cast(str, raw.get("text", "")),
                    token_count=cast(int, raw.get("token_count", 0)),
                    document_id=cast(str | None, raw.get("document_id")),
                    preceding_neighbor_id=cast(
                        str | None, raw.get("preceding_neighbor_id")
                    ),
                    following_neighbor_id=cast(
                        str | None, raw.get("following_neighbor_id")
                    ),
                    height=cast(int, raw.get("height", 0)),
                    level_index=cast(int, raw.get("level_index", 0)),
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[SQLiteTreeNode.id],
                    set_={
                        "text": stmt.excluded.text,
                        "span_start": stmt.excluded.span_start,
                        "span_end": stmt.excluded.span_end,
                        "parent_id": stmt.excluded.parent_id,
                        "left_child_id": stmt.excluded.left_child_id,
                        "right_child_id": stmt.excluded.right_child_id,
                        "document_id": stmt.excluded.document_id,
                        "token_count": stmt.excluded.token_count,
                        "preceding_neighbor_id": stmt.excluded.preceding_neighbor_id,
                        "following_neighbor_id": stmt.excluded.following_neighbor_id,
                        "level_index": stmt.excluded.level_index,
                    },
                )
                session.execute(stmt)
                self.cache_manager.invalidate(node_id)

            if own_session:
                session.commit()

            rows = (
                session.execute(
                    select(SQLiteTreeNode).where(SQLiteTreeNode.id.in_(node_ids))
                )
                .scalars()
                .all()
            )
            detached = _detach_rows(session, rows)
            for node in detached:
                self.cache_manager.put(node.id, node)
            return detached
        finally:
            if own_session:
                session.close()

    # jscpd:ignore-start - Signature mirrors NodeRepository.add_node for protocol parity
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
        level_index: int = 0,
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
                    "level_index": level_index,
                }
            ]
        )
        return created[0]

    # jscpd:ignore-end

    def update_parent_references_batch(
        self,
        updates: Sequence[tuple[str, str | None]],
        *,
        session: Session | None = None,
    ) -> None:
        if not updates:
            return
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            # Perform a single bulk UPDATE using CASE to map ids->parent_ids
            id_list = [node_id for node_id, _ in updates]
            when_clauses = [
                (SQLiteTreeNode.id == node_id, parent_id)
                for node_id, parent_id in updates
            ]
            parent_case = case(*when_clauses, else_=SQLiteTreeNode.parent_id)
            stmt = (
                update(SQLiteTreeNode)
                .where(SQLiteTreeNode.id.in_(id_list))
                .values(parent_id=parent_case)
            )
            session.execute(stmt)
            if own_session:
                session.commit()
        finally:
            if own_session:
                session.close()

    # jscpd:ignore-start
    def update_neighbors_batch(
        self,
        updates: list[tuple[str, str | None, str | None]],
        *,
        session: Session | None = None,
    ) -> None:
        if not updates:
            return
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            for node_id, preceding, following in updates:
                session.execute(
                    update(SQLiteTreeNode)
                    .where(SQLiteTreeNode.id == node_id)
                    .values(
                        preceding_neighbor_id=preceding,
                        following_neighbor_id=following,
                    )
                )
                self.cache_manager.invalidate(node_id)
            if own_session:
                session.commit()
        finally:
            if own_session:
                session.close()

    # jscpd:ignore-end

    # --- Read ---
    def get_node(self, node_id: str) -> TreeNode | None:
        with self.SessionLocal() as session:
            row = session.get(SQLiteTreeNode, node_id)
            if row:
                try:
                    session.expunge(row)
                except Exception:
                    pass
            return row

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        if not node_ids:
            return []
        with self.SessionLocal() as session:
            rows = (
                session.execute(
                    select(SQLiteTreeNode).where(SQLiteTreeNode.id.in_(node_ids))
                )
                .scalars()
                .all()
            )
            return _detach_rows(session, rows)

    def get_root_nodes(self, document_id: str | None = None) -> list[TreeNode]:
        with self.SessionLocal() as session:
            stmt = select(SQLiteTreeNode).where(SQLiteTreeNode.parent_id.is_(None))
            if document_id is not None:
                stmt = stmt.where(SQLiteTreeNode.document_id == document_id)
            rows = session.execute(stmt).scalars().all()
            return _detach_rows(session, rows)

    # jscpd:ignore-start - span query mirrors Postgres implementation for parity
    def get_nodes_overlapping_span(
        self,
        document_id: str | None,
        span_start: int,
        span_end: int,
        *,
        limit: int,
        min_height: int | None = None,
    ) -> tuple[list[TreeNode], int]:
        """Fetch nodes overlapping a span ordered for visualisation."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        if span_end <= span_start:
            raise ValueError("span_end must be greater than span_start")

        with self.SessionLocal() as session:
            filters = [
                SQLiteTreeNode.span_start < span_end,
                SQLiteTreeNode.span_end > span_start,
            ]
            if document_id is not None:
                filters.append(SQLiteTreeNode.document_id == document_id)
            if min_height is not None:
                filters.append(SQLiteTreeNode.height >= int(min_height))

            base_stmt = select(SQLiteTreeNode).where(*filters)

            count_stmt = select(func.count(literal_column("1"))).select_from(
                base_stmt.subquery()
            )
            total = int(session.execute(count_stmt).scalar_one() or 0)

            ordered_stmt = base_stmt.order_by(
                SQLiteTreeNode.height.desc(),
                SQLiteTreeNode.span_start.asc(),
                SQLiteTreeNode.level_index.asc(),
                SQLiteTreeNode.id.asc(),
            ).limit(limit)
            rows = session.execute(ordered_stmt).scalars().all()
            nodes = _detach_rows(session, rows)
            for node in nodes:
                self.cache_manager.put(node.id, node)
            return nodes, total

    # jscpd:ignore-end

    def get_rightmost_leaf_for_document(
        self, document_id: str | None
    ) -> TreeNode | None:
        with self.SessionLocal() as session:
            stmt = select(SQLiteTreeNode).where(SQLiteTreeNode.height == 0)
            if document_id is not None:
                stmt = stmt.where(SQLiteTreeNode.document_id == document_id)
            stmt = stmt.order_by(SQLiteTreeNode.span_end.desc()).limit(1)
            row = session.execute(stmt).scalars().first()
            if not row:
                return None
            try:
                session.expunge(row)
            except Exception:
                pass
            self.cache_manager.put(row.id, row)
            return cast(TreeNode, row)

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        with self.SessionLocal() as session:
            if document_id:
                rows = (
                    session.execute(
                        select(SQLiteTreeNode).where(
                            SQLiteTreeNode.document_id == document_id
                        )
                    )
                    .scalars()
                    .all()
                )
            else:
                rows = session.execute(select(SQLiteTreeNode)).scalars().all()
            return rows  # type: ignore[return-value]

    def count_nodes_for_document(self, document_id: str | None) -> int:
        """Return count of nodes for the given document (fast COUNT(*))"""
        with self.SessionLocal() as session:
            if document_id:
                result = session.execute(
                    select(func.count())
                    .select_from(SQLiteTreeNode)
                    .where(SQLiteTreeNode.document_id == document_id)
                ).scalar_one()
            else:
                result = session.execute(
                    select(func.count()).select_from(SQLiteTreeNode)
                ).scalar_one()
            return int(result or 0)

    def get_all_nodes_for_document_paginated(
        self, document_id: str | None, *, page_size: int = 1000
    ) -> list[list[TreeNode]]:
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        with self.SessionLocal() as session:
            # Build ORM query scoped by document if provided
            q = session.query(SQLiteTreeNode)
            if document_id:
                q = q.filter(SQLiteTreeNode.document_id == document_id)
            # Stable ordering and streaming iteration to avoid large materialization
            q = q.order_by(SQLiteTreeNode.id).yield_per(page_size)

            batches: list[list[TreeNode]] = []
            current: list[SQLiteTreeNode] = []
            for row in q:
                current.append(row)
                if len(current) >= page_size:
                    # Detach loaded rows and append batch
                    for r in current:
                        try:
                            session.expunge(r)
                        except Exception:
                            pass
                    batches.append(cast(list[TreeNode], list(current)))
                    current = []
            if current:
                for r in current:
                    try:
                        session.expunge(r)
                    except Exception:
                        pass
                batches.append(cast(list[TreeNode], list(current)))
            return batches

    def get_leaf_nodes(self) -> list[TreeNode]:
        with self.SessionLocal() as session:
            rows = (
                session.execute(
                    select(SQLiteTreeNode).where(
                        SQLiteTreeNode.left_child_id.is_(None),
                        SQLiteTreeNode.right_child_id.is_(None),
                    )
                )
                .scalars()
                .all()
            )
            return rows  # type: ignore[return-value]

    def get_leaf_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        with self.SessionLocal() as session:
            stmt = select(SQLiteTreeNode).where(
                SQLiteTreeNode.left_child_id.is_(None),
                SQLiteTreeNode.right_child_id.is_(None),
            )
            if document_id:
                stmt = stmt.where(SQLiteTreeNode.document_id == document_id)
            rows = session.execute(stmt).scalars().all()
            return rows  # type: ignore[return-value]

    def count_leaves_for_document(self, document_id: str | None) -> int:
        """Return count of leaf nodes for a document."""
        with self.SessionLocal() as session:
            stmt = (
                select(func.count())
                .select_from(SQLiteTreeNode)
                .where(
                    SQLiteTreeNode.left_child_id.is_(None),
                    SQLiteTreeNode.right_child_id.is_(None),
                )
            )
            if document_id:
                stmt = stmt.where(SQLiteTreeNode.document_id == document_id)
            return int(session.execute(stmt).scalar_one() or 0)

    def max_height_for_document(self, document_id: str | None) -> int:
        """Return maximum node height for a document."""
        with self.SessionLocal() as session:
            stmt = select(func.max(SQLiteTreeNode.height))
            if document_id:
                stmt = stmt.where(SQLiteTreeNode.document_id == document_id)
            result = session.execute(stmt).scalar_one()
            return int(result or 0)

    def get_ready_left_children(self, document_id: str | None) -> list[str]:
        if not document_id:
            return []
        with self.SessionLocal() as session:
            doc_span_end = session.execute(
                select(func.max(SQLiteTreeNode.span_end)).where(
                    SQLiteTreeNode.document_id == document_id
                )
            ).scalar()
            if doc_span_end is None:
                return []

            stmt = (
                select(SQLiteTreeNode.id)
                .where(SQLiteTreeNode.document_id == document_id)
                .where(SQLiteTreeNode.parent_id.is_(None))
                .where((SQLiteTreeNode.level_index % 2) == 0)
                .where(
                    or_(
                        SQLiteTreeNode.span_start > 0,
                        SQLiteTreeNode.span_end < doc_span_end,
                    )
                )
                .where(
                    or_(
                        SQLiteTreeNode.span_start == 0,
                        SQLiteTreeNode.preceding_neighbor_id.is_not(None),
                    )
                )
                .where(
                    or_(
                        SQLiteTreeNode.span_end == doc_span_end,
                        SQLiteTreeNode.following_neighbor_id.is_not(None),
                    )
                )
                .order_by(SQLiteTreeNode.height, SQLiteTreeNode.level_index)
            )
            return [str(row) for row in session.execute(stmt).scalars().all()]

    def get_node_by_height_and_level(
        self,
        document_id: str | None,
        height: int,
        level_index: int,
    ) -> TreeNode | None:
        with self.SessionLocal() as session:
            stmt = select(SQLiteTreeNode).where(
                SQLiteTreeNode.height == height,
                SQLiteTreeNode.level_index == level_index,
            )
            if document_id is not None:
                stmt = stmt.where(SQLiteTreeNode.document_id == document_id)
            row = session.execute(stmt).scalars().first()
            if not row:
                return None
            try:
                session.expunge(row)
            except Exception:
                pass
            return cast(TreeNode, row)

    def get_nodes_by_height_levels(
        self,
        document_id: str | None,
        coordinates: Sequence[tuple[int, int]],
    ) -> list[TreeNode]:
        if not coordinates:
            return []

        unique: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for pair in coordinates:
            if pair in seen:
                continue
            seen.add(pair)
            unique.append(pair)

        results: list[TreeNode] = []
        if not unique:
            return results

        chunk_size = 400  # 400 tuples → 800 bound params; SQLite limit is 999

        with self.SessionLocal() as session:
            tuple_expr = tuple_(SQLiteTreeNode.height, SQLiteTreeNode.level_index)

            for start in range(0, len(unique), chunk_size):
                chunk = unique[start : start + chunk_size]
                stmt = select(SQLiteTreeNode).where(tuple_expr.in_(chunk))
                if document_id is not None:
                    stmt = stmt.where(SQLiteTreeNode.document_id == document_id)

                rows = session.execute(stmt).scalars().all()
                results.extend(_detach_rows(session, rows))

        return results

    def get_parentless_nodes_for_document(
        self, document_id: str | None
    ) -> list[TreeNode]:
        with self.SessionLocal() as session:
            stmt = select(SQLiteTreeNode).where(SQLiteTreeNode.parent_id.is_(None))
            if document_id is None:
                stmt = stmt.where(SQLiteTreeNode.document_id.is_(None))
            else:
                stmt = stmt.where(SQLiteTreeNode.document_id == document_id)

            rows = (
                session.execute(
                    stmt.order_by(
                        SQLiteTreeNode.height,
                        SQLiteTreeNode.level_index,
                        SQLiteTreeNode.span_start,
                    )
                )
                .scalars()
                .all()
            )
            return _detach_rows(session, rows)

    def get_pinned_nodes_for_document(
        self, document_id: str, depth_max: int | None = None
    ) -> list[TreeNode]:
        with self.SessionLocal() as session:
            stmt = select(SQLiteTreeNode).where(
                SQLiteTreeNode.is_pinned == 1,
                SQLiteTreeNode.document_id == document_id,
            )
            rows = session.execute(stmt).scalars().all()
            return cast(list[TreeNode], list(rows))

    def count_pinned_for_document(self, document_id: str | None) -> int:
        with self.SessionLocal() as session:
            stmt = (
                select(func.count())
                .select_from(SQLiteTreeNode)
                .where(SQLiteTreeNode.is_pinned == 1)
            )
            if document_id:
                stmt = stmt.where(SQLiteTreeNode.document_id == document_id)
            return int(session.execute(stmt).scalar_one() or 0)

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        with self.SessionLocal() as session:
            stmt = select(SQLiteTreeNode).where(SQLiteTreeNode.is_pinned == 1)
            rows = session.execute(stmt).scalars().all()
            return cast(list[TreeNode], list(rows))

    # --- Mutations ---
    def pin_node(self, node_id: str) -> None:
        with self.SessionLocal() as session:
            session.execute(
                update(SQLiteTreeNode)
                .where(SQLiteTreeNode.id == node_id)
                .values(is_pinned=1)
            )
            session.commit()

    def update_node_access(self, node_id: str) -> None:
        with self.SessionLocal() as session:
            row = session.get(SQLiteTreeNode, node_id)
            if row:
                row.access_count = (row.access_count or 0) + 1
                session.add(row)
                session.commit()

    def delete_nodes(
        self,
        node_ids: Sequence[str],
        *,
        session: Session | None = None,
    ) -> None:
        if not node_ids:
            return
        own_session = False
        if session is None:
            session = self.SessionLocal()
            own_session = True
        try:
            session.execute(
                delete(SQLiteTreeNode).where(SQLiteTreeNode.id.in_(node_ids))
            )
            if own_session:
                session.commit()
            for node_id in node_ids:
                self.cache_manager.invalidate(node_id)
        finally:
            if own_session:
                session.close()


class SqliteDocumentRepository:
    def __init__(self, db: SqliteDatabaseManager):
        self.db = db
        self.SessionLocal = db.SessionLocal

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
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
                delete(SQLiteTreeNode).where(SQLiteTreeNode.document_id == document_id)
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
                delete(SQLiteTreeNode).where(SQLiteTreeNode.document_id == document_id)
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
