"""Repository for PostgresTreeNode CRUD operations using PostgreSQL with pgvector."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import delete as sa_delete
from sqlalchemy import func, literal_column, or_, text, tuple_, update

from ragzoom.contracts.tree_node import TreeNode
from ragzoom.models import PostgresTreeNode

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
from ragzoom.repositories.base_repository import BaseRepository
from ragzoom.services.cache_manager import CacheManager
from ragzoom.storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class NodeDataDict(TypedDict, total=False):
    """Type definition for node data used in batch operations."""

    # Required fields
    node_id: str
    document_id: str
    # Optional fields
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None
    span_start: int
    span_end: int
    text: str
    embedding: list[float] | None
    node_type: str
    height: int
    token_count: int
    created_at: datetime
    updated_at: datetime
    preceding_neighbor_id: str | None
    following_neighbor_id: str | None
    level_index: int


class PostgresNodeRepository(BaseRepository):
    """Repository for PostgresTreeNode database operations."""

    def __init__(
        self,
        database_manager: DatabaseManager,
        cache_manager: CacheManager[PostgresTreeNode],
    ):
        """Initialize node repository.

        Args:
            database_manager: Database manager for DB operations
            cache_manager: Cache manager for hot nodes
        """
        self.db_manager = database_manager
        self.cache_manager = cache_manager
        self.SessionLocal = database_manager.SessionLocal

    def _force_load_and_detach(self, session: Session, node: PostgresTreeNode) -> None:
        """Force load all attributes and detach node from session."""
        # Force load all attributes before detaching
        _ = (
            node.id,
            node.parent_id,
            node.left_child_id,
            node.right_child_id,
            node.span_start,
            node.span_end,
            node.text,
            node.token_count,
            node.is_pinned,
            node.last_accessed,
            node.access_count,
            node.created_at,
            node.document_id,
            node.preceding_neighbor_id,
            node.following_neighbor_id,
            node.height,
            node.level_index,
        )
        session.expunge(node)

    # jscpd:ignore-start -- signature must mirror NodeRepository.add_node exactly
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
        """Add a node to the database (embedding ignored).

        Args:
            node_id: Unique identifier for the node
            text: Node text content
            embedding: Embedding vector
            span_start: Start position in document
            span_end: End position in document
            parent_id: Parent node ID (optional)
            left_child_id: Left child node ID (optional)
            right_child_id: Right child node ID (optional)
            document_id: Document this node belongs to (optional)
            token_count: Number of tokens in text
            height: Height in tree (0 for leaves)
            is_left_child: Whether this node is a left child (True) or right child (False) of its parent.
                          If None, will attempt to determine from parent's existing child pointers.

        Returns:
            Created PostgresTreeNode
        """
        # Embeddings are not stored in SQL; ignore validation

        with self.SessionLocal() as session:
            node = PostgresTreeNode(
                id=node_id,
                parent_id=parent_id,
                left_child_id=left_child_id,
                right_child_id=right_child_id,
                span_start=span_start,
                span_end=span_end,
                text=text,
                document_id=document_id,
                token_count=token_count,
                height=height,
                level_index=level_index,
            )
            session.add(node)
            session.commit()

            # Refresh to ensure all attributes are loaded
            session.refresh(node)

            # Detach the node from the session
            self._force_load_and_detach(session, node)

            # Add to cache
            self.cache_manager.put(node_id, node)

        return node

    # jscpd:ignore-end

    def add_nodes_batch(
        self,
        nodes_data: list[dict[str, object]],
        *,
        session: Session | None = None,
    ) -> list[TreeNode]:
        """Add multiple nodes to the database in batch.

        Args:
            nodes_data: List of dictionaries containing node data
            session: Optional database session for transactional operations

        Returns:
            List of created PostgresTreeNode objects
        """
        if not nodes_data:
            return []

        # Embeddings not stored in SQL; ignore any provided embedding values

        db_session, should_commit = self._get_session(session)
        try:
            # Create PostgresTreeNode objects for regular session.add_all()
            nodes_pg: list[PostgresTreeNode] = []
            out: list[TreeNode] = []
            for raw in nodes_data:
                data = raw  # accept generic dicts per protocol
                node = PostgresTreeNode(
                    id=str(data["node_id"]),
                    parent_id=cast(str | None, data.get("parent_id")),
                    left_child_id=cast(str | None, data.get("left_child_id")),
                    right_child_id=cast(str | None, data.get("right_child_id")),
                    span_start=cast(int, data["span_start"]),
                    span_end=cast(int, data["span_end"]),
                    text=cast(str, data["text"]),
                    document_id=cast(str | None, data.get("document_id")),
                    token_count=cast(int, data["token_count"]),
                    preceding_neighbor_id=cast(
                        str | None, data.get("preceding_neighbor_id")
                    ),
                    following_neighbor_id=cast(
                        str | None, data.get("following_neighbor_id")
                    ),
                    height=cast(int, data["height"]),
                    level_index=cast(int, data["level_index"]),
                )
                nodes_pg.append(node)
                out.append(node)

            # Use add_all for proper object tracking and session management
            db_session.add_all(nodes_pg)
            if should_commit:
                db_session.commit()

            # Force load and detach all nodes
            for node in nodes_pg:
                db_session.refresh(node)
                self._force_load_and_detach(db_session, node)

            # Add all to cache
            for node in nodes_pg:
                self.cache_manager.put(node.id, node)

            return out

        except Exception:
            if should_commit:
                db_session.rollback()
            raise
        finally:
            if should_commit:
                db_session.close()

    # jscpd:ignore-start - Upsert mirrors SQLite implementation for parity
    def upsert_nodes_batch(
        self,
        nodes_data: list[dict[str, object]],
        *,
        session: Session | None = None,
    ) -> list[TreeNode]:
        if not nodes_data:
            return []

        db_session, should_commit = self._get_session(session)
        insert_sql = text(
            """
            INSERT INTO tree_nodes (
                id,
                text,
                span_start,
                span_end,
                parent_id,
                left_child_id,
                right_child_id,
                document_id,
                token_count,
                height,
                preceding_neighbor_id,
                following_neighbor_id
                , level_index
            ) VALUES (
                :id,
                :text,
                :span_start,
                :span_end,
                :parent_id,
                :left_child_id,
                :right_child_id,
                :document_id,
                :token_count,
                :height,
                :preceding_neighbor_id,
                :following_neighbor_id,
                :level_index
            )
            ON CONFLICT (id) DO UPDATE SET
                text = EXCLUDED.text,
                span_start = EXCLUDED.span_start,
                span_end = EXCLUDED.span_end,
                parent_id = EXCLUDED.parent_id,
                left_child_id = EXCLUDED.left_child_id,
                right_child_id = EXCLUDED.right_child_id,
                document_id = EXCLUDED.document_id,
                token_count = EXCLUDED.token_count,
                preceding_neighbor_id = EXCLUDED.preceding_neighbor_id,
                following_neighbor_id = EXCLUDED.following_neighbor_id,
                level_index = EXCLUDED.level_index
            """
        )

        node_ids: list[str] = []
        try:
            for raw in nodes_data:
                node_id = str(raw["node_id"])
                node_ids.append(node_id)
                params = {
                    "id": node_id,
                    "text": str(raw["text"]),
                    "span_start": int(cast(int | float, raw["span_start"])),
                    "span_end": int(cast(int | float, raw["span_end"])),
                    "parent_id": cast(str | None, raw.get("parent_id")),
                    "left_child_id": cast(str | None, raw.get("left_child_id")),
                    "right_child_id": cast(str | None, raw.get("right_child_id")),
                    "document_id": cast(str | None, raw.get("document_id")),
                    "token_count": int(cast(int | float, raw["token_count"])),
                    "height": int(cast(int | float, raw["height"])),
                    "preceding_neighbor_id": cast(
                        str | None, raw.get("preceding_neighbor_id")
                    ),
                    "following_neighbor_id": cast(
                        str | None, raw.get("following_neighbor_id")
                    ),
                    "level_index": int(cast(int | float, raw["level_index"])),
                }
                db_session.execute(insert_sql, params)
                self.cache_manager.invalidate(node_id)

            if should_commit:
                db_session.commit()

            return self.get_nodes(node_ids)
        except Exception:
            if should_commit:
                db_session.rollback()
            raise
        finally:
            if should_commit:
                db_session.close()

    def update_parent_references_batch(
        self,
        updates: Sequence[tuple[str, str | None]],
        *,
        session: Session | None = None,
    ) -> None:
        """Update parent references for multiple nodes in batch.

        Args:
            updates: List of (node_id, parent_id) tuples
            session: Optional database session for transactional operations
        """
        if not updates:
            return

        with self._session_scope(session) as db_session:
            # Update parent references
            for node_id, parent_id in updates:
                db_session.execute(
                    update(PostgresTreeNode)
                    .where(PostgresTreeNode.id == node_id)
                    .values(parent_id=parent_id)
                )

            # Invalidate cache for updated nodes
            for node_id, _ in updates:
                self.cache_manager.invalidate(node_id)

    def update_neighbors_batch(
        self,
        updates: list[tuple[str, str | None, str | None]],
        *,
        session: Session | None = None,
    ) -> None:
        if not updates:
            return

        with self._session_scope(session) as db_session:
            for node_id, preceding, following in updates:
                db_session.execute(
                    update(PostgresTreeNode)
                    .where(PostgresTreeNode.id == node_id)
                    .values(
                        preceding_neighbor_id=preceding,
                        following_neighbor_id=following,
                    )
                )
                self.cache_manager.invalidate(node_id)

    # jscpd:ignore-end
    def get_node(self, node_id: str) -> TreeNode | None:
        """Get a node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            PostgresTreeNode if found, None otherwise
        """
        # Check cache first
        cached = self.cache_manager.get(node_id)
        if cached:
            return cached

        # Load from database
        with self.SessionLocal() as session:
            node = session.query(PostgresTreeNode).filter_by(id=node_id).first()
            if node:
                # Force load and detach
                self._force_load_and_detach(session, node)
                # Add to cache
                self.cache_manager.put(node_id, node)
                return node

        return None

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes by their IDs.

        Args:
            node_ids: List of node IDs to retrieve

        Returns:
            List of PostgresTreeNode objects found
        """
        if not node_ids:
            return []

        nodes: list[TreeNode] = []
        uncached_ids = []

        # Check cache first
        for node_id in node_ids:
            cached = self.cache_manager.get(node_id)
            if cached:
                nodes.append(cached)
            else:
                uncached_ids.append(node_id)

        # Load uncached nodes from database
        if uncached_ids:
            with self.SessionLocal() as session:
                db_nodes = (
                    session.query(PostgresTreeNode)
                    .filter(PostgresTreeNode.id.in_(uncached_ids))
                    .all()
                )
                for node in db_nodes:
                    # Force load and detach
                    self._force_load_and_detach(session, node)
                    # Add to cache
                    self.cache_manager.put(node.id, node)
                    nodes.append(node)

        return nodes

    def get_root_nodes(self, document_id: str | None = None) -> list[TreeNode]:
        """Return all nodes whose parent is null, optionally filtered by document."""

        with self.SessionLocal() as session:
            query = session.query(PostgresTreeNode).filter(
                PostgresTreeNode.parent_id.is_(None)
            )
            if document_id is not None:
                query = query.filter(PostgresTreeNode.document_id == document_id)

            roots = query.all()
            extracted: list[TreeNode] = []
            for node in roots:
                self._force_load_and_detach(session, node)
                self.cache_manager.put(node.id, node)
                extracted.append(node)

            return extracted

    def get_nodes_overlapping_span(
        self,
        document_id: str | None,
        span_start: int,
        span_end: int,
        *,
        limit: int,
        min_height: int | None = None,
    ) -> tuple[list[TreeNode], int]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if span_end <= span_start:
            raise ValueError("span_end must be greater than span_start")

        with self.SessionLocal() as session:
            filters = [
                PostgresTreeNode.span_start < span_end,
                PostgresTreeNode.span_end > span_start,
            ]
            if document_id is not None:
                filters.append(PostgresTreeNode.document_id == document_id)
            if min_height is not None:
                filters.append(PostgresTreeNode.height >= int(min_height))

            query = session.query(PostgresTreeNode).filter(*filters)

            total = int(
                query.with_entities(func.count(literal_column("1"))).scalar() or 0
            )

            ordered = (
                query.order_by(
                    PostgresTreeNode.height.desc(),
                    PostgresTreeNode.span_start.asc(),
                    PostgresTreeNode.level_index.asc(),
                    PostgresTreeNode.id.asc(),
                )
                .limit(limit)
                .all()
            )

            nodes: list[TreeNode] = []
            for node in ordered:
                self._force_load_and_detach(session, node)
                self.cache_manager.put(node.id, node)
                nodes.append(node)

        return nodes, total

    def get_rightmost_leaf_for_document(
        self, document_id: str | None
    ) -> TreeNode | None:
        with self.SessionLocal() as session:
            query = session.query(PostgresTreeNode).filter(PostgresTreeNode.height == 0)
            if document_id is not None:
                query = query.filter(PostgresTreeNode.document_id == document_id)

            node = query.order_by(PostgresTreeNode.span_end.desc()).limit(1).first()
            if not node:
                return None

            self._force_load_and_detach(session, node)
            self.cache_manager.put(node.id, node)
            return node

    # jscpd:ignore-start - Same SQL for both SQLite and Postgres (standard window functions)
    # Static SQL queries to avoid f-string interpolation
    _SQL_RECENT_LEAVES_ALL_DOCS = text(
        """
        SELECT * FROM (
            SELECT *,
                SUM(token_count) OVER (
                    ORDER BY span_end DESC
                    ROWS UNBOUNDED PRECEDING
                ) as cumsum
            FROM tree_nodes
            WHERE height = 0
        ) sub
        WHERE cumsum - token_count < :budget
        ORDER BY span_start ASC
    """
    )

    _SQL_RECENT_LEAVES_ONE_DOC = text(
        """
        SELECT * FROM (
            SELECT *,
                SUM(token_count) OVER (
                    ORDER BY span_end DESC
                    ROWS UNBOUNDED PRECEDING
                ) as cumsum
            FROM tree_nodes
            WHERE height = 0 AND document_id = :doc_id
        ) sub
        WHERE cumsum - token_count < :budget
        ORDER BY span_start ASC
    """
    )

    def get_recent_leaves_within_budget(
        self, document_id: str | None, token_budget: int
    ) -> list[TreeNode]:
        """Get most recent leaves (by span_end) that fit within token budget.

        Uses a window function to compute cumulative token sum and stops
        when budget is exceeded. Returns leaves in span order (ascending).
        """
        if token_budget <= 0:
            return []

        with self.SessionLocal() as session:
            if document_id is not None:
                query = self._SQL_RECENT_LEAVES_ONE_DOC
                params: dict[str, object] = {
                    "budget": token_budget,
                    "doc_id": document_id,
                }
            else:
                query = self._SQL_RECENT_LEAVES_ALL_DOCS
                params = {"budget": token_budget}

            rows = session.execute(query, params).fetchall()
            if not rows:
                return []

            # Map rows to TreeNode objects
            node_ids = [row.id for row in rows]
            nodes = self.get_nodes(node_ids)
            # Sort by span_start to return in document order
            nodes.sort(key=lambda n: n.span_start)
            return nodes

    # jscpd:ignore-end

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node.

        Args:
            node_id: Node ID to update access info
        """
        with self.SessionLocal() as session:
            node = session.query(PostgresTreeNode).filter_by(id=node_id).first()
            if node:
                node.last_accessed = datetime.utcnow()
                node.access_count += 1
                session.commit()

                # Update cache if present
                cached = self.cache_manager.get(node_id)
                if cached:
                    cached.last_accessed = node.last_accessed
                    cached.access_count = node.access_count

    def delete_nodes(
        self,
        node_ids: Sequence[str],
        *,
        session: Session | None = None,
    ) -> None:
        if not node_ids:
            return

        db_session, should_commit = self._get_session(session)
        try:
            db_session.execute(
                sa_delete(PostgresTreeNode).where(PostgresTreeNode.id.in_(node_ids))
            )
            if should_commit:
                db_session.commit()
            for node_id in node_ids:
                self.cache_manager.invalidate(node_id)
        finally:
            if should_commit:
                db_session.close()

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        """Get all pinned nodes up to optional max depth.

        Args:
            depth_max: Maximum depth for pinned nodes (optional)

        Returns:
            List of pinned PostgresTreeNode objects
        """
        with self.SessionLocal() as session:
            query = session.query(PostgresTreeNode).filter(
                PostgresTreeNode.is_pinned == 1
            )
            db_nodes = query.all()

            # Force load and detach all
            out: list[TreeNode] = []
            for node in db_nodes:
                self._force_load_and_detach(session, node)
                out.append(node)

            return out

    def pin_node(self, node_id: str) -> None:
        """Pin a node (mark as important).

        Args:
            node_id: Node ID to pin
        """
        with self.SessionLocal() as session:
            session.execute(
                update(PostgresTreeNode)
                .where(PostgresTreeNode.id == node_id)
                .values(is_pinned=1)
            )
            session.commit()

            # Update cache if present
            cached = self.cache_manager.get(node_id)
            if cached:
                cached.is_pinned = 1

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes with no children).

        Returns:
            List of leaf PostgresTreeNode objects
        """
        with self.SessionLocal() as session:
            db_nodes = (
                session.query(PostgresTreeNode)
                .filter(
                    PostgresTreeNode.left_child_id.is_(None),
                    PostgresTreeNode.right_child_id.is_(None),
                )
                .all()
            )

            # Force load and detach all
            out: list[TreeNode] = []
            for node in db_nodes:
                self._force_load_and_detach(session, node)
                out.append(node)

            return out

    def count_leaves_for_document(self, document_id: str | None) -> int:
        """Return count of leaf nodes for a document (fast COUNT(*))"""
        with self.SessionLocal() as session:
            q = session.query(func.count(PostgresTreeNode.id)).filter(
                PostgresTreeNode.left_child_id.is_(None),
                PostgresTreeNode.right_child_id.is_(None),
            )
            if document_id:
                q = q.filter(PostgresTreeNode.document_id == document_id)
            return int(q.scalar() or 0)

    def max_height_for_document(self, document_id: str | None) -> int:
        """Return maximum node height for a document (fast MAX(height))"""
        with self.SessionLocal() as session:
            q = session.query(func.max(PostgresTreeNode.height))
            if document_id:
                q = q.filter(PostgresTreeNode.document_id == document_id)
            return int(q.scalar() or 0)

    def get_pinned_nodes_for_document(
        self, document_id: str, depth_max: int | None = None
    ) -> list[TreeNode]:
        """Return pinned nodes filtered by a specific document."""
        with self.SessionLocal() as session:
            q = session.query(PostgresTreeNode).filter(
                PostgresTreeNode.is_pinned == 1,
                PostgresTreeNode.document_id == document_id,
            )
            db_nodes = q.all()
            out: list[TreeNode] = []
            for node in db_nodes:
                self._force_load_and_detach(session, node)
                out.append(node)
            return out

    def get_parentless_nodes_for_document(
        self, document_id: str | None
    ) -> list[TreeNode]:
        with self.SessionLocal() as session:
            query = session.query(PostgresTreeNode).filter(
                PostgresTreeNode.parent_id.is_(None)
            )
            if document_id is None:
                query = query.filter(PostgresTreeNode.document_id.is_(None))
            else:
                query = query.filter(PostgresTreeNode.document_id == document_id)

            db_nodes = query.order_by(
                PostgresTreeNode.height,
                PostgresTreeNode.level_index,
                PostgresTreeNode.span_start,
            ).all()

            out: list[TreeNode] = []
            for node in db_nodes:
                self._force_load_and_detach(session, node)
                self.cache_manager.put(node.id, node)
                out.append(node)
            return out

    def get_ready_left_children(self, document_id: str | None) -> list[str]:
        if not document_id:
            return []
        with self.SessionLocal() as session:
            doc_span_end = (
                session.query(func.max(PostgresTreeNode.span_end))
                .filter(PostgresTreeNode.document_id == document_id)
                .scalar()
            )
            if doc_span_end is None:
                return []

            query = (
                session.query(PostgresTreeNode.id)
                .filter(PostgresTreeNode.document_id == document_id)
                .filter(PostgresTreeNode.parent_id.is_(None))
                .filter(func.mod(PostgresTreeNode.level_index, 2) == 0)
                .filter(
                    or_(
                        PostgresTreeNode.span_start > 0,
                        PostgresTreeNode.span_end < doc_span_end,
                    )
                )
                .filter(
                    or_(
                        PostgresTreeNode.span_start == 0,
                        PostgresTreeNode.preceding_neighbor_id.isnot(None),
                    )
                )
                .filter(
                    or_(
                        PostgresTreeNode.span_end == doc_span_end,
                        PostgresTreeNode.following_neighbor_id.isnot(None),
                    )
                )
                .order_by(PostgresTreeNode.height, PostgresTreeNode.level_index)
            )

            return [str(row[0]) for row in query.all()]

    def get_node_by_height_and_level(
        self,
        document_id: str | None,
        height: int,
        level_index: int,
    ) -> TreeNode | None:
        with self.SessionLocal() as session:
            query = session.query(PostgresTreeNode).filter(
                PostgresTreeNode.height == height,
                PostgresTreeNode.level_index == level_index,
            )
            if document_id is not None:
                query = query.filter(PostgresTreeNode.document_id == document_id)
            node = query.first()
            if node is None:
                return None
            self._force_load_and_detach(session, node)
            return node

    def get_nodes_by_height_levels(
        self,
        document_id: str | None,
        coordinates: Sequence[tuple[int, int]],
    ) -> list[TreeNode]:
        if not coordinates:
            return []

        with self.SessionLocal() as session:
            tuple_expr = tuple_(
                PostgresTreeNode.height,
                PostgresTreeNode.level_index,
            )

            query = session.query(PostgresTreeNode).filter(tuple_expr.in_(coordinates))
            if document_id is not None:
                query = query.filter(PostgresTreeNode.document_id == document_id)

            db_nodes = query.all()

            out: list[TreeNode] = []
            for node in db_nodes:
                self._force_load_and_detach(session, node)
                self.cache_manager.put(node.id, node)
                out.append(node)
            return out

    def count_pinned_for_document(self, document_id: str | None) -> int:
        """Return count of pinned nodes for a document."""
        with self.SessionLocal() as session:
            q = session.query(func.count(PostgresTreeNode.id)).filter(
                PostgresTreeNode.is_pinned == 1
            )
            if document_id:
                q = q.filter(PostgresTreeNode.document_id == document_id)
            return int(q.scalar() or 0)

    def get_all_nodes_for_document_paginated(
        self, document_id: str | None, *, page_size: int = 1000
    ) -> list[list[TreeNode]]:
        """Get all nodes for a document in paginated batches for memory efficiency.

        This method is optimized for large documents with tens of thousands of nodes.
        It yields batches of nodes rather than loading all at once.

        Args:
            document_id: Document ID to get nodes for
            page_size: Number of nodes per batch (default 1000)

        Returns:
            List of batches, where each batch is a list of PostgresTreeNode objects

        Note:
            For small documents (<5000 nodes), get_all_nodes_for_document() is more efficient.
            This method is designed for very large documents that would cause memory issues.
        """
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        batches: list[list[TreeNode]] = []
        offset = 0

        with self.SessionLocal() as session:
            while True:
                # Query one batch at a time
                if document_id:
                    query = (
                        session.query(PostgresTreeNode)
                        .filter_by(document_id=document_id)
                        .offset(offset)
                        .limit(page_size)
                    )
                else:
                    logger.warning(
                        "No document_id provided for paginated query. This will process all nodes."
                    )
                    query = (
                        session.query(PostgresTreeNode).offset(offset).limit(page_size)
                    )

                db_batch = query.all()

                if not db_batch:
                    break  # No more nodes

                # Force load and detach all nodes in this batch
                current: list[TreeNode] = []
                for node in db_batch:
                    self._force_load_and_detach(session, node)
                    current.append(node)

                batches.append(current)
                offset += page_size

                # Log progress for very large documents
                if len(batches) % 10 == 0:  # Every 10 batches
                    total_processed = len(batches) * page_size
                    logger.debug(
                        f"Processed {total_processed} nodes in {len(batches)} batches for document {document_id}"
                    )

        return batches

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        """Get all nodes for a specific document.

        Args:
            document_id: Document ID to get nodes for

        Returns:
            List of PostgresTreeNode objects for the document
        """
        with self.SessionLocal() as session:
            if document_id:
                db_nodes = (
                    session.query(PostgresTreeNode)
                    .filter_by(document_id=document_id)
                    .all()
                )
            else:
                # If no document_id, return all nodes (but this could be memory intensive)
                logger.warning("No document_id provided, returning all nodes in store.")
                db_nodes = session.query(PostgresTreeNode).all()

            # Force load and detach all
            out: list[TreeNode] = []
            for node in db_nodes:
                self._force_load_and_detach(session, node)
                out.append(node)

            return out

    def count_nodes_for_document(self, document_id: str | None) -> int:
        """Return count of nodes for the given document (fast COUNT(*))"""
        with self.SessionLocal() as session:
            if document_id:
                count_val = (
                    session.query(func.count(PostgresTreeNode.id))
                    .filter_by(document_id=document_id)
                    .scalar()
                )
            else:
                count_val = session.query(func.count(PostgresTreeNode.id)).scalar()
            return int(count_val or 0)
