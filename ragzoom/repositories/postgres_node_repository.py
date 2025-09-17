"""Repository for PostgresTreeNode CRUD operations using PostgreSQL with pgvector."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional, TypedDict, cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import func, update

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

    def _force_load_and_detach(
        self, session: "Session", node: PostgresTreeNode
    ) -> None:
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
            node.height,
        )
        session.expunge(node)

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
            path = ""
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
                path=path,
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

    def add_nodes_batch(
        self,
        nodes_data: list[dict[str, object]],
        *,
        session: Optional["Session"] = None,
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
                    token_count=cast(int, data.get("token_count", 0)),
                    preceding_neighbor_id=cast(
                        str | None, data.get("preceding_neighbor_id")
                    ),
                    height=cast(int, data.get("height", 0)),
                    path="",
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

    def update_parent_references_batch(
        self, updates: list[tuple[str, str]], *, session: Optional["Session"] = None
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

    def update_node_paths_from_tree_structure(
        self, *, session: Optional["Session"] = None
    ) -> None:
        """Update node paths based on the current tree structure.

        This method should be called after tree construction is complete to ensure
        all nodes have correct paths assigned based on their parent-child relationships.

        Args:
            session: Optional database session for transactional operations
        """
        with self._session_scope(session) as db_session:
            # Get all nodes and build the tree structure
            nodes = db_session.query(PostgresTreeNode).all()

            # Find root nodes (nodes with no parent)
            root_nodes = [node for node in nodes if node.is_root()]

            # Update paths starting from root nodes
            for root in root_nodes:
                self._update_node_path_recursive(root, "", db_session, set())

    def _update_node_path_recursive(
        self, node: "PostgresTreeNode", path: str, session: "Session", visited: set[str]
    ) -> None:
        """Recursively update node paths in the tree.

        Args:
            node: Current node to update
            path: Path to assign to this node
            session: Database session
            visited: Set of visited node IDs to prevent infinite loops
        """
        if node.id in visited:
            return
        visited.add(node.id)

        # Update this node's path
        session.execute(
            update(PostgresTreeNode)
            .where(PostgresTreeNode.id == node.id)
            .values(path=path)
        )

        # Invalidate cache
        self.cache_manager.invalidate(node.id)

        # Update children
        if node.left_child_id:
            left_child = (
                session.query(PostgresTreeNode).filter_by(id=node.left_child_id).first()
            )
            if left_child:
                self._update_node_path_recursive(
                    left_child, path + "0", session, visited
                )

        if node.right_child_id:
            right_child = (
                session.query(PostgresTreeNode)
                .filter_by(id=node.right_child_id)
                .first()
            )
            if right_child:
                self._update_node_path_recursive(
                    right_child, path + "1", session, visited
                )

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

            # Use database-level path filtering for better performance if depth_max specified
            if depth_max is not None:
                query = query.filter(func.length(PostgresTreeNode.path) <= depth_max)

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
