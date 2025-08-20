"""Repository for TreeNode CRUD operations using PostgreSQL with pgvector."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import update

from ragzoom.models import TreeNode

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
from ragzoom.repositories.base_repository import BaseRepository
from ragzoom.services.cache_manager import CacheManager
from ragzoom.storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class NodeRepository(BaseRepository):
    """Repository for TreeNode database operations."""

    def __init__(
        self, database_manager: DatabaseManager, cache_manager: CacheManager[TreeNode]
    ):
        """Initialize node repository.

        Args:
            database_manager: Database manager for DB operations
            cache_manager: Cache manager for hot nodes
        """
        self.db_manager = database_manager
        self.cache_manager = cache_manager
        self.SessionLocal = database_manager.SessionLocal

    def _force_load_and_detach(self, session: Any, node: TreeNode) -> None:
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
            node.embedding,  # Load embedding too
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
    ) -> TreeNode:
        """Add a node to the database with its embedding.

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

        Returns:
            Created TreeNode
        """
        # Validate embedding dimension
        self.db_manager.validate_embedding_dimension(embedding)

        with self.SessionLocal() as session:
            node = TreeNode(
                id=node_id,
                parent_id=parent_id,
                left_child_id=left_child_id,
                right_child_id=right_child_id,
                span_start=span_start,
                span_end=span_end,
                text=text,
                embedding=list(map(float, embedding)),  # Store embedding in DB
                document_id=document_id,
                token_count=token_count,
                height=height,
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
        self, nodes_data: list[dict[str, Any]], *, session: Optional["Session"] = None
    ) -> list[TreeNode]:
        """Add multiple nodes to the database in batch.

        Args:
            nodes_data: List of dictionaries containing node data
            session: Optional database session for transactional operations

        Returns:
            List of created TreeNode objects
        """
        if not nodes_data:
            return []

        # Validate all embeddings first
        for data in nodes_data:
            self.db_manager.validate_embedding_dimension(data["embedding"])

        db_session, should_commit = self._get_session(session)
        try:
            # Create TreeNode objects for regular session.add_all()
            nodes = []
            for data in nodes_data:
                node = TreeNode(
                    id=data["node_id"],
                    parent_id=data.get("parent_id"),
                    left_child_id=data.get("left_child_id"),
                    right_child_id=data.get("right_child_id"),
                    span_start=data["span_start"],
                    span_end=data["span_end"],
                    text=data["text"],
                    embedding=list(
                        map(float, data["embedding"])
                    ),  # Store embedding in DB
                    document_id=data.get("document_id"),
                    token_count=data.get("token_count", 0),
                    preceding_neighbor_id=data.get("preceding_neighbor_id"),
                    height=data.get("height", 0),
                )
                nodes.append(node)

            # Use add_all for proper object tracking and session management
            db_session.add_all(nodes)
            if should_commit:
                db_session.commit()

            # Force load and detach all nodes
            for node in nodes:
                db_session.refresh(node)
                self._force_load_and_detach(db_session, node)

            # Add all to cache
            for node in nodes:
                self.cache_manager.put(node.id, node)

            return nodes
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
                    update(TreeNode)
                    .where(TreeNode.id == node_id)
                    .values(parent_id=parent_id)
                )

            # Invalidate cache for updated nodes
            for node_id, _ in updates:
                self.cache_manager.invalidate(node_id)

    def get_node(self, node_id: str) -> TreeNode | None:
        """Get a node by ID.

        Args:
            node_id: Node ID to retrieve

        Returns:
            TreeNode if found, None otherwise
        """
        # Check cache first
        cached = self.cache_manager.get(node_id)
        if cached:
            return cached

        # Load from database
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
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
            List of TreeNode objects found
        """
        if not node_ids:
            return []

        nodes = []
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
                    session.query(TreeNode).filter(TreeNode.id.in_(uncached_ids)).all()
                )
                for node in db_nodes:
                    # Force load and detach
                    self._force_load_and_detach(session, node)
                    # Add to cache
                    self.cache_manager.put(node.id, node)
                    nodes.append(node)

        return nodes

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node.

        Args:
            node_id: Node ID to update access info
        """
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
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
            List of pinned TreeNode objects
        """
        with self.SessionLocal() as session:
            query = session.query(TreeNode).filter(TreeNode.is_pinned == 1)
            nodes = query.all()

            # Force load and detach all
            for node in nodes:
                self._force_load_and_detach(session, node)

            # Filter by depth if specified
            if depth_max is not None:
                # Calculate depth for each node and filter
                # This is done post-query for simplicity
                filtered_nodes = []
                for node in nodes:
                    depth = self._calculate_depth(node.id)
                    if depth <= depth_max:
                        filtered_nodes.append(node)
                return filtered_nodes

            return nodes

    def _calculate_depth(self, node_id: str) -> int:
        """Calculate depth of a node from root.

        Args:
            node_id: Node ID to calculate depth for

        Returns:
            Depth from root (0 for root nodes)
        """
        depth = 0
        current_id = node_id

        while current_id:
            node = self.get_node(current_id)
            if not node or not node.parent_id:
                break
            depth += 1
            current_id = node.parent_id

        return depth

    def pin_node(self, node_id: str) -> None:
        """Pin a node (mark as important).

        Args:
            node_id: Node ID to pin
        """
        with self.SessionLocal() as session:
            session.execute(
                update(TreeNode).where(TreeNode.id == node_id).values(is_pinned=1)
            )
            session.commit()

            # Update cache if present
            cached = self.cache_manager.get(node_id)
            if cached:
                cached.is_pinned = 1

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes with no children).

        Returns:
            List of leaf TreeNode objects
        """
        with self.SessionLocal() as session:
            nodes = (
                session.query(TreeNode)
                .filter(
                    TreeNode.left_child_id.is_(None),
                    TreeNode.right_child_id.is_(None),
                )
                .all()
            )

            # Force load and detach all
            for node in nodes:
                self._force_load_and_detach(session, node)

            return nodes

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
            List of batches, where each batch is a list of TreeNode objects

        Note:
            For small documents (<5000 nodes), get_all_nodes_for_document() is more efficient.
            This method is designed for very large documents that would cause memory issues.
        """
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        batches = []
        offset = 0

        with self.SessionLocal() as session:
            while True:
                # Query one batch at a time
                if document_id:
                    query = (
                        session.query(TreeNode)
                        .filter_by(document_id=document_id)
                        .offset(offset)
                        .limit(page_size)
                    )
                else:
                    logger.warning(
                        "No document_id provided for paginated query. This will process all nodes."
                    )
                    query = session.query(TreeNode).offset(offset).limit(page_size)

                batch = query.all()

                if not batch:
                    break  # No more nodes

                # Force load and detach all nodes in this batch
                for node in batch:
                    self._force_load_and_detach(session, node)

                batches.append(batch)
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
            List of TreeNode objects for the document
        """
        with self.SessionLocal() as session:
            if document_id:
                nodes = session.query(TreeNode).filter_by(document_id=document_id).all()
            else:
                # If no document_id, return all nodes (but this could be memory intensive)
                logger.warning("No document_id provided, returning all nodes in store.")
                nodes = session.query(TreeNode).all()

            # Force load and detach all
            for node in nodes:
                self._force_load_and_detach(session, node)

            return nodes
