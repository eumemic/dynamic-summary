"""Repository for TreeNode CRUD operations."""

import logging
from datetime import datetime
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import update

from ragzoom.models import TreeNode
from ragzoom.services.cache_manager import CacheManager
from ragzoom.storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class NodeRepository:
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
        """Add a node to both SQLite and Chroma.

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

        # Add to Chroma
        embedding_array = np.array(embedding, dtype=np.float32)
        self.db_manager.collection.add(
            ids=[node_id],
            embeddings=cast(Any, [embedding_array]),
            metadatas=[
                {
                    "span_start": span_start,
                    "span_end": span_end,
                    "parent_id": parent_id or "",
                    "is_leaf": (
                        1 if (left_child_id is None and right_child_id is None) else 0
                    ),
                    "document_id": document_id or "",
                }
            ],
            documents=[text],
        )

        return node

    def add_nodes_batch(self, nodes_data: list[dict[str, Any]]) -> list[TreeNode]:
        """Add multiple nodes to both SQLite and Chroma in batch.

        Args:
            nodes_data: List of dictionaries containing node data

        Returns:
            List of created TreeNode objects
        """
        if not nodes_data:
            return []

        # Validate all embeddings first
        for data in nodes_data:
            self.db_manager.validate_embedding_dimension(data["embedding"])

        nodes = []
        with self.SessionLocal() as session:
            # Create TreeNode objects
            for data in nodes_data:
                node = TreeNode(
                    id=data["node_id"],
                    parent_id=data.get("parent_id"),
                    left_child_id=data.get("left_child_id"),
                    right_child_id=data.get("right_child_id"),
                    span_start=data["span_start"],
                    span_end=data["span_end"],
                    text=data["text"],
                    document_id=data.get("document_id"),
                    token_count=data.get("token_count", 0),
                    preceding_neighbor_id=data.get("preceding_neighbor_id"),
                    height=data.get("height", 0),
                )
                nodes.append(node)

            # Add all nodes to session
            for node in nodes:
                session.add(node)
            session.commit()

            # Force load attributes and detach all nodes
            for node in nodes:
                self._force_load_and_detach(session, node)

            # Add all to cache
            for node in nodes:
                self.cache_manager.put(node.id, node)

        # Batch add to Chroma
        if nodes:
            ids = []
            embeddings = []
            metadatas = []
            documents = []

            for data, node in zip(nodes_data, nodes):
                ids.append(data["node_id"])
                embeddings.append(np.array(data["embedding"], dtype=np.float32))
                metadatas.append(
                    {
                        "span_start": int(data["span_start"]),
                        "span_end": int(data["span_end"]),
                        "parent_id": data.get("parent_id", ""),
                        "is_leaf": int(
                            1
                            if (
                                data.get("left_child_id") is None
                                and data.get("right_child_id") is None
                            )
                            else 0
                        ),
                        "document_id": data.get("document_id", ""),
                    }
                )
                documents.append(data["text"])

            self.db_manager.collection.add(
                ids=ids,
                embeddings=cast(Any, embeddings),
                metadatas=cast(Any, metadatas),
                documents=documents,
            )

        return nodes

    def update_parent_references_batch(self, updates: list[tuple[str, str]]) -> None:
        """Update parent references for multiple nodes in batch.

        Args:
            updates: List of (child_id, parent_id) tuples
        """
        if not updates:
            return

        with self.SessionLocal() as session:
            # Build update mappings
            update_mappings = [
                {"id": child_id, "parent_id": parent_id}
                for child_id, parent_id in updates
            ]

            # Bulk update
            for mapping in update_mappings:
                stmt = (
                    update(TreeNode)
                    .where(TreeNode.id == mapping["id"])
                    .values(parent_id=mapping["parent_id"])
                )
                session.execute(stmt)
            session.commit()

            # Invalidate cache for updated nodes
            for child_id, _ in updates:
                self.cache_manager.remove(child_id)

    def get_node(self, node_id: str) -> TreeNode | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier

        Returns:
            TreeNode if found, None otherwise
        """
        # Check cache first
        cached = self.cache_manager.get(node_id)
        if cached:
            return cached

        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                # Force load all attributes and detach from session
                self._force_load_and_detach(session, node)
                self.cache_manager.put(node_id, node)
            return node

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes by their IDs.

        Args:
            node_ids: List of node identifiers

        Returns:
            List of TreeNodes found
        """
        # First, try to get as many as possible from the cache
        cached_nodes: list[TreeNode] = []
        cached_ids = set()

        for node_id in node_ids:
            cached_node = self.cache_manager.get(node_id)
            if cached_node is not None:
                cached_nodes.append(cached_node)
                cached_ids.add(node_id)

        # Then, get the rest from the database
        ids_to_fetch = [nid for nid in node_ids if nid not in cached_ids]

        db_nodes = []
        if ids_to_fetch:
            with self.SessionLocal() as session:
                db_nodes = (
                    session.query(TreeNode).filter(TreeNode.id.in_(ids_to_fetch)).all()
                )
                for node in db_nodes:
                    # Force load all attributes and detach from session
                    self._force_load_and_detach(session, node)
                    self.cache_manager.put(node.id, node)

        return cached_nodes + db_nodes

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node.

        Args:
            node_id: Node identifier
        """
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                node.last_accessed = datetime.utcnow()
                node.access_count += 1
                session.commit()

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        """Get all pinned nodes up to optional max depth.

        Args:
            depth_max: Maximum depth to search (optional)

        Returns:
            List of pinned TreeNodes
        """
        with self.SessionLocal() as session:
            query = session.query(TreeNode).filter(TreeNode.is_pinned == 1)
            nodes = query.all()
            for node in nodes:
                session.expunge(node)
            return nodes

    def pin_node(self, node_id: str) -> bool:
        """Pin a node (mark as important).

        Args:
            node_id: Node identifier

        Returns:
            True if node was pinned, False if not found
        """
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if not node:
                return False

            node.is_pinned = 1
            session.commit()

            # Update cache if node is cached
            if self.cache_manager.contains(node_id):
                self.cache_manager.put(node_id, node)

            return True

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes with no children).

        Returns:
            List of leaf TreeNodes
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
            for node in nodes:
                session.expunge(node)
            return nodes

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        """Get all nodes for a document.

        Args:
            document_id: Document identifier (None for global nodes)

        Returns:
            List of TreeNodes for the document
        """
        with self.SessionLocal() as session:
            if document_id is None:
                nodes = (
                    session.query(TreeNode).filter(TreeNode.document_id.is_(None)).all()
                )
            else:
                nodes = (
                    session.query(TreeNode)
                    .filter(TreeNode.document_id == document_id)
                    .all()
                )
            for node in nodes:
                session.expunge(node)
            return nodes
