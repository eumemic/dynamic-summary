"""Storage layer for RagZoom - SQLite for tree structure, Chroma for vectors."""

import hashlib
import logging
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime
from typing import Any, cast

import chromadb
import numpy as np
from chromadb.config import Settings
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from ragzoom.config import OperationalConfig

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TreeNode(Base):
    """SQLite model for tree nodes."""

    __tablename__ = "tree_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("tree_nodes.id"), nullable=True
    )
    left_child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    right_child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    span_start: Mapped[int] = mapped_column(Integer, nullable=False)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # NULL for leaf nodes
    token_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )  # Token count of text/summary
    is_pinned: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    document_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("documents.id"), nullable=True
    )


class Document(Base):
    """SQLite model for documents."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    file_path: Mapped[str | None] = mapped_column(
        String, nullable=True, unique=True
    )  # Path to the source file
    content_hash: Mapped[str] = mapped_column(
        String, nullable=False
    )  # SHA256 hash of content
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    summary_model: Mapped[str] = mapped_column(String, nullable=False)


class Store:
    """Combined storage for tree structure (SQLite) and embeddings (Chroma)."""

    # Class constant for pin depth limit (dormant feature)
    PIN_DEPTH_MAX = 2

    def __init__(
        self, config: OperationalConfig, embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize storage backends.

        Args:
            config: Operational configuration with storage paths
            embedding_model: Name of embedding model (for dimension validation)
        """
        self.config = config
        self.embedding_model = embedding_model

        # Initialize SQLite
        self.engine = create_engine(
            config.sqlite_database_url, connect_args={"check_same_thread": False}
        )

        # Handle migration before creating tables with new schema
        self._run_migrations()

        # Create all tables (will only create missing ones)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

        # Initialize Chroma
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="ragzoom_nodes",
            metadata={"hnsw:space": "cosine"},
        )

        # LRU cache for hot nodes
        self.node_cache: dict[str, TreeNode] = {}
        self.cache_order: deque[str] = deque(maxlen=config.cache_size)

        # Cache expected embedding dimension for validation
        self._expected_embedding_dim = self._get_expected_embedding_dimension()

    def _get_from_cache(self, node_id: str) -> TreeNode | None:
        """Get node from cache if available."""
        if node_id in self.node_cache:
            # Move to end (most recently used)
            self.cache_order.remove(node_id)
            self.cache_order.append(node_id)
            return self.node_cache[node_id]
        return None

    def _add_to_cache(self, node: TreeNode) -> None:
        """Add node to cache, evicting LRU if necessary."""
        if node.id in self.node_cache:
            self.cache_order.remove(node.id)
        elif (
            self.cache_order.maxlen and len(self.cache_order) >= self.cache_order.maxlen
        ):
            # Evict LRU
            lru_id = self.cache_order.popleft()
            del self.node_cache[lru_id]

        self.node_cache[node.id] = node
        self.cache_order.append(node.id)

    def _get_expected_embedding_dimension(self) -> int | None:
        """Get expected embedding dimension from existing data.

        We no longer maintain hardcoded dimension info since OpenAI API
        is the source of truth. This method tries to infer from existing data.
        """
        # Try to infer from existing data
        try:
            # Get any existing embedding from collection
            results = self.collection.peek(limit=1)
            embeddings = results.get("embeddings")
            if (
                embeddings is not None
                and isinstance(embeddings, list)
                and len(embeddings) > 0
            ):
                return len(embeddings[0])
        except Exception as e:
            logger.debug(f"Could not infer embedding dimension: {e}")

        # If no existing data, don't enforce validation
        # This allows tests and first-time setups to work with any dimension
        return None

    def _validate_embedding_dimension(
        self, embedding: list[float] | np.ndarray
    ) -> None:
        """Validate embedding dimension matches expected."""
        if not embedding:
            raise ValueError("Embedding cannot be empty")

        actual_dim = len(embedding)

        # If we don't have an expected dimension yet, use this as the reference
        if self._expected_embedding_dim is None:
            self._expected_embedding_dim = actual_dim
            logger.debug(f"Setting embedding dimension reference to {actual_dim}")
        elif actual_dim != self._expected_embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._expected_embedding_dim}, "
                f"got {actual_dim}. Check embedding_model configuration."
            )

    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: list[float] | np.ndarray,
        span_start: int,
        span_end: int,
        parent_id: str | None = None,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
        summary: str | None = None,
        document_id: str | None = None,
        token_count: int | None = None,
    ) -> TreeNode:
        """Add a node to both SQLite and Chroma."""
        # Validate embedding dimension
        self._validate_embedding_dimension(embedding)

        with self.SessionLocal() as session:
            node = TreeNode(
                id=node_id,
                parent_id=parent_id,
                left_child_id=left_child_id,
                right_child_id=right_child_id,
                span_start=span_start,
                span_end=span_end,
                text=text,
                summary=summary,
                document_id=document_id,
                token_count=token_count,
            )
            session.add(node)
            session.commit()

            # Add to cache
            self._add_to_cache(node)

        # Add to Chroma
        # Convert to numpy array if needed
        embedding_array = np.array(embedding, dtype=np.float32)
        self.collection.add(
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
                    "document_id": document_id
                    or "",  # ChromaDB doesn't accept None values
                }
            ],
            documents=[text],
        )

        return node

    def get_node(self, node_id: str) -> TreeNode | None:
        """Get a node by ID."""
        # Check cache first
        cached = self._get_from_cache(node_id)
        if cached:
            return cached

        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                self._add_to_cache(node)
            return node

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes by their IDs."""
        # First, try to get as many as possible from the cache
        cached_nodes: list[TreeNode] = [
            node for nid in node_ids if (node := self._get_from_cache(nid)) is not None
        ]
        cached_ids = {node.id for node in cached_nodes}

        # Then, get the rest from the database
        ids_to_fetch = [nid for nid in node_ids if nid not in cached_ids]

        db_nodes = []
        if ids_to_fetch:
            with self.SessionLocal() as session:
                db_nodes = (
                    session.query(TreeNode).filter(TreeNode.id.in_(ids_to_fetch)).all()
                )
                for node in db_nodes:
                    self._add_to_cache(node)

        return cached_nodes + db_nodes

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node."""
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                node.last_accessed = datetime.utcnow()
                node.access_count += 1
                session.commit()

    def get_children(self, node_id: str) -> tuple[TreeNode | None, TreeNode | None]:
        """Get left and right children of a node."""
        node = self.get_node(node_id)
        if not node:
            return None, None

        left = self.get_node(node.left_child_id) if node.left_child_id else None
        right = self.get_node(node.right_child_id) if node.right_child_id else None
        return left, right

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Get all ancestors of given nodes using batch loading for efficiency."""
        all_ancestors = set()
        current_level = set(node_ids)

        # Keep going until we've reached all roots
        while current_level:
            # Batch load all nodes at current level
            nodes_at_level = self.get_nodes(list(current_level))

            # Collect parent IDs for next level
            next_level = set()
            for node in nodes_at_level:
                if node.parent_id and node.parent_id not in all_ancestors:
                    all_ancestors.add(node.parent_id)
                    next_level.add(node.parent_id)

            # Move up to next level
            current_level = next_level

        # Batch load all ancestors and return
        if all_ancestors:
            return self.get_nodes(list(all_ancestors))
        return []

    def search_similar(
        self,
        query_embedding: list[float] | np.ndarray,
        n_results: int,
        where: dict | None = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar nodes using Chroma.

        Returns list of (id, similarity, metadata) tuples where similarity is in [0, 1].
        """
        # Convert to numpy array if needed
        query_array = np.array(query_embedding, dtype=np.float32)
        results = self.collection.query(
            query_embeddings=cast(Any, [query_array]),
            n_results=n_results,
            where=where,
        )

        # Return list of (id, similarity, metadata) tuples
        output = []
        ids = results.get("ids")
        distances = results.get("distances")
        metadatas = results.get("metadatas")

        if ids and distances and metadatas and len(ids) > 0:
            for i in range(len(ids[0])):
                # Convert cosine distance to similarity
                # Cosine distance ranges from 0 to 2, where 0 is identical
                # Similarity = 1 - (distance / 2) to map to [0, 1]
                distance = float(distances[0][i])
                similarity = 1.0 - (distance / 2.0)
                # Ensure similarity is in valid range [0, 1]
                similarity = max(0.0, min(1.0, similarity))

                output.append(
                    (
                        ids[0][i],
                        similarity,
                        (
                            dict(metadatas[0][i])
                            if isinstance(metadatas[0][i], dict)
                            else {}
                        ),
                    )
                )

        return output

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        """Get nodes that are pinned (always included in coverage)."""
        with self.SessionLocal() as session:
            pinned_nodes = session.query(TreeNode).filter_by(is_pinned=1).all()

            if depth_max is not None:
                # Filter by calculated depth
                filtered_nodes = []
                for node in pinned_nodes:
                    if self.get_node_depth(node.id) <= depth_max:
                        filtered_nodes.append(node)
                return filtered_nodes

            return pinned_nodes

    def pin_node(self, node_id: str) -> bool:
        """Pin a node if it's within allowed depth."""
        node = self.get_node(node_id)
        if not node:
            return False

        node_depth = self.get_node_depth(node_id)
        if node_depth > self.PIN_DEPTH_MAX:
            return False

        # Check if already pinned
        if node.is_pinned == 1:
            logger.info(f"Node {node_id} is already pinned")
            return False

        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                node.is_pinned = 1
                session.commit()
                return True
        return False

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes without children)."""
        with self.SessionLocal() as session:
            return (
                session.query(TreeNode)
                .filter(
                    TreeNode.left_child_id.is_(None), TreeNode.right_child_id.is_(None)
                )
                .all()
            )

    def get_root_node(self) -> TreeNode | None:
        """Get the root node (node with no parent)."""
        with self.SessionLocal() as session:
            return session.query(TreeNode).filter_by(parent_id=None).first()

    def get_root_node_for_document(self, document_id: str | None) -> TreeNode | None:
        """Get the root node for a specific document."""
        with self.SessionLocal() as session:
            query = session.query(TreeNode).filter_by(parent_id=None)
            if document_id:
                query = query.filter_by(document_id=document_id)
            return query.first()

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        """Get all nodes for a specific document."""
        with self.SessionLocal() as session:
            if document_id:
                return session.query(TreeNode).filter_by(document_id=document_id).all()
            else:
                # If no document_id, maybe return all nodes? Or raise error?
                # For now, let's return all nodes, but this could be memory intensive
                logger.warning("No document_id provided, returning all nodes in store.")
                return session.query(TreeNode).all()

    def get_node_depth(self, node_id: str) -> int:
        """Calculate depth of a node (distance from root).

        Returns 0 for root nodes, incrementing by 1 for each level down.
        This follows the standard tree convention where root is at depth 0.
        """
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        depth = 0
        current_id = node.parent_id

        while current_id:
            depth += 1
            parent = self.get_node(current_id)
            if not parent:
                break
            current_id = parent.parent_id

        return depth

    def get_node_height(self, node_id: str) -> int:
        """Calculate height of a node (distance to furthest leaf).

        Returns 0 for leaf nodes, incrementing by 1 for each level up.
        """
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        # If it's a leaf node (no children), height is 0
        if not node.left_child_id and not node.right_child_id:
            return 0

        # Otherwise, height is 1 + max height of children
        max_child_height = 0

        if node.left_child_id:
            left_height = self.get_node_height(node.left_child_id)
            max_child_height = max(max_child_height, left_height)

        if node.right_child_id:
            right_height = self.get_node_height(node.right_child_id)
            max_child_height = max(max_child_height, right_height)

        return 1 + max_child_height

    def is_leaf_node(self, node_id: str) -> bool:
        """Check if a node is a leaf (has no children)."""
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        return not node.left_child_id and not node.right_child_id

    def is_root_node(self, node_id: str) -> bool:
        """Check if a node is a root (has no parent)."""
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        return node.parent_id is None

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | np.ndarray,
        candidates: list[tuple[str, float, dict]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Apply MMR (Maximal Marginal Relevance) to get diverse results."""
        if not candidates or k <= 0:
            return []

        # Get embeddings for all candidates
        candidate_ids = [c[0] for c in candidates]

        # Batch retrieve embeddings
        results = self.collection.get(ids=candidate_ids, include=["embeddings"])

        # Create ID to embedding mapping for O(1) lookup
        embeddings = results.get("embeddings")
        ids = results.get("ids")
        if embeddings is None or ids is None:
            return []

        id_to_embedding = {ids[i]: np.array(embeddings[i]) for i in range(len(ids))}

        # Build candidate embeddings array in order
        cand_embs = np.array([id_to_embedding[cid] for cid in candidate_ids])
        query_emb = np.array(query_embedding)

        # Vectorized similarity computation
        query_sims = np.dot(cand_embs, query_emb)

        # MMR iterative selection with optimized operations
        selected_mask = np.zeros(len(candidates), dtype="bool")
        selected_indices = []

        # Select first item (highest relevance)
        first_idx = np.argmax(query_sims)
        selected_indices.append(first_idx)
        selected_mask[first_idx] = True

        # Pre-compute pairwise similarities for efficiency
        if len(candidates) > 1:
            pairwise_sims = np.dot(cand_embs, cand_embs.T)

        # Select remaining items
        for _ in range(1, min(k, len(candidates))):
            # Vectorized MMR computation for all unselected
            unselected_mask = ~selected_mask
            if not np.any(unselected_mask):
                break

            # Relevance scores for unselected
            relevances = query_sims[unselected_mask]

            # Max similarity to any selected item (vectorized)
            max_sims = (
                np.max(pairwise_sims[np.ix_(unselected_mask, selected_mask)], axis=1)
                if np.any(selected_mask)
                else np.zeros(int(np.sum(unselected_mask)))
            )

            # MMR scores
            mmr_scores = lambda_param * relevances - (1 - lambda_param) * max_sims

            # Get index in unselected subset
            best_unselected_idx = np.argmax(mmr_scores)

            # Convert to original index
            unselected_indices = np.where(unselected_mask)[0]
            best_idx = unselected_indices[best_unselected_idx]

            selected_indices.append(best_idx)
            selected_mask[best_idx] = True

        # Return selected node IDs
        return [candidates[i][0] for i in selected_indices]

    def get_document_by_path(self, file_path: str) -> Document | None:
        """Get a document by file path."""
        with self.SessionLocal() as session:
            return session.query(Document).filter_by(file_path=file_path).first()

    def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by ID."""
        with self.SessionLocal() as session:
            return session.query(Document).filter_by(id=document_id).first()

    def get_document_embedding_model(self, document_id: str) -> str | None:
        """Get the embedding model used for a specific document."""
        doc = self.get_document_by_id(document_id)
        return doc.embedding_model if doc else None

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
        summary_model: str,
    ) -> Document:
        """Add a document record."""
        with self.SessionLocal() as session:
            doc = Document(
                id=document_id,
                file_path=file_path,
                content_hash=content_hash,
                chunk_count=chunk_count,
                embedding_model=embedding_model,
                summary_model=summary_model,
            )
            session.add(doc)
            session.commit()
            return doc

    def delete_document_nodes(self, document_id: str) -> int:
        """Delete all nodes associated with a document."""
        with self.SessionLocal() as session:
            # Get all nodes for this document
            nodes = session.query(TreeNode).filter_by(document_id=document_id).all()
            node_ids = [n.id for n in nodes]

            # Delete from SQLite
            deleted_count = (
                session.query(TreeNode).filter_by(document_id=document_id).delete()
            )
            session.commit()

            # Delete from Chroma
            if node_ids:
                self.collection.delete(ids=node_ids)

            # Clear from cache
            for node_id in node_ids:
                if node_id in self.node_cache:
                    del self.node_cache[node_id]
                    try:
                        self.cache_order.remove(node_id)
                    except ValueError:
                        pass

            return deleted_count

    def get_document_token_stats(self, document_id: str) -> dict[str, float | int]:
        """Get token statistics for a document using efficient SQL aggregation.

        Returns:
            Dict with keys: avg_tokens, min_tokens, max_tokens, total_tokens, node_count
        """
        with self.SessionLocal() as session:
            from sqlalchemy import func

            result = (
                session.query(
                    func.avg(TreeNode.token_count).label("avg_tokens"),
                    func.min(TreeNode.token_count).label("min_tokens"),
                    func.max(TreeNode.token_count).label("max_tokens"),
                    func.sum(TreeNode.token_count).label("total_tokens"),
                    func.count(TreeNode.id).label("node_count"),
                )
                .filter(
                    TreeNode.document_id == document_id,
                    TreeNode.token_count.isnot(None),
                )
                .one()
            )

            return {
                "avg_tokens": float(result.avg_tokens) if result.avg_tokens else 0.0,
                "min_tokens": result.min_tokens or 0,
                "max_tokens": result.max_tokens or 0,
                "total_tokens": result.total_tokens or 0,
                "node_count": result.node_count or 0,
            }

    def _run_migrations(self):
        """Run any necessary database migrations."""
        try:
            with self.engine.connect() as conn:
                # Check if tree_nodes table exists first
                result = conn.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='tree_nodes'"
                    )
                )
                if result.fetchone():
                    # Table exists, check columns
                    result = conn.execute(text("PRAGMA table_info(tree_nodes)"))
                    columns = [row[1] for row in result.fetchall()]

                    # Migration: Drop is_dirty column if present
                    if "is_dirty" in columns:
                        self._drop_column_migration(conn, "is_dirty")

                    # Migration: Drop depth column if present
                    elif "depth" in columns:
                        self._drop_column_migration(
                            conn,
                            "depth",
                            cleanup_callback=self._clean_chromadb_metadata,
                        )
                else:
                    logger.debug(
                        "tree_nodes table does not exist yet, will be created by SQLAlchemy"
                    )

                # Check if documents table needs model columns migration
                result = conn.execute(
                    text(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='documents'"
                    )
                )
                if result.fetchone():
                    # Table exists, check if it has the new columns
                    result = conn.execute(text("PRAGMA table_info(documents)"))
                    columns = [row[1] for row in result.fetchall()]

                    missing_embedding = "embedding_model" not in columns
                    missing_summary = "summary_model" not in columns
                    if missing_embedding or missing_summary:
                        self._add_document_model_columns_migration(conn)
                else:
                    logger.debug(
                        "documents table does not exist yet, will be created by SQLAlchemy"
                    )
        except Exception as e:
            logger.debug(
                f"Migration check failed (this is normal for new databases): {e}"
            )

    def _drop_column_migration(
        self,
        conn: Any,
        column_name: str,
        cleanup_callback: Callable[[], None] | None = None,
    ) -> None:
        """Drop a column from tree_nodes table using SQLite table recreation pattern.

        Args:
            conn: Database connection
            column_name: Name of column to drop
            cleanup_callback: Optional callback to run after successful migration
        """
        logger.info(f"Found deprecated {column_name} column, dropping it...")
        try:
            # SQLite doesn't support DROP COLUMN directly in older versions
            # We need to recreate the table without the column
            conn.execute(text("BEGIN TRANSACTION"))

            # Create new table without the specified column
            conn.execute(
                text(
                    """
                CREATE TABLE tree_nodes_new (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    parent_id VARCHAR,
                    left_child_id VARCHAR,
                    right_child_id VARCHAR,
                    span_start INTEGER NOT NULL,
                    span_end INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    summary TEXT,
                    token_count INTEGER,
                    is_pinned INTEGER DEFAULT 0,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    document_id VARCHAR,
                    FOREIGN KEY(parent_id) REFERENCES tree_nodes (id),
                    FOREIGN KEY(document_id) REFERENCES documents (id)
                )
            """
                )
            )

            # Copy data from old table (excluding the dropped column)
            conn.execute(
                text(
                    """
                INSERT INTO tree_nodes_new
                SELECT id, parent_id, left_child_id, right_child_id,
                       span_start, span_end, text, summary,
                       is_pinned, last_accessed, access_count,
                       created_at, document_id
                FROM tree_nodes
            """
                )
            )

            # Drop old table and rename new one
            conn.execute(text("DROP TABLE tree_nodes"))
            conn.execute(text("ALTER TABLE tree_nodes_new RENAME TO tree_nodes"))

            conn.execute(text("COMMIT"))
            logger.info(
                f"Successfully dropped {column_name} column from tree_nodes table"
            )

            # Run cleanup callback if provided
            if cleanup_callback:
                cleanup_callback()
        except Exception as e:
            conn.execute(text("ROLLBACK"))
            logger.error(f"Failed to drop {column_name} column: {e}")
            raise

    def _add_document_model_columns_migration(self, conn: Any) -> None:
        """Add embedding_model and summary_model columns to documents table."""
        logger.info(
            "Adding embedding_model and summary_model columns to documents table..."
        )
        try:
            conn.execute(text("BEGIN TRANSACTION"))

            # Add the new columns with default values
            # Use the current default models from config as fallback
            default_embedding_model = "text-embedding-3-small"
            default_summary_model = "gpt-4o"

            # Check which columns are missing and add them
            result = conn.execute(text("PRAGMA table_info(documents)"))
            existing_columns = [row[1] for row in result.fetchall()]

            if "embedding_model" not in existing_columns:
                conn.execute(
                    text(
                        f"ALTER TABLE documents ADD COLUMN embedding_model VARCHAR "
                        f"NOT NULL DEFAULT '{default_embedding_model}'"
                    )
                )
                logger.info("Added embedding_model column to documents table")

            if "summary_model" not in existing_columns:
                conn.execute(
                    text(
                        f"ALTER TABLE documents ADD COLUMN summary_model VARCHAR "
                        f"NOT NULL DEFAULT '{default_summary_model}'"
                    )
                )
                logger.info("Added summary_model column to documents table")

            conn.execute(text("COMMIT"))
            logger.info("Successfully added model columns to documents table")

        except Exception as e:
            conn.execute(text("ROLLBACK"))
            logger.error(f"Failed to add model columns to documents table: {e}")
            raise

    def _clean_chromadb_metadata(self):
        """Clean up deprecated fields from ChromaDB metadata."""
        try:
            logger.info("Cleaning up ChromaDB metadata...")

            # Get all entries from ChromaDB
            results = self.collection.get(include=["metadatas"])

            if not results or not results.get("ids"):
                logger.debug("No ChromaDB entries to clean")
                return

            ids = results["ids"]
            metadatas = results["metadatas"]

            # Check if any entries have the deprecated 'depth' field
            needs_update: list[str] = []
            updated_metadatas: list[Mapping[str, str | int | float | bool | None]] = []

            # Handle case where metadatas might be None
            if metadatas is None:
                return

            for i, (node_id, metadata) in enumerate(zip(ids, metadatas)):
                if metadata and "depth" in metadata:
                    needs_update.append(node_id)
                    # Create new metadata without depth field
                    new_metadata = {k: v for k, v in metadata.items() if k != "depth"}
                    updated_metadatas.append(new_metadata)

            if needs_update:
                logger.info(
                    f"Updating {len(needs_update)} ChromaDB entries to remove depth field"
                )
                # Update entries in batches
                batch_size = 100
                for i in range(0, len(needs_update), batch_size):
                    batch_ids = needs_update[i : i + batch_size]
                    batch_metadatas = updated_metadatas[i : i + batch_size]

                    # ChromaDB update requires updating with the same data but new metadata
                    self.collection.update(ids=batch_ids, metadatas=batch_metadatas)

                logger.info(
                    f"Successfully cleaned {len(needs_update)} ChromaDB entries"
                )
            else:
                logger.debug("No ChromaDB entries need cleaning")

        except Exception as e:
            logger.warning(f"Failed to clean ChromaDB metadata: {e}")
            # Don't fail the migration if ChromaDB cleanup fails
            # This is a one-time cleanup that's not critical

    def close(self):
        """Close database connections and cleanup resources."""
        if hasattr(self, "engine"):
            self.engine.dispose()
        # ChromaDB PersistentClient doesn't have a close method, but we can help GC
        if hasattr(self, "collection"):
            del self.collection
        if hasattr(self, "chroma_client"):
            del self.chroma_client

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    @contextmanager
    def temporary():
        """Create a temporary Store instance for testing/benchmarking.

        This creates a Store with temporary SQLite and Chroma databases
        that are automatically cleaned up when the context exits.

        Usage:
            with Store.temporary() as store:
                # Use store for testing
                pass
            # Databases are automatically cleaned up
        """
        import shutil
        import tempfile

        temp_dir = tempfile.mkdtemp()
        try:
            config = OperationalConfig(
                sqlite_database_url=f"sqlite:///{temp_dir}/test.db",
                chroma_persist_directory=f"{temp_dir}/chroma",
            )
            store = Store(config)
            yield store
        finally:
            # Clean up
            store.close()
            shutil.rmtree(temp_dir, ignore_errors=True)
