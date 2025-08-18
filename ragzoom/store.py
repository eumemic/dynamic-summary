"""Storage layer for RagZoom using PostgreSQL with pgvector for embeddings."""

import hashlib
import logging
import os
from collections import deque
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

# Import register_vector from the appropriate module
try:
    from pgvector.psycopg import register_vector
except ImportError:
    from pgvector.psycopg2 import register_vector

from ragzoom.config import OperationalConfig

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TreeNode(Base):
    """Database model for tree nodes with embedded vectors."""

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
    embedding: Mapped[list[float]] = mapped_column(Vector(), nullable=False)
    token_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )  # Token count of text content (raw text for leaves, summary for internal nodes)
    is_pinned: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    document_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("documents.id"), nullable=True
    )
    preceding_neighbor_id: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # ID of the node that immediately precedes this one at the same tree level


class Document(Base):
    """Database model for documents."""

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
    """Combined storage for tree structure and embeddings in PostgreSQL."""

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

        # Auto-start Docker PostgreSQL if using default database URL
        # and no explicit database URL is set via environment
        database_url = config.database_url

        # Check if we should auto-start Docker PostgreSQL
        should_auto_start = (
            database_url == "postgresql+psycopg://localhost/ragzoom"
            and not os.getenv("RAGZOOM_DATABASE_URL")  # User didn't explicitly set URL
            and not os.getenv("RAGZOOM_NO_DOCKER")  # User didn't disable Docker
        )

        if should_auto_start:
            try:
                from ragzoom.docker_postgres import DockerPostgres

                docker_pg = DockerPostgres()
                database_url = docker_pg.ensure_running()
                logger.info("✅ PostgreSQL ready in Docker container")
            except ImportError:
                logger.debug("Docker PostgreSQL management not available")
            except OSError:
                # User-friendly errors from DockerPostgres - re-raise as-is
                # Don't log here since CLI will handle the user-facing message
                raise
            except Exception as e:
                logger.debug(
                    f"Auto-start failed: {e}"
                )  # Debug level for technical details
                # Re-raise with helpful context
                raise OSError(
                    f"\n❌ Failed to start PostgreSQL automatically.\n\n"
                    f"Run 'ragzoom doctor' to diagnose the issue.\n"
                    f"Error: {str(e)}"
                )

        # Initialize PostgreSQL with pgvector
        self.engine = create_engine(database_url)

        # Register pgvector extension using event listener
        @event.listens_for(self.engine, "connect")
        def register_vector_extension(dbapi_conn, connection_record):
            register_vector(dbapi_conn)

        # Handle migration before creating tables with new schema
        self._run_migrations()

        # Create all tables (will only create missing ones)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

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
            with self.SessionLocal() as session:
                result = session.execute(select(TreeNode.embedding).limit(1)).first()
                if result and result[0] is not None:
                    return len(result[0])
        except Exception as e:
            logger.debug(f"Could not infer embedding dimension: {e}")

        # If no existing data, don't enforce validation
        # This allows tests and first-time setups to work with any dimension
        return None

    def _validate_embedding_dimension(
        self, embedding: list[float] | NDArray[np.float64]
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
        embedding: list[float] | NDArray[np.float64],
        span_start: int,
        span_end: int,
        parent_id: str | None = None,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
        document_id: str | None = None,
        token_count: int = 0,
    ) -> TreeNode:
        """Add a node to the database with its embedding."""
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
                embedding=list(map(float, embedding)),
                document_id=document_id,
                token_count=token_count,
            )
            session.add(node)
            session.commit()

            # Add to cache
            self._add_to_cache(node)

        return node

    def add_nodes_batch(self, nodes_data: list[dict[str, Any]]) -> list[TreeNode]:
        """Add multiple nodes to the database in batch.

        Args:
            nodes_data: List of dictionaries containing node data with keys:
                - node_id, text, embedding, span_start, span_end,
                - parent_id (optional), left_child_id (optional),
                - right_child_id (optional), document_id (optional),
                - token_count (defaults to 0)

        Returns:
            List of created TreeNode objects
        """
        if not nodes_data:
            return []

        # Validate all embeddings first
        for data in nodes_data:
            self._validate_embedding_dimension(data["embedding"])

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
                    embedding=list(map(float, data["embedding"])),
                    document_id=data.get("document_id"),
                    token_count=data.get("token_count", 0),
                    preceding_neighbor_id=data.get("preceding_neighbor_id"),
                )
                nodes.append(node)

            # Bulk insert all nodes
            session.bulk_save_objects(nodes)
            session.commit()

            # Add all to cache
            for node in nodes:
                self._add_to_cache(node)

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

            # Bulk update - use execute with update statement for better compatibility
            from sqlalchemy import update

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
                if child_id in self.node_cache:
                    del self.node_cache[child_id]
                    # Also remove from cache order
                    if child_id in self.cache_order:
                        self.cache_order.remove(child_id)

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
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar nodes using pgvector cosine distance.

        Returns list of (id, similarity, metadata) tuples where similarity is in [0, 1].
        """
        query_array = list(map(float, query_embedding))

        with self.SessionLocal() as session:
            stmt = (
                select(
                    TreeNode.id,
                    TreeNode.embedding.cosine_distance(query_array).label("distance"),
                    TreeNode.span_start,
                    TreeNode.span_end,
                    TreeNode.parent_id,
                    TreeNode.document_id,
                )
                .order_by(TreeNode.embedding.cosine_distance(query_array))
                .limit(n_results)
            )
            rows = session.execute(stmt).all()

        output = []
        for row in rows:
            distance = float(row.distance)
            similarity = 1.0 - (distance / 2.0)
            similarity = max(0.0, min(1.0, similarity))
            metadata = {
                "span_start": row.span_start,
                "span_end": row.span_end,
                "parent_id": row.parent_id or "",
                "document_id": row.document_id or "",
            }
            output.append((row.id, similarity, metadata))

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
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, dict[str, Any]]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Apply MMR (Maximal Marginal Relevance) to get diverse results."""
        if not candidates or k <= 0:
            return []

        # Get embeddings for all candidates
        candidate_ids = [c[0] for c in candidates]

        with self.SessionLocal() as session:
            rows = session.execute(
                select(TreeNode.id, TreeNode.embedding).where(
                    TreeNode.id.in_(candidate_ids)
                )
            ).all()

        id_to_embedding = {
            row.id: np.array(row.embedding, dtype=np.float32) for row in rows
        }

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
        """Add a document record.

        Args:
            document_id: Unique identifier for the document
            file_path: Optional path to the source file
            content_hash: SHA256 hash of the document content
            chunk_count: Number of chunks in the document
            embedding_model: Name of the embedding model used for indexing
            summary_model: Name of the summarization model used

        Note: Model name validation is performed by the indexing layer
        to ensure they're valid OpenAI models before storage.
        """
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

            # Delete from database
            deleted_count = (
                session.query(TreeNode).filter_by(document_id=document_id).delete()
            )
            session.commit()

            # Clear from cache
            for node_id in node_ids:
                if node_id in self.node_cache:
                    del self.node_cache[node_id]
                    try:
                        self.cache_order.remove(node_id)
                    except ValueError:
                        pass

            return deleted_count

    def clear_document(self, document_id: str) -> int:
        """Clear all data for a document, including orphaned nodes and document record.

        This handles both complete documents and orphaned nodes from interrupted indexing.
        Unlike delete_document_nodes, this also removes the Document record.

        Args:
            document_id: ID of the document to clear

        Returns:
            Number of nodes deleted
        """
        # Delete all nodes with this document_id (handles orphaned nodes from interrupted runs)
        deleted_count = self.delete_document_nodes(document_id)

        # Also delete document record if it exists
        with self.SessionLocal() as session:
            session.query(Document).filter_by(id=document_id).delete()
            session.commit()

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

    def _run_migrations(self) -> None:
        """Run any necessary database migrations."""
        try:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception as e:
            logger.debug(
                f"Migration check failed (this is normal for new databases): {e}"
            )

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        if hasattr(self, "engine"):
            self.engine.dispose()

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
