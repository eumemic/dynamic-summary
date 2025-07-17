"""Storage layer for RagZoom - SQLite for tree structure, Chroma for vectors."""

import hashlib
import logging
from collections import deque
from datetime import datetime
from typing import Any, Optional, Union, cast

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

from ragzoom.config import RagZoomConfig

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class TreeNode(Base):
    """SQLite model for tree nodes."""

    __tablename__ = "tree_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("tree_nodes.id"), nullable=True
    )
    left_child_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    right_child_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    span_start: Mapped[int] = mapped_column(Integer, nullable=False)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True
    )  # NULL for leaf nodes
    mid_offset: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )  # Position of <<<MID>>> delimiter in parent summaries
    is_dirty: Mapped[int] = mapped_column(
        Integer, default=0
    )  # Boolean flag for re-summarization
    is_pinned: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    document_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("documents.id"), nullable=True
    )


class Document(Base):
    """SQLite model for documents."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    file_path: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, unique=True
    )  # Path to the source file
    content_hash: Mapped[str] = mapped_column(
        String, nullable=False
    )  # SHA256 hash of content
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)


class Store:
    """Combined storage for tree structure (SQLite) and embeddings (Chroma)."""

    def __init__(self, config: RagZoomConfig):
        """Initialize storage backends."""
        self.config = config

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
        self.cache_order: deque[str] = deque(maxlen=1000)

        # Cache expected embedding dimension for validation
        self._expected_embedding_dim = self._get_expected_embedding_dimension()

    def _get_from_cache(self, node_id: str) -> Optional[TreeNode]:
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

    def _get_expected_embedding_dimension(self) -> Optional[int]:
        """Get expected embedding dimension from config or existing data."""
        if self.config.embedding_dimensions:
            return self.config.embedding_dimensions

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

        # If no explicit config and no existing data, don't enforce validation
        # This allows tests and first-time setups to work with any dimension
        return None

    def _validate_embedding_dimension(
        self, embedding: Union[list[float], np.ndarray]
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
        embedding: Union[list[float], np.ndarray],
        depth: int,
        span_start: int,
        span_end: int,
        parent_id: Optional[str] = None,
        left_child_id: Optional[str] = None,
        right_child_id: Optional[str] = None,
        summary: Optional[str] = None,
        mid_offset: Optional[int] = None,
        document_id: Optional[str] = None,
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
                depth=depth,
                span_start=span_start,
                span_end=span_end,
                text=text,
                summary=summary,
                mid_offset=mid_offset,
                document_id=document_id,
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
                    "depth": depth,
                    "span_start": span_start,
                    "span_end": span_end,
                    "parent_id": parent_id or "",
                    "is_leaf": 1 if summary is None else 0,
                    "document_id": document_id
                    or "",  # ChromaDB doesn't accept None values
                }
            ],
            documents=[text],
        )

        return node

    def get_node(self, node_id: str) -> Optional[TreeNode]:
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

    def mark_dirty_upward(self, node_id: str) -> None:
        """Mark a node and all ancestors as dirty (needs re-summarization)."""
        with self.SessionLocal() as session:
            current_id: Optional[str] = node_id
            marked_ids = []
            while current_id:
                node = session.query(TreeNode).filter_by(id=current_id).first()
                if not node:
                    break
                node.is_dirty = 1
                marked_ids.append(current_id)
                current_id = node.parent_id
            session.commit()

            # Invalidate cache for modified nodes
            for node_id in marked_ids:
                if node_id in self.node_cache:
                    del self.node_cache[node_id]
                    # Only remove from cache_order if it exists
                    if node_id in self.cache_order:
                        self.cache_order.remove(node_id)

    def update_summary(
        self,
        node_id: str,
        text: str,
        embedding: Union[list[float], np.ndarray],
        mid_offset: Optional[int] = None,
    ) -> None:
        """Update node summary and clear dirty flag."""
        # Validate embedding dimension
        self._validate_embedding_dimension(embedding)

        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                # Use cast to handle the nullable text field
                node.text = cast(str, text)
                node.summary = text  # These are the same for internal nodes
                if mid_offset is not None:
                    node.mid_offset = mid_offset
                node.is_dirty = 0
                session.commit()

                # Update in vector store
                # Convert to numpy array if needed
                embedding_array = np.array(embedding, dtype=np.float32)
                self.collection.update(
                    ids=[node_id],
                    embeddings=cast(Any, [embedding_array]),
                    metadatas=[{"text": text}],
                )

                # Update cache - refresh the cached node with new data
                if node_id in self.node_cache:
                    del self.node_cache[node_id]
                    if node_id in self.cache_order:
                        self.cache_order.remove(node_id)
                # Re-add to cache with fresh data
                self._add_to_cache(node)

    def get_dirty_nodes(self) -> list[TreeNode]:
        """Get all nodes marked as dirty."""
        with self.SessionLocal() as session:
            return session.query(TreeNode).filter_by(is_dirty=1).all()

    def get_children(
        self, node_id: str
    ) -> tuple[Optional[TreeNode], Optional[TreeNode]]:
        """Get left and right children of a node."""
        node = self.get_node(node_id)
        if not node:
            return None, None

        left = self.get_node(node.left_child_id) if node.left_child_id else None
        right = self.get_node(node.right_child_id) if node.right_child_id else None
        return left, right

    def get_child(self, node_id: str, side: str) -> Optional[TreeNode]:
        """Get the left or right child of a node."""
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if not node:
                return None

            child_id = node.left_child_id if side == "LEFT" else node.right_child_id
            if not child_id:
                return None

            return session.query(TreeNode).filter_by(id=child_id).first()

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Get all ancestors of given nodes."""
        ancestors = set()
        to_process = list(node_ids)

        while to_process:
            node_id = to_process.pop()
            node = self.get_node(node_id)
            if node and node.parent_id and node.parent_id not in ancestors:
                ancestors.add(node.parent_id)
                to_process.append(node.parent_id)

        return [node for aid in ancestors if (node := self.get_node(aid)) is not None]

    def search_similar(
        self,
        query_embedding: Union[list[float], np.ndarray],
        n_results: int,
        where: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar nodes using Chroma."""
        # Convert to numpy array if needed
        query_array = np.array(query_embedding, dtype=np.float32)
        results = self.collection.query(
            query_embeddings=cast(Any, [query_array]),
            n_results=n_results,
            where=where,
        )

        # Return list of (id, distance, metadata) tuples
        output = []
        ids = results.get("ids")
        distances = results.get("distances")
        metadatas = results.get("metadatas")

        if ids and distances and metadatas and len(ids) > 0:
            for i in range(len(ids[0])):
                output.append(
                    (
                        ids[0][i],
                        float(distances[0][i]),
                        (
                            dict(metadatas[0][i])
                            if isinstance(metadatas[0][i], dict)
                            else {}
                        ),
                    )
                )

        return output

    def get_pinned_nodes(self, depth_max: Optional[int] = None) -> list[TreeNode]:
        """Get nodes that are pinned (always included in coverage)."""
        with self.SessionLocal() as session:
            query = session.query(TreeNode).filter_by(is_pinned=1)
            if depth_max is not None:
                query = query.filter(TreeNode.depth <= depth_max)
            return query.all()

    def pin_node(self, node_id: str) -> bool:
        """Pin a node if it's within allowed depth."""
        node = self.get_node(node_id)
        if not node or node.depth > self.config.pin_depth_max:
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
            return session.query(TreeNode).filter_by(summary=None).all()

    def get_root_node(self) -> Optional[TreeNode]:
        """Get the root node (node with no parent)."""
        with self.SessionLocal() as session:
            return session.query(TreeNode).filter_by(parent_id=None).first()

    def get_root_node_for_document(
        self, document_id: Optional[str]
    ) -> Optional[TreeNode]:
        """Get the root node for a specific document."""
        with self.SessionLocal() as session:
            query = session.query(TreeNode).filter_by(parent_id=None)
            if document_id:
                query = query.filter_by(document_id=document_id)
            return query.first()

    def get_all_nodes_for_document(self, document_id: Optional[str]) -> list[TreeNode]:
        """Get all nodes for a specific document."""
        with self.SessionLocal() as session:
            if document_id:
                return session.query(TreeNode).filter_by(document_id=document_id).all()
            else:
                # If no document_id, maybe return all nodes? Or raise error?
                # For now, let's return all nodes, but this could be memory intensive
                logger.warning("No document_id provided, returning all nodes in store.")
                return session.query(TreeNode).all()

    def compute_mmr_diverse_results(
        self,
        query_embedding: Union[list[float], np.ndarray],
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

    def get_document_by_path(self, file_path: str) -> Optional[Document]:
        """Get a document by file path."""
        with self.SessionLocal() as session:
            return session.query(Document).filter_by(file_path=file_path).first()

    def get_document_by_hash(self, content_hash: str) -> Optional[Document]:
        """Get a document by content hash."""
        with self.SessionLocal() as session:
            return session.query(Document).filter_by(content_hash=content_hash).first()

    def add_document(
        self,
        document_id: str,
        file_path: Optional[str],
        content_hash: str,
        chunk_count: int,
    ) -> Document:
        """Add a document record."""
        with self.SessionLocal() as session:
            doc = Document(
                id=document_id,
                file_path=file_path,
                content_hash=content_hash,
                chunk_count=chunk_count,
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
                    # Table exists, check if mid_offset column exists
                    result = conn.execute(text("PRAGMA table_info(tree_nodes)"))
                    columns = [row[1] for row in result.fetchall()]

                    if "mid_offset" not in columns:
                        # Add the missing column (with proper error handling)
                        try:
                            conn.execute(
                                text(
                                    "ALTER TABLE tree_nodes ADD COLUMN mid_offset INTEGER"
                                )
                            )
                            conn.commit()
                            logger.info("Added mid_offset column to tree_nodes table")
                        except Exception as e:
                            if "duplicate column" in str(e).lower():
                                logger.debug("mid_offset column already exists")
                            else:
                                raise
                    else:
                        logger.debug("mid_offset column already exists")
                else:
                    logger.debug(
                        "tree_nodes table does not exist yet, will be created by SQLAlchemy"
                    )
        except Exception as e:
            logger.debug(
                f"Migration check failed (this is normal for new databases): {e}"
            )

    def close(self):
        """Close database connections and cleanup resources."""
        if hasattr(self, "engine"):
            self.engine.dispose()
        # ChromaDB PersistentClient doesn't have a close method, but we can help GC
        if hasattr(self, "collection"):
            self.collection = None
        if hasattr(self, "chroma_client"):
            self.chroma_client = None

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()
