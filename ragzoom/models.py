"""SQLAlchemy models for RagZoom (storage only; no embeddings)."""

import secrets
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def generate_api_key() -> str:
    """Generate a secure API key with 'rz_' prefix."""
    return f"rz_{secrets.token_urlsafe(32)}"


class Base(DeclarativeBase):
    pass


class TreeNodeColumnsMixin:
    """Shared column definitions for tree node models across backends."""

    left_child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    right_child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    span_start: Mapped[int] = mapped_column(Integer, nullable=False)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Contextual indexing fields (populated during context-aware indexing)
    preceding_context: Mapped[str | None] = mapped_column(Text, nullable=True)
    preceding_context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Embedding vector (stored as packed float32 bytes for efficiency)
    # 1536 dimensions * 4 bytes = 6144 bytes for text-embedding-3-small
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    # Cost in USD for creating this node (embedding + summarization)
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)


class PostgresTreeNode(TreeNodeColumnsMixin, Base):
    """Database model for tree nodes (no embeddings in storage)."""

    __tablename__ = "tree_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )  # Owner of this node (denormalized from document for query efficiency)
    parent_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("tree_nodes.id"), nullable=True
    )
    is_pinned: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    document_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("documents.id"), nullable=True
    )
    preceding_neighbor_id: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # ID of the node that immediately precedes this one at the same tree level
    following_neighbor_id: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # ID of the node that immediately follows this one at the same tree level
    level_index: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )  # 0-based index within nodes of the same height
    height: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )  # Distance to furthest leaf (0 for leaves, incrementing upward)

    # Performance indices for frequently queried columns
    __table_args__ = (
        # Index on document_id for document-level operations (clearing, validation, etc.)
        Index("idx_tree_nodes_document_id", "document_id"),
        # Index on parent_id for tree navigation queries
        Index("idx_tree_nodes_parent_id", "parent_id"),
        # Composite index for root node queries with span ordering
        # Covers: WHERE document_id = X AND parent_id IS NULL ORDER BY span_start
        Index(
            "idx_tree_nodes_document_root_span",
            "document_id",
            "parent_id",
            "span_start",
        ),
        # Index on following_neighbor_id for dataflow navigation
        Index("idx_tree_nodes_following_neighbor_id", "following_neighbor_id"),
        # Index for quickly scanning siblings by height and level_index
        Index(
            "idx_tree_nodes_document_height_level",
            "document_id",
            "height",
            "level_index",
        ),
        # Index for leaf queries with span ordering (height=0 queries)
        # Covers: WHERE document_id = X AND height = 0 ORDER BY span_start
        Index(
            "idx_tree_nodes_document_height_span",
            "document_id",
            "height",
            "span_start",
        ),
        # Composite index for multi-tenant queries (user + document)
        Index("idx_tree_nodes_user_document", "user_id", "document_id"),
        # Note: The unique constraint on (document_id, height, level_index) was removed.
        # Single-writer coordination is now enforced by the IndexerLease mechanism
        # which ensures only one IndexingEngine can write to the database at a time.
        # See ragzoom/server/lease.py for details.
    )

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (has no children)."""
        return self.height == 0

    def is_root(self) -> bool:
        """Check if this node is the root node (has no parent)."""
        return self.parent_id is None

    def get_depth(self) -> int:
        """Depth is not persisted; callers must request it structurally."""
        raise NotImplementedError(
            "PostgresTreeNode does not persist depth; use TreeNavigator.get_node_depth"
        )


class Document(Base):
    """Database model for documents."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )  # Owner of this document (None for legacy/local usage)
    file_path: Mapped[str | None] = mapped_column(
        String, nullable=True, unique=True
    )  # Path to the source file
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    summary_model: Mapped[str] = mapped_column(String, nullable=False)


class User(Base):
    """Database model for authenticated users."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    github_id: Mapped[str | None] = mapped_column(
        String, nullable=True, unique=True, index=True
    )  # GitHub user ID for OAuth
    email: Mapped[str | None] = mapped_column(
        String, nullable=True, index=True
    )  # Email address
    api_key: Mapped[str] = mapped_column(
        String, nullable=False, unique=True, index=True, default=generate_api_key
    )  # API key for authentication
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
