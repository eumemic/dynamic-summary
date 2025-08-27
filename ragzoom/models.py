"""SQLAlchemy models for RagZoom using PostgreSQL with pgvector."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
    following_neighbor_id: Mapped[str | None] = mapped_column(
        String, nullable=True
    )  # ID of the node that immediately follows this one at the same tree level
    height: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )  # Distance to furthest leaf (0 for leaves, incrementing upward)
    path: Mapped[str] = mapped_column(
        String, nullable=False, default=""
    )  # Binary path encoding node position in tree (empty string for root)

    # Performance indices for frequently queried columns
    __table_args__ = (
        # Index on document_id for document-level operations (clearing, validation, etc.)
        Index("idx_tree_nodes_document_id", "document_id"),
        # Index on parent_id for tree navigation queries
        Index("idx_tree_nodes_parent_id", "parent_id"),
        # Composite index for root node queries (document + no parent)
        Index("idx_tree_nodes_document_root", "document_id", "parent_id"),
        # Index on path for fast tree traversal operations
        Index("idx_tree_nodes_path", "path"),
        # Composite index for document-scoped path queries
        Index("idx_tree_nodes_document_path", "document_id", "path"),
    )

    def is_left_child(self) -> bool:
        """Check if this node is a left child based on its path."""
        return self.path.endswith("0")

    def is_right_child(self) -> bool:
        """Check if this node is a right child based on its path."""
        return self.path.endswith("1")

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (has no children)."""
        return self.height == 0

    def is_root(self) -> bool:
        """Check if this node is the root node (has no parent)."""
        return self.parent_id is None

    def get_depth(self) -> int:
        """Return the depth of this node in the tree (0 for root)."""
        return len(self.path)


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
