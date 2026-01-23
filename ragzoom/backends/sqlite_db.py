"""SQLite database manager and models for the pluggable storage backend.

This module provides a minimal schema compatible with DocumentStore usage.
Embeddings are not stored here; vector search is handled by a separate
VectorIndex implementation (e.g., PythonVectorIndex).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)
from sqlalchemy.pool import StaticPool

from ragzoom.models import TreeNodeColumnsMixin


class SqliteBase(DeclarativeBase):
    pass


class SQLiteTreeNode(TreeNodeColumnsMixin, SqliteBase):
    __tablename__ = "tree_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    is_pinned: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    document_id: Mapped[str | None] = mapped_column(String, nullable=True)
    preceding_neighbor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    following_neighbor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    level_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node (has no children)."""
        return self.height == 0

    # Compatibility helpers to mirror PostgreSQL TreeNode API
    def is_root(self) -> bool:  # noqa: D401 - trivial helper
        return self.parent_id is None

    def get_depth(self) -> int:  # noqa: D401 - trivial helper
        raise NotImplementedError(
            "SQLiteTreeNode does not persist depth; use TreeNavigator.get_node_depth"
        )


class SqliteDocument(SqliteBase):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(String, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    indexed_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=lambda: dt.datetime.now(dt.timezone.utc)
    )
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    summary_model: Mapped[str] = mapped_column(String, nullable=False)

    # Temporal document flag: determines if document requires timestamps on all chunks
    # See specs/temporal-metadata.md § Requirements > 1. Temporal Documents
    is_temporal: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Custom system prompt for summary generation
    # See specs/custom-prompt-config.md § CLI Override
    # If None, uses the default prompt from IndexConfig
    summary_system_prompt: Mapped[str | None] = mapped_column(
        String, nullable=True, default=None
    )


@dataclass
class SqliteDatabaseManager:
    """SQLite session/engine manager for the backend repositories."""

    url: str

    def __post_init__(self) -> None:
        # Use thread-safe settings for in-memory and test usage
        engine_kwargs: dict[str, object] = {
            "connect_args": {"check_same_thread": False}
        }
        # Share the same in-memory database across threads and sessions
        if self.url.strip().lower() in {"sqlite:///:memory:", "sqlite://"}:
            engine_kwargs["poolclass"] = StaticPool
        self.engine = create_engine(self.url, **engine_kwargs)
        # Apply lightweight performance PRAGMAs suitable for tests/dev
        try:
            with self.engine.connect() as conn:
                # Reduce fsyncs and use in-memory journaling for faster writes
                conn.exec_driver_sql("PRAGMA synchronous=OFF")
                conn.exec_driver_sql("PRAGMA journal_mode=MEMORY")
                # Keep temporary tables in memory and enlarge page cache
                conn.exec_driver_sql("PRAGMA temp_store=MEMORY")
                conn.exec_driver_sql("PRAGMA cache_size=-65536")
        except Exception:
            # Pragmas are best-effort; ignore on unsupported environments
            pass
        SqliteBase.metadata.create_all(self.engine)
        try:
            with self.engine.begin() as conn:
                try:
                    conn.exec_driver_sql("ALTER TABLE documents DROP COLUMN version")
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE documents DROP COLUMN content_hash"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE documents DROP COLUMN chunk_count"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE tree_nodes ADD COLUMN level_index INTEGER NOT NULL DEFAULT 0"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS idx_tree_nodes_document_height_level ON tree_nodes (document_id, height, level_index)"
                    )
                except Exception:
                    pass
                # Unique constraint to prevent duplicate coordinates from concurrent indexers
                try:
                    conn.exec_driver_sql(
                        "CREATE UNIQUE INDEX IF NOT EXISTS uq_tree_nodes_document_height_level ON tree_nodes (document_id, height, level_index)"
                    )
                except Exception:
                    pass
                # Add contextual indexing columns for issue #287
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE tree_nodes ADD COLUMN preceding_context TEXT"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE tree_nodes ADD COLUMN preceding_context_summary TEXT"
                    )
                except Exception:
                    pass
                # Add embedding column for storing vector on leaf nodes
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE tree_nodes ADD COLUMN embedding BLOB"
                    )
                except Exception:
                    pass
                # Add cost column for issue #310
                try:
                    conn.exec_driver_sql("ALTER TABLE tree_nodes ADD COLUMN cost REAL")
                except Exception:
                    pass
                # Add temporal metadata columns for time-windowed queries
                # See specs/temporal-metadata.md § Data Model Changes > Database Schema
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE tree_nodes ADD COLUMN time_start REAL"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE tree_nodes ADD COLUMN time_end REAL"
                    )
                except Exception:
                    pass
                # Add user_id columns for multi-tenancy
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE tree_nodes ADD COLUMN user_id TEXT"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE documents ADD COLUMN user_id TEXT"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS idx_tree_nodes_user_id ON tree_nodes (user_id)"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id)"
                    )
                except Exception:
                    pass
                # Create users table for auth
                try:
                    conn.exec_driver_sql(
                        """CREATE TABLE IF NOT EXISTS users (
                            id TEXT PRIMARY KEY,
                            github_id TEXT UNIQUE,
                            email TEXT,
                            api_key TEXT NOT NULL UNIQUE,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )"""
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS idx_users_api_key ON users (api_key)"
                    )
                except Exception:
                    pass
                try:
                    conn.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS idx_users_github_id ON users (github_id)"
                    )
                except Exception:
                    pass
                # Add is_temporal column for temporal document tracking
                # See specs/temporal-metadata.md § Data Model Changes > Database Schema
                try:
                    conn.exec_driver_sql(
                        "ALTER TABLE documents ADD COLUMN is_temporal INTEGER NOT NULL DEFAULT 0"
                    )
                except Exception:
                    pass
                # Create indexer_leases table for single-writer coordination.
                # Uses a singleton row pattern (id=1) with TTL-based expiration
                # to ensure only one IndexingEngine instance can write at a time.
                try:
                    conn.exec_driver_sql(
                        """
                        CREATE TABLE IF NOT EXISTS indexer_leases (
                            id INTEGER PRIMARY KEY CHECK (id = 1),
                            holder_id VARCHAR(255) NOT NULL,
                            acquired_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            last_heartbeat TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            expires_at TIMESTAMP NOT NULL
                        )
                        """
                    )
                except Exception:
                    pass
        except Exception:
            # Columns already dropped or table newly created; ignore
            pass
        self.SessionLocal = sessionmaker(bind=self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def session(self) -> Iterator[Session]:
        with self.SessionLocal() as session:
            yield session

    def get_document_embedding_model(self, document_id: str) -> str | None:
        with self.SessionLocal() as session:
            row = session.execute(
                select(SqliteDocument.embedding_model).where(
                    SqliteDocument.id == document_id
                )
            ).first()
            return row[0] if row else None
