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
    Text,
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


class SqliteBase(DeclarativeBase):
    pass


class SQLiteTreeNode(SqliteBase):
    __tablename__ = "tree_nodes"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_id: Mapped[str | None] = mapped_column(String, nullable=True)
    left_child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    right_child_id: Mapped[str | None] = mapped_column(String, nullable=True)
    span_start: Mapped[int] = mapped_column(Integer, nullable=False)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_pinned: Mapped[int] = mapped_column(Integer, default=0)
    last_accessed: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow
    )
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow
    )
    document_id: Mapped[str | None] = mapped_column(String, nullable=True)
    preceding_neighbor_id: Mapped[str | None] = mapped_column(String, nullable=True)
    following_neighbor_id: Mapped[str | None] = mapped_column(String, nullable=True)
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
    file_path: Mapped[str | None] = mapped_column(String, nullable=True)
    content_hash: Mapped[str] = mapped_column(String, nullable=False)
    indexed_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    summary_model: Mapped[str] = mapped_column(String, nullable=False)


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
