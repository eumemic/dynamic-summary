"""Lightweight query logging backed by a dedicated SQLite file.

This logger is intentionally decoupled from the primary storage backends to
avoid coupling query history to production data paths. It records minimal
metadata needed to reconstruct query visualizations.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ragzoom.worktree_utils import DEFAULT_DATA_DIR_NAME


@dataclass(frozen=True)
class QuerySummary:
    """Summary metadata for a logged query."""

    id: str
    document_id: str
    query_text: str
    budget_tokens: int | None
    num_seeds: int | None
    created_at: str


@dataclass(frozen=True)
class QueryNodeRow:
    """Logged node entry for a query."""

    node_id: str
    score: float
    is_seed: bool
    position: int


@dataclass(frozen=True)
class QueryDetail:
    """Complete logged query with ordered tiling nodes."""

    id: str
    document_id: str
    query_text: str
    budget_tokens: int | None
    num_seeds: int | None
    created_at: str
    nodes: list[QueryNodeRow]


class QueryLog:
    """Minimal SQLite-backed query history."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    @staticmethod
    def default_path(base_dir: Path | None = None) -> Path:
        """Return the default path for the query log SQLite file."""
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        return root / DEFAULT_DATA_DIR_NAME / "query-log.db"

    def record_query(
        self,
        *,
        document_id: str,
        query_text: str,
        budget_tokens: int | None,
        num_seeds: int | None,
        tiling_ids: Sequence[str],
        scores: dict[str, float],
        seed_ids: set[str],
    ) -> str:
        """Persist a query and return its generated ID."""
        if not tiling_ids:
            raise ValueError("Cannot log query without tiling nodes")

        query_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queries (id, document_id, query_text, budget_tokens, num_seeds, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    document_id,
                    query_text,
                    budget_tokens,
                    num_seeds,
                    created_at,
                ),
            )

            entries: list[tuple[str, str, float, int, int]] = []
            for position, node_id in enumerate(tiling_ids):
                score = float(scores.get(node_id, 0.0))
                is_seed = 1 if node_id in seed_ids else 0
                entries.append((query_id, node_id, score, is_seed, position))

            conn.executemany(
                """
                INSERT INTO query_nodes (query_id, node_id, score, is_seed, position)
                VALUES (?, ?, ?, ?, ?)
                """,
                entries,
            )

        return query_id

    def list_queries(self, document_id: str, limit: int) -> list[QuerySummary]:
        """Return recent queries for a document, most recent first."""
        if limit <= 0:
            raise ValueError("limit must be positive")

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, document_id, query_text, budget_tokens, num_seeds, created_at
                FROM queries
                WHERE document_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (document_id, limit),
            ).fetchall()

        return [
            QuerySummary(
                id=row["id"],
                document_id=row["document_id"],
                query_text=row["query_text"],
                budget_tokens=row["budget_tokens"],
                num_seeds=row["num_seeds"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_query(self, query_id: str) -> QueryDetail | None:
        """Return a logged query with ordered nodes."""
        with self._connect() as conn:
            header = conn.execute(
                """
                SELECT id, document_id, query_text, budget_tokens, num_seeds, created_at
                FROM queries
                WHERE id = ?
                """,
                (query_id,),
            ).fetchone()

            if header is None:
                return None

            node_rows = conn.execute(
                """
                SELECT node_id, score, is_seed, position
                FROM query_nodes
                WHERE query_id = ?
                ORDER BY position ASC
                """,
                (query_id,),
            ).fetchall()

        nodes = [
            QueryNodeRow(
                node_id=row["node_id"],
                score=float(row["score"]),
                is_seed=bool(row["is_seed"]),
                position=int(row["position"]),
            )
            for row in node_rows
        ]

        return QueryDetail(
            id=header["id"],
            document_id=header["document_id"],
            query_text=header["query_text"],
            budget_tokens=header["budget_tokens"],
            num_seeds=header["num_seeds"],
            created_at=header["created_at"],
            nodes=nodes,
        )

    # Internal helpers
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS queries (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    budget_tokens INTEGER,
                    num_seeds INTEGER,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS query_nodes (
                    query_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    is_seed INTEGER NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (query_id, node_id),
                    FOREIGN KEY (query_id) REFERENCES queries(id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_queries_document_created ON queries (document_id, created_at DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_query_nodes_query_position ON query_nodes (query_id, position)"
            )


__all__ = ["QueryDetail", "QueryLog", "QueryNodeRow", "QuerySummary"]
