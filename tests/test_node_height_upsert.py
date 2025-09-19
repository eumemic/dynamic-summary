"""Regression tests for node height updates during upsert operations."""

from __future__ import annotations

from typing import cast

from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import TextClause

from ragzoom.backends.sqlite_db import SqliteDatabaseManager
from ragzoom.backends.sqlite_repositories import SqliteNodeRepository
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.repositories.postgres_node_repository import PostgresNodeRepository
from ragzoom.services.cache_manager import CacheManager


class _DummySession:
    def __init__(self) -> None:
        self.executed: list[tuple[TextClause, dict[str, object] | None]] = []

    def execute(
        self, stmt: TextClause, params: dict[str, object] | None = None
    ) -> None:
        self.executed.append((stmt, params))

    def commit(self) -> None:  # pragma: no cover - trivial
        return

    def rollback(self) -> None:  # pragma: no cover - defensive
        return

    def close(self) -> None:  # pragma: no cover - trivial
        return


class _StubPostgresNodeRepository(PostgresNodeRepository):
    def __init__(self) -> None:
        self.cache_manager = CacheManager()

    def get_nodes(self, _node_ids: list[str]) -> list[TreeNode]:
        return []


def test_sqlite_upsert_updates_height() -> None:
    """A second upsert must persist the updated height for promoted nodes."""

    db = SqliteDatabaseManager("sqlite:///:memory:")
    try:
        repo = SqliteNodeRepository(db, CacheManager())

        node_id = "node-1"
        node_payload = {
            "node_id": node_id,
            "text": "leaf",
            "span_start": 0,
            "span_end": 1,
            "document_id": "doc-1",
            "height": 0,
        }

        repo.add_nodes_batch([node_payload])

        promoted_payload = dict(node_payload)
        promoted_payload["height"] = 1

        repo.upsert_nodes_batch([promoted_payload])

        row = repo.get_node(node_id)
        assert row is not None
        assert row.height == 1
    finally:
        db.close()


def test_postgres_upsert_updates_height_clause() -> None:
    """The ON CONFLICT clause must include height to keep tree state consistent."""

    repo = _StubPostgresNodeRepository()

    dummy_session = _DummySession()

    payload = {
        "node_id": "node-2",
        "text": "internal",
        "span_start": 0,
        "span_end": 1,
        "document_id": "doc-2",
        "token_count": 0,
        "height": 2,
    }

    repo.upsert_nodes_batch([payload], session=cast(Session, dummy_session))

    assert dummy_session.executed, "Expected upsert SQL to execute"
    sql_text = dummy_session.executed[0][0].text
    assert "height = EXCLUDED.height" in sql_text
