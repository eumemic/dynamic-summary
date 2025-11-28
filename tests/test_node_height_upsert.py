"""Regression tests for node height updates during upsert operations."""

from __future__ import annotations

from typing import cast

from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import TextClause

from ragzoom.backends.sqlite_db import SqliteDatabaseManager
from ragzoom.backends.sqlite_repositories import SqliteNodeRepository
from ragzoom.contracts.node_repository import NodeDataDict
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


def test_sqlite_upsert_preserves_height_on_conflict() -> None:
    """An upsert must not overwrite an existing node's height."""

    db = SqliteDatabaseManager("sqlite:///:memory:")
    try:
        repo = SqliteNodeRepository(db, CacheManager())

        node_id = "node-1"
        node_payload: NodeDataDict = {
            "node_id": node_id,
            "text": "leaf",
            "span_start": 0,
            "span_end": 1,
            "document_id": "doc-1",
            "token_count": 1,
            "height": 0,
            "level_index": 0,
        }

        repo.add_nodes_batch([node_payload])

        promoted_payload: NodeDataDict = {
            "node_id": node_id,
            "text": "leaf",
            "span_start": 0,
            "span_end": 1,
            "document_id": "doc-1",
            "token_count": 1,
            "height": 1,
            "level_index": 0,
        }

        repo.upsert_nodes_batch([promoted_payload])

        row = repo.get_node(node_id)
        assert row is not None
        assert row.height == 0
    finally:
        db.close()


def test_postgres_upsert_omits_height_update_clause() -> None:
    """The ON CONFLICT clause must not overwrite height for existing nodes."""

    repo = _StubPostgresNodeRepository()

    dummy_session = _DummySession()

    payload: NodeDataDict = {
        "node_id": "node-2",
        "text": "internal",
        "span_start": 0,
        "span_end": 1,
        "document_id": "doc-2",
        "token_count": 0,
        "height": 2,
        "level_index": 0,
    }

    repo.upsert_nodes_batch([payload], session=cast(Session, dummy_session))

    assert dummy_session.executed, "Expected upsert SQL to execute"
    sql_text = dummy_session.executed[0][0].text
    assert "height = EXCLUDED.height" not in sql_text
