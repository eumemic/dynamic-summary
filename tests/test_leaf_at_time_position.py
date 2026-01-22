"""Tests for get_leaf_at_time_position() implementations.

This tests the time→span mapping foundation: finding boundary leaves that
overlap a query time window to enable using existing span-based queries.

Per the spec (specs/temporal-metadata.md):
- position="start": Earliest leaf L where query_time <= L.time_end
- position="end": Latest leaf L where L.time_start <= query_time
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ragzoom.contracts.node_repository import NodeRepository
from ragzoom.contracts.tree_node import TreeNode

if TYPE_CHECKING:
    from ragzoom.backends.sqlite_db import SqliteDatabaseManager
    from ragzoom.backends.sqlite_repositories import (
        SqliteDocumentRepository,
        SqliteNodeRepository,
    )


def test_protocol_has_get_leaf_at_time_position_method() -> None:
    """Verify NodeRepository protocol defines get_leaf_at_time_position method."""
    assert hasattr(NodeRepository, "get_leaf_at_time_position")

    import inspect

    sig = inspect.signature(NodeRepository.get_leaf_at_time_position)
    params = list(sig.parameters.keys())

    assert "document_id" in params
    assert "time_position" in params
    assert "position" in params


class TestSqliteGetLeafAtTimePosition:
    """Tests for SQLite implementation of get_leaf_at_time_position."""

    @pytest.fixture
    def sqlite_repos(
        self, tmp_path: Path
    ) -> tuple[SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager]:
        """Create SQLite repositories for testing."""
        from ragzoom.backends.sqlite_db import SqliteDatabaseManager
        from ragzoom.backends.sqlite_repositories import (
            SqliteDocumentRepository,
            SqliteNodeRepository,
        )
        from ragzoom.services.cache_manager import CacheManager

        db = SqliteDatabaseManager(url="sqlite:///:memory:")
        cache: CacheManager[TreeNode] = CacheManager()
        node_repo = SqliteNodeRepository(db, cache)
        doc_repo = SqliteDocumentRepository(db)
        return node_repo, doc_repo, db

    def test_returns_none_for_empty_document(
        self,
        sqlite_repos: tuple[
            SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager
        ],
    ) -> None:
        """Return None when document has no nodes."""
        node_repo, doc_repo, _ = sqlite_repos

        # Create document without nodes
        doc_repo.add_document("doc1", None, "test-model", "test-model")

        result = node_repo.get_leaf_at_time_position("doc1", 100.0, "start")
        assert result is None

        result = node_repo.get_leaf_at_time_position("doc1", 100.0, "end")
        assert result is None

    def test_returns_none_for_non_temporal_document(
        self,
        sqlite_repos: tuple[
            SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager
        ],
    ) -> None:
        """Return None when document has leaves but no timestamps."""
        node_repo, doc_repo, _ = sqlite_repos

        doc_repo.add_document("doc1", None, "test-model", "test-model")

        # Add leaf without timestamps
        node_repo.add_nodes_batch(
            [
                {
                    "node_id": "leaf1",
                    "text": "Hello",
                    "span_start": 0,
                    "span_end": 5,
                    "token_count": 1,
                    "height": 0,
                    "level_index": 0,
                    "document_id": "doc1",
                }
            ]
        )

        result = node_repo.get_leaf_at_time_position("doc1", 100.0, "start")
        assert result is None

        result = node_repo.get_leaf_at_time_position("doc1", 100.0, "end")
        assert result is None

    def test_start_finds_earliest_leaf_where_time_lte_time_end(
        self,
        sqlite_repos: tuple[
            SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager
        ],
    ) -> None:
        """position='start' finds earliest leaf where query_time <= leaf.time_end."""
        node_repo, doc_repo, _ = sqlite_repos

        doc_repo.add_document("doc1", None, "test-model", "test-model")

        # Create 3 temporal leaves:
        # leaf1: time [100, 200] span [0, 10]
        # leaf2: time [200, 300] span [10, 20]
        # leaf3: time [300, 400] span [20, 30]
        node_repo.add_nodes_batch(
            [
                {
                    "node_id": "leaf1",
                    "text": "A" * 10,
                    "span_start": 0,
                    "span_end": 10,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 0,
                    "document_id": "doc1",
                    "time_start": 100.0,
                    "time_end": 200.0,
                },
                {
                    "node_id": "leaf2",
                    "text": "B" * 10,
                    "span_start": 10,
                    "span_end": 20,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 1,
                    "document_id": "doc1",
                    "time_start": 200.0,
                    "time_end": 300.0,
                },
                {
                    "node_id": "leaf3",
                    "text": "C" * 10,
                    "span_start": 20,
                    "span_end": 30,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 2,
                    "document_id": "doc1",
                    "time_start": 300.0,
                    "time_end": 400.0,
                },
            ]
        )

        # Query time=150 → leaf1 (150 <= 200)
        result = node_repo.get_leaf_at_time_position("doc1", 150.0, "start")
        assert result is not None
        assert result.id == "leaf1"

        # Query time=200 → leaf1 (200 <= 200)
        result = node_repo.get_leaf_at_time_position("doc1", 200.0, "start")
        assert result is not None
        assert result.id == "leaf1"

        # Query time=201 → leaf2 (201 <= 300, 201 > 200)
        result = node_repo.get_leaf_at_time_position("doc1", 201.0, "start")
        assert result is not None
        assert result.id == "leaf2"

        # Query time=350 → leaf3 (350 <= 400)
        result = node_repo.get_leaf_at_time_position("doc1", 350.0, "start")
        assert result is not None
        assert result.id == "leaf3"

        # Query time=500 → None (no leaf has time_end >= 500)
        result = node_repo.get_leaf_at_time_position("doc1", 500.0, "start")
        assert result is None

    def test_end_finds_latest_leaf_where_time_start_lte_time(
        self,
        sqlite_repos: tuple[
            SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager
        ],
    ) -> None:
        """position='end' finds latest leaf where leaf.time_start <= query_time."""
        node_repo, doc_repo, _ = sqlite_repos

        doc_repo.add_document("doc1", None, "test-model", "test-model")

        # Create 3 temporal leaves:
        # leaf1: time [100, 200] span [0, 10]
        # leaf2: time [200, 300] span [10, 20]
        # leaf3: time [300, 400] span [20, 30]
        node_repo.add_nodes_batch(
            [
                {
                    "node_id": "leaf1",
                    "text": "A" * 10,
                    "span_start": 0,
                    "span_end": 10,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 0,
                    "document_id": "doc1",
                    "time_start": 100.0,
                    "time_end": 200.0,
                },
                {
                    "node_id": "leaf2",
                    "text": "B" * 10,
                    "span_start": 10,
                    "span_end": 20,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 1,
                    "document_id": "doc1",
                    "time_start": 200.0,
                    "time_end": 300.0,
                },
                {
                    "node_id": "leaf3",
                    "text": "C" * 10,
                    "span_start": 20,
                    "span_end": 30,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 2,
                    "document_id": "doc1",
                    "time_start": 300.0,
                    "time_end": 400.0,
                },
            ]
        )

        # Query time=50 → None (no leaf has time_start <= 50)
        result = node_repo.get_leaf_at_time_position("doc1", 50.0, "end")
        assert result is None

        # Query time=100 → leaf1 (100 <= 100)
        result = node_repo.get_leaf_at_time_position("doc1", 100.0, "end")
        assert result is not None
        assert result.id == "leaf1"

        # Query time=150 → leaf1 (100 <= 150, but 200 > 150)
        result = node_repo.get_leaf_at_time_position("doc1", 150.0, "end")
        assert result is not None
        assert result.id == "leaf1"

        # Query time=250 → leaf2 (200 <= 250, but 300 > 250)
        result = node_repo.get_leaf_at_time_position("doc1", 250.0, "end")
        assert result is not None
        assert result.id == "leaf2"

        # Query time=350 → leaf3 (300 <= 350)
        result = node_repo.get_leaf_at_time_position("doc1", 350.0, "end")
        assert result is not None
        assert result.id == "leaf3"

        # Query time=500 → leaf3 (300 <= 500, latest)
        result = node_repo.get_leaf_at_time_position("doc1", 500.0, "end")
        assert result is not None
        assert result.id == "leaf3"

    def test_returns_span_info_for_mapping(
        self,
        sqlite_repos: tuple[
            SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager
        ],
    ) -> None:
        """Returned leaf includes span_start and span_end for time→span mapping."""
        node_repo, doc_repo, _ = sqlite_repos

        doc_repo.add_document("doc1", None, "test-model", "test-model")

        node_repo.add_nodes_batch(
            [
                {
                    "node_id": "leaf1",
                    "text": "Hello world",
                    "span_start": 0,
                    "span_end": 11,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 0,
                    "document_id": "doc1",
                    "time_start": 1000.0,
                    "time_end": 2000.0,
                },
            ]
        )

        result = node_repo.get_leaf_at_time_position("doc1", 1500.0, "start")
        assert result is not None
        assert result.span_start == 0
        assert result.span_end == 11

    def test_only_considers_leaves_not_inner_nodes(
        self,
        sqlite_repos: tuple[
            SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager
        ],
    ) -> None:
        """Only leaf nodes (height=0) are considered, not inner nodes."""
        node_repo, doc_repo, _ = sqlite_repos

        doc_repo.add_document("doc1", None, "test-model", "test-model")

        # Create a leaf and an inner node both with timestamps
        node_repo.add_nodes_batch(
            [
                {
                    "node_id": "leaf1",
                    "text": "Leaf text",
                    "span_start": 0,
                    "span_end": 10,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 0,
                    "document_id": "doc1",
                    "time_start": 100.0,
                    "time_end": 200.0,
                },
                {
                    "node_id": "inner1",
                    "text": "Inner summary",
                    "span_start": 0,
                    "span_end": 20,
                    "token_count": 4,
                    "height": 1,
                    "level_index": 0,
                    "document_id": "doc1",
                    "time_start": 100.0,
                    "time_end": 300.0,  # Wider time range
                    "left_child_id": "leaf1",
                },
            ]
        )

        # Query at time 250 - inner node covers this but leaf doesn't
        # Should return None because only leaves are considered
        result = node_repo.get_leaf_at_time_position("doc1", 250.0, "start")
        assert result is None

    def test_filters_by_document_id(
        self,
        sqlite_repos: tuple[
            SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager
        ],
    ) -> None:
        """Only returns leaves from the specified document."""
        node_repo, doc_repo, _ = sqlite_repos

        doc_repo.add_document("doc1", None, "test-model", "test-model")
        doc_repo.add_document("doc2", None, "test-model", "test-model")

        # Add leaf to doc1
        node_repo.add_nodes_batch(
            [
                {
                    "node_id": "leaf1",
                    "text": "Doc1 text",
                    "span_start": 0,
                    "span_end": 10,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 0,
                    "document_id": "doc1",
                    "time_start": 100.0,
                    "time_end": 200.0,
                },
            ]
        )

        # Add leaf to doc2 with overlapping time
        node_repo.add_nodes_batch(
            [
                {
                    "node_id": "leaf2",
                    "text": "Doc2 text",
                    "span_start": 0,
                    "span_end": 10,
                    "token_count": 2,
                    "height": 0,
                    "level_index": 0,
                    "document_id": "doc2",
                    "time_start": 50.0,
                    "time_end": 150.0,
                },
            ]
        )

        # Query doc1 - should find leaf1
        result = node_repo.get_leaf_at_time_position("doc1", 150.0, "start")
        assert result is not None
        assert result.id == "leaf1"

        # Query doc2 - should find leaf2 (not leaf1)
        result = node_repo.get_leaf_at_time_position("doc2", 150.0, "start")
        assert result is not None
        assert result.id == "leaf2"

    def test_with_single_timestamp_leaves(
        self,
        sqlite_repos: tuple[
            SqliteNodeRepository, SqliteDocumentRepository, SqliteDatabaseManager
        ],
    ) -> None:
        """Works correctly when leaves have time_start == time_end (point timestamps)."""
        node_repo, doc_repo, _ = sqlite_repos

        doc_repo.add_document("doc1", None, "test-model", "test-model")

        # Create leaves with point timestamps (time_start == time_end)
        node_repo.add_nodes_batch(
            [
                {
                    "node_id": "leaf1",
                    "text": "A",
                    "span_start": 0,
                    "span_end": 1,
                    "token_count": 1,
                    "height": 0,
                    "level_index": 0,
                    "document_id": "doc1",
                    "time_start": 100.0,
                    "time_end": 100.0,  # Point timestamp
                },
                {
                    "node_id": "leaf2",
                    "text": "B",
                    "span_start": 1,
                    "span_end": 2,
                    "token_count": 1,
                    "height": 0,
                    "level_index": 1,
                    "document_id": "doc1",
                    "time_start": 200.0,
                    "time_end": 200.0,  # Point timestamp
                },
            ]
        )

        # Query at exact point timestamp
        result = node_repo.get_leaf_at_time_position("doc1", 100.0, "start")
        assert result is not None
        assert result.id == "leaf1"

        result = node_repo.get_leaf_at_time_position("doc1", 100.0, "end")
        assert result is not None
        assert result.id == "leaf1"

        # Query between points
        result = node_repo.get_leaf_at_time_position("doc1", 150.0, "start")
        assert result is not None
        assert result.id == "leaf2"  # First leaf where 150 <= time_end

        result = node_repo.get_leaf_at_time_position("doc1", 150.0, "end")
        assert result is not None
        assert result.id == "leaf1"  # Latest leaf where time_start <= 150
