"""SQLite-specific worker coordinator repository tests.

These cover behaviour tied to the SQLite repository implementation, while core
coordinator logic is exercised via backend-agnostic tests.
"""

from collections.abc import Generator

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.contracts.tree_node import TreeNode


@pytest.fixture()
def sqlite_backend() -> Generator[SQLiteStorageBackend, None, None]:
    backend = SQLiteStorageBackend()
    try:
        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-5-mini",
        )
        yield backend
    finally:
        backend.close()


def _insert(
    backend: SQLiteStorageBackend,
    *,
    node_id: str,
    height: int,
    level_index: int,
    span_start: int,
    span_end: int,
    parent_id: str | None,
    preceding: str | None,
    following: str | None,
) -> TreeNode:
    store = backend.for_document("doc")
    (node,) = store.nodes.add_batch(
        [
            {
                "node_id": node_id,
                "text": node_id,
                "span_start": span_start,
                "span_end": span_end,
                "parent_id": parent_id,
                "left_child_id": None,
                "right_child_id": None,
                "document_id": "doc",
                "token_count": span_end - span_start,
                "height": height,
                "preceding_neighbor_id": preceding,
                "following_neighbor_id": following,
                "level_index": level_index,
            }
        ]
    )
    return node


def test_parentless_nodes_sorted_by_height_and_level(
    sqlite_backend: SQLiteStorageBackend,
) -> None:
    sqlite_backend.clear_document("doc")
    _insert(
        sqlite_backend,
        node_id="leaf-right",
        height=0,
        level_index=1,
        span_start=10,
        span_end=20,
        parent_id=None,
        preceding="leaf-left",
        following=None,
    )
    _insert(
        sqlite_backend,
        node_id="leaf-left",
        height=0,
        level_index=0,
        span_start=0,
        span_end=10,
        parent_id=None,
        preceding=None,
        following="leaf-right",
    )
    _insert(
        sqlite_backend,
        node_id="intermediate",
        height=1,
        level_index=0,
        span_start=0,
        span_end=20,
        parent_id=None,
        preceding=None,
        following=None,
    )

    store = sqlite_backend.for_document("doc")
    nodes = store.nodes.get_parentless_nodes()

    ordered_ids = [node.id for node in nodes]
    assert ordered_ids == ["leaf-left", "leaf-right", "intermediate"]


def test_parentless_nodes_respects_document_scope(
    sqlite_backend: SQLiteStorageBackend,
) -> None:
    sqlite_backend.clear_document("doc")
    other_store = sqlite_backend.add_document(
        document_id="other",
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )
    other_store.nodes.add_batch(
        [
            {
                "node_id": "foreign",
                "text": "foreign",
                "span_start": 0,
                "span_end": 5,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
                "document_id": "other",
                "token_count": 5,
                "height": 0,
                "preceding_neighbor_id": None,
                "following_neighbor_id": None,
                "level_index": 0,
            }
        ]
    )

    _insert(
        sqlite_backend,
        node_id="doc-node",
        height=0,
        level_index=0,
        span_start=0,
        span_end=5,
        parent_id=None,
        preceding=None,
        following=None,
    )

    nodes = sqlite_backend.for_document("doc").nodes.get_parentless_nodes()
    ids = {node.id for node in nodes}
    assert ids == {"doc-node"}


def test_ready_left_children_returns_ids(sqlite_backend: SQLiteStorageBackend) -> None:
    sqlite_backend.clear_document("doc")
    store = sqlite_backend.for_document("doc")
    _insert(
        sqlite_backend,
        node_id="left",
        height=0,
        level_index=0,
        span_start=0,
        span_end=10,
        parent_id=None,
        preceding=None,
        following="right",
    )
    _insert(
        sqlite_backend,
        node_id="right",
        height=0,
        level_index=1,
        span_start=10,
        span_end=20,
        parent_id=None,
        preceding="left",
        following=None,
    )

    assert store.nodes.get_ready_left_children() == ["left"]
