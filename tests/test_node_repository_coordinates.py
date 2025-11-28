"""Tests for bulk coordinate lookups in node repositories."""

from __future__ import annotations

from collections.abc import Sequence

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.tree_coordinate import TreeCoordinate


def test_sqlite_coordinate_lookup(storage_backend: StorageBackend) -> None:
    store = storage_backend.for_document("doc-coordinate")

    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf_left",
            "text": "left",
            "span_start": 0,
            "span_end": 5,
            "parent_id": "parent",
            "document_id": "doc-coordinate",
            "token_count": 2,
            "height": 0,
            "level_index": 0,
        },
        {
            "node_id": "leaf_right",
            "text": "right",
            "span_start": 5,
            "span_end": 10,
            "parent_id": "parent",
            "document_id": "doc-coordinate",
            "token_count": 2,
            "height": 0,
            "level_index": 1,
        },
        {
            "node_id": "parent",
            "text": "summary",
            "span_start": 0,
            "span_end": 10,
            "parent_id": None,
            "document_id": "doc-coordinate",
            "token_count": 4,
            "height": 1,
            "level_index": 0,
        },
    ]
    store.nodes.add_batch(nodes)

    coords: Sequence[tuple[int, int]] = [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    fetched = store.nodes.get_by_height_levels(coords)
    fetched_ids = {node.id for node in fetched}

    assert fetched_ids == {"leaf_left", "leaf_right", "parent"}

    coord_objects = TreeCoordinate.unique(
        [TreeCoordinate(store.document_id, h, i) for h, i in coords]
    )
    refetched = store.nodes.get_by_height_levels([c.as_tuple() for c in coord_objects])
    assert {node.id for node in refetched} == fetched_ids
