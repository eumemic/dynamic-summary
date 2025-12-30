"""Tests for streaming iterator methods in node repositories."""

from __future__ import annotations

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend


def test_iter_root_nodes_returns_only_parentless(
    storage_backend: StorageBackend,
) -> None:
    """iter_root_nodes should only return nodes without parents."""
    store = storage_backend.for_document("doc-iter-roots")

    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf_left",
            "text": "left",
            "span_start": 0,
            "span_end": 5,
            "parent_id": "parent",
            "document_id": "doc-iter-roots",
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
            "document_id": "doc-iter-roots",
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
            "document_id": "doc-iter-roots",
            "token_count": 4,
            "height": 1,
            "level_index": 0,
        },
    ]
    store.nodes.add_batch(nodes)

    root_ids = [node.id for node in store.nodes.iter_root_nodes()]
    assert root_ids == ["parent"]


def test_iter_root_nodes_ordered_by_span_start(
    storage_backend: StorageBackend,
) -> None:
    """iter_root_nodes should return nodes ordered by span_start."""
    store = storage_backend.for_document("doc-iter-roots-order")

    nodes: list[NodeDataDict] = [
        {
            "node_id": "root_c",
            "text": "third",
            "span_start": 20,
            "span_end": 30,
            "parent_id": None,
            "document_id": "doc-iter-roots-order",
            "token_count": 2,
            "height": 0,
            "level_index": 2,
        },
        {
            "node_id": "root_a",
            "text": "first",
            "span_start": 0,
            "span_end": 10,
            "parent_id": None,
            "document_id": "doc-iter-roots-order",
            "token_count": 2,
            "height": 0,
            "level_index": 0,
        },
        {
            "node_id": "root_b",
            "text": "second",
            "span_start": 10,
            "span_end": 20,
            "parent_id": None,
            "document_id": "doc-iter-roots-order",
            "token_count": 2,
            "height": 0,
            "level_index": 1,
        },
    ]
    store.nodes.add_batch(nodes)

    root_ids = [node.id for node in store.nodes.iter_root_nodes()]
    assert root_ids == ["root_a", "root_b", "root_c"]


def test_iter_leaves_returns_only_height_zero(
    storage_backend: StorageBackend,
) -> None:
    """iter_leaves should only return nodes with height 0."""
    store = storage_backend.for_document("doc-iter-leaves")

    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf_left",
            "text": "left",
            "span_start": 0,
            "span_end": 5,
            "parent_id": "parent",
            "document_id": "doc-iter-leaves",
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
            "document_id": "doc-iter-leaves",
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
            "document_id": "doc-iter-leaves",
            "token_count": 4,
            "height": 1,
            "level_index": 0,
        },
    ]
    store.nodes.add_batch(nodes)

    leaf_ids = [node.id for node in store.nodes.iter_leaves()]
    assert set(leaf_ids) == {"leaf_left", "leaf_right"}


def test_iter_leaves_ordered_by_span_start(
    storage_backend: StorageBackend,
) -> None:
    """iter_leaves should return nodes ordered by span_start."""
    store = storage_backend.for_document("doc-iter-leaves-order")

    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf_c",
            "text": "third",
            "span_start": 20,
            "span_end": 30,
            "parent_id": None,
            "document_id": "doc-iter-leaves-order",
            "token_count": 2,
            "height": 0,
            "level_index": 2,
        },
        {
            "node_id": "leaf_a",
            "text": "first",
            "span_start": 0,
            "span_end": 10,
            "parent_id": None,
            "document_id": "doc-iter-leaves-order",
            "token_count": 2,
            "height": 0,
            "level_index": 0,
        },
        {
            "node_id": "leaf_b",
            "text": "second",
            "span_start": 10,
            "span_end": 20,
            "parent_id": None,
            "document_id": "doc-iter-leaves-order",
            "token_count": 2,
            "height": 0,
            "level_index": 1,
        },
    ]
    store.nodes.add_batch(nodes)

    leaf_ids = [node.id for node in store.nodes.iter_leaves()]
    assert leaf_ids == ["leaf_a", "leaf_b", "leaf_c"]


def test_iter_root_nodes_empty_document(storage_backend: StorageBackend) -> None:
    """iter_root_nodes should return empty iterator for empty document."""
    store = storage_backend.for_document("doc-iter-empty")

    root_ids = list(store.nodes.iter_root_nodes())
    assert root_ids == []


def test_iter_leaves_empty_document(storage_backend: StorageBackend) -> None:
    """iter_leaves should return empty iterator for empty document."""
    store = storage_backend.for_document("doc-iter-empty-leaves")

    leaf_ids = list(store.nodes.iter_leaves())
    assert leaf_ids == []


def test_iter_root_nodes_scoped_to_document(
    storage_backend: StorageBackend,
) -> None:
    """iter_root_nodes should only return nodes for its document."""
    store_a = storage_backend.for_document("doc-a")
    store_b = storage_backend.for_document("doc-b")

    nodes_a: list[NodeDataDict] = [
        {
            "node_id": "root_a",
            "text": "doc a",
            "span_start": 0,
            "span_end": 10,
            "parent_id": None,
            "document_id": "doc-a",
            "token_count": 2,
            "height": 0,
            "level_index": 0,
        },
    ]
    nodes_b: list[NodeDataDict] = [
        {
            "node_id": "root_b",
            "text": "doc b",
            "span_start": 0,
            "span_end": 10,
            "parent_id": None,
            "document_id": "doc-b",
            "token_count": 2,
            "height": 0,
            "level_index": 0,
        },
    ]
    store_a.nodes.add_batch(nodes_a)
    store_b.nodes.add_batch(nodes_b)

    root_ids_a = [node.id for node in store_a.nodes.iter_root_nodes()]
    root_ids_b = [node.id for node in store_b.nodes.iter_root_nodes()]

    assert root_ids_a == ["root_a"]
    assert root_ids_b == ["root_b"]


def test_iter_leaves_scoped_to_document(
    storage_backend: StorageBackend,
) -> None:
    """iter_leaves should only return nodes for its document."""
    store_a = storage_backend.for_document("doc-leaf-a")
    store_b = storage_backend.for_document("doc-leaf-b")

    nodes_a: list[NodeDataDict] = [
        {
            "node_id": "leaf_a",
            "text": "doc a",
            "span_start": 0,
            "span_end": 10,
            "parent_id": None,
            "document_id": "doc-leaf-a",
            "token_count": 2,
            "height": 0,
            "level_index": 0,
        },
    ]
    nodes_b: list[NodeDataDict] = [
        {
            "node_id": "leaf_b",
            "text": "doc b",
            "span_start": 0,
            "span_end": 10,
            "parent_id": None,
            "document_id": "doc-leaf-b",
            "token_count": 2,
            "height": 0,
            "level_index": 0,
        },
    ]
    store_a.nodes.add_batch(nodes_a)
    store_b.nodes.add_batch(nodes_b)

    leaf_ids_a = [node.id for node in store_a.nodes.iter_leaves()]
    leaf_ids_b = [node.id for node in store_b.nodes.iter_leaves()]

    assert leaf_ids_a == ["leaf_a"]
    assert leaf_ids_b == ["leaf_b"]
