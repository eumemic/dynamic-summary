"""Tests for tree completion frontier computation."""

from __future__ import annotations

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend


def test_tree_frontier_single_root(storage_backend: StorageBackend) -> None:
    """Single root: frontier = root's span_end."""
    store = storage_backend.for_document("doc-frontier-single")

    # Create a complete tree with one root
    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf_left",
            "text": "left",
            "span_start": 0,
            "span_end": 50,
            "parent_id": "root",
            "document_id": "doc-frontier-single",
            "token_count": 10,
            "height": 0,
            "level_index": 0,
        },
        {
            "node_id": "leaf_right",
            "text": "right",
            "span_start": 50,
            "span_end": 100,
            "parent_id": "root",
            "document_id": "doc-frontier-single",
            "token_count": 10,
            "height": 0,
            "level_index": 1,
        },
        {
            "node_id": "root",
            "text": "summary",
            "span_start": 0,
            "span_end": 100,
            "parent_id": None,
            "left_child_id": "leaf_left",
            "right_child_id": "leaf_right",
            "document_id": "doc-frontier-single",
            "token_count": 20,
            "height": 1,
            "level_index": 0,
        },
    ]
    store.nodes.add_batch(nodes)

    frontier = store.nodes.get_tree_completion_frontier()
    assert frontier == 100  # The root's span_end


def test_tree_frontier_forest(storage_backend: StorageBackend) -> None:
    """Multiple roots (forest): frontier = first root's span_end."""
    store = storage_backend.for_document("doc-frontier-forest")

    # Create a forest with two roots
    nodes: list[NodeDataDict] = [
        # First tree (root1 at span 0-50)
        {
            "node_id": "leaf1",
            "text": "leaf1",
            "span_start": 0,
            "span_end": 25,
            "parent_id": "root1",
            "document_id": "doc-frontier-forest",
            "token_count": 5,
            "height": 0,
            "level_index": 0,
        },
        {
            "node_id": "leaf2",
            "text": "leaf2",
            "span_start": 25,
            "span_end": 50,
            "parent_id": "root1",
            "document_id": "doc-frontier-forest",
            "token_count": 5,
            "height": 0,
            "level_index": 1,
        },
        {
            "node_id": "root1",
            "text": "summary1",
            "span_start": 0,
            "span_end": 50,
            "parent_id": None,
            "left_child_id": "leaf1",
            "right_child_id": "leaf2",
            "document_id": "doc-frontier-forest",
            "token_count": 10,
            "height": 1,
            "level_index": 0,
        },
        # Second tree (root2 at span 50-100) - NOT yet complete
        {
            "node_id": "leaf3",
            "text": "leaf3",
            "span_start": 50,
            "span_end": 75,
            "parent_id": "root2",
            "document_id": "doc-frontier-forest",
            "token_count": 5,
            "height": 0,
            "level_index": 2,
        },
        {
            "node_id": "leaf4",
            "text": "leaf4",
            "span_start": 75,
            "span_end": 100,
            "parent_id": "root2",
            "document_id": "doc-frontier-forest",
            "token_count": 5,
            "height": 0,
            "level_index": 3,
        },
        {
            "node_id": "root2",
            "text": "summary2",
            "span_start": 50,
            "span_end": 100,
            "parent_id": None,
            "left_child_id": "leaf3",
            "right_child_id": "leaf4",
            "document_id": "doc-frontier-forest",
            "token_count": 10,
            "height": 1,
            "level_index": 1,
        },
    ]
    store.nodes.add_batch(nodes)

    frontier = store.nodes.get_tree_completion_frontier()
    # First root (ordered by span_start) is root1 with span_end=50
    assert frontier == 50


def test_tree_frontier_empty_document(storage_backend: StorageBackend) -> None:
    """Empty document: frontier = 0."""
    store = storage_backend.for_document("doc-frontier-empty")

    frontier = store.nodes.get_tree_completion_frontier()
    assert frontier == 0


def test_tree_frontier_leaves_only(storage_backend: StorageBackend) -> None:
    """Leaves only (no summaries yet): frontier = 0."""
    store = storage_backend.for_document("doc-frontier-leaves")

    # Only leaves, no root nodes
    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf1",
            "text": "leaf1",
            "span_start": 0,
            "span_end": 50,
            "parent_id": None,  # No parent yet
            "document_id": "doc-frontier-leaves",
            "token_count": 10,
            "height": 0,
            "level_index": 0,
        },
        {
            "node_id": "leaf2",
            "text": "leaf2",
            "span_start": 50,
            "span_end": 100,
            "parent_id": None,  # No parent yet
            "document_id": "doc-frontier-leaves",
            "token_count": 10,
            "height": 0,
            "level_index": 1,
        },
    ]
    store.nodes.add_batch(nodes)

    # Leaves have parent_id=None, but height=0, so they're roots by parentless
    # but the frontier should be the first one's span_end
    frontier = store.nodes.get_tree_completion_frontier()
    # First leaf (by span_start) has span_end=50
    assert frontier == 50
