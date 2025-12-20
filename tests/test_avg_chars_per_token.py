"""Tests for get_avg_chars_per_token in node repositories."""

from __future__ import annotations

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend


def test_avg_chars_per_token_empty_document(storage_backend: StorageBackend) -> None:
    """Returns None when document has no leaves."""
    store = storage_backend.for_document("doc-empty")

    result = store.nodes.get_avg_chars_per_token()
    assert result is None


def test_avg_chars_per_token_single_leaf(storage_backend: StorageBackend) -> None:
    """Returns correct ratio for a single leaf."""
    store = storage_backend.for_document("doc-single")

    # Single leaf: 100 chars, 25 tokens → avg = 4.0 chars/token
    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf1",
            "text": "x" * 100,
            "span_start": 0,
            "span_end": 100,
            "document_id": "doc-single",
            "token_count": 25,
            "height": 0,
            "level_index": 0,
        },
    ]
    store.nodes.add_batch(nodes)

    result = store.nodes.get_avg_chars_per_token()
    assert result is not None
    assert result == 4.0


def test_avg_chars_per_token_multiple_leaves(storage_backend: StorageBackend) -> None:
    """Returns aggregate ratio across multiple leaves."""
    store = storage_backend.for_document("doc-multi")

    # Three leaves with varying char/token ratios:
    # leaf1: 100 chars, 20 tokens (5.0 ratio)
    # leaf2: 80 chars, 20 tokens (4.0 ratio)
    # leaf3: 120 chars, 40 tokens (3.0 ratio)
    # Total: 300 chars, 80 tokens → avg = 3.75 chars/token
    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf1",
            "text": "a" * 100,
            "span_start": 0,
            "span_end": 100,
            "document_id": "doc-multi",
            "token_count": 20,
            "height": 0,
            "level_index": 0,
        },
        {
            "node_id": "leaf2",
            "text": "b" * 80,
            "span_start": 100,
            "span_end": 180,
            "document_id": "doc-multi",
            "token_count": 20,
            "height": 0,
            "level_index": 1,
        },
        {
            "node_id": "leaf3",
            "text": "c" * 120,
            "span_start": 180,
            "span_end": 300,
            "document_id": "doc-multi",
            "token_count": 40,
            "height": 0,
            "level_index": 2,
        },
    ]
    store.nodes.add_batch(nodes)

    result = store.nodes.get_avg_chars_per_token()
    assert result is not None
    assert result == 3.75


def test_avg_chars_per_token_ignores_inner_nodes(
    storage_backend: StorageBackend,
) -> None:
    """Only counts leaves (height=0), ignores inner nodes."""
    store = storage_backend.for_document("doc-tree")

    # Two leaves + one parent (height=1)
    # Leaves: 50+50=100 chars, 10+10=20 tokens → avg=5.0
    # Parent would skew result if counted (200 chars, 20 tokens)
    nodes: list[NodeDataDict] = [
        {
            "node_id": "leaf1",
            "text": "a" * 50,
            "span_start": 0,
            "span_end": 50,
            "document_id": "doc-tree",
            "token_count": 10,
            "height": 0,
            "level_index": 0,
        },
        {
            "node_id": "leaf2",
            "text": "b" * 50,
            "span_start": 50,
            "span_end": 100,
            "document_id": "doc-tree",
            "token_count": 10,
            "height": 0,
            "level_index": 1,
        },
        {
            "node_id": "parent",
            "text": "summary" * 28,  # ~200 chars
            "span_start": 0,
            "span_end": 100,
            "document_id": "doc-tree",
            "token_count": 20,
            "height": 1,
            "level_index": 0,
        },
    ]
    store.nodes.add_batch(nodes)

    result = store.nodes.get_avg_chars_per_token()
    assert result is not None
    # Only leaves: 100 chars / 20 tokens = 5.0
    assert result == 5.0


def test_avg_chars_per_token_document_specific(
    storage_backend: StorageBackend,
) -> None:
    """Filters by document_id when specified."""
    store_a = storage_backend.for_document("doc-a")
    store_b = storage_backend.for_document("doc-b")

    # Doc A: 100 chars, 25 tokens → 4.0
    nodes_a: list[NodeDataDict] = [
        {
            "node_id": "leaf_a",
            "text": "a" * 100,
            "span_start": 0,
            "span_end": 100,
            "document_id": "doc-a",
            "token_count": 25,
            "height": 0,
            "level_index": 0,
        },
    ]
    store_a.nodes.add_batch(nodes_a)

    # Doc B: 200 chars, 100 tokens → 2.0
    nodes_b: list[NodeDataDict] = [
        {
            "node_id": "leaf_b",
            "text": "b" * 200,
            "span_start": 0,
            "span_end": 200,
            "document_id": "doc-b",
            "token_count": 100,
            "height": 0,
            "level_index": 0,
        },
    ]
    store_b.nodes.add_batch(nodes_b)

    # Each should return its own average
    assert store_a.nodes.get_avg_chars_per_token() == 4.0
    assert store_b.nodes.get_avg_chars_per_token() == 2.0
