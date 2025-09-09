"""SQLite-backed test to ensure document isolation prevents cross-document contamination.

This converts the contamination test to run against the real in-memory
SQLite backend using the `sqlite_backend` and `sqlite_store_factory`
fixtures. Use this as a template for migrating similar tests.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ragzoom.document_store import DocumentStore
from ragzoom.retrieval.coverage_builder import CoverageBuilder


@pytest.mark.usefixtures("sqlite_backend")
def test_document_isolation_prevents_contamination_sqlite(
    sqlite_store_factory: Callable[[str | None], DocumentStore],
) -> None:
    """CoverageBuilder only includes nodes from the specified document."""
    # Create document-scoped stores
    doc1_store = sqlite_store_factory("doc1")
    doc2_store = sqlite_store_factory("doc2")

    # Add nodes for doc1
    doc1_store.nodes.add_batch(
        [
            {
                "node_id": "doc1_root",
                "text": "Document 1 root",
                "span_start": 0,
                "span_end": 100,
                "parent_id": None,
                "embedding": [0.5] * 1536,
                "left_child_id": "doc1_left",
                "right_child_id": "doc1_right",
                "path": "",
            },
            {
                "node_id": "doc1_left",
                "text": "Document 1 left",
                "span_start": 0,
                "span_end": 50,
                "parent_id": "doc1_root",
                "embedding": [0.5] * 1536,
                "path": "0",
            },
            {
                "node_id": "doc1_right",
                "text": "Document 1 right",
                "span_start": 50,
                "span_end": 100,
                "parent_id": "doc1_root",
                "embedding": [0.5] * 1536,
                "path": "1",
            },
        ]
    )

    # Add nodes for doc2
    doc2_store.nodes.add_batch(
        [
            {
                "node_id": "doc2_root",
                "text": "Document 2 root",
                "span_start": 0,
                "span_end": 100,
                "parent_id": None,
                "embedding": [0.5] * 1536,
                "left_child_id": "doc2_left",
                "right_child_id": "doc2_right",
                "path": "",
            },
            {
                "node_id": "doc2_left",
                "text": "Document 2 left",
                "span_start": 0,
                "span_end": 50,
                "parent_id": "doc2_root",
                "embedding": [0.5] * 1536,
                "path": "0",
            },
            {
                "node_id": "doc2_right",
                "text": "Document 2 right",
                "span_start": 50,
                "span_end": 100,
                "parent_id": "doc2_root",
                "embedding": [0.5] * 1536,
                "path": "1",
            },
        ]
    )

    # Build coverage map for doc1_left within doc1 scope
    coverage_builder = CoverageBuilder(doc1_store)
    coverage_map = coverage_builder.build_coverage_map(["doc1_left"])

    # Verify only doc1 nodes are included
    assert "doc1_left" in coverage_map
    assert "doc1_right" in coverage_map  # Sibling included
    assert "doc1_root" in coverage_map  # Ancestor included

    # Verify doc2 nodes are NOT included
    assert "doc2_left" not in coverage_map
    assert "doc2_right" not in coverage_map
    assert "doc2_root" not in coverage_map

    # Also test retrieval through document store
    doc1_nodes = doc1_store.nodes.get_all()
    doc1_node_ids = {node.id for node in doc1_nodes}

    assert doc1_node_ids == {"doc1_root", "doc1_left", "doc1_right"}
    assert "doc2_root" not in doc1_node_ids
    assert "doc2_left" not in doc1_node_ids
    assert "doc2_right" not in doc1_node_ids
