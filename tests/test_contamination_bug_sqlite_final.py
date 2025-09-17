"""SQLite-based tests for document isolation to prevent cross-document contamination.

These tests ensure that coverage builder only includes nodes from the specified
document using the real in-memory SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.document_store import DocumentStore


@pytest.mark.usefixtures("sqlite_backend")
class TestContaminationBugSQLite:
    """Test that document isolation prevents cross-document contamination."""

    @pytest.fixture
    def doc1_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("doc1")

    @pytest.fixture
    def doc2_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("doc2")

    def test_document_isolation_prevents_contamination(
        self, doc1_store: DocumentStore, doc2_store: DocumentStore
    ) -> None:
        """Test that document stores only contain nodes from their specified document."""
        # Add nodes for doc1 to doc1_store
        doc1_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc1_root",
                "text": "Document 1 root",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 17,
                "height": 1,
                "left_child_id": "doc1_left",
                "right_child_id": "doc1_right",
            },
            {
                "node_id": "doc1_left",
                "text": "Document 1 left",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 16,
                "height": 0,
                "parent_id": "doc1_root",
            },
            {
                "node_id": "doc1_right",
                "text": "Document 1 right",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 17,
                "height": 0,
                "parent_id": "doc1_root",
            },
        ]
        doc1_store.nodes.add_batch(doc1_nodes)
        doc1_store.nodes.update_parent_references_batch(
            [
                ("doc1_left", "doc1_root"),
                ("doc1_right", "doc1_root"),
            ]
        )

        # Add nodes for doc2 to doc2_store
        doc2_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc2_root",
                "text": "Document 2 root",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc2",
                "token_count": 17,
                "height": 1,
                "left_child_id": "doc2_left",
                "right_child_id": "doc2_right",
            },
            {
                "node_id": "doc2_left",
                "text": "Document 2 left",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc2",
                "token_count": 16,
                "height": 0,
                "parent_id": "doc2_root",
            },
            {
                "node_id": "doc2_right",
                "text": "Document 2 right",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc2",
                "token_count": 17,
                "height": 0,
                "parent_id": "doc2_root",
            },
        ]
        doc2_store.nodes.add_batch(doc2_nodes)
        doc2_store.nodes.update_parent_references_batch(
            [
                ("doc2_left", "doc2_root"),
                ("doc2_right", "doc2_root"),
            ]
        )

        # Test isolation: doc1_store should only see doc1 nodes
        doc1_nodes_retrieved = doc1_store.nodes.get_all()
        doc1_node_ids = {node.id for node in doc1_nodes_retrieved}

        assert doc1_node_ids == {"doc1_root", "doc1_left", "doc1_right"}
        assert "doc2_root" not in doc1_node_ids
        assert "doc2_left" not in doc1_node_ids
        assert "doc2_right" not in doc1_node_ids

        # Test isolation: doc2_store should only see doc2 nodes
        doc2_nodes_retrieved = doc2_store.nodes.get_all()
        doc2_node_ids = {node.id for node in doc2_nodes_retrieved}

        assert doc2_node_ids == {"doc2_root", "doc2_left", "doc2_right"}
        assert "doc1_root" not in doc2_node_ids
        assert "doc1_left" not in doc2_node_ids
        assert "doc1_right" not in doc2_node_ids

        # Test node retrieval by ID
        doc1_left_node = doc1_store.nodes.get_node("doc1_left")
        assert doc1_left_node is not None
        assert doc1_left_node.id == "doc1_left"

        # doc1_store should not be able to retrieve doc2 nodes
        doc2_left_node = doc1_store.nodes.get_node("doc2_left")
        assert doc2_left_node is None
