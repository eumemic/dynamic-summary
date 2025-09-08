"""Backend-agnostic document isolation tests to prevent cross-document contamination.

Tests for document isolation to ensure the CoverageBuilder
only includes nodes from the specified document.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.retrieval.coverage_builder import CoverageBuilder


class TestContaminationBug:
    @pytest.fixture
    def doc1_store(self, storage_backend: StorageBackend) -> DocumentStore:
        doc_store = storage_backend.for_document("doc1")

        # Set up document metadata
        doc_store.set_metadata(
            file_path="contamination_doc1.txt",
            content_hash="contamination-doc1-hash",
            chunk_count=3,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        return doc_store

    @pytest.fixture
    def doc2_store(self, storage_backend: StorageBackend) -> DocumentStore:
        doc_store = storage_backend.for_document("doc2")

        # Set up document metadata
        doc_store.set_metadata(
            file_path="contamination_doc2.txt",
            content_hash="contamination-doc2-hash",
            chunk_count=3,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        return doc_store

    def test_document_isolation_prevents_contamination(
        self, doc1_store: DocumentStore, doc2_store: DocumentStore
    ) -> None:
        """Test that coverage builder only includes nodes from the specified document."""
        # Add nodes for doc1
        doc1_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc1_root",
                "text": "Document 1 root",
                "span_start": 0,
                "span_end": 100,
                "parent_id": None,
                "document_id": "doc1",
                "token_count": 50,
                "height": 1,
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
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "path": "0",
            },
            {
                "node_id": "doc1_right",
                "text": "Document 1 right",
                "span_start": 50,
                "span_end": 100,
                "parent_id": "doc1_root",
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "path": "1",
            },
        ]
        doc1_store.nodes.add_batch(doc1_nodes)
        doc1_store.nodes.update_parent_references_batch(
            [("doc1_left", "doc1_root"), ("doc1_right", "doc1_root")]
        )

        # Add nodes for doc2 using separate document store
        doc2_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc2_root",
                "text": "Document 2 root",
                "span_start": 0,
                "span_end": 100,
                "parent_id": None,
                "document_id": "doc2",
                "token_count": 50,
                "height": 1,
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
                "document_id": "doc2",
                "token_count": 25,
                "height": 0,
                "path": "0",
            },
            {
                "node_id": "doc2_right",
                "text": "Document 2 right",
                "span_start": 50,
                "span_end": 100,
                "parent_id": "doc2_root",
                "document_id": "doc2",
                "token_count": 25,
                "height": 0,
                "path": "1",
            },
        ]
        doc2_store.nodes.add_batch(doc2_nodes)
        doc2_store.nodes.update_parent_references_batch(
            [("doc2_left", "doc2_root"), ("doc2_right", "doc2_root")]
        )

        # Build coverage map for doc1_left using document-scoped store
        coverage_builder = CoverageBuilder(doc1_store)
        coverage_map = coverage_builder.build_coverage_map(["doc1_left"])

        # Verify only doc1 nodes are included
        assert "doc1_left" in coverage_map
        assert "doc1_right" in coverage_map  # Sibling included
        assert "doc1_root" in coverage_map  # Ancestor included

        # Verify doc2 nodes are NOT included (contamination prevented)
        assert "doc2_left" not in coverage_map
        assert "doc2_right" not in coverage_map
        assert "doc2_root" not in coverage_map

        # Also test retrieval through document store
        doc1_nodes_retrieved = doc1_store.nodes.get_all()
        doc1_node_ids = {node.id for node in doc1_nodes_retrieved}

        assert doc1_node_ids == {"doc1_root", "doc1_left", "doc1_right"}
        assert "doc2_root" not in doc1_node_ids
        assert "doc2_left" not in doc1_node_ids
        assert "doc2_right" not in doc1_node_ids
