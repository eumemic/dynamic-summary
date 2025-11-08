"""Backend-agnostic tests for DocumentStore methods.

Tests for new DocumentStore methods added in Phase 4
with the configured backend.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore


class TestDocumentStoreMethods:
    """Test the new methods added to DocumentStore for Phase 4."""

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> DocumentStore:
        doc_store = storage_backend.for_document("doc1")
        doc_store.set_metadata(
            file_path="test_file.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        return doc_store

    def test_get_embedding_model(
        self, doc_store: DocumentStore, storage_backend: StorageBackend
    ) -> None:
        """Test that DocumentStore correctly retrieves embedding model."""
        # Document metadata already set in fixture

        # Test getting embedding model
        model = doc_store.get_embedding_model()
        assert model == "text-embedding-3-small"

    def test_get_embedding_model_missing(self, storage_backend: StorageBackend) -> None:
        """Test that DocumentStore returns None when embedding model is missing."""
        # Create doc store with empty embedding model
        doc_store = storage_backend.for_document("doc1")
        doc_store.set_metadata(
            file_path="test_file.txt",
            embedding_model="",
            summary_model="gpt-4o-mini",
        )

        # Test getting embedding model
        model = doc_store.get_embedding_model()
        assert model == "" or model is None

    def test_get_avg_leaf_tokens(self, doc_store: DocumentStore) -> None:
        """Test that DocumentStore correctly calculates average leaf tokens."""
        # Seed leaf nodes with different token counts
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leaf_0",
                "text": "Leaf text 0",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 100,
                "height": 0,
            },
            {
                "node_id": "leaf_1",
                "text": "Leaf text 1",
                "embedding": [],
                "span_start": 100,
                "span_end": 200,
                "document_id": "doc1",
                "token_count": 150,
                "height": 0,
            },
            {
                "node_id": "leaf_2",
                "text": "Leaf text 2",
                "embedding": [],
                "span_start": 200,
                "span_end": 300,
                "document_id": "doc1",
                "token_count": 200,
                "height": 0,
            },
            # Parent node (not a leaf)
            {
                "node_id": "parent",
                "text": "Parent text",
                "embedding": [],
                "span_start": 0,
                "span_end": 300,
                "document_id": "doc1",
                "token_count": 300,
                "height": 1,
                "left_child_id": "leaf_0",
                "right_child_id": "leaf_1",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("leaf_0", "parent"), ("leaf_1", "parent")]
        )

        # Test getting average leaf tokens
        avg_tokens = doc_store.get_avg_leaf_tokens()
        # Average of 100, 150, 200 = 150
        assert avg_tokens == 150

    def test_get_avg_leaf_tokens_no_leaves(self, doc_store: DocumentStore) -> None:
        """Test that DocumentStore returns None when no leaf nodes exist."""
        # Test getting average leaf tokens from empty document
        avg_tokens = doc_store.get_avg_leaf_tokens()
        assert avg_tokens is None

    def test_document_id_mismatch_safety(self, storage_backend: StorageBackend) -> None:
        """Test that DocumentStore validates document ID matches."""
        # Create separate document stores for doc1 and doc2
        doc1_store = storage_backend.for_document("doc1")
        doc1_store.set_metadata(
            file_path="test1.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc2_store = storage_backend.for_document("doc2")
        doc2_store.set_metadata(
            file_path="test2.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add node to doc1
        doc1_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc1_node",
                "text": "Doc 1 content",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 100,
                "height": 0,
            }
        ]
        doc1_store.nodes.add_batch(doc1_nodes)

        # Add node to doc2
        doc2_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc2_node",
                "text": "Doc 2 content",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc2",
                "token_count": 100,
                "height": 0,
            }
        ]
        doc2_store.nodes.add_batch(doc2_nodes)

        # Verify doc1_store can get doc1 node
        node1 = doc1_store.nodes.get_node("doc1_node")
        assert node1 is not None
        assert node1.id == "doc1_node"

        # Verify doc1_store cannot get doc2 node (should be filtered out)
        node2 = doc1_store.nodes.get_node("doc2_node")
        assert node2 is None  # Should be filtered out

    def test_cross_document_store(self, storage_backend: StorageBackend) -> None:
        """Test that DocumentStore with None document_id allows cross-document access."""
        # Create cross-document store
        cross_store = storage_backend.for_document(None)

        # Seed nodes to different documents
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc1_node",
                "text": "Doc 1 content",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 100,
                "height": 0,
            },
            {
                "node_id": "doc2_node",
                "text": "Doc 2 content",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc2",
                "token_count": 100,
                "height": 0,
            },
        ]
        cross_store.nodes.add_batch(nodes)

        # Should be able to access both documents
        node1 = cross_store.nodes.get_node("doc1_node")
        assert node1 is not None

        node2 = cross_store.nodes.get_node("doc2_node")
        assert node2 is not None

    def test_get_nodes_in_span_requires_document_scope(
        self, storage_backend: StorageBackend
    ) -> None:
        """Span queries should fail when the store is not scoped to a document."""
        cross_store = storage_backend.for_document(None)
        with pytest.raises(ValueError, match="document scope"):
            cross_store.get_nodes_in_span(0, 100, limit=10)
