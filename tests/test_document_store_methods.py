"""SQLite-based tests for DocumentStore methods.

SQLite-based tests for new DocumentStore methods added in Phase 4
with the real in-memory SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.document_store import DocumentStore


@pytest.mark.usefixtures("sqlite_backend")
class TestDocumentStoreMethodsSQLite:
    """Test the new methods added to DocumentStore for Phase 4."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("doc1")

    def test_get_embedding_model(
        self, doc_store: DocumentStore, sqlite_backend: object
    ) -> None:
        """Test that DocumentStore correctly retrieves embedding model."""
        # Insert document with metadata using the document repository
        doc_repo = sqlite_backend.doc_repo  # type: ignore[attr-defined]
        doc_repo.add_document(
            document_id="doc1",
            file_path=None,
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4",
        )

        # Test getting embedding model
        model = doc_store.get_embedding_model()
        assert model == "text-embedding-3-small"

    def test_get_embedding_model_missing(
        self, doc_store: DocumentStore, sqlite_backend: object
    ) -> None:
        """Test that DocumentStore returns None when embedding model is missing."""
        # Insert document without embedding_model
        doc_repo = sqlite_backend.doc_repo  # type: ignore[attr-defined]
        doc_repo.add_document(
            document_id="doc1",
            file_path=None,
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="",
            summary_model="gpt-4",
        )

        # Test getting embedding model
        model = doc_store.get_embedding_model()
        assert model == "" or model is None  # SQLite uses empty string instead of null

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
                "path": "00",
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
                "path": "01",
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
                "path": "10",
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
                "path": "0",
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

    def test_document_id_mismatch_safety(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test that DocumentStore validates document ID matches."""
        # Create separate document stores for doc1 and doc2
        doc1_store = sqlite_store_factory("doc1")
        doc2_store = sqlite_store_factory("doc2")

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
                "path": "0",
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
                "path": "0",
            }
        ]
        doc2_store.nodes.add_batch(doc2_nodes)

        # Verify doc1_store can get doc1 node
        node1 = doc1_store.nodes.get("doc1_node")
        assert node1 is not None
        assert node1.id == "doc1_node"

        # Verify doc1_store cannot get doc2 node (should be filtered out)
        node2 = doc1_store.nodes.get("doc2_node")
        assert node2 is None  # Should be filtered out

    def test_cross_document_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test that DocumentStore with None document_id allows cross-document access."""
        # Create cross-document store
        cross_store = sqlite_store_factory(None)

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
                "path": "0",
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
                "path": "0",
            },
        ]
        cross_store.nodes.add_batch(nodes)

        # Should be able to access both documents
        node1 = cross_store.nodes.get("doc1_node")
        assert node1 is not None

        node2 = cross_store.nodes.get("doc2_node")
        assert node2 is not None
