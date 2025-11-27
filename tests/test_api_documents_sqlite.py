"""SQLite-based tests for document API endpoints.

This file contains the converted tests from test_api_documents.py using the real
SQLite backend for higher fidelity testing. Full API endpoint testing requires
complex service integration that is beyond the scope of SQLite backend testing.
The core document isolation functionality is tested at the storage layer.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.contracts.vector_filter import DocumentIdFilter
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore


@pytest.mark.usefixtures("sqlite_backend")
class TestDocumentAPISQLite:
    """Test document-related functionality with SQLite backend.

    These tests verify the core document isolation and storage functionality
    that the API endpoints depend on, using the real SQLite backend for
    higher fidelity testing.
    """

    def test_sqlite_backend_document_storage_isolation(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test document storage isolation using SQLite backend.

        Verifies that different document stores properly isolate their data,
        which is fundamental to the API endpoints working correctly.
        """
        # Create separate document stores
        dragons_store = sqlite_store_factory("dragons-doc")
        wizards_store = sqlite_store_factory("wizards-doc")

        # Add nodes to dragons document
        dragon_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "dragon_node1",
                "text": "Dragons breathe fire and soar through skies",
                "span_start": 0,
                "span_end": 43,
                "document_id": "dragons-doc",
                "token_count": 8,
                "height": 0,
            }
        ]
        dragons_store.nodes.add_batch(dragon_nodes)

        # Add nodes to wizards document
        wizard_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "wizard_node1",
                "text": "Wizards practice magic and cast spells",
                "span_start": 0,
                "span_end": 38,
                "document_id": "wizards-doc",
                "token_count": 7,
                "height": 0,
            }
        ]
        wizards_store.nodes.add_batch(wizard_nodes)

        # Verify isolation - each store only sees its own data
        dragon_nodes_retrieved = dragons_store.nodes.get_all()
        assert len(dragon_nodes_retrieved) == 1
        assert dragon_nodes_retrieved[0].document_id == "dragons-doc"
        assert dragon_nodes_retrieved[0].id == "dragon_node1"
        assert "dragon" in dragon_nodes_retrieved[0].text.lower()

        wizard_nodes_retrieved = wizards_store.nodes.get_all()
        assert len(wizard_nodes_retrieved) == 1
        assert wizard_nodes_retrieved[0].document_id == "wizards-doc"
        assert wizard_nodes_retrieved[0].id == "wizard_node1"
        assert "wizard" in wizard_nodes_retrieved[0].text.lower()

        # Cross-document queries should fail
        assert dragons_store.nodes.get_node("wizard_node1") is None
        assert wizards_store.nodes.get_node("dragon_node1") is None

    def test_sqlite_backend_multi_document_isolation(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test isolation across multiple documents simultaneously."""
        # Create stores for three different documents
        doc_stores = {
            "tech-manual": sqlite_store_factory("tech-manual"),
            "recipe-book": sqlite_store_factory("recipe-book"),
            "story-collection": sqlite_store_factory("story-collection"),
        }

        # Add content to each document
        test_data: dict[str, dict[str, str | int]] = {
            "tech-manual": {
                "node_id": "tech_node1",
                "text": "Configure the server with SSL certificates",
                "token_count": 7,
            },
            "recipe-book": {
                "node_id": "recipe_node1",
                "text": "Mix flour and eggs until smooth",
                "token_count": 6,
            },
            "story-collection": {
                "node_id": "story_node1",
                "text": "Once upon a time in a distant kingdom",
                "token_count": 8,
            },
        }

        # Add nodes to each document store
        for doc_id, store in doc_stores.items():
            data = test_data[doc_id]
            node_id = data["node_id"]
            text = data["text"]
            token_count = data["token_count"]

            # Ensure correct types
            assert isinstance(node_id, str)
            assert isinstance(text, str)
            assert isinstance(token_count, int)

            nodes: list[
                dict[
                    str,
                    str | int | float | bool | list[float] | NDArray[np.float64] | None,
                ]
            ] = [
                {
                    "node_id": node_id,
                    "text": text,
                    "span_start": 0,
                    "span_end": len(text),
                    "document_id": doc_id,
                    "token_count": token_count,
                    "height": 0,
                }
            ]
            store.nodes.add_batch(nodes)

        # Verify each document store only sees its own content
        for doc_id, store in doc_stores.items():
            all_nodes = store.nodes.get_all()
            assert len(all_nodes) == 1

            node = all_nodes[0]
            data = test_data[doc_id]
            assert node.document_id == doc_id
            assert node.id == data["node_id"]
            assert node.text == data["text"]

            # Verify cross-document isolation
            for other_doc_id, other_data in test_data.items():
                if other_doc_id != doc_id:
                    other_node_id = other_data["node_id"]
                    assert isinstance(other_node_id, str)
                    assert store.nodes.get_node(other_node_id) is None

    def test_sqlite_backend_vector_search_isolation(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
        sqlite_backend: SQLiteStorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test vector search isolation between documents."""
        # Create stores for different document types
        technical_store = sqlite_store_factory("technical-docs")
        creative_store = sqlite_store_factory("creative-writing")

        # Add technical document content
        technical_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "tech_node1",
                "text": "Database optimization and indexing strategies",
                "span_start": 0,
                "span_end": 45,
                "document_id": "technical-docs",
                "token_count": 6,
                "height": 0,
            }
        ]
        technical_store.nodes.add_batch(technical_nodes)

        # Add creative writing content
        creative_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "creative_node1",
                "text": "The mystical forest whispered ancient secrets",
                "span_start": 0,
                "span_end": 45,
                "document_id": "creative-writing",
                "token_count": 7,
                "height": 0,
            }
        ]
        creative_store.nodes.add_batch(creative_nodes)

        # Add embeddings to vector index with distinct patterns
        technical_embedding = [0.9, 0.1, 0.1] + [0.2] * 1533  # High first dimension
        creative_embedding = [0.1, 0.9, 0.1] + [0.2] * 1533  # High second dimension

        vector_index.upsert(
            [
                (
                    "tech_node1",
                    technical_embedding,
                    {
                        "document_id": "technical-docs",
                        "span_start": 0,
                        "span_end": 45,
                        "is_leaf": 1,
                        "parent_id": "",  # Empty string instead of None
                    },
                ),
                (
                    "creative_node1",
                    creative_embedding,
                    {
                        "document_id": "creative-writing",
                        "span_start": 0,
                        "span_end": 45,
                        "is_leaf": 1,
                        "parent_id": "",  # Empty string instead of None
                    },
                ),
            ]
        )

        # Test technical document search
        technical_query = [0.9, 0.1, 0.1] + [0.2] * 1533  # Similar to technical
        technical_results = vector_index.search_similar(
            technical_query, 3, [DocumentIdFilter("technical-docs")]
        )

        # Should find technical content
        technical_found = False
        for v in technical_results:
            if v.id == "tech_node1":
                technical_found = True
                assert v.meta.get("document_id") == "technical-docs"
            # Should not find creative content in technical store results
            assert v.id != "creative_node1"
        assert technical_found

        # Test creative document search
        creative_query = [0.1, 0.9, 0.1] + [0.2] * 1533  # Similar to creative
        creative_results = vector_index.search_similar(
            creative_query, 3, [DocumentIdFilter("creative-writing")]
        )

        # Should find creative content
        creative_found = False
        for v in creative_results:
            if v.id == "creative_node1":
                creative_found = True
                assert v.meta.get("document_id") == "creative-writing"
            # Should not find technical content in creative store results
            assert v.id != "tech_node1"
        assert creative_found

    def test_sqlite_backend_node_operations(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test basic node operations with document isolation."""
        store = sqlite_store_factory("test-operations")

        # Add multiple nodes
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "node1",
                "text": "First test node",
                "span_start": 0,
                "span_end": 15,
                "document_id": "test-operations",
                "token_count": 3,
                "height": 0,
            },
            {
                "node_id": "node2",
                "text": "Second test node",
                "span_start": 16,
                "span_end": 32,
                "document_id": "test-operations",
                "token_count": 3,
                "height": 0,
            },
        ]
        store.nodes.add_batch(nodes)

        # Test individual node retrieval
        node1 = store.nodes.get_node("node1")
        assert node1 is not None
        assert node1.text == "First test node"
        assert node1.document_id == "test-operations"

        node2 = store.nodes.get_node("node2")
        assert node2 is not None
        assert node2.text == "Second test node"
        assert node2.document_id == "test-operations"

        # Test bulk retrieval
        both_nodes = store.nodes.get_nodes(["node1", "node2"])
        assert len(both_nodes) == 2
        node_ids = {node.id for node in both_nodes}
        assert node_ids == {"node1", "node2"}

        # Test get_all
        all_nodes = store.nodes.get_all()
        assert len(all_nodes) == 2
        all_node_ids = {node.id for node in all_nodes}
        assert all_node_ids == {"node1", "node2"}

        # All nodes should belong to the same document
        for node in all_nodes:
            assert node.document_id == "test-operations"
