"""SQLite-based tests for Phase 4 auxiliary services.

SQLite-based tests ensuring EmbeddingService and BudgetPlanner work correctly
with the real in-memory SQLite backend for higher fidelity testing.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray
from openai import OpenAI

from ragzoom.document_store import DocumentStore
from ragzoom.retrieval.budget_planner import BudgetPlanner
from ragzoom.retrieval.embedding_service import EmbeddingService


@pytest.mark.usefixtures("sqlite_backend")
class TestEmbeddingServiceIsolation:
    """Test that EmbeddingService properly uses DocumentStore for isolation."""

    def test_embedding_service_uses_document_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test that EmbeddingService gets embedding model from DocumentStore."""
        # Create document stores for different documents
        doc1_store = sqlite_store_factory("doc1")
        doc2_store = sqlite_store_factory("doc2")

        # Create mock OpenAI client
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.5] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        # Test with doc1_store - uses default fallback since SQLite doesn't have document metadata
        service1 = EmbeddingService(mock_client, doc1_store, "text-embedding-ada-002")
        _ = service1.get_query_embedding("test query", document_id="doc1")

        # Verify default model was used (SQLite backend uses fallback behavior)
        mock_client.embeddings.create.assert_called_with(
            model="text-embedding-ada-002",
            input="test query",
        )

        # Reset mock
        mock_client.reset_mock()

        # Test with doc2_store
        service2 = EmbeddingService(mock_client, doc2_store, "text-embedding-ada-002")
        _ = service2.get_query_embedding("test query", document_id="doc2")

        # Verify same default model was used
        mock_client.embeddings.create.assert_called_with(
            model="text-embedding-ada-002",
            input="test query",
        )

    def test_embedding_service_fallback_to_default(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test that EmbeddingService falls back to default when no document model."""
        doc_store = sqlite_store_factory("doc1")

        # Create mock OpenAI client
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.5] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        # Test with DocumentStore
        default_model = "text-embedding-ada-002"
        service = EmbeddingService(mock_client, doc_store, default_model)
        _ = service.get_query_embedding("test query")

        # Verify default model was used
        mock_client.embeddings.create.assert_called_with(
            model=default_model,
            input="test query",
        )


@pytest.mark.usefixtures("sqlite_backend")
class TestBudgetPlannerIsolation:
    """Test that BudgetPlanner properly uses DocumentStore for isolation."""

    def test_budget_planner_uses_document_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test that BudgetPlanner gets token stats from DocumentStore."""
        # Create document stores
        doc1_store = sqlite_store_factory("doc1")
        doc2_store = sqlite_store_factory("doc2")

        # Add nodes for doc1 (small chunks)
        doc1_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": f"doc1_leaf_{i}",
                "text": f"Small text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "document_id": "doc1",
                "token_count": 50,  # Small chunks
                "height": 0,
                "path": f"{i:03b}",
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
            for i in range(5)
        ]
        doc1_store.nodes.add_batch(doc1_nodes)

        # Add nodes for doc2 (large chunks)
        doc2_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": f"doc2_leaf_{i}",
                "text": f"Large text content here {i}" * 10,
                "span_start": i * 100,
                "span_end": (i + 1) * 100,
                "document_id": "doc2",
                "token_count": 200,  # Large chunks
                "height": 0,
                "path": f"{i:03b}",
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
            for i in range(5)
        ]
        doc2_store.nodes.add_batch(doc2_nodes)

        # Test with doc1_store (small chunks)
        planner1 = BudgetPlanner(doc1_store, default_chunk_tokens=100)
        seeds1 = planner1.calculate_conservative_num_seeds(500, document_id="doc1")

        # With 50-token chunks, 500 budget should give ~10 seeds
        assert seeds1 == 10

        # Test with doc2_store (large chunks)
        planner2 = BudgetPlanner(doc2_store, default_chunk_tokens=100)
        seeds2 = planner2.calculate_conservative_num_seeds(500, document_id="doc2")

        # With 200-token chunks, 500 budget should give ~2 seeds
        assert seeds2 == 2

    def test_budget_planner_without_document_id(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test that BudgetPlanner handles cross-document queries properly."""
        # Create a document store without document_id (cross-document)
        cross_doc_store = sqlite_store_factory(None)

        # Test with cross-document store
        default_chunk_tokens = 100
        planner = BudgetPlanner(cross_doc_store, default_chunk_tokens)
        seeds = planner.calculate_conservative_num_seeds(500)

        # Should use default chunk size for estimation
        assert seeds == 500 // default_chunk_tokens

    def test_budget_planner_minimum_seeds(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test that BudgetPlanner always returns at least 1 seed."""
        doc_store = sqlite_store_factory("doc1")

        # Add document with very large chunks
        large_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "doc1_leaf",
                "text": "Very large text" * 100,
                "span_start": 0,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 1000,  # Very large chunk
                "height": 0,
                "path": "0",
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
        ]
        doc_store.nodes.add_batch(large_nodes)

        planner = BudgetPlanner(doc_store, default_chunk_tokens=100)

        # Even with tiny budget, should return at least 1 seed
        seeds = planner.calculate_conservative_num_seeds(10)
        assert seeds == 1
