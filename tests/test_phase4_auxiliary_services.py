"""Test Phase 4 auxiliary services (EmbeddingService and BudgetPlanner) with DocumentStore."""

from unittest.mock import Mock

from openai import OpenAI

from ragzoom.retrieval.budget_planner import BudgetPlanner
from ragzoom.retrieval.embedding_service import EmbeddingService
from tests.mock_store import SimpleMockStore


class TestEmbeddingServiceIsolation:
    """Test that EmbeddingService properly uses DocumentStore for isolation."""

    def test_embedding_service_uses_document_store(self):
        """Test that EmbeddingService gets embedding model from DocumentStore."""
        # Create mock store with documents having different embedding models
        store = SimpleMockStore()

        # Add document metadata
        store.documents["doc1"] = {
            "id": "doc1",
            "embedding_model": "text-embedding-3-small",
            "summary_model": "gpt-4",
        }
        store.documents["doc2"] = {
            "id": "doc2",
            "embedding_model": "text-embedding-3-large",
            "summary_model": "gpt-4",
        }

        # Create document stores
        doc1_store = store.for_document("doc1")
        doc2_store = store.for_document("doc2")

        # Create mock OpenAI client
        mock_client = Mock(spec=OpenAI)
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.5] * 1536)]
        mock_client.embeddings.create.return_value = mock_response

        # Test with doc1_store
        service1 = EmbeddingService(mock_client, doc1_store, "text-embedding-ada-002")
        _ = service1.get_query_embedding("test query", document_id="doc1")

        # Verify correct model was used
        mock_client.embeddings.create.assert_called_with(
            model="text-embedding-3-small",
            input="test query",
        )

        # Reset mock
        mock_client.reset_mock()

        # Test with doc2_store
        service2 = EmbeddingService(mock_client, doc2_store, "text-embedding-ada-002")
        _ = service2.get_query_embedding("test query", document_id="doc2")

        # Verify different model was used
        mock_client.embeddings.create.assert_called_with(
            model="text-embedding-3-large",
            input="test query",
        )

    def test_embedding_service_fallback_to_default(self):
        """Test that EmbeddingService falls back to default when no document model."""
        store = SimpleMockStore()

        # Document without embedding_model metadata
        store.documents["doc1"] = {
            "id": "doc1",
            "summary_model": "gpt-4",
        }

        doc_store = store.for_document("doc1")

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


class TestBudgetPlannerIsolation:
    """Test that BudgetPlanner properly uses DocumentStore for isolation."""

    def test_budget_planner_uses_document_store(self):
        """Test that BudgetPlanner gets token stats from DocumentStore."""
        # Create mock store with documents having different token sizes
        store = SimpleMockStore()

        # Add nodes for doc1 (small chunks)
        for i in range(5):
            store.add_node(
                node_id=f"doc1_leaf_{i}",
                text=f"Small text {i}",
                span_start=i * 10,
                span_end=(i + 1) * 10,
                document_id="doc1",
                embedding=[0.5] * 1536,
                token_count=50,  # Small chunks
            )

        # Add nodes for doc2 (large chunks)
        for i in range(5):
            store.add_node(
                node_id=f"doc2_leaf_{i}",
                text=f"Large text content here {i}" * 10,
                span_start=i * 100,
                span_end=(i + 1) * 100,
                document_id="doc2",
                embedding=[0.5] * 1536,
                token_count=200,  # Large chunks
            )

        # Create document stores
        doc1_store = store.for_document("doc1")
        doc2_store = store.for_document("doc2")

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

    def test_budget_planner_without_document_id(self):
        """Test that BudgetPlanner handles cross-document queries properly."""
        store = SimpleMockStore()

        # Create a document store without document_id (cross-document)
        cross_doc_store = store.for_document(None)

        # Test with cross-document store
        default_chunk_tokens = 100
        planner = BudgetPlanner(cross_doc_store, default_chunk_tokens)
        seeds = planner.calculate_conservative_num_seeds(500)

        # Should use default chunk size for estimation
        assert seeds == 500 // default_chunk_tokens

    def test_budget_planner_minimum_seeds(self):
        """Test that BudgetPlanner always returns at least 1 seed."""
        store = SimpleMockStore()

        # Add document with very large chunks
        store.add_node(
            node_id="doc1_leaf",
            text="Very large text" * 100,
            span_start=0,
            span_end=1000,
            document_id="doc1",
            embedding=[0.5] * 1536,
            token_count=1000,  # Very large chunk
        )

        doc_store = store.for_document("doc1")
        planner = BudgetPlanner(doc_store, default_chunk_tokens=100)

        # Even with tiny budget, should return at least 1 seed
        seeds = planner.calculate_conservative_num_seeds(10)
        assert seeds == 1
