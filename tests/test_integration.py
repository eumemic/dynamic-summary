"""End-to-end workflow tests for RagZoom (index → query → assemble).

These tests verify the complete pipeline from document indexing through
retrieval to final assembly, using mock OpenAI clients.
"""

import shutil
import tempfile
from unittest.mock import Mock, patch

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import Store


class TestIntegration:
    """Test complete workflow integration scenarios.

    Focus: End-to-end testing of index → retrieve → assemble pipeline
    with various document types and retrieval scenarios.
    """

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API calls."""
        with (
            patch("ragzoom.index.AsyncOpenAI") as mock_index_client,
            patch("ragzoom.retrieve.OpenAI") as mock_retrieve_client,
        ):

            # Create async mock functions
            async def mock_embeddings_create(*args, **kwargs):
                # Handle both single and batch embedding requests
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    # Batch request - return multiple embeddings
                    return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_data])
                else:
                    # Single request
                    return Mock(data=[Mock(embedding=[0.1] * 1536)])

            async def mock_chat_create(*args, **kwargs):
                return Mock(
                    choices=[
                        Mock(
                            message=Mock(content="Summary of left half and right half.")
                        )
                    ]
                )

            # Create sync mock functions for retriever and assembler
            def mock_embeddings_create_sync(*args, **kwargs):
                # Handle both single and batch embedding requests
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    # Batch request - return multiple embeddings
                    return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_data])
                else:
                    # Single request
                    return Mock(data=[Mock(embedding=[0.1] * 1536)])

            def mock_chat_create_sync(*args, **kwargs):
                return Mock(
                    choices=[
                        Mock(
                            message=Mock(content="Summary of left half and right half.")
                        )
                    ]
                )

            # Mock embeddings for async client (index)
            mock_embeddings_async = Mock()
            mock_embeddings_async.create = Mock(side_effect=mock_embeddings_create)

            # Mock embeddings for sync clients (retrieve, assemble)
            mock_embeddings_sync = Mock()
            mock_embeddings_sync.create = Mock(side_effect=mock_embeddings_create_sync)

            # Mock chat completions
            mock_chat_async = Mock()
            mock_chat_async.completions = Mock()
            mock_chat_async.completions.create = Mock(side_effect=mock_chat_create)

            mock_chat_sync = Mock()
            mock_chat_sync.completions = Mock()
            mock_chat_sync.completions.create = Mock(side_effect=mock_chat_create_sync)

            # Set up async client for index
            instance_async = Mock()
            instance_async.embeddings = mock_embeddings_async
            instance_async.chat = mock_chat_async
            mock_index_client.return_value = instance_async

            # Set up sync client for retrieve
            instance_sync = Mock()
            instance_sync.embeddings = mock_embeddings_sync
            instance_sync.chat = mock_chat_sync
            mock_retrieve_client.return_value = instance_sync

            yield

    @pytest.fixture
    def temp_system(self, mock_openai):
        """Create a complete temporary RagZoom system."""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        chroma_dir = f"{temp_dir}/chroma"
        db_path = f"{temp_dir}/test.db"

        # Create separate configs
        index_config = IndexConfig.load(
            target_chunk_tokens=50,
            preceding_context_tokens=25,  # Must be less than leaf_tokens
        )
        query_config = QueryConfig(budget_tokens=500)
        operational_config = OperationalConfig(
            openai_api_key="test-key",
            chroma_persist_directory=chroma_dir,
            sqlite_database_url=f"sqlite:///{db_path}",
        )

        store = Store(operational_config, embedding_model=index_config.embedding_model)
        tree_builder = TreeBuilder(
            index_config, store, api_key=operational_config.openai_api_key
        )
        retriever = Retriever(
            query_config, store, api_key=operational_config.openai_api_key
        )
        assembler = Assembler(store)

        # Create a config wrapper for backward compatibility
        from tests.conftest import BackwardCompatibilityConfig

        config = BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )

        yield config, store, tree_builder, retriever, assembler

        # Cleanup - close store first to release file handles
        store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_index_and_query(self, temp_system):
        """Test indexing a document and querying it."""
        config, store, tree_builder, retriever, assembler = temp_system

        # Index a simple document
        text = "The quick brown fox jumps over the lazy dog. " * 20
        doc_id = tree_builder.add_document(text, "test-doc")

        assert doc_id == "test-doc"

        # Check tree was built
        leaf_nodes = store.get_leaf_nodes()
        assert len(leaf_nodes) > 0

        root = store.get_root_node()
        assert root is not None

        # Query the system
        query = "Tell me about the fox"
        result = retriever.retrieve(query)

        assert len(result.node_ids) > 0
        assert result.tiling is not None
        assert len(result.tiling) > 0

        # Assemble summary
        summary = assembler.assemble(result)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_multiple_documents(self, temp_system):
        """Test indexing multiple documents."""
        config, store, tree_builder, retriever, assembler = temp_system

        # Index initial document
        text1 = "First document content. " * 10
        tree_builder.add_document(text1, "doc1")

        initial_leaf_count = len(store.get_leaf_nodes())

        # Index second document
        text2 = "Second document content. " * 10
        tree_builder.add_document(text2, "doc2")

        # Check new leaves were added
        new_leaf_count = len(store.get_leaf_nodes())
        assert new_leaf_count > initial_leaf_count

        # Check we have nodes from both documents
        doc1_nodes = [n for n in store.get_leaf_nodes() if n.document_id == "doc1"]
        doc2_nodes = [n for n in store.get_leaf_nodes() if n.document_id == "doc2"]
        assert len(doc1_nodes) > 0
        assert len(doc2_nodes) > 0

    def test_mmr_diversity(self, temp_system):
        """Test that MMR returns diverse results."""
        config, store, tree_builder, retriever, assembler = temp_system

        # Create documents with different topics
        texts = [
            "The cat sat on the mat. Cats are feline animals.",
            "Dogs are loyal pets. The dog barked loudly.",
            "Birds can fly. Eagles are large birds.",
            "Fish swim in water. Salmon swim upstream.",
            "Cats and dogs are common pets. Many people love cats.",
        ]

        for i, text in enumerate(texts):
            tree_builder.add_document(text, f"doc-{i}")

        # Query about cats (should get diverse cat-related content)
        result = retriever.retrieve("Tell me about cats", num_seeds=3)

        assert len(result.node_ids) <= 3
        # Should get results from different documents, not just repeated similar ones
        assert len(set(result.node_ids)) == len(result.node_ids)

    def test_token_budget_enforcement(self, temp_system):
        """Test that assembly respects token budget."""
        config, store, tree_builder, retriever, assembler = temp_system

        # Create a large document
        text = "This is a test sentence. " * 200
        tree_builder.add_document(text)

        # Query with small budget
        result = retriever.retrieve("test sentence", budget_tokens=100)
        summary = assembler.assemble(result)
        token_count = assembler.get_token_count(summary)

        # Check budget is respected (with some tolerance for token counting differences)
        assert token_count <= 110  # Allow 10% tolerance
        assert len(summary) > 0

    def test_node_pinning(self, temp_system):
        """Test that pinned nodes are always included."""
        config, store, tree_builder, retriever, assembler = temp_system

        # Create documents
        texts = ["Important content.", "Other content.", "More content."]
        for i, text in enumerate(texts):
            tree_builder.add_document(text * 10, f"doc-{i}")

        # Pin a specific node
        all_nodes = store.get_leaf_nodes()
        if all_nodes:
            important_node = all_nodes[0]
            store.pin_node(important_node.id)

            # Query for unrelated content
            result = retriever.retrieve("unrelated query")

            # Check pinned node is in coverage
            assert important_node.id in result.coverage_map
