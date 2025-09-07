"""End-to-end workflow tests for RagZoom (index → query → assemble).

These tests verify the complete pipeline from document indexing through
retrieval to final assembly, using mock OpenAI clients.
"""

from collections.abc import Generator
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from ragzoom.index import TreeBuilder

    TreeBuilderFactory = Callable[[str], TreeBuilder]

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.index import TreeBuilder
from tests.utils import mock_openai_context


@pytest.mark.integration
class TestIntegration:
    """Test complete workflow integration scenarios.

    Focus: End-to-end testing of index → retrieve → assemble pipeline
    with various document types and retrieval scenarios.
    """

    @pytest.fixture
    def mock_openai(self) -> Generator[tuple[Mock, Mock, Mock], None, None]:
        """Mock OpenAI API calls using centralized utilities."""
        with mock_openai_context() as mocks:
            yield mocks

    @pytest.fixture
    def temp_system(
        self,
        request: pytest.FixtureRequest,
        mock_openai: tuple[Mock, Mock, Mock],
        storage_backend: StorageBackend,
    ) -> Generator[
        tuple[IndexConfig, QueryConfig, StorageBackend, "TreeBuilderFactory", Mock],
        None,
        None,
    ]:
        """Create a complete temporary RagZoom system."""
        # Create separate configs
        index_config = IndexConfig.load(
            target_chunk_tokens=50,
            preceding_context_tokens=25,  # Must be less than leaf_tokens
        )
        query_config = QueryConfig(budget_tokens=500)

        # Use the mocked OpenAI client
        mock_index_client, mock_retrieve_client, mock_assemble_client = mock_openai

        # Helper function to create TreeBuilder for a specific document
        def create_tree_builder(document_id: str) -> "TreeBuilder":
            # Create DocumentStore for that document
            doc_store = storage_backend.for_document(document_id)
            # Create TreeBuilder with that DocumentStore
            return TreeBuilder(index_config, doc_store, api_key="test-key")

        yield index_config, query_config, storage_backend, create_tree_builder, mock_retrieve_client

    def test_index_and_query(
        self,
        temp_system: tuple[
            IndexConfig, QueryConfig, StorageBackend, "TreeBuilderFactory", Mock
        ],
    ) -> None:
        """Test indexing a document and querying it."""
        (
            index_config,
            query_config,
            storage_backend,
            create_tree_builder,
            mock_client,
        ) = temp_system

        # Create TreeBuilder for this specific document
        tree_builder = create_tree_builder("test-doc")

        # Index a simple document
        text = "The quick brown fox jumps over the lazy dog. " * 20
        doc_id = tree_builder.add_document(text)

        assert doc_id == "test-doc"

        # Check tree was built - get leaf nodes from the specific document
        doc_store = storage_backend.for_document(doc_id)
        leaf_nodes = doc_store.nodes.get_leaves()
        assert len(leaf_nodes) > 0

        root = doc_store.tree.get_root()
        assert root is not None

        # Create retriever and assembler for this document
        from tests.utils import create_retriever

        retriever = create_retriever(
            query_config,
            doc_store,
            client=mock_client,
        )
        assembler = Assembler(doc_store)

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

    def test_multiple_documents(
        self,
        temp_system: tuple[
            IndexConfig, QueryConfig, StorageBackend, "TreeBuilderFactory", Mock
        ],
    ) -> None:
        """Test indexing multiple documents."""
        (
            index_config,
            query_config,
            storage_backend,
            create_tree_builder,
            mock_client,
        ) = temp_system

        # Create TreeBuilder and index first document
        text1 = "First document content. " * 10
        tree_builder1 = create_tree_builder("doc1")
        tree_builder1.add_document(text1)

        # Get initial leaf count from doc1
        doc1_store = storage_backend.for_document("doc1")
        initial_leaf_count = len(doc1_store.nodes.get_leaves())

        # Create TreeBuilder and index second document
        text2 = "Second document content. " * 10
        tree_builder2 = create_tree_builder("doc2")
        tree_builder2.add_document(text2)

        # Check new leaves were added by checking both documents
        doc2_store = storage_backend.for_document("doc2")
        doc1_leaf_count = len(doc1_store.nodes.get_leaves())
        doc2_leaf_count = len(doc2_store.nodes.get_leaves())
        total_leaf_count = doc1_leaf_count + doc2_leaf_count
        assert total_leaf_count > initial_leaf_count

        # Check we have nodes from both documents
        doc1_nodes = doc1_store.nodes.get_leaves()
        doc2_nodes = doc2_store.nodes.get_leaves()
        assert len(doc1_nodes) > 0
        assert len(doc2_nodes) > 0

    def test_mmr_diversity(
        self,
        temp_system: tuple[
            IndexConfig, QueryConfig, StorageBackend, "TreeBuilderFactory", Mock
        ],
    ) -> None:
        """Test that MMR returns diverse results."""
        (
            index_config,
            query_config,
            storage_backend,
            create_tree_builder,
            mock_client,
        ) = temp_system

        # Create a single document with different topics
        combined_text = """
        The cat sat on the mat. Cats are feline animals.
        Dogs are loyal pets. The dog barked loudly.
        Birds can fly. Eagles are large birds.
        Fish swim in water. Salmon swim upstream.
        Cats and dogs are common pets. Many people love cats.
        """

        # Create TreeBuilder and index the document
        tree_builder = create_tree_builder("doc-diverse")
        tree_builder.add_document(combined_text)

        # Create retriever for this document
        from tests.utils import create_retriever

        doc_store = storage_backend.for_document("doc-diverse")
        retriever = create_retriever(
            query_config,
            doc_store,
            client=mock_client,
        )

        # Query about cats (should get diverse cat-related content)
        result = retriever.retrieve("Tell me about cats", num_seeds=3)

        assert len(result.node_ids) <= 3
        # Should get results from different documents, not just repeated similar ones
        assert len(set(result.node_ids)) == len(result.node_ids)

    def test_token_budget_enforcement(
        self,
        temp_system: tuple[
            IndexConfig, QueryConfig, StorageBackend, "TreeBuilderFactory", Mock
        ],
    ) -> None:
        """Test that assembly respects token budget."""
        (
            index_config,
            query_config,
            storage_backend,
            create_tree_builder,
            mock_client,
        ) = temp_system

        # Create TreeBuilder and index a large document
        text = "This is a test sentence. " * 200
        tree_builder = create_tree_builder("budget-test")
        tree_builder.add_document(text)

        # Create retriever for this document
        from tests.utils import create_retriever

        doc_store = storage_backend.for_document("budget-test")
        retriever = create_retriever(
            query_config,
            doc_store,
            client=mock_client,
        )
        assembler = Assembler(doc_store)

        # Query with small budget
        result = retriever.retrieve("test sentence", budget_tokens=100)
        summary = assembler.assemble(result)
        token_count = assembler.get_token_count(summary)

        # Check budget is respected (with some tolerance for token counting differences)
        assert token_count <= 110  # Allow 10% tolerance
        assert len(summary) > 0

    def test_node_pinning(
        self,
        temp_system: tuple[
            IndexConfig, QueryConfig, StorageBackend, "TreeBuilderFactory", Mock
        ],
    ) -> None:
        """Test that pinned nodes are always included."""
        (
            index_config,
            query_config,
            storage_backend,
            create_tree_builder,
            mock_client,
        ) = temp_system

        # Create a single document with multiple content sections
        combined_text = (
            "Important content. " * 10 + "Other content. " * 10 + "More content. " * 10
        )
        tree_builder = create_tree_builder("doc-pinning")
        tree_builder.add_document(combined_text)

        # Pin a specific node from the document
        doc_store = storage_backend.for_document("doc-pinning")
        all_nodes = doc_store.nodes.get_leaves()
        if all_nodes:
            important_node = all_nodes[0]
            # Use the backend's pin_node method if available
            if hasattr(storage_backend, "pin_node"):
                storage_backend.pin_node(important_node.id)
            else:
                # Backend doesn't support pinning, skip this part of test
                pytest.skip("Node pinning not implemented for this backend")

            # Create retriever for this document
            from tests.utils import create_retriever

            retriever = create_retriever(
                query_config,
                doc_store,
                client=mock_client,
            )

            # Query for unrelated content
            result = retriever.retrieve("unrelated query")

            # Check pinned node is in coverage
            assert important_node.id in result.coverage_map
