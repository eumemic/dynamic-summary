"""End-to-end workflow tests for RagZoom (index → query → assemble).

These tests verify the complete pipeline from document indexing through
retrieval to final assembly, using mock OpenAI clients.
"""

from collections.abc import Generator
from typing import Any

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.index import TreeBuilder
from tests.utils import mock_openai_context


@pytest.mark.integration
class TestIntegration:
    """Test complete workflow integration scenarios.

    Focus: End-to-end testing of index → retrieve → assemble pipeline
    with various document types and retrieval scenarios.
    """

    @pytest.fixture
    def mock_openai(self) -> Generator[tuple[Any, Any, Any], None, None]:
        """Mock OpenAI API calls using centralized utilities."""
        with mock_openai_context() as mocks:
            yield mocks

    @pytest.fixture
    def temp_system(
        self, request: Any, mock_openai: tuple[Any, Any, Any], base_config: Any
    ) -> Generator[tuple[Any, Any, Any, Any], None, None]:
        """Create a complete temporary RagZoom system."""
        # Always use real PostgreSQL store for integration tests
        from tests.conftest import _create_real_store

        real_store = _create_real_store(base_config)
        if real_store is None:
            pytest.skip("PostgreSQL not available for integration test")

        # Create separate configs
        index_config = IndexConfig.load(
            target_chunk_tokens=50,
            preceding_context_tokens=25,  # Must be less than leaf_tokens
        )
        query_config = QueryConfig(budget_tokens=500)

        # Create operational config
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test-key"),
            database_url=real_store.config.database_url,  # Use the store's database URL
        )

        # Don't create TreeBuilder here - tests will create their own
        # after creating documents

        # Use the mocked OpenAI client
        mock_index_client, mock_retrieve_client, mock_assemble_client = mock_openai

        # Helper function to create TreeBuilder for a specific document
        def create_tree_builder(document_id: str) -> "TreeBuilder":
            # Create document in store first
            real_store.add_document(
                document_id=document_id,
                file_path=None,
                content_hash=real_store.compute_content_hash(""),
                chunk_count=0,
                embedding_model=index_config.embedding_model,
                summary_model=index_config.summary_model,
            )
            # Create DocumentStore for that document
            doc_store = real_store.for_document(document_id)
            # Create TreeBuilder with that DocumentStore
            return TreeBuilder(
                index_config, doc_store, api_key=operational_config.openai_api_key
            )

        # Create a config wrapper for backward compatibility
        from tests.conftest import BackwardCompatibilityConfig

        config = BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )

        yield config, real_store, create_tree_builder, mock_retrieve_client

        # Cleanup PostgreSQL store and unique database
        if hasattr(real_store, "_test_db_cleanup"):
            cleanup_info = real_store._test_db_cleanup
            try:
                from sqlalchemy import create_engine, text

                admin_engine = create_engine(
                    cleanup_info["admin_url"], isolation_level="AUTOCOMMIT"
                )
                with admin_engine.connect() as conn:
                    # Terminate connections and drop database
                    conn.execute(
                        text(
                            "SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = :db_name AND pid <> pg_backend_pid()"
                        ),
                        {"db_name": cleanup_info["db_name"]},
                    )
                    conn.execute(
                        text(f"DROP DATABASE IF EXISTS {cleanup_info['db_name']}")
                    )  # nosec B608
                admin_engine.dispose()
            except Exception:
                pass  # Ignore cleanup errors
            real_store.close()

    def test_index_and_query(self, temp_system: tuple[Any, Any, Any, Any]) -> None:
        """Test indexing a document and querying it."""
        config, store, create_tree_builder, mock_client = temp_system

        # Create TreeBuilder for this specific document
        tree_builder = create_tree_builder("test-doc")

        # Index a simple document
        text = "The quick brown fox jumps over the lazy dog. " * 20
        doc_id = tree_builder.add_document(text)

        assert doc_id == "test-doc"

        # Check tree was built - get leaf nodes from the specific document
        doc_store = store.for_document(doc_id)
        leaf_nodes = doc_store.nodes.get_leaves()
        assert len(leaf_nodes) > 0

        root = doc_store.tree.get_root()
        assert root is not None

        # Create retriever and assembler for this document
        from tests.utils import create_retriever

        retriever = create_retriever(
            config.query_config,
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

    def test_multiple_documents(self, temp_system: tuple[Any, Any, Any, Any]) -> None:
        """Test indexing multiple documents."""
        config, store, create_tree_builder, mock_client = temp_system

        # Create TreeBuilder and index first document
        text1 = "First document content. " * 10
        tree_builder1 = create_tree_builder("doc1")
        tree_builder1.add_document(text1)

        # Get initial leaf count from doc1
        doc1_store = store.for_document("doc1")
        initial_leaf_count = len(doc1_store.nodes.get_leaves())

        # Create TreeBuilder and index second document
        text2 = "Second document content. " * 10
        tree_builder2 = create_tree_builder("doc2")
        tree_builder2.add_document(text2)

        # Check new leaves were added by checking both documents
        doc2_store = store.for_document("doc2")
        doc1_leaf_count = len(doc1_store.nodes.get_leaves())
        doc2_leaf_count = len(doc2_store.nodes.get_leaves())
        total_leaf_count = doc1_leaf_count + doc2_leaf_count
        assert total_leaf_count > initial_leaf_count

        # Check we have nodes from both documents
        doc1_nodes = doc1_store.nodes.get_leaves()
        doc2_nodes = doc2_store.nodes.get_leaves()
        assert len(doc1_nodes) > 0
        assert len(doc2_nodes) > 0

    def test_mmr_diversity(self, temp_system: tuple[Any, Any, Any, Any]) -> None:
        """Test that MMR returns diverse results."""
        config, store, create_tree_builder, mock_client = temp_system

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

        doc_store = store.for_document("doc-diverse")
        retriever = create_retriever(
            config.query_config,
            doc_store,
            client=mock_client,
        )

        # Query about cats (should get diverse cat-related content)
        result = retriever.retrieve("Tell me about cats", num_seeds=3)

        assert len(result.node_ids) <= 3
        # Should get results from different documents, not just repeated similar ones
        assert len(set(result.node_ids)) == len(result.node_ids)

    def test_token_budget_enforcement(
        self, temp_system: tuple[Any, Any, Any, Any]
    ) -> None:
        """Test that assembly respects token budget."""
        config, store, create_tree_builder, mock_client = temp_system

        # Create TreeBuilder and index a large document
        text = "This is a test sentence. " * 200
        tree_builder = create_tree_builder("budget-test")
        tree_builder.add_document(text)

        # Create retriever for this document
        from tests.utils import create_retriever

        doc_store = store.for_document("budget-test")
        retriever = create_retriever(
            config.query_config,
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

    def test_node_pinning(self, temp_system: tuple[Any, Any, Any, Any]) -> None:
        """Test that pinned nodes are always included."""
        config, store, create_tree_builder, mock_client = temp_system

        # Create a single document with multiple content sections
        combined_text = (
            "Important content. " * 10 + "Other content. " * 10 + "More content. " * 10
        )
        tree_builder = create_tree_builder("doc-pinning")
        tree_builder.add_document(combined_text)

        # Pin a specific node from the document
        doc_store = store.for_document("doc-pinning")
        all_nodes = doc_store.nodes.get_leaves()
        if all_nodes:
            important_node = all_nodes[0]
            store.pin_node(important_node.id)

            # Create retriever for this document
            from tests.utils import create_retriever

            retriever = create_retriever(
                config.query_config,
                doc_store,
                client=mock_client,
            )

            # Query for unrelated content
            result = retriever.retrieve("unrelated query")

            # Check pinned node is in coverage
            assert important_node.id in result.coverage_map
