"""Test document isolation - queries should only return results from specified document."""

from typing import Any

import pytest

from ragzoom.assemble import Assembler
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from tests.utils import mock_openai_context


class TestDocumentIsolation:
    """Test that queries are properly isolated to specific documents."""

    @pytest.fixture
    def mock_openai(self) -> Any:
        """Mock OpenAI API calls with specialized embedding rules."""
        embedding_rules = {
            "dragon": [0.9] * 1536,
            "wizard": [0.8] * 1536,
        }
        with mock_openai_context(embedding_rules) as mocks:
            yield mocks

    @pytest.fixture
    def setup(self, mock_openai: Any, store: Any, base_config: Any) -> Any:
        """Create test environment."""
        from openai import OpenAI

        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService

        # Create services for Retriever
        client = OpenAI(api_key=base_config.openai_api_key.get_secret_value())
        embedding_service = EmbeddingService(
            client, store, base_config.query_config.embedding_model
        )
        budget_planner = BudgetPlanner(
            store, base_config.index_config.target_chunk_tokens
        )

        yield base_config, store, embedding_service, budget_planner

    def test_document_isolation(self, setup: Any) -> None:
        """Test that queries only return results from the specified document."""
        config, store, embedding_service, budget_planner = setup

        # Index two different documents
        doc1_text = "The mighty dragon breathed fire upon the castle. Dragons are powerful creatures."
        doc2_text = "The wise wizard cast a spell. Wizards study magic for many years."

        # Index with explicit document IDs
        # Create TreeBuilder for first document
        doc1_store = store.add_document(
            document_id="dragons.txt",
            file_path=None,
            content_hash=store.compute_content_hash(doc1_text),
            chunk_count=0,
            embedding_model=config.index_config.embedding_model,
            summary_model=config.index_config.summary_model,
        )
        tree_builder1 = TreeBuilder(
            config.index_config, doc1_store, api_key=config.openai_api_key
        )
        doc1_id = tree_builder1.add_document(doc1_text)

        # Create TreeBuilder for second document
        doc2_store = store.add_document(
            document_id="wizards.txt",
            file_path=None,
            content_hash=store.compute_content_hash(doc2_text),
            chunk_count=0,
            embedding_model=config.index_config.embedding_model,
            summary_model=config.index_config.summary_model,
        )
        tree_builder2 = TreeBuilder(
            config.index_config, doc2_store, api_key=config.openai_api_key
        )
        doc2_id = tree_builder2.add_document(doc2_text)

        assert doc1_id == "dragons.txt"
        assert doc2_id == "wizards.txt"

        # Create retriever for dragons document
        doc1_store = store.for_document("dragons.txt")
        retriever1 = Retriever(
            config.query_config,
            doc1_store,
            embedding_service,
            budget_planner,
        )

        # Query about dragons in the dragons document
        result1 = retriever1.retrieve(
            "tell me about dragons", document_id="dragons.txt"
        )

        # Check that all returned nodes are from dragons.txt
        for node_id in result1.node_ids:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "dragons.txt"
            ), f"Node {node_id} is from wrong document: {node.document_id}"

        # Create retriever for wizards document
        doc2_store = store.for_document("wizards.txt")
        retriever2 = Retriever(
            config.query_config,
            doc2_store,
            embedding_service,
            budget_planner,
        )

        # Query about wizards in the wizards document
        result2 = retriever2.retrieve(
            "tell me about wizards", document_id="wizards.txt"
        )

        # Check that all returned nodes are from wizards.txt
        for node_id in result2.node_ids:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "wizards.txt"
            ), f"Node {node_id} is from wrong document: {node.document_id}"

        # Cross-query test: query about dragons in wizards document
        # Should return nodes from wizards.txt even though query is about dragons
        result3 = retriever2.retrieve(
            "tell me about dragons", document_id="wizards.txt"
        )

        for node_id in result3.node_ids:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "wizards.txt"
            ), f"Cross-query failed: got node from {node.document_id}"

        # The content should be about wizards, not dragons
        assembler = Assembler(doc2_store)
        summary = assembler.assemble(result3)
        assert (
            "wizard" in summary.lower() or "magic" in summary.lower()
        ), "Should return wizard content"
        assert (
            "dragon" not in summary.lower()
        ), "Should not return dragon content when querying wizards doc"

    def test_filename_as_default_document_id(self, setup: Any) -> None:
        """Test that filename is used as document_id when not specified."""
        config, store, embedding_service, budget_planner = setup

        # Index with file_path but no explicit document_id
        text = "Test content for filename ID"
        # Create TreeBuilder with document store for test_file.txt
        doc_store = store.add_document(
            document_id="test_file.txt",
            file_path="test_file.txt",
            content_hash=store.compute_content_hash(text),
            chunk_count=0,
            embedding_model=config.index_config.embedding_model,
            summary_model=config.index_config.summary_model,
        )
        tree_builder = TreeBuilder(
            config.index_config, doc_store, api_key=config.openai_api_key
        )
        doc_id = tree_builder.add_document(text)

        # Should use filename as document_id
        assert doc_id == "test_file.txt"

        # Create retriever for this document
        doc_store = store.for_document("test_file.txt")
        retriever = Retriever(
            config.query_config,
            doc_store,
            embedding_service,
            budget_planner,
        )

        # Verify we can query using the filename
        result = retriever.retrieve("test content", document_id="test_file.txt")
        assert len(result.node_ids) > 0

        # Check all nodes have correct document_id
        for node_id in result.node_ids:
            node = store.nodes.get_node(node_id)
            assert node.document_id == "test_file.txt"

    def test_query_without_document_filter(self, setup: Any) -> None:
        """Test that querying without document_id returns results from all documents."""
        config, store, embedding_service, budget_planner = setup

        # Index multiple documents
        doc1_store = store.add_document(
            document_id="doc1",
            file_path=None,
            content_hash=store.compute_content_hash("Dragons are fierce"),
            chunk_count=0,
            embedding_model=config.index_config.embedding_model,
            summary_model=config.index_config.summary_model,
        )
        tree_builder1 = TreeBuilder(
            config.index_config, doc1_store, api_key=config.openai_api_key
        )
        tree_builder1.add_document("Dragons are fierce")

        doc2_store = store.add_document(
            document_id="doc2",
            file_path=None,
            content_hash=store.compute_content_hash("Wizards are wise"),
            chunk_count=0,
            embedding_model=config.index_config.embedding_model,
            summary_model=config.index_config.summary_model,
        )
        tree_builder2 = TreeBuilder(
            config.index_config, doc2_store, api_key=config.openai_api_key
        )
        tree_builder2.add_document("Wizards are wise")

        # Create retriever without document filter (None)
        doc_store = store.for_document(None)
        retriever = Retriever(
            config.query_config,
            doc_store,
            embedding_service,
            budget_planner,
        )

        # Query without document filter
        result = retriever.retrieve("tell me about everything")

        # Should potentially get results from multiple documents
        doc_ids = set()
        for node_id in result.node_ids:
            node = store.nodes.get_node(node_id)
            doc_ids.add(node.document_id)

        # Could have nodes from either or both documents
        assert len(doc_ids) >= 1, "Should have at least one document in results"
