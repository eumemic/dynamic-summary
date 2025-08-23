"""Test document isolation - queries should only return results from specified document."""

import pytest

from ragzoom.assemble import Assembler
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from tests.utils import mock_openai_context


class TestDocumentIsolation:
    """Test that queries are properly isolated to specific documents."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API calls with specialized embedding rules."""
        embedding_rules = {
            "dragon": [0.9] * 1536,
            "wizard": [0.8] * 1536,
        }
        with mock_openai_context(embedding_rules) as mocks:
            yield mocks

    @pytest.fixture
    def setup(self, mock_openai, store, base_config):
        """Create test environment."""
        tree_builder = TreeBuilder(
            base_config.index_config, store, api_key=base_config.openai_api_key
        )
        retriever = Retriever(
            base_config.query_config,
            store,
            api_key=base_config.openai_api_key.get_secret_value(),
        )
        assembler = Assembler(store)

        yield base_config, store, tree_builder, retriever, assembler

    def test_document_isolation(self, setup):
        """Test that queries only return results from the specified document."""
        config, store, tree_builder, retriever, assembler = setup

        # Index two different documents
        doc1_text = "The mighty dragon breathed fire upon the castle. Dragons are powerful creatures."
        doc2_text = "The wise wizard cast a spell. Wizards study magic for many years."

        # Index with explicit document IDs
        doc1_id = tree_builder.add_document(doc1_text, document_id="dragons.txt")
        doc2_id = tree_builder.add_document(doc2_text, document_id="wizards.txt")

        assert doc1_id == "dragons.txt"
        assert doc2_id == "wizards.txt"

        # Query about dragons in the dragons document
        result1 = retriever.retrieve("tell me about dragons", document_id="dragons.txt")

        # Check that all returned nodes are from dragons.txt
        for node_id in result1.node_ids:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "dragons.txt"
            ), f"Node {node_id} is from wrong document: {node.document_id}"

        # Query about wizards in the wizards document
        result2 = retriever.retrieve("tell me about wizards", document_id="wizards.txt")

        # Check that all returned nodes are from wizards.txt
        for node_id in result2.node_ids:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "wizards.txt"
            ), f"Node {node_id} is from wrong document: {node.document_id}"

        # Cross-query test: query about dragons in wizards document
        # Should return nodes from wizards.txt even though query is about dragons
        result3 = retriever.retrieve("tell me about dragons", document_id="wizards.txt")

        for node_id in result3.node_ids:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "wizards.txt"
            ), f"Cross-query failed: got node from {node.document_id}"

        # The content should be about wizards, not dragons
        summary = assembler.assemble(result3)
        assert (
            "wizard" in summary.lower() or "magic" in summary.lower()
        ), "Should return wizard content"
        assert (
            "dragon" not in summary.lower()
        ), "Should not return dragon content when querying wizards doc"

    def test_filename_as_default_document_id(self, setup):
        """Test that filename is used as document_id when not specified."""
        config, store, tree_builder, retriever, assembler = setup

        # Index with file_path but no explicit document_id
        text = "Test content for filename ID"
        doc_id = tree_builder.add_document(text, file_path="/path/to/test_file.txt")

        # Should use filename as document_id
        assert doc_id == "test_file.txt"

        # Verify we can query using the filename
        result = retriever.retrieve("test content", document_id="test_file.txt")
        assert len(result.node_ids) > 0

        # Check all nodes have correct document_id
        for node_id in result.node_ids:
            node = store.nodes.get_node(node_id)
            assert node.document_id == "test_file.txt"

    def test_query_without_document_filter(self, setup):
        """Test that querying without document_id returns results from all documents."""
        config, store, tree_builder, retriever, assembler = setup

        # Index multiple documents
        tree_builder.add_document("Dragons are fierce", document_id="doc1")
        tree_builder.add_document("Wizards are wise", document_id="doc2")

        # Query without document filter
        result = retriever.retrieve("tell me about everything")

        # Should potentially get results from multiple documents
        doc_ids = set()
        for node_id in result.node_ids:
            node = store.nodes.get_node(node_id)
            doc_ids.add(node.document_id)

        # Could have nodes from either or both documents
        assert len(doc_ids) >= 1, "Should have at least one document in results"
