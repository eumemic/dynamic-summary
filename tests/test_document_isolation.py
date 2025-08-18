"""Test document isolation - queries should only return results from specified document."""

import tempfile
from unittest.mock import Mock, patch

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import Store
from tests.conftest import BackwardCompatibilityConfig


class TestDocumentIsolation:
    """Test that queries are properly isolated to specific documents."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API calls."""
        with (
            patch("ragzoom.index.AsyncOpenAI") as mock_index_client,
            patch("ragzoom.retrieve.OpenAI") as mock_retrieve_client,
        ):

            # Mock embeddings - return different embeddings for different content
            async def mock_embeddings_create(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    embeddings = []
                    for text in input_data:
                        if "dragon" in text.lower():
                            embeddings.append(Mock(embedding=[0.9] * 1536))
                        elif "wizard" in text.lower():
                            embeddings.append(Mock(embedding=[0.8] * 1536))
                        else:
                            embeddings.append(Mock(embedding=[0.5] * 1536))
                    return Mock(data=embeddings)
                else:
                    if "dragon" in input_data.lower():
                        return Mock(data=[Mock(embedding=[0.9] * 1536)])
                    elif "wizard" in input_data.lower():
                        return Mock(data=[Mock(embedding=[0.8] * 1536)])
                    else:
                        return Mock(data=[Mock(embedding=[0.5] * 1536)])

            async def mock_chat_create(*args, **kwargs):
                return Mock(
                    choices=[
                        Mock(message=Mock(content="Summary of left and right content"))
                    ]
                )

            def mock_embeddings_create_sync(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if "dragon" in input_data.lower():
                    return Mock(data=[Mock(embedding=[0.9] * 1536)])
                elif "wizard" in input_data.lower():
                    return Mock(data=[Mock(embedding=[0.8] * 1536)])
                else:
                    return Mock(data=[Mock(embedding=[0.5] * 1536)])

            # Set up async client
            instance_async = Mock()
            instance_async.embeddings = Mock()
            instance_async.embeddings.create = Mock(side_effect=mock_embeddings_create)
            instance_async.chat = Mock()
            instance_async.chat.completions = Mock()
            instance_async.chat.completions.create = Mock(side_effect=mock_chat_create)
            mock_index_client.return_value = instance_async

            # Set up sync client for retrieve only
            instance_sync = Mock()
            instance_sync.embeddings = Mock()
            instance_sync.embeddings.create = Mock(
                side_effect=mock_embeddings_create_sync
            )
            mock_retrieve_client.return_value = instance_sync

            yield

    @pytest.fixture
    def setup(self, mock_openai):
        """Create test environment."""
        with tempfile.TemporaryDirectory():
            index_config = IndexConfig.load(
                target_chunk_tokens=50,
                preceding_context_tokens=25,
            )
            query_config = QueryConfig(
                budget_tokens=1000,
            )
            operational_config = OperationalConfig(
                openai_api_key="test-key",
                database_url="postgresql:///:memory:",
            )
            config = BackwardCompatibilityConfig(
                index_config, query_config, operational_config
            )

            store = Store(
                operational_config, embedding_model=index_config.embedding_model
            )
            tree_builder = TreeBuilder(
                index_config, store, api_key=operational_config.openai_api_key
            )
            retriever = Retriever(
                query_config,
                store,
                api_key=operational_config.openai_api_key,
            )
            assembler = Assembler(store)

            yield config, store, tree_builder, retriever, assembler

            store.close()

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
            node = store.get_node(node_id)
            assert (
                node.document_id == "dragons.txt"
            ), f"Node {node_id} is from wrong document: {node.document_id}"

        # Query about wizards in the wizards document
        result2 = retriever.retrieve("tell me about wizards", document_id="wizards.txt")

        # Check that all returned nodes are from wizards.txt
        for node_id in result2.node_ids:
            node = store.get_node(node_id)
            assert (
                node.document_id == "wizards.txt"
            ), f"Node {node_id} is from wrong document: {node.document_id}"

        # Cross-query test: query about dragons in wizards document
        # Should return nodes from wizards.txt even though query is about dragons
        result3 = retriever.retrieve("tell me about dragons", document_id="wizards.txt")

        for node_id in result3.node_ids:
            node = store.get_node(node_id)
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
            node = store.get_node(node_id)
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
            node = store.get_node(node_id)
            doc_ids.add(node.document_id)

        # Could have nodes from either or both documents
        assert len(doc_ids) >= 1, "Should have at least one document in results"
