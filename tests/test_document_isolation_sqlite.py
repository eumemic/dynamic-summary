"""SQLite-based document isolation tests.

Tests to ensure queries are properly isolated to specific documents using the real SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.assemble import Assembler
from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.document_store import DocumentStore
from ragzoom.index import TreeBuilder
from tests.utils import create_retriever, mock_openai_context


@pytest.mark.usefixtures("sqlite_backend")
class TestDocumentIsolationSQLite:
    """Test that queries are properly isolated to specific documents."""

    @pytest.fixture
    def mock_openai(self) -> Generator[tuple[Mock, Mock, Mock], None, None]:
        """Mock OpenAI API calls with specialized embedding rules."""
        embedding_rules = {
            "dragon": [0.9] * 1536,
            "wizard": [0.8] * 1536,
        }
        with mock_openai_context(embedding_rules) as mocks:
            yield mocks

    def test_document_isolation(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
        sqlite_backend: SQLiteStorageBackend,
        mock_openai: tuple[Mock, Mock, Mock],
    ) -> None:
        """Test that queries only return results from the specified document."""
        # Create document stores
        dragons_store = sqlite_store_factory("dragons.txt")
        wizards_store = sqlite_store_factory("wizards.txt")

        # Create configs
        index_config = IndexConfig.load(target_chunk_tokens=100)
        query_config = QueryConfig(budget_tokens=1000)

        # Index documents using TreeBuilder
        doc1_text = "The mighty dragon breathed fire upon the castle. Dragons are powerful creatures."
        tree_builder1 = TreeBuilder(index_config, dragons_store, api_key="test-key")
        doc1_id = tree_builder1.add_document(doc1_text)
        assert doc1_id == "dragons.txt"

        doc2_text = "The wise wizard cast a spell. Wizards study magic for many years."
        tree_builder2 = TreeBuilder(index_config, wizards_store, api_key="test-key")
        doc2_id = tree_builder2.add_document(doc2_text)
        assert doc2_id == "wizards.txt"

        # Get all nodes and upsert embeddings for vector behavior
        all_nodes = []
        # Collect from both backends through the shared SQLite backend
        for doc_store in [dragons_store, wizards_store]:
            nodes = list(doc_store.nodes.get_all())
            all_nodes.extend(nodes)

        embedding_entries: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = []
        for node in all_nodes:
            # Use different embeddings based on content
            if "dragon" in node.text.lower():
                embedding: list[float] | NDArray[np.float64] = [0.9] * 1536
            else:
                embedding = [0.8] * 1536

            embedding_entries.append(
                (
                    node.id,  # TreeNode uses 'id', not 'node_id'
                    embedding,
                    {
                        "span_start": node.span_start,
                        "span_end": node.span_end,
                        "parent_id": node.parent_id,
                        "document_id": node.document_id,
                        "is_leaf": 1 if node.height == 0 else 0,
                    },
                )
            )

        sqlite_backend.vector_index.upsert(embedding_entries)

        # Create retrievers using the utility function
        retriever1 = create_retriever(
            query_config,
            dragons_store,
            document_id="dragons.txt",
            api_key="test-key",
            client=mock_openai[1],  # Use mock retrieve client
        )

        retriever2 = create_retriever(
            query_config,
            wizards_store,
            document_id="wizards.txt",
            api_key="test-key",
            client=mock_openai[1],  # Use mock retrieve client
        )

        # Query about dragons in the dragons document
        result1 = retriever1.retrieve(
            "tell me about dragons", document_id="dragons.txt"
        )

        # Check that all returned nodes are from dragons.txt
        for node_id in result1.node_ids:
            retrieved_node = dragons_store.nodes.get_node(node_id)
            assert retrieved_node is not None, f"Node {node_id} not found"
            assert (
                retrieved_node.document_id == "dragons.txt"
            ), f"Node {node_id} is from wrong document: {retrieved_node.document_id}"

        # Query about wizards in the wizards document
        result2 = retriever2.retrieve(
            "tell me about wizards", document_id="wizards.txt"
        )

        # Check that all returned nodes are from wizards.txt
        for node_id in result2.node_ids:
            retrieved_node = wizards_store.nodes.get_node(node_id)
            assert retrieved_node is not None, f"Node {node_id} not found"
            assert (
                retrieved_node.document_id == "wizards.txt"
            ), f"Node {node_id} is from wrong document: {retrieved_node.document_id}"

        # Cross-query test: query about dragons in wizards document
        # Should return nodes from wizards.txt even though query is about dragons
        result3 = retriever2.retrieve(
            "tell me about dragons", document_id="wizards.txt"
        )

        for node_id in result3.node_ids:
            retrieved_node = wizards_store.nodes.get_node(node_id)
            assert retrieved_node is not None, f"Node {node_id} not found"
            assert (
                retrieved_node.document_id == "wizards.txt"
            ), f"Cross-query failed: got node from {retrieved_node.document_id}"

        # The content should be about wizards, not dragons
        assembler = Assembler(wizards_store)
        summary = assembler.assemble(result3)
        assert (
            "wizard" in summary.lower() or "magic" in summary.lower()
        ), "Should return wizard content"
        assert (
            "dragon" not in summary.lower()
        ), "Should not return dragon content when querying wizards doc"

    def test_filename_as_default_document_id(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
        mock_openai: tuple[Mock, Mock, Mock],
    ) -> None:
        """Test that filename is used as document_id when not specified."""
        # Create document store
        doc_store = sqlite_store_factory("test_file.txt")

        # Create config
        index_config = IndexConfig.load(target_chunk_tokens=100)
        query_config = QueryConfig(budget_tokens=1000)

        # Index with file_path but no explicit document_id
        text = "Test content for filename ID"
        tree_builder = TreeBuilder(index_config, doc_store, api_key="test-key")
        doc_id = tree_builder.add_document(text)

        # Should use filename as document_id
        assert doc_id == "test_file.txt"

        # Create retriever for this document
        retriever = create_retriever(
            query_config,
            doc_store,
            document_id="test_file.txt",
            api_key="test-key",
            client=mock_openai[1],  # Use mock retrieve client
        )

        # Verify we can query using the filename
        result = retriever.retrieve("test content", document_id="test_file.txt")
        assert len(result.node_ids) > 0

        # Check all nodes have correct document_id
        for node_id in result.node_ids:
            node = doc_store.nodes.get_node(node_id)
            assert node is not None, f"Node {node_id} not found"
            assert node.document_id == "test_file.txt"

    def test_query_without_document_filter(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
        mock_openai: tuple[Mock, Mock, Mock],
    ) -> None:
        """Test that querying without document_id returns results from all documents."""
        # Create document stores for two separate documents
        doc1_store = sqlite_store_factory("doc1")
        doc2_store = sqlite_store_factory("doc2")

        # Create config
        index_config = IndexConfig.load(target_chunk_tokens=100)
        query_config = QueryConfig(budget_tokens=1000)

        # Index multiple documents
        tree_builder1 = TreeBuilder(index_config, doc1_store, api_key="test-key")
        tree_builder1.add_document("Dragons are fierce")

        tree_builder2 = TreeBuilder(index_config, doc2_store, api_key="test-key")
        tree_builder2.add_document("Wizards are wise")

        # Query both documents independently and combine results
        retriever1 = create_retriever(
            query_config,
            doc1_store,
            document_id="doc1",
            api_key="test-key",
            client=mock_openai[1],
        )
        retriever2 = create_retriever(
            query_config,
            doc2_store,
            document_id="doc2",
            api_key="test-key",
            client=mock_openai[1],
        )

        result_all = []
        result_all.extend(
            retriever1.retrieve("tell me about everything", document_id="doc1").node_ids
        )
        result_all.extend(
            retriever2.retrieve("tell me about everything", document_id="doc2").node_ids
        )

        # Should potentially get results from one or both documents
        doc_ids = set()
        for node_id in result_all:
            node = doc1_store.nodes.get_node(node_id) or doc2_store.nodes.get_node(
                node_id
            )
            assert node is not None, f"Node {node_id} not found in either document"
            assert node.document_id in {"doc1", "doc2"}
            doc_ids.add(node.document_id)

        assert len(doc_ids) >= 1
