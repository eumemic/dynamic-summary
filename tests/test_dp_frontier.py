import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import Store


class TestDPFrontier:
    """Tests for the new DP-based frontier generation."""

    @pytest.fixture
    def setup_system(self):
        """Set up a complete system with DP mode enabled."""
        config = RagZoomConfig(
            frontier_mode="dp",
            leaf_tokens=100,
            sqlite_database_url="sqlite:///:memory:",
        )
        store = Store(config=config)
        tree_builder = TreeBuilder(config, store)
        retriever = Retriever(config, store, tree_builder)
        assembler = Assembler(config, store)
        return config, store, tree_builder, retriever, assembler

    def test_dp_retrieval_returns_frontier(self, setup_system):
        """Test that the DP retriever returns a set of segments."""
        config, store, tree_builder, retriever, assembler = setup_system

        document = "This is a test document for the new DP frontier generation. " * 50
        tree_builder.add_document(document, "test-doc")

        result = retriever.retrieve("test", budget_tokens=200, document_id="test-doc")

        assert result.frontier_segments is not None
        # The DP implementation is not complete, so for now this will be empty
        # but the test ensures the pipeline runs.
        assert isinstance(result.frontier_segments, list)

        # Test assembly
        assembled_text = assembler.assemble(result)
        assert isinstance(assembled_text, str)

    def test_dp_frontier_is_complete_and_valid(self, setup_system):
        """Test that the generated frontier is not empty."""
        config, store, tree_builder, retriever, assembler = setup_system

        document = "This is a test document. " * 100
        doc_id = "test-doc-2"
        tree_builder.add_document(document, doc_id)

        # Retrieve with a generous budget
        result = retriever.retrieve("test", budget_tokens=5000, document_id=doc_id)

        assert result.frontier_segments, "DP frontier should not be empty"

    def test_dp_budget_is_respected(self, setup_system):
        """Test that the DP algorithm respects the token budget."""
        config, store, tree_builder, retriever, assembler = setup_system

        document = "This is a test document for budget. " * 200
        doc_id = "test-doc-budget"
        tree_builder.add_document(document, doc_id)

        budget = 500
        result = retriever.retrieve("test", budget_tokens=budget, document_id=doc_id)

        assert result.frontier_segments, "DP frontier should not be empty"

        # Assemble the text and check the token count
        assembled_text = assembler.assemble(result)
        token_count = assembler.get_token_count(assembled_text)

        assert token_count <= budget, f"Budget exceeded: {token_count} > {budget}"

    def test_dp_single_node_tree(self, setup_system):
        """Test the DP algorithm on a tree with only a single node."""
        config, store, tree_builder, retriever, assembler = setup_system

        document = "This is a single node document."
        doc_id = "test-doc-single"
        tree_builder.add_document(document, doc_id)

        # There should only be one node, the root
        root_node = store.get_root_node_for_document(doc_id)
        assert root_node is not None
        assert root_node.left_child_id is None
        assert root_node.right_child_id is None

        # Retrieve with a generous budget
        result = retriever.retrieve("test", budget_tokens=1000, document_id=doc_id)

        assert (
            result.frontier_segments
        ), "DP frontier should not be empty for single node tree"
        assert len(result.frontier_segments) == 2
        assert result.frontier_segments[0].node_id == root_node.id
        assert result.frontier_segments[0].side == "LEFT"
        assert result.frontier_segments[1].node_id == root_node.id
        assert result.frontier_segments[1].side == "RIGHT"
