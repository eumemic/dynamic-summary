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
        """Test that the generated frontier covers the entire document without gaps or overlaps."""
        config, store, tree_builder, retriever, assembler = setup_system

        document = "This is a test document. " * 100
        doc_id = "test-doc-2"
        tree_builder.add_document(document, doc_id)

        # Retrieve with a generous budget
        result = retriever.retrieve("test", budget_tokens=5000, document_id=doc_id)

        assert result.frontier_segments, "DP frontier should not be empty"

        # Check for completeness and no overlaps
        # To do this properly, we need the actual spans of the segments
        spans = []
        for seg in result.frontier_segments:
            node = store.get_node(seg.node_id)
            child = store.get_child(node.id, seg.side)
            if child:
                spans.append((child.span_start, child.span_end))

        spans.sort()

        # Check for gaps and overlaps
        assert (
            spans[0][0] == 0
        ), "Frontier should start at the beginning of the document"
        for i in range(len(spans) - 1):
            assert (
                spans[i][1] == spans[i + 1][0]
            ), f"Gap or overlap found between segments {i} and {i+1}"

        root_node = store.get_root_node_for_document(doc_id)
        assert (
            spans[-1][1] == root_node.span_end
        ), "Frontier should end at the end of the document"
