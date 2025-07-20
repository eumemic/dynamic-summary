import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import Retriever
from tests.mock_store import SimpleMockStore


class TestDPFrontier:
    """Tests for the new DP-based frontier generation."""

    @pytest.fixture
    def setup_system(self):
        """Set up a complete system with DP mode enabled and a mock store."""
        config = RagZoomConfig(leaf_tokens=100)
        store = SimpleMockStore(config=config)
        retriever = Retriever(config, store, tree_builder=None)
        assembler = Assembler(config, store)
        dp_generator = retriever.dp_generator
        return config, store, retriever, assembler, dp_generator

    def test_dp_single_node_tree(self, setup_system):
        """Test the DP algorithm on a tree with only a single node."""
        config, store, retriever, assembler, dp_generator = setup_system

        # Manually create a single-node tree
        store.add_node(
            "root",
            "single node",
            [0.1] * 1536,
            0,
            0,
            100,
            document_id="test-doc-single",
        )
        store.set_mock_scores({"root": 1.0})

        # We need to manually call the DP generator for now
        dp_result = dp_generator.find_optimal_frontier(
            1000, {"root": 1.0}, "test-doc-single"
        )
        segments = dp_result.segments

        assert segments, "DP frontier should not be empty for single node tree"
        assert len(segments) == 1
        assert segments[0].node_id == "root"
        assert segments[0].side is None  # Leaf nodes have side=None
