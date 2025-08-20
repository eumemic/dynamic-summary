"""Tests for DP assembly path and DP algorithm.

This file contains both assembly tests and core DP algorithm tests.
The DP algorithm test was moved here from test_dp_tiling.py to consolidate
all DP-related testing in one place.
"""

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import OperationalConfig, QueryConfig
from ragzoom.retrieve import Retriever


class TestDPAssembly:
    """Test the DP assembly path and core DP algorithm.

    This class contains both assembly tests (using node IDs) and
    core DP algorithm tests (moved from test_dp_tiling.py).
    """

    @pytest.fixture
    def query_config(self):
        """Create test query configuration."""
        return QueryConfig(budget_tokens=1000)

    @pytest.fixture
    def operational_config(self):
        """Create test operational configuration."""
        return OperationalConfig(openai_api_key="test-key")

    @pytest.fixture
    def assembler(self, store):
        """Create assembler with mock store."""
        return Assembler(store)

    @pytest.fixture
    def mock_nodes(self, store):
        """Create mock nodes in the store."""
        # Create a simple tree structure
        # Root (depth=2)
        #  / \
        # L   R (depth=1)
        # /\ /\
        # 1 2 3 4 (depth=0)

        # Leaf nodes
        store.add_node(
            node_id="leaf1",
            text="First chunk of text.",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=20,
            document_id="doc1",
        )

        store.add_node(
            node_id="leaf2",
            text="Second chunk of text.",
            embedding=[0.2] * 1536,
            span_start=20,
            span_end=41,
            document_id="doc1",
        )

        store.add_node(
            node_id="leaf3",
            text="Third chunk of text.",
            embedding=[0.3] * 1536,
            span_start=41,
            span_end=61,
            document_id="doc1",
        )

        store.add_node(
            node_id="leaf4",
            text="Fourth chunk of text.",
            embedding=[0.4] * 1536,
            span_start=61,
            span_end=82,
            document_id="doc1",
        )

        # Internal nodes (no mid_offset in new design)
        store.add_node(
            node_id="left",
            text="Summary of first and second chunks.",
            embedding=[0.15] * 1536,
            span_start=0,
            span_end=41,
            left_child_id="leaf1",
            right_child_id="leaf2",
            document_id="doc1",
        )

        store.add_node(
            node_id="right",
            text="Summary of third and fourth chunks.",
            embedding=[0.35] * 1536,
            span_start=41,
            span_end=82,
            left_child_id="leaf3",
            right_child_id="leaf4",
            document_id="doc1",
        )

        store.add_node(
            node_id="root",
            text="Overall document summary.",
            embedding=[0.25] * 1536,
            span_start=0,
            span_end=82,
            left_child_id="left",
            right_child_id="right",
            document_id="doc1",
        )

    def test_basic_dp_assembly(self, assembler, mock_nodes):
        """Test basic DP assembly with leaf nodes."""
        # List of node IDs
        tiling = ["leaf1", "leaf2"]

        result = assembler.assemble_dp(tiling)

        # Leaf nodes return full text
        assert result == "First chunk of text.\n\nSecond chunk of text."

    def test_internal_node_assembly(self, assembler, mock_nodes):
        """Test assembly with internal nodes (atomic units)."""
        tiling = ["left", "leaf3"]

        result = assembler.assemble_dp(tiling)

        # Internal nodes return their full summary
        assert result == "Summary of first and second chunks.\n\nThird chunk of text."

    def test_mixed_nodes_assembly(self, assembler, mock_nodes):
        """Test assembly with mix of leaf and internal nodes."""
        tiling = ["leaf1", "right"]

        result = assembler.assemble_dp(tiling)

        # Each node returns its full text
        assert result == "First chunk of text.\n\nSummary of third and fourth chunks."

    def test_all_internal_nodes(self, assembler, mock_nodes):
        """Test assembly with only internal nodes."""
        tiling = ["left", "right", "root"]

        result = assembler.assemble_dp(tiling)

        expected = "Summary of first and second chunks.\n\nSummary of third and fourth chunks.\n\nOverall document summary."
        assert result == expected

    def test_all_leaf_nodes(self, assembler, mock_nodes):
        """Test assembly with all leaf nodes."""
        tiling = ["leaf1", "leaf2", "leaf3", "leaf4"]

        result = assembler.assemble_dp(tiling)

        # Should get full text for all leaf nodes
        expected = "First chunk of text.\n\nSecond chunk of text.\n\nThird chunk of text.\n\nFourth chunk of text."
        assert result == expected

    def test_empty_tiling(self, assembler, mock_nodes):
        """Test handling of empty tiling list."""
        tiling = []

        result = assembler.assemble_dp(tiling)

        assert result == ""

    def test_missing_node(self, assembler, mock_nodes):
        """Test handling when tiling references a missing node."""
        tiling = ["leaf1", "missing", "leaf3"]

        result = assembler.assemble_dp(tiling)

        # Should skip missing node
        assert result == "First chunk of text.\n\nThird chunk of text."

    def test_node_with_no_text(self, assembler, mock_nodes, store):
        """Test handling of nodes with empty text."""
        # Add a node with empty text
        store.add_node(
            node_id="empty",
            text="",
            embedding=[0.5] * 1536,
            span_start=82,
            span_end=82,
            document_id="doc1",
        )

        tiling = ["leaf1", "empty", "leaf3"]

        result = assembler.assemble_dp(tiling)

        # Should skip empty node
        assert result == "First chunk of text.\n\nThird chunk of text."

    def test_single_root_node(self, assembler, mock_nodes):
        """Test assembly with just the root node."""
        tiling = ["root"]

        result = assembler.assemble_dp(tiling)

        # Should return root's full summary
        assert result == "Overall document summary."

    def test_complex_tiling_assembly(self, assembler, mock_nodes):
        """Test a complex tiling that resembles real DP output."""
        # Simulate a tiling that might come from DP algorithm
        # Mix of internal and leaf nodes
        tiling = ["left", "leaf3", "leaf4"]

        result = assembler.assemble_dp(tiling)

        expected = (
            "Summary of first and second chunks.\n\n"
            "Third chunk of text.\n\n"
            "Fourth chunk of text."
        )
        assert result == expected

    def test_ordering_preservation(self, assembler, mock_nodes):
        """Test that tiling order is preserved in output."""
        # Nodes in non-sequential order
        tiling = ["leaf3", "leaf1", "leaf4", "leaf2"]

        result = assembler.assemble_dp(tiling)

        # Output should follow tiling order, not span order
        expected = (
            "Third chunk of text.\n\n"
            "First chunk of text.\n\n"
            "Fourth chunk of text.\n\n"
            "Second chunk of text."
        )
        assert result == expected

    def test_dp_single_node_tree(self, store):
        """Test the DP algorithm on a tree with only a single node.

        This test was moved from test_dp_tiling.py to consolidate DP tests.
        It focuses on the DP algorithm itself rather than assembly.
        """
        # Set up configuration similar to the original test
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(
            openai_api_key="test-key",
            database_url="postgresql:///:memory:",
        )

        # Use the store fixture instead of SimpleMockStore for consistency
        retriever = Retriever(
            query_config=query_config,
            store=store,
            api_key=operational_config.openai_api_key,
        )
        dp_generator = retriever.dp_generator

        # Manually create a single-node tree
        store.add_node(
            node_id="root",
            text="single node",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            document_id="test-doc-single",
        )

        # Set up mock scores for the SimpleMockStore if available
        if hasattr(store, "set_mock_scores"):
            store.set_mock_scores({"root": 1.0})

        # Create coverage map
        coverage_map = {"root": True}

        # Load nodes from coverage map
        nodes = {}
        for node_id in coverage_map:
            node = store.get_node(node_id)
            if node:
                nodes[node_id] = node

        # Find root node
        root_id = "root"

        # Test the DP algorithm
        dp_result = dp_generator.find_optimal_tiling(
            1000, {"root": 1.0}, nodes, root_id
        )
        tiling = dp_result.tiling

        assert tiling, "DP tiling should not be empty for single node tree"
        assert len(tiling.node_ids) == 1
        assert tiling.node_ids[0] == "root"
