"""Tests for coverage tree completeness requirements."""

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import Retriever
from tests.mock_store import SimpleMockStore


class TestCoverageTreeCompleteness:
    """Tests that ensure coverage trees are complete binary trees."""

    @pytest.fixture
    def setup_incomplete_tree(self):
        """Set up a system with a tree that will produce incomplete coverage."""
        config = RagZoomConfig(leaf_tokens=100, adjacent_context_tokens=50)
        store = SimpleMockStore(config=config)

        # Create a simple tree structure:
        #         root
        #        /    \
        #      P1      P2
        #     /  \    /  \
        #    L1  L2  L3  L4

        # Add leaf nodes with parent relationships
        store.add_node(
            node_id="L1",
            text="Chapter 1 content",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=20,
            document_id="test-doc",
            parent_id="P1",
        )
        store.add_node(
            node_id="L2",
            text="Chapter 2 content",
            embedding=[0.2] * 1536,
            span_start=20,
            span_end=40,
            document_id="test-doc",
            parent_id="P1",
        )
        store.add_node(
            node_id="L3",
            text="Chapter 3 content",
            embedding=[0.3] * 1536,
            span_start=40,
            span_end=60,
            document_id="test-doc",
            parent_id="P2",
        )
        store.add_node(
            node_id="L4",
            text="Chapter 4 content",
            embedding=[0.4] * 1536,
            span_start=60,
            span_end=80,
            document_id="test-doc",
            parent_id="P2",
        )

        # Add parent nodes
        store.add_node(
            node_id="P1",
            text="Summary of chapters 1-2",
            embedding=[0.15] * 1536,
            span_start=0,
            span_end=40,
            document_id="test-doc",
            parent_id="root",
            left_child_id="L1",
            right_child_id="L2",
            summary="Summary of chapters 1-2",
        )
        store.add_node(
            node_id="P2",
            text="Summary of chapters 3-4",
            embedding=[0.35] * 1536,
            span_start=40,
            span_end=80,
            document_id="test-doc",
            parent_id="root",
            left_child_id="L3",
            right_child_id="L4",
            summary="Summary of chapters 3-4",
        )

        # Add root
        store.add_node(
            node_id="root",
            text="Full document summary",
            embedding=[0.25] * 1536,
            span_start=0,
            span_end=80,
            document_id="test-doc",
            left_child_id="P1",
            right_child_id="P2",
            summary="Full document summary",
        )

        retriever = Retriever(config, store, tree_builder=None)
        dp_generator = retriever.dp_generator

        return config, store, retriever, dp_generator

    def test_incomplete_coverage_tree_raises_error(self, setup_incomplete_tree):
        """Test that incomplete coverage trees are detected and raise an error."""
        config, store, retriever, dp_generator = setup_incomplete_tree

        # Simulate what happens with --n-max 1: only L3 is selected
        # This creates an incomplete coverage tree
        coverage_map = {"L3": True}

        # Add ancestors (this is what current retriever does)
        current_id = "L3"
        node = store.get_node(current_id)
        while node and node.parent_id:
            parent = store.get_node(node.parent_id)
            if parent:
                coverage_map[parent.id] = True
                current_id = parent.id
                node = parent
            else:
                break

        # Load nodes from coverage map
        nodes = {}
        for node_id in coverage_map:
            node = store.get_node(node_id)
            if node:
                nodes[node_id] = node

        # This should have L3, P2, and root, but missing L4 (sibling of L3)
        assert "L3" in nodes
        assert "P2" in nodes
        assert "root" in nodes
        assert "L4" not in nodes  # This is the problem!

        # Try to run DP algorithm - should raise error
        with pytest.raises(
            ValueError, match="Coverage tree is incomplete.*missing.*child"
        ):
            dp_generator.find_optimal_tiling(
                budget_tokens=1000,
                scores={"L3": 1.0},
                nodes=nodes,
                root_id="root",
            )

    def test_complete_coverage_tree_works(self, setup_incomplete_tree):
        """Test that complete coverage trees work correctly."""
        config, store, retriever, dp_generator = setup_incomplete_tree

        # Create a complete coverage tree by including all nodes
        nodes = {}
        for node_id in ["root", "P1", "P2", "L1", "L2", "L3", "L4"]:
            node = store.get_node(node_id)
            if node:
                nodes[node_id] = node

        # This should work without errors
        result = dp_generator.find_optimal_tiling(
            budget_tokens=1000,
            scores={"L3": 1.0},  # L3 is most relevant
            nodes=nodes,
            root_id="root",
        )

        # Should produce a valid tiling
        assert result.tiling is not None
        assert len(result.tiling.node_ids) > 0

    def test_coverage_tree_with_siblings_included(self, setup_incomplete_tree):
        """Test the correct way to build coverage tree with siblings."""
        config, store, retriever, dp_generator = setup_incomplete_tree

        # Start with selected node
        selected_nodes = ["L3"]
        coverage_nodes = set(selected_nodes)

        # Build complete coverage tree
        # Step 1: Add all ancestors
        for node_id in selected_nodes:
            current_id = node_id
            while current_id:
                coverage_nodes.add(current_id)
                node = store.get_node(current_id)
                if node and node.parent_id:
                    current_id = node.parent_id
                else:
                    break

        # Step 2: For each node in coverage, ensure both children are included
        # This ensures completeness
        nodes_to_check = list(coverage_nodes)
        while nodes_to_check:
            node_id = nodes_to_check.pop(0)
            node = store.get_node(node_id)
            if node:
                # If this node has children, both must be in coverage
                if node.left_child_id:
                    if node.left_child_id not in coverage_nodes:
                        coverage_nodes.add(node.left_child_id)
                        nodes_to_check.append(node.left_child_id)
                if node.right_child_id:
                    if node.right_child_id not in coverage_nodes:
                        coverage_nodes.add(node.right_child_id)
                        nodes_to_check.append(node.right_child_id)

        # Load all nodes
        nodes = {}
        for node_id in coverage_nodes:
            node = store.get_node(node_id)
            if node:
                nodes[node_id] = node

        # Should have complete subtree
        assert "L3" in nodes
        assert "L4" in nodes  # Sibling included
        assert "P2" in nodes
        assert "P1" in nodes  # Sibling of P2
        assert "L1" in nodes  # Children of P1
        assert "L2" in nodes
        assert "root" in nodes

        # This should work without errors
        result = dp_generator.find_optimal_tiling(
            budget_tokens=1000,
            scores={"L3": 1.0},
            nodes=nodes,
            root_id="root",
        )

        assert result.tiling is not None
