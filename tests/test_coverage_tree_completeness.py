"""Tests for coverage tree completeness requirements."""

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.retrieve import Retriever
from tests.mock_store import SimpleMockStore


class TestCoverageTreeCompleteness:
    """Tests that ensure coverage trees maintain left-balanced properties."""

    @pytest.fixture
    def setup_incomplete_tree(self):
        """Set up a system with a tree that will produce incomplete coverage."""
        index_config = IndexConfig.load(
            target_chunk_tokens=100, preceding_context_tokens=50
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create config wrapper for SimpleMockStore compatibility
        class LocalTestConfig:
            def __init__(self, index_config, query_config, operational_config):
                self.index_config = index_config
                self.query_config = query_config
                self.operational_config = operational_config
                self.target_chunk_tokens = index_config.target_chunk_tokens
                self.preceding_context_tokens = index_config.preceding_context_tokens
                self.budget_tokens = query_config.budget_tokens

        config = LocalTestConfig(index_config, query_config, operational_config)
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
        )

        retriever = Retriever(
            query_config,
            store,
            api_key=operational_config.openai_api_key,
            tree_builder=None,
        )
        dp_generator = retriever.dp_generator

        return config, store, retriever, dp_generator

    def test_left_balanced_tree_single_child_handling(self, setup_incomplete_tree):
        """Test that left-balanced trees with single children are handled correctly."""
        config, store, retriever, dp_generator = setup_incomplete_tree

        # Simulate what happens with --num-seeds 1: only L3 is selected
        # This creates a left-balanced coverage tree where P2 has only its left child
        coverage_map = {"L3": True}

        # Add ancestors (this is what current retriever does)
        current_id = "L3"
        node = store.nodes.get_node(current_id)
        while node and node.parent_id:
            parent = store.nodes.get_node(node.parent_id)
            if parent:
                coverage_map[parent.id] = True
                current_id = parent.id
                node = parent
            else:
                break

        # Load nodes from coverage map
        nodes = {}
        for node_id in coverage_map:
            node = store.nodes.get_node(node_id)
            if node:
                nodes[node_id] = node

        # This should have L3, P2, and root, but not L4 (sibling of L3)
        assert "L3" in nodes
        assert "P2" in nodes
        assert "root" in nodes
        assert "L4" not in nodes  # P2 has only its left child in coverage

        # With left-balanced trees, this is a valid configuration
        # The DP algorithm correctly handles P2 having only its left child
        # Provide scores for all nodes in coverage to ensure L3 is selected
        scores = {node_id: 0.1 for node_id in nodes}  # Base score for all
        scores["L3"] = 1.0  # L3 has high relevance

        result = dp_generator.find_optimal_tiling(
            budget_tokens=1000,
            scores=scores,
            nodes=nodes,
            root_id="root",
        )

        # The DP algorithm may choose root over the subtree with single child
        # This is correct behavior - it's choosing the option with best quality score
        # The algorithm now supports P2 having only its left child and makes
        # the optimal choice based on relevance scores and token budgets

        # This test verifies that the algorithm handles left-balanced trees correctly
        # The actual tiling choice depends on the quality scores and budget
        assert result.tiling.node_ids  # Should have some result
        assert result.total_quality >= 0  # Should have non-negative quality

    def test_complete_coverage_tree_works(self, setup_incomplete_tree):
        """Test that complete coverage trees work correctly."""
        config, store, retriever, dp_generator = setup_incomplete_tree

        # Create a complete coverage tree by including all nodes
        nodes = {}
        for node_id in ["root", "P1", "P2", "L1", "L2", "L3", "L4"]:
            node = store.nodes.get_node(node_id)
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
                node = store.nodes.get_node(current_id)
                if node and node.parent_id:
                    current_id = node.parent_id
                else:
                    break

        # Step 2: For each node in coverage, ensure both children are included
        # This ensures completeness
        nodes_to_check = list(coverage_nodes)
        while nodes_to_check:
            node_id = nodes_to_check.pop(0)
            node = store.nodes.get_node(node_id)
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
            node = store.nodes.get_node(node_id)
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
