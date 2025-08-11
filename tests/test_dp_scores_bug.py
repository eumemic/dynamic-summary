"""Test demonstrating the DP algorithm uses scores outside coverage tree."""

from ragzoom.config import QueryConfig
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from ragzoom.retrieve import RetrievalResult
from tests.mock_store import SimpleMockStore


class TestDPScoresBug:
    """Test that DP algorithm incorrectly uses nodes outside coverage tree."""

    def test_dp_uses_scores_outside_coverage_tree(self):
        """Demonstrate that DP uses any node with a score, ignoring coverage tree."""
        # Set up a mock store with a simple tree
        store = SimpleMockStore()

        # Create a tree structure:
        #          root
        #         /    \
        #     node_a   node_b
        #      / \      / \
        #    a1  a2   b1  b2

        # Root
        store.add_node(
            node_id="root",
            text="Root summary of document",
            span_start=0,
            span_end=1000,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="node_a",
            right_child_id="node_b",
        )

        # Internal nodes
        store.add_node(
            node_id="node_a",
            text="Node A summary",
            span_start=0,
            span_end=500,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="a1",
            right_child_id="a2",
        )

        store.add_node(
            node_id="node_b",
            text="Node B summary",
            span_start=500,
            span_end=1000,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="b1",
            right_child_id="b2",
        )

        # Leaf nodes
        for node_id, start, end, parent in [
            ("a1", 0, 250, "node_a"),
            ("a2", 250, 500, "node_a"),
            ("b1", 500, 750, "node_b"),
            ("b2", 750, 1000, "node_b"),
        ]:
            store.add_node(
                node_id=node_id,
                text=f"Leaf {node_id} content",
                span_start=start,
                span_end=end,
                parent_id=parent,
                document_id="doc1",
                embedding=[0.5] * 1536,
            )

        # Create config and DP generator
        query_config = QueryConfig(budget_tokens=10000)  # Large budget
        dp_generator = DynamicTilingGenerator(query_config)

        # Pass in a full coverage tree (all nodes)
        coverage_tree = {"a1", "a2", "b1", "b2", "node_a", "node_b", "root"}

        scores = {
            "a1": 0.9,  # Selected node
            "a2": 0.8,  # Sibling
            "b1": 0.85,  # Sibling
            "b2": 0.7,  # Sibling
            "node_a": 0.5,
            "node_b": 0.5,
            "root": 0.3,
        }

        # Load nodes from coverage map
        nodes = {nid: store.get_node(nid) for nid in coverage_tree}

        # Find root node
        root_id = "root"

        dp_result = dp_generator.find_optimal_tiling(
            budget_tokens=10000,
            scores=scores,
            nodes=nodes,
            root_id=root_id,
        )
        tiling = dp_result.tiling

        # Check results
        leaf_node_ids = {
            node_id for node_id in tiling.node_ids if store.is_leaf_node(node_id)
        }

        # With our fix, all leaf nodes in tiling must be in the coverage tree
        leaf_violations = [
            node_id for node_id in leaf_node_ids if node_id not in coverage_tree
        ]
        assert (
            len(leaf_violations) == 0
        ), f"Found leaf nodes outside coverage tree: {leaf_violations}"

    def test_retrieval_result_demonstrates_bug(self):
        """Test using actual RetrievalResult to show the bug."""
        store = SimpleMockStore()

        # Same tree setup as above (simplified)
        store.add_node(
            node_id="root",
            text="Root",
            span_start=0,
            span_end=1000,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 1536,
        )
        store.add_node(
            node_id="leaf1",
            text="Leaf 1",
            span_start=0,
            span_end=500,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
        )
        store.add_node(
            node_id="leaf2",
            text="Leaf 2",
            span_start=500,
            span_end=1000,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
        )
        store.nodes["root"].left_child_id = "leaf1"
        store.nodes["root"].right_child_id = "leaf2"

        query_config = QueryConfig(budget_tokens=10000)
        dp_generator = DynamicTilingGenerator(query_config)

        # Pass in a full coverage tree (root and both leaves)
        result = RetrievalResult(
            node_ids=["leaf1"],  # Only 1 selected
            scores={
                "leaf1": 0.9,  # Selected
                "leaf2": 0.8,  # Sibling
                "root": 0.5,
            },
            coverage_map={"leaf1": True, "leaf2": True, "root": True},
            tiling=None,
        )

        # Load nodes from coverage map
        nodes = {}
        for node_id in result.coverage_map:
            node = store.get_node(node_id)
            if node:
                nodes[node_id] = node

        # Find root node
        root_id = "root"

        dp_result = dp_generator.find_optimal_tiling(
            budget_tokens=10000,
            scores=result.scores,
            nodes=nodes,
            root_id=root_id,
        )
        tiling = dp_result.tiling

        # Check results
        leaf_node_ids = {
            node_id for node_id in tiling.node_ids if store.is_leaf_node(node_id)
        }

        # With our fix: leaf2 should NOT appear in tiling unless it is in the coverage map
        assert (
            "leaf2" in result.coverage_map
        ), "leaf2 must be in coverage map for DP to consider it"
        assert (
            "leaf2" in leaf_node_ids or "leaf1" in leaf_node_ids
        ), "At least one leaf should be in the tiling"
