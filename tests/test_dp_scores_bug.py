"""Test demonstrating the DP algorithm uses scores outside coverage tree."""

from ragzoom.config import RagZoomConfig
from ragzoom.dynamic_frontier import DynamicFrontierGenerator
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
            text="Root summary <<<MID>>> of document",
            span_start=0,
            span_end=1000,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 384,
            mid_offset=12,
            left_child_id="node_a",
            right_child_id="node_b",
        )

        # Internal nodes
        store.add_node(
            node_id="node_a",
            text="Node A left <<<MID>>> Node A right",
            span_start=0,
            span_end=500,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 384,
            mid_offset=12,
            left_child_id="a1",
            right_child_id="a2",
        )

        store.add_node(
            node_id="node_b",
            text="Node B left <<<MID>>> Node B right",
            span_start=500,
            span_end=1000,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 384,
            mid_offset=12,
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
                embedding=[0.5] * 384,
            )

        # Create config and DP generator
        config = RagZoomConfig(
            openai_api_key="test-key", budget_tokens=10000  # Large budget
        )
        dp_generator = DynamicFrontierGenerator(config, store)

        # Simulate the bug scenario:
        # 1. Coverage tree contains only a1 and its ancestors
        coverage_tree = {"a1", "node_a", "root"}

        # 2. But scores contain ALL leaf nodes (this is the bug!)
        scores = {
            "a1": 0.9,  # Selected node
            "a2": 0.8,  # NOT in coverage tree
            "b1": 0.85,  # NOT in coverage tree
            "b2": 0.7,  # NOT in coverage tree
            "node_a": 0.5,
            "node_b": 0.5,
            "root": 0.3,
        }

        # Run DP algorithm
        # Note: DP now takes coverage_map as parameter
        coverage_map = {node: True for node in coverage_tree}
        dp_result = dp_generator.find_optimal_frontier(
            budget_tokens=10000,
            scores=scores,
            document_id="doc1",
            coverage_map=coverage_map,
        )
        segments = dp_result.segments

        # Collect which nodes are used in the frontier
        nodes_in_frontier = {seg.node_id for seg in segments}
        print(f"\nCoverage tree: {coverage_tree}")
        print(f"Nodes in frontier: {nodes_in_frontier}")

        # Check if any nodes are outside coverage tree
        violations = nodes_in_frontier - coverage_tree

        # The bug: DP uses nodes outside coverage tree because they have scores
        assert len(violations) > 0, (
            "Expected DP to use nodes outside coverage tree, but it didn't. "
            "The bug might be fixed!"
        )

        print(
            f"\nBUG CONFIRMED: DP used these nodes outside coverage tree: {violations}"
        )

        # Specifically check for leaf nodes outside coverage
        leaf_violations = []
        for seg in segments:
            node = store.get_node(seg.node_id)
            if (
                node
                and store.is_leaf_node(seg.node_id)
                and seg.node_id not in coverage_tree
            ):
                leaf_violations.append(seg.node_id)

        print(f"Leaf nodes outside coverage tree: {leaf_violations}")

        # With high scores on b1, b2, a2, DP will likely use them
        assert len(leaf_violations) > 0, "Expected leaf nodes outside coverage tree"

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
            embedding=[0.5] * 384,
            mid_offset=10,
        )
        store.add_node(
            node_id="leaf1",
            text="Leaf 1",
            span_start=0,
            span_end=500,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 384,
        )
        store.add_node(
            node_id="leaf2",
            text="Leaf 2",
            span_start=500,
            span_end=1000,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 384,
        )
        store.nodes["root"].left_child_id = "leaf1"
        store.nodes["root"].right_child_id = "leaf2"

        config = RagZoomConfig(openai_api_key="test-key", budget_tokens=10000)
        dp_generator = DynamicFrontierGenerator(config, store)

        # Create a RetrievalResult that mimics the bug:
        # - node_ids has only 1 selected node
        # - but scores has multiple nodes
        result = RetrievalResult(
            node_ids=["leaf1"],  # Only 1 selected
            scores={
                "leaf1": 0.9,  # Selected
                "leaf2": 0.8,  # NOT selected but has score!
                "root": 0.5,
            },
            coverage_map={"leaf1": True, "root": True},  # Only selected + ancestors
            frontier_segments=None,
        )

        # This is what retriever.py does - passes ALL scores to DP
        dp_result = dp_generator.find_optimal_frontier(
            budget_tokens=10000,
            scores=result.scores,  # BUG: includes leaf2 which isn't in coverage!
            document_id="doc1",
            coverage_map=result.coverage_map,
        )
        segments = dp_result.segments

        # Check results
        leaf_segments = [s for s in segments if store.is_leaf_node(s.node_id)]
        leaf_node_ids = {s.node_id for s in leaf_segments}

        print(f"\nSelected nodes: {result.node_ids}")
        print(f"Coverage map: {list(result.coverage_map.keys())}")
        print(f"Scores include: {list(result.scores.keys())}")
        print(f"Leaf nodes in frontier: {leaf_node_ids}")

        # The bug: leaf2 can appear in frontier even though it's not in coverage
        if "leaf2" in leaf_node_ids:
            print("\nBUG CONFIRMED: leaf2 is in frontier but not in coverage map!")
            assert "leaf2" not in result.coverage_map
            assert "leaf2" not in result.node_ids
            assert "leaf2" in result.scores  # But it has a score!

        # This demonstrates the root cause:
        # scores dict contains nodes outside the coverage tree
