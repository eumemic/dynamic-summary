"""Tests for transient pinning mechanism.

Transient pinning allows nodes to be marked as "pinned" for a single query,
without modifying the database. Pinned nodes get relevance=1.0, making them
strongly preferred by the DP algorithm while still respecting budget constraints.
"""

from typing import cast

from ragzoom.config import QueryConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.dynamic_tiling import DynamicTilingGenerator


class MockNode:
    """Mock node for direct DP testing without database."""

    def __init__(
        self,
        node_id: str,
        token_count: int,
        span_start: int,
        span_end: int,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
    ) -> None:
        self.id = node_id
        self.token_count = token_count
        self.span_start = span_start
        self.span_end = span_end
        self.left_child_id = left_child_id
        self.right_child_id = right_child_id


class TestTransientPinning:
    """Test that transient pinning via relevance=1.0 works correctly."""

    def test_pinned_node_gets_relevance_override(self) -> None:
        """Pinned nodes should have their relevance set to 1.0."""
        # Create tree: root -> (leaf1, leaf2)
        leaf1 = MockNode("leaf1", 100, 0, 50)
        leaf2 = MockNode("leaf2", 100, 50, 100)
        root = MockNode("root", 50, 0, 100, "leaf1", "leaf2")

        nodes = cast(
            dict[str, TreeNode], {"leaf1": leaf1, "leaf2": leaf2, "root": root}
        )

        # Scores: leaf2 has low relevance normally
        scores = {"leaf1": 0.8, "leaf2": 0.1, "root": 0.7}

        query_config = QueryConfig(budget_tokens=200)
        generator = DynamicTilingGenerator(query_config)

        # With pinning: leaf2 should be selected (relevance overridden to 1.0)
        result_with_pin = generator.find_optimal_tiling_over_roots(
            ["root"], 200, scores, nodes, pinned_ids={"leaf2"}
        )

        # Verify pinned leaf2 is in the result
        assert "leaf2" in result_with_pin.tiling.node_ids

    def test_pinned_leaf_preferred_over_parent_summary(self) -> None:
        """Pinned leaf should be selected even when parent has decent relevance."""
        # Tree: parent (50 tokens) -> leaf (150 tokens, pinned)
        # Parent has higher relevance per token, but leaf is pinned
        leaf = MockNode("leaf", 150, 0, 100)
        parent = MockNode("parent", 50, 0, 100, "leaf", None)

        nodes = cast(dict[str, TreeNode], {"leaf": leaf, "parent": parent})
        scores = {"leaf": 0.3, "parent": 0.9}  # Parent normally preferred

        query_config = QueryConfig(budget_tokens=200)
        generator = DynamicTilingGenerator(query_config)

        # With budget=200 and leaf pinned, should select leaf (150 tokens)
        result = generator.find_optimal_tiling_over_roots(
            ["parent"], 200, scores, nodes, pinned_ids={"leaf"}
        )

        assert "leaf" in result.tiling.node_ids
        assert "parent" not in result.tiling.node_ids

    def test_pinned_node_excluded_when_over_budget(self) -> None:
        """Pinned node that exceeds budget should be gracefully excluded."""
        # Tree: parent (30 tokens) -> leaf (200 tokens, pinned but too expensive)
        leaf = MockNode("leaf", 200, 0, 100)
        parent = MockNode("parent", 30, 0, 100, "leaf", None)

        nodes = cast(dict[str, TreeNode], {"leaf": leaf, "parent": parent})
        scores = {"leaf": 0.5, "parent": 0.5}

        query_config = QueryConfig(budget_tokens=100)
        generator = DynamicTilingGenerator(query_config)

        # Budget=100, leaf costs 200 - should fall back to parent (30 tokens)
        # No error should be raised
        result = generator.find_optimal_tiling_over_roots(
            ["parent"], 100, scores, nodes, pinned_ids={"leaf"}
        )

        # Parent should be selected since leaf doesn't fit
        assert "parent" in result.tiling.node_ids
        assert "leaf" not in result.tiling.node_ids

    def test_multiple_pinned_nodes_all_get_high_relevance(self) -> None:
        """Multiple pinned nodes should all get relevance=1.0."""
        # Tree: root -> (left, right), both children pinned
        left = MockNode("left", 80, 0, 50)
        right = MockNode("right", 80, 50, 100)
        root = MockNode("root", 40, 0, 100, "left", "right")

        nodes = cast(dict[str, TreeNode], {"left": left, "right": right, "root": root})
        # Low relevance for both leaves normally
        scores = {"left": 0.1, "right": 0.1, "root": 0.9}

        query_config = QueryConfig(budget_tokens=200)
        generator = DynamicTilingGenerator(query_config)

        # With both leaves pinned, they should both be selected
        result = generator.find_optimal_tiling_over_roots(
            ["root"], 200, scores, nodes, pinned_ids={"left", "right"}
        )

        assert "left" in result.tiling.node_ids
        assert "right" in result.tiling.node_ids
        assert "root" not in result.tiling.node_ids

    def test_pinned_ids_none_behaves_normally(self) -> None:
        """When pinned_ids is None, behavior should be unchanged."""
        leaf = MockNode("leaf", 100, 0, 100)
        root = MockNode("root", 50, 0, 100, "leaf", None)

        nodes = cast(dict[str, TreeNode], {"leaf": leaf, "root": root})
        scores = {"leaf": 0.3, "root": 0.9}

        query_config = QueryConfig(budget_tokens=200)
        generator = DynamicTilingGenerator(query_config)

        # Without pinned_ids, should behave as before
        result_none = generator.find_optimal_tiling_over_roots(
            ["root"], 200, scores, nodes, pinned_ids=None
        )

        # Root has higher relevance, should be preferred
        assert "root" in result_none.tiling.node_ids

    def test_pinned_ids_empty_set_behaves_normally(self) -> None:
        """When pinned_ids is empty set, behavior should be unchanged."""
        leaf = MockNode("leaf", 100, 0, 100)
        root = MockNode("root", 50, 0, 100, "leaf", None)

        nodes = cast(dict[str, TreeNode], {"leaf": leaf, "root": root})
        scores = {"leaf": 0.3, "root": 0.9}

        query_config = QueryConfig(budget_tokens=200)
        generator = DynamicTilingGenerator(query_config)

        # With empty pinned_ids, should behave normally
        result_empty = generator.find_optimal_tiling_over_roots(
            ["root"], 200, scores, nodes, pinned_ids=set()
        )

        # Root has higher relevance, should be preferred
        assert "root" in result_empty.tiling.node_ids

    def test_budget_guarantee_maintained_with_pinning(self) -> None:
        """Even with pinned nodes, output should never exceed budget."""
        # Create a deeper tree
        l1 = MockNode("l1", 60, 0, 25)
        l2 = MockNode("l2", 60, 25, 50)
        l3 = MockNode("l3", 60, 50, 75)
        l4 = MockNode("l4", 60, 75, 100)
        m1 = MockNode("m1", 30, 0, 50, "l1", "l2")
        m2 = MockNode("m2", 30, 50, 100, "l3", "l4")
        root = MockNode("root", 20, 0, 100, "m1", "m2")

        nodes = cast(
            dict[str, TreeNode],
            {
                "l1": l1,
                "l2": l2,
                "l3": l3,
                "l4": l4,
                "m1": m1,
                "m2": m2,
                "root": root,
            },
        )
        scores = {
            "l1": 0.5,
            "l2": 0.5,
            "l3": 0.5,
            "l4": 0.5,
            "m1": 0.6,
            "m2": 0.6,
            "root": 0.7,
        }

        query_config = QueryConfig(budget_tokens=150)
        generator = DynamicTilingGenerator(query_config)

        # Pin some leaves
        result = generator.find_optimal_tiling_over_roots(
            ["root"], 150, scores, nodes, pinned_ids={"l1", "l4"}
        )

        # Calculate total tokens in tiling
        total_tokens = sum(
            nodes[node_id].token_count for node_id in result.tiling.node_ids
        )

        # Budget guarantee: never exceed budget
        assert total_tokens <= 150
