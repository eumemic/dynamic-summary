"""Test budget splitting logic in dynamic tiling."""

import math
from types import SimpleNamespace
from typing import cast

from ragzoom.config import QueryConfig
from ragzoom.contracts.tree_node import TreeNode as ProtoTreeNode
from ragzoom.dynamic_tiling import DynamicTilingGenerator


def _make_node(
    *,
    id: str,
    text: str,
    span_start: int,
    span_end: int,
    document_id: str,
    token_count: int,
    parent_id: str | None = None,
    left_child_id: str | None = None,
    right_child_id: str | None = None,
) -> SimpleNamespace:
    """Create a minimal TreeNode-like object for testing."""
    node = SimpleNamespace(
        id=id,
        text=text,
        span_start=span_start,
        span_end=span_end,
        document_id=document_id,
        token_count=token_count,
        parent_id=parent_id,
        left_child_id=left_child_id,
        right_child_id=right_child_id,
        height=0 if left_child_id is None else 1,
        is_pinned=False,
        preceding_neighbor_id=None,
        following_neighbor_id=None,
        level_index=0,
    )
    return node


class TestBudgetSplitting:
    """Test the proportional budget splitting logic."""

    def test_proportional_split_maintains_ratios(self) -> None:
        """Test that budget splitting maintains intended proportions."""
        query_config = QueryConfig()
        generator = DynamicTilingGenerator(query_config)

        # Create mock nodes
        # Need to create text that actually results in 100 tokens
        # "word " repeated gives approximately 1 token per word
        left_text = " ".join(["word"] * 100)  # ~100 tokens
        right_text = " ".join(["word"] * 100)  # ~100 tokens

        left_child = _make_node(
            id="left",
            text=left_text,
            span_start=0,
            span_end=100,
            parent_id="parent",
            document_id="doc",
            token_count=100,
        )

        right_child = _make_node(
            id="right",
            text=right_text,
            span_start=100,
            span_end=200,
            parent_id="parent",
            document_id="doc",
            token_count=100,
        )

        parent = _make_node(
            id="parent",
            text="summary",
            span_start=0,
            span_end=200,
            left_child_id="left",
            right_child_id="right",
            document_id="doc",
            token_count=10,
        )

        # Set up generator's nodes dict
        generator._nodes = {
            "left": cast(ProtoTreeNode, left_child),
            "right": cast(ProtoTreeNode, right_child),
            "parent": cast(ProtoTreeNode, parent),
        }

        # Test case from user: budget=300, both children need 100, relevance ratio 1:2
        budget = 300
        scores = {
            "left": 1.0,  # Relevance 1
            "right": 2.0,  # Relevance 2
        }

        # Calculate subtree relevances (in this case, just the node scores)
        generator._subtree_relevance_cache = {"left": 1.0, "right": 2.0}

        allocation = generator._split_budget_proportionally(
            budget, cast(ProtoTreeNode, parent), scores
        )
        assert allocation is not None
        budget_l, budget_r = allocation

        # Compute minimum cover costs and ensure they're respected
        min_left = generator._get_min_cover_cost(cast(ProtoTreeNode, left_child))
        min_right = generator._get_min_cover_cost(cast(ProtoTreeNode, right_child))
        assert budget_l >= min_left
        assert budget_r >= min_right
        assert budget_l + budget_r == budget, "Total allocation must equal budget"

        extra_total = budget - (min_left + min_right)
        extra_left = budget_l - min_left
        extra_right = budget_r - min_right
        assert extra_left + extra_right == extra_total

        if extra_total > 0 and extra_right > 0:
            ratio = extra_left / extra_right
            expected_ratio = scores["left"] / scores["right"]
            assert math.isclose(
                ratio, expected_ratio, rel_tol=0.05
            ), f"Extra ratio {ratio:.2f} != expected {expected_ratio:.2f}"

    def test_minimum_constraints_respected(self) -> None:
        """Test that minimum token requirements are always met."""
        query_config = QueryConfig()
        generator = DynamicTilingGenerator(query_config)

        # Create nodes with different costs
        left_text = " ".join(["word"] * 150)  # ~150 tokens
        right_text = " ".join(["word"] * 50)  # ~50 tokens

        left_child = _make_node(
            id="left",
            text=left_text,
            span_start=0,
            span_end=150,
            parent_id="parent",
            document_id="doc",
            token_count=150,
        )

        right_child = _make_node(
            id="right",
            text=right_text,
            span_start=150,
            span_end=200,
            parent_id="parent",
            document_id="doc",
            token_count=50,
        )

        parent = _make_node(
            id="parent",
            text="summary",
            span_start=0,
            span_end=200,
            left_child_id="left",
            right_child_id="right",
            document_id="doc",
            token_count=10,
        )

        generator._nodes = {
            "left": cast(ProtoTreeNode, left_child),
            "right": cast(ProtoTreeNode, right_child),
            "parent": cast(ProtoTreeNode, parent),
        }

        # High relevance on right, but it needs fewer tokens
        scores = {"left": 1.0, "right": 9.0}  # 10% vs 90% relevance
        generator._subtree_relevance_cache = {"left": 1.0, "right": 9.0}

        # Budget of 250 tokens
        # Proportionally: left should get 25, right should get 225
        # But left needs minimum 150, so it should get 150 and right gets 100
        budget = 250
        allocation = generator._split_budget_proportionally(
            budget, cast(ProtoTreeNode, parent), scores
        )
        assert allocation is not None
        budget_l, budget_r = allocation

        # Verify minimums are met
        assert budget_l >= 150, f"Left minimum not met: {budget_l} < 150"
        assert budget_r >= 50, f"Right minimum not met: {budget_r} < 50"
        assert budget_l + budget_r == budget

    def test_tight_budget_handling(self) -> None:
        """Test behavior when budget barely covers minimums."""
        query_config = QueryConfig()
        generator = DynamicTilingGenerator(query_config)

        left_text = " ".join(["word"] * 100)  # ~100 tokens
        right_text = " ".join(["word"] * 100)  # ~100 tokens

        left_child = _make_node(
            id="left",
            text=left_text,
            span_start=0,
            span_end=100,
            parent_id="parent",
            document_id="doc",
            token_count=100,
        )

        right_child = _make_node(
            id="right",
            text=right_text,
            span_start=100,
            span_end=200,
            parent_id="parent",
            document_id="doc",
            token_count=100,
        )

        parent = _make_node(
            id="parent",
            text="summary",
            span_start=0,
            span_end=200,
            left_child_id="left",
            right_child_id="right",
            document_id="doc",
            token_count=10,
        )

        generator._nodes = {
            "left": cast(ProtoTreeNode, left_child),
            "right": cast(ProtoTreeNode, right_child),
            "parent": cast(ProtoTreeNode, parent),
        }

        # Budget exactly equals minimum requirements
        budget = 200  # Each child needs 100
        scores = {"left": 1.0, "right": 2.0}
        generator._subtree_relevance_cache = {"left": 1.0, "right": 2.0}

        allocation = generator._split_budget_proportionally(
            budget, cast(ProtoTreeNode, parent), scores
        )
        assert allocation is not None
        budget_l, budget_r = allocation

        # With tight budget, should split by minimum costs (equal in this case)
        assert budget_l == 100
        assert budget_r == 100
        assert budget_l + budget_r == budget
