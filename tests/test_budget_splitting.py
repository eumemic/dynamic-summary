"""Test budget splitting logic in dynamic tiling."""

from ragzoom.config import QueryConfig
from ragzoom.dynamic_tiling import DynamicTilingGenerator
from ragzoom.store import TreeNode


class TestBudgetSplitting:
    """Test the proportional budget splitting logic."""

    def test_proportional_split_maintains_ratios(self):
        """Test that budget splitting maintains intended proportions."""
        query_config = QueryConfig()
        generator = DynamicTilingGenerator(query_config)

        # Create mock nodes
        # Need to create text that actually results in 100 tokens
        # "word " repeated gives approximately 1 token per word
        left_text = " ".join(["word"] * 100)  # ~100 tokens
        right_text = " ".join(["word"] * 100)  # ~100 tokens

        left_child = TreeNode(
            id="left", text=left_text, span_start=0, span_end=100, parent_id="parent"
        )

        right_child = TreeNode(
            id="right",
            text=right_text,
            span_start=100,
            span_end=200,
            parent_id="parent",
        )

        parent = TreeNode(
            id="parent",
            text="summary",
            span_start=0,
            span_end=200,
            left_child_id="left",
            right_child_id="right",
        )

        # Set up generator's nodes dict
        generator._nodes = {"left": left_child, "right": right_child, "parent": parent}

        # Test case from user: budget=300, both children need 100, relevance ratio 1:2
        budget = 300
        scores = {
            "left": 1.0,  # Relevance 1
            "right": 2.0,  # Relevance 2
        }

        # Calculate subtree relevances (in this case, just the node scores)
        generator._subtree_relevance_cache = {"left": 1.0, "right": 2.0}

        budget_l, budget_r = generator._split_budget_proportionally(
            budget, parent, scores
        )

        # Should allocate 100 to left (1/3 of 300) and 200 to right (2/3 of 300)
        assert budget_l == 100, f"Left should get 100 tokens, got {budget_l}"
        assert budget_r == 200, f"Right should get 200 tokens, got {budget_r}"
        assert budget_l + budget_r == budget, "Total allocation must equal budget"

        # Test the ratio
        ratio = budget_l / budget_r
        expected_ratio = 1.0 / 2.0
        assert (
            abs(ratio - expected_ratio) < 0.01
        ), f"Ratio {ratio:.2f} != expected {expected_ratio}"

    def test_minimum_constraints_respected(self):
        """Test that minimum token requirements are always met."""
        query_config = QueryConfig()
        generator = DynamicTilingGenerator(query_config)

        # Create nodes with different costs
        left_text = " ".join(["word"] * 150)  # ~150 tokens
        right_text = " ".join(["word"] * 50)  # ~50 tokens

        left_child = TreeNode(
            id="left", text=left_text, span_start=0, span_end=150, parent_id="parent"
        )

        right_child = TreeNode(
            id="right",
            text=right_text,
            span_start=150,
            span_end=200,
            parent_id="parent",
        )

        parent = TreeNode(
            id="parent",
            text="summary",
            span_start=0,
            span_end=200,
            left_child_id="left",
            right_child_id="right",
        )

        generator._nodes = {"left": left_child, "right": right_child, "parent": parent}

        # High relevance on right, but it needs fewer tokens
        scores = {"left": 1.0, "right": 9.0}  # 10% vs 90% relevance
        generator._subtree_relevance_cache = {"left": 1.0, "right": 9.0}

        # Budget of 250 tokens
        # Proportionally: left should get 25, right should get 225
        # But left needs minimum 150, so it should get 150 and right gets 100
        budget = 250
        budget_l, budget_r = generator._split_budget_proportionally(
            budget, parent, scores
        )

        # Verify minimums are met
        assert budget_l >= 150, f"Left minimum not met: {budget_l} < 150"
        assert budget_r >= 50, f"Right minimum not met: {budget_r} < 50"
        assert budget_l + budget_r == budget

    def test_tight_budget_handling(self):
        """Test behavior when budget barely covers minimums."""
        query_config = QueryConfig()
        generator = DynamicTilingGenerator(query_config)

        left_text = " ".join(["word"] * 100)  # ~100 tokens
        right_text = " ".join(["word"] * 100)  # ~100 tokens

        left_child = TreeNode(
            id="left", text=left_text, span_start=0, span_end=100, parent_id="parent"
        )

        right_child = TreeNode(
            id="right",
            text=right_text,
            span_start=100,
            span_end=200,
            parent_id="parent",
        )

        parent = TreeNode(
            id="parent",
            text="summary",
            span_start=0,
            span_end=200,
            left_child_id="left",
            right_child_id="right",
        )

        generator._nodes = {"left": left_child, "right": right_child, "parent": parent}

        # Budget exactly equals minimum requirements
        budget = 200  # Each child needs 100
        scores = {"left": 1.0, "right": 2.0}
        generator._subtree_relevance_cache = {"left": 1.0, "right": 2.0}

        budget_l, budget_r = generator._split_budget_proportionally(
            budget, parent, scores
        )

        # With tight budget, should split by minimum costs (equal in this case)
        assert budget_l == 100
        assert budget_r == 100
        assert budget_l + budget_r == budget
