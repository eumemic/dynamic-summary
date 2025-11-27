"""Tests for verbatim recent leaves feature.

The verbatim feature allows specifying a token budget for recent content
that should be included verbatim (not summarized). Recent leaves are
selected right-to-left until the verbatim budget is exhausted, then
pinned via the transient pinning mechanism.
"""

from unittest.mock import MagicMock

from ragzoom.config import QueryConfig
from ragzoom.dynamic_tiling import DPResult
from ragzoom.retrieval.verbatim_selector import select_verbatim_leaves
from ragzoom.tiling import Tiling


class MockLeaf:
    """Mock leaf node for verbatim selection testing."""

    def __init__(
        self,
        node_id: str,
        token_count: int,
        span_start: int,
        span_end: int,
    ) -> None:
        self.id = node_id
        self.token_count = token_count
        self.span_start = span_start
        self.span_end = span_end


class TestVerbatimLeafSelection:
    """Test the select_verbatim_leaves function."""

    def test_selects_rightmost_leaves_first(self) -> None:
        """Leaves should be selected from most recent (rightmost) first."""
        # 4 leaves in order: l1, l2, l3, l4
        # Budget fits exactly 2 leaves (100 tokens each)
        leaves = [
            MockLeaf("l1", 100, 0, 25),
            MockLeaf("l2", 100, 25, 50),
            MockLeaf("l3", 100, 50, 75),
            MockLeaf("l4", 100, 75, 100),
        ]

        selected, horizon = select_verbatim_leaves(leaves, verbatim_budget=200)

        # Should select l4 and l3 (most recent), not l1 and l2
        selected_ids = {leaf.id for leaf in selected}
        assert selected_ids == {"l3", "l4"}
        assert horizon == 50  # Leftmost span_start of selected leaves

    def test_budget_exhausted_exactly(self) -> None:
        """Should select leaves until budget is exhausted, not exceeded."""
        leaves = [
            MockLeaf("l1", 60, 0, 25),
            MockLeaf("l2", 60, 25, 50),
            MockLeaf("l3", 80, 50, 75),
            MockLeaf("l4", 50, 75, 100),
        ]

        # Budget = 130: fits l4 (50) + l3 (80) = 130 exactly
        selected, horizon = select_verbatim_leaves(leaves, verbatim_budget=130)

        selected_ids = {leaf.id for leaf in selected}
        assert selected_ids == {"l3", "l4"}
        total_tokens = sum(leaf.token_count for leaf in selected)
        assert total_tokens == 130

    def test_budget_not_exceeded(self) -> None:
        """Should not exceed budget even if leaves remain."""
        leaves = [
            MockLeaf("l1", 100, 0, 25),
            MockLeaf("l2", 100, 25, 50),
            MockLeaf("l3", 100, 50, 75),
            MockLeaf("l4", 100, 75, 100),
        ]

        # Budget = 150: fits l4 (100), but not l4 + l3 (200)
        selected, horizon = select_verbatim_leaves(leaves, verbatim_budget=150)

        selected_ids = {leaf.id for leaf in selected}
        assert selected_ids == {"l4"}
        assert horizon == 75

    def test_empty_budget_returns_nothing(self) -> None:
        """Zero budget should return no leaves."""
        leaves = [MockLeaf("l1", 100, 0, 100)]

        selected, horizon = select_verbatim_leaves(leaves, verbatim_budget=0)

        assert selected == []
        assert horizon == 0

    def test_empty_leaves_returns_nothing(self) -> None:
        """Empty leaf list should return no leaves."""
        empty_leaves: list[MockLeaf] = []
        selected, horizon = select_verbatim_leaves(empty_leaves, verbatim_budget=1000)

        assert selected == []
        assert horizon == 0

    def test_all_leaves_fit(self) -> None:
        """When all leaves fit in budget, select all of them."""
        leaves = [
            MockLeaf("l1", 50, 0, 25),
            MockLeaf("l2", 50, 25, 50),
            MockLeaf("l3", 50, 50, 75),
        ]

        selected, horizon = select_verbatim_leaves(leaves, verbatim_budget=200)

        selected_ids = {leaf.id for leaf in selected}
        assert selected_ids == {"l1", "l2", "l3"}
        assert horizon == 0  # Spans from the beginning

    def test_single_leaf_too_large(self) -> None:
        """Single leaf larger than budget should not be selected."""
        leaves = [MockLeaf("l1", 200, 0, 100)]

        selected, horizon = select_verbatim_leaves(leaves, verbatim_budget=100)

        assert selected == []
        assert horizon == 0

    def test_returns_leaves_in_span_order(self) -> None:
        """Selected leaves should be returned in span order (left to right)."""
        leaves = [
            MockLeaf("l1", 50, 0, 25),
            MockLeaf("l2", 50, 25, 50),
            MockLeaf("l3", 50, 50, 75),
            MockLeaf("l4", 50, 75, 100),
        ]

        # Budget fits all 4 leaves
        selected, _ = select_verbatim_leaves(leaves, verbatim_budget=200)

        # Should be ordered by span_start
        span_starts = [leaf.span_start for leaf in selected]
        assert span_starts == sorted(span_starts)

    def test_horizon_is_leftmost_selected_span(self) -> None:
        """Horizon should be the span_start of the leftmost selected leaf."""
        leaves = [
            MockLeaf("l1", 100, 0, 33),
            MockLeaf("l2", 100, 33, 66),
            MockLeaf("l3", 100, 66, 100),
        ]

        # Budget fits only l3
        selected, horizon = select_verbatim_leaves(leaves, verbatim_budget=100)

        assert horizon == 66
        assert len(selected) == 1
        assert selected[0].id == "l3"

    def test_variable_sized_leaves(self) -> None:
        """Handle leaves with different token counts correctly."""
        leaves = [
            MockLeaf("small", 20, 0, 10),
            MockLeaf("medium", 80, 10, 50),
            MockLeaf("large", 150, 50, 80),
            MockLeaf("tiny", 10, 80, 100),
        ]

        # Budget = 100: should fit tiny (10) + medium (80) or just large (150)?
        # Since we go right-to-left: tiny (10) first, then try large (150) - too big
        # Then medium (80) fits! Total = 90
        selected, horizon = select_verbatim_leaves(leaves, verbatim_budget=100)

        selected_ids = {leaf.id for leaf in selected}
        # tiny (10) + medium (80) = 90 fits
        assert selected_ids == {"tiny", "medium"}
        total_tokens = sum(leaf.token_count for leaf in selected)
        assert total_tokens <= 100


class TestVerbatimSeparation:
    """Test that verbatim leaves are appended after tiling, not mixed in."""

    def test_tiling_uses_base_budget_only(self) -> None:
        """Tiling should use base_budget only; verbatim leaves are appended after.

        This ensures verbatim selection doesn't constrain the seed-based tiling.
        """
        from unittest.mock import patch

        from ragzoom.retrieve import Retriever

        # Create minimal mocks for retriever construction
        mock_query_config = QueryConfig(budget_tokens=2000)
        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_leaves.return_value = []
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding.return_value = [0.1] * 1536
        mock_budget_planner = MagicMock()
        mock_budget_planner.calculate_conservative_num_seeds.return_value = 1
        mock_vector_index = MagicMock()
        mock_vector_index.search_similar.return_value = []

        retriever = Retriever(
            mock_query_config,
            mock_doc_store,
            mock_embedding_service,
            mock_budget_planner,
            mock_vector_index,
        )

        # Capture the budget passed to tiling
        captured_budgets: list[int] = []

        def capture_tiling_call(
            root_ids: object,
            budget_tokens: int,
            scores: object,
            nodes: object,
        ) -> DPResult:
            captured_budgets.append(budget_tokens)
            return DPResult(Tiling.empty(), [], 0.0, {})

        # Create a mock root node that will be found in coverage
        mock_root_node = MagicMock()
        mock_root_node.id = "node1"
        mock_root_node.parent_id = None  # Root node
        mock_root_node.span_start = 0
        mock_root_node.is_root = lambda: True

        # Patch at multiple levels to ensure we reach the tiling call
        with (
            patch.object(
                retriever.greedy_generator,
                "find_optimal_tiling_over_roots",
                side_effect=capture_tiling_call,
            ),
            patch("ragzoom.retrieve.CoverageBuilder") as mock_coverage_builder_class,
            patch("ragzoom.retrieve.ScoringService") as mock_scoring_service_class,
        ):
            # Set up coverage builder mock with actual node
            mock_coverage_result = MagicMock()
            mock_coverage_result.coverage_map = {"node1": True}
            mock_coverage_result.nodes = {"node1": mock_root_node}
            mock_coverage_builder = MagicMock()
            mock_coverage_builder.build_complete_coverage.return_value = (
                mock_coverage_result
            )
            mock_coverage_builder_class.return_value = mock_coverage_builder

            # Set up scoring service mock
            mock_scoring_service = MagicMock()
            mock_scoring_service.compute_scores.return_value = {"node1": 0.5}
            mock_scoring_service_class.return_value = mock_scoring_service

            # Run retrieval with verbatim budget
            base_budget = 1000
            verbatim_budget = 500

            retriever.retrieve(
                "test query",
                budget_tokens=base_budget,
                recent_verbatim_budget=verbatim_budget,
            )

        # Tiling should receive only base_budget (verbatim appended after)
        assert len(captured_budgets) == 1, "Tiling generator was not called"
        assert captured_budgets[0] == base_budget, (
            f"Expected base budget {base_budget}, got {captured_budgets[0]}. "
            f"Tiling should use base_budget only; verbatim leaves appended after."
        )
