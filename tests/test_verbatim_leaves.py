"""Tests for verbatim recent leaves feature.

The verbatim feature allows specifying a token budget for recent content
that should be included verbatim (not summarized). Recent leaves are
selected right-to-left until the verbatim budget is exhausted, then
pinned via the transient pinning mechanism.
"""

from collections.abc import Mapping
from unittest.mock import AsyncMock, MagicMock

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


class TestVerbatimTilingInvariant:
    """Regression tests for tiling invariants with verbatim leaves."""

    def test_verbatim_leaves_must_not_overlap_with_tiling(self) -> None:
        """Verbatim leaves appended to tiling must not create overlaps.

        Regression test: The naive approach of appending verbatim leaves after
        tiling can create overlaps if the tiling already includes a summary node
        that covers the verbatim region.

        Example: Tiling includes summary [0, 1000), verbatim leaf is [800, 900).
        Appending the leaf creates overlap - the summary covers [800, 900) too.
        """
        from ragzoom.validate import validate_tiling

        # Create mock nodes representing the bug scenario:
        # - A summary node covering the whole document [0, 1000)
        # - A verbatim leaf at the end [800, 1000) which is a descendant

        class MockNode:
            def __init__(
                self,
                node_id: str,
                span_start: int,
                span_end: int,
                token_count: int,
            ) -> None:
                self.id = node_id
                self.span_start = span_start
                self.span_end = span_end
                self.token_count = token_count
                self.parent_id: str | None = None
                self.left_child_id: str | None = None
                self.right_child_id: str | None = None

            def is_root(self) -> bool:
                return self.parent_id is None

        # Build a simple tree:
        # root [0, 1000) -> left [0, 800), right [800, 1000) (verbatim leaf)
        root = MockNode("root", 0, 1000, 100)
        left_child = MockNode("left", 0, 800, 80)
        verbatim_leaf = MockNode("verbatim", 800, 1000, 20)

        root.left_child_id = "left"
        root.right_child_id = "verbatim"
        left_child.parent_id = "root"
        verbatim_leaf.parent_id = "root"

        nodes = {
            "root": root,
            "left": left_child,
            "verbatim": verbatim_leaf,
        }

        # Mock document store that returns our nodes
        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_node.side_effect = lambda nid: nodes.get(nid)
        mock_doc_store.nodes.get_all.return_value = list(nodes.values())

        # Scenario: tiling algorithm chose root (covers whole doc),
        # then we naively append verbatim leaf -> OVERLAP!
        tiling_with_overlap = ["root", "verbatim"]

        # This SHOULD fail validation due to overlap
        error = validate_tiling(
            tiling_with_overlap,
            mock_doc_store,
        )

        assert error is not None, (
            "Tiling validation should detect overlap between root [0,1000) "
            "and verbatim [800,1000)"
        )
        assert "overlap" in error.lower(), f"Expected overlap error, got: {error}"

    def test_retriever_verbatim_must_not_create_overlaps(self) -> None:
        """Retriever must not produce overlapping tiling when adding verbatim leaves.

        Regression test for bug: retrieve.py appended verbatim leaves to tiling
        without checking if they overlap with existing tiling nodes.

        This test mocks a scenario where:
        1. Greedy tiling returns a summary node covering [0, 1000)
        2. Verbatim selection picks a leaf at [800, 1000)
        3. Naive append would create overlap -> validation MUST fail
        """
        from unittest.mock import patch

        from ragzoom.retrieve import Retriever
        from ragzoom.validate import validate_tiling

        class MockNode:
            def __init__(
                self,
                node_id: str,
                span_start: int,
                span_end: int,
                token_count: int,
                parent_id: str | None = None,
            ) -> None:
                self.id = node_id
                self.span_start = span_start
                self.span_end = span_end
                self.token_count = token_count
                self.parent_id = parent_id
                self.left_child_id: str | None = None
                self.right_child_id: str | None = None
                self.text = f"text for {node_id}"

            def is_root(self) -> bool:
                return self.parent_id is None

        # Build tree: root [0, 1000) with children
        root = MockNode("root", 0, 1000, 50)
        left = MockNode("left", 0, 800, 40, parent_id="root")
        verbatim_leaf = MockNode("verbatim_leaf", 800, 1000, 10, parent_id="root")

        root.left_child_id = "left"
        root.right_child_id = "verbatim_leaf"

        all_nodes = {"root": root, "left": left, "verbatim_leaf": verbatim_leaf}

        # Setup retriever with mocks
        mock_query_config = QueryConfig(budget_tokens=2000)
        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_leaves.return_value = [verbatim_leaf]
        mock_doc_store.nodes.get_nodes.return_value = list(all_nodes.values())
        mock_doc_store.nodes.get_node.side_effect = lambda nid: all_nodes.get(nid)
        mock_doc_store.nodes.get_all.return_value = list(all_nodes.values())

        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding_async = AsyncMock(
            return_value=[0.1] * 1536
        )
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

        # Mock greedy tiling to return just the root (covers whole doc)
        # This simulates the bug: tiling returns root which covers verbatim region
        def mock_tiling(
            root_ids: object,
            budget_tokens: int,
            scores: Mapping[str, float],
            nodes: object,
        ) -> DPResult:
            # With score boosting fix, verbatim_leaf gets score=1.0 which prevents
            # it from being rolled up in favor of parent. Check if verbatim_leaf
            # has boosted score to determine correct vs bug behavior.
            if scores.get("verbatim_leaf", 0.0) >= 1.0:
                # Correct behavior: verbatim_leaf has max score, won't be rolled up
                return DPResult(
                    Tiling(node_ids=["left", "verbatim_leaf"], relevance_tokens=50.0),
                    [],
                    50.0,
                    {},
                )
            # Bug behavior: returns root which overlaps with verbatim_leaf
            return DPResult(
                Tiling(node_ids=["root"], relevance_tokens=50.0),
                [],
                50.0,
                {},
            )

        with (
            patch.object(
                retriever.greedy_generator,
                "find_optimal_tiling_over_roots",
                side_effect=mock_tiling,
            ),
            patch("ragzoom.retrieve.CoverageBuilder") as mock_coverage_class,
            patch("ragzoom.retrieve.ScoringService") as mock_scoring_class,
        ):
            mock_coverage = MagicMock()
            mock_coverage.coverage_map = {"root": True}
            mock_coverage.nodes = {"root": root}
            mock_coverage_builder = MagicMock()
            mock_coverage_builder.build_complete_coverage.return_value = mock_coverage
            mock_coverage_class.return_value = mock_coverage_builder

            mock_scoring = MagicMock()
            mock_scoring.compute_scores.return_value = {"root": 0.5}
            mock_scoring_class.return_value = mock_scoring

            # Run retrieval with verbatim budget
            result = retriever.retrieve(
                "test query",
                budget_tokens=1000,
                recent_verbatim_budget=100,  # Should select verbatim_leaf
            )

        # The tiling should NOT have overlaps
        # Current buggy implementation appends verbatim_leaf to ["root"],
        # creating ["root", "verbatim_leaf"] which overlaps!
        assert result.tiling is not None, "Tiling should not be None"
        error = validate_tiling(result.tiling, mock_doc_store)

        assert error is None, (
            f"Tiling produced overlaps! Error: {error}\n"
            f"Tiling IDs: {result.tiling}\n"
            f"This is a regression - verbatim leaves must not overlap with "
            f"summary nodes that already cover their span."
        )

    def test_valid_tiling_without_overlap(self) -> None:
        """A properly constructed tiling should pass validation."""
        from ragzoom.validate import validate_tiling

        class MockNode:
            def __init__(
                self,
                node_id: str,
                span_start: int,
                span_end: int,
                token_count: int,
            ) -> None:
                self.id = node_id
                self.span_start = span_start
                self.span_end = span_end
                self.token_count = token_count
                self.parent_id: str | None = None
                self.left_child_id: str | None = None
                self.right_child_id: str | None = None

            def is_root(self) -> bool:
                return self.parent_id is None

        # Valid tiling: left_child [0, 800) + verbatim [800, 1000) = no overlap
        left_child = MockNode("left", 0, 800, 80)
        verbatim_leaf = MockNode("verbatim", 800, 1000, 20)

        nodes = {
            "left": left_child,
            "verbatim": verbatim_leaf,
        }

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_node.side_effect = lambda nid: nodes.get(nid)
        mock_doc_store.nodes.get_all.return_value = list(nodes.values())

        # This tiling has no overlaps - adjacent spans
        valid_tiling = ["left", "verbatim"]

        error = validate_tiling(
            valid_tiling,
            mock_doc_store,
        )

        assert error is None, f"Valid tiling should pass, but got error: {error}"


class TestVerbatimBudgetIntegration:
    """Test that verbatim budget is correctly integrated with tiling."""

    def test_tiling_uses_combined_budget(self) -> None:
        """Tiling should use combined budget (base + verbatim).

        Since pinned verbatim leaves are in coverage and cannot be rolled up,
        the tiling budget must include both base and verbatim portions.
        """
        from unittest.mock import patch

        from ragzoom.retrieve import Retriever

        # Create minimal mocks for retriever construction
        mock_query_config = QueryConfig(budget_tokens=2000)
        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_leaves.return_value = []
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding_async = AsyncMock(
            return_value=[0.1] * 1536
        )
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

        # Tiling should receive combined budget (base + verbatim) since pinned
        # leaves are included in coverage and can't be rolled up
        assert len(captured_budgets) == 1, "Tiling generator was not called"
        expected_budget = base_budget + verbatim_budget
        assert captured_budgets[0] == expected_budget, (
            f"Expected combined budget {expected_budget}, got {captured_budgets[0]}. "
            f"Tiling should use base_budget + verbatim_budget since pinned leaves "
            f"are in coverage and budget must account for them."
        )

    def test_seeds_filtered_to_exclude_verbatim_region(self) -> None:
        """Vector search should filter seeds to exclude the verbatim region.

        When verbatim budget is specified, seeds must come from BEFORE the
        verbatim horizon (span_end < horizon) to prevent overlap between
        relevance-based seeds and verbatim content.
        """
        from collections.abc import Sequence
        from unittest.mock import patch

        from ragzoom.contracts.vector_filter import (
            SpanEndLtFilter,
            VectorFilter,
        )
        from ragzoom.retrieve import Retriever

        # Create leaves: l1 [0,50), l2 [50,100), l3 [100,150), l4 [150,200)
        # With verbatim_budget=100, the repo returns l3+l4 (most recent within budget)
        # in span order, so horizon = l3.span_start = 100
        verbatim_leaves = [
            MockLeaf("l3", 50, 100, 150),
            MockLeaf("l4", 50, 150, 200),
        ]

        mock_query_config = QueryConfig(budget_tokens=2000)
        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_recent_leaves_within_budget.return_value = (
            verbatim_leaves
        )
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding_async = AsyncMock(
            return_value=[0.1] * 1536
        )
        mock_budget_planner = MagicMock()
        mock_budget_planner.calculate_conservative_num_seeds.return_value = 1

        # Capture what filters are passed to search_similar
        captured_filters: list[Sequence[VectorFilter] | None] = []

        mock_vector_index = MagicMock()

        def capture_search(
            query_embedding: object,
            k: int,
            filters: Sequence[VectorFilter] | None = None,
        ) -> list[object]:
            captured_filters.append(filters)
            return []  # No candidates for simplicity

        mock_vector_index.search_similar.side_effect = capture_search

        retriever = Retriever(
            mock_query_config,
            mock_doc_store,
            mock_embedding_service,
            mock_budget_planner,
            mock_vector_index,
        )

        # Mock root node for coverage
        mock_root_node = MagicMock()
        mock_root_node.id = "root"
        mock_root_node.parent_id = None
        mock_root_node.span_start = 0
        mock_root_node.is_root = lambda: True

        with (
            patch("ragzoom.retrieve.CoverageBuilder") as mock_coverage_class,
            patch("ragzoom.retrieve.ScoringService") as mock_scoring_class,
            patch.object(
                retriever.greedy_generator,
                "find_optimal_tiling_over_roots",
                return_value=DPResult(Tiling.empty(), [], 0.0, {}),
            ),
        ):
            mock_coverage = MagicMock()
            mock_coverage.coverage_map = {"root": True}
            mock_coverage.nodes = {"root": mock_root_node}
            mock_coverage_builder = MagicMock()
            mock_coverage_builder.build_complete_coverage.return_value = mock_coverage
            mock_coverage_class.return_value = mock_coverage_builder

            mock_scoring = MagicMock()
            mock_scoring.compute_scores.return_value = {"root": 0.5}
            mock_scoring_class.return_value = mock_scoring

            # Retrieve with verbatim budget - should establish horizon at 100
            retriever.retrieve(
                "test query",
                budget_tokens=1000,
                recent_verbatim_budget=100,  # Selects l4+l3, horizon=100
            )

        # Verify search_similar was called with SpanEndLtFilter
        assert len(captured_filters) == 1, "search_similar should be called once"
        filters = captured_filters[0]
        assert filters is not None, "Filters should not be None when verbatim is set"

        # Find the SpanEndLtFilter
        span_filter = None
        for f in filters:
            if isinstance(f, SpanEndLtFilter):
                span_filter = f
                break

        assert span_filter is not None, (
            f"Expected SpanEndLtFilter in filters, got: {filters}. "
            f"Seeds must be filtered to exclude verbatim region."
        )
        assert span_filter.threshold == 100, (
            f"Expected horizon=100 (span_start of leftmost verbatim leaf l3), "
            f"got {span_filter.threshold}"
        )
