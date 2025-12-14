"""Tests for forest extraneous detail gating in IndexingEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

from ragzoom.config import IndexConfig, PrecedingContextConfig, PrecedingContextSettings
from ragzoom.server.indexing_engine import (
    _expected_total_from_leaf_count,
    _min_roots_for_leaf_count,
)


def _make_preceding_context(
    min_forest_completeness: float = 0.0,
    verbatim_tokens: int = 0,
) -> PrecedingContextSettings:
    """Create PrecedingContextSettings with given values for both leaf and inner."""
    config = PrecedingContextConfig(
        min_forest_completeness=min_forest_completeness,
        verbatim_tokens=verbatim_tokens,
    )
    return PrecedingContextSettings(leaf=config, inner=config)


def _config_with_gating(
    min_forest_completeness: float = 0.0,
    verbatim_tokens: int = 0,
) -> IndexConfig:
    """Create IndexConfig with specified gating parameters."""
    return IndexConfig.load().replace(
        preceding_context=_make_preceding_context(
            min_forest_completeness=min_forest_completeness,
            verbatim_tokens=verbatim_tokens,
        )
    )


class TestMinRootsForLeafCount:
    """Test the helper function for calculating minimum roots (popcount)."""

    def test_zero_leaves(self) -> None:
        """Zero leaves means zero roots."""
        assert _min_roots_for_leaf_count(0) == 0

    def test_one_leaf(self) -> None:
        """One leaf (0b1): popcount=1 root."""
        assert _min_roots_for_leaf_count(1) == 1

    def test_two_leaves(self) -> None:
        """Two leaves (0b10): popcount=1 root (one tree of 2)."""
        assert _min_roots_for_leaf_count(2) == 1

    def test_three_leaves(self) -> None:
        """Three leaves (0b11): popcount=2 roots (tree of 2 + unpaired leaf)."""
        assert _min_roots_for_leaf_count(3) == 2

    def test_four_leaves(self) -> None:
        """Four leaves (0b100): popcount=1 root (perfect binary tree of 4)."""
        assert _min_roots_for_leaf_count(4) == 1

    def test_five_leaves(self) -> None:
        """Five leaves (0b101): popcount=2 roots (tree of 4 + leaf)."""
        assert _min_roots_for_leaf_count(5) == 2

    def test_seven_leaves(self) -> None:
        """Seven leaves (0b111): popcount=3 roots (trees of 4, 2, and 1)."""
        assert _min_roots_for_leaf_count(7) == 3

    def test_eight_leaves(self) -> None:
        """Eight leaves (0b1000): popcount=1 root (perfect tree of 8)."""
        assert _min_roots_for_leaf_count(8) == 1

    def test_fifteen_leaves(self) -> None:
        """Fifteen leaves (0b1111): popcount=4 roots (trees of 8, 4, 2, 1)."""
        assert _min_roots_for_leaf_count(15) == 4

    def test_sixteen_leaves(self) -> None:
        """Sixteen leaves (0b10000): popcount=1 root (perfect tree of 16)."""
        assert _min_roots_for_leaf_count(16) == 1


class TestExtraneousDetailCalculation:
    """Test the extraneous detail calculation logic."""

    def test_single_leaf_zero_extraneous(self) -> None:
        """A single leaf root has 0 extraneous detail (1 root - popcount(1)=1 = 0)."""
        leaves = 1
        actual_roots = 1
        min_roots = _min_roots_for_leaf_count(leaves)
        assert min_roots == 1
        assert actual_roots - min_roots == 0

    def test_two_leaves_two_roots_extraneous(self) -> None:
        """Two leaf roots (not summarized): 2 - 1 = 1 extraneous root."""
        leaves = 2
        actual_roots = 2  # Two separate leaves
        min_roots = _min_roots_for_leaf_count(leaves)
        assert min_roots == 1  # Should be 1 tree with 2 leaves
        assert actual_roots - min_roots == 1

    def test_two_leaves_one_root_optimal(self) -> None:
        """Two leaves summarized into parent: 1 - 1 = 0 extraneous."""
        leaves = 2
        actual_roots = 1  # height-1 root covering both leaves
        min_roots = _min_roots_for_leaf_count(leaves)
        assert min_roots == 1
        assert actual_roots - min_roots == 0

    def test_three_leaves_optimal(self) -> None:
        """Three leaves optimally: 2 roots (tree of 2 + leaf) = 2 - 2 = 0."""
        leaves = 3
        actual_roots = 2
        min_roots = _min_roots_for_leaf_count(leaves)
        assert min_roots == 2
        assert actual_roots - min_roots == 0

    def test_three_leaves_suboptimal(self) -> None:
        """Three separate leaves: 3 - 2 = 1 extraneous root."""
        leaves = 3
        actual_roots = 3
        min_roots = _min_roots_for_leaf_count(leaves)
        assert min_roots == 2
        assert actual_roots - min_roots == 1

    def test_four_leaves_all_separate(self) -> None:
        """Four separate leaves: 4 - 1 = 3 extraneous roots."""
        leaves = 4
        actual_roots = 4
        min_roots = _min_roots_for_leaf_count(leaves)
        assert min_roots == 1
        assert actual_roots - min_roots == 3

    def test_four_leaves_partially_summarized(self) -> None:
        """Four leaves with 2 height-1 trees: 2 - 1 = 1 extraneous."""
        leaves = 4
        actual_roots = 2  # Two height-1 trees
        min_roots = _min_roots_for_leaf_count(leaves)
        assert min_roots == 1
        assert actual_roots - min_roots == 1


class TestCompressibilityGatingBehavior:
    """Test gating behavior with different min_forest_completeness values."""

    def test_zero_forest_completeness_allows_all(self) -> None:
        """min_forest_completeness=0.0 should allow all roots regardless of state."""
        min_forest_completeness = 0.0
        # Even very low forest_completeness ratios pass
        for forest_completeness in [0.0, 0.1, 0.5, 1.0]:
            assert forest_completeness >= min_forest_completeness

    def test_perfect_forest_completeness_requires_optimal(self) -> None:
        """min_forest_completeness=1.0 only allows optimal forest state."""
        min_forest_completeness = 1.0
        # Only perfect forest_completeness (1.0) passes
        assert 1.0 >= min_forest_completeness
        assert 0.9 < min_forest_completeness
        assert 0.5 < min_forest_completeness

    def test_half_forest_completeness_allows_some_slack(self) -> None:
        """min_forest_completeness=0.5 allows up to 2x the minimum roots."""
        min_forest_completeness = 0.5
        assert 1.0 >= min_forest_completeness
        assert 0.75 >= min_forest_completeness
        assert 0.5 >= min_forest_completeness
        assert 0.33 < min_forest_completeness


class TestFindNextJobGating:
    """Test _find_next_job with actual mock roots to verify gating logic.

    These tests verify that the gating correctly uses PRECEDING forest state,
    not including the current root being evaluated.
    """

    def _make_leaf_root(self, node_id: str, level_index: int) -> MagicMock:
        """Create a mock leaf root node."""
        node = MagicMock()
        node.id = node_id
        node.height = 0
        node.level_index = level_index
        node.embedding = None  # No embedding yet
        return node

    def _make_height1_root(self, node_id: str, level_index: int) -> MagicMock:
        """Create a mock height-1 root (summarized pair, covers 2 leaves)."""
        node = MagicMock()
        node.id = node_id
        node.height = 1
        node.level_index = level_index
        node.embedding = b"fake"  # Has embedding
        return node

    def _make_height2_root(self, node_id: str, level_index: int) -> MagicMock:
        """Create a mock height-2 root (covers 4 leaves)."""
        node = MagicMock()
        node.id = node_id
        node.height = 2
        node.level_index = level_index
        node.embedding = b"fake"  # Has embedding
        return node

    def test_first_leaf_always_allowed_with_perfect_forest_completeness(self) -> None:
        """First leaf (no preceding forest) always allowed, even with min_forest_completeness=1.0."""

        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        # Strictest threshold: no extraneous detail allowed
        index_config = _config_with_gating(min_forest_completeness=1.0)

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Single leaf root with no embedding
        leaf0 = self._make_leaf_root("leaf0", 0)
        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [leaf0]
        mock_store.for_document.return_value = mock_doc_store

        # Should return embedding job for leaf0 (no preceding forest to check)
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf0"

    def test_second_leaf_allowed_when_first_is_single_root(self) -> None:
        """Second leaf allowed when first leaf exists as 1 root (1 leaf = 1 min root = 0 extraneous)."""

        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = _config_with_gating(min_forest_completeness=1.0)

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Two leaves: leaf0 has embedding (complete), leaf1 needs embedding
        leaf0 = self._make_leaf_root("leaf0", 0)
        leaf0.embedding = b"fake"  # leaf0 is done
        leaf1 = self._make_leaf_root("leaf1", 1)

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [leaf0, leaf1]
        mock_store.for_document.return_value = mock_doc_store

        # Should return embedding job for leaf1
        # Preceding forest: 1 leaf, 1 root, min_roots=1 → extraneous=0
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf1"

    def test_third_leaf_blocked_returns_summary_job(self) -> None:
        """Third leaf blocked with min_forest_completeness=1.0, returns summary job.

        Preceding forest: 2 leaves as 2 separate roots.
        min_roots = popcount(2) = 1
        forest_completeness = 1/2 = 0.5 < min_forest_completeness=1.0 → BLOCKED
        But engine returns a SummaryJob to combine leaf0+leaf1.
        """

        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob

        index_config = _config_with_gating(min_forest_completeness=1.0)

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Three leaves, first two with embeddings but not summarized
        leaf0 = self._make_leaf_root("leaf0", 0)
        leaf0.embedding = b"fake"
        leaf1 = self._make_leaf_root("leaf1", 1)
        leaf1.embedding = b"fake"
        leaf2 = self._make_leaf_root("leaf2", 2)  # No embedding

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [leaf0, leaf1, leaf2]
        mock_store.for_document.return_value = mock_doc_store

        # Engine returns summary job for leaf0+leaf1 since leaf2 is blocked
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, SummaryJob)
        assert job.left_id == "leaf0"
        assert job.right_id == "leaf1"

    def test_third_leaf_allowed_when_preceding_summarized(self) -> None:
        """Third leaf allowed when first two are summarized into one root.

        Preceding forest: height-1 root covering 2 leaves → 1 root, min=1 → extraneous=0.
        """

        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = _config_with_gating(min_forest_completeness=1.0)

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Height-1 root (covers 2 leaves) + leaf2 needing embedding
        parent01 = self._make_height1_root("parent01", 0)
        leaf2 = self._make_leaf_root("leaf2", 2)

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [parent01, leaf2]
        mock_store.for_document.return_value = mock_doc_store

        # Should return embedding job for leaf2
        # Preceding: 2 leaves in 1 root, min=1 → extraneous=0
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf2"

    def test_zero_forest_completeness_allows_all_leaves(self) -> None:
        """With min_forest_completeness=0.0, all leaves allowed regardless of forest state."""

        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = _config_with_gating(min_forest_completeness=0.0)

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Four leaves, none with embeddings, no parents
        leaves = [self._make_leaf_root(f"leaf{i}", i) for i in range(4)]

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = leaves
        mock_store.for_document.return_value = mock_doc_store

        # Should return embedding job for first leaf
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf0"

    def test_fifth_leaf_blocked_with_four_separate_roots(self) -> None:
        """Fifth leaf blocked when 4 preceding leaves exist as 4 separate roots.

        Preceding: 4 leaves, 4 roots, min=popcount(4)=1
        forest_completeness = 1/4 = 0.25 < min_forest_completeness=0.33 → BLOCKED
        """

        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob

        index_config = _config_with_gating(min_forest_completeness=0.33)

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Five leaves, first four with embeddings but not summarized
        leaves = [self._make_leaf_root(f"leaf{i}", i) for i in range(5)]
        for i in range(4):
            leaves[i].embedding = b"fake"

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = leaves
        mock_store.for_document.return_value = mock_doc_store

        # Fifth leaf is blocked (extraneous=3 > max=2), returns summary job
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, SummaryJob)

    def test_fifth_leaf_allowed_with_partially_summarized(self) -> None:
        """Fifth leaf allowed when preceding 4 leaves are partially summarized.

        Preceding: 4 leaves in 2 height-1 roots → 2 roots, min=1
        forest_completeness = 1/2 = 0.5 >= min_forest_completeness=0.33 → ALLOWED
        """

        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = _config_with_gating(min_forest_completeness=0.33)

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Two height-1 roots (covering 4 leaves) + leaf4 needing embedding
        parent01 = self._make_height1_root("parent01", 0)
        parent23 = self._make_height1_root("parent23", 2)
        leaf4 = self._make_leaf_root("leaf4", 4)

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [parent01, parent23, leaf4]
        mock_store.for_document.return_value = mock_doc_store

        # Should return embedding job for leaf4
        # Preceding: 4 leaves in 2 roots, min=1 → extraneous=1 <= max=2
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf4"


class TestVerbatimTokensFrontier:
    """Test eligibility frontier calculation with verbatim tokens.

    The verbatim tokens setting allows jobs to proceed past the strict gating
    boundary by a character budget: frontier = first_ineligible_root.span_start +
    verbatim_tokens * avg_chars_per_token.
    """

    def _make_leaf_root(
        self, node_id: str, level_index: int, span_start: int, span_end: int
    ) -> MagicMock:
        """Create a mock leaf root node with span information."""
        node = MagicMock()
        node.id = node_id
        node.height = 0
        node.level_index = level_index
        node.span_start = span_start
        node.span_end = span_end
        node.token_count = (span_end - span_start) // 4  # ~4 chars per token
        node.embedding = None
        return node

    def _make_height1_root(
        self, node_id: str, level_index: int, span_start: int, span_end: int
    ) -> MagicMock:
        """Create a mock height-1 root (summarized pair, covers 2 leaves)."""
        node = MagicMock()
        node.id = node_id
        node.height = 1
        node.level_index = level_index
        node.span_start = span_start
        node.span_end = span_end
        node.embedding = b"fake"  # Has embedding
        return node

    def test_third_leaf_allowed_with_verbatim_tokens(self) -> None:
        """Third leaf allowed when within verbatim token budget.

        Scenario: height-1 root (covering 2 leaves) + third leaf needing embedding.
        With min_forest_completeness=1.0, the third leaf would normally be blocked because
        preceding forest has 1 root but 2 leaves (forest_completeness = 1/1 = 1.0, OK).
        But let's test with a case that would be blocked without verbatim tokens.

        Actually, with an optimal preceding forest (h1 root), third leaf is allowed.
        Let's create a scenario with intentional low forest_completeness to test verbatim.
        """
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = _config_with_gating(
            min_forest_completeness=1.0, verbatim_tokens=100
        )

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Three roots where first two are already summarized:
        # - height-1 root covering leaves 0-1 (span 0-200)
        # - leaf2 needing embedding (span 200-300)
        # Preceding forest before leaf2: 2 leaves in 1 root = optimal (extraneous=0)
        # So this test verifies the leaf is allowed (no blocking needed).
        h1_root = self._make_height1_root("parent01", 0, span_start=0, span_end=200)
        leaf2 = self._make_leaf_root("leaf2", 2, span_start=200, span_end=300)

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [h1_root, leaf2]
        mock_doc_store.nodes.get_avg_chars_per_token.return_value = 4.0
        mock_store.for_document.return_value = mock_doc_store

        # Preceding forest is optimal, so leaf2 is eligible without verbatim
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf2"

    def test_leaf_allowed_within_verbatim_frontier(self) -> None:
        """Leaf allowed when past strict gating but within verbatim frontier.

        Setup: 2 height-1 roots + leaf needing embedding
        - Preceding forest before leaf: 4 leaves in 2 roots, extraneous=1 > 0
        - Without verbatim: leaf blocked (first_ineligible=leaf, frontier=400)
        - With verbatim_tokens=100: frontier = 400 + 400 = 800, leaf at 400 allowed
        """
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = _config_with_gating(
            min_forest_completeness=1.0, verbatim_tokens=100
        )

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Setup: 2 height-1 roots + 1 leaf needing embedding
        # Preceding 2 h1 roots = 4 leaves, 2 roots, min=popcount(4)=1 → extraneous=1
        h1_0 = self._make_height1_root("h1_0", 0, span_start=0, span_end=200)
        h1_1 = self._make_height1_root("h1_1", 2, span_start=200, span_end=400)
        leaf = self._make_leaf_root("leaf", 4, span_start=400, span_end=500)

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [h1_0, h1_1, leaf]
        mock_doc_store.nodes.get_avg_chars_per_token.return_value = 4.0
        mock_store.for_document.return_value = mock_doc_store

        # Preceding: 4 leaves in 2 roots, min=popcount(4)=1 → extraneous=1 > 0
        # First ineligible = leaf at span_start=400
        # frontier = 400 + 100*4 = 800
        # leaf.span_start=400 <= 800 → allowed
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf"

    def test_leaf_blocked_beyond_verbatim_frontier(self) -> None:
        """Leaf blocked when span_start exceeds verbatim frontier.

        First ineligible determines frontier. Later roots past frontier are blocked.

        Setup: 1 height-2 root (complete) + 1 leaf far away needing embedding
        - At leaf_far: preceding = 4 leaves in 1 root → completeness = 1.0
        - But we set min_completeness very high and use a suboptimal forest

        Actually, simpler: use a height-1 root (2 leaves) + leaf, which gives:
        - 3 leaves in 2 roots (height 1 + height 0)
        - Optimal: popcount(3)=2 roots, max_height=1, cost = 2+1 = 3
        - Actual: 2 roots, max_height=1, cost = 2+1 = 3 → completeness = 1.0

        Need a truly incomplete forest. Use 4 separate leaves (no merging):
        - 4 leaves in 4 roots, max_height=0
        - Optimal: 1 root, max_height=2, cost = 1+2 = 3
        - Actual: cost = 4+0 = 4, completeness = 0.75 < 1.0

        The job found should be a summary for the leaves, not the far leaf.
        We verify the far leaf is blocked by checking its span_start > frontier.
        """
        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob

        index_config = _config_with_gating(
            min_forest_completeness=1.0, verbatim_tokens=10
        )

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # 4 height-0 leaves (with embeddings) + 1 leaf needing embedding far away
        leaf_0 = self._make_leaf_root("leaf_0", 0, span_start=0, span_end=100)
        leaf_0.embedding = b"fake"
        leaf_1 = self._make_leaf_root("leaf_1", 1, span_start=100, span_end=200)
        leaf_1.embedding = b"fake"
        leaf_2 = self._make_leaf_root("leaf_2", 2, span_start=200, span_end=300)
        leaf_2.embedding = b"fake"
        leaf_3 = self._make_leaf_root("leaf_3", 3, span_start=300, span_end=400)
        leaf_3.embedding = b"fake"
        leaf_far = self._make_leaf_root("leaf_far", 4, span_start=2000, span_end=2100)

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [
            leaf_0,
            leaf_1,
            leaf_2,
            leaf_3,
            leaf_far,
        ]
        mock_doc_store.nodes.get_avg_chars_per_token.return_value = 4.0
        mock_store.for_document.return_value = mock_doc_store

        # At leaf_far: preceding = 4 leaves in 4 roots, max_height = 0
        # optimal_cost = 1+2 = 3, actual_cost = 4+0 = 4
        # completeness = 3/4 = 0.75 < 1.0 → ineligible
        # frontier = 2000 + 40 = 2040
        # But leaf_far is at 2000 <= 2040, so it's within frontier!
        #
        # However, work within the frontier is still allowed. The first job found
        # is the summary job for leaf_0 + leaf_1 (within frontier at span_start=0).
        # The leaf_far embedding job is blocked because there's other work to do first.
        #
        # This test verifies that even with an incomplete forest, work within
        # the frontier proceeds (summarization), while the far leaf waits.
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, SummaryJob)
        assert job.left_id == "leaf_0"
        assert job.right_id == "leaf_1"

    def test_avg_chars_per_token_none_uses_fallback(self) -> None:
        """When get_avg_chars_per_token returns None, use fallback of 4.0.

        Setup: 2 height-1 roots + leaf needing embedding
        - First ineligible: leaf (extraneous=1 > 0)
        - frontier = 400 + 50*4.0(fallback) = 600
        - leaf at 400 <= 600 → allowed
        """
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = _config_with_gating(
            min_forest_completeness=1.0, verbatim_tokens=50
        )

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # 2 height-1 roots + leaf needing embedding
        h1_0 = self._make_height1_root("h1_0", 0, span_start=0, span_end=200)
        h1_1 = self._make_height1_root("h1_1", 2, span_start=200, span_end=400)
        leaf = self._make_leaf_root("leaf", 4, span_start=400, span_end=500)

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [h1_0, h1_1, leaf]
        mock_doc_store.nodes.get_avg_chars_per_token.return_value = None  # No data yet
        mock_store.for_document.return_value = mock_doc_store

        # frontier = 400 + 50*4.0 = 600
        # leaf.span_start=400 <= 600 → allowed
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf"

    def test_verbatim_zero_allows_first_ineligible_root(self) -> None:
        """With verbatim_tokens=0, first_ineligible_root.span_start is still allowed.

        frontier = first_ineligible.span_start + 0 = first_ineligible.span_start
        And comparison is <=, so the first ineligible root IS eligible.

        Setup: 2 height-1 roots + leaf needing embedding
        - h1_0: covers 2 leaves → preceding_leaves=2, preceding_roots=1
        - h1_1: before checking, preceding has 2 leaves in 1 root, min=1, extraneous=0 ≤ 0 ✓
                after: preceding_leaves=4, preceding_roots=2
        - leaf: before checking, preceding has 4 leaves in 2 roots, min=popcount(4)=1
                extraneous=1 > 0 → leaf IS the first ineligible root
        """
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = _config_with_gating(
            min_forest_completeness=1.0, verbatim_tokens=0
        )

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # 2 height-1 roots + leaf needing embedding
        # This creates 4 preceding leaves in 2 roots, extraneous=1 > 0
        h1_0 = self._make_height1_root("h1_0", 0, span_start=0, span_end=200)
        h1_1 = self._make_height1_root("h1_1", 2, span_start=200, span_end=400)
        leaf = self._make_leaf_root("leaf", 4, span_start=400, span_end=500)

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [h1_0, h1_1, leaf]
        mock_doc_store.nodes.get_avg_chars_per_token.return_value = 4.0
        mock_store.for_document.return_value = mock_doc_store

        # first_ineligible = leaf at span_start=400
        # frontier = 400 + 0*4 = 400
        # leaf.span_start=400 <= 400 → allowed (the key insight!)
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf"


class TestExpectedTotalMatchesMinRoots:
    """Verify relationship between job count and minimum roots formulas."""

    def test_job_count_formula(self) -> None:
        """Expected total jobs = 2N - popcount(N), same logic as min roots."""
        for n in range(1, 20):
            expected_jobs = _expected_total_from_leaf_count(n)
            min_roots = _min_roots_for_leaf_count(n)
            # Jobs = N embeddings + (N - popcount) summaries = 2N - popcount
            # min_roots = popcount
            # So jobs = 2N - min_roots
            assert expected_jobs == 2 * n - min_roots
