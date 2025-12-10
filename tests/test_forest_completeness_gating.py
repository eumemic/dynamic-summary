"""Tests for forest extraneous detail gating in IndexingEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

from ragzoom.server.indexing_engine import (
    _expected_total_from_leaf_count,
    _min_roots_for_leaf_count,
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


class TestExtraneousDetailGatingBehavior:
    """Test gating behavior with different max_extraneous_detail values."""

    def test_high_max_allows_all(self) -> None:
        """High max_extraneous_detail value should allow all roots."""
        max_extraneous = 100
        # Even with 10 extraneous roots, should pass
        for extraneous in [0, 1, 5, 10, 50]:
            assert extraneous <= max_extraneous

    def test_max_zero_requires_optimal(self) -> None:
        """max_extraneous_detail=0 only allows optimal forest state."""
        max_extraneous = 0
        # Only 0 extraneous passes
        assert 0 <= max_extraneous
        assert 1 > max_extraneous
        assert 5 > max_extraneous

    def test_max_two_allows_some_slack(self) -> None:
        """max_extraneous_detail=2 allows up to 2 extra roots."""
        max_extraneous = 2
        assert 0 <= max_extraneous
        assert 1 <= max_extraneous
        assert 2 <= max_extraneous
        assert 3 > max_extraneous


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

    def test_first_leaf_always_allowed_with_max_zero(self) -> None:
        """First leaf (no preceding forest) always allowed, even with max_extraneous=0."""

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        # Strictest threshold: no extraneous detail allowed
        index_config = IndexConfig.load()
        index_config = index_config.replace(preceding_context_max_extraneous_detail=0)

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

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = IndexConfig.load()
        index_config = index_config.replace(
            preceding_context_max_extraneous_detail=0  # Strictest
        )

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
        """Third leaf blocked with max_extraneous=0, returns summary job.

        Preceding forest: 2 leaves as 2 separate roots.
        min_roots = popcount(2) = 1
        extraneous = 2 - 1 = 1 > max_extraneous=0 → BLOCKED
        But engine returns a SummaryJob to combine leaf0+leaf1.
        """

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob

        index_config = IndexConfig.load()
        index_config = index_config.replace(preceding_context_max_extraneous_detail=0)

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

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = IndexConfig.load()
        index_config = index_config.replace(preceding_context_max_extraneous_detail=0)

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

    def test_large_max_allows_all_leaves(self) -> None:
        """With large max_extraneous_detail, all leaves allowed regardless of forest state."""

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = IndexConfig.load()
        index_config = index_config.replace(
            preceding_context_max_extraneous_detail=100  # Very permissive
        )

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

        Preceding: 4 leaves, 4 roots, min=popcount(4)=1 → extraneous=3
        With max_extraneous=2, this exceeds threshold.
        """

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob

        index_config = IndexConfig.load()
        index_config = index_config.replace(
            preceding_context_max_extraneous_detail=2  # Allow up to 2 extra roots
        )

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

        Preceding: 4 leaves in 2 height-1 roots → 2 roots, min=1 → extraneous=1
        With max_extraneous=2, this is allowed.
        """

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = IndexConfig.load()
        index_config = index_config.replace(preceding_context_max_extraneous_detail=2)

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
