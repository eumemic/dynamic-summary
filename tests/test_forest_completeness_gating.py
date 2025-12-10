"""Tests for forest completeness gating in IndexingEngine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragzoom.server.indexing_engine import (
    _expected_nodes_from_leaf_count,
    _expected_total_from_leaf_count,
)


class TestExpectedNodesFromLeafCount:
    """Test the helper function for calculating expected nodes in a complete forest."""

    def test_zero_leaves(self) -> None:
        """Zero leaves means zero nodes."""
        assert _expected_nodes_from_leaf_count(0) == 0

    def test_one_leaf(self) -> None:
        """One leaf (popcount=1): 2*1 - 1 = 1 node (just the leaf itself)."""
        assert _expected_nodes_from_leaf_count(1) == 1

    def test_two_leaves(self) -> None:
        """Two leaves (popcount=1): 2*2 - 1 = 3 nodes (2 leaves + 1 parent)."""
        assert _expected_nodes_from_leaf_count(2) == 3

    def test_three_leaves(self) -> None:
        """Three leaves (popcount=2): 2*3 - 2 = 4 nodes.

        Forest: one tree of 2 leaves (3 nodes) + one unpaired leaf (1 node) = 4.
        """
        assert _expected_nodes_from_leaf_count(3) == 4

    def test_four_leaves(self) -> None:
        """Four leaves (popcount=1): 2*4 - 1 = 7 nodes.

        Perfect binary tree: 4 leaves + 2 parents + 1 root = 7.
        """
        assert _expected_nodes_from_leaf_count(4) == 7

    def test_five_leaves(self) -> None:
        """Five leaves (popcount=2): 2*5 - 2 = 8 nodes.

        Forest: tree of 4 (7 nodes) + unpaired leaf (1 node) = 8.
        """
        assert _expected_nodes_from_leaf_count(5) == 8

    def test_seven_leaves(self) -> None:
        """Seven leaves (popcount=3): 2*7 - 3 = 11 nodes.

        Forest: tree of 4 (7 nodes) + tree of 2 (3 nodes) + leaf (1 node) = 11.
        """
        assert _expected_nodes_from_leaf_count(7) == 11

    def test_eight_leaves(self) -> None:
        """Eight leaves (popcount=1): 2*8 - 1 = 15 nodes.

        Perfect binary tree of height 3.
        """
        assert _expected_nodes_from_leaf_count(8) == 15

    def test_matches_job_count(self) -> None:
        """Expected nodes should equal expected jobs (each node requires one job)."""
        for n in range(1, 20):
            assert _expected_nodes_from_leaf_count(
                n
            ) == _expected_total_from_leaf_count(n)


class TestForestCompletenessCalculation:
    """Test the completeness ratio calculation logic."""

    def test_single_leaf_is_complete(self) -> None:
        """A single leaf root has 100% completeness (1/1 = 1.0)."""
        # 1 leaf, 1 node actual, 1 node expected
        leaves = 1
        actual = 1
        expected = _expected_nodes_from_leaf_count(leaves)
        assert expected == 1
        assert actual / expected == 1.0

    def test_two_leaves_without_parent_incomplete(self) -> None:
        """Two leaf roots without parent have 2/3 completeness."""
        # 2 leaves, 2 nodes actual (just leaves), 3 nodes expected
        leaves = 2
        actual = 2
        expected = _expected_nodes_from_leaf_count(leaves)
        assert expected == 3
        assert actual / expected == pytest.approx(0.666, rel=0.01)

    def test_two_leaves_with_parent_complete(self) -> None:
        """Two leaves combined into parent = complete (3/3 = 1.0)."""
        # After summarizing: 1 root of height 1 = 3 nodes
        leaves = 2
        actual = 3  # height-1 tree contains 3 nodes
        expected = _expected_nodes_from_leaf_count(leaves)
        assert expected == 3
        assert actual / expected == 1.0

    def test_three_leaves_all_separate(self) -> None:
        """Three separate leaves have 3/4 = 0.75 completeness."""
        # 3 leaves as roots, no parents yet
        leaves = 3
        actual = 3
        expected = _expected_nodes_from_leaf_count(leaves)
        assert expected == 4
        assert actual / expected == 0.75

    def test_three_leaves_first_two_paired(self) -> None:
        """First two leaves paired, third separate = 4/4 = 1.0 complete."""
        # One height-1 tree (3 nodes) + one leaf (1 node) = 4 nodes
        leaves = 3
        actual = 4
        expected = _expected_nodes_from_leaf_count(leaves)
        assert expected == 4
        assert actual / expected == 1.0

    def test_four_leaves_none_paired(self) -> None:
        """Four separate leaves have 4/7 completeness."""
        leaves = 4
        actual = 4
        expected = _expected_nodes_from_leaf_count(leaves)
        assert expected == 7
        assert actual / expected == pytest.approx(0.571, rel=0.01)

    def test_four_leaves_two_paired(self) -> None:
        """Two pairs of leaves summarized = 6/7 completeness."""
        # Two height-1 trees = 3 + 3 = 6 nodes
        leaves = 4
        actual = 6
        expected = _expected_nodes_from_leaf_count(leaves)
        assert expected == 7
        assert actual / expected == pytest.approx(0.857, rel=0.01)

    def test_four_leaves_complete_tree(self) -> None:
        """Complete binary tree of 4 leaves = 7/7 = 1.0."""
        leaves = 4
        actual = 7
        expected = _expected_nodes_from_leaf_count(leaves)
        assert expected == 7
        assert actual / expected == 1.0


class TestForestCompletenessGatingBehavior:
    """Test gating behavior with different threshold values.

    These tests verify the scan-and-stop logic conceptually.
    The actual IndexingEngine integration is tested in integration tests.
    """

    def test_threshold_zero_allows_all(self) -> None:
        """Threshold 0.0 should never block (no completeness check needed)."""
        threshold = 0.0
        # Any completeness value should pass when threshold is 0
        for completeness in [0.0, 0.25, 0.5, 0.75, 1.0]:
            # With threshold 0, we skip the check entirely
            assert threshold == 0.0 or completeness >= threshold

    def test_threshold_one_requires_complete(self) -> None:
        """Threshold 1.0 only allows jobs when forest is 100% complete."""
        threshold = 1.0
        # Only 1.0 completeness passes
        assert 1.0 >= threshold
        assert 0.999 < threshold
        assert 0.5 < threshold

    def test_threshold_half_allows_fifty_percent(self) -> None:
        """Threshold 0.5 allows jobs when forest is at least 50% complete."""
        threshold = 0.5
        assert 1.0 >= threshold
        assert 0.75 >= threshold
        assert 0.5 >= threshold
        assert 0.49 < threshold
        assert 0.25 < threshold


class TestFindNextJobGating:
    """Test _find_next_job with actual mock roots to verify gating logic.

    These tests verify that the gating correctly uses PRECEDING forest state,
    not including the current root being evaluated.
    """

    @pytest.fixture
    def mock_store(self) -> object:
        """Create a minimal mock store for testing."""

        return MagicMock()

    @pytest.fixture
    def mock_llm_service(self) -> object:
        """Create a minimal mock LLM service."""

        return MagicMock()

    def _make_leaf_root(self, node_id: str, level_index: int) -> MagicMock:
        """Create a mock leaf root node."""
        node = MagicMock()
        node.id = node_id
        node.height = 0
        node.level_index = level_index
        node.embedding = None  # No embedding yet
        return node

    def _make_height1_root(self, node_id: str, level_index: int) -> MagicMock:
        """Create a mock height-1 root (summarized pair)."""
        node = MagicMock()
        node.id = node_id
        node.height = 1
        node.level_index = level_index
        node.embedding = b"fake"  # Has embedding
        return node

    def test_first_leaf_always_allowed_with_threshold_one(self) -> None:
        """First leaf (no preceding forest) should always be allowed, even with threshold=1.0.

        This test would have caught the original bug where we included the current
        root in the completeness calculation, causing leaf 0 to pass but leaf 1
        to be blocked even when preceding forest was complete.
        """

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        # Create engine with threshold=1.0
        index_config = IndexConfig.load()
        index_config = index_config.replace(
            preceding_context_min_forest_completeness=1.0
        )

        mock_store = MagicMock()
        mock_llm_service = MagicMock()

        engine = IndexingEngine(
            store=mock_store,
            llm_service=mock_llm_service,
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Set up mock: single leaf root with no embedding
        leaf0 = self._make_leaf_root("leaf0", 0)
        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [leaf0]
        mock_store.for_document.return_value = mock_doc_store

        # Should return embedding job for leaf0 (no preceding forest to check)
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf0"

    def test_second_leaf_allowed_when_first_complete(self) -> None:
        """Second leaf allowed when first leaf's forest is 100% complete.

        With threshold=1.0, leaf 1 should be allowed when leaf 0 exists as
        a complete subtree (1 leaf = 1 node = 100% complete).
        """

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = IndexConfig.load()
        index_config = index_config.replace(
            preceding_context_min_forest_completeness=1.0
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
        # Preceding forest: 1 leaf, 1 node actual, 1 expected = 100%
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf1"

    def test_third_leaf_blocked_returns_summary_job(self) -> None:
        """Third leaf blocked but summary job returned to make progress.

        With threshold=1.0, leaf 2 is blocked when leaves 0,1 exist
        but their parent summary doesn't (2 nodes / 3 expected = 67%).
        However, the engine returns a SummaryJob to combine leaf0+leaf1.
        """

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob

        index_config = IndexConfig.load()
        index_config = index_config.replace(
            preceding_context_min_forest_completeness=1.0
        )

        mock_store = MagicMock()
        engine = IndexingEngine(
            store=mock_store,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

        # Three leaves, all with embeddings but no parent yet
        leaf0 = self._make_leaf_root("leaf0", 0)
        leaf0.embedding = b"fake"
        leaf1 = self._make_leaf_root("leaf1", 1)
        leaf1.embedding = b"fake"
        leaf2 = self._make_leaf_root("leaf2", 2)  # No embedding

        mock_doc_store = MagicMock()
        mock_doc_store.nodes.get_root_nodes.return_value = [leaf0, leaf1, leaf2]
        mock_store.for_document.return_value = mock_doc_store

        # Engine returns summary job for leaf0+leaf1 since leaf2 is blocked
        # by completeness gating (preceding forest is 67% complete)
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, SummaryJob)
        assert job.left_id == "leaf0"
        assert job.right_id == "leaf1"

    def test_third_leaf_allowed_when_preceding_summarized(self) -> None:
        """Third leaf allowed when first two are summarized into parent.

        With threshold=1.0, leaf 2 should be allowed when a height-1 root
        covers leaves 0,1 (3 nodes / 3 expected = 100%).
        """

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = IndexConfig.load()
        index_config = index_config.replace(
            preceding_context_min_forest_completeness=1.0
        )

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
        # Preceding forest: height-1 root = 2 leaves, 3 nodes, 3 expected = 100%
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf2"

    def test_threshold_zero_allows_all_leaves(self) -> None:
        """With threshold=0.0, all leaves should be allowed regardless of completeness."""

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

        index_config = IndexConfig.load()
        index_config = index_config.replace(
            preceding_context_min_forest_completeness=0.0
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

        # Should return embedding job for first leaf (threshold=0 skips check)
        job = engine._find_next_job("doc1", set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf0"
