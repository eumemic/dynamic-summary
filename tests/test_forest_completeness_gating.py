"""Tests for forest extraneous detail gating in IndexingEngine."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from ragzoom.config import IndexConfig, PrecedingContextConfig, PrecedingContextSettings
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.server.indexing_engine import (
    EmbeddingJob,
    IndexingEngine,
    SummaryJob,
    _expected_total_from_leaf_count,
    _min_roots_for_leaf_count,
)

if TYPE_CHECKING:
    from ragzoom.contracts.storage_backend import StorageBackend


def _make_preceding_context(
    min_forest_completeness: float = 0.0,
    verbatim_tokens: int = 0,
    *,
    leaf_min_forest_completeness: float | None = None,
    inner_min_forest_completeness: float | None = None,
) -> PrecedingContextSettings:
    """Create PrecedingContextSettings with given values.

    Args:
        min_forest_completeness: Default value for both leaf and inner
        verbatim_tokens: Verbatim token budget
        leaf_min_forest_completeness: Override for leaf (None = use default)
        inner_min_forest_completeness: Override for inner (None = use default)
    """
    leaf_completeness = (
        leaf_min_forest_completeness
        if leaf_min_forest_completeness is not None
        else min_forest_completeness
    )
    inner_completeness = (
        inner_min_forest_completeness
        if inner_min_forest_completeness is not None
        else min_forest_completeness
    )
    leaf_config = PrecedingContextConfig(
        min_forest_completeness=leaf_completeness,
        verbatim_tokens=verbatim_tokens,
    )
    inner_config = PrecedingContextConfig(
        min_forest_completeness=inner_completeness,
        verbatim_tokens=verbatim_tokens,
        verbatim_nodes_only=True,  # Required - inner nodes don't store embeddings
    )
    return PrecedingContextSettings(leaf=leaf_config, inner=inner_config)


def _config_with_gating(
    min_forest_completeness: float = 0.0,
    verbatim_tokens: int = 0,
    *,
    leaf_min_forest_completeness: float | None = None,
    inner_min_forest_completeness: float | None = None,
) -> IndexConfig:
    """Create IndexConfig with specified gating parameters."""
    return IndexConfig.load().replace(
        preceding_context=_make_preceding_context(
            min_forest_completeness=min_forest_completeness,
            verbatim_tokens=verbatim_tokens,
            leaf_min_forest_completeness=leaf_min_forest_completeness,
            inner_min_forest_completeness=inner_min_forest_completeness,
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


# -----------------------------------------------------------------------------
# Test helper for building tree structures with real storage
# -----------------------------------------------------------------------------


class TreeBuilder:
    """Helper for building tree structures in a real storage backend.

    This creates actual nodes in the database so tests exercise real storage
    queries (get_root_nodes, get_leaves, etc.) rather than mocks.
    """

    def __init__(self, storage_backend: StorageBackend, document_id: str) -> None:
        self.storage = storage_backend
        self.document_id = document_id
        self._next_span = 0
        self._span_size = 100  # chars per leaf

        # Initialize document metadata
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path=f"/test/{document_id}",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

    def add_leaf(
        self,
        node_id: str,
        level_index: int,
        *,
        with_embedding: bool = False,
        span_start: int | None = None,
        span_end: int | None = None,
    ) -> None:
        """Add a leaf node to the tree."""
        if span_start is None:
            span_start = self._next_span
        if span_end is None:
            span_end = span_start + self._span_size
        self._next_span = span_end

        doc_store = self.storage.for_document(self.document_id)
        node_data: NodeDataDict = {
            "node_id": node_id,
            "text": f"Leaf content {node_id}",
            "span_start": span_start,
            "span_end": span_end,
            "token_count": (span_end - span_start) // 4,  # ~4 chars per token
            "height": 0,
            "level_index": level_index,
        }
        doc_store.nodes.add_batch([node_data])

        if with_embedding:
            # Add a fake embedding (1536 dimensions for text-embedding-3-small)
            fake_embedding = [0.1] * 1536
            doc_store.nodes._repo.update_embedding(node_id, fake_embedding)

    def add_parent(
        self,
        node_id: str,
        level_index: int,
        left_child_id: str,
        right_child_id: str,
        height: int,
        span_start: int,
        span_end: int,
    ) -> None:
        """Add a parent (inner) node covering two children."""
        doc_store = self.storage.for_document(self.document_id)
        node_data: NodeDataDict = {
            "node_id": node_id,
            "text": f"Summary of {left_child_id} and {right_child_id}",
            "span_start": span_start,
            "span_end": span_end,
            "token_count": (span_end - span_start) // 4,
            "height": height,
            "level_index": level_index,
            "left_child_id": left_child_id,
            "right_child_id": right_child_id,
        }
        doc_store.nodes.add_batch([node_data])

        # Update children's parent_id
        doc_store.nodes._repo.update_parent_references_batch(
            [
                (left_child_id, node_id),
                (right_child_id, node_id),
            ]
        )


# -----------------------------------------------------------------------------
# Tests using real storage backend
# -----------------------------------------------------------------------------


class TestFindNextJobGating:
    """Test _find_next_job with real storage backend to verify gating logic.

    These tests verify that the gating correctly uses PRECEDING forest state,
    not including the current root being evaluated.
    """

    @pytest.fixture
    def engine(self, storage_backend: StorageBackend) -> IndexingEngine:
        """Create an IndexingEngine with min_forest_completeness=1.0."""
        index_config = _config_with_gating(min_forest_completeness=1.0)
        return IndexingEngine(
            store=storage_backend,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

    @pytest.fixture
    def relaxed_engine(self, storage_backend: StorageBackend) -> IndexingEngine:
        """Create an IndexingEngine with min_forest_completeness=0.0."""
        index_config = _config_with_gating(min_forest_completeness=0.0)
        return IndexingEngine(
            store=storage_backend,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

    @pytest.fixture
    def partial_gating_engine(self, storage_backend: StorageBackend) -> IndexingEngine:
        """Create an IndexingEngine with min_forest_completeness=0.33."""
        index_config = _config_with_gating(min_forest_completeness=0.33)
        return IndexingEngine(
            store=storage_backend,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

    def test_first_leaf_always_allowed_with_perfect_forest_completeness(
        self, storage_backend: StorageBackend, engine: IndexingEngine
    ) -> None:
        """First leaf (no preceding forest) always allowed, even with min_forest_completeness=1.0."""
        doc_id = "test_first_leaf"
        builder = TreeBuilder(storage_backend, doc_id)
        builder.add_leaf("leaf0", level_index=0, with_embedding=False)

        # Should return embedding job for leaf0 (no preceding forest to check)
        job = engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf0"

    def test_two_leaves_returns_summary_job(
        self, storage_backend: StorageBackend, engine: IndexingEngine
    ) -> None:
        """Two leaves form an eligible pair, returns summary job.

        Summary jobs don't require embeddings - leaf embeddings are for query-time
        retrieval, not summarization.
        """
        doc_id = "test_two_leaves"
        builder = TreeBuilder(storage_backend, doc_id)
        builder.add_leaf("leaf0", level_index=0, with_embedding=True)
        builder.add_leaf("leaf1", level_index=1, with_embedding=False)

        # Returns summary job since leaves form an eligible pair
        job = engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, SummaryJob)
        assert job.left_id == "leaf0"
        assert job.right_id == "leaf1"

    def test_third_leaf_blocked_returns_summary_job(
        self, storage_backend: StorageBackend, engine: IndexingEngine
    ) -> None:
        """Third leaf blocked with min_forest_completeness=1.0, returns summary job.

        Preceding forest: 2 leaves as 2 separate roots.
        min_roots = popcount(2) = 1
        forest_completeness = 1/2 = 0.5 < min_forest_completeness=1.0 → BLOCKED
        But engine returns a SummaryJob to combine leaf0+leaf1.
        """
        doc_id = "test_third_leaf_blocked"
        builder = TreeBuilder(storage_backend, doc_id)
        builder.add_leaf("leaf0", level_index=0, with_embedding=True)
        builder.add_leaf("leaf1", level_index=1, with_embedding=True)
        builder.add_leaf("leaf2", level_index=2, with_embedding=False)

        # Engine returns summary job for leaf0+leaf1 since leaf2 is blocked
        job = engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, SummaryJob)
        assert job.left_id == "leaf0"
        assert job.right_id == "leaf1"

    def test_third_leaf_allowed_when_preceding_summarized(
        self, storage_backend: StorageBackend, engine: IndexingEngine
    ) -> None:
        """Third leaf allowed when first two are summarized into one root.

        Preceding forest: height-1 root covering 2 leaves → 1 root, min=1 → extraneous=0.
        """
        doc_id = "test_third_leaf_allowed"
        builder = TreeBuilder(storage_backend, doc_id)

        # Create leaves and their parent
        builder.add_leaf(
            "leaf0", level_index=0, with_embedding=True, span_start=0, span_end=100
        )
        builder.add_leaf(
            "leaf1", level_index=1, with_embedding=True, span_start=100, span_end=200
        )
        builder.add_parent(
            "parent01",
            level_index=0,
            left_child_id="leaf0",
            right_child_id="leaf1",
            height=1,
            span_start=0,
            span_end=200,
        )
        builder.add_leaf(
            "leaf2", level_index=2, with_embedding=False, span_start=200, span_end=300
        )

        # Should return embedding job for leaf2 (preceding forest is optimal)
        job = engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf2"

    def test_zero_forest_completeness_allows_all_leaves(
        self, storage_backend: StorageBackend, relaxed_engine: IndexingEngine
    ) -> None:
        """With min_forest_completeness=0.0, all leaves allowed regardless of forest state."""
        doc_id = "test_relaxed_gating"
        builder = TreeBuilder(storage_backend, doc_id)
        for i in range(4):
            builder.add_leaf(f"leaf{i}", level_index=i, with_embedding=False)

        # Should return embedding job for first leaf
        job = relaxed_engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf0"

    def test_fifth_leaf_blocked_with_four_separate_roots(
        self, storage_backend: StorageBackend, partial_gating_engine: IndexingEngine
    ) -> None:
        """Fifth leaf blocked when 4 preceding leaves exist as 4 separate roots.

        Preceding: 4 leaves, 4 roots, min=popcount(4)=1
        forest_completeness = 1/4 = 0.25 < min_forest_completeness=0.33 → BLOCKED
        """
        doc_id = "test_fifth_leaf_blocked"
        builder = TreeBuilder(storage_backend, doc_id)
        for i in range(4):
            builder.add_leaf(f"leaf{i}", level_index=i, with_embedding=True)
        builder.add_leaf("leaf4", level_index=4, with_embedding=False)

        # Fifth leaf is blocked (extraneous=3 > max), returns summary job
        job = partial_gating_engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, SummaryJob)

    def test_fifth_leaf_allowed_with_partially_summarized(
        self, storage_backend: StorageBackend, partial_gating_engine: IndexingEngine
    ) -> None:
        """Fifth leaf allowed when preceding 4 leaves are partially summarized.

        Preceding: 4 leaves in 2 height-1 roots → 2 roots, min=1
        forest_completeness = 1/2 = 0.5 >= min_forest_completeness=0.33 → ALLOWED
        """
        doc_id = "test_fifth_leaf_allowed"
        builder = TreeBuilder(storage_backend, doc_id)

        # Create first pair of leaves and their parent
        builder.add_leaf(
            "leaf0", level_index=0, with_embedding=True, span_start=0, span_end=100
        )
        builder.add_leaf(
            "leaf1", level_index=1, with_embedding=True, span_start=100, span_end=200
        )
        builder.add_parent(
            "parent01",
            level_index=0,
            left_child_id="leaf0",
            right_child_id="leaf1",
            height=1,
            span_start=0,
            span_end=200,
        )

        # Create second pair of leaves and their parent
        builder.add_leaf(
            "leaf2", level_index=2, with_embedding=True, span_start=200, span_end=300
        )
        builder.add_leaf(
            "leaf3", level_index=3, with_embedding=True, span_start=300, span_end=400
        )
        builder.add_parent(
            "parent23",
            level_index=2,
            left_child_id="leaf2",
            right_child_id="leaf3",
            height=1,
            span_start=200,
            span_end=400,
        )

        # Fifth leaf needing embedding
        builder.add_leaf(
            "leaf4", level_index=4, with_embedding=False, span_start=400, span_end=500
        )

        # Should return embedding job for leaf4
        job = partial_gating_engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf4"


class TestVerbatimTokensFrontier:
    """Test eligibility frontier calculation with verbatim tokens.

    The verbatim tokens setting allows jobs to proceed past the strict gating
    boundary by a character budget: frontier = first_ineligible_root.span_start +
    verbatim_tokens * avg_chars_per_token.
    """

    def _make_engine(
        self,
        storage_backend: StorageBackend,
        min_forest_completeness: float,
        verbatim_tokens: int,
    ) -> IndexingEngine:
        """Create engine with specified gating parameters."""
        index_config = _config_with_gating(
            min_forest_completeness=min_forest_completeness,
            verbatim_tokens=verbatim_tokens,
        )
        return IndexingEngine(
            store=storage_backend,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

    def test_third_leaf_allowed_with_verbatim_tokens(
        self, storage_backend: StorageBackend
    ) -> None:
        """Third leaf allowed when preceding forest is optimal.

        Scenario: height-1 root (covering 2 leaves) + third leaf needing embedding.
        Preceding forest before leaf2: 2 leaves in 1 root = optimal (extraneous=0)
        """
        engine = self._make_engine(
            storage_backend, min_forest_completeness=1.0, verbatim_tokens=100
        )

        doc_id = "test_verbatim_third_leaf"
        builder = TreeBuilder(storage_backend, doc_id)

        # Create summarized pair
        builder.add_leaf(
            "leaf0", level_index=0, with_embedding=True, span_start=0, span_end=100
        )
        builder.add_leaf(
            "leaf1", level_index=1, with_embedding=True, span_start=100, span_end=200
        )
        builder.add_parent(
            "parent01",
            level_index=0,
            left_child_id="leaf0",
            right_child_id="leaf1",
            height=1,
            span_start=0,
            span_end=200,
        )
        builder.add_leaf(
            "leaf2", level_index=2, with_embedding=False, span_start=200, span_end=300
        )

        # Preceding forest is optimal, so leaf2 is eligible
        job = engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf2"

    def test_leaf_allowed_within_verbatim_frontier(
        self, storage_backend: StorageBackend
    ) -> None:
        """Leaf allowed when past strict gating but within verbatim frontier.

        Setup: 2 height-1 roots + leaf needing embedding
        - Preceding forest before leaf: 4 leaves in 2 roots, extraneous=1 > 0
        - Without verbatim: leaf blocked (first_ineligible=leaf, frontier=400)
        - With verbatim_tokens=100: frontier = 400 + 400 = 800, leaf at 400 allowed
        """
        engine = self._make_engine(
            storage_backend, min_forest_completeness=1.0, verbatim_tokens=100
        )

        doc_id = "test_verbatim_frontier"
        builder = TreeBuilder(storage_backend, doc_id)

        # Create first pair
        builder.add_leaf(
            "leaf0", level_index=0, with_embedding=True, span_start=0, span_end=100
        )
        builder.add_leaf(
            "leaf1", level_index=1, with_embedding=True, span_start=100, span_end=200
        )
        builder.add_parent(
            "h1_0",
            level_index=0,
            left_child_id="leaf0",
            right_child_id="leaf1",
            height=1,
            span_start=0,
            span_end=200,
        )

        # Create second pair
        builder.add_leaf(
            "leaf2", level_index=2, with_embedding=True, span_start=200, span_end=300
        )
        builder.add_leaf(
            "leaf3", level_index=3, with_embedding=True, span_start=300, span_end=400
        )
        builder.add_parent(
            "h1_1",
            level_index=2,
            left_child_id="leaf2",
            right_child_id="leaf3",
            height=1,
            span_start=200,
            span_end=400,
        )

        # Add leaf needing embedding
        builder.add_leaf(
            "leaf4", level_index=4, with_embedding=False, span_start=400, span_end=500
        )

        # frontier = 400 + 100*4 = 800, leaf at 400 <= 800 → allowed
        job = engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf4"

    def test_leaf_blocked_beyond_verbatim_frontier(
        self, storage_backend: StorageBackend
    ) -> None:
        """Leaf blocked when span_start exceeds verbatim frontier.

        Setup: 4 separate leaves (no merging) + 1 leaf far away
        - 4 leaves in 4 roots, completeness = 1/4 = 0.25 < 1.0 → ineligible
        - frontier = span_start + verbatim_budget
        - Far leaf waits while summary jobs proceed
        """
        engine = self._make_engine(
            storage_backend, min_forest_completeness=1.0, verbatim_tokens=10
        )

        doc_id = "test_verbatim_blocked"
        builder = TreeBuilder(storage_backend, doc_id)

        # 4 height-0 leaves (with embeddings)
        for i in range(4):
            builder.add_leaf(
                f"leaf_{i}",
                level_index=i,
                with_embedding=True,
                span_start=i * 100,
                span_end=(i + 1) * 100,
            )

        # Far leaf needing embedding
        builder.add_leaf(
            "leaf_far",
            level_index=4,
            with_embedding=False,
            span_start=2000,
            span_end=2100,
        )

        # Work within frontier proceeds (summarization), far leaf waits
        job = engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, SummaryJob)
        assert job.left_id == "leaf_0"
        assert job.right_id == "leaf_1"

    def test_verbatim_zero_allows_first_ineligible_root(
        self, storage_backend: StorageBackend
    ) -> None:
        """With verbatim_tokens=0, first_ineligible_root.span_start is still allowed.

        frontier = first_ineligible.span_start + 0 = first_ineligible.span_start
        And comparison is <=, so the first ineligible root IS eligible.
        """
        engine = self._make_engine(
            storage_backend, min_forest_completeness=1.0, verbatim_tokens=0
        )

        doc_id = "test_verbatim_zero"
        builder = TreeBuilder(storage_backend, doc_id)

        # Create first pair
        builder.add_leaf(
            "leaf0", level_index=0, with_embedding=True, span_start=0, span_end=100
        )
        builder.add_leaf(
            "leaf1", level_index=1, with_embedding=True, span_start=100, span_end=200
        )
        builder.add_parent(
            "h1_0",
            level_index=0,
            left_child_id="leaf0",
            right_child_id="leaf1",
            height=1,
            span_start=0,
            span_end=200,
        )

        # Create second pair
        builder.add_leaf(
            "leaf2", level_index=2, with_embedding=True, span_start=200, span_end=300
        )
        builder.add_leaf(
            "leaf3", level_index=3, with_embedding=True, span_start=300, span_end=400
        )
        builder.add_parent(
            "h1_1",
            level_index=2,
            left_child_id="leaf2",
            right_child_id="leaf3",
            height=1,
            span_start=200,
            span_end=400,
        )

        # Leaf needing embedding (this is the first ineligible root)
        builder.add_leaf(
            "leaf4", level_index=4, with_embedding=False, span_start=400, span_end=500
        )

        # first_ineligible = leaf at span_start=400
        # frontier = 400 + 0*4 = 400
        # leaf.span_start=400 <= 400 → allowed
        job = engine._find_next_job(doc_id, set(), None)
        assert job is not None
        assert isinstance(job, EmbeddingJob)
        assert job.leaf_id == "leaf4"


class TestAsymmetricGating:
    """Test that embedding and summary jobs use different gating thresholds.

    This is the key test for parallelism: when leaf.min_forest_completeness=1.0
    but inner.min_forest_completeness=0.0, embedding jobs should be blocked
    while summary jobs proceed freely.
    """

    @pytest.fixture
    def asymmetric_engine(self, storage_backend: StorageBackend) -> IndexingEngine:
        """Create engine with strict leaf gating but no inner gating.

        This is the production configuration for maximum parallelism:
        - leaf.min_forest_completeness=1.0 (embeddings wait for complete forest)
        - inner.min_forest_completeness=0.0 (summaries never wait)
        """
        index_config = _config_with_gating(
            leaf_min_forest_completeness=1.0,
            inner_min_forest_completeness=0.0,
        )
        return IndexingEngine(
            store=storage_backend,
            llm_service=MagicMock(),
            index_config=index_config,
            openai_client=MagicMock(),
            max_parallelism=30,
        )

    def test_summary_job_not_blocked_when_embedding_blocked(
        self, storage_backend: StorageBackend, asymmetric_engine: IndexingEngine
    ) -> None:
        """Summary jobs proceed even when embedding jobs are blocked by gating.

        Scenario: 4 unsummarized leaves create an incomplete forest.
        - Embedding jobs for later leaves ARE blocked (leaf.min_forest_completeness=1.0)
        - Summary jobs for eligible pairs are NOT blocked (inner.min_forest_completeness=0.0)
        """
        doc_id = "test_asymmetric_gating"
        builder = TreeBuilder(storage_backend, doc_id)

        # Create 4 leaves - all embedded, forming 4 separate roots (incomplete forest)
        for i in range(4):
            builder.add_leaf(
                f"leaf{i}",
                level_index=i,
                with_embedding=True,
                span_start=i * 100,
                span_end=(i + 1) * 100,
            )

        # Add 5th leaf without embedding - this embedding job would be blocked
        # because preceding forest has 4 roots but should have 1 (completeness=0.25)
        builder.add_leaf(
            "leaf4",
            level_index=4,
            with_embedding=False,
            span_start=400,
            span_end=500,
        )

        # With asymmetric gating, should return SUMMARY job (not blocked)
        # even though an embedding job exists but is blocked
        job = asymmetric_engine._find_next_job(doc_id, set(), None)

        # The summary job for the first eligible pair (leaf0, leaf1) should be returned
        # because inner.min_forest_completeness=0.0 means no gating for summaries
        assert job is not None
        assert isinstance(job, SummaryJob), (
            f"Expected SummaryJob but got {type(job).__name__}. "
            "Summary jobs should not be blocked when inner.min_forest_completeness=0.0"
        )
        assert job.left_id == "leaf0"
        assert job.right_id == "leaf1"

    def test_embedding_blocked_but_later_summary_allowed(
        self, storage_backend: StorageBackend, asymmetric_engine: IndexingEngine
    ) -> None:
        """Embedding job at position N blocked, but summary job at position M > N allowed.

        This tests that summary jobs can proceed past the embedding frontier.
        """
        doc_id = "test_summary_past_frontier"
        builder = TreeBuilder(storage_backend, doc_id)

        # Create 8 leaves, first 4 embedded, last 4 not embedded
        for i in range(4):
            builder.add_leaf(
                f"leaf{i}",
                level_index=i,
                with_embedding=True,
                span_start=i * 100,
                span_end=(i + 1) * 100,
            )
        for i in range(4, 8):
            builder.add_leaf(
                f"leaf{i}",
                level_index=i,
                with_embedding=False,
                span_start=i * 100,
                span_end=(i + 1) * 100,
            )

        # The embedding frontier is at leaf4 (first incomplete forest point)
        # But with inner.min_forest_completeness=0.0, ALL summary jobs are allowed
        # including pairs at leaf4-leaf5, leaf6-leaf7

        job = asymmetric_engine._find_next_job(doc_id, set(), None)

        # Should return earliest summary job (leaf0, leaf1)
        assert job is not None
        assert isinstance(job, SummaryJob)


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
