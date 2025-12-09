"""Tests for ScoringService using pre-indexed embeddings on nodes."""

import struct
from unittest.mock import Mock

import pytest

from ragzoom.retrieval.scoring_service import ScoringService

# Type alias for candidates list
CandidateList = list[tuple[str, float, dict[str, str | int | float | bool | None]]]


def pack_embedding(vec: list[float]) -> bytes:
    """Pack a list of floats into bytes (float32 format)."""
    return struct.pack(f"{len(vec)}f", *vec)


@pytest.fixture
def mock_store() -> Mock:
    """Create a mock DocumentStore."""
    return Mock()


@pytest.fixture
def mock_vector_index() -> Mock:
    """Create a mock VectorIndex."""
    return Mock()


@pytest.fixture
def scoring_service(mock_store: Mock, mock_vector_index: Mock) -> ScoringService:
    """Create a ScoringService with mock dependencies."""
    return ScoringService(mock_store, mock_vector_index)


def make_node(
    node_id: str,
    height: int = 0,
    parent_id: str | None = None,
    left_child_id: str | None = None,
    right_child_id: str | None = None,
    embedding: bytes | None = None,
) -> Mock:
    """Create a mock TreeNode with optional embedding."""
    node = Mock()
    node.id = node_id
    node.height = height
    node.parent_id = parent_id
    node.left_child_id = left_child_id
    node.right_child_id = right_child_id
    node.embedding = embedding
    return node


class TestComputeScoresWithEmbeddings:
    """Tests for compute_scores using pre-indexed node embeddings."""

    def test_node_with_embedding_gets_similarity_score(
        self, scoring_service: ScoringService
    ) -> None:
        """Node with embedding gets dot product similarity with query."""
        # Create embedding aligned with query for high score
        embedding = [0.6, 0.8]
        nodes = {
            "node1": make_node("node1", embedding=pack_embedding(embedding)),
        }
        coverage_map = {"node1": True}
        candidates: CandidateList = []
        query = [0.6, 0.8]  # Same direction

        scores = scoring_service.compute_scores(query, coverage_map, candidates, nodes)

        # Normalized vectors pointing same direction -> similarity ~1.0
        assert scores["node1"] == pytest.approx(1.0, abs=0.01)

    def test_node_without_embedding_gets_zero(
        self, scoring_service: ScoringService
    ) -> None:
        """Node without embedding (not yet indexed) gets 0.0."""
        nodes = {
            "node1": make_node("node1", embedding=None),
        }
        coverage_map = {"node1": True}
        candidates: CandidateList = []

        scores = scoring_service.compute_scores(
            [0.6, 0.8], coverage_map, candidates, nodes
        )

        assert scores["node1"] == 0.0

    def test_orthogonal_embedding_gets_zero_score(
        self, scoring_service: ScoringService
    ) -> None:
        """Embedding orthogonal to query gets 0.0 similarity."""
        # Query along x-axis, embedding along y-axis
        embedding = [0.0, 1.0]
        nodes = {
            "node1": make_node("node1", embedding=pack_embedding(embedding)),
        }
        coverage_map = {"node1": True}
        candidates: CandidateList = []
        query = [1.0, 0.0]

        scores = scoring_service.compute_scores(query, coverage_map, candidates, nodes)

        assert scores["node1"] == pytest.approx(0.0, abs=0.01)

    def test_multiple_nodes_scored_independently(
        self, scoring_service: ScoringService
    ) -> None:
        """Each node gets its own similarity score."""
        # High relevance embedding
        high_emb = [0.6, 0.8]
        # Low relevance embedding (orthogonal-ish)
        low_emb = [0.8, -0.6]

        nodes = {
            "high": make_node("high", embedding=pack_embedding(high_emb)),
            "low": make_node("low", embedding=pack_embedding(low_emb)),
        }
        coverage_map = {"high": True, "low": True}
        candidates: CandidateList = []
        query = [0.6, 0.8]

        scores = scoring_service.compute_scores(query, coverage_map, candidates, nodes)

        assert scores["high"] > 0.9  # High similarity
        assert scores["low"] < 0.2  # Low similarity

    def test_inner_nodes_use_their_own_embeddings(
        self, scoring_service: ScoringService
    ) -> None:
        """Inner nodes get scored using their pre-computed embeddings."""
        # Parent embedding (would be avg of children, but we test it directly)
        parent_emb = [0.7, 0.7]  # Normalized: [0.707, 0.707]
        child_emb = [0.6, 0.8]

        nodes = {
            "child": make_node(
                "child",
                height=0,
                parent_id="parent",
                embedding=pack_embedding(child_emb),
            ),
            "parent": make_node(
                "parent",
                height=1,
                left_child_id="child",
                embedding=pack_embedding(parent_emb),
            ),
        }
        coverage_map = {"child": True, "parent": True}
        candidates: CandidateList = []
        query = [0.7, 0.7]

        scores = scoring_service.compute_scores(query, coverage_map, candidates, nodes)

        # Parent scored using its own embedding, not propagated from child
        assert "parent" in scores
        assert scores["parent"] == pytest.approx(1.0, abs=0.01)

    def test_mixed_embedded_and_unembedded_nodes(
        self, scoring_service: ScoringService
    ) -> None:
        """Tree with some embedded and some unembedded nodes."""
        emb = [0.6, 0.8]
        nodes = {
            "embedded": make_node("embedded", embedding=pack_embedding(emb)),
            "not_embedded": make_node("not_embedded", embedding=None),
        }
        coverage_map = {"embedded": True, "not_embedded": True}
        candidates: CandidateList = []
        query = [0.6, 0.8]

        scores = scoring_service.compute_scores(query, coverage_map, candidates, nodes)

        assert scores["embedded"] > 0.9
        assert scores["not_embedded"] == 0.0


class TestComputeScoresFallback:
    """Tests for fallback behavior when nodes are not provided."""

    def test_without_nodes_uses_candidate_scores(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Without nodes dict, falls back to candidate scores only."""
        service = ScoringService(mock_store, mock_vector_index)

        coverage_map = {"seed1": True, "seed2": True, "other": True}
        candidates: CandidateList = [("seed1", 0.75, {}), ("seed2", 0.5, {})]

        scores = service.compute_scores(
            [0.1, 0.2], coverage_map, candidates, nodes=None
        )

        assert scores == {"seed1": 0.75, "seed2": 0.5}

    def test_candidate_not_in_coverage_excluded(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Candidates not in coverage are excluded from scores."""
        service = ScoringService(mock_store, mock_vector_index)

        coverage_map = {"in_coverage": True}
        candidates: CandidateList = [
            ("in_coverage", 0.8, {}),
            ("not_in_coverage", 0.9, {}),
        ]

        scores = service.compute_scores(
            [0.1, 0.2], coverage_map, candidates, nodes=None
        )

        assert "in_coverage" in scores
        assert "not_in_coverage" not in scores


class TestComputeScoresNormalization:
    """Tests for embedding normalization during scoring."""

    def test_unnormalized_embedding_normalized_before_scoring(
        self, scoring_service: ScoringService
    ) -> None:
        """Embeddings are normalized before computing similarity."""
        # Unnormalized embedding (magnitude != 1)
        unnorm_emb = [3.0, 4.0]  # Magnitude 5
        nodes = {
            "node1": make_node("node1", embedding=pack_embedding(unnorm_emb)),
        }
        coverage_map = {"node1": True}
        candidates: CandidateList = []
        query = [0.6, 0.8]  # Same direction as [3,4] normalized

        scores = scoring_service.compute_scores(query, coverage_map, candidates, nodes)

        # After normalization, [3,4] -> [0.6, 0.8], same as query
        assert scores["node1"] == pytest.approx(1.0, abs=0.01)

    def test_averaged_embedding_renormalized(
        self, scoring_service: ScoringService
    ) -> None:
        """Parent embeddings (averages) are renormalized correctly."""
        # Simulate a parent embedding that's an average (may not be normalized)
        # avg([1,0], [0,1]) = [0.5, 0.5], magnitude = 0.707
        avg_emb = [0.5, 0.5]
        nodes = {
            "parent": make_node("parent", height=1, embedding=pack_embedding(avg_emb)),
        }
        coverage_map = {"parent": True}
        candidates: CandidateList = []
        query = [0.707, 0.707]  # Same direction as [0.5, 0.5] normalized

        scores = scoring_service.compute_scores(query, coverage_map, candidates, nodes)

        assert scores["parent"] == pytest.approx(1.0, abs=0.01)
