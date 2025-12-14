"""Tests for ScoringService with bottom-up score propagation."""

from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np
import pytest

from ragzoom.retrieval.scoring_service import ScoringService

# Type alias for candidates list
CandidateList = list[tuple[str, float, dict[str, str | int | float | bool | None]]]


@dataclass
class MockVector:
    """Mock vector returned from vector index."""

    id: str
    vec: np.ndarray[tuple[int], np.dtype[np.float64]]


def make_node(
    node_id: str,
    height: int = 0,
    parent_id: str | None = None,
    left_child_id: str | None = None,
    right_child_id: str | None = None,
) -> Mock:
    """Create a mock TreeNode."""
    node = Mock()
    node.id = node_id
    node.height = height
    node.parent_id = parent_id
    node.left_child_id = left_child_id
    node.right_child_id = right_child_id
    return node


@pytest.fixture
def mock_store() -> Mock:
    """Create a mock DocumentStore."""
    return Mock()


@pytest.fixture
def mock_vector_index() -> Mock:
    """Create a mock VectorIndex."""
    index = Mock()
    index.get_vectors.return_value = []
    return index


@pytest.fixture
def scoring_service(mock_store: Mock, mock_vector_index: Mock) -> ScoringService:
    """Create a ScoringService with mock dependencies."""
    return ScoringService(mock_store, mock_vector_index)


class TestSeedScoring:
    """Tests for seed node scoring from candidates."""

    def test_seed_gets_precomputed_score(self, scoring_service: ScoringService) -> None:
        """Seeds in candidates get their pre-computed scores."""
        coverage_map = {"seed1": True, "seed2": True}
        candidates: CandidateList = [
            ("seed1", 0.9, {}),
            ("seed2", 0.7, {}),
        ]
        query = [0.6, 0.8]

        scores = scoring_service.compute_scores(
            query, coverage_map, candidates, nodes=None
        )

        assert scores["seed1"] == 0.9
        assert scores["seed2"] == 0.7

    def test_candidate_not_in_coverage_excluded(
        self, scoring_service: ScoringService
    ) -> None:
        """Candidates not in coverage are excluded from scores."""
        coverage_map = {"in_coverage": True}
        candidates: CandidateList = [
            ("in_coverage", 0.8, {}),
            ("not_in_coverage", 0.9, {}),
        ]
        query = [0.6, 0.8]

        scores = scoring_service.compute_scores(
            query, coverage_map, candidates, nodes=None
        )

        assert "in_coverage" in scores
        assert "not_in_coverage" not in scores


class TestSiblingScoring:
    """Tests for seed sibling scoring via vector index."""

    def test_sibling_scores_fetched_from_vector_index(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Siblings of seeds get scores via vector index lookup."""
        # Build tree: parent has seed (left) and sibling (right)
        seed = make_node("seed", height=0, parent_id="parent")
        sibling = make_node("sibling", height=0, parent_id="parent")
        parent = make_node(
            "parent",
            height=1,
            left_child_id="seed",
            right_child_id="sibling",
        )
        nodes = {"seed": seed, "sibling": sibling, "parent": parent}
        coverage_map = {"seed": True, "sibling": True, "parent": True}

        # Seed score from candidates
        candidates: CandidateList = [("seed", 0.9, {})]

        # Sibling score from vector index
        # Query [0.6, 0.8] dot sibling [0.6, 0.8] = 1.0
        sibling_vec = np.array([0.6, 0.8], dtype=np.float64)
        mock_vector_index.get_vectors.return_value = [
            MockVector(id="sibling", vec=sibling_vec)
        ]

        service = ScoringService(mock_store, mock_vector_index)
        query = [0.6, 0.8]

        scores = service.compute_scores(query, coverage_map, candidates, nodes)

        # Seed gets candidate score
        assert scores["seed"] == 0.9
        # Sibling gets vector-based score
        assert scores["sibling"] == pytest.approx(1.0, abs=0.01)

    def test_sibling_not_in_coverage_not_scored(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Siblings not in coverage are not fetched or scored."""
        # Build tree: parent has seed (left) and sibling (right)
        # But sibling is not in coverage
        seed = make_node("seed", height=0, parent_id="parent")
        sibling = make_node("sibling", height=0, parent_id="parent")
        parent = make_node(
            "parent",
            height=1,
            left_child_id="seed",
            right_child_id="sibling",
        )
        nodes = {"seed": seed, "sibling": sibling, "parent": parent}
        coverage_map = {"seed": True, "parent": True}  # sibling NOT in coverage

        candidates: CandidateList = [("seed", 0.9, {})]
        mock_vector_index.get_vectors.return_value = []

        service = ScoringService(mock_store, mock_vector_index)
        query = [0.6, 0.8]

        scores = service.compute_scores(query, coverage_map, candidates, nodes)

        assert "sibling" not in scores
        # Vector index should not have been called for sibling
        # (or called with empty list)


class TestBottomUpPropagation:
    """Tests for bottom-up score propagation to inner nodes."""

    def test_parent_gets_average_of_children(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Parent score is average of children scores."""
        # Build tree: parent with two leaf children
        left = make_node("left", height=0, parent_id="parent")
        right = make_node("right", height=0, parent_id="parent")
        parent = make_node(
            "parent",
            height=1,
            left_child_id="left",
            right_child_id="right",
        )
        nodes = {"left": left, "right": right, "parent": parent}
        coverage_map = {"left": True, "right": True, "parent": True}

        # Both children are seeds
        candidates: CandidateList = [
            ("left", 0.8, {}),
            ("right", 0.4, {}),
        ]
        mock_vector_index.get_vectors.return_value = []

        service = ScoringService(mock_store, mock_vector_index)
        query = [0.6, 0.8]

        scores = service.compute_scores(query, coverage_map, candidates, nodes)

        # Parent = avg(0.8, 0.4) = 0.6
        assert scores["parent"] == pytest.approx(0.6, abs=0.01)

    def test_multi_level_propagation(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Scores propagate through multiple levels of the tree."""
        # Build tree:
        #         grandparent (height 2)
        #        /            \
        #     parent1         parent2
        #     /    \          /    \
        #   leaf1  leaf2   leaf3  leaf4
        leaf1 = make_node("leaf1", height=0, parent_id="parent1")
        leaf2 = make_node("leaf2", height=0, parent_id="parent1")
        leaf3 = make_node("leaf3", height=0, parent_id="parent2")
        leaf4 = make_node("leaf4", height=0, parent_id="parent2")
        parent1 = make_node(
            "parent1",
            height=1,
            parent_id="grandparent",
            left_child_id="leaf1",
            right_child_id="leaf2",
        )
        parent2 = make_node(
            "parent2",
            height=1,
            parent_id="grandparent",
            left_child_id="leaf3",
            right_child_id="leaf4",
        )
        grandparent = make_node(
            "grandparent",
            height=2,
            left_child_id="parent1",
            right_child_id="parent2",
        )
        nodes = {
            "leaf1": leaf1,
            "leaf2": leaf2,
            "leaf3": leaf3,
            "leaf4": leaf4,
            "parent1": parent1,
            "parent2": parent2,
            "grandparent": grandparent,
        }
        coverage_map = {nid: True for nid in nodes}

        # All leaves are seeds with different scores
        candidates: CandidateList = [
            ("leaf1", 1.0, {}),
            ("leaf2", 0.8, {}),
            ("leaf3", 0.6, {}),
            ("leaf4", 0.4, {}),
        ]
        mock_vector_index.get_vectors.return_value = []

        service = ScoringService(mock_store, mock_vector_index)
        query = [0.6, 0.8]

        scores = service.compute_scores(query, coverage_map, candidates, nodes)

        # parent1 = avg(1.0, 0.8) = 0.9
        assert scores["parent1"] == pytest.approx(0.9, abs=0.01)
        # parent2 = avg(0.6, 0.4) = 0.5
        assert scores["parent2"] == pytest.approx(0.5, abs=0.01)
        # grandparent = avg(0.9, 0.5) = 0.7
        assert scores["grandparent"] == pytest.approx(0.7, abs=0.01)

    def test_inner_node_with_one_child_scored_uses_that_child(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Inner node with only one child in coverage uses that child's score."""
        left = make_node("left", height=0, parent_id="parent")
        right = make_node("right", height=0, parent_id="parent")
        parent = make_node(
            "parent",
            height=1,
            left_child_id="left",
            right_child_id="right",
        )
        nodes = {"left": left, "right": right, "parent": parent}
        # Only left child in coverage (and scored)
        coverage_map = {"left": True, "parent": True}

        candidates: CandidateList = [("left", 0.8, {})]
        mock_vector_index.get_vectors.return_value = []

        service = ScoringService(mock_store, mock_vector_index)
        query = [0.6, 0.8]

        scores = service.compute_scores(query, coverage_map, candidates, nodes)

        # Parent = avg of scored children = just left = 0.8
        assert scores["parent"] == pytest.approx(0.8, abs=0.01)

    def test_inner_node_with_no_scored_children_gets_zero(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Inner node with no scored children gets 0.0."""
        left = make_node("left", height=0, parent_id="parent")
        right = make_node("right", height=0, parent_id="parent")
        parent = make_node(
            "parent",
            height=1,
            left_child_id="left",
            right_child_id="right",
        )
        nodes = {"left": left, "right": right, "parent": parent}
        # Only parent in coverage, children not in coverage
        coverage_map = {"parent": True}

        candidates: CandidateList = []
        mock_vector_index.get_vectors.return_value = []

        service = ScoringService(mock_store, mock_vector_index)
        query = [0.6, 0.8]

        scores = service.compute_scores(query, coverage_map, candidates, nodes)

        # Parent has no scored children
        assert scores["parent"] == 0.0


class TestQueryNormalization:
    """Tests for query embedding normalization."""

    def test_query_normalized_before_scoring(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Query embedding is normalized before computing similarity."""
        seed = make_node("seed", height=0, parent_id="parent")
        sibling = make_node("sibling", height=0, parent_id="parent")
        parent = make_node(
            "parent",
            height=1,
            left_child_id="seed",
            right_child_id="sibling",
        )
        nodes = {"seed": seed, "sibling": sibling, "parent": parent}
        coverage_map = {"seed": True, "sibling": True, "parent": True}

        candidates: CandidateList = [("seed", 0.9, {})]

        # Sibling vector same direction as unnormalized query
        sibling_vec = np.array([0.6, 0.8], dtype=np.float64)
        mock_vector_index.get_vectors.return_value = [
            MockVector(id="sibling", vec=sibling_vec)
        ]

        service = ScoringService(mock_store, mock_vector_index)
        # Unnormalized query [3, 4] has magnitude 5, same direction as [0.6, 0.8]
        query = [3.0, 4.0]

        scores = service.compute_scores(query, coverage_map, candidates, nodes)

        # After normalization, query is [0.6, 0.8], same as sibling
        assert scores["sibling"] == pytest.approx(1.0, abs=0.01)
