"""Tests for ScoringService with bottom-up score propagation."""

from unittest.mock import Mock

import numpy as np
import pytest

from ragzoom.retrieval.scoring_service import ScoringService
from ragzoom.vector_api import Vector

# Type alias for candidates list
CandidateList = list[tuple[str, float, dict[str, str | int | float | bool | None]]]


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
) -> Mock:
    """Create a mock TreeNode."""
    node = Mock()
    node.id = node_id
    node.height = height
    node.parent_id = parent_id
    node.left_child_id = left_child_id
    node.right_child_id = right_child_id
    return node


class TestGetSiblingIds:
    """Tests for _get_sibling_ids method."""

    def test_finds_sibling_of_seed(self, scoring_service: ScoringService) -> None:
        """Seed's sibling (via parent) should be found."""
        # Tree: parent -> [seed, sibling]
        nodes = {
            "seed": make_node("seed", height=0, parent_id="parent"),
            "sibling": make_node("sibling", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="seed", right_child_id="sibling"
            ),
        }
        coverage_map = {"seed": True, "sibling": True, "parent": True}

        result = scoring_service._get_sibling_ids({"seed"}, nodes, coverage_map)

        assert result == {"sibling"}

    def test_excludes_seeds_from_siblings(
        self, scoring_service: ScoringService
    ) -> None:
        """Seeds that are also siblings should not be returned."""
        # Both children are seeds
        nodes = {
            "seed1": make_node("seed1", height=0, parent_id="parent"),
            "seed2": make_node("seed2", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="seed1", right_child_id="seed2"
            ),
        }
        coverage_map = {"seed1": True, "seed2": True, "parent": True}

        result = scoring_service._get_sibling_ids(
            {"seed1", "seed2"}, nodes, coverage_map
        )

        assert result == set()  # No siblings, both are seeds

    def test_sibling_not_in_coverage_excluded(
        self, scoring_service: ScoringService
    ) -> None:
        """Siblings not in coverage should not be returned."""
        nodes = {
            "seed": make_node("seed", height=0, parent_id="parent"),
            "sibling": make_node("sibling", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="seed", right_child_id="sibling"
            ),
        }
        coverage_map = {"seed": True, "parent": True}  # sibling NOT in coverage

        result = scoring_service._get_sibling_ids({"seed"}, nodes, coverage_map)

        assert result == set()

    def test_seed_without_parent(self, scoring_service: ScoringService) -> None:
        """Root seed (no parent) has no sibling."""
        nodes = {
            "root_seed": make_node("root_seed", height=1, parent_id=None),
        }
        coverage_map = {"root_seed": True}

        result = scoring_service._get_sibling_ids({"root_seed"}, nodes, coverage_map)

        assert result == set()

    def test_multiple_seeds_find_all_siblings(
        self, scoring_service: ScoringService
    ) -> None:
        """Multiple seeds from different parents find their respective siblings."""
        nodes = {
            "seed1": make_node("seed1", height=0, parent_id="parent1"),
            "sib1": make_node("sib1", height=0, parent_id="parent1"),
            "parent1": make_node(
                "parent1", height=1, left_child_id="seed1", right_child_id="sib1"
            ),
            "seed2": make_node("seed2", height=0, parent_id="parent2"),
            "sib2": make_node("sib2", height=0, parent_id="parent2"),
            "parent2": make_node(
                "parent2", height=1, left_child_id="seed2", right_child_id="sib2"
            ),
        }
        coverage_map = {
            "seed1": True,
            "sib1": True,
            "parent1": True,
            "seed2": True,
            "sib2": True,
            "parent2": True,
        }

        result = scoring_service._get_sibling_ids(
            {"seed1", "seed2"}, nodes, coverage_map
        )

        assert result == {"sib1", "sib2"}


class TestPropagateScoresBottomUp:
    """Tests for _propagate_scores_bottom_up method."""

    def test_parent_gets_avg_of_children(self, scoring_service: ScoringService) -> None:
        """Parent score = avg(left_child, right_child)."""
        nodes = {
            "left": make_node("left", height=0, parent_id="parent"),
            "right": make_node("right", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="left", right_child_id="right"
            ),
        }
        coverage_map = {"left": True, "right": True, "parent": True}
        scores = {"left": 0.8, "right": 0.4}

        scoring_service._propagate_scores_bottom_up(scores, nodes, coverage_map)

        assert scores["parent"] == pytest.approx(0.6)  # avg(0.8, 0.4)

    def test_single_child_uses_child_score(
        self, scoring_service: ScoringService
    ) -> None:
        """Parent with one child gets that child's score."""
        nodes = {
            "child": make_node("child", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="child", right_child_id=None
            ),
        }
        coverage_map = {"child": True, "parent": True}
        scores = {"child": 0.7}

        scoring_service._propagate_scores_bottom_up(scores, nodes, coverage_map)

        assert scores["parent"] == pytest.approx(0.7)

    def test_no_children_in_scores_gets_zero(
        self, scoring_service: ScoringService
    ) -> None:
        """Parent with no scored children gets 0.0."""
        nodes = {
            "child": make_node("child", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="child", right_child_id=None
            ),
        }
        coverage_map = {"parent": True}  # child not in coverage
        scores: dict[str, float] = {}  # child has no score

        scoring_service._propagate_scores_bottom_up(scores, nodes, coverage_map)

        assert scores["parent"] == 0.0

    def test_multi_level_propagation(self, scoring_service: ScoringService) -> None:
        """Scores propagate through multiple levels bottom-up."""
        #       root (h=2)
        #      /    \
        #   mid1    mid2 (h=1)
        #   /  \    /  \
        #  l1  l2  l3  l4 (h=0)
        nodes = {
            "l1": make_node("l1", height=0, parent_id="mid1"),
            "l2": make_node("l2", height=0, parent_id="mid1"),
            "l3": make_node("l3", height=0, parent_id="mid2"),
            "l4": make_node("l4", height=0, parent_id="mid2"),
            "mid1": make_node(
                "mid1",
                height=1,
                parent_id="root",
                left_child_id="l1",
                right_child_id="l2",
            ),
            "mid2": make_node(
                "mid2",
                height=1,
                parent_id="root",
                left_child_id="l3",
                right_child_id="l4",
            ),
            "root": make_node(
                "root", height=2, left_child_id="mid1", right_child_id="mid2"
            ),
        }
        coverage_map = {k: True for k in nodes}
        scores = {"l1": 1.0, "l2": 0.5, "l3": 0.8, "l4": 0.2}

        scoring_service._propagate_scores_bottom_up(scores, nodes, coverage_map)

        assert scores["mid1"] == pytest.approx(0.75)  # avg(1.0, 0.5)
        assert scores["mid2"] == pytest.approx(0.5)  # avg(0.8, 0.2)
        assert scores["root"] == pytest.approx(0.625)  # avg(0.75, 0.5)

    def test_ancestor_seed_score_overwritten(
        self, scoring_service: ScoringService
    ) -> None:
        """Seed at inner node has score overwritten by propagation."""
        nodes = {
            "child": make_node("child", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="child", right_child_id=None
            ),
        }
        coverage_map = {"child": True, "parent": True}
        # parent is a seed with its own score
        scores = {"child": 0.6, "parent": 0.9}

        scoring_service._propagate_scores_bottom_up(scores, nodes, coverage_map)

        # parent's seed score should be overwritten with propagated score
        assert scores["parent"] == pytest.approx(0.6)


class TestComputeScoresIntegration:
    """Integration tests for compute_scores with bottom-up propagation."""

    def test_seeds_get_candidate_scores(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Seeds use scores from candidates list."""
        service = ScoringService(mock_store, mock_vector_index)
        mock_vector_index.get_vectors.return_value = []

        nodes = {
            "seed": make_node("seed", height=0),
        }
        coverage_map = {"seed": True}
        candidates: CandidateList = [("seed", 0.85, {})]

        scores = service.compute_scores([0.1, 0.2], coverage_map, candidates, nodes)

        assert scores["seed"] == 0.85

    def test_siblings_get_embedding_scores(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Siblings of seeds get scores from embedding fetch."""
        # Sibling embedding will dot to ~0.5 with normalized query
        sibling_vec = np.array([0.6, 0.8], dtype=np.float32)
        sibling_vec = sibling_vec / np.linalg.norm(sibling_vec)

        mock_vector_index.get_vectors.return_value = [
            Vector("sibling", sibling_vec, {}, "model", 2)
        ]

        service = ScoringService(mock_store, mock_vector_index)

        nodes = {
            "seed": make_node("seed", height=0, parent_id="parent"),
            "sibling": make_node("sibling", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="seed", right_child_id="sibling"
            ),
        }
        coverage_map = {"seed": True, "sibling": True, "parent": True}
        candidates: CandidateList = [("seed", 0.9, {})]

        query_vec = [0.6, 0.8]  # Same direction as sibling
        scores = service.compute_scores(query_vec, coverage_map, candidates, nodes)

        assert scores["sibling"] == pytest.approx(1.0, abs=0.01)
        mock_vector_index.get_vectors.assert_called_once_with(["sibling"])

    def test_inner_nodes_get_propagated_scores(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Inner nodes get avg of children's scores."""
        mock_vector_index.get_vectors.return_value = []

        service = ScoringService(mock_store, mock_vector_index)

        nodes = {
            "seed1": make_node("seed1", height=0, parent_id="parent"),
            "seed2": make_node("seed2", height=0, parent_id="parent"),
            "parent": make_node(
                "parent", height=1, left_child_id="seed1", right_child_id="seed2"
            ),
        }
        coverage_map = {"seed1": True, "seed2": True, "parent": True}
        candidates: CandidateList = [("seed1", 0.8, {}), ("seed2", 0.4, {})]

        scores = service.compute_scores([0.1, 0.2], coverage_map, candidates, nodes)

        assert scores["parent"] == pytest.approx(0.6)

    def test_ancestor_sibling_gets_zero(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Ancestor's sibling with no scored children gets 0.0."""
        mock_vector_index.get_vectors.return_value = []

        service = ScoringService(mock_store, mock_vector_index)

        #       root (h=2)
        #      /    \
        #   left    right (h=1) <- ancestor sibling, no seeds below
        #   /  \
        # seed  sib (h=0)
        nodes = {
            "seed": make_node("seed", height=0, parent_id="left"),
            "sib": make_node("sib", height=0, parent_id="left"),
            "left": make_node(
                "left",
                height=1,
                parent_id="root",
                left_child_id="seed",
                right_child_id="sib",
            ),
            "right": make_node(
                "right",
                height=1,
                parent_id="root",
                left_child_id=None,
                right_child_id=None,
            ),
            "root": make_node(
                "root", height=2, left_child_id="left", right_child_id="right"
            ),
        }
        # right subtree has no children in coverage (only the inner node)
        coverage_map = {
            "seed": True,
            "sib": True,
            "left": True,
            "right": True,
            "root": True,
        }
        candidates: CandidateList = [("seed", 0.8, {})]

        # sib is fetched but has no vector
        mock_vector_index.get_vectors.return_value = []

        scores = service.compute_scores([0.1, 0.2], coverage_map, candidates, nodes)

        # right has no children in scores -> 0.0
        assert scores["right"] == 0.0
        # left gets avg of seed (0.8) and sib (not scored -> not in child_scores)
        # Actually sib won't be in scores since get_vectors returned empty
        # So left = avg(0.8) = 0.8
        assert scores["left"] == pytest.approx(0.8)
        # root = avg(0.8, 0.0) = 0.4
        assert scores["root"] == pytest.approx(0.4)

    def test_without_nodes_returns_seed_scores_only(
        self, mock_store: Mock, mock_vector_index: Mock
    ) -> None:
        """Without nodes dict, only seeds get scores (no propagation)."""
        service = ScoringService(mock_store, mock_vector_index)

        coverage_map = {"seed": True, "other": True}
        candidates: CandidateList = [("seed", 0.75, {})]

        scores = service.compute_scores(
            [0.1, 0.2], coverage_map, candidates, nodes=None
        )

        assert scores == {"seed": 0.75}
