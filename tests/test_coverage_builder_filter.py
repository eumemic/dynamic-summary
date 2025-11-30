"""Unit tests for CoverageBuilder._filter_pinned_ancestors method."""

from unittest.mock import MagicMock

import pytest

from ragzoom.retrieval.coverage_builder import CoverageBuilder
from ragzoom.tree_coordinate import TreeCoordinate


@pytest.fixture
def coverage_builder() -> CoverageBuilder:
    """Create a CoverageBuilder with a mock store."""
    mock_store = MagicMock()
    return CoverageBuilder(mock_store)


class TestFilterPinnedAncestors:
    """Tests for _filter_pinned_ancestors method."""

    def test_empty_pinned_coords_returns_unchanged(
        self, coverage_builder: CoverageBuilder
    ) -> None:
        """Empty pinned_coords should return the original coords unchanged."""
        coords = [
            TreeCoordinate("doc", 0, 0),
            TreeCoordinate("doc", 1, 0),
            TreeCoordinate("doc", 2, 0),
        ]
        pinned_coords: list[TreeCoordinate] = []

        filtered = coverage_builder._filter_pinned_ancestors(
            coords, pinned_coords, max_height=2
        )

        assert filtered == coords

    def test_filters_ancestors_of_pinned_leaf(
        self, coverage_builder: CoverageBuilder
    ) -> None:
        """Ancestors of a pinned leaf node should be removed."""
        # Tree structure (height 2):
        #       (2,0)        <- root
        #      /     \
        #   (1,0)   (1,1)    <- parents
        #   /  \    /  \
        # (0,0)(0,1)(0,2)(0,3) <- leaves
        #
        # If (0,0) is pinned, ancestors (1,0) and (2,0) should be removed
        coords = [
            TreeCoordinate("doc", 0, 0),  # pinned leaf
            TreeCoordinate("doc", 0, 1),  # sibling
            TreeCoordinate("doc", 1, 0),  # parent of pinned - should be removed
            TreeCoordinate("doc", 1, 1),  # uncle - should remain
            TreeCoordinate("doc", 2, 0),  # root - should be removed
        ]
        pinned_coords = [TreeCoordinate("doc", 0, 0)]

        filtered = coverage_builder._filter_pinned_ancestors(
            coords, pinned_coords, max_height=2
        )

        filtered_tuples = {c.as_tuple() for c in filtered}
        assert (0, 0) in filtered_tuples  # pinned leaf remains
        assert (0, 1) in filtered_tuples  # sibling remains
        assert (1, 0) not in filtered_tuples  # parent removed
        assert (1, 1) in filtered_tuples  # uncle remains
        assert (2, 0) not in filtered_tuples  # root removed

    def test_preserves_siblings_of_ancestors(
        self, coverage_builder: CoverageBuilder
    ) -> None:
        """Siblings of removed ancestors should be preserved."""
        # Pinned: (0,2) at leaf level
        # Parent: (1,1) - should be removed
        # Parent's sibling: (1,0) - should remain
        coords = [
            TreeCoordinate("doc", 0, 2),  # pinned
            TreeCoordinate("doc", 1, 0),  # sibling of parent
            TreeCoordinate("doc", 1, 1),  # parent - should be removed
            TreeCoordinate("doc", 2, 0),  # root - should be removed
        ]
        pinned_coords = [TreeCoordinate("doc", 0, 2)]

        filtered = coverage_builder._filter_pinned_ancestors(
            coords, pinned_coords, max_height=2
        )

        filtered_tuples = {c.as_tuple() for c in filtered}
        assert (1, 0) in filtered_tuples  # sibling preserved
        assert (1, 1) not in filtered_tuples  # parent removed

    def test_shared_ancestor_chain_early_exit(
        self, coverage_builder: CoverageBuilder
    ) -> None:
        """Multiple pinned nodes sharing ancestors should handle deduplication."""
        # Two pinned leaves (0,0) and (0,1) share parent (1,0) and root (2,0)
        coords = [
            TreeCoordinate("doc", 0, 0),
            TreeCoordinate("doc", 0, 1),
            TreeCoordinate("doc", 1, 0),  # shared parent
            TreeCoordinate("doc", 2, 0),  # shared root
        ]
        pinned_coords = [
            TreeCoordinate("doc", 0, 0),
            TreeCoordinate("doc", 0, 1),
        ]

        filtered = coverage_builder._filter_pinned_ancestors(
            coords, pinned_coords, max_height=2
        )

        # Both pinned leaves remain, ancestors removed
        filtered_tuples = {c.as_tuple() for c in filtered}
        assert (0, 0) in filtered_tuples
        assert (0, 1) in filtered_tuples
        assert (1, 0) not in filtered_tuples
        assert (2, 0) not in filtered_tuples

    def test_pinned_node_at_non_leaf_height(
        self, coverage_builder: CoverageBuilder
    ) -> None:
        """Pinned nodes can be at any height, not just leaves."""
        # Pinned node at height 1
        coords = [
            TreeCoordinate("doc", 1, 0),  # pinned
            TreeCoordinate("doc", 1, 1),  # sibling
            TreeCoordinate("doc", 2, 0),  # root - should be removed
        ]
        pinned_coords = [TreeCoordinate("doc", 1, 0)]

        filtered = coverage_builder._filter_pinned_ancestors(
            coords, pinned_coords, max_height=2
        )

        filtered_tuples = {c.as_tuple() for c in filtered}
        assert (1, 0) in filtered_tuples  # pinned remains
        assert (1, 1) in filtered_tuples  # sibling remains
        assert (2, 0) not in filtered_tuples  # root removed

    def test_multiple_separate_trees(self, coverage_builder: CoverageBuilder) -> None:
        """Pinned nodes in separate subtrees should each filter their ancestors."""
        # Two pinned leaves in different subtrees
        # (0,0) under (1,0) under (2,0)
        # (0,3) under (1,1) under (2,0)
        coords = [
            TreeCoordinate("doc", 0, 0),  # pinned
            TreeCoordinate("doc", 0, 3),  # pinned
            TreeCoordinate("doc", 1, 0),  # ancestor of (0,0) - remove
            TreeCoordinate("doc", 1, 1),  # ancestor of (0,3) - remove
            TreeCoordinate("doc", 2, 0),  # shared root - remove
        ]
        pinned_coords = [
            TreeCoordinate("doc", 0, 0),
            TreeCoordinate("doc", 0, 3),
        ]

        filtered = coverage_builder._filter_pinned_ancestors(
            coords, pinned_coords, max_height=2
        )

        filtered_tuples = {c.as_tuple() for c in filtered}
        assert (0, 0) in filtered_tuples
        assert (0, 3) in filtered_tuples
        assert (1, 0) not in filtered_tuples
        assert (1, 1) not in filtered_tuples
        assert (2, 0) not in filtered_tuples

    def test_ancestors_removed_up_to_max_height(
        self, coverage_builder: CoverageBuilder
    ) -> None:
        """Ancestors are removed up to max_height even if not in coords."""
        # Pinned at height 1, max_height=3
        # Ancestors (2,0) should be removed from coords
        coords = [
            TreeCoordinate("doc", 1, 0),  # pinned
            TreeCoordinate("doc", 2, 0),  # parent - removed
        ]
        pinned_coords = [TreeCoordinate("doc", 1, 0)]

        filtered = coverage_builder._filter_pinned_ancestors(
            coords, pinned_coords, max_height=3
        )

        filtered_tuples = {c.as_tuple() for c in filtered}
        assert (1, 0) in filtered_tuples  # pinned remains
        assert (2, 0) not in filtered_tuples  # parent removed
