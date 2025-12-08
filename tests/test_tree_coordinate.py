import pytest

from ragzoom.tree_coordinate import TreeCoordinate


def test_coordinate_basic_parent_child_relationships() -> None:
    coord = TreeCoordinate(document_id="doc-1", height=2, level_index=3)
    parent = coord.parent()
    assert parent.height == 3
    assert parent.level_index == 1

    left, right = parent.children()
    assert left.height == 2 and left.level_index == 2
    assert right.height == 2 and right.level_index == 3


def test_sibling_and_neighbors() -> None:
    coord = TreeCoordinate(document_id=None, height=4, level_index=6)
    sibling = coord.sibling()
    assert sibling.height == 4
    assert sibling.level_index == 7

    preceding = coord.preceding()
    assert preceding.level_index == 5

    following = coord.following()
    assert following.level_index == 7

    walk = list(coord.walk_neighbors(steps=3, direction=1))
    assert [c.level_index for c in walk] == [7, 8, 9]


@pytest.mark.parametrize("height, level_index", [(-1, 0), (0, -5)])
def test_invalid_coordinates_rejected(height: int, level_index: int) -> None:
    with pytest.raises(ValueError):
        TreeCoordinate(document_id=None, height=height, level_index=level_index)


def test_ancestors_requires_stop_height() -> None:
    leaf = TreeCoordinate(document_id="doc", height=0, level_index=5)
    ancestors = list(leaf.ancestors(stop_height=3))
    assert [(c.height, c.level_index) for c in ancestors] == [
        (1, 2),
        (2, 1),
        (3, 0),
    ]

    inclusive = list(leaf.ancestors(include_self=True, stop_height=2))
    assert [(c.height, c.level_index) for c in inclusive] == [
        (0, 5),
        (1, 2),
        (2, 1),
    ]


def test_descendants_traversal() -> None:
    root = TreeCoordinate(document_id="doc", height=3, level_index=0)
    descendants = list(root.descendants(depth=2))
    # Depth 1 -> level_index 0 and 1, depth 2 -> four nodes
    expected = [
        (2, 0),
        (2, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]
    assert [(c.height, c.level_index) for c in descendants] == expected


def test_unique_preserves_order() -> None:
    coords = [
        TreeCoordinate("doc", 1, 0),
        TreeCoordinate("doc", 1, 0),
        TreeCoordinate("doc", 2, 1),
        TreeCoordinate("doc", 1, 1),
        TreeCoordinate("doc", 2, 1),
    ]
    deduped = TreeCoordinate.unique(coords)
    assert len(deduped) == 3
    assert deduped[0].height == 1 and deduped[0].level_index == 0
    assert deduped[1].height == 2 and deduped[1].level_index == 1
    assert deduped[2].height == 1 and deduped[2].level_index == 1


# --- Tests for windowed query support ---


class TestLeftRightChild:
    """Tests for is_left_child and is_right_child methods."""

    def test_even_level_index_is_left_child(self) -> None:
        """Even level_index nodes are left children (share span_start with parent)."""
        for level_index in [0, 2, 4, 6, 100]:
            coord = TreeCoordinate("doc", height=0, level_index=level_index)
            assert coord.is_left_child() is True
            assert coord.is_right_child() is False

    def test_odd_level_index_is_right_child(self) -> None:
        """Odd level_index nodes are right children (share span_end with parent)."""
        for level_index in [1, 3, 5, 7, 101]:
            coord = TreeCoordinate("doc", height=0, level_index=level_index)
            assert coord.is_left_child() is False
            assert coord.is_right_child() is True

    def test_applies_at_all_heights(self) -> None:
        """Left/right child determination works at any height."""
        for height in [0, 1, 5, 10]:
            left = TreeCoordinate("doc", height=height, level_index=4)
            right = TreeCoordinate("doc", height=height, level_index=5)
            assert left.is_left_child() is True
            assert right.is_right_child() is True


class TestHighestAncestorOnBoundary:
    """Tests for edge-max computation used in windowed queries."""

    def test_left_edge_stops_at_right_child(self) -> None:
        """Walking up left edge stops when we hit a right child."""
        # Leaf at level_index=5 (odd = right child) should return itself
        leaf = TreeCoordinate("doc", height=0, level_index=5)
        edge_max = leaf.highest_ancestor_on_boundary(left_edge=True, max_height=10)
        assert edge_max == leaf

    def test_left_edge_walks_up_left_children(self) -> None:
        """Walking up left edge continues through left children."""
        # Leaf at level_index=4 (even = left child)
        # Parent at (1, 2) - even = left child
        # Grandparent at (2, 1) - odd = right child -> stop here
        leaf = TreeCoordinate("doc", height=0, level_index=4)
        edge_max = leaf.highest_ancestor_on_boundary(left_edge=True, max_height=10)
        assert edge_max.height == 2
        assert edge_max.level_index == 1

    def test_right_edge_stops_at_left_child(self) -> None:
        """Walking up right edge stops when we hit a left child."""
        # Leaf at level_index=4 (even = left child) should return itself
        leaf = TreeCoordinate("doc", height=0, level_index=4)
        edge_max = leaf.highest_ancestor_on_boundary(left_edge=False, max_height=10)
        assert edge_max == leaf

    def test_right_edge_walks_up_right_children(self) -> None:
        """Walking up right edge continues through right children."""
        # Leaf at level_index=5 (odd = right child)
        # Parent at (1, 2) - even = left child -> stop here
        leaf = TreeCoordinate("doc", height=0, level_index=5)
        edge_max = leaf.highest_ancestor_on_boundary(left_edge=False, max_height=10)
        assert edge_max.height == 1
        assert edge_max.level_index == 2

    def test_level_index_zero_respects_max_height(self) -> None:
        """Leaf at level_index=0 is always left child - must respect max_height."""
        # This is the critical test that would have caught the infinite loop bug!
        # level_index=0 is always even (left child), so without max_height it loops forever
        leaf = TreeCoordinate("doc", height=0, level_index=0)
        edge_max = leaf.highest_ancestor_on_boundary(left_edge=True, max_height=5)
        # Should walk up to height 5, where level_index becomes 0
        assert edge_max.height == 5
        assert edge_max.level_index == 0

    def test_rightmost_leaf_respects_max_height(self) -> None:
        """Rightmost leaf (all ancestors are right children) respects max_height."""
        # If we have 8 leaves (0-7), leaf 7 is always a right child up the tree
        # 7 -> (1,3) -> (2,1) -> (3,0) - actually 3 is odd, 1 is odd, 0 is even
        # Let's use leaf 7: parent (1,3) odd, grandparent (2,1) odd, great-grandparent (3,0) even
        leaf = TreeCoordinate("doc", height=0, level_index=7)
        edge_max = leaf.highest_ancestor_on_boundary(left_edge=False, max_height=10)
        # Walks up: (0,7) -> (1,3) -> (2,1) -> (3,0) stops because 0 is even (left child)
        assert edge_max.height == 3
        assert edge_max.level_index == 0

    def test_max_height_none_still_terminates(self) -> None:
        """Without max_height, still terminates when boundary changes."""
        # Leaf at level_index=3 (odd = right child) returns itself for left_edge
        leaf = TreeCoordinate("doc", height=0, level_index=3)
        edge_max = leaf.highest_ancestor_on_boundary(left_edge=True)
        assert edge_max == leaf


class TestLeafSpan:
    """Tests for leaf_span method."""

    def test_leaf_spans_itself(self) -> None:
        """A leaf (height=0) spans only itself."""
        leaf = TreeCoordinate("doc", height=0, level_index=5)
        left, right = leaf.leaf_span()
        assert left == 5
        assert right == 5

    def test_height_1_spans_two_leaves(self) -> None:
        """Height 1 node spans 2 leaves."""
        node = TreeCoordinate("doc", height=1, level_index=3)
        left, right = node.leaf_span()
        # level_index=3 at height=1 covers leaves 6 and 7
        assert left == 6
        assert right == 7

    def test_height_2_spans_four_leaves(self) -> None:
        """Height 2 node spans 4 leaves."""
        node = TreeCoordinate("doc", height=2, level_index=1)
        left, right = node.leaf_span()
        # level_index=1 at height=2 covers leaves 4, 5, 6, 7
        assert left == 4
        assert right == 7

    def test_root_spans_all_leaves(self) -> None:
        """Root at height=3, level_index=0 spans 8 leaves."""
        root = TreeCoordinate("doc", height=3, level_index=0)
        left, right = root.leaf_span()
        assert left == 0
        assert right == 7


class TestHighestAncestorWithinWindow:
    """Tests for edge-max computation that respects window boundaries.

    Unlike highest_ancestor_on_boundary which only checks edge alignment,
    this method also checks that climbing doesn't extend beyond the OTHER
    edge of the window.

    Tree structure for reference (8 leaves):
    Height 0: [0] [1] [2] [3] [4] [5] [6] [7]   (leaves)
    Height 1:  [0,1]  [2,3]  [4,5]  [6,7]       (indices 0,1,2,3)
    Height 2:    [0-3]      [4-7]              (indices 0,1)
    Height 3:        [0-7]                     (index 0 = root)
    """

    # --- Edge alignment tests (same as highest_ancestor_on_boundary) ---

    def test_left_edge_stops_at_right_child(self) -> None:
        """Leaf at odd index (right child) stops immediately."""
        leaf = TreeCoordinate("doc", height=0, level_index=5)
        result = leaf.highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=0, right_leaf_idx=7, max_height=10
        )
        assert result == leaf

    def test_left_edge_walks_up_left_children(self) -> None:
        """Left edge climbs through left children until hitting right child."""
        # Leaf 4 → (1,2) → (2,1) stops at (2,1) because 1 is odd
        leaf = TreeCoordinate("doc", height=0, level_index=4)
        result = leaf.highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=0, right_leaf_idx=7, max_height=10
        )
        assert result.height == 2
        assert result.level_index == 1

    def test_right_edge_stops_at_left_child(self) -> None:
        """Leaf at even index (left child) stops immediately."""
        leaf = TreeCoordinate("doc", height=0, level_index=4)
        result = leaf.highest_ancestor_within_window(
            left_edge=False, left_leaf_idx=0, right_leaf_idx=7, max_height=10
        )
        assert result == leaf

    def test_right_edge_walks_up_right_children(self) -> None:
        """Right edge climbs through right children in full document window."""
        # Leaf 7 → (1,3) → (2,1) → (3,0) stops at (3,0) because 0 is even
        leaf = TreeCoordinate("doc", height=0, level_index=7)
        result = leaf.highest_ancestor_within_window(
            left_edge=False, left_leaf_idx=0, right_leaf_idx=7, max_height=10
        )
        assert result.height == 3
        assert result.level_index == 0

    # --- Window boundary constraint tests (NEW behavior) ---

    def test_left_edge_stopped_by_right_boundary(self) -> None:
        """Left edge stops when parent would exceed right boundary."""
        # Leaf 0 in window [0,3]: climbs to (2,0) which spans [0,3]
        # Next parent (3,0) spans [0,7] which exceeds right_leaf_idx=3
        leaf = TreeCoordinate("doc", height=0, level_index=0)
        result = leaf.highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=0, right_leaf_idx=3, max_height=10
        )
        assert result.height == 2
        assert result.level_index == 0

    def test_right_edge_stopped_by_left_boundary(self) -> None:
        """Right edge stops when parent would exceed left boundary."""
        # Leaf 7 in window [4,7]: climbs to (2,1) which spans [4,7]
        # Next parent (3,0) spans [0,7] which is below left_leaf_idx=4
        leaf = TreeCoordinate("doc", height=0, level_index=7)
        result = leaf.highest_ancestor_within_window(
            left_edge=False, left_leaf_idx=4, right_leaf_idx=7, max_height=10
        )
        assert result.height == 2
        assert result.level_index == 1

    def test_narrow_window_limits_climbing(self) -> None:
        """Narrow window stops climbing early due to boundary check."""
        # Leaf 2 in window [2,5]: climbs to (1,1) which spans [2,3]
        # Stops at (1,1) because 1 is odd (right child), not boundary
        leaf = TreeCoordinate("doc", height=0, level_index=2)
        result = leaf.highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=2, right_leaf_idx=5, max_height=10
        )
        assert result.height == 1
        assert result.level_index == 1

    # --- Full document window tests ---

    def test_left_edge_climbs_to_root_full_doc(self) -> None:
        """Left edge climbs to root when window is full document."""
        leaf = TreeCoordinate("doc", height=0, level_index=0)
        result = leaf.highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=0, right_leaf_idx=7, max_height=3
        )
        assert result.height == 3
        assert result.level_index == 0

    # --- Single leaf window tests ---

    def test_single_leaf_window_cannot_climb(self) -> None:
        """Single leaf window prevents climbing beyond self."""
        # Leaf 2 in window [2,2]: parent (1,1) spans [2,3] > 2
        leaf = TreeCoordinate("doc", height=0, level_index=2)
        result = leaf.highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=2, right_leaf_idx=2, max_height=10
        )
        assert result == leaf

    # --- Max height constraint tests ---

    def test_max_height_stops_climbing(self) -> None:
        """Max height constraint stops climbing before boundary check."""
        leaf = TreeCoordinate("doc", height=0, level_index=0)
        result = leaf.highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=0, right_leaf_idx=7, max_height=2
        )
        assert result.height == 2
        assert result.level_index == 0

    # --- Both edges converge tests ---

    def test_both_edges_same_ancestor(self) -> None:
        """Both edges of narrow window converge to same ancestor."""
        # Window [0,1]: both leaf 0 (left) and leaf 1 (right) → (1,0)
        left_result = TreeCoordinate("doc", 0, 0).highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=0, right_leaf_idx=1, max_height=10
        )
        right_result = TreeCoordinate("doc", 0, 1).highest_ancestor_within_window(
            left_edge=False, left_leaf_idx=0, right_leaf_idx=1, max_height=10
        )
        assert left_result.height == 1 and left_result.level_index == 0
        assert right_result.height == 1 and right_result.level_index == 0
        assert left_result == right_result

    def test_neither_edge_can_climb(self) -> None:
        """Window where both boundary leaves are wrong child type."""
        # Window [3,4]: leaf 3 is right child, leaf 4 is left child
        left_result = TreeCoordinate("doc", 0, 3).highest_ancestor_within_window(
            left_edge=True, left_leaf_idx=3, right_leaf_idx=4, max_height=10
        )
        right_result = TreeCoordinate("doc", 0, 4).highest_ancestor_within_window(
            left_edge=False, left_leaf_idx=3, right_leaf_idx=4, max_height=10
        )
        # Both return themselves - can't climb at all
        assert left_result.height == 0 and left_result.level_index == 3
        assert right_result.height == 0 and right_result.level_index == 4


class TestIsWithinLeafRange:
    """Tests for coordinate window filtering."""

    def test_leaf_within_range(self) -> None:
        """Leaf inside range returns True."""
        leaf = TreeCoordinate("doc", height=0, level_index=5)
        assert leaf.is_within_leaf_range(3, 7) is True

    def test_leaf_outside_range_left(self) -> None:
        """Leaf before range returns False."""
        leaf = TreeCoordinate("doc", height=0, level_index=2)
        assert leaf.is_within_leaf_range(3, 7) is False

    def test_leaf_outside_range_right(self) -> None:
        """Leaf after range returns False."""
        leaf = TreeCoordinate("doc", height=0, level_index=8)
        assert leaf.is_within_leaf_range(3, 7) is False

    def test_leaf_at_boundary_inclusive(self) -> None:
        """Leaves at boundaries are included."""
        left_boundary = TreeCoordinate("doc", height=0, level_index=3)
        right_boundary = TreeCoordinate("doc", height=0, level_index=7)
        assert left_boundary.is_within_leaf_range(3, 7) is True
        assert right_boundary.is_within_leaf_range(3, 7) is True

    def test_parent_within_range(self) -> None:
        """Parent node fully within range returns True."""
        # Height 1, level_index=2 spans leaves 4-5, which is within 3-7
        node = TreeCoordinate("doc", height=1, level_index=2)
        assert node.is_within_leaf_range(3, 7) is True

    def test_parent_partially_outside_range(self) -> None:
        """Parent node extending beyond range returns False."""
        # Height 1, level_index=1 spans leaves 2-3, left edge (2) is outside 3-7
        node = TreeCoordinate("doc", height=1, level_index=1)
        assert node.is_within_leaf_range(3, 7) is False

    def test_large_node_spanning_beyond_range(self) -> None:
        """Large node extending beyond both sides returns False."""
        # Height 3, level_index=0 spans leaves 0-7, but range is 2-5
        node = TreeCoordinate("doc", height=3, level_index=0)
        assert node.is_within_leaf_range(2, 5) is False

    def test_node_exactly_matching_range(self) -> None:
        """Node exactly matching range returns True."""
        # Height 2, level_index=0 spans leaves 0-3
        node = TreeCoordinate("doc", height=2, level_index=0)
        assert node.is_within_leaf_range(0, 3) is True
