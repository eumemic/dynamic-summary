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
