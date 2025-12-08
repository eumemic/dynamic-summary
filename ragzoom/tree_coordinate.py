"""Coordinate abstraction for tree nodes based on height and level index.

The binary summary tree persists two structural numbers for every node:
- ``height``: distance to the furthest leaf (0 for leaves, increasing upward)
- ``level_index``: position within the nodes of the same height, counted from
  the leftmost node (0-based)

These coordinates are stable across incremental appends and will become the
canonical addressing scheme for bulk structural operations.  ``TreeCoordinate``
wraps the pair and offers small, composable helpers for navigating relatives
without consulting the database.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TreeCoordinate:
    """Immutable coordinate for locating a node within a document tree.

    Args:
        document_id: Optional document the coordinate belongs to.  The coordinate
            math is document agnostic, but knowing the scope is convenient when
            constructing bulk queries.
        height: Distance to the furthest leaf (0 for leaves, increasing upward).
        level_index: 0-based position among nodes with the same ``height``.

    ``TreeCoordinate`` is intentionally lightweight so we can create them on the
    fly while assembling ancestor/sibling batches.  Every helper returns another
    ``TreeCoordinate`` (or an iterator) making it simple to compose operations
    such as "ancestors plus their immediate siblings" without repeatedly
    reimplementing the math.
    """

    document_id: str | None
    height: int
    level_index: int

    def __post_init__(self) -> None:  # pragma: no cover - dataclass hook
        if self.height < 0:
            raise ValueError("height must be non-negative")
        if self.level_index < 0:
            raise ValueError("level_index must be non-negative")

    # ------------------------------------------------------------------
    # Basic relatives
    # ------------------------------------------------------------------
    def parent(self) -> TreeCoordinate:
        """Return the structural parent coordinate.

        Caller is responsible for ensuring that the resulting node actually
        exists in storage (e.g. by bounding with the document's max height).
        """

        return TreeCoordinate(self.document_id, self.height + 1, self.level_index // 2)

    def children(self) -> tuple[TreeCoordinate, TreeCoordinate]:
        """Return left and right child coordinates.

        The children are only valid when ``height > 0``.  The caller can inspect
        ``height`` before using this helper.
        """

        if self.height == 0:
            raise ValueError("Leaf nodes do not have children (height == 0)")
        child_height = self.height - 1
        base_index = self.level_index * 2
        return (
            TreeCoordinate(self.document_id, child_height, base_index),
            TreeCoordinate(self.document_id, child_height, base_index + 1),
        )

    def left_child(self) -> TreeCoordinate:
        """Return the left child coordinate."""

        return self.children()[0]

    def right_child(self) -> TreeCoordinate:
        """Return the right child coordinate."""

        return self.children()[1]

    def sibling(self) -> TreeCoordinate:
        """Return the coordinate of the immediate sibling at the same height."""

        if self.level_index == 0:
            return TreeCoordinate(self.document_id, self.height, 1)
        return TreeCoordinate(self.document_id, self.height, self.level_index ^ 1)

    # ------------------------------------------------------------------
    # Neighbor traversal helpers
    # ------------------------------------------------------------------
    def neighbor(self, offset: int) -> TreeCoordinate:
        """Return a neighbor ``offset`` positions away on the same height."""

        new_index = self.level_index + offset
        if new_index < 0:
            raise ValueError("neighbor offset leads to negative level_index")
        return TreeCoordinate(self.document_id, self.height, new_index)

    def preceding(self) -> TreeCoordinate:
        """Return the immediate preceding neighbor coordinate."""

        if self.level_index == 0:
            raise ValueError("preceding neighbor does not exist for level_index=0")
        return self.neighbor(-1)

    def following(self) -> TreeCoordinate:
        """Return the immediate following neighbor coordinate."""

        return self.neighbor(1)

    def walk_neighbors(self, *, steps: int, direction: int) -> Iterator[TreeCoordinate]:
        """Yield a linear walk of neighbors.

        Args:
            steps: Number of coordinates to emit.
            direction: +1 for rightward, -1 for leftward traversal.
        """

        if steps < 0:
            raise ValueError("steps must be non-negative")
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        current = self
        for _ in range(steps):
            current = current.neighbor(direction)
            yield current

    # ------------------------------------------------------------------
    # Ancestor / descendant sequences
    # ------------------------------------------------------------------
    def ancestors(
        self, *, include_self: bool = False, stop_height: int
    ) -> Iterator[TreeCoordinate]:
        """Yield ancestor coordinates from this node up to ``stop_height``.

        Args:
            include_self: Yield the current coordinate first if True.
            stop_height: Inclusive upper bound for heights; typically the
                document's maximum height.  Iteration stops before emitting any
                coordinate whose height would exceed this value.
        """

        if stop_height < self.height:
            raise ValueError("stop_height must be >= current height")

        current = self if include_self else self.parent()
        while current.height <= stop_height:
            yield current
            if current.height == stop_height:
                break
            current = current.parent()

    def descendants(
        self, depth: int, *, include_self: bool = False
    ) -> Iterator[TreeCoordinate]:
        """Yield coordinates ``depth`` levels below this node.

        Args:
            depth: Number of descendant levels (0 returns nothing unless
                ``include_self`` is True).
            include_self: Yield the origin coordinate before descending.
        """

        if depth < 0:
            raise ValueError("depth must be non-negative")
        if include_self:
            yield self
        if depth == 0:
            return

        queue: list[TreeCoordinate] = [self]
        current_depth = 0
        while queue and current_depth < depth:
            next_level: list[TreeCoordinate] = []
            for coord in queue:
                if coord.height == 0:
                    continue
                left, right = coord.children()
                yield left
                yield right
                next_level.extend((left, right))
            queue = next_level
            current_depth += 1

    # ------------------------------------------------------------------
    # Window / boundary helpers
    # ------------------------------------------------------------------
    def is_left_child(self) -> bool:
        """Return True if this coordinate represents a left child.

        Left children have even level_index and share span_start with their parent.
        """
        return self.level_index % 2 == 0

    def is_right_child(self) -> bool:
        """Return True if this coordinate represents a right child.

        Right children have odd level_index and share span_end with their parent.
        """
        return self.level_index % 2 == 1

    def highest_ancestor_on_boundary(
        self, *, left_edge: bool, max_height: int | None = None
    ) -> TreeCoordinate:
        """Find the highest ancestor that shares the same boundary edge.

        For left_edge=True: walk up while node is a left child (shares span_start).
        For left_edge=False: walk up while node is a right child (shares span_end).

        Note: For the RIGHT edge at the document end, callers should skip this
        computation entirely - the tree naturally covers to the end without
        needing a synthetic edge-max node.

        Args:
            left_edge: If True, find ancestor sharing left boundary; else right.
            max_height: Optional maximum height to stop at (prevents infinite loop).

        Returns:
            The highest ancestor coordinate that shares the specified boundary.
            May return self if this node doesn't share the boundary with its parent.
        """
        current = self
        while True:
            # Stop if we've reached max height
            if max_height is not None and current.height >= max_height:
                break
            # Check if we should continue based on edge alignment
            if left_edge and not current.is_left_child():
                break
            if not left_edge and not current.is_right_child():
                break
            current = current.parent()
        return current

    def highest_ancestor_within_window(
        self,
        *,
        left_edge: bool,
        left_leaf_idx: int,
        right_leaf_idx: int,
        max_height: int | None = None,
    ) -> TreeCoordinate:
        """Find highest ancestor on boundary that stays within window.

        Like highest_ancestor_on_boundary, but stops climbing if the next
        ancestor would extend beyond the window's OTHER edge.

        For left_edge=True: shares span_start with ancestor, but ancestor's
        span_end must not exceed right_leaf_idx.

        For left_edge=False: shares span_end with ancestor, but ancestor's
        span_start must not precede left_leaf_idx.

        Args:
            left_edge: If True, find ancestor sharing left boundary; else right.
            left_leaf_idx: Minimum allowed leaf index (inclusive).
            right_leaf_idx: Maximum allowed leaf index (inclusive).
            max_height: Optional maximum height to stop at.

        Returns:
            The highest ancestor that shares the boundary AND stays within
            the window. May return self if climbing would exceed the window.
        """
        current = self
        while True:
            # Stop if we've reached max height
            if max_height is not None and current.height >= max_height:
                break

            # Check if we should continue based on edge alignment
            if left_edge and not current.is_left_child():
                break
            if not left_edge and not current.is_right_child():
                break

            # Check if parent would exceed window bounds on the OTHER edge
            parent = current.parent()
            parent_left, parent_right = parent.leaf_span()

            if left_edge:
                # For left edge, check if parent extends beyond right boundary
                if parent_right > right_leaf_idx:
                    break
            else:
                # For right edge, check if parent extends before left boundary
                if parent_left < left_leaf_idx:
                    break

            current = parent
        return current

    def leaf_span(self) -> tuple[int, int]:
        """Return the range of leaf-level indices covered by this coordinate.

        Returns:
            Tuple of (left_leaf_idx, right_leaf_idx) where left is inclusive
            and right is the last covered leaf index (inclusive).
        """
        left_leaf = self.level_index << self.height
        right_leaf = ((self.level_index + 1) << self.height) - 1
        return left_leaf, right_leaf

    def is_within_leaf_range(self, left_leaf_idx: int, right_leaf_idx: int) -> bool:
        """Check if this coordinate's leaf span is within [left, right].

        Args:
            left_leaf_idx: Minimum allowed leaf index (inclusive).
            right_leaf_idx: Maximum allowed leaf index (inclusive).

        Returns:
            True if all leaves covered by this coordinate fall within the range.
        """
        my_left, my_right = self.leaf_span()
        return my_left >= left_leaf_idx and my_right <= right_leaf_idx

    # ------------------------------------------------------------------
    # Conversions / utilities
    # ------------------------------------------------------------------
    def as_tuple(self) -> tuple[int, int]:
        """Return ``(height, level_index)`` for dictionary or SQL usage."""

        return self.height, self.level_index

    @classmethod
    def from_tuple(cls, document_id: str | None, pair: Sequence[int]) -> TreeCoordinate:
        """Create a coordinate from a ``(height, level_index)`` pair."""

        if len(pair) != 2:
            raise ValueError("coordinate tuple must contain exactly two elements")
        return cls(
            document_id=document_id, height=int(pair[0]), level_index=int(pair[1])
        )

    @classmethod
    def unique(cls, coordinates: Iterable[TreeCoordinate]) -> list[TreeCoordinate]:
        """Return coordinates de-duplicated while preserving input order."""

        seen: set[tuple[str | None, int, int]] = set()
        ordered: list[TreeCoordinate] = []
        for coord in coordinates:
            key = (coord.document_id, coord.height, coord.level_index)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(coord)
        return ordered
