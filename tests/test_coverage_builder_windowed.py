"""Unit tests for CoverageBuilder windowed coverage methods.

Tests cover:
- compute_window_bounds: window boundary computation and edge-max discovery
- build_windowed_coverage: full windowed coverage construction

Note: Window coordinate filtering is tested via TreeCoordinate.is_within_leaf_range
in test_tree_coordinate.py and through integration tests in TestBuildWindowedCoverage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from ragzoom.retrieval.coverage_builder import CoverageBuilder

if TYPE_CHECKING:
    pass


@dataclass
class MockTreeNode:
    """Minimal mock tree node for testing."""

    id: str
    document_id: str
    span_start: int
    span_end: int
    height: int
    level_index: int
    token_count: int = 10
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    text: str = "test content"


class TestComputeWindowBounds:
    """Tests for compute_window_bounds method."""

    def _create_mock_store(
        self,
        leaves: list[MockTreeNode],
        max_height: int = 3,
        doc_span_end: int | None = None,
    ) -> MagicMock:
        """Create a mock store with configured node repository."""
        mock_store = MagicMock()

        # Create mock repository
        mock_repo = MagicMock()

        def get_leaf_at_position(
            document_id: str, position: int
        ) -> MockTreeNode | None:
            for leaf in leaves:
                if (
                    leaf.document_id == document_id
                    and leaf.span_start <= position < leaf.span_end
                ):
                    return leaf
            return None

        mock_repo.get_leaf_at_span_position = MagicMock(
            side_effect=get_leaf_at_position
        )
        mock_repo.get_document_span_end = MagicMock(return_value=doc_span_end)

        # Wire up store.nodes._repo
        mock_nodes = MagicMock()
        mock_nodes._repo = mock_repo
        mock_nodes.max_height = MagicMock(return_value=max_height)
        mock_store.nodes = mock_nodes

        return mock_store

    def test_basic_window_bounds(self) -> None:
        """Compute window bounds for a simple case."""
        # Create leaves: [0-100), [100-200), [200-300), [300-400)
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
            MockTreeNode("L2", "doc", 200, 300, 0, 2),
            MockTreeNode("L3", "doc", 300, 400, 0, 3),
        ]
        store = self._create_mock_store(leaves, max_height=2, doc_span_end=400)
        builder = CoverageBuilder(store)

        # Request window [150, 250) - should expand to [100, 300)
        bounds = builder.compute_window_bounds(
            span_start=150, span_end=250, document_id="doc"
        )

        assert bounds.actual_start == 100  # L1.span_start
        assert bounds.actual_end == 300  # L2.span_end
        assert bounds.left_leaf_index == 1
        assert bounds.right_leaf_index == 2

    def test_window_at_document_start(self) -> None:
        """Window starting at document beginning."""
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
            MockTreeNode("L2", "doc", 200, 300, 0, 2),
        ]
        store = self._create_mock_store(leaves, max_height=2, doc_span_end=300)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=0, span_end=150, document_id="doc"
        )

        assert bounds.actual_start == 0
        assert bounds.actual_end == 200  # L1.span_end
        assert bounds.left_leaf_index == 0
        assert bounds.right_leaf_index == 1

    def test_window_at_document_end(self) -> None:
        """Window extending to document end."""
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
            MockTreeNode("L2", "doc", 200, 300, 0, 2),
        ]
        store = self._create_mock_store(leaves, max_height=2, doc_span_end=300)
        builder = CoverageBuilder(store)

        # Request window to document end
        bounds = builder.compute_window_bounds(
            span_start=150, span_end=300, document_id="doc"
        )

        assert bounds.actual_start == 100  # L1.span_start
        assert bounds.actual_end == 300  # L2.span_end (document end)
        assert bounds.left_leaf_index == 1
        assert bounds.right_leaf_index == 2

    def test_edge_max_left_walks_up_left_children(self) -> None:
        """Left edge-max walks up through left children (when not at doc start)."""
        # Leaf at index 2 (even = left child)
        # Parent at (1,1) is odd = right child -> stop here
        # Window [200, 400) does NOT start at doc start, so edge-max applies
        leaves = [
            MockTreeNode(f"L{i}", "doc", i * 100, (i + 1) * 100, 0, i) for i in range(8)
        ]
        store = self._create_mock_store(leaves, max_height=3, doc_span_end=800)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=200, span_end=400, document_id="doc"
        )

        # Left boundary at index 2 (even = left child)
        # Parent (1,1) is odd = right child -> stop at (1,1)
        assert bounds.left_edge_max is not None
        assert bounds.left_edge_max.height == 1
        assert bounds.left_edge_max.level_index == 1

    def test_edge_max_left_skipped_at_document_start(self) -> None:
        """Left edge-max is skipped when window starts at document beginning."""
        # When the window starts at document start, we skip edge-max computation
        # because the tree naturally covers from the beginning.
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
        ]
        store = self._create_mock_store(leaves, max_height=3, doc_span_end=200)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=0, span_end=100, document_id="doc"
        )

        # Window covers only leaf 0. Left edge-max starts at leaf 0 (left child),
        # but cannot climb because parent would cover leaf 1 which is outside window.
        assert bounds.left_edge_max is not None
        assert bounds.left_edge_max.height == 0
        assert bounds.left_edge_max.level_index == 0

    def test_edge_max_left_stops_at_right_child(self) -> None:
        """Left edge-max stops when encountering a right child."""
        # Leaf at index 3 is a right child (odd)
        # Should return itself immediately
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
            MockTreeNode("L2", "doc", 200, 300, 0, 2),
            MockTreeNode("L3", "doc", 300, 400, 0, 3),
        ]
        store = self._create_mock_store(leaves, max_height=3, doc_span_end=400)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=300, span_end=400, document_id="doc"
        )

        # Left leaf is at index 3 (odd = right child), returns itself
        assert bounds.left_edge_max is not None
        assert bounds.left_edge_max.height == 0
        assert bounds.left_edge_max.level_index == 3

    def test_edge_max_right_walks_up_right_children(self) -> None:
        """Right edge-max walks up through right children only if staying in window."""
        # Window [500, 600) covers only leaf 5
        # Leaf 5's parent (1, 2) covers leaves 4-5, but leaf 4 is outside window
        # So right_edge_max stays at leaf 5 (doesn't climb)
        leaves = [
            MockTreeNode(f"L{i}", "doc", i * 100, (i + 1) * 100, 0, i) for i in range(8)
        ]
        store = self._create_mock_store(leaves, max_height=5, doc_span_end=800)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=500, span_end=600, document_id="doc"
        )

        # Right boundary at index 5 - parent would extend beyond left edge of window
        # So edge-max stays at the leaf itself
        assert bounds.right_edge_max is not None
        assert bounds.right_edge_max.height == 0
        assert bounds.right_edge_max.level_index == 5

    def test_edge_max_right_climbs_when_parent_within_window(self) -> None:
        """Right edge-max climbs when parent stays within window bounds."""
        # Window [400, 600) covers leaves 4-5
        # Leaf 5's parent (1, 2) covers leaves 4-5, both within window
        # So right_edge_max climbs to (1, 2)
        leaves = [
            MockTreeNode(f"L{i}", "doc", i * 100, (i + 1) * 100, 0, i) for i in range(8)
        ]
        store = self._create_mock_store(leaves, max_height=5, doc_span_end=800)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=400, span_end=600, document_id="doc"
        )

        # Right boundary at index 5 - parent (1, 2) covers leaves 4-5
        # Both are within window, so edge-max climbs to parent
        # Parent (1, 2) is a left child (even), so climb stops there
        assert bounds.right_edge_max is not None
        assert bounds.right_edge_max.height == 1
        assert bounds.right_edge_max.level_index == 2

    def test_edge_max_right_at_document_end(self) -> None:
        """Right edge-max at document end climbs within window bounds."""
        # When the window goes to document end, edge-max climbs as high as
        # possible while staying within window bounds.
        leaves = [
            MockTreeNode(f"L{i}", "doc", i * 100, (i + 1) * 100, 0, i) for i in range(8)
        ]
        store = self._create_mock_store(leaves, max_height=5, doc_span_end=800)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=700, span_end=800, document_id="doc"
        )

        # Window covers only leaf 7 (index 7, odd = right child)
        # Parent (1, 3) would cover leaves 6-7, but leaf 6 is outside window
        # So right_edge_max stays at the leaf
        assert bounds.right_edge_max is not None
        assert bounds.right_edge_max.height == 0
        assert bounds.right_edge_max.level_index == 7

    def test_edge_max_right_stops_at_left_child(self) -> None:
        """Right edge-max stops when encountering a left child."""
        # Leaf at index 4 is a left child (even)
        # Should return itself immediately
        leaves = [
            MockTreeNode(f"L{i}", "doc", i * 100, (i + 1) * 100, 0, i) for i in range(8)
        ]
        store = self._create_mock_store(leaves, max_height=3, doc_span_end=800)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=400, span_end=500, document_id="doc"
        )

        # Right leaf is at index 4 (even = left child), returns itself
        assert bounds.right_edge_max is not None
        assert bounds.right_edge_max.height == 0
        assert bounds.right_edge_max.level_index == 4

    def test_single_leaf_window(self) -> None:
        """Window covering a single leaf."""
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
            MockTreeNode("L2", "doc", 200, 300, 0, 2),
        ]
        store = self._create_mock_store(leaves, max_height=2, doc_span_end=300)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=120, span_end=180, document_id="doc"
        )

        # Both boundaries resolve to leaf 1
        assert bounds.actual_start == 100
        assert bounds.actual_end == 200
        assert bounds.left_leaf_index == 1
        assert bounds.right_leaf_index == 1

    def test_raises_on_missing_left_leaf(self) -> None:
        """Raises ValueError when no leaf found at span_start."""
        leaves = [MockTreeNode("L0", "doc", 100, 200, 0, 0)]  # Gap at start
        store = self._create_mock_store(leaves, doc_span_end=200)
        builder = CoverageBuilder(store)

        with pytest.raises(ValueError, match="No leaf found containing position 50"):
            builder.compute_window_bounds(
                span_start=50, span_end=150, document_id="doc"
            )

    def test_raises_on_missing_right_leaf(self) -> None:
        """Raises ValueError when no leaf found at span_end."""
        # Gap between leaves - no leaf at position 150-199
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 200, 300, 0, 1),
        ]
        store = self._create_mock_store(leaves, doc_span_end=None)
        builder = CoverageBuilder(store)

        with pytest.raises(ValueError, match="No leaf found containing position"):
            builder.compute_window_bounds(span_start=0, span_end=180, document_id="doc")

    def test_raises_on_missing_repository(self) -> None:
        """Raises ValueError when node repository is not available."""
        mock_store = MagicMock()
        mock_store.nodes = None
        builder = CoverageBuilder(mock_store)

        with pytest.raises(ValueError, match="Node repository not available"):
            builder.compute_window_bounds(span_start=0, span_end=100, document_id="doc")


class TestBuildWindowedCoverage:
    """Tests for build_windowed_coverage method."""

    def _create_mock_store_with_nodes(
        self,
        nodes: dict[str, MockTreeNode],
        max_height: int = 3,
        doc_span_end: int | None = None,
    ) -> MagicMock:
        """Create a mock store with nodes available for fetching."""
        mock_store = MagicMock()
        mock_store.document_id = "doc"

        # Get leaves for span position lookups
        leaves = [n for n in nodes.values() if n.height == 0]
        leaves.sort(key=lambda n: n.level_index)

        # Compute doc_span_end if not provided
        if doc_span_end is None and leaves:
            doc_span_end = max(leaf.span_end for leaf in leaves)

        # Create mock repository for compute_window_bounds support
        mock_repo = MagicMock()

        def get_leaf_at_position(
            document_id: str, position: int
        ) -> MockTreeNode | None:
            for leaf in leaves:
                if (
                    leaf.document_id == document_id
                    and leaf.span_start <= position < leaf.span_end
                ):
                    return leaf
            return None

        mock_repo.get_leaf_at_span_position = MagicMock(
            side_effect=get_leaf_at_position
        )
        mock_repo.get_document_span_end = MagicMock(return_value=doc_span_end)

        # Mock nodes wrapper
        mock_nodes = MagicMock()
        mock_nodes.max_height = MagicMock(return_value=max_height)
        mock_nodes._repo = mock_repo

        def get_nodes(node_ids: list[str]) -> list[MockTreeNode]:
            return [nodes[nid] for nid in node_ids if nid in nodes]

        mock_nodes.get_nodes = MagicMock(side_effect=get_nodes)

        def get_by_height_levels(coords: list[tuple[int, int]]) -> list[MockTreeNode]:
            result = []
            for node in nodes.values():
                if (node.height, node.level_index) in coords:
                    result.append(node)
            return result

        mock_nodes.get_by_height_levels = MagicMock(side_effect=get_by_height_levels)

        mock_store.nodes = mock_nodes
        return mock_store

    def test_includes_edge_max_nodes_in_coverage(self) -> None:
        """Edge-max nodes are included as synthetic seeds."""
        # Create a simple tree with 4 leaves
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "P0": MockTreeNode("P0", "doc", 0, 200, 1, 0),
            "P1": MockTreeNode("P1", "doc", 200, 400, 1, 1),
            "root": MockTreeNode("root", "doc", 0, 400, 2, 0),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=2)
        builder = CoverageBuilder(store)

        # No seeds from vector search - only edge-max nodes
        # Window covers leaves 1-2 (span [100, 300))
        result = builder.build_windowed_coverage([], "doc", 100, 300)

        # Should include nodes related to edge-max coordinates
        # Edge-max at (0,1) and (0,2) should bring in their ancestors and siblings
        assert len(result.coverage_map) > 0

    def test_filters_out_nodes_outside_window(self) -> None:
        """Nodes outside window are filtered from coverage."""
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "P0": MockTreeNode("P0", "doc", 0, 200, 1, 0),
            "P1": MockTreeNode("P1", "doc", 200, 400, 1, 1),
            "root": MockTreeNode("root", "doc", 0, 400, 2, 0),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=2)
        builder = CoverageBuilder(store)

        # Window covers leaves 1-2 only (span [100, 300))
        result = builder.build_windowed_coverage([], "doc", 100, 300)

        # Nodes outside window should not be in coverage
        # L0 (index 0) is outside window [1,2]
        # L3 (index 3) is outside window [1,2]
        # root spans 0-3, so it's also outside
        covered_ids = set(result.coverage_map.keys())
        assert "L0" not in covered_ids
        assert "L3" not in covered_ids
        # P0 spans leaves 0-1, outside [1,2]
        assert "P0" not in covered_ids

    def test_includes_seeds_and_their_relatives(self) -> None:
        """Seeds from vector search and their ancestors/siblings are included."""
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "P0": MockTreeNode("P0", "doc", 0, 200, 1, 0),
            "P1": MockTreeNode("P1", "doc", 200, 400, 1, 1),
            "root": MockTreeNode("root", "doc", 0, 400, 2, 0),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=2)
        builder = CoverageBuilder(store)

        # L1 is a seed, window covers all leaves [0, 400)
        result = builder.build_windowed_coverage(["L1"], "doc", 0, 400)

        # L1 should be in coverage
        assert "L1" in result.coverage_map
        # With the full window, many relatives should be included

    def test_handles_pinned_ids(self) -> None:
        """Pinned IDs have their ancestors filtered from coverage."""
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "P0": MockTreeNode("P0", "doc", 0, 200, 1, 0),
            "P1": MockTreeNode("P1", "doc", 200, 400, 1, 1),
            "root": MockTreeNode("root", "doc", 0, 400, 2, 0),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=2)
        builder = CoverageBuilder(store)

        # L0 is pinned and also a seed, window covers all leaves [0, 400)
        result = builder.build_windowed_coverage(
            ["L0"], "doc", 0, 400, pinned_ids={"L0"}
        )

        # L0 should be in coverage
        assert "L0" in result.coverage_map

    def test_returns_coverage_result_with_nodes(self) -> None:
        """build_windowed_coverage returns CoverageResult with nodes dict."""
        nodes = {
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=1)
        builder = CoverageBuilder(store)

        result = builder.build_windowed_coverage([], "doc", 100, 300)

        # Result should have both coverage_map and nodes
        assert hasattr(result, "coverage_map")
        assert hasattr(result, "nodes")
        assert isinstance(result.coverage_map, dict)
        assert isinstance(result.nodes, dict)

    def test_fallback_fetch_respects_window_bounds(self) -> None:
        """Fallback fetch for missing seeds respects window bounds.

        The fallback path handles seeds identified via metadata but not found
        through coordinate-based fetch. These should be filtered by window bounds.
        """
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=1)
        builder = CoverageBuilder(store)

        # Provide metadata for L0 (window only covers L1 at [100, 200)) so it goes through coordinate path
        # with coord_version=1 so it's recognized as having coordinates
        seed_metadata: dict[str, dict[str, str | int | float | bool | None]] = {
            "L0": {
                "coord_version": 1,
                "height": 0,
                "level_index": 0,
                "document_id": "doc",
            }
        }

        # Make get_by_height_levels return empty so L0 falls back
        store.nodes.get_by_height_levels = MagicMock(return_value=[])
        # Fallback fetch returns L0
        store.nodes.get_nodes = MagicMock(return_value=[nodes["L0"]])

        result = builder.build_windowed_coverage(
            ["L0"], "doc", 100, 200, seed_metadata=seed_metadata  # L0 is outside window
        )

        # L0 should NOT be in coverage because its span [0,100) is outside
        # window [100,200). The fallback fetch filters by span bounds.
        assert "L0" not in result.coverage_map


class TestWindowBoundsEdgeCases:
    """Edge case tests for window bounds computation."""

    def _create_mock_store(
        self,
        leaves: list[MockTreeNode],
        max_height: int = 3,
        doc_span_end: int | None = None,
    ) -> MagicMock:
        """Create a mock store with configured node repository."""
        mock_store = MagicMock()

        mock_repo = MagicMock()

        def get_leaf_at_position(
            document_id: str, position: int
        ) -> MockTreeNode | None:
            for leaf in leaves:
                if (
                    leaf.document_id == document_id
                    and leaf.span_start <= position < leaf.span_end
                ):
                    return leaf
            return None

        mock_repo.get_leaf_at_span_position = MagicMock(
            side_effect=get_leaf_at_position
        )
        mock_repo.get_document_span_end = MagicMock(return_value=doc_span_end)

        mock_nodes = MagicMock()
        mock_nodes._repo = mock_repo
        mock_nodes.max_height = MagicMock(return_value=max_height)
        mock_store.nodes = mock_nodes

        return mock_store

    def test_window_at_exact_leaf_boundaries(self) -> None:
        """Window request exactly matching leaf boundaries."""
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
        ]
        store = self._create_mock_store(leaves, max_height=1, doc_span_end=200)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=100, span_end=200, document_id="doc"
        )

        assert bounds.actual_start == 100
        assert bounds.actual_end == 200
        assert bounds.left_leaf_index == 1
        assert bounds.right_leaf_index == 1

    def test_window_spanning_full_document(self) -> None:
        """Window covering the entire document."""
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
            MockTreeNode("L2", "doc", 200, 300, 0, 2),
            MockTreeNode("L3", "doc", 300, 400, 0, 3),
        ]
        store = self._create_mock_store(leaves, max_height=2, doc_span_end=400)
        builder = CoverageBuilder(store)

        bounds = builder.compute_window_bounds(
            span_start=0, span_end=400, document_id="doc"
        )

        assert bounds.actual_start == 0
        assert bounds.actual_end == 400
        assert bounds.left_leaf_index == 0
        assert bounds.right_leaf_index == 3
        # For a full document window, edge-max climbs all the way to the root
        # since nothing extends beyond the window.
        assert bounds.left_edge_max is not None
        assert bounds.left_edge_max.height == 2  # root
        assert bounds.left_edge_max.level_index == 0
        assert bounds.right_edge_max is not None
        assert bounds.right_edge_max.height == 2  # root
        assert bounds.right_edge_max.level_index == 0

    def test_adjacent_windows_share_boundary(self) -> None:
        """Two adjacent windows share boundary leaf."""
        leaves = [
            MockTreeNode("L0", "doc", 0, 100, 0, 0),
            MockTreeNode("L1", "doc", 100, 200, 0, 1),
            MockTreeNode("L2", "doc", 200, 300, 0, 2),
        ]
        store = self._create_mock_store(leaves, max_height=2, doc_span_end=300)
        builder = CoverageBuilder(store)

        # First window [0, 200)
        bounds1 = builder.compute_window_bounds(
            span_start=0, span_end=200, document_id="doc"
        )

        # Second window [200, 300)
        bounds2 = builder.compute_window_bounds(
            span_start=200, span_end=300, document_id="doc"
        )

        # Windows should be adjacent without overlap
        assert bounds1.actual_end == 200
        assert bounds2.actual_start == 200
        assert bounds1.right_leaf_index == 1
        assert bounds2.left_leaf_index == 2


class TestWindowedCoverageBugs:
    """Tests exposing bugs in windowed coverage implementation.

    These tests document missing functionality that should exist:
    1. Store pinned nodes missing from windowed coverage
    2. Roots missing when edge_max is None at document boundaries
    3. Edge-max extending beyond window and getting filtered out
    """

    def _create_mock_store_with_nodes(
        self,
        nodes: dict[str, MockTreeNode],
        max_height: int = 3,
        pinned_nodes: list[MockTreeNode] | None = None,
    ) -> MagicMock:
        """Create a mock store with nodes and optional pinned nodes."""
        mock_store = MagicMock()
        mock_store.document_id = "doc"
        mock_store.PIN_DEPTH_MAX = 2

        # Get leaves for span position lookups
        leaves = [n for n in nodes.values() if n.height == 0]
        leaves.sort(key=lambda n: n.level_index)

        # Compute doc_span_end from leaves
        doc_span_end = max(leaf.span_end for leaf in leaves) if leaves else 0

        # Create mock repository for compute_window_bounds support
        mock_repo = MagicMock()

        def get_leaf_at_position(
            document_id: str, position: int
        ) -> MockTreeNode | None:
            for leaf in leaves:
                if (
                    leaf.document_id == document_id
                    and leaf.span_start <= position < leaf.span_end
                ):
                    return leaf
            return None

        mock_repo.get_leaf_at_span_position = MagicMock(
            side_effect=get_leaf_at_position
        )
        mock_repo.get_document_span_end = MagicMock(return_value=doc_span_end)

        # Mock nodes wrapper
        mock_nodes = MagicMock()
        mock_nodes.max_height = MagicMock(return_value=max_height)
        mock_nodes._repo = mock_repo

        def get_nodes(node_ids: list[str]) -> list[MockTreeNode]:
            return [nodes[nid] for nid in node_ids if nid in nodes]

        mock_nodes.get_nodes = MagicMock(side_effect=get_nodes)

        def get_by_height_levels(coords: list[tuple[int, int]]) -> list[MockTreeNode]:
            result = []
            for node in nodes.values():
                if (node.height, node.level_index) in coords:
                    result.append(node)
            return result

        mock_nodes.get_by_height_levels = MagicMock(side_effect=get_by_height_levels)

        # Mock get_root_nodes for complete coverage path
        def get_root_nodes() -> list[MockTreeNode]:
            return [n for n in nodes.values() if n.height == max_height]

        mock_nodes.get_root_nodes = MagicMock(side_effect=get_root_nodes)

        mock_store.nodes = mock_nodes

        # Mock get_pinned_nodes on store
        if pinned_nodes is not None:
            mock_store.get_pinned_nodes = MagicMock(return_value=pinned_nodes)

        return mock_store

    def test_bug1_windowed_coverage_missing_store_pinned_nodes(self) -> None:
        """BUG: build_windowed_coverage doesn't include store pinned nodes.

        build_complete_coverage adds pinned nodes from the store (lines 87-107),
        but build_windowed_coverage does not. This is a bug because pinned nodes
        should be included in all coverage scenarios.
        """
        # Tree: 4 leaves, 2 parents, 1 root
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "P0": MockTreeNode("P0", "doc", 0, 200, 1, 0),
            "P1": MockTreeNode("P1", "doc", 200, 400, 1, 1),
            "root": MockTreeNode("root", "doc", 0, 400, 2, 0),
        }

        # L1 is a pinned node (within window)
        pinned_l1 = nodes["L1"]
        store = self._create_mock_store_with_nodes(
            nodes, max_height=2, pinned_nodes=[pinned_l1]
        )
        builder = CoverageBuilder(store)

        # No seeds - only pinned nodes should appear (window covers all leaves [0, 400))
        result = builder.build_windowed_coverage([], "doc", 0, 400)

        # L1 (pinned node) should be in coverage
        assert "L1" in result.coverage_map, (
            "Pinned node L1 should be in windowed coverage. "
            "build_windowed_coverage must add store pinned nodes."
        )

    def test_full_document_window_includes_root(self) -> None:
        """Full document window should include root via edge-max.

        When the window covers the entire document, highest_ancestor_within_window
        climbs all the way to the root since no boundary is exceeded.
        """
        # Tree: 4 leaves, root at height 2
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "P0": MockTreeNode("P0", "doc", 0, 200, 1, 0),
            "P1": MockTreeNode("P1", "doc", 200, 400, 1, 1),
            "root": MockTreeNode("root", "doc", 0, 400, 2, 0),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=2)
        builder = CoverageBuilder(store)

        # No seeds - coverage should include root via edge-max (full document window)
        result = builder.build_windowed_coverage([], "doc", 0, 400)

        # Root should be in coverage via edge-max
        assert "root" in result.coverage_map, (
            "Root should be in coverage for full document window. "
            "Edge-max should climb to root when window covers full document."
        )

    def test_edge_max_stays_within_window(self) -> None:
        """Edge-max computation should stay within window bounds.

        With highest_ancestor_within_window, the edge-max climbs up the tree
        but stops before it would extend beyond the window's OTHER edge.

        For a window covering leaves 0-3 on a tree with 8 leaves:
        - Left edge-max for leaf 0: climbs to Q0123 at (2, 0)
          - Root at (3, 0) would cover leaves 0-7, exceeding window
        - Right edge-max for leaf 3: climbs to Q0123 at (2, 0)
          - Leaf 3 is a right child, parent P23 at (1, 1) covers leaves 2-3
          - P23's parent Q0123 at (2, 0) covers leaves 0-3, exactly the window
        """
        # 8 leaves (indices 0-7), window covers only leaves 0-3
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "L4": MockTreeNode("L4", "doc", 400, 500, 0, 4),
            "L5": MockTreeNode("L5", "doc", 500, 600, 0, 5),
            "L6": MockTreeNode("L6", "doc", 600, 700, 0, 6),
            "L7": MockTreeNode("L7", "doc", 700, 800, 0, 7),
            # Height 1: pairs of leaves
            "P01": MockTreeNode("P01", "doc", 0, 200, 1, 0),
            "P23": MockTreeNode("P23", "doc", 200, 400, 1, 1),
            "P45": MockTreeNode("P45", "doc", 400, 600, 1, 2),
            "P67": MockTreeNode("P67", "doc", 600, 800, 1, 3),
            # Height 2: pairs of height-1 nodes
            "Q0123": MockTreeNode("Q0123", "doc", 0, 400, 2, 0),
            "Q4567": MockTreeNode("Q4567", "doc", 400, 800, 2, 1),
            # Height 3: root
            "root": MockTreeNode("root", "doc", 0, 800, 3, 0),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=3)
        builder = CoverageBuilder(store)

        # No seeds, window covers leaves 0-3 (first half of document)
        # Edge-max climbs to Q0123 (2, 0) which covers exactly leaves 0-3
        result = builder.build_windowed_coverage([], "doc", 0, 400)

        # Q0123 is added as edge-max, covering leaves 0-3 (exactly the window).
        # Its sibling Q4567 covers leaves 4-7, which is outside window and filtered.
        # Note: edge-max doesn't auto-include descendants - those come from seeds.

        # Coverage should include the edge-max node
        assert "Q0123" in result.coverage_map, (
            "Q0123 should be in coverage as edge-max. "
            "It covers leaves 0-3 which matches the window exactly."
        )

    def test_pinned_nodes_outside_window_are_filtered(self) -> None:
        """Pinned nodes outside window bounds should NOT be in coverage.

        When a pinned node's span falls outside the window, it should be
        filtered out of coverage. This test ensures the filtering logic
        works correctly (once pinned nodes are added to windowed coverage).
        """
        # Tree: 4 leaves
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "P0": MockTreeNode("P0", "doc", 0, 200, 1, 0),
            "P1": MockTreeNode("P1", "doc", 200, 400, 1, 1),
            "root": MockTreeNode("root", "doc", 0, 400, 2, 0),
        }

        # L0 is pinned but OUTSIDE the window
        pinned_l0 = nodes["L0"]
        store = self._create_mock_store_with_nodes(
            nodes, max_height=2, pinned_nodes=[pinned_l0]
        )
        builder = CoverageBuilder(store)

        # Window covers only leaves 2-3 [200, 400)
        result = builder.build_windowed_coverage([], "doc", 200, 400)

        # L0 (pinned) is at [0, 100), window is [200, 400)
        # L0 should be filtered out
        assert "L0" not in result.coverage_map, (
            "Pinned node L0 should be filtered out because its span [0, 100) "
            "is outside the window [200, 400)."
        )

    def test_roots_outside_window_are_filtered(self) -> None:
        """Roots that extend beyond window bounds should be filtered.

        This confirms that the window filtering correctly excludes nodes
        whose span exceeds the window, even if they are roots.
        """
        # Tree: 8 leaves, window covers leaves 2-5 (middle portion)
        nodes = {
            "L0": MockTreeNode("L0", "doc", 0, 100, 0, 0),
            "L1": MockTreeNode("L1", "doc", 100, 200, 0, 1),
            "L2": MockTreeNode("L2", "doc", 200, 300, 0, 2),
            "L3": MockTreeNode("L3", "doc", 300, 400, 0, 3),
            "L4": MockTreeNode("L4", "doc", 400, 500, 0, 4),
            "L5": MockTreeNode("L5", "doc", 500, 600, 0, 5),
            "L6": MockTreeNode("L6", "doc", 600, 700, 0, 6),
            "L7": MockTreeNode("L7", "doc", 700, 800, 0, 7),
            # Height 1
            "P01": MockTreeNode("P01", "doc", 0, 200, 1, 0),
            "P23": MockTreeNode("P23", "doc", 200, 400, 1, 1),
            "P45": MockTreeNode("P45", "doc", 400, 600, 1, 2),
            "P67": MockTreeNode("P67", "doc", 600, 800, 1, 3),
            # Height 2
            "Q0123": MockTreeNode("Q0123", "doc", 0, 400, 2, 0),
            "Q4567": MockTreeNode("Q4567", "doc", 400, 800, 2, 1),
            # Height 3
            "root": MockTreeNode("root", "doc", 0, 800, 3, 0),
        }
        store = self._create_mock_store_with_nodes(nodes, max_height=3)
        builder = CoverageBuilder(store)

        # Window covers leaves 2-5 (middle of document, neither at doc start nor doc end)
        result = builder.build_windowed_coverage([], "doc", 200, 600)

        # Root covers leaves 0-7, window is 2-5
        # is_within_leaf_range(2, 5): root.leaf_span() = (0, 7)
        # 0 >= 2 is False, so root should be filtered
        assert "root" not in result.coverage_map, (
            "Root should be filtered because it covers leaves 0-7 but "
            "window only covers leaves 2-5."
        )

        # Q0123 covers leaves 0-3, window is 2-5
        # 0 >= 2 is False, so Q0123 should be filtered
        assert "Q0123" not in result.coverage_map, (
            "Q0123 should be filtered because it covers leaves 0-3 but "
            "window starts at leaf 2."
        )

        # Q4567 covers leaves 4-7, window is 2-5
        # 7 <= 5 is False, so Q4567 should be filtered
        assert "Q4567" not in result.coverage_map, (
            "Q4567 should be filtered because it covers leaves 4-7 but "
            "window ends at leaf 5."
        )

    def test_edge_max_enqueued_fills_gap_between_edge_and_seed(self) -> None:
        """Edge-max must be enqueued (not just appended) to fill gaps.

        When a seed is far from the window edge, there's a gap between
        the edge-max coverage and the seed coverage. This gap is filled
        by siblings of edge-max's ancestors.

        Example:
        - Window covers leaves 2-15
        - Seed at leaf 12
        - Left edge-max at leaf 2 (or small ancestor)

        Seed's ancestors/siblings:
        - H1 [12,13], sibling [14,15]
        - H2 [12,15], sibling [8,11]
        - H3 [8,15], sibling [0,7] <- filtered (extends past leaf 2)

        Without enqueuing edge-max, leaves 3-7 have no coverage!

        With enqueuing edge-max (leaf 2):
        - H1 [2,3], sibling [0,1] <- filtered
        - H2 [0,3], sibling [4,7] <- ancestor filtered, sibling KEPT!
        - H3 [0,7], sibling [8,15] <- ancestor filtered, sibling KEPT!

        The siblings [4,7] and [8,15] fill the gap.
        """
        # 16 leaves, window covers leaves 2-15
        nodes: dict[str, MockTreeNode] = {}

        # Create leaves 0-15
        for i in range(16):
            nodes[f"L{i}"] = MockTreeNode(f"L{i}", "doc", i * 100, (i + 1) * 100, 0, i)

        # Height 1: pairs [0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]
        for i in range(8):
            start_leaf = i * 2
            nodes[f"H1_{i}"] = MockTreeNode(
                f"H1_{i}", "doc", start_leaf * 100, (start_leaf + 2) * 100, 1, i
            )

        # Height 2: quads [0-3], [4-7], [8-11], [12-15]
        for i in range(4):
            start_leaf = i * 4
            nodes[f"H2_{i}"] = MockTreeNode(
                f"H2_{i}", "doc", start_leaf * 100, (start_leaf + 4) * 100, 2, i
            )

        # Height 3: octets [0-7], [8-15]
        for i in range(2):
            start_leaf = i * 8
            nodes[f"H3_{i}"] = MockTreeNode(
                f"H3_{i}", "doc", start_leaf * 100, (start_leaf + 8) * 100, 3, i
            )

        # Height 4: root [0-15]
        nodes["root"] = MockTreeNode("root", "doc", 0, 1600, 4, 0)

        store = self._create_mock_store_with_nodes(nodes, max_height=4)
        builder = CoverageBuilder(store)

        # Seed at leaf 12 - far from left edge
        # Window covers leaves 2-15 (not at doc start, at doc end)
        # Left edge-max is H1_1 [2,3] - can't climb higher without exceeding window
        result = builder.build_windowed_coverage(["L12"], "doc", 200, 1600)

        # Seed L12 should be in coverage
        assert "L12" in result.coverage_map

        # Edge-max H1_1 [2,3] should be in coverage
        assert "H1_1" in result.coverage_map, "Edge-max H1_1 should be in coverage"

        # KEY ASSERTION: H2_1 [4,7] should be in coverage
        # This is the sibling of edge-max's ancestor H2_0 [0,3]
        # H2_0 gets filtered (extends to leaf 0), but H2_1 is within window
        # This only works if edge-max is ENQUEUED (not just appended)
        assert "H2_1" in result.coverage_map, (
            "H2_1 [4,7] should be in coverage to fill gap between edge-max and seed. "
            "Edge-max must be enqueued (not just appended) so that siblings of its "
            "ancestors are added. H2_1 is the sibling of H2_0 (edge-max's ancestor)."
        )

        # H3_1 [8,15] should also be in coverage (sibling of H3_0)
        assert "H3_1" in result.coverage_map, (
            "H3_1 [8,15] should be in coverage. It's the sibling of H3_0 "
            "(ancestor of edge-max that gets filtered)."
        )
