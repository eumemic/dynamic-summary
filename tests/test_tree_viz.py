"""Tests for tree visualization functionality."""

from ragzoom.dynamic_frontier import Segment
from ragzoom.tree_viz import build_ascii_tree
from tests.mock_store import SimpleMockStore


class TestTreeVisualization:
    """Test tree visualization functionality."""

    def test_basic_tree_visualization(self):
        """Test basic tree visualization with selected segments."""
        # Create a mock store with a simple tree
        store = SimpleMockStore()

        # Root node
        store.add_node(
            node_id="root",
            text="Root left <<<MID>>> Root right",
            span_start=0,
            span_end=100,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 384,
            mid_offset=10,
            left_child_id="left",
            right_child_id="right",
        )

        # Left child
        store.add_node(
            node_id="left",
            text="Left content",
            span_start=0,
            span_end=50,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 384,
            left_child_id="leaf1",
            right_child_id="leaf2",
        )

        # Right child
        store.add_node(
            node_id="right",
            text="Right content",
            span_start=50,
            span_end=100,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 384,
            left_child_id="leaf3",
            right_child_id="leaf4",
        )

        # Leaf nodes
        for i, (node_id, start, end, parent) in enumerate(
            [
                ("leaf1", 0, 25, "left"),
                ("leaf2", 25, 50, "left"),
                ("leaf3", 50, 75, "right"),
                ("leaf4", 75, 100, "right"),
            ]
        ):
            store.add_node(
                node_id=node_id,
                text=f"Leaf {i} text",
                span_start=start,
                span_end=end,
                parent_id=parent,
                document_id="doc1",
                embedding=[0.5] * 384,
            )

        # Create segments for visualization
        segments = [
            Segment(node_id="root", side="LEFT"),
            Segment(node_id="leaf3", side=None),
            Segment(node_id="leaf4", side=None),
        ]

        # Build visualization with coverage map
        coverage_map = {"root": True, "left": True, "leaf1": True, "leaf2": True}
        viz = build_ascii_tree(
            segments, store, "doc1", width=40, coverage_map=coverage_map
        )

        # Check basic structure
        assert "H2 " in viz
        assert "H1 " in viz
        assert "H0 " in viz

        # Check that selected segments are labeled with indices
        assert "0" in viz  # First segment (root-L)
        assert "1" in viz  # Second segment (leaf3)
        assert "2" in viz  # Third segment (leaf4)

    def test_empty_segments(self):
        """Test visualization with no selected segments."""
        store = SimpleMockStore()

        # Add a single node
        store.add_node(
            node_id="root",
            text="Root",
            span_start=0,
            span_end=100,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 384,
        )

        # Empty segments list
        segments = []

        viz = build_ascii_tree(segments, store, "doc1", width=40)

        # Should still show document structure
        assert "H0 " in viz

    def test_no_nodes_for_document(self):
        """Test visualization when document has no nodes."""
        store = SimpleMockStore()
        segments = []

        viz = build_ascii_tree(segments, store, "nonexistent", width=40)

        assert viz == "No nodes found for document"

    def test_coverage_visualization(self):
        """Test visualization with coverage map showing covered but not selected nodes."""
        # Create a mock store with a simple tree
        store = SimpleMockStore()

        # Root node
        store.add_node(
            node_id="root",
            text="Root left <<<MID>>> Root right",
            span_start=0,
            span_end=100,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 384,
            mid_offset=10,
            left_child_id="leaf1",
            right_child_id="leaf2",
        )

        # Leaf nodes
        store.add_node(
            node_id="leaf1",
            text="Leaf 1 text",
            span_start=0,
            span_end=50,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 384,
        )

        store.add_node(
            node_id="leaf2",
            text="Leaf 2 text",
            span_start=50,
            span_end=100,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 384,
        )

        # Only the left segment of root is selected
        segments = [
            Segment(node_id="root", side="LEFT"),
            Segment(node_id="leaf2", side=None),
        ]

        # Coverage map includes root and leaf1 (covered but not selected)
        coverage_map = {"root": True, "leaf1": True, "leaf2": True}

        viz = build_ascii_tree(
            segments, store, "doc1", width=60, coverage_map=coverage_map
        )

        # Check that the visualization includes all expected elements
        assert "0" in viz  # First segment (root-L)
        assert "1" in viz  # Second segment (leaf2)
        # The covered but not selected leaf1 should be shown with ░ characters
