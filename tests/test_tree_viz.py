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
            depth=2,
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
            depth=1,
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
            depth=1,
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
                depth=0,
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

        # Build visualization
        viz = build_ascii_tree(segments, store, "doc1", width=40)

        # Check basic structure
        assert "Document span: 0-100" in viz
        assert "Level 2:" in viz
        assert "Level 1:" in viz
        assert "Level 0:" in viz

        # Check that selected segments are labeled
        assert "root-L" in viz
        assert "leaf3" in viz
        assert "leaf4" in viz

    def test_empty_segments(self):
        """Test visualization with no selected segments."""
        store = SimpleMockStore()

        # Add a single node
        store.add_node(
            node_id="root",
            text="Root",
            depth=0,
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
        assert "Document span: 0-100" in viz
        assert "Level 0:" in viz

    def test_no_nodes_for_document(self):
        """Test visualization when document has no nodes."""
        store = SimpleMockStore()
        segments = []

        viz = build_ascii_tree(segments, store, "nonexistent", width=40)

        assert viz == "No nodes found for document"
