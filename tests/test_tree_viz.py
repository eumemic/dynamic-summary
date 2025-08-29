"""Tests for tree visualization functionality."""

import pytest

from ragzoom.tree_viz import build_ascii_tree
from tests.mock_store import SimpleMockStore


@pytest.mark.skip_ci
class TestTreeVisualization:
    """Test tree visualization functionality."""

    def test_basic_tree_visualization(self):
        """Test basic tree visualization with selected nodes."""
        # Create a mock store with a simple tree
        store = SimpleMockStore()

        # Root node (no mid_offset in new design)
        store.add_node(
            node_id="root",
            text="Root summary",
            span_start=0,
            span_end=100,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="left",
            right_child_id="right",
            height=2,
        )

        # Left child
        store.add_node(
            node_id="left",
            text="Left content",
            span_start=0,
            span_end=50,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="leaf1",
            right_child_id="leaf2",
            height=1,
        )

        # Right child
        store.add_node(
            node_id="right",
            text="Right content",
            span_start=50,
            span_end=100,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="leaf3",
            right_child_id="leaf4",
            height=1,
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
                embedding=[0.5] * 1536,
                height=0,  # Leaf nodes have height 0
            )

        # Create tiling (list of node IDs)
        tiling = ["left", "leaf3", "leaf4"]

        # Build visualization with coverage map
        coverage_map = {
            "root": True,
            "left": True,
            "leaf1": True,
            "leaf2": True,
            "leaf3": True,
            "leaf4": True,
        }

        # Preload nodes as the production code does
        preloaded_nodes = {}
        for node_id in coverage_map:
            node = store.nodes.get_node(node_id)
            if node:
                preloaded_nodes[node_id] = node

        doc_store = store.for_document("doc1")
        viz = build_ascii_tree(
            tiling,
            doc_store,
            width=40,
            coverage_map=coverage_map,
            preloaded_nodes=preloaded_nodes,
        )

        # Check basic structure
        assert "H2 " in viz
        assert "H1 " in viz
        assert "H0 " in viz

        # Check that selected nodes are labeled with indices
        assert "0" in viz  # First node (left)
        assert "1" in viz  # Second node (leaf3)
        assert "2" in viz  # Third node (leaf4)

    def test_empty_tiling(self):
        """Test visualization with no selected nodes."""
        store = SimpleMockStore()

        # Add a single node
        store.add_node(
            node_id="root",
            text="Root",
            span_start=0,
            span_end=100,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 1536,
        )

        # Empty tiling list
        tiling = []

        doc_store = store.for_document("doc1")
        viz = build_ascii_tree(tiling, doc_store, width=40)

        # Should still show document structure
        assert "H0 " in viz

    def test_no_nodes_for_document(self):
        """Test visualization when document has no nodes."""
        store = SimpleMockStore()
        tiling = []

        doc_store = store.for_document("nonexistent")
        viz = build_ascii_tree(tiling, doc_store, width=40)

        assert viz == "No nodes found for document"

    def test_coverage_visualization(self):
        """Test visualization with coverage map showing covered but not selected nodes."""
        # Create a mock store with a simple tree
        store = SimpleMockStore()

        # Root node
        store.add_node(
            node_id="root",
            text="Root summary",
            span_start=0,
            span_end=100,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 1536,
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
            embedding=[0.5] * 1536,
        )

        store.add_node(
            node_id="leaf2",
            text="Leaf 2 text",
            span_start=50,
            span_end=100,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
        )

        # Only leaf2 is selected
        tiling = ["leaf2"]

        # Coverage map includes all nodes
        coverage_map = {"root": True, "leaf1": True, "leaf2": True}

        # Preload nodes as the production code does
        preloaded_nodes = {}
        for node_id in coverage_map:
            node = store.nodes.get_node(node_id)
            if node:
                preloaded_nodes[node_id] = node

        doc_store = store.for_document("doc1")
        viz = build_ascii_tree(
            tiling,
            doc_store,
            width=60,
            coverage_map=coverage_map,
            preloaded_nodes=preloaded_nodes,
        )

        # Check that the visualization includes expected elements
        assert "0" in viz  # First node (leaf2)
        # The covered but not selected nodes should be shown with ░ characters

    def test_mixed_height_tiling(self):
        """Test visualization with nodes at different heights."""
        store = SimpleMockStore()

        # Create a deeper tree
        # Root (H3)
        store.add_node(
            node_id="root",
            text="Root",
            span_start=0,
            span_end=100,
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="l1",
            right_child_id="r1",
            height=3,
        )

        # Height 2
        store.add_node(
            node_id="l1",
            text="L1",
            span_start=0,
            span_end=50,
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="l2",
            right_child_id="r2",
            height=2,
        )

        store.add_node(
            node_id="r1",
            text="R1",
            span_start=50,
            span_end=100,
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="l3",
            right_child_id="r3",
            height=2,
        )

        # Height 1
        for node_id, start, end in [
            ("l2", 0, 25),
            ("r2", 25, 50),
            ("l3", 50, 75),
            ("r3", 75, 100),
        ]:
            store.add_node(
                node_id=node_id,
                text=node_id,
                span_start=start,
                span_end=end,
                document_id="doc1",
                embedding=[0.5] * 1536,
                left_child_id=f"{node_id}_l",
                right_child_id=f"{node_id}_r",
                height=1,
            )

        # Height 0 (leaves)
        for i, (node_id, start, end) in enumerate(
            [
                ("l2_l", 0, 12),
                ("l2_r", 12, 25),
                ("r2_l", 25, 37),
                ("r2_r", 37, 50),
                ("l3_l", 50, 62),
                ("l3_r", 62, 75),
                ("r3_l", 75, 87),
                ("r3_r", 87, 100),
            ]
        ):
            store.add_node(
                node_id=node_id,
                text=f"Leaf {i}",
                span_start=start,
                span_end=end,
                document_id="doc1",
                embedding=[0.5] * 1536,
                height=0,
            )

        # Mixed height tiling
        tiling = ["l1", "l3", "r3_r"]

        doc_store = store.for_document("doc1")
        viz = build_ascii_tree(tiling, doc_store, width=80)

        # Should show all heights
        assert "H3 " in viz
        assert "H2 " in viz
        assert "H1 " in viz
        assert "H0 " in viz

        # Check labels
        assert "0" in viz  # l1
        assert "1" in viz  # l3
        assert "2" in viz  # r3_r
