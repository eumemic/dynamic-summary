"""SQLite-based tests for tree visualization functionality.

SQLite-based tests for ASCII tree visualization functionality
with the real in-memory SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.document_store import DocumentStore
from ragzoom.tree_viz import build_ascii_tree


@pytest.mark.skip_ci
@pytest.mark.usefixtures("sqlite_backend")
class TestTreeVisualizationSQLite:
    """Test tree visualization functionality."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("doc1")

    def test_basic_tree_visualization(self, doc_store: DocumentStore) -> None:
        """Test basic tree visualization with selected nodes."""
        # Create a tree structure with proper parent references
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Root node
            {
                "node_id": "root",
                "text": "Root summary",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 100,
                "height": 2,
                "left_child_id": "left",
                "right_child_id": "right",
            },
            # Left child
            {
                "node_id": "left",
                "text": "Left content",
                "embedding": [],
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 50,
                "height": 1,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
                "parent_id": "root",
            },
            # Right child
            {
                "node_id": "right",
                "text": "Right content",
                "embedding": [],
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 50,
                "height": 1,
                "left_child_id": "leaf3",
                "right_child_id": "leaf4",
                "parent_id": "root",
            },
            # Leaf nodes
            {
                "node_id": "leaf1",
                "text": "Leaf 0 text",
                "embedding": [],
                "span_start": 0,
                "span_end": 25,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "parent_id": "left",
            },
            {
                "node_id": "leaf2",
                "text": "Leaf 1 text",
                "embedding": [],
                "span_start": 25,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "parent_id": "left",
            },
            {
                "node_id": "leaf3",
                "text": "Leaf 2 text",
                "embedding": [],
                "span_start": 50,
                "span_end": 75,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "parent_id": "right",
            },
            {
                "node_id": "leaf4",
                "text": "Leaf 3 text",
                "embedding": [],
                "span_start": 75,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 25,
                "height": 0,
                "parent_id": "right",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("left", "root"),
                ("right", "root"),
                ("leaf1", "left"),
                ("leaf2", "left"),
                ("leaf3", "right"),
                ("leaf4", "right"),
            ]
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
            node = doc_store.nodes.get_node(node_id)
            if node:
                preloaded_nodes[node_id] = node

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

    def test_node_with_only_left_child(self, doc_store: DocumentStore) -> None:
        """Test visualization of nodes with only a left child (document boundary case)."""
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Root node
            {
                "node_id": "root",
                "text": "Root summary",
                "embedding": [],
                "span_start": 0,
                "span_end": 150,
                "document_id": "doc1",
                "token_count": 150,
                "height": 2,
                "left_child_id": "left",
                "right_child_id": "right",
            },
            # Left subtree (complete)
            {
                "node_id": "left",
                "text": "Left content",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 100,
                "height": 1,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
                "parent_id": "root",
            },
            # Right subtree (only left child - document boundary)
            {
                "node_id": "right",
                "text": "Right content",
                "embedding": [],
                "span_start": 100,
                "span_end": 150,
                "document_id": "doc1",
                "token_count": 50,
                "height": 1,
                "left_child_id": "leaf3",
                "right_child_id": None,  # No right child - document boundary
                "parent_id": "root",
            },
            # Leaf nodes
            {
                "node_id": "leaf1",
                "text": "Leaf 1",
                "embedding": [],
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 50,
                "height": 0,
                "parent_id": "left",
            },
            {
                "node_id": "leaf2",
                "text": "Leaf 2",
                "embedding": [],
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 50,
                "height": 0,
                "parent_id": "left",
            },
            {
                "node_id": "leaf3",
                "text": "Leaf 3",
                "embedding": [],
                "span_start": 100,
                "span_end": 150,
                "document_id": "doc1",
                "token_count": 50,
                "height": 0,
                "parent_id": "right",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("left", "root"),
                ("right", "root"),
                ("leaf1", "left"),
                ("leaf2", "left"),
                ("leaf3", "right"),
            ]
        )

        # Tiling includes the node with only left child
        tiling = ["left", "right"]  # "right" has only left child

        # Coverage map
        coverage_map = {
            "root": True,
            "left": True,
            "right": True,
            "leaf1": True,
            "leaf2": True,
            "leaf3": True,
        }

        # Preload nodes
        preloaded_nodes = {}
        for node_id in coverage_map:
            node = doc_store.nodes.get_node(node_id)
            if node:
                preloaded_nodes[node_id] = node

        viz = build_ascii_tree(
            tiling,
            doc_store,
            width=40,
            coverage_map=coverage_map,
            preloaded_nodes=preloaded_nodes,
        )

        # Both selected nodes should appear
        lines = viz.split("\n")
        h1_line = None
        for line in lines:
            if line.startswith("H1 "):
                h1_line = line[3:]  # Remove "H1 " prefix
                break

        assert h1_line is not None, "H1 line not found"

        # Check H1 line to see if both nodes are visualized
        # "left" spans 0-100, "right" spans 100-150
        # With width 40, we expect:
        # - "left" should occupy roughly 0-27 (100/150 * 40)
        # - "right" should occupy roughly 27-40 (50/150 * 40)

        # Since the two nodes are adjacent (left: 0-100, right: 100-150),
        # they will appear as one continuous filled region
        filled_blocks = h1_line.count("█")

        # The filled region should cover most of the width
        # (both nodes together span the entire document)
        assert (
            filled_blocks > 20
        ), f"Expected significant filled region for both nodes, got {filled_blocks} blocks. H1 line: {h1_line}"

        # Check that both nodes are labeled
        assert "0" in viz, "First selected node (left) not labeled"
        assert (
            "1" in viz
        ), "Second selected node (right with only left child) not labeled"

    def test_empty_tiling(self, doc_store: DocumentStore) -> None:
        """Test visualization with no selected nodes."""
        # Add a single node
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "root",
                "text": "Root",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 100,
                "height": 0,
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Empty tiling list
        tiling: list[str] = []

        viz = build_ascii_tree(tiling, doc_store, width=40)

        # Should still show document structure
        assert "H0 " in viz

    def test_no_nodes_for_document(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        """Test visualization when document has no nodes."""
        doc_store = sqlite_store_factory("nonexistent")
        tiling: list[str] = []

        viz = build_ascii_tree(tiling, doc_store, width=40)

        assert viz == "No nodes found for document"

    def test_coverage_visualization(self, doc_store: DocumentStore) -> None:
        """Test visualization with coverage map showing covered but not selected nodes."""
        # Create a simple tree
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Root node
            {
                "node_id": "root",
                "text": "Root summary",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 100,
                "height": 1,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
            },
            # Leaf nodes
            {
                "node_id": "leaf1",
                "text": "Leaf 1 text",
                "embedding": [],
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 50,
                "height": 0,
                "parent_id": "root",
            },
            {
                "node_id": "leaf2",
                "text": "Leaf 2 text",
                "embedding": [],
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 50,
                "height": 0,
                "parent_id": "root",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("leaf1", "root"), ("leaf2", "root")]
        )

        # Only leaf2 is selected
        tiling = ["leaf2"]

        # Coverage map includes all nodes
        coverage_map = {"root": True, "leaf1": True, "leaf2": True}

        # Preload nodes as the production code does
        preloaded_nodes = {}
        for node_id in coverage_map:
            node = doc_store.nodes.get_node(node_id)
            if node:
                preloaded_nodes[node_id] = node

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

    def test_mixed_height_tiling(self, doc_store: DocumentStore) -> None:
        """Test visualization with nodes at different heights."""
        # Create a deeper tree structure
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Root (H3)
            {
                "node_id": "root",
                "text": "Root",
                "embedding": [],
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 100,
                "height": 3,
                "left_child_id": "l1",
                "right_child_id": "r1",
            },
            # Height 2
            {
                "node_id": "l1",
                "text": "L1",
                "embedding": [],
                "span_start": 0,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 50,
                "height": 2,
                "left_child_id": "l2",
                "right_child_id": "r2",
                "parent_id": "root",
            },
            {
                "node_id": "r1",
                "text": "R1",
                "embedding": [],
                "span_start": 50,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 50,
                "height": 2,
                "left_child_id": "l3",
                "right_child_id": "r3",
                "parent_id": "root",
            },
            # Height 1
            {
                "node_id": "l2",
                "text": "l2",
                "embedding": [],
                "span_start": 0,
                "span_end": 25,
                "document_id": "doc1",
                "token_count": 25,
                "height": 1,
                "left_child_id": "l2_l",
                "right_child_id": "l2_r",
                "parent_id": "l1",
            },
            {
                "node_id": "r2",
                "text": "r2",
                "embedding": [],
                "span_start": 25,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 25,
                "height": 1,
                "left_child_id": "r2_l",
                "right_child_id": "r2_r",
                "parent_id": "l1",
            },
            {
                "node_id": "l3",
                "text": "l3",
                "embedding": [],
                "span_start": 50,
                "span_end": 75,
                "document_id": "doc1",
                "token_count": 25,
                "height": 1,
                "left_child_id": "l3_l",
                "right_child_id": "l3_r",
                "parent_id": "r1",
            },
            {
                "node_id": "r3",
                "text": "r3",
                "embedding": [],
                "span_start": 75,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 25,
                "height": 1,
                "left_child_id": "r3_l",
                "right_child_id": "r3_r",
                "parent_id": "r1",
            },
            # Height 0 (leaves) - only create the ones we reference
            {
                "node_id": "l2_l",
                "text": "Leaf 0",
                "embedding": [],
                "span_start": 0,
                "span_end": 12,
                "document_id": "doc1",
                "token_count": 12,
                "height": 0,
                "parent_id": "l2",
            },
            {
                "node_id": "l2_r",
                "text": "Leaf 1",
                "embedding": [],
                "span_start": 12,
                "span_end": 25,
                "document_id": "doc1",
                "token_count": 13,
                "height": 0,
                "parent_id": "l2",
            },
            {
                "node_id": "r2_l",
                "text": "Leaf 2",
                "embedding": [],
                "span_start": 25,
                "span_end": 37,
                "document_id": "doc1",
                "token_count": 12,
                "height": 0,
                "parent_id": "r2",
            },
            {
                "node_id": "r2_r",
                "text": "Leaf 3",
                "embedding": [],
                "span_start": 37,
                "span_end": 50,
                "document_id": "doc1",
                "token_count": 13,
                "height": 0,
                "parent_id": "r2",
            },
            {
                "node_id": "l3_l",
                "text": "Leaf 4",
                "embedding": [],
                "span_start": 50,
                "span_end": 62,
                "document_id": "doc1",
                "token_count": 12,
                "height": 0,
                "parent_id": "l3",
            },
            {
                "node_id": "l3_r",
                "text": "Leaf 5",
                "embedding": [],
                "span_start": 62,
                "span_end": 75,
                "document_id": "doc1",
                "token_count": 13,
                "height": 0,
                "parent_id": "l3",
            },
            {
                "node_id": "r3_l",
                "text": "Leaf 6",
                "embedding": [],
                "span_start": 75,
                "span_end": 87,
                "document_id": "doc1",
                "token_count": 12,
                "height": 0,
                "parent_id": "r3",
            },
            {
                "node_id": "r3_r",
                "text": "Leaf 7",
                "embedding": [],
                "span_start": 87,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 13,
                "height": 0,
                "parent_id": "r3",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("l1", "root"),
                ("r1", "root"),
                ("l2", "l1"),
                ("r2", "l1"),
                ("l3", "r1"),
                ("r3", "r1"),
                ("l2_l", "l2"),
                ("l2_r", "l2"),
                ("r2_l", "r2"),
                ("r2_r", "r2"),
                ("l3_l", "l3"),
                ("l3_r", "l3"),
                ("r3_l", "r3"),
                ("r3_r", "r3"),
            ]
        )

        # Mixed height tiling
        tiling = ["l1", "l3", "r3_r"]

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
