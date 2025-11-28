"""SQLite-based tests for DP assembly path.

SQLite-based tests for DP assembly functionality with the real
in-memory SQLite backend, providing higher fidelity testing of the
DP assembly functionality.

Complex DP algorithm tests that require retrieval functionality are kept
in the original test_dp_assembly.py for now.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.assemble import Assembler
from ragzoom.document_store import DocumentStore


@pytest.mark.usefixtures("sqlite_backend")
class TestDPAssemblySQLiteExtended:
    """Test DP assembly with real SQLite backend."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        """Create a document-scoped store for doc1."""
        return sqlite_store_factory("doc1")

    @pytest.fixture
    def assembler(self, doc_store: DocumentStore) -> Assembler:
        """Create assembler with SQLite document store."""
        return Assembler(doc_store)

    @pytest.fixture
    def seed_nodes(self, doc_store: DocumentStore) -> None:
        """Create a comprehensive tree directly in the SQLite backend.

        Structure:
            root (0-82)
            /          \
         left(0-41)   right(41-82)
         /    \\        /     \
      leaf1 leaf2   leaf3  leaf4
        """
        nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            # Leaf nodes
            {
                "node_id": "leaf1",
                "text": "First chunk of text.",
                "embedding": [],
                "span_start": 0,
                "span_end": 20,
                "document_id": "doc1",
                "token_count": 3,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf2",
                "text": "Second chunk of text.",
                "embedding": [],
                "span_start": 20,
                "span_end": 41,
                "document_id": "doc1",
                "token_count": 4,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf3",
                "text": "Third chunk of text.",
                "embedding": [],
                "span_start": 41,
                "span_end": 61,
                "document_id": "doc1",
                "token_count": 4,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf4",
                "text": "Fourth chunk of text.",
                "embedding": [],
                "span_start": 61,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 4,
                "height": 0,
                "level_index": 0,
            },
            # Internal nodes
            {
                "node_id": "left",
                "text": "Summary of first and second chunks.",
                "embedding": [],
                "span_start": 0,
                "span_end": 41,
                "document_id": "doc1",
                "token_count": 7,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
            },
            {
                "node_id": "right",
                "text": "Summary of third and fourth chunks.",
                "embedding": [],
                "span_start": 41,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 7,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf3",
                "right_child_id": "leaf4",
            },
            {
                "node_id": "root",
                "text": "Overall document summary.",
                "embedding": [],
                "span_start": 0,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 5,
                "height": 2,
                "level_index": 0,
                "left_child_id": "left",
                "right_child_id": "right",
            },
        ]
        doc_store.nodes.add_batch(nodes)
        # Update parent references
        doc_store.nodes.update_parent_references_batch(
            [
                ("leaf1", "left"),
                ("leaf2", "left"),
                ("leaf3", "right"),
                ("leaf4", "right"),
                ("left", "root"),
                ("right", "root"),
            ]
        )

    def test_basic_dp_assembly(self, assembler: Assembler, seed_nodes: None) -> None:
        """Test basic DP assembly with leaf nodes."""
        tiling = ["leaf1", "leaf2"]
        result = assembler.assemble_dp(tiling)
        assert result == "First chunk of text.\n\nSecond chunk of text."

    def test_internal_node_assembly(
        self, assembler: Assembler, seed_nodes: None
    ) -> None:
        """Test assembly with internal nodes (atomic units)."""
        tiling = ["left", "leaf3"]
        result = assembler.assemble_dp(tiling)
        assert result == "Summary of first and second chunks.\n\nThird chunk of text."

    def test_mixed_nodes_assembly(self, assembler: Assembler, seed_nodes: None) -> None:
        """Test assembly with mix of leaf and internal nodes."""
        tiling = ["leaf1", "right"]
        result = assembler.assemble_dp(tiling)
        assert result == "First chunk of text.\n\nSummary of third and fourth chunks."

    def test_all_internal_nodes(self, assembler: Assembler, seed_nodes: None) -> None:
        """Test assembly with only internal nodes."""
        tiling = ["left", "right", "root"]
        result = assembler.assemble_dp(tiling)
        expected = "Summary of first and second chunks.\n\nSummary of third and fourth chunks.\n\nOverall document summary."
        assert result == expected

    def test_all_leaf_nodes(self, assembler: Assembler, seed_nodes: None) -> None:
        """Test assembly with all leaf nodes."""
        tiling = ["leaf1", "leaf2", "leaf3", "leaf4"]
        result = assembler.assemble_dp(tiling)
        expected = "First chunk of text.\n\nSecond chunk of text.\n\nThird chunk of text.\n\nFourth chunk of text."
        assert result == expected

    def test_empty_tiling(self, assembler: Assembler, seed_nodes: None) -> None:
        """Test handling of empty tiling list."""
        tiling: list[str] = []
        result = assembler.assemble_dp(tiling)
        assert result == ""

    def test_missing_node(self, assembler: Assembler, seed_nodes: None) -> None:
        """Test handling when tiling references a missing node."""
        tiling = ["leaf1", "missing", "leaf3"]
        result = assembler.assemble_dp(tiling)
        # Should skip missing node
        assert result == "First chunk of text.\n\nThird chunk of text."

    def test_node_with_no_text(
        self, assembler: Assembler, seed_nodes: None, doc_store: DocumentStore
    ) -> None:
        """Test handling of nodes with empty text."""
        # Add a node with empty text
        empty_nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "empty",
                "text": "",
                "embedding": [],
                "span_start": 82,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 0,
                "height": 0,
                "level_index": 0,
            }
        ]
        doc_store.nodes.add_batch(empty_nodes)

        tiling = ["leaf1", "empty", "leaf3"]
        result = assembler.assemble_dp(tiling)
        # Should skip empty node
        assert result == "First chunk of text.\n\nThird chunk of text."

    def test_single_root_node(self, assembler: Assembler, seed_nodes: None) -> None:
        """Test assembly with just the root node."""
        tiling = ["root"]
        result = assembler.assemble_dp(tiling)
        # Should return root's full summary
        assert result == "Overall document summary."

    def test_complex_tiling_assembly(
        self, assembler: Assembler, seed_nodes: None
    ) -> None:
        """Test a complex tiling that resembles real DP output."""
        # Simulate a tiling that might come from DP algorithm
        # Mix of internal and leaf nodes
        tiling = ["left", "leaf3", "leaf4"]
        result = assembler.assemble_dp(tiling)
        expected = (
            "Summary of first and second chunks.\n\n"
            "Third chunk of text.\n\n"
            "Fourth chunk of text."
        )
        assert result == expected

    def test_ordering_preservation(
        self, assembler: Assembler, seed_nodes: None
    ) -> None:
        """Test that tiling order is preserved in output."""
        # Nodes in non-sequential order
        tiling = ["leaf3", "leaf1", "leaf4", "leaf2"]
        result = assembler.assemble_dp(tiling)
        # Output should follow tiling order, not span order
        expected = (
            "Third chunk of text.\n\n"
            "First chunk of text.\n\n"
            "Fourth chunk of text.\n\n"
            "Second chunk of text."
        )
        assert result == expected
