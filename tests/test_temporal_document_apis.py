"""Tests for Temporal Document APIs.

Tests document-status and truncate_from_time APIs as specified in
specs/temporal-document-apis.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ragzoom.server.servicers import complete_forest_size

if TYPE_CHECKING:
    from ragzoom.contracts.node_repository import NodeDataDict
    from ragzoom.contracts.storage_backend import StorageBackend
    from ragzoom.document_store import DocumentStore


class TestCompleteForestSize:
    """Tests for the complete_forest_size helper function.

    The formula is: 2N - popcount(N)
    where popcount(N) is the number of 1-bits in N's binary representation.

    This represents the total nodes (leaves + inner) in a complete binary forest.
    """

    def test_complete_forest_size_zero(self) -> None:
        """Zero leaves means zero nodes."""
        assert complete_forest_size(0) == 0

    def test_complete_forest_size_negative(self) -> None:
        """Negative leaf count should return 0."""
        assert complete_forest_size(-1) == 0
        assert complete_forest_size(-100) == 0

    def test_complete_forest_size_powers_of_two(self) -> None:
        """Powers of two have popcount=1, so 2N - 1.

        These form a single perfect binary tree.
        """
        # 1 leaf (0b1): 2*1 - 1 = 1
        assert complete_forest_size(1) == 1
        # 2 leaves (0b10): 2*2 - 1 = 3 (2 leaves + 1 root)
        assert complete_forest_size(2) == 3
        # 4 leaves (0b100): 2*4 - 1 = 7 (perfect tree of depth 2)
        assert complete_forest_size(4) == 7
        # 8 leaves (0b1000): 2*8 - 1 = 15
        assert complete_forest_size(8) == 15
        # 16 leaves (0b10000): 2*16 - 1 = 31
        assert complete_forest_size(16) == 31

    def test_complete_forest_size_mixed(self) -> None:
        """Non-power-of-two counts have popcount > 1.

        These form a forest of multiple perfect binary trees.
        """
        # 3 leaves (0b11): popcount=2, 2*3 - 2 = 4
        # (tree of 2 + single leaf = 3 nodes + the root... wait, no)
        # Actually: 3 leaves, 1 inner node (pairs 2 leaves), so 3+1=4
        assert complete_forest_size(3) == 4

        # 5 leaves (0b101): popcount=2, 2*5 - 2 = 8
        # Tree of 4 (7 nodes) + 1 leaf = 8 nodes
        assert complete_forest_size(5) == 8

        # 6 leaves (0b110): popcount=2, 2*6 - 2 = 10
        # Tree of 4 + tree of 2 = 7 + 3 = 10
        assert complete_forest_size(6) == 10

        # 7 leaves (0b111): popcount=3, 2*7 - 3 = 11
        # Tree of 4 (7) + tree of 2 (3) + leaf (1) = 11
        assert complete_forest_size(7) == 11

        # 100 leaves (0b1100100): popcount=3, 2*100 - 3 = 197
        assert complete_forest_size(100) == 197

    def test_complete_forest_size_formula_correctness(self) -> None:
        """Verify formula against explicit calculation for small values."""
        # For N leaves, a complete binary forest has:
        # - N leaves
        # - N - popcount(N) inner nodes
        # Total = 2N - popcount(N)

        for n in range(1, 100):
            popcount = bin(n).count("1")
            expected = 2 * n - popcount
            actual = complete_forest_size(n)
            assert (
                actual == expected
            ), f"Failed for n={n}: expected {expected}, got {actual}"


class TestDocumentStoreNodeCount:
    """Tests for DocumentStore.get_node_count() method.

    Verifies that the document store correctly returns the total count
    of nodes (leaves + inner nodes) for a document.
    """

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> DocumentStore:
        """Create a document store with test metadata."""

        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        return doc_store

    def test_document_store_node_count_empty(self, doc_store: DocumentStore) -> None:
        """Empty document returns 0 node count."""
        assert doc_store.get_node_count() == 0

    def test_document_store_node_count_leaves_only(
        self, doc_store: DocumentStore
    ) -> None:
        """Document with only leaves returns leaf count."""
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"leaf-{i}",
                "text": f"Leaf text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "token_count": 5,
                "height": 0,
                "level_index": i,
            }
            for i in range(5)
        ]
        doc_store.nodes.add_batch(nodes_data)

        assert doc_store.get_node_count() == 5

    def test_document_store_node_count_with_inner_nodes(
        self, doc_store: DocumentStore
    ) -> None:
        """Document with leaves and inner nodes returns total count."""
        # Create 4 leaves
        leaves: list[NodeDataDict] = [
            {
                "node_id": f"leaf-{i}",
                "text": f"Leaf text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "token_count": 5,
                "height": 0,
                "level_index": i,
            }
            for i in range(4)
        ]
        doc_store.nodes.add_batch(leaves)

        # Create 2 inner nodes at height 1
        inner_h1: list[NodeDataDict] = [
            {
                "node_id": "inner-0",
                "text": "Summary of leaves 0-1",
                "span_start": 0,
                "span_end": 20,
                "token_count": 10,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf-0",
                "right_child_id": "leaf-1",
            },
            {
                "node_id": "inner-1",
                "text": "Summary of leaves 2-3",
                "span_start": 20,
                "span_end": 40,
                "token_count": 10,
                "height": 1,
                "level_index": 1,
                "left_child_id": "leaf-2",
                "right_child_id": "leaf-3",
            },
        ]
        doc_store.nodes.add_batch(inner_h1)

        # Create 1 root at height 2
        root: list[NodeDataDict] = [
            {
                "node_id": "root",
                "text": "Summary of all",
                "span_start": 0,
                "span_end": 40,
                "token_count": 15,
                "height": 2,
                "level_index": 0,
                "left_child_id": "inner-0",
                "right_child_id": "inner-1",
            },
        ]
        doc_store.nodes.add_batch(root)

        # 4 leaves + 2 inner + 1 root = 7 total nodes
        assert doc_store.get_node_count() == 7


class TestDocumentStoreTemporalRange:
    """Tests for DocumentStore.get_temporal_range() method.

    Verifies that the document store correctly returns the temporal range
    (min time_start, max time_end) from leaf nodes only.
    """

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> DocumentStore:
        """Create a document store with test metadata."""
        doc_store = storage_backend.for_document("test-temporal-doc")
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        return doc_store

    def test_temporal_range_empty_document(self, doc_store: DocumentStore) -> None:
        """Empty document returns (None, None) for temporal range."""
        result = doc_store.get_temporal_range()
        assert result == (None, None)

    def test_temporal_range_non_temporal_document(
        self, doc_store: DocumentStore
    ) -> None:
        """Document with leaves but no timestamps returns (None, None)."""
        # Add leaves without timestamps
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"leaf-{i}",
                "text": f"Leaf text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "token_count": 5,
                "height": 0,
                "level_index": i,
            }
            for i in range(3)
        ]
        doc_store.nodes.add_batch(nodes_data)

        result = doc_store.get_temporal_range()
        assert result == (None, None)

    def test_temporal_range_leaves_only(self, doc_store: DocumentStore) -> None:
        """Document with temporal leaves returns correct min/max range."""
        # Create leaves with timestamps
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": "leaf-0",
                "text": "First message",
                "span_start": 0,
                "span_end": 10,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1100.0,
            },
            {
                "node_id": "leaf-1",
                "text": "Second message",
                "span_start": 10,
                "span_end": 20,
                "token_count": 5,
                "height": 0,
                "level_index": 1,
                "time_start": 1100.0,
                "time_end": 1200.0,
            },
            {
                "node_id": "leaf-2",
                "text": "Third message",
                "span_start": 20,
                "span_end": 30,
                "token_count": 5,
                "height": 0,
                "level_index": 2,
                "time_start": 1200.0,
                "time_end": 1300.0,
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        time_start, time_end = doc_store.get_temporal_range()
        assert time_start == 1000.0
        assert time_end == 1300.0

    def test_temporal_range_with_inner_nodes(self, doc_store: DocumentStore) -> None:
        """Temporal range only considers leaves, not inner nodes."""
        # Create leaves with timestamps
        leaves: list[NodeDataDict] = [
            {
                "node_id": "leaf-0",
                "text": "First",
                "span_start": 0,
                "span_end": 10,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1050.0,
            },
            {
                "node_id": "leaf-1",
                "text": "Second",
                "span_start": 10,
                "span_end": 20,
                "token_count": 5,
                "height": 0,
                "level_index": 1,
                "time_start": 1050.0,
                "time_end": 1100.0,
            },
        ]
        doc_store.nodes.add_batch(leaves)

        # Create inner node spanning both leaves (with wider time range)
        inner: list[NodeDataDict] = [
            {
                "node_id": "inner-0",
                "text": "Summary",
                "span_start": 0,
                "span_end": 20,
                "token_count": 8,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf-0",
                "right_child_id": "leaf-1",
                "time_start": 900.0,  # Wider range than leaves (shouldn't be used)
                "time_end": 1200.0,
            },
        ]
        doc_store.nodes.add_batch(inner)

        # Should only use leaf timestamps
        time_start, time_end = doc_store.get_temporal_range()
        assert time_start == 1000.0
        assert time_end == 1100.0

    def test_temporal_range_single_leaf(self, doc_store: DocumentStore) -> None:
        """Single temporal leaf returns that leaf's time range."""
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": "leaf-0",
                "text": "Only message",
                "span_start": 0,
                "span_end": 10,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1500.0,
                "time_end": 1600.0,
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        time_start, time_end = doc_store.get_temporal_range()
        assert time_start == 1500.0
        assert time_end == 1600.0

    def test_temporal_range_no_document_id(
        self, storage_backend: StorageBackend
    ) -> None:
        """Document store without document_id returns (None, None)."""
        doc_store = storage_backend.for_document(None)
        result = doc_store.get_temporal_range()
        assert result == (None, None)
