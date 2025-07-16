"""Tests for validation functions."""

from unittest.mock import MagicMock

from ragzoom.validate import (
    validate_chunk_sizes,
    validate_document_coverage,
    validate_frontier_completeness,
    validate_no_overlap,
    validate_tree_structure,
)


class TestDocumentCoverage:
    """Test document coverage validation."""

    def test_valid_coverage(self):
        """Test valid document coverage passes."""
        # Create mock leaf nodes - now with exact adjacency (no gaps, no overlaps)
        leaves = [
            MagicMock(span_start=0, span_end=100, id="node1"),
            MagicMock(span_start=100, span_end=200, id="node2"),  # Exactly adjacent
            MagicMock(span_start=200, span_end=300, id="node3"),  # Exactly adjacent
        ]

        original_text = "x" * 300
        error = validate_document_coverage(original_text, leaves)
        assert error is None  # Should be valid

    def test_missing_start_coverage(self):
        """Test detection of missing coverage at start."""
        leaves = [
            MagicMock(span_start=10, span_end=100, id="node1"),  # Doesn't start at 0
            MagicMock(span_start=100, span_end=200, id="node2"),  # Exactly adjacent
        ]

        original_text = "x" * 200
        error = validate_document_coverage(original_text, leaves)
        assert error is not None
        assert "First leaf node starts at 10" in error

    def test_missing_end_coverage(self):
        """Test detection of missing coverage at end."""
        leaves = [
            MagicMock(span_start=0, span_end=100, id="node1"),
            MagicMock(span_start=100, span_end=180, id="node2"),  # Doesn't reach 200
        ]

        original_text = "x" * 200
        error = validate_document_coverage(original_text, leaves)
        assert error is not None
        assert "Last leaf node ends at 180" in error

    def test_gap_in_coverage(self):
        """Test detection of gaps between nodes."""
        leaves = [
            MagicMock(span_start=0, span_end=100, id="node1"),
            MagicMock(span_start=110, span_end=200, id="node2"),  # Gap from 100-110
        ]

        original_text = "x" * 200
        error = validate_document_coverage(original_text, leaves)
        assert error is not None
        assert "Non-contiguous chunks found" in error


class TestChunkSizes:
    """Test chunk size validation."""

    def test_valid_chunk_sizes(self):
        """Test chunks within tolerance pass."""
        leaves = [
            MagicMock(text="x" * 800, id="node1"),  # ~200 tokens
            MagicMock(text="x" * 820, id="node2"),  # ~205 tokens
            MagicMock(text="x" * 780, id="node3"),  # ~195 tokens
        ]

        error = validate_chunk_sizes(leaves, target_tokens=200)
        assert error is None  # Should be valid

    def test_oversized_chunks(self):
        """Test detection of oversized chunks."""
        leaves = [
            MagicMock(text="x" * 800, id="node1"),
            MagicMock(text="x" * 1000, id="node2"),  # ~250 tokens - too big
        ]

        # Should log warning but not raise
        error = validate_chunk_sizes(leaves, target_tokens=200, tolerance=0.2)
        assert error is None  # Should still return None, just log warnings

    def test_undersized_chunks(self):
        """Test detection of undersized chunks."""
        leaves = [
            MagicMock(text="x" * 600, id="node1"),  # ~150 tokens - too small
            MagicMock(text="x" * 800, id="node2"),
        ]

        # Should log warning but not raise
        error = validate_chunk_sizes(leaves, target_tokens=200, tolerance=0.2)
        assert error is None  # Should still return None, just log warnings


class TestTreeStructure:
    """Test tree structure validation."""

    def test_invalid_span(self):
        """Test detection of invalid spans."""
        store = MagicMock()

        # Mock invalid node with span_end < span_start
        node = MagicMock(
            id="bad_node",
            span_start=100,
            span_end=50,  # Invalid!
            depth=1,
            left_child_id=None,
            right_child_id=None,
            summary="test",
            mid_offset=10,
        )

        with store.SessionLocal() as session:
            session.query().filter_by().all.return_value = [node]

        error = validate_tree_structure(store, "doc1")
        assert error is not None
        assert "Tree structure validation failed" in error

    def test_missing_summary(self):
        """Test detection of missing summaries in non-leaf nodes."""
        store = MagicMock()

        # Mock non-leaf node without summary
        node = MagicMock(
            id="bad_node",
            span_start=0,
            span_end=100,
            depth=1,  # Non-leaf
            left_child_id="child1",
            right_child_id="child2",
            summary=None,  # Missing!
            mid_offset=None,
        )

        # Mock children
        child1 = MagicMock(span_start=0, span_end=50)
        child2 = MagicMock(span_start=50, span_end=100)
        store.get_node.side_effect = lambda x: child1 if x == "child1" else child2

        with store.SessionLocal() as session:
            session.query().filter_by().all.return_value = [node]

        error = validate_tree_structure(store, "doc1")
        assert error is not None
        assert "Tree structure validation failed" in error


class TestFrontierValidation:
    """Test frontier validation functions."""

    def test_valid_frontier_completeness(self):
        """Test valid complete frontier."""
        segments = [
            ("node1", "text1", 0, 100),
            ("node2", "text2", 100, 200),
            ("node3", "text3", 200, 300),
        ]

        error = validate_frontier_completeness(segments, (0, 300))
        assert error is None  # Should be valid

    def test_frontier_with_gap(self):
        """Test detection of gaps in frontier."""
        segments = [
            ("node1", "text1", 0, 100),
            ("node2", "text2", 110, 200),  # Gap from 100-110
        ]

        error = validate_frontier_completeness(segments, (0, 200))
        assert error is not None
        assert "Gap in frontier" in error

    def test_valid_no_overlap(self):
        """Test non-overlapping frontier."""
        segments = [
            ("node1", "text1", 0, 100),
            ("node2", "text2", 100, 200),
        ]

        error = validate_no_overlap(segments)
        assert error is None  # Should be valid

    def test_overlapping_segments(self):
        """Test detection of overlapping segments."""
        segments = [
            ("node1", "text1", 0, 110),
            ("node2", "text2", 100, 200),  # Overlaps from 100-110
        ]

        error = validate_no_overlap(segments)
        assert error is not None
        assert "Overlapping segments" in error
