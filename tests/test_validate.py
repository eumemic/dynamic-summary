"""Tests for validation functions."""

from unittest.mock import MagicMock

from ragzoom.validate import (
    validate_chunk_sizes,
    validate_document_coverage,
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

        # Mock node with invalid span
        node = MagicMock(
            id="node1",
            span_start=100,
            span_end=50,  # Invalid: start > end
            left_child_id=None,
            right_child_id=None,
            summary="test",
        )

        # Mock the query
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter_by.return_value.all.return_value = [node]
        mock_session.query.return_value = mock_query
        store.SessionLocal.return_value.__enter__.return_value = mock_session

        # Mock is_leaf_node to return True for node without children
        store.is_leaf_node.return_value = True

        error = validate_tree_structure(store, "doc1")
        assert error is not None
        assert "validation failed" in error

    def test_missing_summary(self):
        """Test detection of missing text content on nodes."""
        store = MagicMock()

        # Mock parent node without text
        parent = MagicMock(
            id="parent",
            span_start=0,
            span_end=100,
            left_child_id="child1",
            right_child_id="child2",
            text=None,  # Missing text content
        )

        # Mock children
        child1 = MagicMock(
            id="child1",
            span_start=0,
            span_end=50,
        )
        child2 = MagicMock(
            id="child2",
            span_start=50,
            span_end=100,
        )

        # Mock the query
        mock_session = MagicMock()
        mock_query = MagicMock()
        mock_query.filter_by.return_value.all.return_value = [parent]
        mock_session.query.return_value = mock_query
        store.SessionLocal.return_value.__enter__.return_value = mock_session

        # Mock get_node to return children
        def get_node_side_effect(node_id):
            if node_id == "child1":
                return child1
            elif node_id == "child2":
                return child2
            return None

        store.get_node.side_effect = get_node_side_effect
        store.is_leaf_node.return_value = False

        error = validate_tree_structure(store, "doc1")
        assert error is not None
        assert "validation failed" in error
