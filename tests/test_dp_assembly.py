"""Tests for DP assembly path."""

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.dynamic_tiling import Segment


class TestDPAssembly:
    """Test the DP assembly path that uses Segments."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RagZoomConfig(
            openai_api_key="test-key", slope_cap=True, budget_tokens=1000
        )

    @pytest.fixture
    def assembler(self, config, store):
        """Create assembler with mock store."""
        return Assembler(config, store)

    @pytest.fixture
    def mock_nodes(self, store):
        """Create mock nodes in the store."""
        # Create a simple tree structure
        # Root (depth=2)
        #  / \
        # L   R (depth=1)
        # /\ /\
        # 1 2 3 4 (depth=0)

        # Leaf nodes
        store.add_node(
            node_id="leaf1",
            text="First chunk of text.",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=20,
            document_id="doc1",
        )

        store.add_node(
            node_id="leaf2",
            text="Second chunk of text.",
            embedding=[0.2] * 1536,
            span_start=20,
            span_end=41,
            document_id="doc1",
        )

        store.add_node(
            node_id="leaf3",
            text="Third chunk of text.",
            embedding=[0.3] * 1536,
            span_start=41,
            span_end=61,
            document_id="doc1",
        )

        store.add_node(
            node_id="leaf4",
            text="Fourth chunk of text.",
            embedding=[0.4] * 1536,
            span_start=61,
            span_end=82,
            document_id="doc1",
        )

        # Internal nodes with MID delimiters
        store.add_node(
            node_id="left",
            text="Summary of first half. <<<MID>>> Summary of second half.",
            embedding=[0.15] * 1536,
            span_start=0,
            span_end=41,
            left_child_id="leaf1",
            right_child_id="leaf2",
            mid_offset=23,  # Position of <<<MID>>>
            document_id="doc1",
        )

        store.add_node(
            node_id="right",
            text="Summary of third chunk. <<<MID>>> Summary of fourth chunk.",
            embedding=[0.35] * 1536,
            span_start=41,
            span_end=82,
            left_child_id="leaf3",
            right_child_id="leaf4",
            mid_offset=24,  # Position of <<<MID>>>
            document_id="doc1",
        )

        store.add_node(
            node_id="root",
            text="Overall document summary. <<<MID>>> More summary content.",
            embedding=[0.25] * 1536,
            span_start=0,
            span_end=82,
            left_child_id="left",
            right_child_id="right",
            mid_offset=26,  # Position of <<<MID>>>
            document_id="doc1",
        )

    def test_basic_dp_assembly(self, assembler, mock_nodes):
        """Test basic DP assembly with leaf segments."""
        # Leaf nodes now have side=None
        segments = [
            Segment(node_id="leaf1", side=None),
            Segment(node_id="leaf2", side=None),
        ]

        result = assembler.assemble_dp(segments)

        # Leaf nodes return full text
        assert result == "First chunk of text.\n\nSecond chunk of text."

    def test_left_side_extraction(self, assembler, mock_nodes):
        """Test extracting LEFT side of a node with MID delimiter."""
        segments = [
            Segment(node_id="left", side="LEFT"),
            Segment(node_id="leaf3", side=None),
        ]

        result = assembler.assemble_dp(segments)

        # Should get left half of "left" node (before <<<MID>>>)
        # Leaf nodes return full text regardless of side
        assert result == "Summary of first half.\n\nThird chunk of text."

    def test_right_side_extraction(self, assembler, mock_nodes):
        """Test extracting RIGHT side of a node with MID delimiter."""
        segments = [
            Segment(node_id="leaf1", side=None),
            Segment(node_id="right", side="RIGHT"),
        ]

        result = assembler.assemble_dp(segments)

        # Leaf returns full text, right returns text after <<<MID>>> (cleaned)
        assert result == "First chunk of text.\n\nSummary of fourth chunk."

    def test_mixed_sides_assembly(self, assembler, mock_nodes):
        """Test assembly with mixed LEFT/RIGHT segments."""
        segments = [
            Segment(node_id="left", side="LEFT"),
            Segment(node_id="left", side="RIGHT"),
            Segment(node_id="right", side="LEFT"),
            Segment(node_id="right", side="RIGHT"),
        ]

        result = assembler.assemble_dp(segments)

        expected = "Summary of first half.\n\nSummary of second half.\n\nSummary of third chunk.\n\nSummary of fourth chunk."
        assert result == expected

    def test_leaf_node_handling(self, assembler, mock_nodes):
        """Test that leaf nodes use side=None."""
        # Leaf nodes must have side=None
        segments = [
            Segment(node_id="leaf1", side=None),
            Segment(node_id="leaf2", side=None),
        ]

        result = assembler.assemble_dp(segments)

        # Should get full text for leaf nodes
        assert result == "First chunk of text.\n\nSecond chunk of text."

    def test_empty_segments(self, assembler, mock_nodes):
        """Test handling of empty segment list."""
        segments = []

        result = assembler.assemble_dp(segments)

        assert result == ""

    def test_missing_node(self, assembler, mock_nodes):
        """Test handling when a segment references a missing node."""
        segments = [
            Segment(node_id="leaf1", side=None),
            Segment(node_id="missing", side="LEFT"),
            Segment(node_id="leaf3", side=None),
        ]

        result = assembler.assemble_dp(segments)

        # Should skip missing node
        assert result == "First chunk of text.\n\nThird chunk of text."

    def test_node_with_no_text(self, assembler, mock_nodes, store):
        """Test handling of nodes with empty text."""
        # Add a node with empty text
        store.add_node(
            node_id="empty",
            text="",
            embedding=[0.5] * 1536,
            span_start=82,
            span_end=82,
            document_id="doc1",
        )

        segments = [
            Segment(node_id="leaf1", side=None),
            Segment(node_id="empty", side="LEFT"),
            Segment(node_id="leaf3", side=None),
        ]

        result = assembler.assemble_dp(segments)

        # Should skip empty node
        assert result == "First chunk of text.\n\nThird chunk of text."

    def test_node_without_mid_offset(self, assembler, mock_nodes, store):
        """Test handling of internal node without mid_offset."""
        # Add an internal node without mid_offset (shouldn't happen, but test fallback)
        store.add_node(
            node_id="bad_internal",
            text="Summary without MID delimiter",
            embedding=[0.6] * 1536,
            span_start=82,
            span_end=100,
            left_child_id="leaf1",
            right_child_id="leaf2",
            mid_offset=None,  # No MID offset
            document_id="doc1",
        )

        segments = [
            Segment(node_id="bad_internal", side=None),
        ]

        result = assembler.assemble_dp(segments)

        # Should return full text since mid_offset is None
        assert result == "Summary without MID delimiter"

    def test_complex_tiling_assembly(self, assembler, mock_nodes):
        """Test a complex tiling that resembles real DP output."""
        # Simulate a tiling that might come from DP algorithm
        segments = [
            Segment(node_id="left", side="LEFT"),
            Segment(node_id="leaf2", side=None),
            Segment(node_id="right", side="LEFT"),
            Segment(node_id="leaf4", side=None),
        ]

        result = assembler.assemble_dp(segments)

        expected = (
            "Summary of first half.\n\n"
            "Second chunk of text.\n\n"
            "Summary of third chunk.\n\n"
            "Fourth chunk of text."
        )
        assert result == expected

    def test_ordering_preservation(self, assembler, mock_nodes):
        """Test that segment order is preserved in output."""
        # Segments in non-sequential order
        segments = [
            Segment(node_id="leaf3", side=None),
            Segment(node_id="leaf1", side=None),
            Segment(node_id="leaf4", side=None),
            Segment(node_id="leaf2", side=None),
        ]

        result = assembler.assemble_dp(segments)

        # Output should follow segment order, not span order
        expected = (
            "Third chunk of text.\n\n"
            "First chunk of text.\n\n"
            "Fourth chunk of text.\n\n"
            "Second chunk of text."
        )
        assert result == expected
