"""Test budget validation in validate_tiling."""

from ragzoom.config import RagZoomConfig
from ragzoom.dynamic_tiling import Segment
from ragzoom.validate import validate_tiling
from tests.mock_store import SimpleMockStore


class TestBudgetValidation:
    """Test that budget validation catches overflows."""

    def test_budget_validation_catches_overflow(self):
        """Test that validation fails when tiling exceeds budget."""
        config = RagZoomConfig(leaf_tokens=100)
        store = SimpleMockStore(config=config)

        # Create some nodes with known token costs
        # "test " * 20 = ~20 tokens
        store.add_node(
            node_id="node1",
            text="test " * 20,
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            document_id="test-doc",
        )

        store.add_node(
            node_id="node2",
            text="test " * 30,
            embedding=[0.2] * 1536,
            span_start=100,
            span_end=200,
            document_id="test-doc",
        )

        # Create segments that would exceed a small budget
        segments = [
            Segment("node1", None),  # ~20 tokens
            Segment("node2", None),  # ~30 tokens
        ]

        # Validate with budget that's too small
        error = validate_tiling(segments, store, "test-doc", budget_tokens=40)

        assert error is not None
        assert "exceeds budget" in error
        assert "> 40 budget" in error

    def test_budget_validation_passes_within_budget(self):
        """Test that validation passes when tiling is within budget."""
        config = RagZoomConfig(leaf_tokens=100)
        store = SimpleMockStore(config=config)

        # Create a node
        store.add_node(
            node_id="node1",
            text="test " * 10,
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=50,
            document_id="test-doc",
        )

        # Create segment within budget
        segments = [Segment("node1", None)]  # ~10 tokens

        # Validate with sufficient budget
        error = validate_tiling(segments, store, "test-doc", budget_tokens=100)

        assert error is None

    def test_budget_validation_with_segment_sides(self):
        """Test budget validation with LEFT/RIGHT segments."""
        config = RagZoomConfig(leaf_tokens=100)
        store = SimpleMockStore(config=config)

        # Create an internal node with mid_offset
        full_text = "left part " * 10 + "right part " * 10

        # First create children
        store.add_node(
            node_id="left_child",
            text="left part " * 10,
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            parent_id="internal",
            document_id="test-doc",
        )

        store.add_node(
            node_id="right_child",
            text="right part " * 10,
            embedding=[0.1] * 1536,
            span_start=100,
            span_end=200,
            parent_id="internal",
            document_id="test-doc",
        )

        # Now create internal node
        store.add_node(
            node_id="internal",
            text=full_text,
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=200,
            document_id="test-doc",
            summary=full_text,
            mid_offset=len("left part " * 10),
            left_child_id="left_child",
            right_child_id="right_child",
        )

        # Create segments for left and right
        segments = [
            Segment("internal", "LEFT"),  # ~10 tokens
            Segment("internal", "RIGHT"),  # ~10 tokens
        ]

        # Should pass with budget of 50 (actual is ~40)
        error = validate_tiling(segments, store, "test-doc", budget_tokens=50)
        assert error is None

        # Should fail with budget of 30
        error = validate_tiling(segments, store, "test-doc", budget_tokens=30)
        assert error is not None
        assert "exceeds budget" in error
