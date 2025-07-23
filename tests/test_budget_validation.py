"""Test budget validation in validate_tiling."""

from ragzoom.config import RagZoomConfig
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

        # Create tiling that would exceed a small budget
        tiling = ["node1", "node2"]  # ~20 + ~30 = ~50 tokens

        # Validate with budget that's too small
        error = validate_tiling(tiling, store, "test-doc", budget_tokens=40)

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

        # Create tiling within budget
        tiling = ["node1"]  # ~10 tokens

        # Validate with sufficient budget
        error = validate_tiling(tiling, store, "test-doc", budget_tokens=100)

        assert error is None

    def test_budget_validation_with_parent_child(self):
        """Test budget validation with parent and child nodes."""
        config = RagZoomConfig(leaf_tokens=100)
        store = SimpleMockStore(config=config)

        # Create child nodes
        store.add_node(
            node_id="left_child",
            text="left part " * 10,
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            document_id="test-doc",
        )

        store.add_node(
            node_id="right_child",
            text="right part " * 10,
            embedding=[0.1] * 1536,
            span_start=100,
            span_end=200,
            document_id="test-doc",
        )

        # Create parent node
        store.add_node(
            node_id="parent",
            text="Summary of left and right parts",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=200,
            document_id="test-doc",
            summary="Summary of left and right parts",
            left_child_id="left_child",
            right_child_id="right_child",
        )

        # Create tiling with child nodes
        tiling = ["left_child", "right_child"]  # ~20 tokens total

        # Should pass with budget of 50
        error = validate_tiling(tiling, store, "test-doc", budget_tokens=50)
        assert error is None

        # Should fail with budget of 15
        error = validate_tiling(tiling, store, "test-doc", budget_tokens=15)
        assert error is not None
        assert "exceeds budget" in error
