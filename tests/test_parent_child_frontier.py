"""Test parent-child frontier handling with <<<MID>>> extraction."""

from unittest.mock import MagicMock

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import RetrievalResult


@pytest.mark.skip(
    reason="Legacy assembler tests, will be removed with DP implementation"
)
class TestParentChildFrontier:
    """Tests for assembling frontiers containing both parent and child nodes."""

    @pytest.fixture
    def setup_assembler(self):
        """Set up assembler with mock store."""
        config = RagZoomConfig(
            budget_tokens=1000,
            leaf_tokens=100,
            openai_api_key="test-key",
            slope_cap=False,
            smoothing_pass_enabled=False,
        )
        store = MagicMock()
        assembler = Assembler(config, store)
        return assembler, store

    def test_parent_and_left_child_in_frontier(self, setup_assembler):
        """Test when parent and left child are both in frontier - should keep both."""
        assembler, store = setup_assembler

        # Create nodes: parent with left and right children
        parent = MagicMock()
        parent.id = "parent"
        parent.text = "Summary of left content. <<<MID>>> Summary of right content."
        parent.mid_offset = 25  # Position of first "<" in <<<MID>>>
        parent.span_start = 0
        parent.span_end = 200
        parent.depth = 1
        parent.parent_id = None

        left_child = MagicMock()
        left_child.id = "left"
        left_child.text = "Detailed left child content with specific information."
        left_child.span_start = 0
        left_child.span_end = 100
        left_child.depth = 0
        left_child.parent_id = "parent"

        right_child = MagicMock()
        right_child.id = "right"
        right_child.text = "Detailed right child content with other information."
        right_child.span_start = 100
        right_child.span_end = 200
        right_child.depth = 0
        right_child.parent_id = "parent"

        # Mock store methods
        def get_node(node_id):
            if node_id == "parent":
                return parent
            elif node_id == "left":
                return left_child
            elif node_id == "right":
                return right_child
            return None

        def get_children(node_id):
            if node_id == "parent":
                return left_child, right_child
            return None, None

        store.get_node.side_effect = get_node
        store.get_children.side_effect = get_children

        # Create retrieval result with parent and left child in frontier
        # This happens when left child is explicitly selected but right child isn't
        retrieval_result = MagicMock(spec=RetrievalResult)
        retrieval_result.frontier_nodes = ["parent", "left"]
        retrieval_result.coverage_map = {"parent": True, "left": True}

        # Assemble - with current bug, this removes left child
        result = assembler.assemble(retrieval_result)

        # EXPECTED: Should include left child's full text + parent's right summary
        # "Detailed left child content with specific information."
        # "Summary of right content."
        expected = "Detailed left child content with specific information.\n\nSummary of right content."

        # ACTUAL with bug: Only parent's full summary
        # "Summary of left content. <<<MID>>> Summary of right content."

        assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"

    def test_parent_and_right_child_in_frontier(self, setup_assembler):
        """Test when parent and right child are both in frontier - should keep both."""
        assembler, store = setup_assembler

        # Create nodes: parent with left and right children
        parent = MagicMock()
        parent.id = "parent"
        parent.text = "Summary of left content. <<<MID>>> Summary of right content."
        parent.mid_offset = 25  # Position of first "<" in <<<MID>>>
        parent.span_start = 0
        parent.span_end = 200
        parent.depth = 1
        parent.parent_id = None

        left_child = MagicMock()
        left_child.id = "left"
        left_child.text = "Detailed left child content."
        left_child.span_start = 0
        left_child.span_end = 100
        left_child.depth = 0
        left_child.parent_id = "parent"

        right_child = MagicMock()
        right_child.id = "right"
        right_child.text = "Detailed right child content with specific information."
        right_child.span_start = 100
        right_child.span_end = 200
        right_child.depth = 0
        right_child.parent_id = "parent"

        # Mock store methods
        def get_node(node_id):
            if node_id == "parent":
                return parent
            elif node_id == "left":
                return left_child
            elif node_id == "right":
                return right_child
            return None

        def get_children(node_id):
            if node_id == "parent":
                return left_child, right_child
            return None, None

        store.get_node.side_effect = get_node
        store.get_children.side_effect = get_children

        # Create retrieval result with parent and right child in frontier
        retrieval_result = MagicMock(spec=RetrievalResult)
        retrieval_result.frontier_nodes = ["parent", "right"]
        retrieval_result.coverage_map = {"parent": True, "right": True}

        # Assemble
        result = assembler.assemble(retrieval_result)

        # EXPECTED: Should include parent's left summary + right child's full text
        # "Summary of left content."
        # "Detailed right child content with specific information."
        expected = "Summary of left content.\n\nDetailed right child content with specific information."

        assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"

    def test_parent_and_both_children_in_frontier(self, setup_assembler):
        """Test when parent and both children are in frontier - should remove parent."""
        assembler, store = setup_assembler

        # Create nodes
        parent = MagicMock()
        parent.id = "parent"
        parent.text = "Parent summary"
        parent.span_start = 0
        parent.span_end = 200
        parent.depth = 1
        parent.parent_id = None

        left_child = MagicMock()
        left_child.id = "left"
        left_child.text = "Left child content"
        left_child.span_start = 0
        left_child.span_end = 100
        left_child.depth = 0
        left_child.parent_id = "parent"

        right_child = MagicMock()
        right_child.id = "right"
        right_child.text = "Right child content"
        right_child.span_start = 100
        right_child.span_end = 200
        right_child.depth = 0
        right_child.parent_id = "parent"

        # Mock store methods
        def get_node(node_id):
            if node_id == "parent":
                return parent
            elif node_id == "left":
                return left_child
            elif node_id == "right":
                return right_child
            return None

        def get_children(node_id):
            if node_id == "parent":
                return left_child, right_child
            return None, None

        store.get_node.side_effect = get_node
        store.get_children.side_effect = get_children

        # All three in frontier - this is the case where parent SHOULD be removed
        retrieval_result = MagicMock(spec=RetrievalResult)
        retrieval_result.frontier_nodes = ["parent", "left", "right"]
        retrieval_result.coverage_map = {"parent": True, "left": True, "right": True}

        # Assemble
        result = assembler.assemble(retrieval_result)

        # Should only include children, not parent
        expected = "Left child content\n\nRight child content"

        assert result == expected, f"Expected:\n{expected}\n\nGot:\n{result}"
