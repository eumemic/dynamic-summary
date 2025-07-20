"""Tests for position resolvers in tree visualization."""

from ragzoom.dynamic_frontier import Segment, SegmentInfo
from ragzoom.store import TreeNode
from ragzoom.tree_viz import CharacterPositionResolver, TokenPositionResolver
from tests.mock_store import SimpleMockStore


class TestPositionResolvers:
    """Test position resolver implementations."""

    def test_character_position_resolver(self):
        """Test character-based position resolver."""
        # Create mock nodes
        nodes = [
            TreeNode(
                id="root",
                depth=1,
                span_start=0,
                span_end=1000,
                text="Root summary",
                summary=None,
                parent_id=None,
                left_child_id="left",
                right_child_id="right",
                mid_offset=6,
                is_dirty=False,
                document_id="doc1",
            ),
            TreeNode(
                id="left",
                depth=0,
                span_start=0,
                span_end=500,
                text="Left leaf text",
                summary=None,
                parent_id="root",
                left_child_id=None,
                right_child_id=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
            TreeNode(
                id="right",
                depth=0,
                span_start=500,
                span_end=1000,
                text="Right leaf text",
                summary=None,
                parent_id="root",
                left_child_id=None,
                right_child_id=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
        ]

        store = SimpleMockStore()
        resolver = CharacterPositionResolver(nodes, store)

        # Test extent
        assert resolver.get_extent() == 1000.0

        # Test node positions
        assert resolver.get_node_position(nodes[0]) == (0.0, 1000.0)
        assert resolver.get_node_position(nodes[1]) == (0.0, 500.0)
        assert resolver.get_node_position(nodes[2]) == (500.0, 1000.0)

        # Test segment positions (same as node positions for character-based)
        seg = Segment("left", None)
        store.nodes["left"] = nodes[1]
        assert resolver.get_segment_position(seg, 0) == (0.0, 500.0)

    def test_token_position_resolver(self):
        """Test token-based position resolver."""
        # Create segments with token costs
        segment_infos = [
            SegmentInfo(Segment("left", None), 100),
            SegmentInfo(Segment("root", "RIGHT"), 50),
        ]

        coverage_map = {"left": True, "root": True, "right": True}

        # Set up mock store with nodes
        store = SimpleMockStore()
        store.nodes = {
            "root": TreeNode(
                id="root",
                depth=1,
                span_start=0,
                span_end=1000,
                text="Root summary",
                summary=None,
                parent_id=None,
                left_child_id="left",
                right_child_id="right",
                mid_offset=6,
                is_dirty=False,
                document_id="doc1",
            ),
            "left": TreeNode(
                id="left",
                depth=0,
                span_start=0,
                span_end=500,
                text="Left leaf text",
                summary=None,
                parent_id="root",
                left_child_id=None,
                right_child_id=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
            "right": TreeNode(
                id="right",
                depth=0,
                span_start=500,
                span_end=1000,
                text="Right leaf text",
                summary=None,
                parent_id="root",
                left_child_id=None,
                right_child_id=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
        }

        resolver = TokenPositionResolver(segment_infos, coverage_map, store)

        # Test extent (total tokens)
        assert resolver.get_extent() == 150.0

        # Test segment positions
        assert resolver.get_segment_position(segment_infos[0].segment, 0) == (
            0.0,
            100.0,
        )
        assert resolver.get_segment_position(segment_infos[1].segment, 1) == (
            100.0,
            150.0,
        )

        # Test node positions
        # Left node should match its segment position
        assert resolver.get_node_position(store.nodes["left"]) == (0.0, 100.0)
        # Root node should span both segments
        assert resolver.get_node_position(store.nodes["root"]) == (0.0, 150.0)
        # Right node has no selected segments, should be empty
        assert resolver.get_node_position(store.nodes["right"]) == (0.0, 0.0)

    def test_token_resolver_complex_tree(self):
        """Test token resolver with more complex tree structure."""
        # Create a more complex tree with partial selections
        segment_infos = [
            SegmentInfo(Segment("leaf1", None), 80),
            SegmentInfo(Segment("parent2", "LEFT"), 60),
            SegmentInfo(Segment("leaf4", None), 90),
        ]

        coverage_map = {
            "root": True,
            "parent1": True,
            "parent2": True,
            "leaf1": True,
            "leaf2": True,
            "leaf3": True,
            "leaf4": True,
        }

        store = SimpleMockStore()
        store.nodes = {
            "root": TreeNode(
                id="root",
                depth=2,
                span_start=0,
                span_end=400,
                parent_id=None,
                left_child_id="parent1",
                right_child_id="parent2",
                text="",
                summary=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
            "parent1": TreeNode(
                id="parent1",
                depth=1,
                span_start=0,
                span_end=200,
                parent_id="root",
                left_child_id="leaf1",
                right_child_id="leaf2",
                text="",
                summary=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
            "parent2": TreeNode(
                id="parent2",
                depth=1,
                span_start=200,
                span_end=400,
                parent_id="root",
                left_child_id="leaf3",
                right_child_id="leaf4",
                text="",
                summary=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
            "leaf1": TreeNode(
                id="leaf1",
                depth=0,
                span_start=0,
                span_end=100,
                parent_id="parent1",
                left_child_id=None,
                right_child_id=None,
                text="",
                summary=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
            "leaf2": TreeNode(
                id="leaf2",
                depth=0,
                span_start=100,
                span_end=200,
                parent_id="parent1",
                left_child_id=None,
                right_child_id=None,
                text="",
                summary=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
            "leaf3": TreeNode(
                id="leaf3",
                depth=0,
                span_start=200,
                span_end=300,
                parent_id="parent2",
                left_child_id=None,
                right_child_id=None,
                text="",
                summary=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
            "leaf4": TreeNode(
                id="leaf4",
                depth=0,
                span_start=300,
                span_end=400,
                parent_id="parent2",
                left_child_id=None,
                right_child_id=None,
                text="",
                summary=None,
                mid_offset=None,
                is_dirty=False,
                document_id="doc1",
            ),
        }

        resolver = TokenPositionResolver(segment_infos, coverage_map, store)

        # Total tokens: 80 + 60 + 90 = 230
        assert resolver.get_extent() == 230.0

        # Check segment positions
        assert resolver.get_segment_position(segment_infos[0].segment, 0) == (0.0, 80.0)
        assert resolver.get_segment_position(segment_infos[1].segment, 1) == (
            80.0,
            140.0,
        )
        assert resolver.get_segment_position(segment_infos[2].segment, 2) == (
            140.0,
            230.0,
        )

        # Check node positions
        # parent1 contains only leaf1 (selected)
        assert resolver.get_node_position(store.nodes["parent1"]) == (0.0, 80.0)
        # parent2 contains LEFT segment (60) and leaf4 (90)
        assert resolver.get_node_position(store.nodes["parent2"]) == (80.0, 230.0)
        # root contains everything
        assert resolver.get_node_position(store.nodes["root"]) == (0.0, 230.0)
        # leaf2 and leaf3 are not selected, should be empty
        assert resolver.get_node_position(store.nodes["leaf2"]) == (0.0, 0.0)
        assert resolver.get_node_position(store.nodes["leaf3"]) == (0.0, 0.0)
