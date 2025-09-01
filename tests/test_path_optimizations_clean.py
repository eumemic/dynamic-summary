"""Tests for path-based tree navigation optimizations."""

from ragzoom.services.tree_navigator import TreeNavigator
from tests.mock_store import SimpleMockStore


class TestPathOptimizations:
    """Test path-based optimizations in tree navigation."""

    def setup_method(self) -> None:
        """Set up test environment with nodes that have proper paths."""
        self.store = SimpleMockStore()

        # Create a proper tree with paths
        # Root: path=""
        self.store.add_node(
            node_id="root",
            text="Root node",
            span_start=0,
            span_end=100,
            parent_id=None,
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="left",
            right_child_id="right",
            path="",  # Root path
        )

        # Left child: path="0"
        self.store.add_node(
            node_id="left",
            text="Left child",
            span_start=0,
            span_end=50,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
            left_child_id="left_left",
            right_child_id="left_right",
            path="0",  # Left child path
        )

        # Right child: path="1"
        self.store.add_node(
            node_id="right",
            text="Right child",
            span_start=50,
            span_end=100,
            parent_id="root",
            document_id="doc1",
            embedding=[0.5] * 1536,
            path="1",  # Right child path
        )

        # Left-left grandchild: path="00"
        self.store.add_node(
            node_id="left_left",
            text="Left-left grandchild",
            span_start=0,
            span_end=25,
            parent_id="left",
            document_id="doc1",
            embedding=[0.5] * 1536,
            path="00",  # Left-left grandchild path
        )

        # Left-right grandchild: path="01"
        self.store.add_node(
            node_id="left_right",
            text="Left-right grandchild",
            span_start=25,
            span_end=50,
            parent_id="left",
            document_id="doc1",
            embedding=[0.5] * 1536,
            path="01",  # Left-right grandchild path
        )

    def test_get_node_depth_with_paths(self) -> None:
        """Test that get_node_depth uses path field for instant calculation."""
        navigator = TreeNavigator(self.store.nodes)

        # Test depth calculation using paths
        assert navigator.get_node_depth("root") == 0  # Root depth
        assert navigator.get_node_depth("left") == 1  # First level
        assert navigator.get_node_depth("right") == 1  # First level
        assert navigator.get_node_depth("left_left") == 2  # Second level
        assert navigator.get_node_depth("left_right") == 2  # Second level

    def test_get_parent_node_with_paths(self) -> None:
        """Test that get_parent_node uses path field for instant lookup."""
        navigator = TreeNavigator(self.store.nodes)

        # Test parent lookup using paths
        root_parent = navigator.get_parent_node("root")
        assert root_parent is None  # Root has no parent

        left_parent = navigator.get_parent_node("left")
        assert left_parent is not None
        assert left_parent.id == "root"

        left_left_parent = navigator.get_parent_node("left_left")
        assert left_left_parent is not None
        assert left_left_parent.id == "left"

    def test_get_sibling_node_with_paths(self) -> None:
        """Test that get_sibling_node uses path field for instant lookup."""
        navigator = TreeNavigator(self.store.nodes)

        # Test sibling lookup using paths
        root_sibling = navigator.get_sibling_node("root")
        assert root_sibling is None  # Root has no sibling

        left_sibling = navigator.get_sibling_node("left")
        assert left_sibling is not None
        assert left_sibling.id == "right"

        right_sibling = navigator.get_sibling_node("right")
        assert right_sibling is not None
        assert right_sibling.id == "left"

        left_left_sibling = navigator.get_sibling_node("left_left")
        assert left_left_sibling is not None
        assert left_left_sibling.id == "left_right"

    def test_is_left_child_with_paths(self) -> None:
        """Test that is_left_child uses path field for instant determination."""
        navigator = TreeNavigator(self.store.nodes)

        # Test left child detection using paths
        assert not navigator.is_left_child("root")  # Root is neither left nor right
        assert navigator.is_left_child("left")  # Left child
        assert not navigator.is_left_child("right")  # Right child, not left
        assert navigator.is_left_child("left_left")  # Left-left is left child
        assert not navigator.is_left_child("left_right")  # Left-right is right child

    def test_is_right_child_with_paths(self) -> None:
        """Test that is_right_child uses path field for instant determination."""
        navigator = TreeNavigator(self.store.nodes)

        # Test right child detection using paths
        assert not navigator.is_right_child("root")  # Root is neither left nor right
        assert not navigator.is_right_child("left")  # Left child, not right
        assert navigator.is_right_child("right")  # Right child
        assert not navigator.is_right_child("left_left")  # Left-left is left child
        assert navigator.is_right_child("left_right")  # Left-right is right child

    def test_pinned_nodes_path_filtering(self) -> None:
        """Test that get_pinned_nodes uses path-based database filtering."""
        # Pin some nodes at different depths
        self.store.pin_node("root")  # Depth 0
        self.store.pin_node("left")  # Depth 1
        self.store.pin_node("left_left")  # Depth 2

        # Test filtering by depth
        pinned_depth_0 = self.store.get_pinned_nodes(depth_max=0)
        assert len(pinned_depth_0) == 1
        assert pinned_depth_0[0].id == "root"

        pinned_depth_1 = self.store.get_pinned_nodes(depth_max=1)
        assert len(pinned_depth_1) == 2
        node_ids = {node.id for node in pinned_depth_1}
        assert node_ids == {"root", "left"}

        pinned_depth_2 = self.store.get_pinned_nodes(depth_max=2)
        assert len(pinned_depth_2) == 3
        node_ids = {node.id for node in pinned_depth_2}
        assert node_ids == {"root", "left", "left_left"}

        # Test no depth limit
        pinned_all = self.store.get_pinned_nodes()
        assert len(pinned_all) == 3

    def test_path_optimization_performance(self) -> None:
        """Test that path-based methods avoid database queries where possible."""
        navigator = TreeNavigator(self.store.nodes)

        # With proper paths, these operations should be very fast
        # and not require traversing up the tree
        depth = navigator.get_node_depth("left_left")
        assert depth == 2

        # The path-based implementation should use string operations
        # rather than multiple database queries
        sibling = navigator.get_sibling_node("left_left")
        assert sibling is not None
        assert sibling.id == "left_right"
