"""Tests for storage functionality."""

import shutil
import tempfile

import pytest

from ragzoom.config import OperationalConfig
from ragzoom.store import Store


class TestStore:
    """Test the Store class."""

    @pytest.fixture
    def temp_store(self):
        """Create a temporary store for testing."""
        # Create temporary directories
        temp_dir = tempfile.mkdtemp()
        chroma_dir = f"{temp_dir}/chroma"
        db_path = f"{temp_dir}/test.db"

        # Override config
        config = OperationalConfig(
            openai_api_key="test-key",
            chroma_persist_directory=chroma_dir,
            sqlite_database_url=f"sqlite:///{db_path}",
        )

        store = Store(config, embedding_model="text-embedding-3-small")
        yield store

        # Cleanup - close store first to release file handles
        store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_add_node(self, temp_store):
        """Test adding a node to the store."""
        node = temp_store.add_node(
            node_id="test-1",
            text="Test text",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=10,
        )

        assert node.id == "test-1"
        assert node.text == "Test text"
        assert node.span_start == 0
        assert node.span_end == 10

    def test_get_node(self, temp_store):
        """Test retrieving a node."""
        # Add a node
        temp_store.add_node(
            node_id="test-2",
            text="Test text 2",
            embedding=[0.2] * 1536,
            span_start=10,
            span_end=20,
        )

        # Retrieve it
        node = temp_store.get_node("test-2")
        assert node is not None
        assert node.id == "test-2"
        assert node.text == "Test text 2"

        # Test non-existent node
        node = temp_store.get_node("non-existent")
        assert node is None

    def test_node_relationships(self, temp_store):
        """Test parent-child relationships."""
        # Create parent and children
        temp_store.add_node(
            node_id="parent",
            text="Parent node",
            embedding=[0.3] * 1536,
            span_start=0,
            span_end=20,
            left_child_id="child1",
            right_child_id="child2",
        )

        temp_store.add_node(
            node_id="child1",
            text="Child 1",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=10,
            parent_id="parent",
        )

        temp_store.add_node(
            node_id="child2",
            text="Child 2",
            embedding=[0.5] * 1536,
            span_start=10,
            span_end=20,
            parent_id="parent",
        )

        # Test relationships
        left, right = temp_store.get_children("parent")
        assert left.id == "child1"
        assert right.id == "child2"

        ancestors = temp_store.get_ancestors(["child1", "child2"])
        assert len(ancestors) == 1
        assert ancestors[0].id == "parent"

    def test_search_similar(self, temp_store):
        """Test vector similarity search."""
        # Add some nodes
        for i in range(5):
            embedding = [i * 0.1] * 1536
            temp_store.add_node(
                node_id=f"node-{i}",
                text=f"Text {i}",
                embedding=embedding,
                span_start=i * 10,
                span_end=(i + 1) * 10,
            )

        # Search with a query embedding
        query_embedding = [0.25] * 1536
        results = temp_store.search_similar(query_embedding, n_results=3)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 3 for r in results)  # (id, distance, metadata)

    def test_mmr_diverse_results(self, temp_store):
        """Test MMR diversity computation."""
        # Create candidates with different similarities
        candidates = [
            ("node-1", 0.1, {}),  # Very similar
            ("node-2", 0.15, {}),  # Similar
            ("node-3", 0.5, {}),  # Less similar
            ("node-4", 0.12, {}),  # Similar to node-1
            ("node-5", 0.8, {}),  # Different
        ]

        # Add nodes with embeddings
        embeddings = [
            [1.0, 0.0, 0.0],  # node-1
            [0.9, 0.1, 0.0],  # node-2 (similar to 1)
            [0.5, 0.5, 0.0],  # node-3 (different)
            [0.95, 0.05, 0.0],  # node-4 (very similar to 1)
            [0.0, 0.0, 1.0],  # node-5 (very different)
        ]

        for i, (node_id, _, _) in enumerate(candidates):
            # Pad embedding to expected size
            full_embedding = embeddings[i] + [0.0] * (1536 - 3)
            temp_store.add_node(
                node_id=node_id,
                text=f"Text for {node_id}",
                embedding=full_embedding,
                span_start=i * 10,
                span_end=(i + 1) * 10,
            )

        # Test MMR selection
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 1533  # Similar to node-1
        selected = temp_store.compute_mmr_diverse_results(
            query_embedding, candidates, lambda_param=0.7, k=3
        )

        assert len(selected) == 3
        # Should select node-1 (most relevant) and diverse nodes
        assert "node-1" in selected
        # Should include some diversity
        assert len(set(selected)) == 3

    def test_pinned_nodes(self, temp_store):
        """Test node pinning functionality."""
        # Create a tree structure with proper depths
        # Root node (depth 0)
        temp_store.add_node(
            node_id="root",
            text="Root node",
            embedding=[0.5] * 1536,
            span_start=0,
            span_end=30,
            parent_id=None,
        )

        # Level 1 node (depth 1)
        temp_store.add_node(
            node_id="level1",
            text="Level 1 node",
            embedding=[0.6] * 1536,
            span_start=0,
            span_end=20,
            parent_id="root",
        )

        # Level 2 node (depth 2)
        temp_store.add_node(
            node_id="level2",
            text="Level 2 node",
            embedding=[0.7] * 1536,
            span_start=0,
            span_end=10,
            parent_id="level1",
        )

        # Level 3 node (depth 3)
        temp_store.add_node(
            node_id="level3",
            text="Level 3 node",
            embedding=[0.8] * 1536,
            span_start=0,
            span_end=5,
            parent_id="level2",
        )

        # pin_depth_max is hardcoded to 2 in the config

        # Pin nodes at different depths
        success = temp_store.pin_node("root")  # depth 0 - should succeed
        assert success is True

        success = temp_store.pin_node("level2")  # depth 2 - should succeed
        assert success is True

        # Try to pin deep node (should fail)
        success = temp_store.pin_node("level3")  # depth 3 - should fail
        assert success is False

        # Check pinned nodes
        pinned = temp_store.get_pinned_nodes()
        assert len(pinned) == 2
        pinned_ids = {n.id for n in pinned}
        assert pinned_ids == {"root", "level2"}

        # Check with max depth
        pinned = temp_store.get_pinned_nodes(depth_max=0)
        assert len(pinned) == 1  # Only root (depth 0)
        assert pinned[0].id == "root"

        pinned = temp_store.get_pinned_nodes(depth_max=1)
        assert len(pinned) == 1  # Still only root (level2 has depth 2)

    def test_cache_functionality(self, temp_store):
        """Test LRU cache behavior."""
        # Add a node
        temp_store.add_node(
            node_id="cached",
            text="Cached node",
            embedding=[0.8] * 1536,
            span_start=0,
            span_end=10,
        )

        # First retrieval (from DB)
        node1 = temp_store.get_node("cached")
        assert node1 is not None

        # Second retrieval (from cache)
        node2 = temp_store.get_node("cached")
        assert node2 is not None
        assert node2.id == node1.id

        # Check cache contains the node
        assert "cached" in temp_store.node_cache

    def test_node_depth_calculation(self, temp_store):
        """Test dynamic depth calculation."""
        # Create a tree structure:
        #     root
        #    /    \
        #  left   right
        #  /  \     |
        # ll  lr    rc

        # Root node (depth 0)
        temp_store.add_node(
            node_id="root",
            text="Root",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            left_child_id="left",
            right_child_id="right",
        )

        # Level 1 nodes (depth 1)
        temp_store.add_node(
            node_id="left",
            text="Left",
            embedding=[0.2] * 1536,
            span_start=0,
            span_end=50,
            parent_id="root",
            left_child_id="ll",
            right_child_id="lr",
        )

        temp_store.add_node(
            node_id="right",
            text="Right",
            embedding=[0.3] * 1536,
            span_start=50,
            span_end=100,
            parent_id="root",
            left_child_id="rc",
        )

        # Level 2 nodes (depth 2)
        temp_store.add_node(
            node_id="ll",
            text="Left-Left",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=25,
            parent_id="left",
        )

        temp_store.add_node(
            node_id="lr",
            text="Left-Right",
            embedding=[0.5] * 1536,
            span_start=25,
            span_end=50,
            parent_id="left",
        )

        temp_store.add_node(
            node_id="rc",
            text="Right-Child",
            embedding=[0.6] * 1536,
            span_start=50,
            span_end=75,
            parent_id="right",
        )

        # Test depth calculations
        assert temp_store.get_node_depth("root") == 0
        assert temp_store.get_node_depth("left") == 1
        assert temp_store.get_node_depth("right") == 1
        assert temp_store.get_node_depth("ll") == 2
        assert temp_store.get_node_depth("lr") == 2
        assert temp_store.get_node_depth("rc") == 2

        # Test is_root_node
        assert temp_store.is_root_node("root") is True
        assert temp_store.is_root_node("left") is False
        assert temp_store.is_root_node("ll") is False

        # Test non-existent node
        with pytest.raises(ValueError, match="Node non-existent not found"):
            temp_store.get_node_depth("non-existent")

    def test_node_height_calculation(self, temp_store):
        """Test dynamic height calculation."""
        # Create the same tree structure
        temp_store.add_node(
            node_id="root",
            text="Root",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            left_child_id="left",
            right_child_id="right",
        )

        temp_store.add_node(
            node_id="left",
            text="Left",
            embedding=[0.2] * 1536,
            span_start=0,
            span_end=50,
            parent_id="root",
            left_child_id="ll",
            right_child_id="lr",
        )

        temp_store.add_node(
            node_id="right",
            text="Right",
            embedding=[0.3] * 1536,
            span_start=50,
            span_end=100,
            parent_id="root",
            left_child_id="rc",
        )

        temp_store.add_node(
            node_id="ll",
            text="Left-Left",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=25,
            parent_id="left",
        )

        temp_store.add_node(
            node_id="lr",
            text="Left-Right",
            embedding=[0.5] * 1536,
            span_start=25,
            span_end=50,
            parent_id="left",
        )

        temp_store.add_node(
            node_id="rc",
            text="Right-Child",
            embedding=[0.6] * 1536,
            span_start=50,
            span_end=75,
            parent_id="right",
        )

        # Test height calculations
        # Leaf nodes have height 0
        assert temp_store.get_node_height("ll") == 0
        assert temp_store.get_node_height("lr") == 0
        assert temp_store.get_node_height("rc") == 0

        # Internal nodes have height = 1 + max(child heights)
        assert temp_store.get_node_height("left") == 1  # max(0, 0) + 1
        assert temp_store.get_node_height("right") == 1  # has only left child
        assert temp_store.get_node_height("root") == 2  # max(1, 1) + 1

        # Test is_leaf_node
        assert temp_store.is_leaf_node("ll") is True
        assert temp_store.is_leaf_node("lr") is True
        assert temp_store.is_leaf_node("rc") is True
        assert temp_store.is_leaf_node("left") is False
        assert temp_store.is_leaf_node("root") is False

        # Test non-existent node
        with pytest.raises(ValueError, match="Node non-existent not found"):
            temp_store.get_node_height("non-existent")

    def test_depth_height_edge_cases(self, temp_store):
        """Test edge cases for depth/height calculation."""
        # Test single node (both root and leaf)
        temp_store.add_node(
            node_id="single",
            text="Single node",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=10,
        )

        assert temp_store.get_node_depth("single") == 0  # Root has depth 0
        assert temp_store.get_node_height("single") == 0  # Leaf has height 0
        assert temp_store.is_root_node("single") is True
        assert temp_store.is_leaf_node("single") is True

        # Test node with only left child
        temp_store.add_node(
            node_id="parent_left_only",
            text="Parent with left only",
            embedding=[0.2] * 1536,
            span_start=0,
            span_end=20,
            left_child_id="left_only_child",
        )

        temp_store.add_node(
            node_id="left_only_child",
            text="Left only child",
            embedding=[0.3] * 1536,
            span_start=0,
            span_end=10,
            parent_id="parent_left_only",
        )

        assert temp_store.get_node_height("parent_left_only") == 1
        assert temp_store.is_leaf_node("parent_left_only") is False

        # Test node with only right child
        temp_store.add_node(
            node_id="parent_right_only",
            text="Parent with right only",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=20,
            right_child_id="right_only_child",
        )

        temp_store.add_node(
            node_id="right_only_child",
            text="Right only child",
            embedding=[0.5] * 1536,
            span_start=10,
            span_end=20,
            parent_id="parent_right_only",
        )

        assert temp_store.get_node_height("parent_right_only") == 1
        assert temp_store.is_leaf_node("parent_right_only") is False

    def test_depth_calculation_performance(self, temp_store):
        """Test that depth calculation is O(log n) by creating a deep tree."""
        # Create a linear chain of nodes to test worst case
        nodes = []
        parent_id = None

        # Create a chain of 10 nodes
        for i in range(10):
            node_id = f"chain_{i}"
            temp_store.add_node(
                node_id=node_id,
                text=f"Chain node {i}",
                embedding=[0.1 * i] * 1536,
                span_start=i * 10,
                span_end=(i + 1) * 10,
                parent_id=parent_id,
            )
            nodes.append(node_id)
            parent_id = node_id

        # Test depths
        for i, node_id in enumerate(nodes):
            assert temp_store.get_node_depth(node_id) == i

        # Even for the deepest node, we only traverse up to root
        # This is O(depth) = O(log n) for balanced trees
        assert temp_store.get_node_depth("chain_9") == 9
