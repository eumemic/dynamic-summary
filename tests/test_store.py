"""Tests for storage functionality."""

import shutil
import tempfile

import pytest

from ragzoom.config import RagZoomConfig
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
        config = RagZoomConfig(
            openai_api_key="test-key",
            chroma_persist_directory=chroma_dir,
            sqlite_database_url=f"sqlite:///{db_path}",
        )

        store = Store(config)
        yield store

        # Cleanup - close store first to release file handles
        store.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_add_node(self, temp_store):
        """Test adding a node to the store."""
        node = temp_store.add_node(
            node_id="test-1",
            text="Test text",
            embedding=[0.1] * 384,
            depth=0,
            span_start=0,
            span_end=10,
        )

        assert node.id == "test-1"
        assert node.text == "Test text"
        assert node.depth == 0
        assert node.span_start == 0
        assert node.span_end == 10

    def test_get_node(self, temp_store):
        """Test retrieving a node."""
        # Add a node
        temp_store.add_node(
            node_id="test-2",
            text="Test text 2",
            embedding=[0.2] * 384,
            depth=1,
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
            embedding=[0.3] * 384,
            depth=1,
            span_start=0,
            span_end=20,
            left_child_id="child1",
            right_child_id="child2",
            summary="Parent summary",
        )

        temp_store.add_node(
            node_id="child1",
            text="Child 1",
            embedding=[0.4] * 384,
            depth=0,
            span_start=0,
            span_end=10,
            parent_id="parent",
        )

        temp_store.add_node(
            node_id="child2",
            text="Child 2",
            embedding=[0.5] * 384,
            depth=0,
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
            embedding = [i * 0.1] * 384
            temp_store.add_node(
                node_id=f"node-{i}",
                text=f"Text {i}",
                embedding=embedding,
                depth=0,
                span_start=i * 10,
                span_end=(i + 1) * 10,
            )

        # Search with a query embedding
        query_embedding = [0.25] * 384
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
            full_embedding = embeddings[i] + [0.0] * (384 - 3)
            temp_store.add_node(
                node_id=node_id,
                text=f"Text for {node_id}",
                embedding=full_embedding,
                depth=0,
                span_start=i * 10,
                span_end=(i + 1) * 10,
            )

        # Test MMR selection
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 381  # Similar to node-1
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
        # Add nodes at different depths
        temp_store.add_node(
            node_id="shallow",
            text="Shallow node",
            embedding=[0.6] * 384,
            depth=1,
            span_start=0,
            span_end=10,
        )

        temp_store.add_node(
            node_id="deep",
            text="Deep node",
            embedding=[0.7] * 384,
            depth=5,
            span_start=10,
            span_end=20,
        )

        # Pin shallow node (should succeed)
        temp_store.config.pin_depth_max = 2
        success = temp_store.pin_node("shallow")
        assert success is True

        # Try to pin deep node (should fail)
        success = temp_store.pin_node("deep")
        assert success is False

        # Check pinned nodes
        pinned = temp_store.get_pinned_nodes()
        assert len(pinned) == 1
        assert pinned[0].id == "shallow"

        # Check with max depth
        pinned = temp_store.get_pinned_nodes(max_depth=0)
        assert len(pinned) == 0

    def test_cache_functionality(self, temp_store):
        """Test LRU cache behavior."""
        # Add a node
        temp_store.add_node(
            node_id="cached",
            text="Cached node",
            embedding=[0.8] * 384,
            depth=0,
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

    def test_dirty_marking(self, temp_store):
        """Test marking nodes as dirty."""
        # Create a tree structure
        temp_store.add_node(
            node_id="root",
            text="Root",
            embedding=[0.9] * 384,
            depth=2,
            span_start=0,
            span_end=40,
        )

        temp_store.add_node(
            node_id="parent",
            text="Parent",
            embedding=[0.85] * 384,
            depth=1,
            span_start=0,
            span_end=20,
            parent_id="root",
        )

        temp_store.add_node(
            node_id="child",
            text="Child",
            embedding=[0.8] * 384,
            depth=0,
            span_start=0,
            span_end=10,
            parent_id="parent",
        )

        # Mark child as dirty
        temp_store.mark_dirty_upward("child")

        # Check all ancestors are marked dirty
        child_node = temp_store.get_node("child")
        parent_node = temp_store.get_node("parent")
        root_node = temp_store.get_node("root")

        # Debug: Check if nodes exist and have correct parent relationships
        assert child_node is not None, "Child node not found"
        assert parent_node is not None, "Parent node not found"
        assert root_node is not None, "Root node not found"
        assert (
            child_node.parent_id == "parent"
        ), f"Child parent_id is {child_node.parent_id}"
        assert (
            parent_node.parent_id == "root"
        ), f"Parent parent_id is {parent_node.parent_id}"

        # Now check is_dirty
        assert child_node.is_dirty == 1, f"Child is_dirty is {child_node.is_dirty}"
        assert parent_node.is_dirty == 1, f"Parent is_dirty is {parent_node.is_dirty}"
        assert root_node.is_dirty == 1, f"Root is_dirty is {root_node.is_dirty}"
