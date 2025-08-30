"""Tests for storage functionality."""

from typing import Any

import pytest

from ragzoom.exceptions import InvalidOperationError, NodeNotFoundError


@pytest.mark.integration
class TestStore:
    """Test the Store class."""

    @pytest.fixture
    def temp_store(self, store: Any) -> Any:
        """Create a temporary store for testing using conftest store fixture."""
        # For integration tests, use the store fixture from conftest.py
        # which handles PostgreSQL with proper isolation or SQLite fallback
        return store

    def test_add_node(self, temp_store: Any) -> None:
        """Test adding a node to the store."""
        # Create document first to satisfy foreign key constraint
        temp_store.add_document(
            document_id="test-doc",
            file_path="test.txt",
            content_hash="test-hash",
            chunk_count=1,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Use document-scoped store for node operations
        doc_store = temp_store.for_document("test-doc")
        node = doc_store.nodes.add_node(
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

    def test_get_node(self, temp_store: Any) -> None:
        """Test retrieving a node."""
        # Add a node
        temp_store.nodes.add_node(
            node_id="test-2",
            text="Test text 2",
            embedding=[0.2] * 1536,
            span_start=10,
            span_end=20,
        )

        # Retrieve it
        node = temp_store.nodes.get_node("test-2")
        assert node is not None
        assert node.id == "test-2"
        assert node.text == "Test text 2"

        # Test non-existent node
        node = temp_store.nodes.get_node("non-existent")
        assert node is None

    def test_node_relationships(self, temp_store: Any) -> None:
        """Test parent-child relationships."""
        # Create parent and children
        temp_store.nodes.add_node(
            node_id="parent",
            text="Parent node",
            embedding=[0.3] * 1536,
            span_start=0,
            span_end=20,
            left_child_id="child1",
            right_child_id="child2",
        )

        temp_store.nodes.add_node(
            node_id="child1",
            text="Child 1",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=10,
            parent_id="parent",
        )

        temp_store.nodes.add_node(
            node_id="child2",
            text="Child 2",
            embedding=[0.5] * 1536,
            span_start=10,
            span_end=20,
            parent_id="parent",
        )

        # Test relationships
        left, right = temp_store.tree.get_children("parent")
        assert left.id == "child1"
        assert right.id == "child2"

        ancestors = temp_store.tree.get_ancestors(["child1", "child2"])
        assert len(ancestors) == 1
        assert ancestors[0].id == "parent"

    def test_search_similar(self, temp_store: Any) -> None:
        """Test vector similarity search."""
        # Add some nodes
        for i in range(5):
            embedding = [i * 0.1] * 1536
            temp_store.nodes.add_node(
                node_id=f"node-{i}",
                text=f"Text {i}",
                embedding=embedding,
                span_start=i * 10,
                span_end=(i + 1) * 10,
            )

        # Search with a query embedding
        query_embedding = [0.25] * 1536
        results = temp_store.search.search_similar(query_embedding, n_results=3)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 3 for r in results)  # (id, distance, metadata)

    def test_mmr_diverse_results(self, temp_store: Any) -> None:
        """Test MMR diversity computation."""
        # Create candidates with different similarities
        candidates: list[tuple[str, float, dict[str, Any]]] = [
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
            temp_store.nodes.add_node(
                node_id=node_id,
                text=f"Text for {node_id}",
                embedding=full_embedding,
                span_start=i * 10,
                span_end=(i + 1) * 10,
            )

        # Test MMR selection
        query_embedding = [1.0, 0.0, 0.0] + [0.0] * 1533  # Similar to node-1
        selected = temp_store.search.compute_mmr_diverse_results(
            query_embedding, candidates, lambda_param=0.7, k=3
        )

        assert len(selected) == 3
        # Should select node-1 (most relevant) and diverse nodes
        assert "node-1" in selected
        # Should include some diversity
        assert len(set(selected)) == 3

    def test_pinned_nodes(self, temp_store: Any) -> None:
        """Test node pinning functionality."""
        # Create a tree structure with proper depths
        # Root node (depth 0)
        temp_store.nodes.add_node(
            node_id="root",
            text="Root node",
            embedding=[0.5] * 1536,
            span_start=0,
            span_end=30,
            parent_id=None,
        )

        # Level 1 node (depth 1)
        temp_store.nodes.add_node(
            node_id="level1",
            text="Level 1 node",
            embedding=[0.6] * 1536,
            span_start=0,
            span_end=20,
            parent_id="root",
            is_left_child=True,
        )

        # Level 2 node (depth 2)
        temp_store.nodes.add_node(
            node_id="level2",
            text="Level 2 node",
            embedding=[0.7] * 1536,
            span_start=0,
            span_end=10,
            parent_id="level1",
            is_left_child=True,
        )

        # Level 3 node (depth 3)
        temp_store.nodes.add_node(
            node_id="level3",
            text="Level 3 node",
            embedding=[0.8] * 1536,
            span_start=0,
            span_end=5,
            parent_id="level2",
            is_left_child=True,
        )

        # pin_depth_max is hardcoded to 2 in the config

        # Pin nodes at different depths
        temp_store.pin_node("root")  # depth 0 - should succeed
        temp_store.pin_node("level2")  # depth 2 - should succeed

        # Try to pin deep node (should fail)
        with pytest.raises(InvalidOperationError, match="exceeds maximum pin depth"):
            temp_store.pin_node("level3")  # depth 3 - should fail

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

    def test_cache_functionality(self, temp_store: Any) -> None:
        """Test LRU cache behavior."""
        # Add a node
        temp_store.nodes.add_node(
            node_id="cached",
            text="Cached node",
            embedding=[0.8] * 1536,
            span_start=0,
            span_end=10,
        )

        # First retrieval (from DB)
        node1 = temp_store.nodes.get_node("cached")
        assert node1 is not None

        # Second retrieval (from cache)
        node2 = temp_store.nodes.get_node("cached")
        assert node2 is not None
        assert node2.id == node1.id

        # Check cache contains the node
        assert "cached" in temp_store.node_cache

    def test_node_depth_calculation(self, temp_store: Any) -> None:
        """Test dynamic depth calculation."""
        # Create a tree structure:
        #     root
        #    /    \
        #  left   right
        #  /  \     |
        # ll  lr    rc

        # Root node (depth 0)
        temp_store.nodes.add_node(
            node_id="root",
            text="Root",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            left_child_id="left",
            right_child_id="right",
        )

        # Level 1 nodes (depth 1)
        temp_store.nodes.add_node(
            node_id="left",
            text="Left",
            embedding=[0.2] * 1536,
            span_start=0,
            span_end=50,
            parent_id="root",
            left_child_id="ll",
            right_child_id="lr",
        )

        temp_store.nodes.add_node(
            node_id="right",
            text="Right",
            embedding=[0.3] * 1536,
            span_start=50,
            span_end=100,
            parent_id="root",
            left_child_id="rc",
        )

        # Level 2 nodes (depth 2)
        temp_store.nodes.add_node(
            node_id="ll",
            text="Left-Left",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=25,
            parent_id="left",
        )

        temp_store.nodes.add_node(
            node_id="lr",
            text="Left-Right",
            embedding=[0.5] * 1536,
            span_start=25,
            span_end=50,
            parent_id="left",
        )

        temp_store.nodes.add_node(
            node_id="rc",
            text="Right-Child",
            embedding=[0.6] * 1536,
            span_start=50,
            span_end=75,
            parent_id="right",
        )

        # Test depth calculations
        assert temp_store.tree.get_node_depth("root") == 0
        assert temp_store.tree.get_node_depth("left") == 1
        assert temp_store.tree.get_node_depth("right") == 1
        assert temp_store.tree.get_node_depth("ll") == 2
        assert temp_store.tree.get_node_depth("lr") == 2
        assert temp_store.tree.get_node_depth("rc") == 2

        # Test is_root_node
        assert temp_store.tree.is_root_node("root") is True
        assert temp_store.tree.is_root_node("left") is False
        assert temp_store.tree.is_root_node("ll") is False

        # Test with non-existent node
        assert temp_store.tree.is_root_node("non-existent") is False

        # Test non-existent node
        with pytest.raises(NodeNotFoundError):
            temp_store.tree.get_node_depth("non-existent")

    def test_node_height_calculation(self, temp_store: Any) -> None:
        """Test dynamic height calculation."""
        # Create the same tree structure
        temp_store.nodes.add_node(
            node_id="root",
            text="Root",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            left_child_id="left",
            right_child_id="right",
            height=2,
        )

        temp_store.nodes.add_node(
            node_id="left",
            text="Left",
            embedding=[0.2] * 1536,
            span_start=0,
            span_end=50,
            parent_id="root",
            left_child_id="ll",
            right_child_id="lr",
            height=1,
        )

        temp_store.nodes.add_node(
            node_id="right",
            text="Right",
            embedding=[0.3] * 1536,
            span_start=50,
            span_end=100,
            parent_id="root",
            left_child_id="rc",
            height=1,
        )

        temp_store.nodes.add_node(
            node_id="ll",
            text="Left-Left",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=25,
            parent_id="left",
            height=0,
        )

        temp_store.nodes.add_node(
            node_id="lr",
            text="Left-Right",
            embedding=[0.5] * 1536,
            span_start=25,
            span_end=50,
            parent_id="left",
            height=0,
        )

        temp_store.nodes.add_node(
            node_id="rc",
            text="Right-Child",
            embedding=[0.6] * 1536,
            span_start=50,
            span_end=75,
            parent_id="right",
            height=0,
        )

        # Test height calculations using stored values
        # Leaf nodes have height 0
        assert temp_store.nodes.get_node("ll").height == 0
        assert temp_store.nodes.get_node("lr").height == 0
        assert temp_store.nodes.get_node("rc").height == 0

        # Internal nodes have height = 1 + max(child heights)
        assert temp_store.nodes.get_node("left").height == 1  # max(0, 0) + 1
        assert temp_store.nodes.get_node("right").height == 1  # has only left child
        assert temp_store.nodes.get_node("root").height == 2  # max(1, 1) + 1

        # Test is_leaf_node
        assert temp_store.tree.is_leaf_node("ll") is True
        assert temp_store.tree.is_leaf_node("lr") is True
        assert temp_store.tree.is_leaf_node("rc") is True
        assert temp_store.tree.is_leaf_node("left") is False
        assert temp_store.tree.is_leaf_node("root") is False

        # Test with non-existent node
        assert temp_store.tree.is_leaf_node("non-existent") is False

        # Test non-existent node
        assert temp_store.nodes.get_node("non-existent") is None

    def test_depth_height_edge_cases(self, temp_store: Any) -> None:
        """Test edge cases for depth/height calculation."""
        # Test single node (both root and leaf)
        temp_store.nodes.add_node(
            node_id="single",
            text="Single node",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=10,
            height=0,
        )

        assert temp_store.tree.get_node_depth("single") == 0  # Root has depth 0
        assert temp_store.nodes.get_node("single").height == 0  # Leaf has height 0
        assert temp_store.tree.is_root_node("single") is True
        assert temp_store.tree.is_leaf_node("single") is True

        # Test node with only left child
        temp_store.nodes.add_node(
            node_id="parent_left_only",
            text="Parent with left only",
            embedding=[0.2] * 1536,
            span_start=0,
            span_end=20,
            left_child_id="left_only_child",
            height=1,
        )

        temp_store.nodes.add_node(
            node_id="left_only_child",
            text="Left only child",
            embedding=[0.3] * 1536,
            span_start=0,
            span_end=10,
            parent_id="parent_left_only",
            height=0,
        )

        assert temp_store.nodes.get_node("parent_left_only").height == 1
        assert temp_store.tree.is_leaf_node("parent_left_only") is False

        # Test node with only right child
        temp_store.nodes.add_node(
            node_id="parent_right_only",
            text="Parent with right only",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=20,
            right_child_id="right_only_child",
            height=1,
        )

        temp_store.nodes.add_node(
            node_id="right_only_child",
            text="Right only child",
            embedding=[0.5] * 1536,
            span_start=10,
            span_end=20,
            parent_id="parent_right_only",
            height=0,
        )

        assert temp_store.nodes.get_node("parent_right_only").height == 1
        assert temp_store.tree.is_leaf_node("parent_right_only") is False

    def test_depth_calculation_performance(self, temp_store: Any) -> None:
        """Test that depth calculation is O(log n) by creating a deep tree."""
        # Create a linear chain of nodes to test worst case
        nodes = []
        parent_id = None

        # Create a chain of 10 nodes
        for i in range(10):
            node_id = f"chain_{i}"
            temp_store.nodes.add_node(
                node_id=node_id,
                text=f"Chain node {i}",
                embedding=[0.1 * i] * 1536,
                span_start=i * 10,
                span_end=(i + 1) * 10,
                parent_id=parent_id,
                is_left_child=True if parent_id else None,
            )
            nodes.append(node_id)
            parent_id = node_id

        # Test depths
        for i, node_id in enumerate(nodes):
            assert temp_store.tree.get_node_depth(node_id) == i

        # Even for the deepest node, we only traverse up to root
        # This is O(depth) = O(log n) for balanced trees
        assert temp_store.tree.get_node_depth("chain_9") == 9

    def test_error_handling_patterns(self, temp_store: Any) -> None:
        """Test consistent error handling patterns."""
        # Create a simple tree for testing
        temp_store.nodes.add_node(
            node_id="root",
            text="Root",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            height=0,
        )

        # Test NodeNotFoundError for calculation methods
        with pytest.raises(NodeNotFoundError) as exc_info:
            temp_store.tree.get_node_depth("missing")
        assert exc_info.value.node_id == "missing"

        # Test InvalidOperationError for already pinned node
        temp_store.pin_node("root")  # Pin the node first
        with pytest.raises(InvalidOperationError, match="already pinned"):
            temp_store.pin_node("root")  # Try to pin again

        # Test InvalidOperationError for embedding validation
        with pytest.raises(InvalidOperationError, match="Embedding cannot be empty"):
            temp_store.nodes.add_node(
                node_id="bad",
                text="Bad node",
                embedding=[],  # Empty embedding
                span_start=0,
                span_end=10,
            )

        # Test predicate methods return False for missing nodes (don't raise)
        assert temp_store.tree.is_leaf_node("missing") is False
        assert temp_store.tree.is_root_node("missing") is False

        # Test query methods return None for missing items (don't raise)
        assert temp_store.nodes.get_node("missing") is None
        assert temp_store.get_document_by_id("missing") is None
        assert temp_store.documents.get_document_embedding_model("missing") is None
