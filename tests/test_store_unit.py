"""Unit tests for storage functionality using mock store."""

import pytest

from tests.mock_store import SimpleMockStore


class TestStoreMock:
    """Test the Store interface using SimpleMockStore."""

    @pytest.fixture
    def mock_store(self):
        """Create a mock store for testing."""
        return SimpleMockStore()

    def test_add_node(self, mock_store):
        """Test adding a node to the store."""
        node = mock_store.add_node(
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

    def test_get_node(self, mock_store):
        """Test retrieving a node."""
        # Add a node
        mock_store.add_node(
            node_id="test-2",
            text="Test text 2",
            embedding=[0.2] * 1536,
            span_start=10,
            span_end=20,
        )

        # Retrieve it
        node = mock_store.get_node("test-2")
        assert node is not None
        assert node.id == "test-2"
        assert node.text == "Test text 2"

        # Test non-existent node
        node = mock_store.get_node("non-existent")
        assert node is None

    def test_node_relationships(self, mock_store):
        """Test parent-child relationships."""
        # Create parent and children
        mock_store.add_node(
            node_id="parent",
            text="Parent node",
            embedding=[0.3] * 1536,
            span_start=0,
            span_end=20,
            left_child_id="child1",
            right_child_id="child2",
        )

        mock_store.add_node(
            node_id="child1",
            text="Child 1",
            embedding=[0.4] * 1536,
            span_start=0,
            span_end=10,
            parent_id="parent",
        )

        mock_store.add_node(
            node_id="child2",
            text="Child 2",
            embedding=[0.5] * 1536,
            span_start=10,
            span_end=20,
            parent_id="parent",
        )

        # Test relationships
        left, right = mock_store.get_children("parent")
        assert left.id == "child1"
        assert right.id == "child2"

        ancestors = mock_store.get_ancestors(["child1", "child2"])
        assert len(ancestors) == 1
        assert ancestors[0].id == "parent"

    def test_search_similar(self, mock_store):
        """Test vector similarity search."""
        # Add some nodes
        for i in range(5):
            embedding = [i * 0.1] * 1536
            mock_store.add_node(
                node_id=f"node-{i}",
                text=f"Text {i}",
                embedding=embedding,
                span_start=i * 10,
                span_end=(i + 1) * 10,
            )

        # Search with a query embedding
        query_embedding = [0.25] * 1536
        results = mock_store.search_similar(query_embedding, n_results=3)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 3 for r in results)  # (id, distance, metadata)

    def test_session_local_count(self, mock_store):
        """Test that SessionLocal mock properly returns count."""
        # Add some nodes
        for i in range(5):
            mock_store.add_node(
                node_id=f"node-{i}",
                text=f"Text {i}",
                embedding=[0.1] * 1536,
                span_start=i * 10,
                span_end=(i + 1) * 10,
            )

        # Test SessionLocal count query (used by api.py and cli.py)
        with mock_store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            count = session.query(TreeNode).count()
            assert count == 5

    def test_add_node_returns_node(self, mock_store):
        """Test that add_node returns the created node object."""
        node = mock_store.add_node(
            node_id="test-return",
            text="Test return value",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=15,
        )

        # Should return the node object, not None
        assert node is not None
        assert node.id == "test-return"
        assert node.text == "Test return value"
