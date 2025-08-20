"""Unit tests for storage functionality using mock store.

This file focuses on mock-specific functionality and interface compliance tests.
Basic CRUD tests have been removed to eliminate duplication with test_store.py,
which provides comprehensive integration testing.
"""

import pytest

from tests.mock_store import SimpleMockStore


class TestStoreMock:
    """Test the Store interface using SimpleMockStore.

    This class tests mock-specific functionality and interface compliance.
    Basic CRUD operations are tested in test_store.py integration tests.
    """

    @pytest.fixture
    def mock_store(self):
        """Create a mock store for testing."""
        return SimpleMockStore()

    # NOTE: Basic CRUD tests (test_add_node, test_get_node, test_node_relationships,
    # test_search_similar) have been removed to eliminate duplication with test_store.py.
    # See test_store.py for comprehensive integration testing of these operations.

    def test_session_local_count(self, mock_store):
        """Test that SessionLocal mock properly returns count."""
        # Add some nodes
        for i in range(3):
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
            assert count == 3

    def test_add_node_returns_node(self, mock_store):
        """Test that add_node returns the created node."""
        node = mock_store.add_node(
            node_id="return-test",
            text="Return test text",
            embedding=[0.3] * 1536,
            span_start=0,
            span_end=10,
        )

        # Should return the node object
        assert node is not None
        assert node.id == "return-test"
        assert node.text == "Return test text"

    def test_document_operations(self, mock_store):
        """Test document operations."""
        # Create document directly
        doc = mock_store.add_document(
            document_id="test-doc",
            file_path="/test/file.txt",
            content_hash="abc123",
            chunk_count=3,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        assert doc.id == "test-doc"
        assert doc.file_path == "/test/file.txt"
        assert doc.chunk_count == 3

        # Test retrieval if method exists
        if hasattr(mock_store, "get_document_by_id"):
            retrieved = mock_store.get_document_by_id("test-doc")
            assert retrieved.id == "test-doc"

    def test_interface_compliance(self, mock_store):
        """Test that mock store implements the core interface."""
        # Test that core methods exist (only those actually implemented)
        core_methods = [
            "add_node",
            "get_node",
            "search_similar",
            "get_children",
            "get_ancestors",
        ]

        for method_name in core_methods:
            assert hasattr(mock_store, method_name), f"Missing method: {method_name}"
            assert callable(
                getattr(mock_store, method_name)
            ), f"Not callable: {method_name}"

    def test_real_store_interface_compliance(self):
        """Test that real Store class has the same interface as mock."""
        from ragzoom.store import Store

        # Get method names from both classes
        mock_methods = {
            name for name in dir(SimpleMockStore) if not name.startswith("_")
        }
        real_methods = {name for name in dir(Store) if not name.startswith("_")}

        # Mock should implement core Store methods
        core_methods = {
            "add_node",
            "get_node",
            "search_similar",
            "get_children",
            "get_ancestors",
        }
        for method in core_methods:
            assert method in mock_methods, f"Mock missing core method: {method}"
            assert method in real_methods, f"Store missing core method: {method}"

    def test_builder_advanced_features(self, mock_store, tree_node_builder):
        """Test advanced builder features with mock store."""
        # Test complex node creation with builder
        node_data = (
            tree_node_builder.with_id("advanced-test")
            .with_text("Advanced test text")
            .with_span(100, 200)
            .with_height(2)
            .with_document("advanced-doc")
            .build("dict")
        )

        node = mock_store.add_node(**node_data)

        assert node.id == "advanced-test"
        assert node.span_start == 100
        assert node.span_end == 200
        assert node.height == 2
        assert node.document_id == "advanced-doc"
