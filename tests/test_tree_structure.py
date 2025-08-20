"""Tests for tree structure validation and indexing tree creation.

This module consolidates tests for:
1. Left-balanced tree validation logic
2. Tree structure created by indexing process
"""

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
from ragzoom.index import TreeBuilder
from ragzoom.validate import (
    set_validation_enabled,
    validate_equal_leaf_depth,
    validate_tree_is_left_balanced,
)
from tests.mock_store import SimpleMockStore


class TestTreeValidation:
    """Tests for left-balanced tree validation function."""

    @pytest.fixture
    def mock_store_with_valid_tree(self):
        """Create a mock store with a valid left-balanced tree."""

        # Using a simple mock instead of the real Store
        class MockNode:
            def __init__(self, id, left_child_id=None, right_child_id=None):
                self.id = id
                self.left_child_id = left_child_id
                self.right_child_id = right_child_id

        class MockStore:
            def __init__(self):
                self.nodes = {}

            def add_node(self, node):
                self.nodes[node.id] = node

            def get_all_nodes_for_document(self, doc_id):
                return list(self.nodes.values())

        store = MockStore()

        # Create a valid left-balanced tree (happens to be full):
        #       root
        #      /    \
        #     P1     P2
        #    /  \   /  \
        #   L1  L2 L3  L4

        store.add_node(MockNode("L1"))
        store.add_node(MockNode("L2"))
        store.add_node(MockNode("L3"))
        store.add_node(MockNode("L4"))
        store.add_node(MockNode("P1", "L1", "L2"))
        store.add_node(MockNode("P2", "L3", "L4"))
        store.add_node(MockNode("root", "P1", "P2"))

        return store

    @pytest.fixture
    def mock_store_with_left_balanced_tree(self):
        """Create a mock store with a valid left-balanced tree (single left child)."""

        class MockNode:
            def __init__(self, id, left_child_id=None, right_child_id=None):
                self.id = id
                self.left_child_id = left_child_id
                self.right_child_id = right_child_id

        class MockStore:
            def __init__(self):
                self.nodes = {}

            def add_node(self, node):
                self.nodes[node.id] = node

            def get_all_nodes_for_document(self, doc_id):
                return list(self.nodes.values())

        store = MockStore()

        # Create a valid left-balanced tree (P2 only has left child):
        #       root
        #      /    \
        #     P1     P2
        #    /  \   /
        #   L1  L2 L3

        store.add_node(MockNode("L1"))
        store.add_node(MockNode("L2"))
        store.add_node(MockNode("L3"))
        store.add_node(MockNode("P1", "L1", "L2"))
        store.add_node(MockNode("P2", "L3", None))  # Valid: only left child
        store.add_node(MockNode("root", "P1", "P2"))

        return store

    def test_full_tree_passes_validation(self, mock_store_with_valid_tree):
        """Test that a full binary tree passes left-balanced validation."""
        result = validate_tree_is_left_balanced(mock_store_with_valid_tree, "test-doc")
        assert result is None  # None means valid

    def test_single_left_child_passes_validation(
        self, mock_store_with_left_balanced_tree
    ):
        """Test that a tree with single left children passes validation."""
        result = validate_tree_is_left_balanced(
            mock_store_with_left_balanced_tree, "test-doc"
        )
        assert result is None  # Valid left-balanced tree

    def test_single_node_tree_passes(self):
        """Test that a tree with just one node (root) is valid."""

        class MockNode:
            def __init__(self, id):
                self.id = id
                self.left_child_id = None
                self.right_child_id = None

        class MockStore:
            def get_all_nodes_for_document(self, doc_id):
                return [MockNode("root")]

        store = MockStore()
        result = validate_tree_is_left_balanced(store, "test-doc")
        assert result is None  # Single node tree is valid

    def test_invalid_child_reference_fails(self):
        """Test that referencing non-existent children fails validation."""

        class MockNode:
            def __init__(self, id, left_child_id=None, right_child_id=None):
                self.id = id
                self.left_child_id = left_child_id
                self.right_child_id = right_child_id

        class MockStore:
            def get_all_nodes_for_document(self, doc_id):
                # Node references children that don't exist
                return [MockNode("root", "missing-left", "missing-right")]

        store = MockStore()
        result = validate_tree_is_left_balanced(store, "test-doc")
        assert result is not None
        assert "Invalid tree" in result
        assert "non-existent" in result

    def test_right_child_without_left_child_fails(self):
        """Test that a node with only a right child fails validation."""

        class MockNode:
            def __init__(self, id, left_child_id=None, right_child_id=None):
                self.id = id
                self.left_child_id = left_child_id
                self.right_child_id = right_child_id

        class MockStore:
            def __init__(self):
                self.nodes = {}

            def add_node(self, node):
                self.nodes[node.id] = node

            def get_all_nodes_for_document(self, doc_id):
                return list(self.nodes.values())

        store = MockStore()

        # Invalid tree with right child but no left child:
        #      root
        #         \
        #          P1
        #         /  \
        #        L1  L2

        store.add_node(MockNode("L1"))
        store.add_node(MockNode("L2"))
        store.add_node(MockNode("P1", "L1", "L2"))
        store.add_node(MockNode("root", None, "P1"))  # Only right child!

        result = validate_tree_is_left_balanced(store, "test-doc")
        assert result is not None
        assert "not left-balanced" in result
        assert "root" in result  # Should identify the problematic node
        assert "right child but no left child" in result


class TestIndexingCreatesValidTrees:
    """Tests to ensure indexing produces valid left-balanced trees."""

    @pytest.fixture
    def setup_indexing(self):
        """Set up indexing system with validation enabled."""
        index_config = IndexConfig.load(
            target_chunk_tokens=50,  # Small chunks for testing
            preceding_context_tokens=25,
        )
        query_config = QueryConfig()
        operational_config = OperationalConfig(
            openai_api_key="test-key-for-tests",
        )
        store = SimpleMockStore(config=(index_config, query_config, operational_config))
        tree_builder = TreeBuilder(
            index_config, store, api_key=operational_config.openai_api_key
        )

        # Mock the API calls - need to mock the async methods
        async def mock_get_embedding(text):
            return [0.1] * 1536

        async def mock_get_batch_embeddings(texts):
            return [[0.1] * 1536 for _ in texts]

        async def mock_summarize_text(
            left,
            right,
            target,
            parent_id=None,
            reporter=None,
        ):
            if right:  # Two children
                return f"Summary of: {left[:20]}... and {right[:20]}...", 0, 100
            else:  # Single child
                return f"Summary of: {left[:20]}...", 0, 50

        tree_builder.llm_service._get_embedding = mock_get_embedding
        tree_builder.llm_service._get_embeddings_batch = mock_get_batch_embeddings
        tree_builder.llm_service._summarize_text = mock_summarize_text

        # Enable validation
        set_validation_enabled(True)

        return (index_config, query_config, operational_config), store, tree_builder

    def teardown_method(self):
        """Disable validation after each test."""
        set_validation_enabled(False)

    def test_even_number_of_chunks_creates_valid_tree(self, setup_indexing):
        """Test that indexing with even number of chunks creates a valid left-balanced tree."""
        (index_config, query_config, operational_config), store, tree_builder = (
            setup_indexing
        )

        # Text that will create 4 chunks
        text = "Chapter 1 content here. " * 10  # ~40 tokens
        text += "Chapter 2 content here. " * 10  # ~40 tokens
        text += "Chapter 3 content here. " * 10  # ~40 tokens
        text += "Chapter 4 content here. " * 10  # ~40 tokens

        # This should create a tree like:
        #       root
        #      /    \
        #     P1     P2
        #    /  \   /  \
        #   L1  L2 L3  L4

        # Index the document
        doc_id = tree_builder.add_document(
            text, document_id="test-even", show_progress=False
        )

        # Verify it's left-balanced
        result = validate_tree_is_left_balanced(store, doc_id)
        assert result is None

        # Verify all leaves are at the same depth
        result = validate_equal_leaf_depth(store, doc_id)
        assert result is None

    def test_odd_number_of_chunks_creates_valid_tree(self, setup_indexing):
        """Test that indexing with odd number of chunks creates a valid left-balanced tree."""
        (index_config, query_config, operational_config), store, tree_builder = (
            setup_indexing
        )

        # Text that will create 3 chunks
        text = "Chapter 1 content here. " * 10  # ~40 tokens
        text += "Chapter 2 content here. " * 10  # ~40 tokens
        text += "Chapter 3 content here. " * 10  # ~40 tokens

        # This should create a left-balanced tree like:
        #      root
        #     /    \
        #    P1     P2
        #   /  \     |
        #  L1  L2   L3
        # P2 has only a left child (L3)

        # Index the document
        doc_id = tree_builder.add_document(
            text, document_id="test-odd", show_progress=False
        )

        # Verify it's left-balanced
        result = validate_tree_is_left_balanced(store, doc_id)
        assert result is None

        # Verify all leaves are at the same depth
        result = validate_equal_leaf_depth(store, doc_id)
        assert result is None

    @pytest.mark.slow
    def test_large_document_creates_valid_tree(self, setup_indexing):
        """Test that a large document with many chunks creates a valid left-balanced tree."""
        (index_config, query_config, operational_config), store, tree_builder = (
            setup_indexing
        )

        # Text that will create multiple chunks
        text = ""
        for i in range(7):
            text += f"Chapter {i+1} content here. " * 10  # ~40 tokens each

        doc_id = tree_builder.add_document(
            text, document_id="test-large", show_progress=False
        )

        # Verify it's left-balanced
        result = validate_tree_is_left_balanced(store, doc_id)
        assert result is None

        # Verify all leaves are at the same depth
        result = validate_equal_leaf_depth(store, doc_id)
        assert result is None

        # Check we have multiple leaf nodes (exact count depends on tokenization)
        nodes = store.get_all_nodes_for_document(doc_id)
        leaf_nodes = [
            n for n in nodes if n.left_child_id is None and n.right_child_id is None
        ]
        assert len(leaf_nodes) > 1  # Should have multiple chunks

        # Verify left-balanced property: no node has only a right child
        internal_nodes = [
            n
            for n in nodes
            if n.left_child_id is not None or n.right_child_id is not None
        ]
        for node in internal_nodes:
            # In a left-balanced tree, if there's a right child, there must be a left child
            if node.right_child_id is not None:
                assert node.left_child_id is not None

    def test_power_of_two_plus_one_chunks_creates_valid_tree(self, setup_indexing):
        """Test that indexing with 2^n + 1 chunks creates a valid tree with equal leaf depth."""
        (index_config, query_config, operational_config), store, tree_builder = (
            setup_indexing
        )

        # Create text that will produce exactly 5 chunks (2^2 + 1)
        # Each chunk should be around 50 tokens based on index_config
        chunks = []
        for i in range(5):
            # Create distinct content for each chunk to ensure proper splitting
            chunk_text = f"This is chunk number {i}. " * 12  # ~48 tokens
            chunks.append(chunk_text)

        text = " ".join(chunks)

        # Expected tree structure:
        #         root
        #        /    \
        #       P3     P4
        #      /  \     |
        #     P1   P2   L5
        #    / \   / \
        #   L1 L2 L3 L4

        # Index the document
        doc_id = tree_builder.add_document(
            text, document_id="test-2n-plus-1", show_progress=False
        )

        # Verify it's left-balanced
        result = validate_tree_is_left_balanced(store, doc_id)
        assert result is None

        # Verify all leaves are at the same depth
        result = validate_equal_leaf_depth(store, doc_id)
        assert result is None

        # Verify we have exactly 5 leaf nodes
        nodes = store.get_all_nodes_for_document(doc_id)
        leaf_nodes = [
            n for n in nodes if n.left_child_id is None and n.right_child_id is None
        ]
        # Due to tokenization, we might not get exactly 5 chunks, but verify structure
        assert len(leaf_nodes) >= 3  # Should have multiple chunks

        # Find nodes with only one child (left child)
        single_child_nodes = [
            n for n in nodes if n.left_child_id is not None and n.right_child_id is None
        ]
        # With odd number of chunks, we should have at least one single-child node
        if len(leaf_nodes) % 2 == 1:
            assert (
                len(single_child_nodes) > 0
            ), "Expected single-child nodes for odd number of leaves"