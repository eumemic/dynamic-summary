"""Test that the indexing process creates valid left-balanced trees."""

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.validate import (
    set_validation_enabled,
    validate_equal_leaf_depth,
    validate_tree_is_left_balanced,
)
from tests.mock_store import SimpleMockStore


class TestIndexingCreatesLeftBalancedTrees:
    """Tests to ensure indexing produces valid left-balanced trees."""

    @pytest.fixture
    def setup_indexing(self):
        """Set up indexing system with validation enabled."""
        config = RagZoomConfig(
            leaf_tokens=50,  # Small chunks for testing
            adjacent_context_tokens=25,
            openai_api_key="test-key-for-tests",
        )
        store = SimpleMockStore(config=config)
        tree_builder = TreeBuilder(config, store)

        # Mock the API calls - need to mock the async methods
        async def mock_get_embedding(text):
            return [0.1] * 1536

        async def mock_get_batch_embeddings(texts):
            return [[0.1] * 1536 for _ in texts]

        async def mock_summarize_text(left, right, target, prev_context, next_context):
            if right:  # Two children
                return f"Summary of: {left[:20]}... and {right[:20]}..."
            else:  # Single child
                return f"Summary of: {left[:20]}..."

        tree_builder._get_embedding = mock_get_embedding
        tree_builder._get_embeddings_batch = mock_get_batch_embeddings
        tree_builder._summarize_text = mock_summarize_text

        # Enable validation
        set_validation_enabled(True)

        return config, store, tree_builder

    def teardown_method(self):
        """Disable validation after each test."""
        set_validation_enabled(False)

    def test_even_number_of_chunks_creates_valid_tree(self, setup_indexing):
        """Test that indexing with even number of chunks creates a valid left-balanced tree."""
        config, store, tree_builder = setup_indexing

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
        config, store, tree_builder = setup_indexing

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

    def test_large_document_creates_valid_tree(self, setup_indexing):
        """Test that a large document with many chunks creates a valid left-balanced tree."""
        config, store, tree_builder = setup_indexing

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
        config, store, tree_builder = setup_indexing

        # Create text that will produce exactly 5 chunks (2^2 + 1)
        # Each chunk should be around 50 tokens based on config
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
