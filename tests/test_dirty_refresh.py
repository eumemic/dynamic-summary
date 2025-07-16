"""Test dirty node refresh functionality."""

import asyncio
import tempfile
from unittest.mock import Mock, patch

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import Store


class TestDirtyRefresh:
    """Test dirty node refresh and re-summarization."""

    @pytest.fixture
    def setup_system(self):
        """Set up a test system with mocked API."""
        # Mock OpenAI clients
        with (
            patch("ragzoom.index.AsyncOpenAI") as mock_index_client,
            patch("ragzoom.retrieve.OpenAI") as mock_retrieve_client,
        ):

            # Setup async mocks for indexing
            async def mock_embeddings_create(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    return Mock(data=[Mock(embedding=[0.1] * 384) for _ in input_data])
                else:
                    return Mock(data=[Mock(embedding=[0.1] * 384)])

            async def mock_chat_create(*args, **kwargs):
                # Return fresh summary with MID delimiter
                return Mock(
                    choices=[
                        Mock(
                            message=Mock(
                                content="Fresh left summary. <<<MID>>> Fresh right summary."
                            )
                        )
                    ]
                )

            # Setup sync mocks for retrieval
            def mock_embeddings_create_sync(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    return Mock(data=[Mock(embedding=[0.1] * 384) for _ in input_data])
                else:
                    return Mock(data=[Mock(embedding=[0.1] * 384)])

            # Configure mocks
            mock_embeddings_async = Mock()
            mock_embeddings_async.create = Mock(side_effect=mock_embeddings_create)

            mock_embeddings_sync = Mock()
            mock_embeddings_sync.create = Mock(side_effect=mock_embeddings_create_sync)

            mock_chat_async = Mock()
            mock_chat_async.completions = Mock()
            mock_chat_async.completions.create = Mock(side_effect=mock_chat_create)

            # Set up clients
            instance_async = Mock()
            instance_async.embeddings = mock_embeddings_async
            instance_async.chat = mock_chat_async
            mock_index_client.return_value = instance_async

            instance_sync = Mock()
            instance_sync.embeddings = mock_embeddings_sync
            mock_retrieve_client.return_value = instance_sync

            # Create test config with temporary directory for ChromaDB
            with tempfile.TemporaryDirectory() as temp_dir:
                config = RagZoomConfig(
                    openai_api_key="test-key",
                    sqlite_database_url="sqlite:///:memory:",
                    chroma_persist_directory=temp_dir,
                    leaf_tokens=200,
                    budget_tokens=1000,
                )

                store = Store(config)
                tree_builder = TreeBuilder(config, store)
                retriever = Retriever(config, store, tree_builder)

                yield config, store, tree_builder, retriever, mock_chat_async

                # Close store to prevent file handle leaks
                store.close()

    def test_refresh_nodes_async(self, setup_system):
        """Test TreeBuilder.refresh_nodes_async updates dirty nodes."""
        config, store, tree_builder, retriever, mock_chat = setup_system

        # Create a simple tree structure
        # Add leaf nodes
        store.add_node(
            "leaf1",
            "Original leaf 1 text",
            [0.1] * 384,
            0,
            0,
            100,
            document_id="test-doc",
        )
        store.add_node(
            "leaf2",
            "Original leaf 2 text",
            [0.2] * 384,
            0,
            100,
            200,
            document_id="test-doc",
        )

        # Add parent node with old summary
        store.add_node(
            "parent1",
            "Old parent summary. <<<MID>>> Old right summary.",
            [0.15] * 384,
            1,
            0,
            200,
            left_child_id="leaf1",
            right_child_id="leaf2",
            summary="Old parent summary. <<<MID>>> Old right summary.",
            mid_offset=len("Old parent summary. "),
            document_id="test-doc",
        )

        # Update parent references
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            for leaf_id in ["leaf1", "leaf2"]:
                leaf = session.query(TreeNode).filter_by(id=leaf_id).first()
                if leaf:
                    leaf.parent_id = "parent1"
            session.commit()

        # Mark parent as dirty
        store.mark_dirty_upward("parent1")

        # Verify node is dirty
        dirty_nodes = store.get_dirty_nodes()
        assert len(dirty_nodes) == 1
        assert dirty_nodes[0].id == "parent1"

        # Run refresh
        refreshed_count = asyncio.run(tree_builder.refresh_nodes_async(["parent1"]))

        # Verify refresh happened
        assert refreshed_count == 1

        # Check that node is no longer dirty
        dirty_nodes = store.get_dirty_nodes()
        assert len(dirty_nodes) == 0

        # Verify the summary was updated
        parent = store.get_node("parent1")
        assert "Fresh left summary" in parent.text
        assert "Fresh right summary" in parent.text
        assert parent.mid_offset == len("Fresh left summary. ")

        # Verify API was called
        mock_chat.completions.create.assert_called_once()

    def test_retrieve_with_dirty_refresh(self, setup_system):
        """Test that retrieve() refreshes dirty nodes before search."""
        config, store, tree_builder, retriever, mock_chat = setup_system

        # Create a tree with dirty node
        store.add_node(
            "leaf1", "Leaf 1 text", [0.1] * 384, 0, 0, 100, document_id="test-doc"
        )
        store.add_node(
            "leaf2", "Leaf 2 text", [0.2] * 384, 0, 100, 200, document_id="test-doc"
        )
        store.add_node(
            "parent1",
            "Stale summary",
            [0.15] * 384,
            1,
            0,
            200,
            left_child_id="leaf1",
            right_child_id="leaf2",
            document_id="test-doc",
        )

        # Mark as dirty
        store.mark_dirty_upward("parent1")

        # Run retrieval (should trigger refresh)
        retriever.retrieve("test query")

        # Verify node was refreshed
        parent = store.get_node("parent1")
        assert "Fresh left summary" in parent.text
        assert parent.is_dirty == 0

    def test_no_double_refresh(self, setup_system):
        """Test that nodes aren't refreshed twice in same request."""
        config, store, tree_builder, retriever, mock_chat = setup_system

        # Create dirty node
        store.add_node(
            "leaf1", "Leaf 1", [0.1] * 384, 0, 0, 100, document_id="test-doc"
        )
        store.add_node(
            "leaf2", "Leaf 2", [0.2] * 384, 0, 100, 200, document_id="test-doc"
        )
        store.add_node(
            "parent1",
            "Old summary",
            [0.15] * 384,
            1,
            0,
            200,
            left_child_id="leaf1",
            right_child_id="leaf2",
            document_id="test-doc",
        )

        store.mark_dirty_upward("parent1")

        # First retrieval
        retriever.retrieve("query 1")

        # Reset mock to track second call
        mock_chat.completions.create.reset_mock()

        # Second retrieval in same retriever instance (same request cache)
        retriever.retrieve("query 2")

        # Should not refresh again
        mock_chat.completions.create.assert_not_called()

    def test_leaf_nodes_not_refreshed(self, setup_system):
        """Test that leaf nodes are skipped during refresh."""
        config, store, tree_builder, retriever, mock_chat = setup_system

        # Create and mark leaf as dirty
        store.add_node(
            "leaf1", "Leaf text", [0.1] * 384, 0, 0, 100, document_id="test-doc"
        )
        store.mark_dirty_upward("leaf1")

        # Try to refresh
        refreshed_count = asyncio.run(tree_builder.refresh_nodes_async(["leaf1"]))

        # Should skip leaf nodes
        assert refreshed_count == 0
        mock_chat.completions.create.assert_not_called()

    def test_embedding_dimension_preserved(self, setup_system):
        """Test that refresh preserves correct embedding dimensions."""
        config, store, tree_builder, retriever, mock_chat = setup_system

        # ChromaDB defaults to 384, so we'll use that
        test_dim = 384

        # Create nodes
        store.add_node(
            "leaf1", "Leaf 1", [0.1] * test_dim, 0, 0, 100, document_id="test-doc"
        )
        store.add_node(
            "leaf2", "Leaf 2", [0.2] * test_dim, 0, 100, 200, document_id="test-doc"
        )
        store.add_node(
            "parent1",
            "Old summary",
            [0.15] * test_dim,
            1,
            0,
            200,
            left_child_id="leaf1",
            right_child_id="leaf2",
            document_id="test-doc",
        )

        # Update parent references
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            for leaf_id in ["leaf1", "leaf2"]:
                leaf = session.query(TreeNode).filter_by(id=leaf_id).first()
                if leaf:
                    leaf.parent_id = "parent1"
            session.commit()

        store.mark_dirty_upward("parent1")

        # Refresh should preserve dimension
        refreshed_count = asyncio.run(tree_builder.refresh_nodes_async(["parent1"]))
        assert refreshed_count == 1

        # Verify the embedding dimension matches what the store expects
        assert store._expected_embedding_dim == test_dim
