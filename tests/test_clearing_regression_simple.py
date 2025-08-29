"""Simple test to demonstrate the document clearing regression."""

from unittest.mock import Mock, patch

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.services.indexing_service import IndexingService


def test_index_document_always_clears():
    """Test that index_document ALWAYS clears, even when content hash matches."""

    config = OperationalConfig(openai_api_key=SecretStr("test-key"))
    index_config = IndexConfig.load()

    # Create a mock store
    mock_store = Mock()

    # Set up minimal mocks needed
    mock_store.compute_content_hash.return_value = "abc123"
    mock_store.clear_document.return_value = 5  # Cleared 5 nodes
    mock_store.add_document.return_value = None
    mock_store.for_document.return_value = Mock()

    # Mock the tree builder to avoid actual indexing
    with patch(
        "ragzoom.services.indexing_service.TreeBuilder"
    ) as mock_tree_builder_class:
        mock_tree_builder = Mock()
        mock_tree_builder.add_document.return_value = "test.txt"
        mock_tree_builder_class.return_value = mock_tree_builder

        # Mock the session for document stats
        mock_session = Mock()
        mock_leaves = [Mock() for _ in range(3)]
        mock_root = Mock(height=2)
        mock_doc = Mock(id="test.txt", chunk_count=3)

        # Set up query chain
        mock_query = Mock()
        mock_query.filter_by.return_value.filter.return_value.all.return_value = (
            mock_leaves
        )
        mock_query.filter_by.return_value.first.side_effect = [mock_root, mock_doc]
        mock_session.query.return_value = mock_query

        # Set up context manager
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_session)
        mock_context.__exit__ = Mock(return_value=None)
        mock_store.SessionLocal.return_value = mock_context

        # Create the service
        service = IndexingService(mock_store, index_config, config)

        # Index a document
        service.index_document(
            "Test content", document_id="test.txt", show_progress=False
        )

        # WITH FIX: clear_document should be called
        # WITHOUT FIX: clear_document would NOT be called if we had content hash check
        assert mock_store.clear_document.called, "Document should always be cleared!"
        assert mock_store.clear_document.call_args[0][0] == "test.txt"
        print(
            f"✅ Fix verified: clear_document was called for {mock_store.clear_document.call_args[0][0]}"
        )
