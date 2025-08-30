"""Test for session scope regression in IndexingService."""

from unittest.mock import Mock, patch

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.services.indexing_service import IndexingService


def test_tree_height_accessed_within_session() -> None:
    """Test that root.height is accessed while session is still open.

    This is a regression test for a bug where root.height was accessed after
    the session closed, causing: "Instance <TreeNode> is not bound to a Session"
    """

    config = OperationalConfig(openai_api_key=SecretStr("test-key"))
    index_config = IndexConfig.load()

    # Create a mock store
    mock_store = Mock()

    # Set up minimal mocks
    mock_store.compute_content_hash.return_value = "abc123"
    mock_store.clear_document.return_value = 0
    mock_store.add_document.return_value = None
    mock_store.for_document.return_value = Mock()

    # Mock the tree builder
    with patch(
        "ragzoom.services.indexing_service.TreeBuilder"
    ) as mock_tree_builder_class:
        mock_tree_builder = Mock()
        # Since sync now delegates to async, mock the async method
        from unittest.mock import AsyncMock

        mock_tree_builder.add_document_async = AsyncMock(return_value="test.txt")
        mock_tree_builder_class.return_value = mock_tree_builder

        # Create mock root that will raise error if accessed outside session
        class MockRoot:
            def __getattr__(self, name):
                if name == "height":
                    # Check if we're still in the context manager
                    if not hasattr(mock_context, "_in_context"):
                        raise Exception(
                            "Instance <TreeNode at 0x1234> is not bound to a Session; "
                            "attribute refresh operation cannot proceed"
                        )
                    return 3  # Return a height value when accessed properly
                return Mock()

        mock_root = MockRoot()

        # Mock the session
        mock_session = Mock()
        mock_leaves = [Mock() for _ in range(3)]
        mock_doc = Mock(id="test.txt", chunk_count=3)

        # Set up query chain
        mock_query = Mock()
        mock_query.filter_by.return_value.filter.return_value.all.return_value = (
            mock_leaves
        )
        mock_query.filter_by.return_value.first.side_effect = [mock_root, mock_doc]
        mock_session.query.return_value = mock_query

        # Create context manager that tracks whether we're inside it
        mock_context = Mock()

        def enter_context() -> Mock:
            mock_context._in_context = True
            return mock_session

        def exit_context(*args: object) -> None:
            # Mark that we've exited the context
            delattr(mock_context, "_in_context")
            return None

        mock_context.__enter__ = Mock(side_effect=enter_context)
        mock_context.__exit__ = Mock(side_effect=exit_context)
        mock_store.SessionLocal.return_value = mock_context

        # Create the service
        service = IndexingService(mock_store, index_config, config)

        # This should NOT raise an error with the fix
        # But WILL raise an error without the fix (when tree_height = root.height is outside session)
        try:
            result = service.index_document(
                "Test content", document_id="test.txt", show_progress=False
            )
            print(
                f"✅ No error - root.height accessed within session (tree_depth={result.tree_depth})"
            )
        except Exception as e:
            if "not bound to a Session" in str(e):
                raise AssertionError(
                    "REGRESSION: root.height was accessed after session closed!\n"
                    f"Error: {e}"
                )
            raise  # Re-raise other unexpected errors
