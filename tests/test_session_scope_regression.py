"""Test for session scope regression in IndexingService."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass
from unittest.mock import MagicMock, patch

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.services.indexing_service import IndexingService


def test_tree_height_accessed_within_session(storage_backend: StorageBackend) -> None:
    """Test that root.height is accessed while session is still open.

    This is a regression test for a bug where root.height was accessed after
    the session closed, causing: "Instance <TreeNode> is not bound to a Session"
    """

    config = OperationalConfig(openai_api_key=SecretStr("test-key"))
    index_config = IndexConfig.load()

    # Mock OpenAI to avoid network
    mock_async_client = MagicMock()

    async def mock_embeddings(*args: object, **kwargs: object) -> object:
        from typing import cast

        input_texts = cast(list[str] | str, kwargs.get("input", []))
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        return MagicMock(data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts])

    mock_async_client.embeddings.create = mock_embeddings
    mock_async_client.chat.completions.create = MagicMock(
        return_value=MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(content="Summary of left and right content")
                )
            ]
        )
    )

    with patch(
        "ragzoom.services.llm_service.AsyncOpenAI", return_value=mock_async_client
    ):
        service = IndexingService(storage_backend, index_config, config)  # type: ignore[arg-type]
        result = service.index_document(
            "Test content", document_id="test.txt", show_progress=False
        )
        # Should not raise; ensure tree_depth computed
        assert isinstance(result.tree_depth, int)
        print(
            f"✅ No error - tree depth computed without session exposure (tree_depth={result.tree_depth})"
        )
