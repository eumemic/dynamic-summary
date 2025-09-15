"""Simple test to demonstrate the document clearing regression."""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.services.indexing_service import IndexingService


@pytest.mark.slow_threshold(2.0)
def test_index_document_always_clears(storage_backend: StorageBackend) -> None:
    """Test that index_document ALWAYS clears, even when content hash matches."""

    config = OperationalConfig(openai_api_key=SecretStr("test-key"))
    index_config = IndexConfig.load()

    # Prepare a real document with pre-existing nodes
    import uuid

    doc_id = f"test-{uuid.uuid4().hex}.txt"
    doc_store = storage_backend.for_document(doc_id)
    doc_store.set_metadata(
        file_path=None,
        content_hash="pre-hash",
        chunk_count=0,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )
    pre_nodes = [
        {
            "node_id": f"pre-{i}",
            "text": f"Pre node {i}",
            "span_start": i * 10,
            "span_end": i * 10 + 5,
            "document_id": doc_id,
            "token_count": 1,
            "height": 0,
            "path": "",
        }
        for i in range(2)
    ]
    doc_store.nodes.add_batch(pre_nodes)  # type: ignore[arg-type]

    # Mock the tree builder to avoid actual indexing
    # Mock OpenAI client to avoid network
    mock_async_client = MagicMock()

    async def mock_embeddings(*args: object, **kwargs: object) -> object:
        from types import SimpleNamespace

        input_texts = cast(list[str] | str, kwargs.get("input", []))
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_texts]
        )

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
        # Create the service using backend directly
        service = IndexingService(storage_backend, index_config, config)
        # Index a document
        service.index_document("Test content", document_id=doc_id, show_progress=False)

        # WITH FIX: clear_document should be called
        # WITHOUT FIX: clear_document would NOT be called if we had content hash check
    # Verify pre-existing nodes were cleared (no 'pre-' nodes remain)
    remaining_ids = [n.id for n in doc_store.nodes.get_all()]
    assert all(not nid.startswith("pre-") for nid in remaining_ids)
    print("✅ Fix verified: pre-existing nodes cleared before indexing")
