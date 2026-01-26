"""Acceptance tests for embedding text optimization.

These tests verify the end-to-end behavior described in specs/embedding-text-optimization.md.
"""

from __future__ import annotations

import pytest

from ragzoom.config import IndexConfig
from ragzoom.utils.tokenization import tokenizer


@pytest.mark.asyncio
async def test_oversized_leaf_embeds_successfully() -> None:
    """Test that oversized leaves (9000+ tokens) embed successfully.

    Acceptance test for specs/embedding-text-optimization.md § Acceptance Criteria > 1:
    "Oversized leaves embed successfully: Leaves with 9000+ tokens no longer fail"

    Prior to the embedding text optimization feature, leaves exceeding 8000 tokens
    would fail with ValueError: "Item 0 exceeds embedding token limit". Now the
    _prepare_embedding_text workflow compresses large content to fit within limits.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from ragzoom.contracts.embedding_model import EmbeddingResult
    from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine
    from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult

    # Create oversized leaf text (9000+ tokens)
    # "word " is approximately 1 token, so we need 10000+ repetitions
    oversized_leaf_text = "word " * 10000
    token_count = tokenizer.count_tokens(oversized_leaf_text)
    assert token_count > 9000, (
        f"Test setup error: expected >9000 tokens, got {token_count}. "
        "Increase repetitions to ensure oversized content."
    )

    # Setup mocks for storage backend
    mock_store = MagicMock()
    mock_doc_store = MagicMock()
    mock_store.for_document.return_value = mock_doc_store

    # Create mock leaf with oversized text
    mock_leaf = MagicMock()
    mock_leaf.text = oversized_leaf_text
    mock_leaf.id = "oversized-leaf-id"
    mock_leaf.span_start = 0  # First leaf, no preceding context needed
    mock_leaf.span_end = len(oversized_leaf_text)
    mock_leaf.level_index = 0
    mock_leaf.coord_version = 1
    mock_leaf.parent_id = None
    mock_doc_store.nodes.get.return_value = mock_leaf

    # Setup mock LLM service
    mock_llm_service = MagicMock()

    # Track embedding calls
    embed_texts_received: list[str] = []

    async def mock_embed_texts_with_usage(texts: list[str]) -> EmbeddingResult:
        embed_texts_received.extend(texts)
        # Return valid embedding for each text
        return {
            "embeddings": [[0.1] * 1536 for _ in texts],
            "usage": {"total_tokens": 100, "model": "text-embedding-3-small"},
        }

    mock_llm_service.embed_texts_with_usage = mock_embed_texts_with_usage

    # Mock _prepare_embedding_text to simulate LLM compression
    # This is the key functionality: it compresses oversized content to fit
    compressed_text = "Compressed summary of the large document content"

    async def mock_prepare_embedding_text(
        preceding_context: str,
        leaf_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object = None,
    ) -> SummaryResult:
        # Verify we received the oversized content
        received_tokens = tokenizer.count_tokens(leaf_text)
        assert (
            received_tokens > 9000
        ), f"Expected oversized leaf text >9000 tokens, got {received_tokens}"
        # Return compressed text within embedding limits
        return SummaryResult(
            summary=compressed_text,
            retry_count=0,
            summary_tokens=10,
            usage=AccumulatedUsage(prompt_tokens=500, completion_tokens=10),
        )

    mock_llm_service._prepare_embedding_text = mock_prepare_embedding_text

    # Setup config
    config = IndexConfig.load().replace(target_embedding_tokens=500)

    # Create engine
    engine = IndexingEngine(
        store=mock_store,
        llm_service=mock_llm_service,
        index_config=config,
        openai_client=AsyncMock(),
    )

    # Create mock vector index to capture upserts
    mock_vector_index = MagicMock()
    upsert_calls: list[tuple[str, list[float], dict[str, object]]] = []

    def capture_upsert(items: list[tuple[str, list[float], dict[str, object]]]) -> None:
        upsert_calls.extend(items)

    mock_vector_index.upsert = capture_upsert

    # Execute: Run embedding job for oversized leaf
    with patch.object(engine, "_get_vector_index", return_value=mock_vector_index):
        job = EmbeddingJob(document_id="test-doc", leaf_id="oversized-leaf-id")
        # This should NOT raise - the key acceptance criterion
        await engine._embed_leaf(job)

    # Verify: Embedding succeeded
    assert (
        len(embed_texts_received) == 1
    ), "embed_texts_with_usage should have been called once with compressed text"
    assert (
        embed_texts_received[0] == compressed_text
    ), f"Expected compressed text to be embedded, got: {embed_texts_received[0][:100]}"

    # Verify: Vector was stored
    assert len(upsert_calls) == 1, (
        "vector_index.upsert should have been called once. "
        "Embedding should succeed for oversized leaves when compression is applied."
    )
    node_id, embedding, metadata = upsert_calls[0]
    assert node_id == "oversized-leaf-id"
    assert len(embedding) == 1536, "Embedding should have correct dimension"
    assert metadata["document_id"] == "test-doc"
