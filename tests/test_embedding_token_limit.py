"""Test that embedding text never exceeds the embedding model's token limit.

Regression test for issue where text_to_embed (context_prefix + leaf_text)
could exceed the 8000 token limit for text-embedding-3-small.
"""

from __future__ import annotations

import pytest

from ragzoom.config import IndexConfig
from ragzoom.services.llm_service import LLMService
from ragzoom.utils.tokenization import tokenizer

# OpenAI's text-embedding-3-small has an 8191 token limit
# We use 8000 as a safe limit in llm_service.py
EMBEDDING_TOKEN_LIMIT = 8000


@pytest.mark.asyncio
async def test_llm_service_rejects_oversized_embedding_text() -> None:
    """Test that LLMService.embed_texts rejects text over 8000 tokens.

    This is a unit test verifying the validation in embed_texts works.
    """
    config = IndexConfig.load()
    llm_service = LLMService(config)

    # Create text that's definitely over 8000 tokens
    # ~4 chars per token on average, so 40000 chars should be ~10000 tokens
    oversized_text = "word " * 10000  # ~10000 tokens

    # Verify the text is actually over the limit
    token_count = tokenizer.count_tokens(oversized_text)
    assert (
        token_count > EMBEDDING_TOKEN_LIMIT
    ), f"Test setup error: expected >8000 tokens, got {token_count}"

    # This should raise ValueError due to token limit
    with pytest.raises(ValueError, match="exceeds embedding token limit"):
        await llm_service.embed_texts([oversized_text])


def test_combined_context_and_leaf_within_limit() -> None:
    """Test that context_prefix + leaf_text can't exceed the embedding limit.

    This test validates the invariant that should hold during indexing:
    the combined text for embedding (context + leaf) must stay under 8000 tokens.

    Given:
    - preceding_summary_budget_tokens controls context retrieval budget
    - target_chunk_tokens controls leaf chunk size
    - Combined must be <= EMBEDDING_TOKEN_LIMIT

    The system should enforce that:
    preceding_summary_budget_tokens + target_chunk_tokens + margin <= 8000
    """
    # Default config values
    config = IndexConfig.load()

    # Calculate the maximum possible combined size
    # Note: In practice, chunks can exceed target_chunk_tokens due to
    # gap reconstruction in the splitter, so we need margin
    max_context = config.preceding_summary_budget_tokens
    max_chunk = config.target_chunk_tokens

    # With gap reconstruction, a chunk could be up to ~2x target in edge cases
    # Be conservative and assume 3x for safety margin
    worst_case_chunk = max_chunk * 3

    combined_max = max_context + worst_case_chunk

    # This assertion documents the EXPECTED invariant
    # If this fails, it means the config allows oversized embeddings
    assert combined_max <= EMBEDDING_TOKEN_LIMIT, (
        f"Config allows oversized embedding text: "
        f"context ({max_context}) + worst_case_chunk ({worst_case_chunk}) = {combined_max} "
        f"exceeds limit ({EMBEDDING_TOKEN_LIMIT}). "
        f"Either reduce preceding_summary_budget_tokens or enforce chunk size limits."
    )


@pytest.mark.asyncio
async def test_indexing_engine_limits_embedding_text() -> None:
    """Regression test: _embed_leaf must limit text_to_embed to 8000 tokens.

    This simulates the production bug where:
    - Leaf text: could be large due to chunking
    - Context prefix: from retrieve_for_context()
    - Combined: could exceed 8000 tokens

    The fix should limit context_prefix based on leaf_text size.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

    # Create mocks
    mock_store = MagicMock()
    mock_doc_store = MagicMock()
    mock_store.for_document.return_value = mock_doc_store

    # Create a leaf with large text (~5000 tokens)
    large_leaf_text = "word " * 5000  # ~5000 tokens
    leaf_tokens = tokenizer.count_tokens(large_leaf_text)
    assert leaf_tokens > 4000, f"Test setup: expected >4000 tokens, got {leaf_tokens}"

    mock_leaf = MagicMock()
    mock_leaf.text = large_leaf_text
    mock_leaf.id = "test-leaf-id"
    mock_leaf.span_start = 1000  # Non-zero to trigger context retrieval
    mock_leaf.span_end = 2000
    mock_doc_store.nodes.get.return_value = mock_leaf

    # Create config with large context budget (5000 tokens)
    # Combined would be ~10000 tokens without the fix
    config = IndexConfig.load(preceding_summary_budget_tokens=5000)

    # Create mock LLM service
    mock_llm_service = MagicMock()

    # Track what text is passed to embed_texts
    embed_texts_received: list[str] = []

    async def capture_embed_texts(texts: list[str]) -> list[list[float]]:
        embed_texts_received.extend(texts)
        return [[0.1] * 1536 for _ in texts]

    mock_llm_service.embed_texts = capture_embed_texts

    # Create mock retriever that returns large context (~5000 tokens)
    large_context = "context " * 5000  # ~5000 tokens
    context_tokens = tokenizer.count_tokens(large_context)
    assert context_tokens > 4000, "Test setup: expected >4000 context tokens"

    mock_retriever = AsyncMock()
    mock_retriever.retrieve_for_context = AsyncMock(return_value=large_context)

    # Create engine
    engine = IndexingEngine(
        store=mock_store,
        llm_service=mock_llm_service,
        index_config=config,
        openai_client=AsyncMock(),
    )

    # Patch _create_retriever to return our mock
    with patch.object(engine, "_create_retriever", return_value=mock_retriever):
        with patch.object(engine, "_get_vector_index", return_value=None):
            job = EmbeddingJob(document_id="test-doc", leaf_id="test-leaf-id")
            await engine._embed_leaf(job)

    # Verify embed_texts was called
    assert len(embed_texts_received) == 1, "embed_texts should have been called once"

    # The key assertion: text_to_embed must not exceed the limit
    text_to_embed = embed_texts_received[0]
    total_tokens = tokenizer.count_tokens(text_to_embed)

    assert total_tokens <= EMBEDDING_TOKEN_LIMIT, (
        f"text_to_embed has {total_tokens} tokens, exceeding limit of {EMBEDDING_TOKEN_LIMIT}. "
        f"This is the production bug - context_prefix + leaf_text is not being limited. "
        f"Leaf had {leaf_tokens} tokens, context had {context_tokens} tokens."
    )
