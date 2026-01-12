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


def test_summarized_context_and_leaf_within_embedding_limit() -> None:
    """Test that context_summary + leaf_text stays under the embedding limit.

    The embedding text is composed of:
    - context_summary: summary of preceding context (target: target_embedding_context_tokens)
    - leaf_text: the leaf chunk itself (target: target_chunk_tokens)

    Note: preceding_context_budget controls how much context is retrieved for
    summarization (sent to the LLM), NOT what gets embedded. The context is
    summarized down to target_embedding_context_tokens before being prepended to the leaf.
    """
    config = IndexConfig.load()
    # This test is for fixed-chunking mode
    assert config.target_chunk_tokens is not None

    # The embedding text is: context_summary (~target_chunk_tokens) + leaf (~target_chunk_tokens)
    # With gap reconstruction, chunks can be up to ~3x target in edge cases
    max_summary_tokens = config.target_chunk_tokens * 3  # Worst case summary
    max_leaf_tokens = config.target_chunk_tokens * 3  # Worst case leaf

    combined_max = max_summary_tokens + max_leaf_tokens

    # This should comfortably fit within the 8000 token embedding limit
    # With default target_chunk_tokens=200, this is 1200 tokens max
    assert combined_max <= EMBEDDING_TOKEN_LIMIT, (
        f"Embedding text could exceed limit: "
        f"summary ({max_summary_tokens}) + leaf ({max_leaf_tokens}) = {combined_max} "
        f"exceeds limit ({EMBEDDING_TOKEN_LIMIT}). "
        f"Reduce target_chunk_tokens."
    )


@pytest.mark.asyncio
async def test_embed_leaf_uses_embedding_context_tokens() -> None:
    """Test that _embed_leaf uses target_embedding_context_tokens for contextualization.

    Verifies that when contextualizing preceding context for embedding, the engine
    uses config.target_embedding_context_tokens instead of config.target_chunk_tokens.
    This allows client-managed chunking (target_chunk_tokens=None) while still having
    sensible embedding context summarization.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine
    from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult

    # Create mocks
    mock_store = MagicMock()
    mock_doc_store = MagicMock()
    mock_store.for_document.return_value = mock_doc_store

    # Create a leaf with some text
    leaf_text = "This is a test leaf."
    mock_leaf = MagicMock()
    mock_leaf.text = leaf_text
    mock_leaf.id = "test-leaf-id"
    mock_leaf.span_start = 1000  # Non-zero to trigger context retrieval
    mock_leaf.span_end = 1100
    mock_doc_store.nodes.get.return_value = mock_leaf

    # Create config with custom target_embedding_context_tokens
    config = IndexConfig.load().replace(
        target_chunk_tokens=None,  # Client-managed chunking mode
        target_embedding_context_tokens=300,  # Custom embedding context target
    )

    # Create mock LLM service
    mock_llm_service = MagicMock()

    # Track what target_tokens is passed to _contextualize_text
    contextualize_calls: list[int] = []

    async def mock_contextualize(
        preceding_context: str,
        target_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object = None,
    ) -> SummaryResult:
        contextualize_calls.append(target_tokens)
        return SummaryResult(
            summary="summarized context",
            retry_count=0,
            summary_tokens=50,
            usage=AccumulatedUsage(),
        )

    mock_llm_service._contextualize_text = mock_contextualize

    # Mock embed_texts_with_usage
    from ragzoom.contracts.embedding_model import EmbeddingResult

    async def mock_embed_texts_with_usage(texts: list[str]) -> EmbeddingResult:
        return {
            "embeddings": [[0.1] * 1536 for _ in texts],
            "usage": {"total_tokens": 50, "model": "text-embedding-3-small"},
        }

    mock_llm_service.embed_texts_with_usage = mock_embed_texts_with_usage

    # Mock _get_preceding_context to return some context
    from ragzoom.server.indexing_engine import PrecedingContextResult

    mock_context_node = MagicMock()
    mock_context_node.id = "context-node"
    mock_context_node.span_start = 0
    mock_context_node.span_end = 1000
    mock_context_node.height = 1
    mock_context_node.token_count = 100
    mock_context_node.text = "Some preceding context"

    async def mock_get_preceding_context(
        store: object,
        document_id: str,
        span_start: int,
        config: object,
        query_text: str | None,
        query_embedding: list[float] | None = None,
    ) -> PrecedingContextResult:
        return PrecedingContextResult(
            tiling_ids=["context-node"],
            nodes={"context-node": mock_context_node},
            tiling_tokens=100,
        )

    # Create engine
    engine = IndexingEngine(
        store=mock_store,
        llm_service=mock_llm_service,
        index_config=config,
        openai_client=AsyncMock(),
    )

    # Create mock vector index
    mock_vector_index = MagicMock()

    # Run _embed_leaf
    with patch.object(
        engine, "_get_preceding_context", side_effect=mock_get_preceding_context
    ):
        with patch.object(engine, "_get_vector_index", return_value=mock_vector_index):
            job = EmbeddingJob(document_id="test-doc", leaf_id="test-leaf-id")
            await engine._embed_leaf(job)

    # Verify _contextualize_text was called with target_embedding_context_tokens
    assert (
        len(contextualize_calls) == 1
    ), "_contextualize_text should have been called once"
    assert contextualize_calls[0] == 300, (
        f"_contextualize_text should use target_embedding_context_tokens (300), "
        f"but was called with {contextualize_calls[0]}. "
        f"This is the bug: it's using target_chunk_tokens instead of "
        f"target_embedding_context_tokens for embedding contextualization."
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
    config = IndexConfig.load()

    # Create mock LLM service
    mock_llm_service = MagicMock()

    # Track what text is passed to embed_texts_with_usage
    embed_texts_received: list[str] = []

    from ragzoom.contracts.embedding_model import EmbeddingResult

    async def capture_embed_texts_with_usage(texts: list[str]) -> EmbeddingResult:
        embed_texts_received.extend(texts)
        return {
            "embeddings": [[0.1] * 1536 for _ in texts],
            "usage": {"total_tokens": 100, "model": "text-embedding-3-small"},
        }

    mock_llm_service.embed_texts_with_usage = capture_embed_texts_with_usage

    # Mock _contextualize_text to return a short summary (simulating context summarization)
    from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult

    async def mock_contextualize(
        preceding_context: str,
        target_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object = None,
    ) -> SummaryResult:
        # Return a short summary instead of the full context
        return SummaryResult(
            summary="summarized context",
            retry_count=0,
            summary_tokens=50,
            usage=AccumulatedUsage(),
        )

    mock_llm_service._contextualize_text = mock_contextualize

    # Create mock retriever that returns large context (~5000 tokens)
    from ragzoom.retrieve import RetrievalResult

    large_context = "context " * 5000  # ~5000 tokens
    context_tokens = tokenizer.count_tokens(large_context)
    assert context_tokens > 4000, "Test setup: expected >4000 context tokens"

    # Create a mock TreeNode for the context result
    mock_context_node = MagicMock()
    mock_context_node.id = "context-node"
    mock_context_node.span_start = 0
    mock_context_node.span_end = 1000
    mock_context_node.height = 1
    mock_context_node.token_count = context_tokens
    mock_context_node.text = large_context

    context_result = RetrievalResult(
        node_ids=["context-node"],
        scores={},
        coverage_map={},
        tiling=["context-node"],
        nodes={"context-node": mock_context_node},
    )

    mock_retriever = AsyncMock()
    mock_retriever.retrieve_for_context = AsyncMock(return_value=context_result)
    # Mock embedding_service for query embedding with usage
    mock_embedding_service = MagicMock()

    async def mock_get_query_embedding_with_usage(
        query: str, document_id: str | None = None
    ) -> tuple[list[float], dict[str, object]]:
        return [0.1] * 1536, {"total_tokens": 50, "model": "text-embedding-3-small"}

    mock_embedding_service.get_query_embedding_async_with_usage = (
        mock_get_query_embedding_with_usage
    )
    mock_retriever.embedding_service = mock_embedding_service

    # Create engine
    engine = IndexingEngine(
        store=mock_store,
        llm_service=mock_llm_service,
        index_config=config,
        openai_client=AsyncMock(),
    )

    # Create mock vector index
    mock_vector_index = MagicMock()

    # Patch _create_retriever to return our mock
    with patch.object(engine, "_create_retriever", return_value=mock_retriever):
        with patch.object(engine, "_get_vector_index", return_value=mock_vector_index):
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


@pytest.mark.asyncio
async def test_embed_leaf_records_telemetry() -> None:
    """Test that _embed_leaf records embedding telemetry with timing.

    Regression test for bug where leaf node embeddings were generated but
    no telemetry was recorded, causing missing embedding data in telemetry.json.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from ragzoom.server.indexing_engine import (
        DocumentContext,
        EmbeddingJob,
        IndexingEngine,
    )
    from ragzoom.telemetry_collection import TelemetryCollector

    # Create mocks
    mock_store = MagicMock()
    mock_doc_store = MagicMock()
    mock_store.for_document.return_value = mock_doc_store

    # Create a leaf with some text
    leaf_text = "This is a test leaf with some text content."
    mock_leaf = MagicMock()
    mock_leaf.text = leaf_text
    mock_leaf.id = "test-leaf-id"
    mock_leaf.span_start = 0  # First leaf, no context retrieval
    mock_leaf.span_end = 100
    mock_doc_store.nodes.get.return_value = mock_leaf

    # Create config
    config = IndexConfig.load()

    # Create mock LLM service
    mock_llm_service = MagicMock()

    from ragzoom.contracts.embedding_model import EmbeddingResult

    async def mock_embed_texts_with_usage(texts: list[str]) -> EmbeddingResult:
        return {
            "embeddings": [[0.1] * 1536 for _ in texts],
            "usage": {"total_tokens": 50, "model": "text-embedding-3-small"},
        }

    mock_llm_service.embed_texts_with_usage = mock_embed_texts_with_usage

    # Create telemetry collector and pre-register the leaf node
    telemetry = TelemetryCollector(
        document_id="test-doc",
        source_tokens=100,
        config=config,
    )
    telemetry.track_node_created(
        node_id="test-leaf-id",
        height=0,
        span=(0, 100),
    )

    # Create engine with document context containing telemetry
    engine = IndexingEngine(
        store=mock_store,
        llm_service=mock_llm_service,
        index_config=config,
        openai_client=AsyncMock(),
    )

    # Set up document context with telemetry collector
    engine._document_contexts["test-doc"] = DocumentContext(
        telemetry_collector=telemetry
    )

    # Create mock vector index
    mock_vector_index = MagicMock()

    # Run _embed_leaf
    with patch.object(engine, "_get_vector_index", return_value=mock_vector_index):
        job = EmbeddingJob(document_id="test-doc", leaf_id="test-leaf-id")
        await engine._embed_leaf(job)

    # Verify telemetry was recorded
    node_telemetry = telemetry.node_telemetry.get("test-leaf-id")
    assert node_telemetry is not None, "Node telemetry should exist"
    assert node_telemetry.embedding is not None, (
        "Embedding telemetry should be recorded. "
        "This is the regression - _embed_leaf should call record_embedding_call_v2."
    )

    # Verify embedding timing was captured
    assert node_telemetry.embedding.start_time > 0, "start_time should be set"
    assert node_telemetry.embedding.end_time > 0, "end_time should be set"
    assert (
        node_telemetry.embedding.end_time >= node_telemetry.embedding.start_time
    ), "end_time should be >= start_time"

    # Verify other embedding fields
    assert node_telemetry.embedding.text_tokens > 0, "text_tokens should be > 0"
    assert node_telemetry.embedding.batch_size == 1, "batch_size should be 1"
    assert node_telemetry.embedding.batch_position == 0, "batch_position should be 0"
    assert (
        node_telemetry.embedding.model == config.embedding_model
    ), f"model should be {config.embedding_model}"


@pytest.mark.asyncio
async def test_embed_leaf_includes_level_index_in_vector_metadata() -> None:
    """Regression test: vector metadata must include level_index and coord_version.

    Bug: _embed_leaf builds metadata for vector_index.upsert but omits level_index
    and coord_version. This causes all vectors to get level_index=0 by default,
    which breaks the coverage builder optimization that uses vector metadata to
    avoid database fetches during retrieval.

    Impact: Without proper coordinates in vector metadata, every retrieval falls
    back to database fetches for seed nodes, causing ~1200ms latency on Railway
    (with network latency) vs ~4ms locally.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine

    # Create mocks
    mock_store = MagicMock()
    mock_doc_store = MagicMock()
    mock_store.for_document.return_value = mock_doc_store

    # Create a leaf with level_index = 42 (not 0, to verify it's passed correctly)
    mock_leaf = MagicMock()
    mock_leaf.text = "Test leaf content"
    mock_leaf.id = "test-leaf-id"
    mock_leaf.span_start = 0
    mock_leaf.span_end = 100
    mock_leaf.level_index = 42  # Key: non-zero to detect if it's being passed
    mock_leaf.coord_version = 1
    mock_leaf.parent_id = "parent-123"
    mock_doc_store.nodes.get.return_value = mock_leaf

    config = IndexConfig.load()

    # Create mock LLM service
    mock_llm_service = MagicMock()

    from ragzoom.contracts.embedding_model import EmbeddingResult

    async def mock_embed_texts_with_usage(texts: list[str]) -> EmbeddingResult:
        return {
            "embeddings": [[0.1] * 1536 for _ in texts],
            "usage": {"total_tokens": 50, "model": "text-embedding-3-small"},
        }

    mock_llm_service.embed_texts_with_usage = mock_embed_texts_with_usage

    # Create engine
    engine = IndexingEngine(
        store=mock_store,
        llm_service=mock_llm_service,
        index_config=config,
        openai_client=AsyncMock(),
    )

    # Create mock vector index that captures upsert calls
    mock_vector_index = MagicMock()
    upsert_calls: list[tuple[str, object, dict[str, object]]] = []

    def capture_upsert(items: list[tuple[str, object, dict[str, object]]]) -> None:
        upsert_calls.extend(items)

    mock_vector_index.upsert = capture_upsert

    # Run _embed_leaf
    with patch.object(engine, "_get_vector_index", return_value=mock_vector_index):
        job = EmbeddingJob(document_id="test-doc", leaf_id="test-leaf-id")
        await engine._embed_leaf(job)

    # Verify upsert was called
    assert len(upsert_calls) == 1, "vector_index.upsert should have been called once"

    node_id, _embedding, metadata = upsert_calls[0]
    assert node_id == "test-leaf-id"

    # Key assertions: level_index and coord_version must be present and correct
    assert "level_index" in metadata, (
        "metadata must include level_index for coverage builder optimization. "
        "Without it, retrieval falls back to slow database fetches."
    )
    assert metadata["level_index"] == 42, (
        f"level_index should be 42 (from leaf), got {metadata.get('level_index')}. "
        "This indicates level_index is defaulting to 0 instead of using the leaf's value."
    )

    assert "coord_version" in metadata, (
        "metadata must include coord_version for coverage builder optimization. "
        "coord_version=1 signals valid coordinates; 0 triggers database fallback."
    )
    assert metadata["coord_version"] == 1, (
        f"coord_version should be 1 (valid coordinates), got {metadata.get('coord_version')}. "
        "coord_version=0 causes coverage builder to ignore the metadata."
    )
