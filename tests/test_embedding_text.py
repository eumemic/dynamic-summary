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


@pytest.mark.asyncio
async def test_passthrough_no_llm_call() -> None:
    """Test that small content passes through without LLM call.

    Acceptance test for specs/embedding-text-optimization.md § Acceptance Criteria > 2:
    "Passthrough for small content: No LLM call when combined tokens <= target"

    When combined content (preceding_context + leaf_text) fits within the
    target_embedding_tokens limit, the workflow should return the combined text
    directly without invoking the LLM. This is a performance optimization - most
    leaves are small and shouldn't require LLM processing.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from ragzoom.contracts.embedding_model import EmbeddingResult
    from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine
    from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult

    # Create small content that fits within default target (500 tokens)
    # Each word is roughly 1 token, so ~100 words is well under the limit
    small_leaf_text = "This is a small piece of content. " * 10  # ~80 tokens
    preceding_context = "Prior conversation context. " * 5  # ~25 tokens
    expected_combined = f"{preceding_context}\n{small_leaf_text}"

    # Verify content is small enough for passthrough
    token_count = tokenizer.count_tokens(expected_combined.strip())
    assert token_count < 500, (
        f"Test setup error: combined content should be under 500 tokens, got {token_count}. "
        "This ensures passthrough behavior is tested."
    )

    # Setup mocks for storage backend
    mock_store = MagicMock()
    mock_doc_store = MagicMock()
    mock_store.for_document.return_value = mock_doc_store

    # Create mock leaf with small text
    mock_leaf = MagicMock()
    mock_leaf.text = small_leaf_text
    mock_leaf.id = "small-leaf-id"
    mock_leaf.span_start = 100  # Not first leaf, so preceding context applies
    mock_leaf.span_end = mock_leaf.span_start + len(small_leaf_text)
    mock_leaf.level_index = 0
    mock_leaf.coord_version = 1
    mock_leaf.parent_id = None
    mock_doc_store.nodes.get.return_value = mock_leaf

    # Track LLM calls - this should NOT be called for passthrough
    llm_call_count = 0

    # Setup mock LLM service
    mock_llm_service = MagicMock()

    # Track embedding calls
    embed_texts_received: list[str] = []

    async def mock_embed_texts_with_usage(texts: list[str]) -> EmbeddingResult:
        embed_texts_received.extend(texts)
        return {
            "embeddings": [[0.1] * 1536 for _ in texts],
            "usage": {"total_tokens": 50, "model": "text-embedding-3-small"},
        }

    mock_llm_service.embed_texts_with_usage = mock_embed_texts_with_usage

    # Mock _prepare_embedding_text to track calls and verify passthrough behavior
    async def mock_prepare_embedding_text(
        preceding_context: str,
        leaf_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object = None,
    ) -> SummaryResult:
        nonlocal llm_call_count
        # Calculate combined text the same way the real implementation does
        combined = (
            f"{preceding_context}\n{leaf_text}"
            if preceding_context.strip()
            else leaf_text
        )
        combined_tokens = tokenizer.count_tokens(combined.strip())

        # Passthrough: return combined text directly, no LLM invocation
        if target_tokens <= 0 or combined_tokens <= target_tokens:
            # Return with zero usage - indicates no LLM call was made
            return SummaryResult(
                summary=combined.strip(),
                retry_count=0,
                summary_tokens=combined_tokens,
                usage=AccumulatedUsage(),  # Zero usage = no LLM call
            )

        # If we get here, LLM would be called (shouldn't happen for small content)
        llm_call_count += 1
        return SummaryResult(
            summary="LLM-processed text",
            retry_count=0,
            summary_tokens=50,
            usage=AccumulatedUsage(prompt_tokens=100, completion_tokens=50),
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

    # Mock preceding context retrieval
    # Simulate preceding context being available
    mock_preceding_node = MagicMock()
    mock_preceding_node.text = preceding_context

    mock_vector_index = MagicMock()
    mock_vector_index.upsert = MagicMock()

    # Mock the methods that retrieve preceding context
    from ragzoom.server.indexing_engine import PrecedingContextResult

    # Create a mock tiling node that provides the preceding context
    context_node_id = "context-node-id"
    mock_tiling_node = MagicMock()
    mock_tiling_node.text = preceding_context.strip()

    mock_context_result = PrecedingContextResult(
        tiling_ids=[context_node_id],
        nodes={context_node_id: mock_tiling_node},
        tiling_tokens=tokenizer.count_tokens(preceding_context),
    )

    with (
        patch.object(engine, "_get_vector_index", return_value=mock_vector_index),
        patch.object(
            engine,
            "_get_preceding_context",
            new_callable=AsyncMock,
            return_value=mock_context_result,
        ),
    ):
        job = EmbeddingJob(document_id="test-doc", leaf_id="small-leaf-id")
        await engine._embed_leaf(job)

    # Verify: No LLM call was made (passthrough path taken)
    assert llm_call_count == 0, (
        "LLM should not be called for small content. "
        f"Expected 0 LLM calls, got {llm_call_count}. "
        "Passthrough should skip LLM processing when combined tokens <= target."
    )

    # Verify: Embedding was still created
    assert len(embed_texts_received) == 1, (
        "embed_texts_with_usage should have been called once. "
        "Passthrough skips LLM but still creates embeddings."
    )

    # Verify: The embedded text is the combined passthrough text
    embedded_text = embed_texts_received[0]
    assert preceding_context.strip() in embedded_text, (
        f"Passthrough text should include preceding context. "
        f"Got: {embedded_text[:200]}"
    )
    assert small_leaf_text.strip() in embedded_text, (
        f"Passthrough text should include leaf text. " f"Got: {embedded_text[:200]}"
    )


@pytest.mark.asyncio
async def test_original_text_preserved() -> None:
    """Test that leaf text in database is unchanged; only embedding uses optimized version.

    Acceptance test for specs/embedding-text-optimization.md § Acceptance Criteria > 5:
    "Original text preserved: Leaf text in database unchanged; only embedding uses
    optimized version"

    The embedding text optimization compresses large content for better retrieval, but
    the original verbatim text must be preserved in the database. Users querying the
    leaf node should see the original text, not the compressed version. Only the
    embedding vector should be based on the optimized text.

    The spec says:
    - "Leaf text in database: Unchanged (original verbatim text preserved)"
    - "Embedding vector: Based on retrieval-optimized text"
    - "preceding_context_summary field: Repurposed to store the retrieval-optimized text"
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from ragzoom.contracts.embedding_model import EmbeddingResult
    from ragzoom.server.indexing_engine import EmbeddingJob, IndexingEngine
    from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult

    # Create original leaf text that exceeds target and will be compressed
    original_leaf_text = (
        "This is the original, verbatim content that users uploaded. "
        "It contains important details about the project timeline and deliverables. "
        "The team discussed implementing a new authentication system using OAuth 2.0. "
        "Alice mentioned that the deadline is April 15th, and Bob agreed to handle "
        "the frontend integration. Carol will review the security implications. "
    ) * 20  # Repeat to exceed target

    optimized_text = (
        "OAuth 2.0 authentication implementation. Deadline: April 15th. "
        "Team: Alice (timeline), Bob (frontend), Carol (security review)."
    )

    # Verify original exceeds target (should trigger compression)
    token_count = tokenizer.count_tokens(original_leaf_text)
    assert token_count > 500, (
        f"Test setup error: original text should exceed 500 tokens, got {token_count}. "
        "This ensures compression is triggered."
    )

    # Setup mocks for storage backend
    mock_store = MagicMock()
    mock_doc_store = MagicMock()
    mock_store.for_document.return_value = mock_doc_store

    # Create mock leaf with original text
    mock_leaf = MagicMock()
    mock_leaf.text = original_leaf_text
    mock_leaf.id = "test-leaf-id"
    mock_leaf.span_start = 0
    mock_leaf.span_end = len(original_leaf_text)
    mock_leaf.level_index = 0
    mock_leaf.coord_version = 1
    mock_leaf.parent_id = None

    # Track any changes to the leaf's text attribute
    original_text_at_start = mock_leaf.text

    mock_doc_store.nodes.get.return_value = mock_leaf

    # Track calls to update_preceding_context_summary
    update_context_summary_calls: list[tuple[str, str]] = []
    mock_nodes_repo = MagicMock()

    def capture_update_context_summary(leaf_id: str, text: str) -> None:
        update_context_summary_calls.append((leaf_id, text))

    mock_nodes_repo.update_preceding_context_summary = capture_update_context_summary
    mock_doc_store.nodes._repo = mock_nodes_repo

    # Setup mock LLM service
    mock_llm_service = MagicMock()

    # Track what text is embedded
    embed_texts_received: list[str] = []

    async def mock_embed_texts_with_usage(texts: list[str]) -> EmbeddingResult:
        embed_texts_received.extend(texts)
        return {
            "embeddings": [[0.1] * 1536 for _ in texts],
            "usage": {"total_tokens": 50, "model": "text-embedding-3-small"},
        }

    mock_llm_service.embed_texts_with_usage = mock_embed_texts_with_usage

    # Mock _prepare_embedding_text to return optimized text
    async def mock_prepare_embedding_text(
        preceding_context: str,
        leaf_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object = None,
    ) -> SummaryResult:
        # Return optimized text (simulating LLM compression)
        return SummaryResult(
            summary=optimized_text,
            retry_count=0,
            summary_tokens=tokenizer.count_tokens(optimized_text),
            usage=AccumulatedUsage(prompt_tokens=500, completion_tokens=50),
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

    # Create mock vector index
    mock_vector_index = MagicMock()
    mock_vector_index.upsert = MagicMock()

    # Execute: Run embedding job
    with patch.object(engine, "_get_vector_index", return_value=mock_vector_index):
        job = EmbeddingJob(document_id="test-doc", leaf_id="test-leaf-id")
        await engine._embed_leaf(job)

    # VERIFY 1: Original text in database is UNCHANGED
    # The mock leaf's text attribute should still contain the original text
    assert mock_leaf.text == original_text_at_start, (
        "Original leaf text should be preserved in database. "
        f"Expected: {original_text_at_start[:100]}... "
        f"Got: {mock_leaf.text[:100]}..."
    )
    assert (
        mock_leaf.text == original_leaf_text
    ), "Leaf text should match the original input text exactly"

    # VERIFY 2: Embedding used the OPTIMIZED text, not the original
    assert len(embed_texts_received) == 1, "Embedding should have been called once"
    assert embed_texts_received[0] == optimized_text, (
        "Embedding should use optimized text, not original. "
        f"Expected optimized: {optimized_text[:100]}... "
        f"Got: {embed_texts_received[0][:100]}..."
    )
    assert (
        embed_texts_received[0] != original_leaf_text
    ), "Embedding should NOT use the original text directly"

    # VERIFY 3: Optimized text stored in preceding_context_summary for debugging
    assert (
        len(update_context_summary_calls) == 1
    ), "update_preceding_context_summary should have been called once"
    stored_leaf_id, stored_text = update_context_summary_calls[0]
    assert stored_leaf_id == "test-leaf-id", "Should update the correct leaf"
    assert stored_text == optimized_text, (
        "preceding_context_summary should store the optimized text for inspection. "
        f"Expected: {optimized_text[:100]}... "
        f"Got: {stored_text[:100]}..."
    )
