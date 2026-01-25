"""Tests for temporal document configuration error messages.

This module tests that error messages for temporal document misconfigurations
are clear, actionable, and guide users to the correct fix.
"""

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.document_store import DocumentStore
from ragzoom.server.append_executor import AppendExecutor


class StubEmbedder(EmbeddingProvider):
    """Stub embedder that returns deterministic embeddings."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]


def _create_document(backend: SQLiteStorageBackend, doc_id: str) -> DocumentStore:
    """Create a document and return its store."""
    backend.add_document(
        document_id=doc_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-nano",
    )
    return backend.for_document(doc_id)


class TestHelpfulErrorForTemporalWithoutNullChunks:
    """Test that temporal document config mismatch produces helpful error."""

    @pytest.mark.asyncio
    async def test_helpful_error_for_temporal_without_null_chunks(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Error message includes exact fix for temporal document config mismatch.

        Spec: Phase 33 - Temporal Document UX Improvement
        Success: Error message includes exact fix:
            "Temporal documents require target_chunk_tokens=null in config"
        """
        # Create config WITH target_chunk_tokens (server-controlled chunking)
        # This is the WRONG config for temporal documents
        config = IndexConfig.load(target_chunk_tokens=512)
        store = _create_document(sqlite_backend, "doc-temporal-error")
        executor = AppendExecutor(config, StubEmbedder())

        # Attempt append WITH timestamp should fail with helpful error
        with pytest.raises(ValueError) as exc_info:
            await executor.append(
                store=store,
                document_id="doc-temporal-error",
                new_text="Message with timestamp",
                timestamp="2024-01-21T14:30:00Z",
            )

        error_msg = str(exc_info.value)

        # Error should explain WHAT is wrong
        assert "temporal" in error_msg.lower()

        # Error should explain HOW to fix it
        assert "target_chunk_tokens" in error_msg.lower()
        assert "null" in error_msg.lower() or "none" in error_msg.lower()

        # Error should mention config file
        assert "config" in error_msg.lower()

    @pytest.mark.asyncio
    async def test_helpful_error_mentions_why(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Error message explains WHY target_chunk_tokens must be null.

        Good error messages explain the reason, not just the fix.
        """
        config = IndexConfig.load(target_chunk_tokens=512)
        store = _create_document(sqlite_backend, "doc-temporal-why")
        executor = AppendExecutor(config, StubEmbedder())

        with pytest.raises(ValueError) as exc_info:
            await executor.append(
                store=store,
                document_id="doc-temporal-why",
                new_text="Test message",
                timestamp="2024-01-21T14:30:00Z",
            )

        error_msg = str(exc_info.value)

        # Error should mention "each" or "per" to explain one leaf per unit
        assert (
            "each" in error_msg.lower()
            or "per" in error_msg.lower()
            or "one-to-one" in error_msg.lower()
            or "preserves" in error_msg.lower()
        )

    @pytest.mark.asyncio
    async def test_batch_append_also_has_helpful_error(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Batch append with timestamps also produces helpful error."""
        config = IndexConfig.load(target_chunk_tokens=512)
        store = _create_document(sqlite_backend, "doc-batch-error")
        executor = AppendExecutor(config, StubEmbedder())

        with pytest.raises(ValueError) as exc_info:
            await executor.append_batch(
                store=store,
                document_id="doc-batch-error",
                units=["Turn A", "Turn B"],
                timestamps=["2024-01-21T14:30:00Z", "2024-01-21T14:30:05Z"],
            )

        error_msg = str(exc_info.value)

        # Same helpful error message for batch append
        assert "temporal" in error_msg.lower()
        assert "target_chunk_tokens" in error_msg.lower()
        assert "null" in error_msg.lower() or "none" in error_msg.lower()
        assert "config" in error_msg.lower()
