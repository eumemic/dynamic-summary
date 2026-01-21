"""Tests for is_temporal inference from first append.

Verifies that the document's is_temporal flag is inferred from the first append:
- First append WITH timestamps → document becomes temporal (is_temporal=True)
- First append WITHOUT timestamps → document becomes non-temporal (is_temporal=False)
"""

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.document_store import DocumentStore


class StubEmbedder(EmbeddingProvider):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]


def _create_document(backend: SQLiteStorageBackend, document_id: str) -> DocumentStore:
    backend.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-nano",
    )
    return backend.for_document(document_id)


class TestIsTemporalInference:
    """Test is_temporal inference from first append."""

    @pytest.mark.asyncio
    async def test_first_append_with_timestamp_sets_is_temporal_true(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """First append WITH timestamps → document becomes temporal."""
        from ragzoom.server.append_executor import AppendExecutor

        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-inference-1")
        executor = AppendExecutor(config, StubEmbedder())

        # Before append, is_temporal should be False (default)
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-1")
            is False
        )

        # Append WITH timestamp
        await executor.append(
            store=store,
            document_id="doc-temporal-inference-1",
            new_text="Hello world",
            timestamp="2024-01-21T14:30:00Z",
        )

        # After append, is_temporal should be True
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-1")
            is True
        )

    @pytest.mark.asyncio
    async def test_first_append_without_timestamp_sets_is_temporal_false(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """First append WITHOUT timestamps → document becomes non-temporal."""
        from ragzoom.server.append_executor import AppendExecutor

        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-inference-2")
        executor = AppendExecutor(config, StubEmbedder())

        # Before append, is_temporal should be False (default)
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-2")
            is False
        )

        # Append WITHOUT timestamp
        await executor.append(
            store=store,
            document_id="doc-temporal-inference-2",
            new_text="Hello world",
        )

        # After append, is_temporal should still be False (non-temporal document)
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-2")
            is False
        )

    @pytest.mark.asyncio
    async def test_first_batch_append_with_timestamps_sets_is_temporal_true(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """First batch append WITH timestamps → document becomes temporal."""
        from ragzoom.server.append_executor import AppendExecutor

        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-inference-3")
        executor = AppendExecutor(config, StubEmbedder())

        # Before append, is_temporal should be False (default)
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-3")
            is False
        )

        # Batch append WITH timestamps
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-inference-3",
            units=["Turn A", "Turn B"],
            timestamps=["2024-01-21T14:30:00Z", "2024-01-21T14:30:05Z"],
        )

        # After append, is_temporal should be True
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-3")
            is True
        )

    @pytest.mark.asyncio
    async def test_first_batch_append_without_timestamps_sets_is_temporal_false(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """First batch append WITHOUT timestamps → document becomes non-temporal."""
        from ragzoom.server.append_executor import AppendExecutor

        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-inference-4")
        executor = AppendExecutor(config, StubEmbedder())

        # Before append, is_temporal should be False (default)
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-4")
            is False
        )

        # Batch append WITHOUT timestamps
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-inference-4",
            units=["Turn A", "Turn B"],
        )

        # After append, is_temporal should still be False (non-temporal document)
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-4")
            is False
        )

    @pytest.mark.asyncio
    async def test_subsequent_append_does_not_change_is_temporal(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Subsequent appends should not change is_temporal (already inferred from first)."""
        from ragzoom.server.append_executor import AppendExecutor

        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-inference-5")
        executor = AppendExecutor(config, StubEmbedder())

        # First append WITH timestamp → sets is_temporal=True
        await executor.append(
            store=store,
            document_id="doc-temporal-inference-5",
            new_text="First message",
            timestamp="2024-01-21T14:30:00Z",
        )
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-5")
            is True
        )

        # Second append WITH timestamp → is_temporal should remain True
        await executor.append(
            store=store,
            document_id="doc-temporal-inference-5",
            new_text="Second message",
            timestamp="2024-01-21T14:30:05Z",
        )
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-inference-5")
            is True
        )
