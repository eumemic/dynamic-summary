"""Tests for timestamp validation on subsequent appends.

Verifies that temporal documents enforce all-or-nothing timestamp rules:
- Temporal document + missing timestamps → Error
- Non-temporal document + provided timestamps → Error
"""

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.document_store import DocumentStore
from ragzoom.server.append_executor import AppendExecutor


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


class TestTemporalValidationOnSubsequentAppends:
    """Test timestamp validation on subsequent appends."""

    @pytest.mark.asyncio
    async def test_temporal_doc_append_without_timestamp_raises(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Temporal document + missing timestamp on subsequent append → Error."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-1")
        executor = AppendExecutor(config, StubEmbedder())

        # First append WITH timestamp → document becomes temporal
        await executor.append(
            store=store,
            document_id="doc-temporal-val-1",
            new_text="First message",
            timestamp="2024-01-21T14:30:00Z",
        )

        # Verify document is temporal
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-val-1")
            is True
        )

        # Second append WITHOUT timestamp → Error
        with pytest.raises(ValueError) as exc_info:
            await executor.append(
                store=store,
                document_id="doc-temporal-val-1",
                new_text="Second message",
                # No timestamp provided
            )

        error_msg = str(exc_info.value).lower()
        assert "temporal" in error_msg
        assert "timestamp" in error_msg

    @pytest.mark.asyncio
    async def test_non_temporal_doc_append_with_timestamp_raises(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Non-temporal document + timestamp on subsequent append → Error."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-2")
        executor = AppendExecutor(config, StubEmbedder())

        # First append WITHOUT timestamp → document becomes non-temporal
        await executor.append(
            store=store,
            document_id="doc-temporal-val-2",
            new_text="First message",
        )

        # Verify document is non-temporal
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-val-2")
            is False
        )

        # Second append WITH timestamp → Error
        with pytest.raises(ValueError) as exc_info:
            await executor.append(
                store=store,
                document_id="doc-temporal-val-2",
                new_text="Second message",
                timestamp="2024-01-21T14:30:05Z",
            )

        error_msg = str(exc_info.value).lower()
        assert "non-temporal" in error_msg or "temporal" in error_msg
        assert "timestamp" in error_msg

    @pytest.mark.asyncio
    async def test_temporal_doc_batch_append_without_timestamps_raises(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Temporal document + missing timestamps on batch append → Error."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-3")
        executor = AppendExecutor(config, StubEmbedder())

        # First batch append WITH timestamps → document becomes temporal
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-val-3",
            units=["Turn A"],
            timestamps=["2024-01-21T14:30:00Z"],
        )

        # Verify document is temporal
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-val-3")
            is True
        )

        # Second batch append WITHOUT timestamps → Error
        with pytest.raises(ValueError) as exc_info:
            await executor.append_batch(
                store=store,
                document_id="doc-temporal-val-3",
                units=["Turn B", "Turn C"],
                # No timestamps provided
            )

        error_msg = str(exc_info.value).lower()
        assert "temporal" in error_msg
        assert "timestamp" in error_msg

    @pytest.mark.asyncio
    async def test_non_temporal_doc_batch_append_with_timestamps_raises(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Non-temporal document + timestamps on batch append → Error."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-4")
        executor = AppendExecutor(config, StubEmbedder())

        # First batch append WITHOUT timestamps → document becomes non-temporal
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-val-4",
            units=["Turn A"],
        )

        # Verify document is non-temporal
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-temporal-val-4")
            is False
        )

        # Second batch append WITH timestamps → Error
        with pytest.raises(ValueError) as exc_info:
            await executor.append_batch(
                store=store,
                document_id="doc-temporal-val-4",
                units=["Turn B", "Turn C"],
                timestamps=["2024-01-21T14:30:05Z", "2024-01-21T14:30:10Z"],
            )

        error_msg = str(exc_info.value).lower()
        assert "non-temporal" in error_msg or "temporal" in error_msg
        assert "timestamp" in error_msg

    @pytest.mark.asyncio
    async def test_temporal_doc_valid_subsequent_append_succeeds(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Temporal document + timestamp on subsequent append → Success."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-5")
        executor = AppendExecutor(config, StubEmbedder())

        # First append WITH timestamp
        await executor.append(
            store=store,
            document_id="doc-temporal-val-5",
            new_text="First message",
            timestamp="2024-01-21T14:30:00Z",
        )

        # Second append WITH timestamp → should succeed
        await executor.append(
            store=store,
            document_id="doc-temporal-val-5",
            new_text="Second message",
            timestamp="2024-01-21T14:30:05Z",
        )

        # Verify both leaves exist
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 2

    @pytest.mark.asyncio
    async def test_non_temporal_doc_valid_subsequent_append_succeeds(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Non-temporal document + no timestamp on subsequent append → Success."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-6")
        executor = AppendExecutor(config, StubEmbedder())

        # First append WITHOUT timestamp
        await executor.append(
            store=store,
            document_id="doc-temporal-val-6",
            new_text="First message",
        )

        # Second append WITHOUT timestamp → should succeed
        await executor.append(
            store=store,
            document_id="doc-temporal-val-6",
            new_text="Second message",
        )

        # Verify both leaves exist
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 2

    @pytest.mark.asyncio
    async def test_temporal_doc_valid_batch_append_succeeds(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Temporal document + timestamps on batch append → Success."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-7")
        executor = AppendExecutor(config, StubEmbedder())

        # First batch append WITH timestamps
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-val-7",
            units=["Turn A"],
            timestamps=["2024-01-21T14:30:00Z"],
        )

        # Second batch append WITH timestamps → should succeed
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-val-7",
            units=["Turn B", "Turn C"],
            timestamps=["2024-01-21T14:30:05Z", "2024-01-21T14:30:10Z"],
        )

        # Verify all leaves exist
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 3

    @pytest.mark.asyncio
    async def test_non_temporal_doc_valid_batch_append_succeeds(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Non-temporal document + no timestamps on batch append → Success."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-8")
        executor = AppendExecutor(config, StubEmbedder())

        # First batch append WITHOUT timestamps
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-val-8",
            units=["Turn A"],
        )

        # Second batch append WITHOUT timestamps → should succeed
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-val-8",
            units=["Turn B", "Turn C"],
        )

        # Verify all leaves exist
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 3

    @pytest.mark.asyncio
    async def test_mixed_append_modes_temporal(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Temporal doc: single append then batch append with timestamps → Success."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-9")
        executor = AppendExecutor(config, StubEmbedder())

        # First: single append WITH timestamp
        await executor.append(
            store=store,
            document_id="doc-temporal-val-9",
            new_text="Single message",
            timestamp="2024-01-21T14:30:00Z",
        )

        # Second: batch append WITH timestamps → should succeed
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-val-9",
            units=["Batch A", "Batch B"],
            timestamps=["2024-01-21T14:30:05Z", "2024-01-21T14:30:10Z"],
        )

        leaves = store.nodes.get_leaves()
        assert len(leaves) == 3

    @pytest.mark.asyncio
    async def test_mixed_append_modes_non_temporal(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Non-temporal doc: batch append then single append without timestamps → Success."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-val-10")
        executor = AppendExecutor(config, StubEmbedder())

        # First: batch append WITHOUT timestamps
        await executor.append_batch(
            store=store,
            document_id="doc-temporal-val-10",
            units=["Batch A", "Batch B"],
        )

        # Second: single append WITHOUT timestamp → should succeed
        await executor.append(
            store=store,
            document_id="doc-temporal-val-10",
            new_text="Single message",
        )

        leaves = store.nodes.get_leaves()
        assert len(leaves) == 3


class TestTemporalIgnoresTargetChunkTokens:
    """Test that temporal documents ignore target_chunk_tokens config.

    Temporal documents always use client-controlled chunking to preserve
    the one-to-one mapping between input units and leaf nodes. When
    target_chunk_tokens is set in config, it is silently ignored for
    temporal documents.
    """

    @pytest.mark.asyncio
    async def test_append_with_timestamp_ignores_target_chunk_tokens(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """First append with timestamps + target_chunk_tokens set → Success (ignored)."""
        # Create config WITH target_chunk_tokens (would normally split)
        config = IndexConfig.load(target_chunk_tokens=512)
        store = _create_document(sqlite_backend, "doc-chunking-val-1")
        executor = AppendExecutor(config, StubEmbedder())

        # Append WITH timestamp succeeds - target_chunk_tokens is ignored
        await executor.append(
            store=store,
            document_id="doc-chunking-val-1",
            new_text="First message",
            timestamp="2024-01-21T14:30:00Z",
        )

        # Verify document is temporal and has exactly one leaf
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-chunking-val-1")
            is True
        )
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 1

    @pytest.mark.asyncio
    async def test_batch_append_with_timestamps_ignores_target_chunk_tokens(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """First batch append with timestamps + target_chunk_tokens set → Success."""
        # Create config WITH target_chunk_tokens (would normally split)
        config = IndexConfig.load(target_chunk_tokens=512)
        store = _create_document(sqlite_backend, "doc-chunking-val-2")
        executor = AppendExecutor(config, StubEmbedder())

        # Batch append WITH timestamps succeeds - target_chunk_tokens is ignored
        await executor.append_batch(
            store=store,
            document_id="doc-chunking-val-2",
            units=["Turn A", "Turn B"],
            timestamps=["2024-01-21T14:30:00Z", "2024-01-21T14:30:05Z"],
        )

        # Verify document is temporal and has exactly two leaves (one per unit)
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-chunking-val-2")
            is True
        )
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 2

    @pytest.mark.asyncio
    async def test_append_without_timestamp_respects_target_chunk_tokens(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """First append without timestamps + target_chunk_tokens set → uses chunking."""
        config = IndexConfig.load(target_chunk_tokens=512)
        store = _create_document(sqlite_backend, "doc-chunking-val-3")
        executor = AppendExecutor(config, StubEmbedder())

        # Append WITHOUT timestamp with target_chunk_tokens should succeed
        await executor.append(
            store=store,
            document_id="doc-chunking-val-3",
            new_text="First message without timestamp",
        )

        # Verify document is non-temporal
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-chunking-val-3")
            is False
        )

    @pytest.mark.asyncio
    async def test_append_with_timestamp_tuple_ignores_target_chunk_tokens(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """First append with timestamp tuple + target_chunk_tokens set → Success."""
        config = IndexConfig.load(target_chunk_tokens=512)
        store = _create_document(sqlite_backend, "doc-chunking-val-4")
        executor = AppendExecutor(config, StubEmbedder())

        # Append WITH timestamp tuple succeeds - target_chunk_tokens is ignored
        await executor.append(
            store=store,
            document_id="doc-chunking-val-4",
            new_text="First message",
            timestamp=("2024-01-21T14:30:00Z", "2024-01-21T14:30:05Z"),
        )

        # Verify document is temporal and has exactly one leaf
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-chunking-val-4")
            is True
        )
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 1

    @pytest.mark.asyncio
    async def test_subsequent_append_to_temporal_doc_ignores_server_chunking(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Subsequent append to temporal document with different config succeeds.

        Scenario: A temporal document was created with target_chunk_tokens=None,
        then someone appends with an executor that has target_chunk_tokens set
        (e.g., daemon restarted with different config).

        This should succeed - the config setting is ignored for temporal documents.
        """
        # Step 1: Create temporal document with target_chunk_tokens=None
        correct_config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-subsequent-guard")
        correct_executor = AppendExecutor(correct_config, StubEmbedder())

        await correct_executor.append(
            store=store,
            document_id="doc-subsequent-guard",
            new_text="First message",
            timestamp="2024-01-21T14:30:00Z",
        )

        # Verify document is temporal
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal("doc-subsequent-guard")
            is True
        )

        # Step 2: Append with different config (target_chunk_tokens set)
        different_config = IndexConfig.load(target_chunk_tokens=512)
        different_executor = AppendExecutor(different_config, StubEmbedder())

        # This succeeds - target_chunk_tokens is ignored for temporal docs
        await different_executor.append(
            store=store,
            document_id="doc-subsequent-guard",
            new_text="Second message",
            timestamp="2024-01-21T14:30:05Z",
        )

        # Verify we have exactly 2 leaves (one per append, no splitting)
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 2

    @pytest.mark.asyncio
    async def test_subsequent_batch_append_to_temporal_doc_ignores_server_chunking(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Subsequent batch append to temporal document with different config succeeds."""
        # Step 1: Create temporal document
        correct_config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-subsequent-batch-guard")
        correct_executor = AppendExecutor(correct_config, StubEmbedder())

        await correct_executor.append_batch(
            store=store,
            document_id="doc-subsequent-batch-guard",
            units=["Turn A"],
            timestamps=["2024-01-21T14:30:00Z"],
        )

        # Verify document is temporal
        assert (
            sqlite_backend.doc_repo.get_document_is_temporal(
                "doc-subsequent-batch-guard"
            )
            is True
        )

        # Step 2: Batch append with different config
        different_config = IndexConfig.load(target_chunk_tokens=512)
        different_executor = AppendExecutor(different_config, StubEmbedder())

        # This succeeds - target_chunk_tokens is ignored for temporal docs
        await different_executor.append_batch(
            store=store,
            document_id="doc-subsequent-batch-guard",
            units=["Turn B", "Turn C"],
            timestamps=["2024-01-21T14:30:05Z", "2024-01-21T14:30:10Z"],
        )

        # Verify we have exactly 3 leaves (one per unit, no splitting)
        leaves = store.nodes.get_leaves()
        assert len(leaves) == 3
