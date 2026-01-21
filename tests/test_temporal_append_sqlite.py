"""Tests for temporal metadata in append operations."""

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


class TestAppendWithTimestamp:
    """Test append() with optional timestamp parameter."""

    @pytest.mark.asyncio
    async def test_append_with_single_timestamp_string(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Append with a single timestamp string (used for both start and end)."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-1")
        executor = AppendExecutor(config, StubEmbedder())

        await executor.append(
            store=store,
            document_id="doc-temporal-1",
            new_text="Hello world",
            timestamp="2024-01-21T14:30:00Z",
        )

        leaves = store.nodes.get_leaves()
        assert len(leaves) == 1

        leaf = leaves[0]
        # time_start and time_end should both be set to the same Unix timestamp
        assert leaf.time_start == 1705847400.0
        assert leaf.time_end == 1705847400.0

    @pytest.mark.asyncio
    async def test_append_with_timestamp_tuple(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Append with a (start, end) timestamp tuple."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-2")
        executor = AppendExecutor(config, StubEmbedder())

        await executor.append(
            store=store,
            document_id="doc-temporal-2",
            new_text="Hello world",
            timestamp=("2024-01-21T14:30:00Z", "2024-01-21T14:30:12Z"),
        )

        leaves = store.nodes.get_leaves()
        assert len(leaves) == 1

        leaf = leaves[0]
        assert leaf.time_start == 1705847400.0
        assert leaf.time_end == 1705847412.0  # 12 seconds later

    @pytest.mark.asyncio
    async def test_append_without_timestamp(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Append without timestamp leaves temporal fields as None."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-3")
        executor = AppendExecutor(config, StubEmbedder())

        await executor.append(
            store=store,
            document_id="doc-temporal-3",
            new_text="Hello world",
        )

        leaves = store.nodes.get_leaves()
        assert len(leaves) == 1

        leaf = leaves[0]
        assert leaf.time_start is None
        assert leaf.time_end is None

    @pytest.mark.asyncio
    async def test_append_with_timestamp_microseconds(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Append with timestamp containing microseconds."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-4")
        executor = AppendExecutor(config, StubEmbedder())

        await executor.append(
            store=store,
            document_id="doc-temporal-4",
            new_text="Precise timing",
            timestamp="2024-01-21T14:30:00.123456Z",
        )

        leaves = store.nodes.get_leaves()
        assert len(leaves) == 1

        leaf = leaves[0]
        # Check microsecond precision
        assert leaf.time_start is not None
        assert leaf.time_end is not None
        assert abs(leaf.time_start - 1705847400.123456) < 0.000001
        assert abs(leaf.time_end - 1705847400.123456) < 0.000001

    @pytest.mark.asyncio
    async def test_append_with_invalid_timestamp_format_raises(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Append with invalid timestamp format raises ValueError."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-5")
        executor = AppendExecutor(config, StubEmbedder())

        with pytest.raises(ValueError) as exc_info:
            await executor.append(
                store=store,
                document_id="doc-temporal-5",
                new_text="Bad timestamp",
                timestamp="2024-01-21T14:30:00",  # Missing timezone
            )

        assert "timezone" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_append_with_invalid_timestamp_range_raises(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Append with time_end < time_start raises ValueError."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-6")
        executor = AppendExecutor(config, StubEmbedder())

        with pytest.raises(ValueError) as exc_info:
            await executor.append(
                store=store,
                document_id="doc-temporal-6",
                new_text="Invalid range",
                timestamp=("2024-01-21T15:00:00Z", "2024-01-21T14:30:00Z"),
            )

        assert "time_end" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_append_with_timezone_offset(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Append with timestamp using timezone offset notation."""
        config = IndexConfig.load(target_chunk_tokens=None)
        store = _create_document(sqlite_backend, "doc-temporal-7")
        executor = AppendExecutor(config, StubEmbedder())

        await executor.append(
            store=store,
            document_id="doc-temporal-7",
            new_text="Offset timezone",
            timestamp="2024-01-21T09:30:00-05:00",  # Same as 14:30:00 UTC
        )

        leaves = store.nodes.get_leaves()
        assert len(leaves) == 1

        leaf = leaves[0]
        assert leaf.time_start == 1705847400.0
        assert leaf.time_end == 1705847400.0
