"""Tests for batch_append() accepting list[AppendUnit].

These tests verify that batch_append() can accept AppendUnit objects that bundle
text with optional timestamps, instead of requiring parallel arrays.
"""

import pytest

from ragzoom import AppendUnit, AsyncRagZoom, RagZoom
from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.server.append_executor import AppendExecutor


class StubEmbedder(EmbeddingProvider):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]


@pytest.fixture(name="index_config")
def config_fixture() -> IndexConfig:
    return IndexConfig.load()


def _create_document(backend: SQLiteStorageBackend, document_id: str) -> None:
    backend.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-nano",
    )


class TestBatchAppendAcceptsAppendUnits:
    """Test batch_append() accepts list[AppendUnit]."""

    @pytest.mark.asyncio
    async def test_batch_append_with_non_temporal_units(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """batch_append() accepts AppendUnit objects without timestamps."""
        _create_document(sqlite_backend, "doc-units")
        store = sqlite_backend.for_document("doc-units")
        executor = AppendExecutor(index_config, StubEmbedder())

        units = [
            AppendUnit(text="First message."),
            AppendUnit(text="Second message."),
            AppendUnit(text="Third message."),
        ]

        outcome = await executor.append_batch(
            store=store,
            document_id="doc-units",
            units=units,
        )

        leaves = list(store.nodes.get_leaves())
        assert len(leaves) >= 3
        all_text = " ".join(leaf.text for leaf in leaves)
        assert "First message." in all_text
        assert "Second message." in all_text
        assert "Third message." in all_text
        assert len(outcome.new_leaf_ids) >= 3

    @pytest.mark.asyncio
    async def test_batch_append_with_temporal_units(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """batch_append() accepts AppendUnit objects with timestamps."""
        # Use client-controlled chunking (target_chunk_tokens=None) for temporal
        _create_document(sqlite_backend, "doc-temporal-units")
        store = sqlite_backend.for_document("doc-temporal-units")
        client_controlled_config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(client_controlled_config, StubEmbedder())

        units = [
            AppendUnit(
                text="User asked a question",
                time_start="2024-01-21T14:30:00Z",
                time_end="2024-01-21T14:30:05Z",
            ),
            AppendUnit(
                text="Assistant responded",
                time_start="2024-01-21T14:30:06Z",
                time_end="2024-01-21T14:30:15Z",
            ),
        ]

        await executor.append_batch(
            store=store,
            document_id="doc-temporal-units",
            units=units,
        )

        leaves = sorted(store.nodes.get_leaves(), key=lambda n: n.span_start)
        assert len(leaves) == 2

        # Check timestamps are stored
        assert leaves[0].time_start is not None
        assert leaves[0].time_end is not None
        assert leaves[1].time_start is not None
        assert leaves[1].time_end is not None

        # Verify document is marked temporal
        doc_is_temporal = sqlite_backend.doc_repo.get_document_is_temporal(
            "doc-temporal-units"
        )
        assert doc_is_temporal is True

    @pytest.mark.asyncio
    async def test_batch_append_equivalent_to_parallel_arrays(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """batch_append() with AppendUnits produces same result as parallel arrays."""
        # Setup two documents for comparison
        _create_document(sqlite_backend, "doc-append-units")
        _create_document(sqlite_backend, "doc-parallel")
        store_units = sqlite_backend.for_document("doc-append-units")
        store_parallel = sqlite_backend.for_document("doc-parallel")

        client_controlled_config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(client_controlled_config, StubEmbedder())

        # AppendUnit approach
        append_units = [
            AppendUnit(
                text="First message.",
                time_start="2024-01-21T14:30:00Z",
                time_end="2024-01-21T14:30:05Z",
            ),
            AppendUnit(
                text="Second message.",
                time_start="2024-01-21T14:30:06Z",
                time_end="2024-01-21T14:30:10Z",
            ),
        ]

        await executor.append_batch(
            store=store_units,
            document_id="doc-append-units",
            units=append_units,
        )

        # Parallel arrays approach (current API)
        texts = ["First message.", "Second message."]
        timestamps = [
            ("2024-01-21T14:30:00Z", "2024-01-21T14:30:05Z"),
            ("2024-01-21T14:30:06Z", "2024-01-21T14:30:10Z"),
        ]

        await executor.append_batch(
            store=store_parallel,
            document_id="doc-parallel",
            units=texts,
            timestamps=timestamps,
        )

        # Compare results
        leaves_units = sorted(
            store_units.nodes.get_leaves(), key=lambda n: n.span_start
        )
        leaves_parallel = sorted(
            store_parallel.nodes.get_leaves(), key=lambda n: n.span_start
        )

        assert len(leaves_units) == len(leaves_parallel)
        for u, p in zip(leaves_units, leaves_parallel):
            assert u.text == p.text
            assert u.time_start == p.time_start
            assert u.time_end == p.time_end

    @pytest.mark.asyncio
    async def test_batch_append_empty_append_units(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """batch_append() with empty AppendUnit list creates no leaves."""
        _create_document(sqlite_backend, "doc-empty-units")
        store = sqlite_backend.for_document("doc-empty-units")
        executor = AppendExecutor(index_config, StubEmbedder())

        units: list[AppendUnit] = []

        outcome = await executor.append_batch(
            store=store,
            document_id="doc-empty-units",
            units=units,
        )

        leaves = list(store.nodes.get_leaves())
        assert len(leaves) == 0
        assert len(outcome.new_leaf_ids) == 0


class TestRagZoomBatchAppendAcceptsAppendUnits:
    """Test RagZoom.batch_append() wrapper accepts list[AppendUnit]."""

    def test_batch_append_validates_document_id(self) -> None:
        """batch_append() raises on empty document_id."""
        client = RagZoom(server_address="localhost:50051")
        with pytest.raises(ValueError, match="document_id"):
            client.batch_append("", [AppendUnit(text="Hello")])

    def test_batch_append_validates_empty_units(self) -> None:
        """batch_append() raises on empty units list."""
        client = RagZoom(server_address="localhost:50051")
        with pytest.raises(ValueError, match="units"):
            client.batch_append("doc", [])


class TestAsyncRagZoomBatchAppendAcceptsAppendUnits:
    """Test AsyncRagZoom.batch_append() wrapper accepts list[AppendUnit]."""

    @pytest.mark.asyncio
    async def test_batch_append_validates_document_id(self) -> None:
        """batch_append() raises on empty document_id."""
        client = AsyncRagZoom(server_address="localhost:50051")
        with pytest.raises(ValueError, match="document_id"):
            await client.batch_append("", [AppendUnit(text="Hello")])

    @pytest.mark.asyncio
    async def test_batch_append_validates_empty_units(self) -> None:
        """batch_append() raises on empty units list."""
        client = AsyncRagZoom(server_address="localhost:50051")
        with pytest.raises(ValueError, match="units"):
            await client.batch_append("doc", [])
