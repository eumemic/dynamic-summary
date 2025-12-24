"""Tests for batch append functionality.

Batch append allows multiple text units to be appended in a single call,
with each unit creating a forced split boundary. This is semantically
equivalent to N separate append calls but executed server-side in one
transaction.
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


@pytest.fixture(name="index_config")
def config_fixture() -> IndexConfig:
    return IndexConfig.load()


def _create_document(backend: SQLiteStorageBackend, document_id: str) -> DocumentStore:
    backend.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-nano",
    )
    return backend.for_document(document_id)


class TestBatchAppendEquivalence:
    """Verify batch_append produces identical results to sequential appends."""

    @pytest.mark.asyncio
    async def test_batch_append_equivalent_to_sequential_appends(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """batch_append([a,b,c]) produces same leaves as append(a);append(b);append(c)."""
        # Setup two documents for comparison
        store_batch = _create_document(sqlite_backend, "doc-batch")
        store_seq = _create_document(sqlite_backend, "doc-sequential")
        executor = AppendExecutor(index_config, StubEmbedder())

        units = ["First message.", "Second message.", "Third message."]

        # Batch append
        batch_outcome = await executor.append_batch(
            store=store_batch,
            document_id="doc-batch",
            units=units,
        )

        # Sequential appends
        seq_outcomes = []
        for unit in units:
            outcome = await executor.append(
                store=store_seq,
                document_id="doc-sequential",
                new_text=unit,
            )
            seq_outcomes.append(outcome)

        # Compare results
        batch_leaves = list(store_batch.nodes.get_leaves())
        seq_leaves = list(store_seq.nodes.get_leaves())

        # Same number of leaves
        assert len(batch_leaves) == len(seq_leaves)

        # Same leaf texts (in order)
        batch_texts = [
            leaf.text for leaf in sorted(batch_leaves, key=lambda n: n.span_start)
        ]
        seq_texts = [
            leaf.text for leaf in sorted(seq_leaves, key=lambda n: n.span_start)
        ]
        assert batch_texts == seq_texts

        # Same span coordinates
        batch_spans = [
            (leaf.span_start, leaf.span_end)
            for leaf in sorted(batch_leaves, key=lambda n: n.span_start)
        ]
        seq_spans = [
            (leaf.span_start, leaf.span_end)
            for leaf in sorted(seq_leaves, key=lambda n: n.span_start)
        ]
        assert batch_spans == seq_spans

        # Same total span
        assert batch_outcome.appended_span_start == 0
        assert batch_outcome.appended_span_end == seq_outcomes[-1].appended_span_end

    @pytest.mark.asyncio
    async def test_batch_append_single_unit_same_as_append(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """batch_append([x]) produces identical result to append(x)."""
        store_batch = _create_document(sqlite_backend, "doc-batch-single")
        store_single = _create_document(sqlite_backend, "doc-single")
        executor = AppendExecutor(index_config, StubEmbedder())

        text = "This is a single message."

        batch_outcome = await executor.append_batch(
            store=store_batch,
            document_id="doc-batch-single",
            units=[text],
        )

        single_outcome = await executor.append(
            store=store_single,
            document_id="doc-single",
            new_text=text,
        )

        batch_leaves = list(store_batch.nodes.get_leaves())
        single_leaves = list(store_single.nodes.get_leaves())

        assert len(batch_leaves) == len(single_leaves)
        assert batch_leaves[0].text == single_leaves[0].text
        assert batch_outcome.appended_span_end == single_outcome.appended_span_end


class TestBatchAppendBoundaries:
    """Verify batch append creates correct split boundaries."""

    @pytest.mark.asyncio
    async def test_batch_append_preserves_split_boundaries(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """Each unit creates a forced boundary - text never merged across units."""
        store = _create_document(sqlite_backend, "doc-boundaries")
        executor = AppendExecutor(index_config, StubEmbedder())

        # Two short units that would normally be merged by the splitter
        units = ["Short.", "Also short."]

        await executor.append_batch(
            store=store,
            document_id="doc-boundaries",
            units=units,
        )

        leaves = list(store.nodes.get_leaves())

        # Should have at least 2 leaves (one per unit minimum)
        assert len(leaves) >= 2

        # First leaf should contain first unit's text
        sorted_leaves = sorted(leaves, key=lambda n: n.span_start)
        assert "Short." in sorted_leaves[0].text
        # Second unit should start in a different leaf
        assert (
            "Also short." in sorted_leaves[-1].text
            or "Also short." in sorted_leaves[1].text
        )

    @pytest.mark.asyncio
    async def test_batch_append_large_unit_still_split(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """A single large unit can still be split by the internal splitter."""
        store = _create_document(sqlite_backend, "doc-large-unit")
        # Use small chunk size to force splitting
        small_config = IndexConfig.load(target_chunk_tokens=10)
        executor = AppendExecutor(small_config, StubEmbedder())

        # Large unit that exceeds target_chunk_tokens
        large_unit = "This is a much longer message. " * 20

        outcome = await executor.append_batch(
            store=store,
            document_id="doc-large-unit",
            units=[large_unit],
        )

        leaves = list(store.nodes.get_leaves())

        # Should have multiple leaves from splitting the large unit
        assert len(leaves) > 1
        assert len(outcome.new_leaf_ids) > 1


class TestBatchAppendEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_batch_append_empty_units_skipped(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """Empty strings in batch are skipped."""
        store = _create_document(sqlite_backend, "doc-empty")
        executor = AppendExecutor(index_config, StubEmbedder())

        units = ["First.", "", "  ", "Second."]

        await executor.append_batch(
            store=store,
            document_id="doc-empty",
            units=units,
        )

        leaves = list(store.nodes.get_leaves())

        # Only non-empty units should create leaves
        assert len(leaves) >= 2
        all_text = " ".join(leaf.text for leaf in leaves)
        assert "First." in all_text
        assert "Second." in all_text

    @pytest.mark.asyncio
    async def test_batch_append_all_empty_units(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """Batch of all empty units creates no leaves."""
        store = _create_document(sqlite_backend, "doc-all-empty")
        executor = AppendExecutor(index_config, StubEmbedder())

        units = ["", "  ", "\n"]

        outcome = await executor.append_batch(
            store=store,
            document_id="doc-all-empty",
            units=units,
        )

        leaves = list(store.nodes.get_leaves())
        assert len(leaves) == 0
        assert len(outcome.new_leaf_ids) == 0

    @pytest.mark.asyncio
    async def test_batch_append_empty_list(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """Empty batch creates no leaves."""
        store = _create_document(sqlite_backend, "doc-empty-list")
        executor = AppendExecutor(index_config, StubEmbedder())

        outcome = await executor.append_batch(
            store=store,
            document_id="doc-empty-list",
            units=[],
        )

        leaves = list(store.nodes.get_leaves())
        assert len(leaves) == 0
        assert len(outcome.new_leaf_ids) == 0


class TestBatchAppendWithExistingLeaves:
    """Test batch append when document already has leaves."""

    @pytest.mark.asyncio
    async def test_batch_append_links_to_existing_leaves(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """Batch append correctly links to existing rightmost leaf."""
        store = _create_document(sqlite_backend, "doc-existing")
        executor = AppendExecutor(index_config, StubEmbedder())

        # Create existing leaf
        store.nodes.add_batch(
            [
                {
                    "node_id": "existing",
                    "text": "Existing content.",
                    "span_start": 0,
                    "span_end": 17,
                    "parent_id": None,
                    "left_child_id": None,
                    "right_child_id": None,
                    "document_id": "doc-existing",
                    "token_count": 3,
                    "height": 0,
                    "level_index": 0,
                },
            ]
        )

        units = ["New first.", "New second."]

        outcome = await executor.append_batch(
            store=store,
            document_id="doc-existing",
            units=units,
        )

        # Existing leaf should link to first new leaf
        existing = store.nodes.get("existing")
        assert existing is not None
        assert existing.following_neighbor_id == outcome.new_leaf_ids[0]

        # First new leaf should link back to existing
        first_new = store.nodes.get(outcome.new_leaf_ids[0])
        assert first_new is not None
        assert first_new.preceding_neighbor_id == "existing"

        # Span should start where existing ended
        assert outcome.appended_span_start == 17

    @pytest.mark.asyncio
    async def test_batch_append_neighbor_chain_integrity(
        self, sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
    ) -> None:
        """All leaves form a valid doubly-linked list after batch append."""
        store = _create_document(sqlite_backend, "doc-chain")
        executor = AppendExecutor(index_config, StubEmbedder())

        units = ["One.", "Two.", "Three.", "Four."]

        await executor.append_batch(
            store=store,
            document_id="doc-chain",
            units=units,
        )

        leaves = sorted(store.nodes.get_leaves(), key=lambda n: n.span_start)

        # Check forward chain
        for i, leaf in enumerate(leaves[:-1]):
            assert leaf.following_neighbor_id == leaves[i + 1].id

        # Check backward chain
        for i, leaf in enumerate(leaves[1:], start=1):
            assert leaf.preceding_neighbor_id == leaves[i - 1].id

        # First has no preceding, last has no following
        assert leaves[0].preceding_neighbor_id is None
        assert leaves[-1].following_neighbor_id is None
