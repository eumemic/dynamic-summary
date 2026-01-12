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


@pytest.mark.asyncio
async def test_append_creates_leaves_from_scratch(
    sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
) -> None:
    store = _create_document(sqlite_backend, "doc-1")
    executor = AppendExecutor(index_config, StubEmbedder())

    outcome = await executor.append(
        store=store,
        document_id="doc-1",
        new_text="The quick brown fox",
    )

    leaves = store.nodes.get_leaves()
    assert len(leaves) == len(outcome.new_leaf_ids)
    assert outcome.total_leaves == len(leaves)
    assert all(node.parent_id is None for node in leaves)
    assert set(outcome.new_leaf_ids) == {leaf.id for leaf in leaves}


@pytest.mark.asyncio
async def test_append_preserves_existing_leaves_and_links_neighbors(
    sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
) -> None:
    """Appending new text creates new leaves without modifying existing ones."""
    store = _create_document(sqlite_backend, "doc-2")
    executor = AppendExecutor(index_config, StubEmbedder())

    # Create two existing leaves
    store.nodes.add_batch(
        [
            {
                "node_id": "left",
                "text": "AAA",
                "span_start": 0,
                "span_end": 3,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
                "document_id": "doc-2",
                "token_count": 3,
                "height": 0,
                "level_index": 0,
                "following_neighbor_id": "tail",
            },
            {
                "node_id": "tail",
                "text": "BBB",
                "span_start": 3,
                "span_end": 6,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
                "document_id": "doc-2",
                "token_count": 3,
                "height": 0,
                "level_index": 1,
                "preceding_neighbor_id": "left",
            },
        ]
    )

    outcome = await executor.append(
        store=store,
        document_id="doc-2",
        new_text="CCC",
    )

    # No nodes should be deleted - append is append-only
    assert outcome.deleted_node_ids == []

    # Original leaves should still exist unchanged
    left_leaf = store.nodes.get("left")
    assert left_leaf is not None
    assert left_leaf.text == "AAA"

    tail_leaf = store.nodes.get("tail")
    assert tail_leaf is not None
    assert tail_leaf.text == "BBB"
    # tail's following_neighbor should now point to the new leaf
    assert tail_leaf.following_neighbor_id == outcome.new_leaf_ids[0]

    # New leaf should link back to tail
    new_leaf = store.nodes.get(outcome.new_leaf_ids[0])
    assert new_leaf is not None
    assert new_leaf.preceding_neighbor_id == "tail"
    # New leaf starts where tail ended
    assert new_leaf.span_start == 6


@pytest.mark.asyncio
async def test_append_truncates_large_units_with_warning(
    sqlite_backend: SQLiteStorageBackend, caplog: pytest.LogCaptureFixture
) -> None:
    """When target_chunk_tokens is None, units > 50k chars are truncated with warning."""
    config = IndexConfig.load(target_chunk_tokens=None)
    store = _create_document(sqlite_backend, "doc-truncate")
    executor = AppendExecutor(config, StubEmbedder())

    # Create a unit larger than 50k characters
    large_text = "A" * 60000

    await executor.append(
        store=store,
        document_id="doc-truncate",
        new_text=large_text,
    )

    # Should log a warning
    assert any(
        "truncating" in record.message.lower() and "50000" in record.message
        for record in caplog.records
    )

    # Should create exactly one leaf (client-managed chunking)
    leaves = store.nodes.get_leaves()
    assert len(leaves) == 1

    # Leaf should contain exactly 50k characters
    leaf = leaves[0]
    assert len(leaf.text) == 50000
    assert leaf.text == "A" * 50000


@pytest.mark.asyncio
async def test_append_empty_string_creates_leaf_when_none(
    sqlite_backend: SQLiteStorageBackend,
) -> None:
    """When target_chunk_tokens is None, empty strings create a leaf."""
    config = IndexConfig.load(target_chunk_tokens=None)
    store = _create_document(sqlite_backend, "doc-empty")
    executor = AppendExecutor(config, StubEmbedder())

    outcome = await executor.append(
        store=store,
        document_id="doc-empty",
        new_text="",
    )

    # Should create exactly one leaf with empty text
    leaves = store.nodes.get_leaves()
    assert len(leaves) == 1
    assert len(outcome.new_leaf_ids) == 1

    leaf = leaves[0]
    assert leaf.text == ""
    assert leaf.span_start == 0
    assert leaf.span_end == 0


@pytest.mark.asyncio
async def test_append_batch_truncates_large_units(
    sqlite_backend: SQLiteStorageBackend, caplog: pytest.LogCaptureFixture
) -> None:
    """When target_chunk_tokens is None, units > 50k chars are truncated with warning."""
    config = IndexConfig.load(target_chunk_tokens=None)
    store = _create_document(sqlite_backend, "doc-batch-truncate")
    executor = AppendExecutor(config, StubEmbedder())

    # Create batch with one large unit and one normal unit
    large_unit = "B" * 55000
    normal_unit = "C" * 100

    await executor.append_batch(
        store=store,
        document_id="doc-batch-truncate",
        units=[large_unit, normal_unit],
    )

    # Should log a warning for the large unit
    assert any(
        "truncating" in record.message.lower() and "50000" in record.message
        for record in caplog.records
    )

    # Should create exactly 2 leaves (client-managed chunking)
    leaves = store.nodes.get_leaves()
    assert len(leaves) == 2

    # First leaf should be truncated to 50k
    assert len(leaves[0].text) == 50000
    assert leaves[0].text == "B" * 50000

    # Second leaf should be unchanged
    assert len(leaves[1].text) == 100
    assert leaves[1].text == "C" * 100


@pytest.mark.asyncio
async def test_append_batch_preserves_empty_units_when_none(
    sqlite_backend: SQLiteStorageBackend,
) -> None:
    """When target_chunk_tokens is None, empty and whitespace units create leaves."""
    config = IndexConfig.load(target_chunk_tokens=None)
    store = _create_document(sqlite_backend, "doc-batch-empty")
    executor = AppendExecutor(config, StubEmbedder())

    await executor.append_batch(
        store=store,
        document_id="doc-batch-empty",
        units=["", "  ", "foo"],
    )

    # Should create exactly 3 leaves
    leaves = store.nodes.get_leaves()
    assert len(leaves) == 3

    # Verify each leaf preserves the original unit
    assert leaves[0].text == ""
    assert leaves[1].text == "  "
    assert leaves[2].text == "foo"


@pytest.mark.asyncio
async def test_append_batch_preserves_atomic_units(
    sqlite_backend: SQLiteStorageBackend,
) -> None:
    """When target_chunk_tokens is None, each unit becomes exactly one leaf."""
    config = IndexConfig.load(target_chunk_tokens=None)
    store = _create_document(sqlite_backend, "doc-batch-atomic")
    executor = AppendExecutor(config, StubEmbedder())

    units = ["Turn A", "Turn B", "Turn C"]

    outcome = await executor.append_batch(
        store=store,
        document_id="doc-batch-atomic",
        units=units,
    )

    # Should create exactly N leaves for N units
    leaves = store.nodes.get_leaves()
    assert len(leaves) == 3
    assert len(outcome.new_leaf_ids) == 3

    # Each leaf should contain the full unit text (no splitting)
    assert leaves[0].text == "Turn A"
    assert leaves[1].text == "Turn B"
    assert leaves[2].text == "Turn C"

    # Verify span continuity
    assert leaves[0].span_start == 0
    assert leaves[0].span_end == len("Turn A")
    assert leaves[1].span_start == len("Turn A")
    assert leaves[1].span_end == len("Turn A") + len("Turn B")
    assert leaves[2].span_start == len("Turn A") + len("Turn B")
    assert leaves[2].span_end == len("Turn A") + len("Turn B") + len("Turn C")
