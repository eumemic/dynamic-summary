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
