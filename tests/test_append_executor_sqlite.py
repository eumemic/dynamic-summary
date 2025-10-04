import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.server.append_executor import AppendExecutor, EmbeddingProvider
from ragzoom.vector_api import Vector


class StubEmbedder(EmbeddingProvider):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]


class FakeVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self.vectors: dict[str, Vector] = {}

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[Vector]:  # pragma: no cover - not used
        raise NotImplementedError

    def get_vectors(self, ids: list[str]) -> list[Vector]:
        found: list[Vector] = []
        for node_id in ids:
            vec = self.vectors.get(node_id)
            if vec is not None:
                found.append(vec)
        return found

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        for node_id, embedding, meta in items:
            arr = np.asarray(embedding, dtype=np.float32)
            norm = float(np.linalg.norm(arr))
            if norm == 0.0:
                raise ValueError("Cannot upsert zero vector")
            normalized = arr / norm
            normalized_meta: dict[str, str | int | float | bool | None] = {}
            for key, value in meta.items():
                if isinstance(value, str | int | float | bool) or value is None:
                    normalized_meta[str(key)] = value
                else:
                    normalized_meta[str(key)] = str(value)
            self.vectors[node_id] = Vector(
                id=node_id,
                vec=normalized,
                meta=normalized_meta,
                model_id="test",
                dim=normalized.shape[0],
            )

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        if not ids:
            return 0
        deleted = 0
        for node_id in ids:
            if node_id in self.vectors:
                del self.vectors[node_id]
                deleted += 1
        return deleted


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
    vector_index = FakeVectorIndex()
    executor = AppendExecutor(index_config, StubEmbedder())

    outcome = await executor.append(
        store=store,
        vector_index=vector_index,
        document_id="doc-1",
        new_text="The quick brown fox",
    )

    leaves = store.nodes.get_leaves()
    assert len(leaves) == len(outcome.new_leaf_ids)
    assert outcome.total_leaves == len(leaves)
    assert all(node.parent_id is None for node in leaves)
    assert set(outcome.new_leaf_ids) == {leaf.id for leaf in leaves}
    assert vector_index.vectors.keys() == set(outcome.new_leaf_ids)


@pytest.mark.asyncio
async def test_append_replaces_rightmost_path_and_updates_neighbors(
    sqlite_backend: SQLiteStorageBackend, index_config: IndexConfig
) -> None:
    store = _create_document(sqlite_backend, "doc-2")
    vector_index = FakeVectorIndex()
    executor = AppendExecutor(index_config, StubEmbedder())

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
                "following_neighbor_id": "tail",
            },
            {
                "node_id": "tail",
                "text": "BBB",
                "span_start": 3,
                "span_end": 6,
                "parent_id": "parent",
                "left_child_id": None,
                "right_child_id": None,
                "document_id": "doc-2",
                "token_count": 3,
                "height": 0,
                "preceding_neighbor_id": "left",
            },
            {
                "node_id": "parent",
                "text": "parent",
                "span_start": 3,
                "span_end": 6,
                "parent_id": None,
                "left_child_id": "tail",
                "right_child_id": None,
                "document_id": "doc-2",
                "token_count": 6,
                "height": 1,
            },
        ]
    )

    vector_index.upsert(
        [
            ("left", [1.0, 0.0, 0.0, 0.0], {"document_id": "doc-2"}),
            ("tail", [0.0, 1.0, 0.0, 0.0], {"document_id": "doc-2"}),
            ("parent", [0.0, 0.0, 1.0, 0.0], {"document_id": "doc-2"}),
        ]
    )

    outcome = await executor.append(
        store=store,
        vector_index=vector_index,
        document_id="doc-2",
        new_text="CCC",
    )

    assert "parent" in outcome.deleted_node_ids
    leaves = {leaf.id: leaf for leaf in store.nodes.get_leaves()}
    assert "parent" not in leaves
    tail_leaf = leaves[outcome.new_leaf_ids[0]]
    assert tail_leaf.preceding_neighbor_id == "left"
    assert tail_leaf.parent_id is None
    left_leaf = store.nodes.get("left")
    assert left_leaf is not None
    assert left_leaf.following_neighbor_id == tail_leaf.id
    assert vector_index.vectors.keys() == {"left", *outcome.new_leaf_ids}
