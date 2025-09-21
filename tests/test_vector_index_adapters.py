"""Unit tests for vector index adapters."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.backends.vector_index_python import PythonVectorIndexAdapter
from ragzoom.vector_api import Vector

chromadb: ModuleType | None
Settings: type | None

try:  # pragma: no cover - optional dependency
    import chromadb as _chromadb_module
except Exception:
    chromadb = None
    Settings = None
else:
    chromadb = _chromadb_module
    try:
        from chromadb.config import Settings as _ChromaSettings
    except Exception:
        Settings = None
    else:
        Settings = _ChromaSettings


def _install_fake_chroma_client(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeCollection:
        def __init__(self) -> None:
            self._store: dict[str, tuple[list[float], dict[str, object]]] = {}
            self.batch_sizes: list[int] = []

        def upsert(
            self,
            ids: list[str],
            embeddings: list[Iterable[float]],
            metadatas: list[dict[str, object]],
        ) -> None:
            self.batch_sizes.append(len(ids))
            for node_id, emb, meta in zip(ids, embeddings, metadatas, strict=False):
                self._store[node_id] = ([float(x) for x in emb], dict(meta))

        def get(
            self,
            ids: list[str] | None = None,
            *,
            include: list[str] | None = None,
            where: dict[str, object] | None = None,
        ) -> dict[str, object]:
            if ids:
                present_ids = [node_id for node_id in ids if node_id in self._store]
                embeddings = [self._store[node_id][0] for node_id in present_ids]
                metadatas = [self._store[node_id][1] for node_id in present_ids]
                return {
                    "ids": present_ids,
                    "embeddings": embeddings if include else [],
                    "metadatas": metadatas if include else [],
                }
            if where:
                eq_filters = {
                    key: value.get("$eq") if isinstance(value, dict) else value
                    for key, value in where.items()
                }
                matching = [
                    node_id
                    for node_id, (_, meta) in self._store.items()
                    if all(meta.get(k) == v for k, v in eq_filters.items())
                ]
                return {"ids": matching}
            return {"ids": list(self._store)}

        def query(
            self,
            query_embeddings: list[Iterable[float]],
            n_results: int,
            include: list[str],
            where: dict[str, object] | None = None,
        ) -> dict[str, object]:
            if where:
                eq_filters = {
                    key: value.get("$eq") if isinstance(value, dict) else value
                    for key, value in where.items()
                }
                candidate_ids = [
                    node_id
                    for node_id, (_, meta) in self._store.items()
                    if all(meta.get(k) == v for k, v in eq_filters.items())
                ]
            else:
                candidate_ids = list(self._store.keys())
            candidate_ids = candidate_ids[:n_results]
            metadatas = [self._store[node_id][1] for node_id in candidate_ids]
            distances = [[0.0 for _ in candidate_ids]] if candidate_ids else [[]]
            return {
                "ids": [candidate_ids],
                "distances": distances,
                "metadatas": [metadatas],
            }

        def delete(
            self,
            ids: list[str] | None = None,
            *,
            where: dict[str, object] | None = None,
        ) -> int:
            removed = 0
            if ids:
                for node_id in ids:
                    if self._store.pop(node_id, None) is not None:
                        removed += 1
                return removed
            if where:
                eq_filters = {
                    key: value.get("$eq") if isinstance(value, dict) else value
                    for key, value in where.items()
                }
                to_remove = [
                    node_id
                    for node_id, (_, meta) in list(self._store.items())
                    if all(meta.get(k) == v for k, v in eq_filters.items())
                ]
                for node_id in to_remove:
                    self._store.pop(node_id, None)
                removed = len(to_remove)
            return removed

    class _FakeClient:
        def __init__(self, path: str, settings: object | None = None) -> None:
            self._collection = _FakeCollection()

        def get_or_create_collection(
            self, name: str, metadata: dict[str, object] | None = None
        ) -> _FakeCollection:
            return self._collection

    monkeypatch.setattr(
        "ragzoom.backends.chroma_vector_index.chromadb.PersistentClient",
        _FakeClient,
    )


@pytest.fixture
def sample_items() -> (
    list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]]
):
    """Create a small deterministic set of vectors and metadata."""
    meta = {
        "span_start": 0,
        "span_end": 10,
        "parent_id": "",
        "document_id": "doc-1",
        "is_leaf": 1,
        "doc_version": 1,
    }
    return [
        ("n1", [1.0, 0.0], dict(meta)),
        ("n2", [0.0, 1.0], dict(meta)),
    ]


def _vector_ids(vectors: Iterable[Vector]) -> list[str]:
    return [v.id for v in vectors]


def test_python_vector_index_adapter_round_trip(
    tmp_path: Path,
    sample_items: list[
        tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
    ],
) -> None:
    idx_dir = tmp_path / "py-idx"
    idx_dir.mkdir()
    adapter = PythonVectorIndexAdapter(str(idx_dir), "test-model")
    adapter.upsert(sample_items)

    vectors = adapter.get_vectors(["n1", "n2"])
    assert _vector_ids(vectors) == ["n1", "n2"]
    assert all(v.dim == 2 for v in vectors)
    assert all(isinstance(v.vec, np.ndarray) for v in vectors)

    # Query nearest neighbour for basis vector
    results = adapter.search_similar([1.0, 0.0], k=2)
    assert results[0].id == "n1"
    assert results[0].meta["document_id"] == "doc-1"
    assert results[0].meta["doc_version"] == 1

    # Delete by id
    deleted = adapter.delete(ids=["n1"])
    assert deleted == 1
    with pytest.raises(KeyError):
        adapter.get_vectors(["n1"])

    # Delete remaining by document filter
    deleted_by_filter = adapter.delete(filter={"document_id": "doc-1"})
    assert deleted_by_filter == 1
    assert adapter.search_similar([0.0, 1.0], k=1) == []


def test_chroma_vector_index_adapter_round_trip(
    tmp_path: Path,
    sample_items: list[
        tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ragzoom.backends.vector_index_chroma import ChromaVectorIndexAdapter

    if chromadb is None or Settings is None:
        pytest.skip("chromadb is required for adapter tests")

    _install_fake_chroma_client(monkeypatch)

    chroma_dir = tmp_path / "chroma-idx"
    chroma_dir.mkdir()
    adapter = ChromaVectorIndexAdapter(str(chroma_dir), "test-model")
    adapter.upsert(sample_items)

    vectors = adapter.get_vectors(["n1", "n2"])
    assert _vector_ids(vectors) == ["n1", "n2"]
    assert all(v.dim == 2 for v in vectors)

    results = adapter.search_similar([1.0, 0.0], k=2)
    assert results and results[0].id == "n1"
    assert results[0].meta["doc_version"] == 1

    deleted = adapter.delete(ids=["n1"])
    assert deleted == 1
    with pytest.raises(KeyError):
        adapter.get_vectors(["n1"])

    # Remaining vector should be removed by document filter delete
    deleted_by_filter = adapter.delete(filter={"document_id": "doc-1"})
    assert deleted_by_filter >= 1
    assert adapter.search_similar([0.0, 1.0], k=1) == []


def test_chroma_vector_index_adapter_chunks_large_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ragzoom.backends.vector_index_chroma import ChromaVectorIndexAdapter

    if chromadb is None or Settings is None:
        pytest.skip("chromadb is required for adapter tests")

    _install_fake_chroma_client(monkeypatch)

    idx_dir = tmp_path / "chroma-large"
    idx_dir.mkdir()
    adapter = ChromaVectorIndexAdapter(str(idx_dir), "test-model")

    meta = {
        "span_start": 0,
        "span_end": 1,
        "parent_id": "",
        "document_id": "doc",
        "is_leaf": 1,
        "doc_version": 1,
    }
    large_items: list[
        tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
    ] = [(f"node-{i}", [float(i), float(i + 1)], dict(meta)) for i in range(6000)]

    adapter.upsert(large_items)

    collection = getattr(adapter._under, "_collection")
    sizes = getattr(collection, "batch_sizes")
    assert sum(sizes) == len(large_items)
    assert len(sizes) > 1
    assert all(size <= adapter._max_batch_size for size in sizes)


def test_chroma_adapter_combines_multiple_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ragzoom.backends.vector_index_chroma import ChromaVectorIndexAdapter

    if chromadb is None:
        pytest.skip("chromadb is required for adapter tests")

    _install_fake_chroma_client(monkeypatch)

    captured: dict[str, object] = {}

    idx_dir = tmp_path / f"chroma-filter-{uuid4().hex}"
    idx_dir.mkdir(parents=True, exist_ok=True)
    adapter = ChromaVectorIndexAdapter(str(idx_dir), "test-model")

    def fake_query(*args: object, **kwargs: object) -> dict[str, list[list[object]]]:
        captured["where"] = kwargs.get("where")
        return {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    monkeypatch.setattr(adapter._under._collection, "query", fake_query)

    adapter.search_similar(
        [1.0, 0.0],
        1,
        {"document_id": "doc-1", "doc_version": 1},
    )

    where = captured.get("where")
    assert isinstance(where, dict)
    assert "$and" in where
