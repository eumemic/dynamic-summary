"""Unit tests for vector index adapters."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.backends.vector_index_python import PythonVectorIndexAdapter
from ragzoom.vector_api import Vector


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


@pytest.mark.slow_threshold(5.0)
def test_chroma_vector_index_adapter_round_trip(
    tmp_path: Path,
    sample_items: list[
        tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
    ],
) -> None:
    pytest.importorskip("chromadb", reason="chromadb is required for adapter tests")
    from ragzoom.backends.vector_index_chroma import ChromaVectorIndexAdapter

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


def test_chroma_adapter_combines_multiple_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    try:
        __import__("chromadb")
    except Exception:
        pytest.skip("chromadb is required for adapter tests")
    from ragzoom.backends.vector_index_chroma import ChromaVectorIndexAdapter

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
