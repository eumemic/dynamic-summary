"""VectorIndex v2 wrapper over PythonVectorIndex (in-memory).

This adapts the existing PythonVectorIndex to the new VectorIndex v2 protocol
by returning canonical Vector objects with normalized float32 arrays.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from ragzoom.backends.python_vector_index import PythonVectorIndex
from ragzoom.backends.vector_common import (
    NormalizedUpsertItem,
    VectorUpsertItem,
    normalize_metadata_from_dict,
    normalize_metadata_from_object,
    normalize_upsert_items,
)
from ragzoom.contracts.vector_filter import (
    DocumentIdFilter,
    SpanEndLtFilter,
    VectorFilter,
)
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.exceptions import UnsupportedFilterError
from ragzoom.vector_api import MetaDict, Vector


def _filters_to_where(
    filters: Sequence[VectorFilter] | None,
) -> dict[str, str | int | float | bool | None] | None:
    """Convert typed filters to legacy dict format for underlying implementation."""
    if not filters:
        return None
    where: dict[str, str | int | float | bool | None] = {}
    for f in filters:
        match f:
            case DocumentIdFilter(value=doc_id):
                where["document_id"] = doc_id
            case SpanEndLtFilter(threshold=threshold):
                where["span_end_lt"] = threshold
            case _:
                raise UnsupportedFilterError(type(f).__name__, "PythonVectorIndex")
    return where if where else None


class PythonVectorIndexAdapter(VectorIndex):
    def __init__(self, persist_dir: str | None, model_id: str) -> None:
        # Intentionally ignore persist_dir: PythonVectorIndex is in-memory only and
        # never persists to disk. This adapter is intended for tests/dev only.
        self._idx = PythonVectorIndex(None)
        self._model_id = model_id

    # --- VectorIndex v2 ---
    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        filters: Sequence[VectorFilter] | None = None,
    ) -> list[Vector]:
        # Convert typed filters to legacy dict format for underlying implementation
        where = _filters_to_where(filters)
        # Use underlying search for ordering and scores; construct Vectors from internal arrays
        results = self._idx.search_similar(query_embedding, k, where)
        out: list[Vector] = []
        for node_id, _score, meta in results:
            vec = self._vector_for_id(node_id)
            out.append(self._wrap(node_id, vec, _as_meta(meta)))
        return out

    def get_vectors(self, ids: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for node_id in ids:
            try:
                vec = self._vector_for_id(node_id)
                meta = self._meta_for_id(node_id)
                out.append(self._wrap(node_id, vec, meta))
            except KeyError:
                # Skip missing vectors - caller handles partial results
                continue
        return out

    def upsert(self, items: Sequence[VectorUpsertItem]) -> None:
        normalized: list[NormalizedUpsertItem] = normalize_upsert_items(items)
        if not normalized:
            return

        # In-memory only; do not persist to disk
        payload: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = [(node_id, vector, meta) for node_id, vector, meta in normalized]
        self._idx.upsert(payload, persist=False)

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        # Simple implementation: recreate without deleted IDs or those matching filter
        if ids:
            remaining: list[
                tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
            ] = []
            # Snapshot all current ids from internal mapping
            all_ids = list(self._idx._ids)
            for i in all_ids:
                if i in ids:
                    continue
                # Reconstruct tuple for upsert
                v = self._vector_for_id(i)
                meta = self._meta_for_id(i)
                remaining.append(
                    (i, list(map(float, v.tolist())), dict(meta))
                )  # upcast meta
            # Reset internal state by reinitializing index
            self._idx = PythonVectorIndex(getattr(self._idx, "_persist_dir", None))
            self._idx.upsert(remaining)
            return len(ids)
        if filter and "document_id" in filter:
            doc = (
                str(filter["document_id"]) if filter["document_id"] is not None else ""
            )
            remaining2: list[
                tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
            ] = []
            deleted = 0
            all_ids = list(self._idx._ids)
            for i in all_ids:
                meta = self._meta_for_id(i)
                same_doc = str(meta.get("document_id", "")) == doc
                if same_doc:
                    deleted += 1
                    continue
                v = self._vector_for_id(i)
                remaining2.append((i, list(map(float, v.tolist())), dict(meta)))
            self._idx = PythonVectorIndex(getattr(self._idx, "_persist_dir", None))
            if remaining2:
                self._idx.upsert(remaining2)
            return deleted
        return 0

    # --- internals ---
    def _vector_for_id(self, node_id: str) -> NDArray[np.float32]:
        id_to_row = self._idx._id_to_row
        row = id_to_row.get(node_id)
        if row is None:
            raise KeyError(f"Vector not found for id {node_id}")
        mat = self._idx._vectors
        if mat is None:
            raise RuntimeError("Vector matrix is empty")
        return np.asarray(mat[int(row), :], dtype=np.float32)

    def _meta_for_id(self, node_id: str) -> MetaDict:
        meta_obj = self._idx._meta.get(node_id)
        if meta_obj is None:
            return {}
        if isinstance(meta_obj, dict):
            return normalize_metadata_from_dict(meta_obj)
        if hasattr(meta_obj, "span_start"):
            return normalize_metadata_from_object(meta_obj)
        return {}

    def _wrap(self, node_id: str, vec: NDArray[np.float32], meta: MetaDict) -> Vector:
        return Vector(
            id=node_id,
            vec=vec,
            meta=meta,
            model_id=self._model_id,
            dim=int(vec.shape[0]),
        )


def _as_meta(meta: object) -> MetaDict:
    if isinstance(meta, dict):
        return normalize_metadata_from_dict(meta)
    if hasattr(meta, "span_start"):
        return normalize_metadata_from_object(meta)
    return {}
