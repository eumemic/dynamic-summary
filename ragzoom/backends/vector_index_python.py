"""VectorIndex v2 wrapper over PythonVectorIndex (in-memory).

This adapts the existing PythonVectorIndex to the new VectorIndex v2 protocol
by returning canonical Vector objects with normalized float32 arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ragzoom.backends.python_vector_index import PythonVectorIndex
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.vector_api import MetaDict, Vector


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
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[Vector]:
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
            vec = self._vector_for_id(node_id)
            meta = self._meta_for_id(node_id)
            out.append(self._wrap(node_id, vec, meta))
        return out

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        # In-memory only; do not persist to disk
        self._idx.upsert(items, persist=False)

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        # Simple implementation: recreate without deleted IDs if provided; otherwise no-op
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
        if hasattr(meta_obj, "span_start"):
            return {
                "span_start": int(getattr(meta_obj, "span_start")),
                "span_end": int(getattr(meta_obj, "span_end")),
                "parent_id": str(getattr(meta_obj, "parent_id")),
                "document_id": str(getattr(meta_obj, "document_id")),
                "is_leaf": int(getattr(meta_obj, "is_leaf")),
            }
        if isinstance(meta_obj, dict):
            return {
                "span_start": int(meta_obj.get("span_start", 0)),
                "span_end": int(meta_obj.get("span_end", 0)),
                "parent_id": str(meta_obj.get("parent_id", "")),
                "document_id": str(meta_obj.get("document_id", "")),
                "is_leaf": int(meta_obj.get("is_leaf", 0)),
            }
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
    if hasattr(meta, "span_start"):
        return {
            "span_start": int(getattr(meta, "span_start")),
            "span_end": int(getattr(meta, "span_end")),
            "parent_id": str(getattr(meta, "parent_id")),
            "document_id": str(getattr(meta, "document_id")),
            "is_leaf": int(getattr(meta, "is_leaf")),
        }
    if isinstance(meta, dict):
        return {
            "span_start": int(meta.get("span_start", 0)),
            "span_end": int(meta.get("span_end", 0)),
            "parent_id": str(meta.get("parent_id", "")),
            "document_id": str(meta.get("document_id", "")),
            "is_leaf": int(meta.get("is_leaf", 0)),
        }
    return {}
