"""Test helpers providing in-memory VectorIndex implementations."""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray

from ragzoom.vector_api import Vector, ensure_normalized

_MetaValue = str | int | float | bool | None


def _coerce_meta(value: object, default: _MetaValue = None) -> _MetaValue:
    if value is None:
        return default
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, np.generic):
        coerced = value.item()
        if isinstance(coerced, str | int | float | bool):
            return coerced
    return default


class RecordingVectorIndex:
    """Minimal in-memory VectorIndex that records upserts for assertions."""

    def __init__(self) -> None:
        self._vectors: dict[str, Vector] = {}

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        where: dict[str, _MetaValue] | None = None,
    ) -> list[Vector]:
        if k <= 0:
            return []
        normalized_query = ensure_normalized(query_embedding)
        candidates = [
            vector
            for vector in self._vectors.values()
            if self._matches_where(vector, where)
        ]
        if not candidates:
            return []
        scored = sorted(
            candidates,
            key=lambda candidate: float(np.dot(candidate.vec, normalized_query)),
            reverse=True,
        )
        return scored[:k]

    def get_vectors(self, ids: list[str]) -> list[Vector]:
        return [self._vectors[node_id] for node_id in ids if node_id in self._vectors]

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        for node_id, embedding, metadata in items:
            normalized = ensure_normalized(embedding)
            coerced_meta: dict[str, _MetaValue] = {}
            for key, value in metadata.items():
                coerced_meta[key] = _coerce_meta(value)

            doc_id = metadata.get("document_id")
            coerced_meta["document_id"] = (
                str(doc_id) if isinstance(doc_id, str) else _coerce_meta(doc_id, "")
            )

            coerced_meta["span_start"] = int(
                _coerce_meta(metadata.get("span_start"), 0) or 0
            )
            coerced_meta["span_end"] = int(
                _coerce_meta(metadata.get("span_end"), 0) or 0
            )

            parent = metadata.get("parent_id")
            if parent is None:
                coerced_meta["parent_id"] = None
            else:
                parent_str = _coerce_meta(parent)
                coerced_meta["parent_id"] = (
                    None if parent_str is None else str(parent_str)
                )

            coerced_meta["is_leaf"] = int(_coerce_meta(metadata.get("is_leaf"), 0) or 0)

            model_raw = metadata.get("model_id", "")
            if isinstance(model_raw, np.generic):
                model_raw = model_raw.item()
            model_id_str = str(model_raw)
            coerced_meta["model_id"] = model_id_str

            self._vectors[node_id] = Vector(
                id=node_id,
                vec=normalized,
                meta=coerced_meta,
                model_id=model_id_str,
                dim=int(normalized.shape[0]),
            )

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        removed = 0
        if filter and "document_id" in filter:
            target = str(filter["document_id"])
            to_delete = [
                node_id
                for node_id, vector in self._vectors.items()
                if vector.meta.get("document_id") == target
            ]
            for node_id in to_delete:
                del self._vectors[node_id]
            return len(to_delete)

        if ids:
            for node_id in ids:
                if node_id in self._vectors:
                    del self._vectors[node_id]
                    removed += 1
            return removed

        removed = len(self._vectors)
        self._vectors.clear()
        return removed

    def _matches_where(
        self,
        vector: Vector,
        where: dict[str, _MetaValue] | None,
    ) -> bool:
        if not where:
            return True
        for key, value in where.items():
            if value is None:
                continue
            if vector.meta.get(key) != value:
                return False
        return True

    def __len__(self) -> int:
        return len(self._vectors)


__all__: Final = ["RecordingVectorIndex"]
