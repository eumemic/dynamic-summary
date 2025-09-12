"""VectorIndex v2 wrapper over ChromaVectorIndex.

Uses Chroma's persistent collection to fetch embeddings and wrap them into
canonical Vector objects for core consumption.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.backends.chroma_vector_index import ChromaVectorIndex
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.vector_api import MetaDict, Vector


class ChromaVectorIndexAdapter(VectorIndex):
    def __init__(self, persist_dir: str, model_id: str) -> None:
        self._under = ChromaVectorIndex(persist_dir)
        self._model_id = model_id

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[Vector]:
        # Filter where to remove None values per adapter's expected type
        where_clean: dict[str, str | int | float | bool] | None = None
        if where:
            tmp: dict[str, str | int | float | bool] = {}
            for key, val in where.items():
                if val is None:
                    continue
                if isinstance(val, str | int | float | bool):
                    tmp[key] = val
            where_clean = tmp or None
        base_results = self._under.search_similar(query_embedding, k, where_clean)
        ids = [r[0] for r in base_results]
        vecs = self.get_vectors(ids)
        meta_map: dict[str, dict[str, object]] = {
            str(r[0]): dict(r[2]) for r in base_results
        }
        out: list[Vector] = []
        for v in vecs:
            out.append(
                Vector(
                    id=v.id,
                    vec=v.vec,
                    meta=_as_meta(meta_map.get(v.id, {})),
                    model_id=self._model_id,
                    dim=v.dim,
                )
            )
        return out

    def get_vectors(self, ids: list[str]) -> list[Vector]:
        if not ids:
            return []
        read = self._under._collection.get(ids=ids, include=["embeddings", "metadatas"])

        # Normalize response fields without relying on truthiness of numpy arrays
        raw_ids = read.get("ids")
        got_ids: list[str] = list(raw_ids) if raw_ids is not None else []

        raw_embeddings = read.get("embeddings")
        if raw_embeddings is None:
            emb_list: list[list[float]] = []
        else:
            # Accept iterable of sequences (list or numpy array); coerce to list[list[float]]
            emb_list = [list(map(float, e)) for e in raw_embeddings]

        raw_metas = read.get("metadatas")
        metas: list[dict[str, object]] = []
        if raw_metas is not None:
            for m in raw_metas:
                # Each m is Mapping[str, str|int|float|bool|None]; copy to plain dict[str, object]
                metas.append({k: v for k, v in dict(m).items()})

        by_id = {
            got_ids[i]: (emb_list[i], metas[i] if i < len(metas) else {})
            for i in range(len(got_ids))
        }
        out: list[Vector] = []
        for node_id in ids:
            tup = by_id.get(node_id)
            if tup is None:
                raise KeyError(f"Vector not found for id {node_id}")
            emb = np.asarray(tup[0], dtype=np.float32)
            out.append(
                Vector(
                    id=node_id,
                    vec=emb,
                    meta=_as_meta(tup[1]),
                    model_id=self._model_id,
                    dim=len(emb),
                )
            )
        return out

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        narrowed: list[
            tuple[
                str,
                list[float] | NDArray[np.float64],
                dict[str, str | int | float | bool | None],
            ]
        ] = [
            (
                i,
                e,
                cast(dict[str, str | int | float | bool | None], m),
            )
            for (i, e, m) in items
        ]
        self._under.upsert(narrowed)

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        if ids:
            self._under._collection.delete(ids=ids)
            return len(ids)
        return 0


def _as_meta(meta: dict[str, object]) -> MetaDict:
    def _to_int(x: object) -> int:
        if isinstance(x, bool):
            return 1 if x else 0
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return int(x)
        return 0

    def _to_str(x: object) -> str:
        if isinstance(x, str):
            return x
        return str(x) if x is not None else ""

    return {
        "span_start": _to_int(meta.get("span_start")),
        "span_end": _to_int(meta.get("span_end")),
        "parent_id": _to_str(meta.get("parent_id")),
        "document_id": _to_str(meta.get("document_id")),
        "is_leaf": _to_int(meta.get("is_leaf")),
    }
