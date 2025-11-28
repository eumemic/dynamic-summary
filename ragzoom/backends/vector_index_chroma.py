"""VectorIndex v2 wrapper over ChromaVectorIndex.

Uses Chroma's persistent collection to fetch embeddings and wrap them into
canonical Vector objects for core consumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.backends.chroma_vector_index import ChromaVectorIndex
from ragzoom.backends.vector_common import (
    NormalizedUpsertItem,
    VectorUpsertItem,
    normalize_metadata_from_dict,
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


def _filters_to_chroma_where(
    filters: Sequence[VectorFilter] | None,
) -> dict[str, object] | None:
    """Convert typed filters to Chroma where clause format."""
    if not filters:
        return None
    clauses: list[dict[str, object]] = []
    for f in filters:
        match f:
            case DocumentIdFilter(value=doc_id):
                clauses.append({"document_id": {"$eq": doc_id}})
            case SpanEndLtFilter(threshold=threshold):
                clauses.append({"span_end": {"$lt": threshold}})
            case _:
                raise UnsupportedFilterError(type(f).__name__, "ChromaVectorIndex")
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _normalize_where(
    where: dict[str, str | int | float | bool | None],
) -> dict[str, object]:
    clauses: list[dict[str, object]] = []
    for key, val in where.items():
        if val is None:
            continue
        if isinstance(val, dict):
            clauses.append({key: val})
        elif isinstance(val, str | int | float | bool):
            clauses.append({key: {"$eq": val}})
    if not clauses:
        return {}
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaVectorIndexAdapter(VectorIndex):
    def __init__(self, persist_dir: str, model_id: str) -> None:
        self._under = ChromaVectorIndex(persist_dir)
        self._model_id = model_id
        # Chroma rejects batches larger than 5461; stay comfortably under the ceiling.
        self._max_batch_size = 5000

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        filters: Sequence[VectorFilter] | None = None,
    ) -> list[Vector]:
        include: list[
            Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]
        ] = [
            "metadatas",
            "distances",
        ]
        where_param: Mapping[str, object] | None = _filters_to_chroma_where(filters)

        class _QueryResult(TypedDict, total=False):
            ids: Sequence[Sequence[str]]
            distances: Sequence[Sequence[float]]
            metadatas: Sequence[Sequence[Mapping[str, str | int | float | bool | None]]]

        res = cast(
            _QueryResult,
            self._under._collection.query(
                query_embeddings=[
                    cast(Sequence[float], list(map(float, query_embedding)))
                ],
                n_results=k,
                include=include,
                where=where_param,  # type: ignore[arg-type]
            ),
        )

        ids_nested = res.get("ids")
        ids = list(ids_nested[0]) if ids_nested else []
        metas_nested = res.get("metadatas")
        metas: list[dict[str, object]] = (
            [cast(dict[str, object], dict(m)) for m in metas_nested[0]]
            if metas_nested and len(metas_nested) > 0
            else []
        )

        vecs = self.get_vectors(ids)
        meta_map: dict[str, dict[str, object]] = {}
        for idx, node_id in enumerate(ids):
            meta_map[node_id] = metas[idx] if idx < len(metas) else {}
        out: list[Vector] = []
        for idx, v in enumerate(vecs):
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

        class _GetResult(TypedDict, total=False):
            ids: Sequence[str]
            embeddings: Sequence[Sequence[float]]
            metadatas: Sequence[Mapping[str, object]]

        read = cast(
            _GetResult,
            self._under._collection.get(ids=ids, include=["embeddings", "metadatas"]),
        )

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

    def upsert(self, items: Sequence[VectorUpsertItem]) -> None:
        if not items:
            return

        for start in range(0, len(items), self._max_batch_size):
            chunk = items[start : start + self._max_batch_size]
            normalized: list[NormalizedUpsertItem] = normalize_upsert_items(chunk)
            payload = [
                (
                    node_id,
                    vector,
                    cast(
                        dict[str, str | int | float | bool | None],
                        {k: v for k, v in meta.items()},
                    ),
                )
                for node_id, vector, meta in normalized
            ]
            self._under.upsert(payload)

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        if ids:
            self._under._collection.delete(ids=ids)
            return len(ids)
        if filter:
            # Normalize simple equality filters to Chroma operator form
            where_param: dict[str, object] = {}
            for k, v in filter.items():
                if v is None:
                    continue
                where_param[k] = v if isinstance(v, dict) else {"$eq": v}

            # Fetch matching ids to report a count
            from typing import TypedDict
            from typing import cast as _cast

            class _GetIds(TypedDict, total=False):
                ids: list[str]

            read = _cast(
                _GetIds,
                self._under._collection.get(where=where_param),  # type: ignore[arg-type]
            )
            matched_ids = list(read.get("ids", []))

            self._under._collection.delete(where=where_param)  # type: ignore[arg-type]
            return len(matched_ids)
        return 0


def _as_meta(meta: dict[str, object]) -> MetaDict:
    return normalize_metadata_from_dict(meta)
