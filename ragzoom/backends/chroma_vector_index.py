"""Chroma-based vector index adapter.

This provides a minimal surface over Chroma for:
 - upsert: add/update vectors and metadata
 - search_similar: cosine similarity search with optional metadata filter

Requires the optional dependency: chromadb
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

try:
    import logging

    # Quiet chroma's telemetry loggers before importing the client
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
    logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.ERROR)
    import chromadb
    from chromadb.api.types import Metadatas
except Exception as e:  # pragma: no cover - optional dependency
    raise ImportError(
        "chromadb is not installed. Install with `pip install chromadb`."
    ) from e


@dataclass
class _Meta:
    span_start: int
    span_end: int
    parent_id: str
    document_id: str
    is_leaf: int


class ChromaVectorIndex:
    """Thin wrapper around ChromaDB persistent client/collection."""

    def __init__(self, persist_dir: str) -> None:
        # Use persistent client to keep data across runs
        try:
            settings = chromadb.config.Settings(anonymized_telemetry=False)
            self._client = chromadb.PersistentClient(
                path=persist_dir, settings=settings
            )
        except Exception:
            self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="ragzoom",
            metadata={"hnsw:space": "cosine"},
        )

    # API: list[tuple[id, score, meta]] with score being similarity
    # jscpd:ignore-start - structure intentionally mirrors python adapter
    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, str | int | float | bool] | None = None,
    ) -> list[tuple[str, float, dict[str, str | int | float | bool | None]]]:
        if n_results <= 0:
            return []
        emb = cast(list[float], list(map(float, query_embedding)))
        # Chroma supports: documents, embeddings, metadatas, distances, uris, data
        # 'ids' are always returned independently; do not include it.
        include: list[
            Literal["documents", "embeddings", "metadatas", "distances", "uris", "data"]
        ] = [  # noqa: E501
            "metadatas",
            "distances",
        ]
        where_param: Mapping[str, object] | None = None
        if where:
            # Normalize simple equality filters to Chroma operator form
            cw: dict[str, object] = {}
            for k, v in where.items():
                cw[k] = v if isinstance(v, dict) else {"$eq": v}
            where_param = cw
        res = self._collection.query(
            query_embeddings=[cast(Sequence[float], emb)],
            n_results=n_results,
            include=include,
            where=where_param,  # type: ignore[arg-type]
        )
        # Mypy types for chroma response are loose; cast progressively
        ids = (res.get("ids") or [[]])[0] if res else []
        dists = (res.get("distances") or [[]])[0] if res else []
        metas = cast(
            list[list[dict[str, str | int | float | bool | None]]],
            res.get("metadatas", [[]]) if res else [[]],
        )[0]
        out: list[tuple[str, float, dict[str, str | int | float | bool | None]]] = []
        for i, node_id in enumerate(ids):
            # Convert distance to similarity ~ 1/(1+d)
            dist = float(dists[i]) if i < len(dists) else 0.0
            sim = 1.0 / (1.0 + dist)
            meta = metas[i] if i < len(metas) and isinstance(metas[i], dict) else {}
            out.append((node_id, sim, meta))
        return out

    # jscpd:ignore-end

    def upsert(
        self,
        items: list[
            tuple[
                str,
                list[float] | NDArray[np.float64],
                dict[str, str | int | float | bool | None],
            ]
        ],
    ) -> None:
        if not items:
            return
        ids: list[str] = []
        embeddings: list[Sequence[float]] = []
        metadatas: Metadatas = []
        for node_id, emb, meta in items:
            ids.append(str(node_id))
            embeddings.append([float(x) for x in cast(list[float], emb)])
            # Store only necessary metadata fields (others are allowed)
            m: dict[str, str | int | float | bool | None] = {
                "span_start": meta.get("span_start"),
                "span_end": meta.get("span_end"),
                "parent_id": meta.get("parent_id"),
                "document_id": meta.get("document_id"),
                "is_leaf": meta.get("is_leaf"),
            }
            # Also pass-through any additional keys to support future use
            for k, v in meta.items():
                if k not in m:
                    m[k] = v
            metadatas.append(m)
        # upsert is add+update
        self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)
