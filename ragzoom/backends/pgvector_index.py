"""VectorIndex adapter over the existing SearchService (pgvector).

This allows retrieval to depend on a small, stable VectorIndex protocol while
reusing the current optimized search implementation backed by PostgreSQL +
pgvector.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ragzoom.interfaces.vector_index import VectorIndex, VectorSearchMetadata
from ragzoom.services.search_service import NodeMetadataDict, SearchService
from ragzoom.storage.database_manager import DatabaseManager


class _MetadataProxy:
    """Proxy that adapts NodeMetadataDict to the VectorSearchMetadata Protocol."""

    def __init__(self, meta: NodeMetadataDict) -> None:
        self.span_start = int(meta["span_start"])
        self.span_end = int(meta["span_end"])
        self.parent_id = str(meta["parent_id"])
        self.document_id = str(meta["document_id"])
        self.is_leaf = int(meta["is_leaf"])


class PgVectorIndex(VectorIndex):
    """VectorIndex backed by PostgreSQL pgvector via SearchService."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self._search = SearchService(db_manager)

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[tuple[str, float, VectorSearchMetadata]]:
        # Filter where clause to the stricter type accepted by SearchService
        where_strict: dict[str, str | int | float] | None = None
        if where:
            tmp: dict[str, str | int | float] = {}
            for k, v in where.items():
                if v is None:
                    continue
                if isinstance(v, (str | int | float)):
                    tmp[k] = v
            where_strict = tmp if tmp else None

        rows = self._search.search_similar(query_embedding, n_results, where_strict)
        out: list[tuple[str, float, VectorSearchMetadata]] = []
        for node_id, score, meta in rows:
            out.append((node_id, float(score), _MetadataProxy(meta)))
        return out

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, VectorSearchMetadata]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        # Convert back to SearchService's expected metadata shape
        converted: list[tuple[str, float, NodeMetadataDict]] = []
        for node_id, score, meta in candidates:
            md: NodeMetadataDict = {
                "span_start": int(meta.span_start),
                "span_end": int(meta.span_end),
                "parent_id": str(meta.parent_id),
                "document_id": str(meta.document_id),
                "is_leaf": int(meta.is_leaf),
            }
            converted.append((node_id, float(score), md))

        return self._search.compute_mmr_diverse_results(
            query_embedding, converted, lambda_param, k
        )
