"""VectorIndex implementation using PostgreSQL + pgvector in its own table.

This adapter manages a dedicated table for vectors and minimal metadata, and is
independent of the storage tables. It requires the pgvector extension.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.event import listens_for

from ragzoom.contracts.vector_filter import (
    DocumentIdFilter,
    SpanEndLtFilter,
    VectorFilter,
)
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.exceptions import UnsupportedFilterError
from ragzoom.vector_api import MetaDict, Vector

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import dependent on environment
    from pgvector.psycopg import register_vector as _register_vector
except Exception:  # pragma: no cover - optional dependency
    try:
        from pgvector.psycopg2 import register_vector as _register_vector
    except Exception:  # pragma: no cover
        _register_vector = None


class PgVectorIndexAdapter(VectorIndex):
    def __init__(self, database_url: str, model_id: str) -> None:
        self._model_id = model_id
        self._engine: Engine = create_engine(database_url, pool_pre_ping=True)

        # Register pgvector for connections if possible
        if database_url.startswith("postgres") and _register_vector is not None:

            @listens_for(self._engine, "connect")
            def _on_connect(
                dbapi_conn: object, connection_record: object
            ) -> None:  # noqa: D401 - hook
                try:
                    raw_conn = (
                        dbapi_conn.connection
                        if hasattr(dbapi_conn, "connection")
                        else dbapi_conn
                    )
                    _register_vector(raw_conn)
                except Exception as e:  # pragma: no cover - best effort
                    logger.debug(f"pgvector registration note: {e}")

        self._ensure_schema()

    # --- VectorIndex ---
    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        filters: Sequence[VectorFilter] | None = None,
    ) -> list[Vector]:
        # Build SQL WHERE clauses from typed filters
        where_clauses: list[str] = []
        params: dict[str, object] = {"k": int(k)}

        if filters:
            for f in filters:
                match f:
                    case DocumentIdFilter(value=doc_id):
                        where_clauses.append("document_id = :doc_id")
                        params["doc_id"] = doc_id
                    case SpanEndLtFilter(threshold=threshold):
                        where_clauses.append("span_end < :span_end_lt")
                        params["span_end_lt"] = threshold
                    case _:
                        raise UnsupportedFilterError(type(f).__name__, "PgVectorIndex")

        q: list[float] = [float(x) for x in cast(Sequence[float], query_embedding)]
        params["q"] = q

        sql = (
            "SELECT id, embedding, document_id, span_start, span_end, parent_id, is_leaf, height, level_index, coord_version "
            "FROM node_vectors "
        )
        if where_clauses:
            sql += "WHERE " + " AND ".join(where_clauses) + " "
        # Order by cosine distance; convert to similarity in Python
        sql += "ORDER BY embedding <=> :q LIMIT :k"

        out: list[Vector] = []
        with self._engine.begin() as conn:
            rows = conn.execute(text(sql), params).fetchall()
            for row in rows:
                out.append(self._row_to_vector(row))
        return out

    def get_vectors(self, ids: list[str]) -> list[Vector]:
        if not ids:
            return []
        # Use ANY array binding to avoid constructing SQL fragments
        params: dict[str, object] = {"ids": list(ids)}
        sql = text(
            "SELECT id, embedding, document_id, span_start, span_end, parent_id, is_leaf, height, level_index, coord_version "
            "FROM node_vectors WHERE id = ANY(:ids)"
        )
        out: list[Vector] = []
        with self._engine.begin() as conn:
            rows = conn.execute(sql, params).fetchall()
            by_id = {str(r[0]): r for r in rows}
        for node_id in ids:
            r = by_id.get(node_id)
            if r is None:
                raise KeyError(f"Vector not found for id {node_id}")
            out.append(self._row_to_vector(r))
        return out

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        if not items:
            return
        sql = text(
            """
            INSERT INTO node_vectors (id, embedding, document_id, span_start, span_end, parent_id, is_leaf, height, level_index, coord_version)
            VALUES (:id, :emb, :doc, :ss, :se, :pid, :leaf, :height, :level_index, :coord_version)
            ON CONFLICT (id)
            DO UPDATE SET embedding = EXCLUDED.embedding,
                          document_id = EXCLUDED.document_id,
                          span_start = EXCLUDED.span_start,
                          span_end = EXCLUDED.span_end,
                          parent_id = EXCLUDED.parent_id,
                          is_leaf = EXCLUDED.is_leaf,
                          height = EXCLUDED.height,
                          level_index = EXCLUDED.level_index,
                          coord_version = EXCLUDED.coord_version
            """
        )
        with self._engine.begin() as conn:
            for node_id, emb, meta in items:
                params = {
                    "id": node_id,
                    "emb": [float(x) for x in cast(Sequence[float], emb)],
                    "doc": str(meta.get("document_id", "")),
                    "ss": int(cast(int | float, meta.get("span_start", 0))),
                    "se": int(cast(int | float, meta.get("span_end", 0))),
                    "pid": str(meta.get("parent_id", "")),
                    "leaf": int(cast(int | float | bool, meta.get("is_leaf", 0))),
                    "height": int(cast(int | float, meta.get("height", 0))),
                    "level_index": int(cast(int | float, meta.get("level_index", 0))),
                    "coord_version": int(
                        cast(int | float, meta.get("coord_version", 0))
                    ),
                }
                conn.execute(sql, params)

    def delete(
        self, filter: dict[str, object] | None = None, ids: list[str] | None = None
    ) -> int:
        with self._engine.begin() as conn:
            if ids:
                del_sql = text("DELETE FROM node_vectors WHERE id = ANY(:ids)")
                res = conn.execute(del_sql, {"ids": list(ids)})
                return int(res.rowcount or 0)
            if filter and "document_id" in filter:
                doc = filter["document_id"]
                params: dict[str, object] = {"doc": str(doc)}
                del_sql = text("DELETE FROM node_vectors WHERE document_id = :doc")
                res = conn.execute(del_sql, params)
                return int(res.rowcount or 0)
        return 0

    # --- setup ---
    def _ensure_schema(self) -> None:
        # Create vector extension and table for node vectors
        with self._engine.begin() as conn:
            try:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception as e:  # pragma: no cover - environment-specific
                logger.debug(f"pgvector extension note: {e}")
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS node_vectors (
                        id TEXT PRIMARY KEY,
                        embedding VECTOR,
                        document_id TEXT,
                        span_start INTEGER,
                        span_end INTEGER,
                        parent_id TEXT,
                        is_leaf SMALLINT,
                        height INTEGER DEFAULT 0,
                        level_index INTEGER DEFAULT 0,
                        coord_version SMALLINT DEFAULT 0
                    );
                    """
                )
            )
            # Create basic index on document_id for faster filtering
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_node_vectors_doc ON node_vectors(document_id)"
                )
            )
            # Drop legacy doc_version column if still present
            conn.execute(
                text(
                    """
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'node_vectors'
                            AND column_name = 'doc_version'
                        ) THEN
                            ALTER TABLE node_vectors
                            DROP COLUMN doc_version;
                        END IF;
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'node_vectors'
                            AND column_name = 'height'
                        ) THEN
                            ALTER TABLE node_vectors ADD COLUMN height INTEGER NOT NULL DEFAULT 0;
                        END IF;
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'node_vectors'
                            AND column_name = 'level_index'
                        ) THEN
                            ALTER TABLE node_vectors ADD COLUMN level_index INTEGER NOT NULL DEFAULT 0;
                        END IF;
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'node_vectors'
                            AND column_name = 'coord_version'
                        ) THEN
                            ALTER TABLE node_vectors ADD COLUMN coord_version SMALLINT NOT NULL DEFAULT 0;
                        END IF;
                    END $$;
                    """
                )
            )

    def _row_to_vector(self, row: Sequence[object]) -> Vector:
        node_id = str(row[0])
        emb = np.asarray(row[1], dtype=np.float32)
        meta: MetaDict = {
            "document_id": str(row[2]) if row[2] is not None else "",
            "span_start": int(cast(int | float | None, row[3]) or 0),
            "span_end": int(cast(int | float | None, row[4]) or 0),
            "parent_id": str(row[5]) if row[5] is not None else "",
            "is_leaf": int(cast(int | float | None, row[6]) or 0),
            "height": int(cast(int | float | None, row[7]) or 0),
            "level_index": int(cast(int | float | None, row[8]) or 0),
            "coord_version": int(cast(int | float | None, row[9]) or 0),
        }
        return Vector(
            id=node_id, vec=emb, meta=meta, model_id=self._model_id, dim=len(emb)
        )
