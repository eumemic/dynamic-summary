"""Pure-Python VectorIndex implementation for dev/tests.

Features:
- In-memory storage of embeddings with optional file-backed persistence
- Exact cosine similarity via numpy
- Deterministic, vectorized MMR implementation

This backend enables a file-based (SQLite + Python vector index) developer
experience with no external services. For production, pgvector is still
available via PgVectorIndex.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class _Meta:
    span_start: int
    span_end: int
    parent_id: str
    document_id: str
    is_leaf: int


class PythonVectorIndex:
    """Simple vector index using numpy arrays.

    Persistence (optional):
    - vectors: a single .npy file storing a 2D float32 matrix
    - ids: a JSON list of string IDs aligned to vector rows
    - meta: a JSON dict id -> minimal metadata required by retrieval
    """

    def __init__(self, persist_dir: str | None = None) -> None:
        self._ids: list[str] = []
        self._id_to_row: dict[str, int] = {}
        self._meta: dict[str, _Meta] = {}
        # Store L2-normalized vectors for fast cosine as dot product
        self._vectors: NDArray[np.float32] | None = None
        self._persist_dir = persist_dir
        if persist_dir:
            self._try_load(persist_dir)

    # -------- Persistence ---------
    def _persist_paths(self, base: str) -> tuple[str, str, str]:
        vec_path = os.path.join(base, "vectors.npy")
        ids_path = os.path.join(base, "ids.json")
        meta_path = os.path.join(base, "meta.json")
        return vec_path, ids_path, meta_path

    def _try_load(self, base: str) -> None:
        vec_path, ids_path, meta_path = self._persist_paths(base)
        try:
            if (
                os.path.exists(vec_path)
                and os.path.exists(ids_path)
                and os.path.exists(meta_path)
            ):
                vectors = np.load(vec_path)
                with open(ids_path, encoding="utf-8") as f:
                    ids = json.load(f)
                with open(meta_path, encoding="utf-8") as f:
                    raw_meta = json.load(f)

                # Rebuild structures
                self._ids = list(ids)
                self._id_to_row = {i: r for r, i in enumerate(self._ids)}
                self._vectors = vectors.astype(np.float32)
                meta_out: dict[str, _Meta] = {}
                for k, v in raw_meta.items():
                    if not isinstance(v, dict):
                        raise TypeError("Invalid meta record type")
                    ss = v.get("span_start")
                    se = v.get("span_end")
                    pid = v.get("parent_id")
                    did = v.get("document_id")
                    leaf = v.get("is_leaf")
                    if not isinstance(ss, int) or not isinstance(se, int):
                        raise TypeError("Invalid span types in meta")
                    if not isinstance(pid, str) or not isinstance(did, str):
                        raise TypeError("Invalid ID types in meta")
                    if not isinstance(leaf, int):
                        raise TypeError("Invalid is_leaf type in meta")
                    meta_out[k] = _Meta(
                        span_start=ss,
                        span_end=se,
                        parent_id=pid,
                        document_id=did,
                        is_leaf=leaf,
                    )
                self._meta = meta_out
        except Exception:
            # Corrupt snapshot should not prevent usage; start empty
            self._ids = []
            self._id_to_row = {}
            self._vectors = None
            self._meta = {}

    def _save(self) -> None:
        if not self._persist_dir:
            return
        os.makedirs(self._persist_dir, exist_ok=True)
        vec_path, ids_path, meta_path = self._persist_paths(self._persist_dir)
        if self._vectors is not None:
            np.save(vec_path, self._vectors)
            with open(ids_path, "w", encoding="utf-8") as f:
                json.dump(self._ids, f)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        i: {
                            "span_start": m.span_start,
                            "span_end": m.span_end,
                            "parent_id": m.parent_id,
                            "document_id": m.document_id,
                            "is_leaf": m.is_leaf,
                        }
                        for i, m in self._meta.items()
                    },
                    f,
                )

    # -------- Mutation helpers (not part of protocol yet) ---------
    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
        *,
        normalize: bool = True,
        persist: bool = True,
    ) -> None:
        """Insert or replace vectors.

        Each item: (node_id, embedding, meta{span_start, span_end, parent_id, document_id, is_leaf})
        """
        if not items:
            return
        vecs = []
        for node_id, emb, meta in items:
            v = np.asarray(emb, dtype=np.float32)
            if normalize:
                n = np.linalg.norm(v)
                v = v / n if n > 0 else v
            row = self._id_to_row.get(node_id)
            if row is None:
                # append new row
                row = len(self._ids)
                self._ids.append(node_id)
                self._id_to_row[node_id] = row
                vecs.append(v)
            else:
                # update existing in-place later
                if self._vectors is not None:
                    self._vectors[row, :] = v
                else:
                    vecs.append(v)

            # store meta with strict typing
            def _req_int(key: str) -> int:
                val = meta.get(key)
                if isinstance(val, int | np.integer):
                    return int(val)
                raise TypeError(f"Missing or invalid integer for {key}")

            def _req_str(key: str) -> str:
                val = meta.get(key)
                if isinstance(val, str):
                    return val
                if val is None:
                    return ""
                # Coerce common non-string values to string for robustness in tests
                if isinstance(val, int | float | bool):
                    return str(val)
                raise TypeError(f"Missing or invalid string for {key}")

            is_leaf_val = meta.get("is_leaf")
            if isinstance(is_leaf_val, int | np.integer):
                is_leaf_i = int(is_leaf_val)
            else:
                raise TypeError("Missing or invalid integer for is_leaf")

            self._meta[node_id] = _Meta(
                span_start=_req_int("span_start"),
                span_end=_req_int("span_end"),
                parent_id=_req_str("parent_id"),
                document_id=_req_str("document_id"),
                is_leaf=is_leaf_i,
            )

        if vecs:
            mat = np.vstack(vecs)
            if self._vectors is None:
                self._vectors = mat
            else:
                self._vectors = np.vstack([self._vectors, mat])

        if persist:
            self._save()

    # -------- VectorIndex protocol ---------
    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[tuple[str, float, dict[str, str | int | float | bool | None]]]:
        if self._vectors is None or not self._ids:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        # Normalize to use dot product as cosine similarity
        qn = q / (np.linalg.norm(q) + 1e-12)
        sims = self._vectors @ qn

        mask: NDArray[np.bool_] | None = None
        if where:
            mask = np.ones(len(self._ids), dtype=bool)
            doc_filter = where.get("document_id")
            if doc_filter is not None:
                doc = (
                    str(doc_filter)
                    if isinstance(doc_filter, (str | int | float | bool))
                    else ""
                )
                mask &= np.array(
                    [self._meta[i].document_id == doc for i in self._ids], dtype=bool
                )
            if mask is not None and not mask.any():
                return []
        if mask is not None:
            idxs = np.where(mask)[0]
            if idxs.size == 0:
                return []
            sub_sims = sims[idxs]
            topk_idx = np.argsort(-sub_sims)[:n_results]
            result_rows = idxs[topk_idx]
        else:
            topk_idx = np.argsort(-sims)[:n_results]
            result_rows = topk_idx

        out: list[tuple[str, float, dict[str, str | int | float | bool | None]]] = []
        for r in result_rows:
            node_id = self._ids[int(r)]
            m = self._meta[node_id]
            out.append(
                (
                    node_id,
                    float(sims[int(r)]),
                    {
                        "span_start": m.span_start,
                        "span_end": m.span_end,
                        "parent_id": m.parent_id,
                        "document_id": m.document_id,
                        "is_leaf": m.is_leaf,
                    },
                )
            )
        return out

    # Note: MMR selection is performed in core via generic utilities.
