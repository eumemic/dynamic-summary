"""Server-side append pipeline that creates leaf nodes and embeddings."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import IndexConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.splitter import TextSplitter
from ragzoom.utils.tokenization import tokenizer
from ragzoom.vector_api import Vector

logger = logging.getLogger(__name__)


NodePayload = dict[
    str,
    str | int | float | bool | list[float] | NDArray[np.float64] | None,
]


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class LeafSpec:
    node_id: str
    text: str
    span_start: int
    span_end: int
    token_count: int
    preceding_neighbor_id: str | None
    following_neighbor_id: str | None


@dataclass
class AppendOutcome:
    document_id: str
    appended_span_start: int
    appended_span_end: int
    new_leaf_ids: list[str]
    deleted_node_ids: list[str]
    total_leaves: int


class AppendExecutor:
    """Create new leaves for appended content and write embeddings synchronously."""

    def __init__(
        self,
        config: IndexConfig,
        embedder: EmbeddingProvider,
        *,
        splitter: TextSplitter | None = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._splitter = splitter or TextSplitter(config)

    async def append(
        self,
        *,
        store: DocumentStore,
        vector_index: VectorIndex,
        document_id: str,
        new_text: str,
    ) -> AppendOutcome:
        if not new_text:
            raise ValueError("append requires non-empty text")

        right_leaf = store.nodes.get_rightmost_leaf_for_document(document_id)
        tail_start = int(right_leaf.span_start) if right_leaf else 0
        preceding_neighbor = (
            getattr(right_leaf, "preceding_neighbor_id", None) if right_leaf else None
        )
        following_neighbor = (
            getattr(right_leaf, "following_neighbor_id", None) if right_leaf else None
        )

        existing_tail_text = right_leaf.text or "" if right_leaf else ""
        combined_text = existing_tail_text + new_text
        if not combined_text:
            raise ValueError("append produced no text to index")

        chunks = self._splitter.split_text(combined_text)
        if not chunks:
            raise ValueError("splitter returned no chunks for append")

        leaf_specs = self._build_leaf_specs(
            chunks,
            tail_start=tail_start,
            first_leaf_id=right_leaf.id if right_leaf else None,
            preceding_neighbor_id=preceding_neighbor,
            following_neighbor_id=following_neighbor,
        )

        embeddings = await self._embedder.embed_texts(
            [leaf.text for leaf in leaf_specs]
        )
        if len(embeddings) != len(leaf_specs):
            raise RuntimeError("Embedding provider returned mismatched result count")

        deleted_node_ids = self._collect_deletion_ids(store, right_leaf)
        rollback_vectors = self._load_existing_vectors(vector_index, deleted_node_ids)
        rollback_new_ids: list[str] = []

        payload: list[NodePayload] = []
        for leaf in leaf_specs:
            payload.append(
                {
                    "node_id": leaf.node_id,
                    "text": leaf.text,
                    "span_start": leaf.span_start,
                    "span_end": leaf.span_end,
                    "parent_id": None,
                    "left_child_id": None,
                    "right_child_id": None,
                    "document_id": document_id,
                    "token_count": leaf.token_count,
                    "height": 0,
                    "preceding_neighbor_id": leaf.preceding_neighbor_id,
                    "following_neighbor_id": leaf.following_neighbor_id,
                }
            )

        neighbor_updates = self._build_neighbor_updates(
            store,
            leaf_specs,
            preceding_neighbor,
            following_neighbor,
        )

        try:
            with store.transaction() as session:
                if deleted_node_ids:
                    store.nodes.delete_nodes(deleted_node_ids, session=session)
                store.nodes.add_batch(payload, session=session)
                if neighbor_updates:
                    store.nodes.update_neighbors_batch(
                        neighbor_updates, session=session
                    )

                if deleted_node_ids:
                    self._delete_vectors(vector_index, deleted_node_ids)

                vector_payload: list[
                    tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
                ] = []
                for leaf, embedding in zip(leaf_specs, embeddings, strict=True):
                    embedding_array = np.asarray(embedding, dtype=np.float64)
                    embedding_value: list[float] | NDArray[np.float64] = embedding_array
                    vector_payload.append(
                        (
                            leaf.node_id,
                            embedding_value,
                            {
                                "document_id": document_id,
                                "span_start": leaf.span_start,
                                "span_end": leaf.span_end,
                                "is_leaf": 1,
                            },
                        )
                    )

                vector_index.upsert(vector_payload)
                rollback_new_ids = [leaf.node_id for leaf in leaf_specs]
        except Exception:
            if rollback_vectors:
                try:
                    vector_index.upsert(rollback_vectors)
                except Exception:  # pragma: no cover - best effort rollback
                    logger.exception(
                        "Failed to restore vectors after append rollback",
                        extra={"document_id": document_id},
                    )
            if rollback_new_ids:
                try:
                    vector_index.delete(ids=rollback_new_ids)
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.exception(
                        "Failed to delete new vectors during append rollback",
                        extra={"document_id": document_id},
                    )
            raise

        affected_nodes = set(deleted_node_ids)
        affected_nodes.update(leaf.node_id for leaf in leaf_specs)
        store.tree.clear_depth_cache(list(affected_nodes))

        total_leaves = store.nodes.leaf_count()
        appended_span_end = leaf_specs[-1].span_end

        return AppendOutcome(
            document_id=document_id,
            appended_span_start=leaf_specs[0].span_start,
            appended_span_end=appended_span_end,
            new_leaf_ids=[leaf.node_id for leaf in leaf_specs],
            deleted_node_ids=deleted_node_ids,
            total_leaves=total_leaves,
        )

    def _build_leaf_specs(
        self,
        chunks: Sequence[str],
        *,
        tail_start: int,
        first_leaf_id: str | None,
        preceding_neighbor_id: str | None,
        following_neighbor_id: str | None,
    ) -> list[LeafSpec]:
        specs: list[LeafSpec] = []
        span_cursor = tail_start

        for index, chunk in enumerate(chunks):
            node_id = (
                first_leaf_id if index == 0 and first_leaf_id else str(uuid.uuid4())
            )
            span_end = span_cursor + len(chunk)
            token_count = tokenizer.count_tokens(chunk)

            specs.append(
                LeafSpec(
                    node_id=node_id,
                    text=chunk,
                    span_start=span_cursor,
                    span_end=span_end,
                    token_count=token_count,
                    preceding_neighbor_id=None,
                    following_neighbor_id=None,
                )
            )
            span_cursor = span_end

        for idx, leaf in enumerate(specs):
            prev_id = specs[idx - 1].node_id if idx > 0 else preceding_neighbor_id
            next_id = (
                specs[idx + 1].node_id
                if idx + 1 < len(specs)
                else following_neighbor_id
            )
            specs[idx] = LeafSpec(
                node_id=leaf.node_id,
                text=leaf.text,
                span_start=leaf.span_start,
                span_end=leaf.span_end,
                token_count=leaf.token_count,
                preceding_neighbor_id=prev_id,
                following_neighbor_id=next_id,
            )

        return specs

    def _collect_deletion_ids(
        self,
        store: DocumentStore,
        right_leaf: TreeNode | None,
    ) -> list[str]:
        if right_leaf is None:
            return []
        to_delete: list[str] = []
        current: TreeNode | None = right_leaf
        visited: set[str] = set()
        while current is not None and current.id not in visited:
            visited.add(current.id)
            to_delete.append(current.id)
            parent_id = getattr(current, "parent_id", None)
            if not parent_id:
                break
            current = store.nodes.get(parent_id)
        return to_delete

    def _build_neighbor_updates(
        self,
        store: DocumentStore,
        leaves: Sequence[LeafSpec],
        preceding_neighbor: str | None,
        following_neighbor: str | None,
    ) -> list[tuple[str, str | None, str | None]]:
        updates: list[tuple[str, str | None, str | None]] = []
        if preceding_neighbor:
            prev_node = store.nodes.get(preceding_neighbor)
            if prev_node is not None:
                updates.append(
                    (
                        preceding_neighbor,
                        getattr(prev_node, "preceding_neighbor_id", None),
                        leaves[0].node_id,
                    )
                )
        if following_neighbor:
            next_node = store.nodes.get(following_neighbor)
            if next_node is not None:
                updates.append(
                    (
                        following_neighbor,
                        leaves[-1].node_id,
                        getattr(next_node, "following_neighbor_id", None),
                    )
                )
        for leaf in leaves:
            updates.append(
                (
                    leaf.node_id,
                    leaf.preceding_neighbor_id,
                    leaf.following_neighbor_id,
                )
            )
        return updates

    def _load_existing_vectors(
        self,
        vector_index: VectorIndex,
        node_ids: Sequence[str],
    ) -> list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]]:
        if not node_ids:
            return []
        try:
            existing = vector_index.get_vectors(list(node_ids))
        except Exception:
            logger.exception("Failed to load vectors prior to append deletion")
            return []

        restored: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = []
        for vec in existing:
            if not isinstance(vec, Vector):
                continue
            embedding_value: list[float] | NDArray[np.float64] = np.asarray(
                vec.vec, dtype=np.float64
            )
            restored.append(
                (
                    vec.id,
                    embedding_value,
                    dict(vec.meta),
                )
            )
        return restored

    @staticmethod
    def _delete_vectors(vector_index: VectorIndex, node_ids: Sequence[str]) -> None:
        if not node_ids:
            return
        try:
            vector_index.delete(ids=list(node_ids))
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to delete vectors during append")
