"""Compute summarization fidelity for telemetry nodes."""

from __future__ import annotations

import logging
from collections.abc import Callable, MutableMapping, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer
from ragzoom.vector_api import Vector

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


async def compute_fidelity_for_telemetry(
    *,
    document_store: DocumentStore,
    collector: TelemetryCollector,
    vector_index: VectorIndex,
    embedder: EmbeddingProvider,
    token_limit: int,
    max_batch_items: int,
) -> None:
    """Compute parent/child fidelity for every summarized node."""

    if not collector.node_telemetry:
        return

    parent_ids = [
        node_id
        for node_id, record in collector.node_telemetry.items()
        if record.height > 0
    ]
    if not parent_ids:
        return

    await _compute_fidelity(
        document_store=document_store,
        parent_ids=parent_ids,
        vector_index=vector_index,
        embedder=embedder,
        token_limit=token_limit,
        max_batch_items=max_batch_items,
        record_callback=collector.record_node_fidelity,
    )


async def annotate_telemetry_fidelity(
    *,
    document_store: DocumentStore,
    telemetry_nodes: Sequence[object],
    vector_index: VectorIndex,
    embedder: EmbeddingProvider,
    token_limit: int,
    max_batch_items: int,
) -> None:
    """Compute semantic drift for exported telemetry payload."""

    node_lookup: dict[str, MutableMapping[str, object]] = {}
    parent_ids: list[str] = []
    for entry in telemetry_nodes:
        if not isinstance(entry, MutableMapping):
            continue
        node_id = entry.get("node_id")
        if not isinstance(node_id, str):
            continue
        height = entry.get("height", 0)
        node_lookup[node_id] = entry
        if isinstance(height, int) and height > 0:
            parent_ids.append(node_id)

    if not parent_ids:
        return

    def record(node_id: str, fidelity: float) -> None:
        node = node_lookup.get(node_id)
        if node is not None:
            node["fidelity"] = fidelity

    await _compute_fidelity(
        document_store=document_store,
        parent_ids=parent_ids,
        vector_index=vector_index,
        embedder=embedder,
        token_limit=token_limit,
        max_batch_items=max_batch_items,
        record_callback=record,
    )


async def _compute_fidelity(
    *,
    document_store: DocumentStore,
    parent_ids: Sequence[str],
    vector_index: VectorIndex,
    embedder: EmbeddingProvider,
    token_limit: int,
    max_batch_items: int,
    record_callback: Callable[[str, float], None],
) -> None:
    node_map = {
        node.id: node for node in document_store.nodes.get_many(list(parent_ids))
    }
    for node_id in parent_ids:
        if node_id not in node_map:
            node = document_store.nodes.get(node_id)
            if node is not None:
                node_map[node_id] = node
    child_cache: dict[str, TreeNode] = {}

    parent_vectors = await _resolve_parent_vectors(
        vector_index=vector_index,
        node_ids=parent_ids,
        node_map=node_map,
        embedder=embedder,
        token_limit=token_limit,
        max_batch_items=max_batch_items,
    )
    if not parent_vectors:
        return

    baselines: list[tuple[str, str, int]] = []
    for node_id in parent_ids:
        repo_node = node_map.get(node_id) or document_store.nodes.get(node_id)
        if repo_node is None:
            continue
        baseline_text = _build_child_baseline_text(
            repo_node, document_store, child_cache
        )
        if not baseline_text:
            continue
        truncated = _truncate_text_for_embedding(baseline_text, token_limit)
        if not truncated:
            continue
        token_count = tokenizer.count_tokens(truncated)
        if token_count <= 0:
            continue
        baselines.append((node_id, truncated, token_count))

    batches = _build_embedding_batches(baselines, max_batch_items, token_limit)
    if not batches:
        return

    for node_ids, texts in batches:
        embeddings = await embedder.embed_texts(texts)
        if len(embeddings) != len(node_ids):
            raise ValueError("Embedding provider returned mismatched result count")
        for node_id, child_vec in zip(node_ids, embeddings):
            parent_vec = parent_vectors.get(node_id)
            if parent_vec is None:
                continue
            fidelity = _cosine_similarity(
                parent_vec, np.asarray(child_vec, dtype=np.float64)
            )
            if fidelity is not None:
                record_callback(node_id, fidelity)


async def _resolve_parent_vectors(
    *,
    vector_index: VectorIndex,
    node_ids: Sequence[str],
    node_map: dict[str, TreeNode],
    embedder: EmbeddingProvider,
    token_limit: int,
    max_batch_items: int,
) -> dict[str, NDArray[np.float64]]:
    """Load parent vectors from the index or re-embed summaries if missing."""

    resolved: dict[str, NDArray[np.float64]] = {}
    missing: set[str] = set(node_ids)

    vectors = _safe_get_vectors(vector_index, list(node_ids))
    for vector in vectors:
        try:
            resolved[vector.id] = np.asarray(vector.vec, dtype=np.float64)
            missing.discard(vector.id)
        except Exception:
            continue

    if missing and vectors:
        # get_vectors succeeded, so missing nodes truly have no stored vectors.
        logger.debug("Missing %d parent vectors; will re-embed summaries", len(missing))
    elif missing:
        # Bulk fetch failed entirely, retry per-id for partial salvage.
        recovered: dict[str, NDArray[np.float64]] = {}
        still_missing: set[str] = set()
        for node_id in list(missing):
            single = _safe_get_vectors(vector_index, [node_id])
            if not single:
                still_missing.add(node_id)
                continue
            try:
                recovered[node_id] = np.asarray(single[0].vec, dtype=np.float64)
            except Exception:
                still_missing.add(node_id)
                continue
        resolved.update(recovered)
        missing = still_missing

    if not missing:
        return resolved

    fallback_entries: list[tuple[str, str, int]] = []
    for node_id in missing:
        node = node_map.get(node_id)
        if node is None or not node.text:
            continue
        truncated = _truncate_text_for_embedding(node.text, token_limit)
        if not truncated:
            continue
        token_count = tokenizer.count_tokens(truncated)
        if token_count <= 0:
            continue
        fallback_entries.append((node_id, truncated, token_count))

    if not fallback_entries:
        return resolved

    batches = _build_embedding_batches(fallback_entries, max_batch_items, token_limit)
    for batch_ids, texts in batches:
        embeddings = await embedder.embed_texts(texts)
        if len(embeddings) != len(batch_ids):
            raise ValueError("Embedding provider returned mismatched result count")
        for node_id, values in zip(batch_ids, embeddings):
            resolved[node_id] = np.asarray(values, dtype=np.float64)

    return resolved


def _safe_get_vectors(vector_index: VectorIndex, ids: list[str]) -> list[Vector]:
    if not ids:
        return []
    try:
        return vector_index.get_vectors(ids)
    except Exception as exc:
        logger.debug("Failed to fetch %d vectors: %s", len(ids), exc)
        return []


def _cosine_similarity(
    parent_vec: NDArray[np.float64], child_vec: NDArray[np.float64]
) -> float | None:
    norm_parent = float(np.linalg.norm(parent_vec))
    norm_child = float(np.linalg.norm(child_vec))
    if norm_parent == 0.0 or norm_child == 0.0:
        return None
    cosine = float(np.dot(parent_vec, child_vec) / (norm_parent * norm_child))
    return max(-1.0, min(1.0, cosine))


def _build_child_baseline_text(
    node: TreeNode,
    document_store: DocumentStore,
    cache: dict[str, TreeNode],
) -> str:
    parts: list[str] = []
    for child_id in (
        getattr(node, "left_child_id", None),
        getattr(node, "right_child_id", None),
    ):
        if not child_id:
            continue
        child = cache.get(child_id)
        if child is None:
            child = document_store.nodes.get(child_id)
            if child is not None:
                cache[child_id] = child
        if child and child.text:
            parts.append(child.text)
    return "\n".join(parts)


def _truncate_text_for_embedding(text: str, limit: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= limit:
        return text
    return tokenizer.decode(tokens[:limit])


def _build_embedding_batches(
    entries: Sequence[tuple[str, str, int]],
    max_batch_items: int,
    token_budget: int,
) -> list[tuple[list[str], list[str]]]:
    """Group embedding inputs while respecting provider limits."""

    if not entries:
        return []

    batches: list[tuple[list[str], list[str]]] = []
    current_ids: list[str] = []
    current_texts: list[str] = []
    running_tokens = 0

    for node_id, text, token_count in entries:
        exceeds = current_ids and (
            len(current_ids) >= max_batch_items
            or running_tokens + token_count > token_budget
        )
        if exceeds:
            batches.append((list(current_ids), list(current_texts)))
            current_ids = []
            current_texts = []
            running_tokens = 0

        current_ids.append(node_id)
        current_texts.append(text)
        running_tokens += token_count

    if current_ids:
        batches.append((current_ids, current_texts))

    return batches
