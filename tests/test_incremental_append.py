from uuid import uuid4

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.index import AppendStats
from ragzoom.splitter import TextSplitter
from ragzoom.validate import set_validation_enabled
from ragzoom.vector_api import Vector, ensure_normalized
from tests.conftest import BackwardCompatibilityConfig, IndexerRuntimeHarness


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._vectors: dict[str, Vector] = {}

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[Vector]:
        return []

    def get_vectors(self, ids: list[str]) -> list[Vector]:
        return [self._vectors[node_id] for node_id in ids if node_id in self._vectors]

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        for node_id, embedding, meta in items:
            normalized = ensure_normalized(embedding)
            payload_meta: dict[str, str | int | float | bool | None]
            if isinstance(meta, dict):
                payload_meta = {
                    key: value
                    for key, value in meta.items()
                    if isinstance(value, str | int | float | bool) or value is None
                }
            else:
                payload_meta = {}
            model_id = str(payload_meta.get("model_id", ""))
            self._vectors[node_id] = Vector(
                id=node_id,
                vec=normalized,
                meta=payload_meta,
                model_id=model_id,
                dim=int(normalized.shape[0]),
            )

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        target_ids = list(ids or [])
        if not target_ids and filter is not None:
            count = len(self._vectors)
            self._vectors.clear()
            return count
        removed = 0
        for node_id in target_ids:
            if node_id in self._vectors:
                self._vectors.pop(node_id)
                removed += 1
        return removed


def _configure_runtime(harness: IndexerRuntimeHarness, config: IndexConfig) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config


def _make_runtime_doc(
    doc_id: str,
    storage_backend: StorageBackend,
) -> DocumentStore:
    storage_backend.clear_document(doc_id)
    doc_store = storage_backend.for_document(doc_id)
    doc_store.set_metadata(
        file_path=f"{doc_id}.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )
    return doc_store


def _reconstruct_document(doc_store: DocumentStore) -> str:
    leaves = doc_store.nodes.get_leaves()
    leaves.sort(key=lambda n: int(n.span_start))
    return "".join(leaf.text or "" for leaf in leaves)


def _snapshot_document(doc_store: DocumentStore) -> list[tuple[int, int, int, str]]:
    nodes = doc_store.nodes.get_all()
    return sorted(
        [
            (
                int(node.height),
                int(node.span_start),
                int(node.span_end),
                node.text or "",
            )
            for node in nodes
        ]
    )


def _split_into_segments(text: str, segment_count: int) -> list[str]:
    if segment_count <= 1:
        return [text]
    total = len(text)
    base = total // segment_count
    segments: list[str] = []
    cursor = 0
    for idx in range(segment_count - 1):
        segments.append(text[cursor : cursor + base])
        cursor += base
    segments.append(text[cursor:])
    return segments


async def _build_full_and_incremental_documents(
    storage_backend: StorageBackend,
    runtime: IndexerRuntimeHarness,
    config: IndexConfig,
    full_text: str,
    segments: list[str],
) -> tuple[DocumentStore, DocumentStore, list[AppendStats]]:
    full_doc_id = f"full-{uuid4()}"
    incremental_doc_id = f"inc-{uuid4()}"

    full_store = _make_runtime_doc(full_doc_id, storage_backend)
    incremental_store = _make_runtime_doc(incremental_doc_id, storage_backend)

    append_stats: list[AppendStats] = []

    _configure_runtime(runtime, config)
    full_vector_index = InMemoryVectorIndex()
    incremental_vector_index = InMemoryVectorIndex()

    runtime.runtime._vector_index_factory = lambda _model: full_vector_index
    runtime.worker_coordinator._vector_index_factory = lambda _doc_id: full_vector_index

    await runtime.clear(full_doc_id)
    await runtime.clear(incremental_doc_id)
    await runtime.append(
        full_doc_id,
        full_text,
        replace_existing=True,
        file_path=f"{full_doc_id}.txt",
    )
    runtime.runtime._vector_index_factory = lambda _model: incremental_vector_index
    runtime.worker_coordinator._vector_index_factory = (
        lambda _doc_id: incremental_vector_index
    )

    for idx, segment in enumerate(segments):
        stats = await runtime.append(
            incremental_doc_id,
            segment,
            replace_existing=(idx == 0),
            file_path=f"{incremental_doc_id}.txt",
        )
        append_stats.append(
            AppendStats(
                document_id=stats.document_id,
                mutated_nodes=stats.mutated_nodes or 0,
                resummarized_nodes=stats.resummarized_nodes or 0,
                new_leaves=stats.new_leaves or 0,
                total_leaves=stats.chunks_created,
            )
        )

    await runtime.wait_for_idle(full_doc_id)
    await runtime.wait_for_idle(incremental_doc_id)
    return full_store, incremental_store, append_stats


def _collect_leaf_depths(doc_store: DocumentStore) -> list[int]:
    root = doc_store.tree.get_root()
    if root is None:
        return []

    nodes = {node.id: node for node in doc_store.nodes.get_all()}
    stack: list[tuple[str, int]] = [(root.id, 0)]
    depths: list[int] = []

    while stack:
        node_id, depth = stack.pop()
        node = nodes.get(node_id)
        if node is None:
            continue
        left_id = node.left_child_id
        right_id = node.right_child_id
        if not left_id and not right_id:
            depths.append(depth)
            continue
        if right_id:
            stack.append((right_id, depth + 1))
        if left_id:
            stack.append((left_id, depth + 1))

    return depths


def _assert_left_balanced(doc_store: DocumentStore) -> None:
    nodes = {node.id: node for node in doc_store.nodes.get_all()}

    for node in nodes.values():
        left_id = node.left_child_id
        right_id = node.right_child_id

        if not left_id and not right_id:
            continue

        left = nodes.get(left_id) if left_id else None
        right = nodes.get(right_id) if right_id else None

        if right is None:
            assert left is not None
            assert int(node.height) == int(left.height) + 1
            continue

        assert left is not None
        left_height = int(left.height)
        right_height = int(right.height)

        assert left_height >= right_height
        assert left_height - right_height <= 1
        assert int(node.height) == max(left_height, right_height) + 1


class TestIncrementalAppend:
    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(10.0)
    async def test_incremental_equivalence(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        config = base_config.index_config.replace(
            target_chunk_tokens=32,
            preceding_context_tokens=0,
            embedding_batch_size=4,
        )

        full_text = "".join(
            [
                (
                    f"Paragraph {i}. This is deterministic content for testing. "
                    f"Dragons and dwarves confer in room {i % 7}.\n"
                )
                for i in range(12)
            ]
        )

        segments = _split_into_segments(full_text, 3)

        full_store, incremental_store, _ = await _build_full_and_incremental_documents(
            storage_backend,
            indexer_runtime_harness,
            config,
            full_text,
            segments,
        )

        full_snapshot = _snapshot_document(full_store)
        incremental_snapshot = _snapshot_document(incremental_store)
        full_leaves = [entry for entry in full_snapshot if entry[0] == 0]
        incremental_leaves = [entry for entry in incremental_snapshot if entry[0] == 0]
        assert incremental_leaves == full_leaves

        full_doc = _reconstruct_document(full_store)
        incremental_doc = _reconstruct_document(incremental_store)
        assert incremental_doc == full_doc == full_text
        assert _collect_leaf_depths(full_store)
        assert _collect_leaf_depths(incremental_store)

    @pytest.mark.asyncio
    async def test_append_handles_empty_document_without_fallback(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Ensure append works when no existing nodes without using legacy fallback."""

        document_id = "doc-empty-append"
        _make_runtime_doc(document_id, storage_backend)

        await indexer_runtime_harness.clear(document_id)
        stats = await indexer_runtime_harness.append(
            document_id,
            "Initial content for an empty document.",
            replace_existing=True,
            file_path=f"{document_id}.txt",
        )

        assert stats.document_id == document_id
        assert stats.chunks_created > 0
        assert stats.new_leaves == stats.chunks_created
        assert (
            stats.mutated_nodes is not None
            and stats.mutated_nodes >= stats.chunks_created
        )
        doc_store = storage_backend.for_document(document_id)
        assert doc_store.nodes.count() == stats.mutated_nodes

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(20.0)
    async def test_append_height_matches_full_build(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        config = base_config.index_config.replace(
            target_chunk_tokens=32,
            preceding_context_tokens=0,
            embedding_batch_size=4,
        )

        full_text = "".join(
            [
                (
                    f"Paragraph {i}. This is deterministic content for testing. "
                    f"Dragons and dwarves confer in room {i % 5}.\n"
                )
                for i in range(12)
            ]
        )

        segments = _split_into_segments(full_text, 3)

        full_store, incremental_store, stats = (
            await _build_full_and_incremental_documents(
                storage_backend,
                indexer_runtime_harness,
                config,
                full_text,
                segments,
            )
        )

        assert stats
        full_depths = _collect_leaf_depths(full_store)
        incremental_depths = _collect_leaf_depths(incremental_store)
        assert incremental_depths
        assert max(incremental_depths) <= max(full_depths)
        assert stats[-1].total_leaves == incremental_store.nodes.leaf_count()

    @pytest.mark.asyncio
    async def test_append_respects_validation_toggle(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        document_id = "doc-validation-toggle"
        _make_runtime_doc(document_id, storage_backend)

        await indexer_runtime_harness.clear(document_id)
        set_validation_enabled(True)
        try:
            stats = await indexer_runtime_harness.append(
                document_id,
                "Validation enabled text.",
                replace_existing=True,
                file_path=f"{document_id}.txt",
            )
        finally:
            set_validation_enabled(False)

        assert stats.document_id == document_id
        assert stats.chunks_created > 0

    @pytest.mark.asyncio
    async def test_append_tracks_mutation_stats(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        document_id = "doc-track-stats"
        _make_runtime_doc(document_id, storage_backend)

        await indexer_runtime_harness.clear(document_id)
        stats = await indexer_runtime_harness.append(
            document_id,
            "Appending text to measure stats.",
            replace_existing=True,
            file_path=f"{document_id}.txt",
        )
        assert stats.document_id == document_id
        assert stats.chunks_created > 0
        assert (
            stats.mutated_nodes is not None
            and stats.mutated_nodes >= stats.chunks_created
        )

    @pytest.mark.asyncio
    async def test_incremental_append_chain(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        document_id = "doc-chain"
        _make_runtime_doc(document_id, storage_backend)

        await indexer_runtime_harness.clear(document_id)
        stats_first = await indexer_runtime_harness.append(
            document_id,
            "First segment of append operation.",
            replace_existing=True,
            file_path=f"{document_id}.txt",
        )
        stats_second = await indexer_runtime_harness.append(
            document_id,
            " Second segment appended.",
            replace_existing=False,
            file_path=f"{document_id}.txt",
        )

        assert stats_first.document_id == document_id
        assert stats_second.document_id == document_id
        assert stats_second.chunks_created >= stats_first.chunks_created
