"""SQLite-based tests covering span-corruption regressions using the runtime."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.splitter import TextSplitter
from tests.conftest import IndexerRuntimeHarness
from tests.vector_index_stubs import RecordingVectorIndex


def _configure_runtime(
    harness: IndexerRuntimeHarness,
    config: IndexConfig,
    vector_index: RecordingVectorIndex,
) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config
    harness.runtime._vector_index_factory = lambda _model: vector_index
    harness.worker_coordinator._vector_index_factory = lambda _doc_id: vector_index


@pytest.mark.usefixtures("sqlite_backend")
class TestSpanCorruptionSQLite:
    async def _index_document(
        self,
        harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
        document_id: str,
        index_config: IndexConfig,
        vector_index: RecordingVectorIndex,
        text: str,
        client: AsyncMock,
    ) -> DocumentStore:
        _configure_runtime(harness, index_config, vector_index)
        harness.llm_service.client = client
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path=f"{document_id}.txt",
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )
        await harness.clear(document_id)
        await harness.append(
            document_id,
            text,
            replace_existing=True,
            file_path=f"{document_id}.txt",
        )
        await harness.wait_for_idle(document_id)
        return doc_store

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(3.0)
    async def test_odd_nodes_create_invalid_spans(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        index_config = IndexConfig.load(
            target_chunk_tokens=100,
            preceding_context_tokens=10,
        )
        vector_index = RecordingVectorIndex()
        client = AsyncMock()
        client.embeddings.create = AsyncMock(
            side_effect=lambda **kwargs: Mock(
                data=[Mock(embedding=[0.1] * 1536)]
                * len(kwargs.get("input", [kwargs.get("input")]))
            )
        )
        client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(message=Mock(content="Summary of left and right content"))
                ]
            )
        )

        chunk_text = (
            "This is a longer chunk of text that should be approximately one hundred tokens. "
            * 12
        )
        # Keep an odd chunk count to exercise wraparound pairing without
        # blowing past the global 3s test timeout. Nine chunks still build a
        # multi-level tree while staying comfortably within the budget.
        chunks = [f"Chunk {i}: {chunk_text}" for i in range(9)]
        text = " ".join(chunks)

        document_id = "span-corruption"
        doc_store = await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id,
            index_config,
            vector_index,
            text,
            client,
        )

        nodes = list(doc_store.nodes.get_all())
        corrupt_nodes = []
        for node in nodes:
            if node.span_end < node.span_start:
                corrupt_nodes.append(node)
            elif node.span_start == node.span_end and node.height > 0:
                corrupt_nodes.append(node)

        assert (
            len(corrupt_nodes) == 0
        ), f"Found {len(corrupt_nodes)} nodes with invalid spans"

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(3.0)
    async def test_wraparound_pairing(
        self,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        index_config = IndexConfig.load(target_chunk_tokens=100)
        vector_index = RecordingVectorIndex()
        client = AsyncMock()
        client.embeddings.create = AsyncMock(
            side_effect=lambda **kwargs: Mock(
                data=[Mock(embedding=[0.1] * 1536)]
                * len(kwargs.get("input", [kwargs.get("input")]))
            )
        )
        client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="Summary of the content"))]
            )
        )

        base_text = "The quick brown fox jumps over the lazy dog. " * 20
        text = " ".join(
            [f"CHUNK_{i}_START {base_text} CHUNK_{i}_END" for i in range(5)]
        )

        document_id = "wraparound"
        doc_store = await self._index_document(
            indexer_runtime_harness,
            storage_backend,
            document_id,
            index_config,
            vector_index,
            text,
            client,
        )

        nodes = list(doc_store.nodes.get_all())
        nodes_by_height: dict[int, list[TreeNode]] = {}
        for node in nodes:
            nodes_by_height.setdefault(node.height, []).append(node)
        for height in nodes_by_height:
            nodes_by_height[height].sort(key=lambda n: n.span_start)

        for node in nodes:
            if node.height <= 0:
                continue
            assert node.span_end >= node.span_start
            if node.left_child_id:
                left = doc_store.nodes.get_node(node.left_child_id)
                assert left is not None
                assert (
                    node.span_start <= left.span_start <= left.span_end <= node.span_end
                )
            if node.right_child_id:
                right = doc_store.nodes.get_node(node.right_child_id)
                assert right is not None
                assert (
                    node.span_start
                    <= right.span_start
                    <= right.span_end
                    <= node.span_end
                )

    def test_manual_span_corruption_scenario(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> None:
        doc_store = sqlite_store_factory("test-doc")
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "leaf0",
                "text": "First leaf content",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "parent_id": "internal0",
            },
            {
                "node_id": "leaf1",
                "text": "Second leaf content",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 100,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "parent_id": "internal0",
            },
            {
                "node_id": "leaf2",
                "text": "Third leaf content",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 200,
                "span_end": 300,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "parent_id": "internal1",
            },
            {
                "node_id": "leaf3",
                "text": "Fourth leaf content",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 300,
                "span_end": 400,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "parent_id": "internal1",
            },
            {
                "node_id": "leaf4",
                "text": "Fifth leaf content (the odd one out)",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 400,
                "span_end": 500,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "parent_id": "internal2",
            },
            {
                "node_id": "internal0",
                "text": "Summary of leaves 0-1",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 20,
                "height": 1,
                "left_child_id": "leaf0",
                "right_child_id": "leaf1",
                "parent_id": "internal3",
            },
            {
                "node_id": "internal1",
                "text": "Summary of leaves 2-3",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 200,
                "span_end": 400,
                "document_id": "test-doc",
                "token_count": 20,
                "height": 1,
                "left_child_id": "leaf2",
                "right_child_id": "leaf3",
                "parent_id": "internal3",
            },
            {
                "node_id": "internal2",
                "text": "Summary of leaf 4 (single child)",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 400,
                "span_end": 500,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 1,
                "left_child_id": "leaf4",
                "right_child_id": None,
                "parent_id": "root",
            },
            {
                "node_id": "internal3",
                "text": "Summary of leaves 0-3",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 400,
                "document_id": "test-doc",
                "token_count": 40,
                "height": 2,
                "left_child_id": "internal0",
                "right_child_id": "internal1",
                "parent_id": "root",
            },
            {
                "node_id": "root",
                "text": "Root summary of all content",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 500,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 3,
                "left_child_id": "internal3",
                "right_child_id": "internal2",
            },
        ]

        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("leaf0", "internal0"),
                ("leaf1", "internal0"),
                ("leaf2", "internal1"),
                ("leaf3", "internal1"),
                ("leaf4", "internal2"),
                ("internal0", "internal3"),
                ("internal1", "internal3"),
                ("internal2", "root"),
                ("internal3", "root"),
            ]
        )

        corrupt_nodes = []
        for node in doc_store.nodes.get_all():
            if node.span_end < node.span_start:
                corrupt_nodes.append(node)
            if node.left_child_id:
                left_child = doc_store.nodes.get_node(node.left_child_id)
                if left_child and not (
                    node.span_start <= left_child.span_start
                    and left_child.span_end <= node.span_end
                ):
                    corrupt_nodes.append(node)
            if node.right_child_id:
                right_child = doc_store.nodes.get_node(node.right_child_id)
                if right_child and not (
                    node.span_start <= right_child.span_start
                    and right_child.span_end <= node.span_end
                ):
                    corrupt_nodes.append(node)

        assert corrupt_nodes == []
