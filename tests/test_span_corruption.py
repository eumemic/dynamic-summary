"""Tests for span corruption bug in tree building.

These tests ensure that tree building handles odd numbers of nodes correctly
and prevents span corruption issues.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from tests.conftest import BackwardCompatibilityConfig, IndexerRuntimeHarness


class TestSpanCorruption:
    """Test span corruption issues in tree building."""

    def setup_system(
        self,
        storage_backend: StorageBackend,
        vector_index: _VectorIndexProtocol,
        runtime_harness: IndexerRuntimeHarness,
        *,
        document_id: str = "test-doc",
    ) -> tuple[BackwardCompatibilityConfig, str, AsyncMock]:
        """Set up test system using the runtime harness."""
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path="test_span_corruption.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        index_config = IndexConfig.load(
            target_chunk_tokens=100,
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test-key"),
        )

        runtime_harness.llm_service.config = index_config
        mock_client = AsyncMock()
        runtime_harness.llm_service.client = mock_client

        # Mock the sync OpenAI client used by IndexingEngine's retriever
        mock_sync_client = MagicMock()

        def sync_mock_embeddings(*args: object, **kwargs: object) -> object:
            from types import SimpleNamespace

            input_texts = cast(list[str] | str, kwargs.get("input", []))
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            num_items = len(input_texts)
            return MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts],
                usage=SimpleNamespace(
                    prompt_tokens=num_items * 10, total_tokens=num_items * 10
                ),
            )

        mock_sync_client.embeddings.create = sync_mock_embeddings
        runtime_harness.indexing_engine._openai_client = mock_sync_client

        config = BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )

        return config, document_id, mock_client

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(6.0)
    async def test_odd_nodes_create_invalid_spans(
        self,
        storage_backend: StorageBackend,
        vector_index: _VectorIndexProtocol,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that odd number of nodes creates span corruption."""
        _, document_id, mock_client = self.setup_system(
            storage_backend,
            vector_index,
            indexer_runtime_harness,
        )
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)

        chunk_text = (
            "This is a longer chunk of text that should be approximately one hundred tokens. "
            * 8
        )
        chunks = [f"Chunk {i}: {chunk_text}" for i in range(9)]
        chunks.append(f"Chunk {len(chunks)}: {chunk_text}")
        text = " ".join(chunks)
        text = " ".join(chunks)

        def async_embedding_side_effect(**kwargs: object) -> Mock:
            from types import SimpleNamespace

            input_data = kwargs.get("input", [])
            if isinstance(input_data, str):
                input_data = [input_data]
            input_list = cast(list[str], input_data)
            num_items = len(input_list)
            return Mock(
                data=[Mock(embedding=[0.1] * 1536) for _ in input_list],
                usage=SimpleNamespace(
                    prompt_tokens=num_items * 10, total_tokens=num_items * 10
                ),
            )

        mock_client.embeddings.create = AsyncMock(
            side_effect=async_embedding_side_effect
        )
        mock_client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(message=Mock(content="Summary of left and right content"))
                ],
                usage=Mock(
                    prompt_tokens=100,
                    completion_tokens=10,
                    prompt_tokens_details=Mock(cached_tokens=0),
                ),
            )
        )

        await indexer_runtime_harness.append(
            document_id,
            text,
            replace_existing=True,
            file_path="test_span_corruption.txt",
        )
        await indexer_runtime_harness.wait_for_idle(document_id)

        nodes = cast(Sequence[TreeNode], doc_store.nodes.get_all())

        corrupt_nodes = []
        for node in nodes:
            node_height = node.height
            if node.span_end < node.span_start:
                corrupt_nodes.append(
                    {
                        "id": node.id,
                        "height": node_height,
                        "span_start": node.span_start,
                        "span_end": node.span_end,
                    }
                )
            elif node.span_start == node.span_end and node_height > 0:
                corrupt_nodes.append(
                    {
                        "id": node.id,
                        "height": node_height,
                        "span_start": node.span_start,
                        "span_end": node.span_end,
                    }
                )

        assert (
            len(corrupt_nodes) == 0
        ), f"Found {len(corrupt_nodes)} nodes with invalid spans"

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(4.0)
    async def test_wraparound_pairing(
        self,
        storage_backend: StorageBackend,
        vector_index: _VectorIndexProtocol,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that demonstrates wraparound pairing issue."""
        _, document_id, mock_client = self.setup_system(
            storage_backend,
            vector_index,
            indexer_runtime_harness,
        )
        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)

        base_text = "The quick brown fox jumps over the lazy dog. " * 20
        chunks = [f"CHUNK_{i}_START {base_text} CHUNK_{i}_END" for i in range(5)]
        text = " ".join(chunks)

        def async_embedding_side_effect(**kwargs: object) -> Mock:
            from types import SimpleNamespace

            input_data = kwargs.get("input", [])
            if isinstance(input_data, str):
                input_data = [input_data]
            input_list = cast(list[str], input_data)
            num_items = len(input_list)
            return Mock(
                data=[Mock(embedding=[0.1] * 1536) for _ in input_list],
                usage=SimpleNamespace(
                    prompt_tokens=num_items * 10, total_tokens=num_items * 10
                ),
            )

        mock_client.embeddings.create = AsyncMock(
            side_effect=async_embedding_side_effect
        )
        mock_client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="Summary of the content"))],
                usage=Mock(
                    prompt_tokens=100,
                    completion_tokens=10,
                    prompt_tokens_details=Mock(cached_tokens=0),
                ),
            )
        )

        await indexer_runtime_harness.append(
            document_id,
            text,
            replace_existing=True,
            file_path="test_span_corruption.txt",
        )
        await indexer_runtime_harness.wait_for_idle(document_id)

        nodes = cast(Sequence[TreeNode], doc_store.nodes.get_all())
        nodes_by_id = {node.id: node for node in nodes}

        nodes_by_height: dict[int, list[TreeNode]] = {}
        for node in nodes:
            height = node.height
            nodes_by_height.setdefault(height, []).append(node)

        for height in nodes_by_height:
            nodes_by_height[height].sort(key=lambda node: node.span_start)

        root_height = max(nodes_by_height.keys())
        assert len(nodes_by_height[root_height]) == 1

        for height in range(root_height):
            level_nodes = nodes_by_height.get(height, [])
            for node in level_nodes:
                left_id = node.left_child_id
                right_id = node.right_child_id
                if not left_id:
                    continue
                left = nodes_by_id.get(left_id)
                assert left is not None
                if right_id:
                    right = nodes_by_id.get(right_id)
                    assert right is not None
                    assert node.span_start == left.span_start
                    assert node.span_end == right.span_end
                else:
                    assert node.span_start == left.span_start
                    assert node.span_end >= left.span_end

        leaves = nodes_by_height.get(0, [])
        assert len(leaves) >= 5
        for i, leaf in enumerate(leaves[:-1]):
            assert leaf.span_end <= leaves[i + 1].span_start
