"""Tests for following_neighbor_id column and bidirectional neighbor relationships."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.models import PostgresTreeNode as ORMTreeNode
from ragzoom.splitter import TextSplitter
from tests.conftest import BackwardCompatibilityConfig, IndexerRuntimeHarness


def _mock_sync_embeddings_create(**kwargs: object) -> Mock:
    """Mock for sync OpenAI client embeddings used by EmbeddingService."""
    input_texts = cast(list[str] | str, kwargs.get("input", []))
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    return Mock(data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_texts])


def _configure_runtime(harness: IndexerRuntimeHarness, config: IndexConfig) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.indexing_engine._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config


class TestFollowingNeighbor:
    """Test following_neighbor_id column and relationships."""

    def test_tree_node_has_following_neighbor_id_column(self) -> None:
        """TreeNode model should have following_neighbor_id column."""
        assert hasattr(ORMTreeNode, "following_neighbor_id")
        column = ORMTreeNode.__table__.columns.get("following_neighbor_id")
        assert column is not None
        assert column.nullable is True

    def test_bidirectional_neighbor_consistency(
        self, base_config: BackwardCompatibilityConfig, storage_backend: StorageBackend
    ) -> None:
        """Verify bidirectional consistency: if A.following = B, then B.preceding = A."""
        nodes_data: list[NodeDataDict] = []
        node_ids = ["node1", "node2", "node3", "node4"]

        for i, node_id in enumerate(node_ids):
            nodes_data.append(
                {
                    "node_id": node_id,
                    "text": f"Node {i} text",
                    "span_start": i * 100,
                    "span_end": (i + 1) * 100,
                    "document_id": "test-doc",
                    "height": 0,
                    "token_count": 10,
                    "level_index": i,
                    "preceding_neighbor_id": node_ids[i - 1] if i > 0 else None,
                    "following_neighbor_id": (
                        node_ids[i + 1] if i < len(node_ids) - 1 else None
                    ),
                }
            )

        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        created_nodes = doc_store.nodes.add_batch(nodes_data)
        nodes_by_id = {node.id: node for node in created_nodes}

        for node in created_nodes:
            if node.following_neighbor_id:
                following = nodes_by_id.get(node.following_neighbor_id)
                assert following is not None
                assert following.preceding_neighbor_id == node.id
            if node.preceding_neighbor_id:
                preceding = nodes_by_id.get(node.preceding_neighbor_id)
                assert preceding is not None
                assert preceding.following_neighbor_id == node.id

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(4.0)
    async def test_leaf_nodes_have_correct_neighbor_relationships(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Leaf nodes generated during indexing should have correct neighbors."""
        document_id = "neighbor-test"
        config = base_config.index_config.replace(target_chunk_tokens=5)
        _configure_runtime(indexer_runtime_harness, config)

        storage_backend.clear_document(document_id)
        await indexer_runtime_harness.clear(document_id)

        test_doc = "First chunk. Second chunk. Third chunk. Fourth chunk."
        engine_client = indexer_runtime_harness.indexing_engine._openai_client
        with patch.object(
            engine_client.embeddings, "create", new=_mock_sync_embeddings_create
        ):
            await indexer_runtime_harness.append(
                document_id,
                test_doc,
                replace_existing=True,
                file_path="test.txt",
            )

        doc_store = indexer_runtime_harness.runtime._store.for_document(document_id)
        leaf_nodes = [node for node in doc_store.nodes.get_all() if node.height == 0]
        leaf_nodes.sort(key=lambda n: n.span_start)

        for i, node in enumerate(leaf_nodes):
            if i == 0:
                assert node.preceding_neighbor_id is None
                if len(leaf_nodes) > 1:
                    assert node.following_neighbor_id == leaf_nodes[i + 1].id
            elif i == len(leaf_nodes) - 1:
                assert node.following_neighbor_id is None
                assert node.preceding_neighbor_id == leaf_nodes[i - 1].id
            else:
                assert node.preceding_neighbor_id == leaf_nodes[i - 1].id
                assert node.following_neighbor_id == leaf_nodes[i + 1].id

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(6.0)
    async def test_parent_nodes_have_correct_neighbor_relationships(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Parent nodes at each level should maintain neighbor relationships."""
        document_id = "parent-neighbor-test"
        config = base_config.index_config.replace(target_chunk_tokens=3)
        _configure_runtime(indexer_runtime_harness, config)

        storage_backend.clear_document(document_id)
        await indexer_runtime_harness.clear(document_id)

        test_doc = " ".join([f"Chunk {i}." for i in range(8)])
        engine_client = indexer_runtime_harness.indexing_engine._openai_client
        with patch.object(
            engine_client.embeddings, "create", new=_mock_sync_embeddings_create
        ):
            await indexer_runtime_harness.append(
                document_id,
                test_doc,
                replace_existing=True,
                file_path="test.txt",
            )

        doc_store = indexer_runtime_harness.runtime._store.for_document(document_id)
        nodes_by_height: dict[int, list[TreeNode]] = {}
        for node in doc_store.nodes.get_all():
            nodes_by_height.setdefault(node.height, []).append(node)

        for height, nodes in nodes_by_height.items():
            nodes.sort(key=lambda n: n.span_start)
            for idx, node in enumerate(nodes):
                if height == 0:
                    continue
                if idx == 0:
                    assert node.preceding_neighbor_id is None
                else:
                    assert node.preceding_neighbor_id == nodes[idx - 1].id
                if idx == len(nodes) - 1:
                    assert node.following_neighbor_id is None
                else:
                    assert node.following_neighbor_id == nodes[idx + 1].id
