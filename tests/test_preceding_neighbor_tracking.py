"""Test that preceding_neighbor_id is correctly tracked during indexing."""

from __future__ import annotations

import pytest
from openai import AsyncOpenAI

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.splitter import TextSplitter
from tests.conftest import BackwardCompatibilityConfig, IndexerRuntimeHarness


def _configure_runtime(harness: IndexerRuntimeHarness, config: IndexConfig) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config


class TestPrecedingNeighborTracking:
    """Tests for preceding_neighbor_id field tracking during indexing."""

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(2.0)
    async def test_leaf_nodes_track_preceding_neighbor(
        self,
        storage_backend: StorageBackend,
        base_config: BackwardCompatibilityConfig,
        mock_openai_async_client: AsyncOpenAI,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that leaf nodes correctly track their preceding neighbor."""
        config = base_config.index_config
        document_id = "test-doc"

        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        _configure_runtime(indexer_runtime_harness, config)
        indexer_runtime_harness.llm_service.client = mock_openai_async_client

        test_chunks = [
            f"Chunk {i}: This is content for chunk number {i}. " * 10 for i in range(5)
        ]
        test_document = "\n\n".join(test_chunks)

        try:
            await indexer_runtime_harness.clear(document_id)
            await indexer_runtime_harness.append(
                document_id,
                test_document,
                replace_existing=True,
                file_path=None,
            )

            leaf_nodes = doc_store.nodes.get_leaves()
            leaf_nodes.sort(key=lambda n: n.span_start)

            for i, node in enumerate(leaf_nodes):
                if i == 0:
                    assert (
                        node.preceding_neighbor_id is None
                    ), f"First leaf node {node.id} should have no preceding neighbor"
                else:
                    expected_preceding = leaf_nodes[i - 1].id
                    assert node.preceding_neighbor_id == expected_preceding, (
                        f"Node {node.id} should have preceding_neighbor_id={expected_preceding}, "
                        f"but got {node.preceding_neighbor_id}"
                    )
        finally:
            await indexer_runtime_harness.clear(document_id)

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(2.0)
    async def test_internal_nodes_track_preceding_neighbor(
        self,
        storage_backend: StorageBackend,
        base_config: BackwardCompatibilityConfig,
        mock_openai_async_client: AsyncOpenAI,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that internal nodes at each tree level track their preceding neighbor."""
        config = base_config.index_config
        document_id = "test-doc"

        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        _configure_runtime(indexer_runtime_harness, config)
        indexer_runtime_harness.llm_service.client = mock_openai_async_client

        test_chunks = [
            f"Chunk {i}: This is content for chunk number {i}. " * 20 for i in range(8)
        ]
        test_document = "\n\n".join(test_chunks)

        try:
            await indexer_runtime_harness.clear(document_id)
            await indexer_runtime_harness.append(
                document_id,
                test_document,
                replace_existing=True,
                file_path=None,
            )

            all_nodes: list[TreeNode] = []
            leaf_nodes = doc_store.nodes.get_leaves()
            all_nodes.extend(leaf_nodes)
            if leaf_nodes:
                leaf_ids = [n.id for n in leaf_nodes]
                ancestors = doc_store.tree.get_ancestors(leaf_ids)
                all_nodes.extend(ancestors)

            nodes_by_height: dict[int, list[TreeNode]] = {}
            for node in all_nodes:
                height = _calculate_node_height(node, all_nodes)
                nodes_by_height.setdefault(height, []).append(node)

            for height, nodes in nodes_by_height.items():
                nodes.sort(key=lambda n: n.span_start)
                for i, node in enumerate(nodes):
                    if i == 0:
                        assert (
                            node.preceding_neighbor_id is None
                        ), f"First node {node.id} at height {height} should have no preceding neighbor"
                    else:
                        expected_preceding = nodes[i - 1].id
                        assert node.preceding_neighbor_id == expected_preceding, (
                            f"Node {node.id} at height {height} should have preceding_neighbor_id={expected_preceding}, "
                            f"but got {node.preceding_neighbor_id}"
                        )
        finally:
            await indexer_runtime_harness.clear(document_id)

    @pytest.mark.asyncio
    async def test_preceding_context_reconstruction(
        self,
        storage_backend: StorageBackend,
        base_config: BackwardCompatibilityConfig,
        mock_openai_async_client: AsyncOpenAI,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        """Test that we can reconstruct preceding context using preceding_neighbor_id."""
        config = base_config.index_config
        document_id = "test-doc"

        storage_backend.clear_document(document_id)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        _configure_runtime(indexer_runtime_harness, config)
        indexer_runtime_harness.llm_service.client = mock_openai_async_client

        test_chunks = [
            f"START_CHUNK_{i} Content for chunk {i} END_CHUNK_{i}" for i in range(6)
        ]
        test_document = " ".join(test_chunks)

        try:
            await indexer_runtime_harness.clear(document_id)
            await indexer_runtime_harness.append(
                document_id,
                test_document,
                replace_existing=True,
                file_path=None,
            )

            leaf_nodes = doc_store.nodes.get_leaves()
            leaf_nodes.sort(key=lambda n: n.span_start)

            for i, node in enumerate(leaf_nodes):
                if i == 0:
                    continue
                prev_id = node.preceding_neighbor_id
                assert prev_id is not None
                preceding_node = doc_store.nodes.get_node(prev_id)
                assert (
                    preceding_node is not None
                ), f"Should be able to retrieve preceding node for {node.id}"
                assert (
                    preceding_node.span_end <= node.span_start
                ), "Preceding node should end before current node starts"
        finally:
            await indexer_runtime_harness.clear(document_id)


def _calculate_node_height(node: TreeNode, all_nodes: list[TreeNode]) -> int:
    """Calculate the height of a node in the tree."""
    node_map = {n.id: n for n in all_nodes}

    if not hasattr(node, "left_child_id") or node.left_child_id is None:
        return 0

    left_height = 0
    right_height = 0

    if node.left_child_id and node.left_child_id in node_map:
        left_height = (
            _calculate_node_height(node_map[node.left_child_id], all_nodes) + 1
        )

    if (
        hasattr(node, "right_child_id")
        and node.right_child_id
        and node.right_child_id in node_map
    ):
        right_height = (
            _calculate_node_height(node_map[node.right_child_id], all_nodes) + 1
        )

    return max(left_height, right_height)
