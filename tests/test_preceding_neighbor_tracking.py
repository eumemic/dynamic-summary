"""Test that preceding_neighbor_id is correctly tracked during indexing."""

import asyncio

from openai import AsyncOpenAI

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.index import TreeBuilder
from ragzoom.models import TreeNode
from tests.conftest import BackwardCompatibilityConfig


class TestPrecedingNeighborTracking:
    """Tests for preceding_neighbor_id field tracking during indexing."""

    def test_leaf_nodes_track_preceding_neighbor(
        self,
        storage_backend: StorageBackend,
        base_config: BackwardCompatibilityConfig,
        mock_openai_async_client: AsyncOpenAI,
    ) -> None:
        """Test that leaf nodes correctly track their preceding neighbor."""
        config = base_config.index_config

        # Create test document with clear chunk boundaries
        test_chunks = [
            f"Chunk {i}: This is content for chunk number {i}. " * 10 for i in range(5)
        ]
        test_document = "\n\n".join(test_chunks)

        # Create document-scoped store and ensure metadata exists
        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path=None,
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        tree_builder = TreeBuilder(config, doc_store, max_concurrent=5)
        tree_builder.llm_service.client = mock_openai_async_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document))

        # Get all leaf nodes for the document
        leaf_nodes = doc_store.nodes.get_leaves()

        # Sort by span_start to get document order
        leaf_nodes.sort(key=lambda n: n.span_start)

        # Verify preceding_neighbor_id is set correctly
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

    def test_internal_nodes_track_preceding_neighbor(
        self,
        storage_backend: StorageBackend,
        base_config: BackwardCompatibilityConfig,
        mock_openai_async_client: AsyncOpenAI,
    ) -> None:
        """Test that internal nodes at each tree level track their preceding neighbor."""
        config = base_config.index_config

        # Create test document that will create multiple tree levels
        test_chunks = [
            f"Chunk {i}: This is content for chunk number {i}. " * 20 for i in range(8)
        ]
        test_document = "\n\n".join(test_chunks)

        # Create document-scoped store and ensure metadata exists
        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path=None,
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        tree_builder = TreeBuilder(config, doc_store, max_concurrent=5)
        tree_builder.llm_service.client = mock_openai_async_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document))

        # Collect leaves and their ancestors
        all_nodes: list[TreeNode] = []
        leaf_nodes = doc_store.nodes.get_leaves()
        all_nodes.extend(leaf_nodes)
        if leaf_nodes:
            leaf_ids = [n.id for n in leaf_nodes]
            ancestors = doc_store.tree.get_ancestors(leaf_ids)
            all_nodes.extend(ancestors)

        # Group nodes by height (leaf nodes have no children)
        nodes_by_height: dict[int, list[TreeNode]] = {}
        for node in all_nodes:
            height = _calculate_node_height(node, all_nodes)
            nodes_by_height.setdefault(height, []).append(node)

        # Verify each level
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

    def test_preceding_context_reconstruction(
        self,
        storage_backend: StorageBackend,
        base_config: BackwardCompatibilityConfig,
        mock_openai_async_client: AsyncOpenAI,
    ) -> None:
        """Test that we can reconstruct preceding context using preceding_neighbor_id."""
        config = base_config.index_config

        # Create test document with clear markers
        test_chunks = [
            f"START_CHUNK_{i} Content for chunk {i} END_CHUNK_{i}" for i in range(6)
        ]
        test_document = " ".join(test_chunks)

        # Create document-scoped store and ensure metadata exists
        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path=None,
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        tree_builder = TreeBuilder(config, doc_store, max_concurrent=5)
        tree_builder.llm_service.client = mock_openai_async_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_document))

        # Get leaf nodes for the document
        leaf_nodes = doc_store.nodes.get_leaves()
        leaf_nodes.sort(key=lambda n: n.span_start)

        # For each node (except the first), verify we can get its preceding context
        for i, node in enumerate(leaf_nodes):
            if i > 0:
                prev_id = node.preceding_neighbor_id
                assert prev_id is not None
                preceding_node = doc_store.nodes.get_node(prev_id)
                assert (
                    preceding_node is not None
                ), f"Should be able to retrieve preceding node for {node.id}"

                # The preceding node's text should appear before this node's text in the document
                assert (
                    preceding_node.span_end <= node.span_start
                ), "Preceding node should end before current node starts"


def _calculate_node_height(node: TreeNode, all_nodes: list[TreeNode]) -> int:
    """Calculate the height of a node in the tree."""
    # Create a mapping of node IDs to nodes
    node_map = {n.id: n for n in all_nodes}

    # Leaf nodes have height 0
    if not hasattr(node, "left_child_id") or node.left_child_id is None:
        return 0

    # Calculate height recursively
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
