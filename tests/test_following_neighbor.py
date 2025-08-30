"""Tests for following_neighbor_id column and bidirectional neighbor relationships."""

from typing import Any

from openai import AsyncOpenAI

from ragzoom.models import TreeNode
from ragzoom.store import StoreManager


class TestFollowingNeighbor:
    """Test following_neighbor_id column and relationships."""

    def test_tree_node_has_following_neighbor_id_column(self) -> None:
        """TreeNode model should have following_neighbor_id column."""
        # Verify the column exists on the model
        assert hasattr(TreeNode, "following_neighbor_id")

        # Verify it's a mapped column
        column = TreeNode.__table__.columns.get("following_neighbor_id")
        assert column is not None
        assert (
            column.nullable is True
        )  # Should be nullable (last node has no following)

    def test_bidirectional_neighbor_consistency(
        self, base_config: Any, store: StoreManager
    ) -> None:
        """Verify bidirectional consistency: if A.following = B, then B.preceding = A."""
        # Create some test nodes with neighbor relationships
        nodes_data = []
        node_ids = ["node1", "node2", "node3", "node4"]

        for i, node_id in enumerate(node_ids):
            node_data = {
                "node_id": node_id,
                "text": f"Node {i} text",
                "embedding": [0.1] * 10,  # Simple embedding
                "span_start": i * 100,
                "span_end": (i + 1) * 100,
                "document_id": "test-doc",
                "height": 0,  # Leaf nodes
                "path": format(i, "02b"),  # Binary path
                "preceding_neighbor_id": node_ids[i - 1] if i > 0 else None,
                "following_neighbor_id": (
                    node_ids[i + 1] if i < len(node_ids) - 1 else None
                ),
            }
            nodes_data.append(node_data)

        # Create document with proper metadata and get document store
        doc_store = store.add_document(
            document_id="test-doc",
            file_path=None,
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        created_nodes = doc_store.nodes.add_batch(nodes_data)

        # Verify bidirectional consistency
        nodes_by_id = {node.id: node for node in created_nodes}

        for node in created_nodes:
            # If this node has a following neighbor
            if node.following_neighbor_id:
                following = nodes_by_id.get(node.following_neighbor_id)
                assert (
                    following is not None
                ), f"Following neighbor {node.following_neighbor_id} not found"
                # The following node's preceding should point back to this node
                assert following.preceding_neighbor_id == node.id, (
                    f"Bidirectional inconsistency: {node.id}.following={node.following_neighbor_id}, "
                    f"but {following.id}.preceding={following.preceding_neighbor_id}"
                )

            # If this node has a preceding neighbor
            if node.preceding_neighbor_id:
                preceding = nodes_by_id.get(node.preceding_neighbor_id)
                assert (
                    preceding is not None
                ), f"Preceding neighbor {node.preceding_neighbor_id} not found"
                # The preceding node's following should point to this node
                assert preceding.following_neighbor_id == node.id, (
                    f"Bidirectional inconsistency: {node.id}.preceding={node.preceding_neighbor_id}, "
                    f"but {preceding.id}.following={preceding.following_neighbor_id}"
                )

    def test_leaf_nodes_have_correct_neighbor_relationships(
        self,
        base_config: Any,
        store: StoreManager,
        mock_openai_async_client: AsyncOpenAI,
    ) -> None:
        """Test that leaf nodes created during indexing have correct neighbor relationships."""
        import asyncio

        from ragzoom.index import TreeBuilder

        # Create a simple test document
        test_doc = "First chunk. Second chunk. Third chunk. Fourth chunk."

        # Create tree builder with small chunk size to ensure multiple chunks
        config = base_config.index_config.replace(target_chunk_tokens=5)
        # Create document with proper metadata
        doc_store = store.add_document(
            document_id="neighbor-test",
            file_path=None,
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        tree_builder = TreeBuilder(config, doc_store)
        tree_builder.llm_service.client = mock_openai_async_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_doc))

        # Get all leaf nodes (height=0)
        all_nodes = doc_store.nodes.get_all()
        leaf_nodes = [node for node in all_nodes if node.height == 0]

        # Sort by span_start to get document order
        leaf_nodes.sort(key=lambda n: n.span_start)

        # Verify neighbor relationships
        for i, node in enumerate(leaf_nodes):
            if i == 0:
                # First node has no preceding neighbor
                assert (
                    node.preceding_neighbor_id is None
                ), f"First leaf should have no preceding neighbor, got {node.preceding_neighbor_id}"
                # But should have following (unless it's the only node)
                if len(leaf_nodes) > 1:
                    assert (
                        node.following_neighbor_id == leaf_nodes[i + 1].id
                    ), "First leaf's following should be second leaf"
            elif i == len(leaf_nodes) - 1:
                # Last node has no following neighbor
                assert (
                    node.following_neighbor_id is None
                ), f"Last leaf should have no following neighbor, got {node.following_neighbor_id}"
                # But should have preceding
                assert (
                    node.preceding_neighbor_id == leaf_nodes[i - 1].id
                ), "Last leaf's preceding should be previous leaf"
            else:
                # Middle nodes have both
                assert (
                    node.preceding_neighbor_id == leaf_nodes[i - 1].id
                ), f"Leaf {i} preceding should be leaf {i-1}"
                assert (
                    node.following_neighbor_id == leaf_nodes[i + 1].id
                ), f"Leaf {i} following should be leaf {i+1}"

    def test_parent_nodes_have_correct_neighbor_relationships(
        self,
        base_config: Any,
        store: StoreManager,
        mock_openai_async_client: AsyncOpenAI,
    ) -> None:
        """Test that parent nodes at each level have correct neighbor relationships."""
        import asyncio

        from ragzoom.index import TreeBuilder

        # Create a document that will create multiple levels
        test_doc = " ".join([f"Chunk {i}." for i in range(8)])  # 8 chunks -> 3 levels

        # Create tree builder with small chunk size
        config = base_config.index_config.replace(
            target_chunk_tokens=3
        )  # Create document with proper metadata
        doc_store = store.add_document(
            document_id="parent-neighbor-test",
            file_path=None,
            content_hash="test-hash",
            chunk_count=0,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        tree_builder = TreeBuilder(config, doc_store)
        tree_builder.llm_service.client = mock_openai_async_client

        # Index the document
        asyncio.run(tree_builder.add_document_async(test_doc))

        # Get all nodes and group by height
        all_nodes = doc_store.nodes.get_all()

        nodes_by_height: dict[int, list[Any]] = {}
        for node in all_nodes:
            if node.height not in nodes_by_height:
                nodes_by_height[node.height] = []
            nodes_by_height[node.height].append(node)

        # Check each level
        for height, level_nodes in nodes_by_height.items():
            # Sort by span_start to get logical order
            level_nodes.sort(key=lambda n: n.span_start)
            for i, node in enumerate(level_nodes):
                if i == 0:
                    assert (
                        node.preceding_neighbor_id is None
                    ), f"First node at height {height} should have no preceding"
                    if len(level_nodes) > 1:
                        assert (
                            node.following_neighbor_id == level_nodes[i + 1].id
                        ), f"First node at height {height} should point to second"
                elif i == len(level_nodes) - 1:
                    assert (
                        node.following_neighbor_id is None
                    ), f"Last node at height {height} should have no following"
                    assert (
                        node.preceding_neighbor_id == level_nodes[i - 1].id
                    ), f"Last node at height {height} should have preceding"
                else:
                    assert (
                        node.preceding_neighbor_id == level_nodes[i - 1].id
                    ), f"Node {i} at height {height} has wrong preceding"
                    assert (
                        node.following_neighbor_id == level_nodes[i + 1].id
                    ), f"Node {i} at height {height} has wrong following"
