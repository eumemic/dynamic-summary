"""Test that preceding_neighbor_id is correctly tracked during indexing."""

import asyncio

import pytest

from ragzoom.index import TreeBuilder


class TestPrecedingNeighborTracking:
    """Tests for preceding_neighbor_id field tracking during indexing."""

    @pytest.mark.parametrize("store_type", ["mock", "real"])
    def test_leaf_nodes_track_preceding_neighbor(
        self, request, base_config, mock_openai_async_client, store_type
    ):
        """Test that leaf nodes correctly track their preceding neighbor."""
        # Get the appropriate store based on parameter
        store = request.getfixturevalue(f"{store_type}_store")

        # Skip if real store not available (PostgreSQL not running)
        if store is None:
            pytest.skip("PostgreSQL not available for real store test")

        config = base_config.index_config

        # Create test document with clear chunk boundaries
        test_chunks = [
            f"Chunk {i}: This is content for chunk number {i}. " * 10 for i in range(5)
        ]
        test_document = "\n\n".join(test_chunks)

        # Create document-scoped store and tree builder
        doc_store = store.for_document("test-doc")
        doc_store.ensure_exists()  # Create document record for tree operations
        tree_builder = TreeBuilder(config, doc_store, max_concurrent=5)
        tree_builder.llm_service.client = mock_openai_async_client

        # Index the document
        asyncio.run(
            tree_builder.add_document_async(test_document, document_id="test-doc")
        )

        # Get all leaf nodes for the document
        doc_store = store.for_document("test-doc")
        leaf_nodes = doc_store.nodes.get_leaves()

        # Sort by span_start to get document order
        leaf_nodes.sort(key=lambda n: n.span_start)

        # Verify preceding_neighbor_id is set correctly
        for i, node in enumerate(leaf_nodes):
            if i == 0:
                # First node should have no preceding neighbor
                assert (
                    node.preceding_neighbor_id is None
                ), f"First leaf node {node.id} should have no preceding neighbor"
            else:
                # Each node should point to the previous node
                expected_preceding = leaf_nodes[i - 1].id
                assert node.preceding_neighbor_id == expected_preceding, (
                    f"Node {node.id} should have preceding_neighbor_id={expected_preceding}, "
                    f"but got {node.preceding_neighbor_id}"
                )

    @pytest.mark.parametrize("store_type", ["mock", "real"])
    def test_internal_nodes_track_preceding_neighbor(
        self, request, base_config, mock_openai_async_client, store_type
    ):
        """Test that internal nodes at each tree level track their preceding neighbor."""
        # Get the appropriate store based on parameter
        store = request.getfixturevalue(f"{store_type}_store")

        # Skip if real store not available (PostgreSQL not running)
        if store is None:
            pytest.skip("PostgreSQL not available for real store test")
        config = base_config.index_config

        # Create test document that will create multiple tree levels
        # Need at least 4 chunks to get a tree with height > 1
        test_chunks = [
            f"Chunk {i}: This is content for chunk number {i}. " * 20
            for i in range(8)  # 8 chunks will create 3 levels
        ]
        test_document = "\n\n".join(test_chunks)

        # Create document-scoped store and tree builder
        doc_store = store.for_document("test-doc")
        doc_store.ensure_exists()  # Create document record for tree operations
        tree_builder = TreeBuilder(config, doc_store, max_concurrent=5)
        tree_builder.llm_service.client = mock_openai_async_client

        # Index the document
        asyncio.run(
            tree_builder.add_document_async(test_document, document_id="test-doc")
        )

        # Get all nodes (different approach for real vs mock store)
        if hasattr(store, "get_all_nodes"):
            all_nodes = store.get_all_nodes()
        else:
            # For real store, get nodes from document
            doc_store = store.for_document("test-doc")
            all_nodes = []
            # Get leaf nodes
            leaf_nodes = doc_store.nodes.get_leaves()
            all_nodes.extend(leaf_nodes)
            # Get their ancestors
            if leaf_nodes:
                leaf_ids = [n.id for n in leaf_nodes]
                if hasattr(store, "tree"):
                    # For real store
                    ancestors = store.tree.get_ancestors(leaf_ids)
                else:
                    # For mock store
                    ancestors = store.get_ancestors(leaf_ids)
                all_nodes.extend(ancestors)

        # Group nodes by height (leaf nodes have no children)
        nodes_by_height = {}
        for node in all_nodes:
            height = _calculate_node_height(node, all_nodes)
            if height not in nodes_by_height:
                nodes_by_height[height] = []
            nodes_by_height[height].append(node)

        # Verify each level
        for height, nodes in nodes_by_height.items():
            # Sort by span_start to get document order
            nodes.sort(key=lambda n: n.span_start)

            for i, node in enumerate(nodes):
                if i == 0:
                    # First node at each level should have no preceding neighbor
                    assert (
                        node.preceding_neighbor_id is None
                    ), f"First node {node.id} at height {height} should have no preceding neighbor"
                else:
                    # Each node should point to the previous node at the same level
                    expected_preceding = nodes[i - 1].id
                    assert node.preceding_neighbor_id == expected_preceding, (
                        f"Node {node.id} at height {height} should have preceding_neighbor_id={expected_preceding}, "
                        f"but got {node.preceding_neighbor_id}"
                    )

    @pytest.mark.parametrize("store_type", ["mock", "real"])
    def test_preceding_context_reconstruction(
        self, request, base_config, mock_openai_async_client, store_type
    ):
        """Test that we can reconstruct preceding context using preceding_neighbor_id."""
        # Get the appropriate store based on parameter
        store = request.getfixturevalue(f"{store_type}_store")

        # Skip if real store not available (PostgreSQL not running)
        if store is None:
            pytest.skip("PostgreSQL not available for real store test")

        config = base_config.index_config

        # Create test document with clear markers
        test_chunks = [
            f"START_CHUNK_{i} Content for chunk {i} END_CHUNK_{i}" for i in range(6)
        ]
        test_document = " ".join(test_chunks)

        # Create document-scoped store and tree builder
        doc_store = store.for_document("test-doc")
        doc_store.ensure_exists()  # Create document record for tree operations
        tree_builder = TreeBuilder(config, doc_store, max_concurrent=5)
        tree_builder.llm_service.client = mock_openai_async_client

        # Index the document
        asyncio.run(
            tree_builder.add_document_async(test_document, document_id="test-doc")
        )

        # Get leaf nodes for the document
        if hasattr(store, "get_leaf_nodes"):
            # Mock store
            leaf_nodes = store.get_leaf_nodes()
        else:
            # Real store - use document store
            doc_store = store.for_document("test-doc")
            leaf_nodes = doc_store.nodes.get_leaves()
        leaf_nodes.sort(key=lambda n: n.span_start)

        # For each node (except the first), verify we can get its preceding context
        for i, node in enumerate(leaf_nodes):
            if i > 0:
                # Get the preceding node
                if hasattr(store, "get_node"):
                    # Mock store
                    preceding_node = store.nodes.get_node(node.preceding_neighbor_id)
                else:
                    # Real store - use nodes repository
                    preceding_node = store.nodes.get_node(node.preceding_neighbor_id)
                assert (
                    preceding_node is not None
                ), f"Should be able to retrieve preceding node for {node.id}"

                # The preceding node's text should appear before this node's text in the document
                assert (
                    preceding_node.span_end <= node.span_start
                ), "Preceding node should end before current node starts"


def _calculate_node_height(node, all_nodes):
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
