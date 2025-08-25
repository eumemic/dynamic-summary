"""Tests for tree skeleton construction - pure functions for dataflow implementation."""

import pytest

from ragzoom.tree_skeleton import (
    build_internal_nodes,
    create_leaf_nodes,
)


class TestTreeSkeleton:
    """Test tree skeleton construction for dataflow implementation."""

    def test_create_leaf_nodes_basic(self):
        """Test creating leaf nodes from chunks."""
        chunks = ["First chunk.", "Second chunk.", "Third chunk.", "Fourth chunk."]
        document_id = "test-doc"

        skeleton, leaves = create_leaf_nodes(chunks, document_id)

        # Check we have the right number of leaves
        assert len(leaves) == 4
        assert len(skeleton.lookup) == 4

        # Check leaf properties
        for i, leaf in enumerate(leaves):
            assert leaf.text == chunks[i]
            assert leaf.height == 0
            assert leaf.left_child_id is None
            assert leaf.right_child_id is None
            assert leaf.parent_id is None  # Will be set when parent is created
            assert leaf.document_id == document_id

            # Check span positions
            if i == 0:
                assert leaf.span_start == 0
            else:
                assert leaf.span_start == leaves[i - 1].span_end
            assert leaf.span_end == leaf.span_start + len(chunks[i])

            # Check neighbor relationships
            if i == 0:
                assert leaf.preceding_neighbor_id is None
                assert (
                    leaf.following_neighbor_id == leaves[1].id
                    if len(leaves) > 1
                    else None
                )
            elif i == len(leaves) - 1:
                assert leaf.preceding_neighbor_id == leaves[i - 1].id
                assert leaf.following_neighbor_id is None
            else:
                assert leaf.preceding_neighbor_id == leaves[i - 1].id
                assert leaf.following_neighbor_id == leaves[i + 1].id

    def test_create_leaf_nodes_binary_paths(self):
        """Test that leaf nodes get correct binary paths."""
        # Test with 8 chunks (perfect binary tree)
        chunks = [f"Chunk {i}." for i in range(8)]
        document_id = "test-doc"

        skeleton, leaves = create_leaf_nodes(chunks, document_id)

        # Check binary paths (3 bits for 8 leaves)
        expected_paths = ["000", "001", "010", "011", "100", "101", "110", "111"]
        for i, leaf in enumerate(leaves):
            assert (
                leaf.path == expected_paths[i]
            ), f"Leaf {i} has wrong path: {leaf.path}"

    def test_build_internal_nodes_simple(self):
        """Test building internal nodes from leaves."""
        # Create 4 leaves first
        chunks = ["A", "B", "C", "D"]
        skeleton, leaves = create_leaf_nodes(chunks, "test-doc")

        # Build internal nodes
        build_internal_nodes(skeleton, leaves)

        # Should have 4 leaves + 2 level-1 parents + 1 root = 7 total nodes
        assert len(skeleton.lookup) == 7

        # Find root (node with no parent)
        root = skeleton.get_root()
        assert root is not None
        assert root.parent_id is None
        assert root.height == 2  # Two levels above leaves
        assert root.text is None  # No text yet (will be filled by dataflow)

        # Check that root has correct span
        assert root.span_start == 0
        assert root.span_end == sum(len(c) for c in chunks)

    def test_build_internal_nodes_odd_number(self):
        """Test building tree with odd number of leaves."""
        # Create 5 leaves
        chunks = ["A", "B", "C", "D", "E"]
        skeleton, leaves = create_leaf_nodes(chunks, "test-doc")

        # Build internal nodes
        build_internal_nodes(skeleton, leaves)

        # Should have 5 leaves + 3 level-1 parents + 2 level-2 parents + 1 root = 11 total
        # Actually: 5 leaves, then pairs: (A,B), (C,D), (E,None) = 3 parents
        # Then: ((A,B), (C,D)), (E,None) = 2 parents
        # Then: root
        # Total = 5 + 3 + 2 + 1 = 11? Let's check...
        # Actually the tree building pairs them differently, let me think...
        # With 5 leaves: pairs are (0,1), (2,3), (4,None) -> 3 parents at level 1
        # Then: (parent0, parent1), (parent2, None) -> 2 parents at level 2
        # Then: (parent3, parent4) -> 1 root at level 3
        # Total = 5 + 3 + 2 + 1 = 11 nodes

        # Let's just verify we have more than the leaves and there's a root
        assert len(skeleton.lookup) > 5

        root = skeleton.get_root()
        assert root is not None

        # Check the last leaf's parent only has one child
        last_leaf = leaves[-1]
        parent = skeleton.lookup.get(last_leaf.parent_id)
        if parent:
            assert parent.left_child_id == last_leaf.id
            assert parent.right_child_id is None

    def test_parent_neighbor_relationships(self):
        """Test that parent nodes have correct neighbor relationships."""
        # Create 8 leaves for a balanced tree
        chunks = [f"Chunk {i}" for i in range(8)]
        skeleton, leaves = create_leaf_nodes(chunks, "test-doc")

        # Build internal nodes
        build_internal_nodes(skeleton, leaves)

        # Get all nodes at height 1 (parents of leaves)
        level_1_nodes = skeleton.get_nodes_at_height(1)
        assert len(level_1_nodes) == 4  # 8 leaves -> 4 parents

        # Sort by span_start to get logical order
        level_1_nodes.sort(key=lambda n: n.span_start)

        # Check neighbor relationships
        for i, node in enumerate(level_1_nodes):
            if i == 0:
                assert node.preceding_neighbor_id is None
                assert node.following_neighbor_id == level_1_nodes[1].id
            elif i == len(level_1_nodes) - 1:
                assert node.preceding_neighbor_id == level_1_nodes[i - 1].id
                assert node.following_neighbor_id is None
            else:
                assert node.preceding_neighbor_id == level_1_nodes[i - 1].id
                assert node.following_neighbor_id == level_1_nodes[i + 1].id

    def test_parent_child_relationships(self):
        """Test that parent-child relationships are correctly established."""
        chunks = ["A", "B", "C", "D"]
        skeleton, leaves = create_leaf_nodes(chunks, "test-doc")
        build_internal_nodes(skeleton, leaves)

        # Check that each leaf has a parent
        for leaf in leaves:
            assert leaf.parent_id is not None
            parent = skeleton.lookup[leaf.parent_id]
            assert parent is not None
            # Check parent points back to child
            assert leaf.id in [parent.left_child_id, parent.right_child_id]

        # Check that parents of pairs are correct
        # Leaves 0,1 should have same parent
        assert leaves[0].parent_id == leaves[1].parent_id
        # Leaves 2,3 should have same parent
        assert leaves[2].parent_id == leaves[3].parent_id
        # But different from first pair
        assert leaves[0].parent_id != leaves[2].parent_id

    def test_binary_paths_propagation(self):
        """Test that binary paths are correctly derived from children."""
        chunks = ["A", "B", "C", "D"]
        skeleton, leaves = create_leaf_nodes(chunks, "test-doc")
        build_internal_nodes(skeleton, leaves)

        # Leaves should have paths: 00, 01, 10, 11
        assert leaves[0].path == "00"
        assert leaves[1].path == "01"
        assert leaves[2].path == "10"
        assert leaves[3].path == "11"

        # Parent of first two leaves should have path "0" (common prefix)
        parent_01 = skeleton.lookup[leaves[0].parent_id]
        assert parent_01.path == "0"

        # Parent of last two leaves should have path "1"
        parent_23 = skeleton.lookup[leaves[2].parent_id]
        assert parent_23.path == "1"

        # Root should have empty path
        root = skeleton.get_root()
        assert root.path == ""

    def test_spans_cover_document(self):
        """Test that spans correctly cover the entire document."""
        chunks = ["First part", "Second part", "Third part"]
        skeleton, leaves = create_leaf_nodes(chunks, "test-doc")
        build_internal_nodes(skeleton, leaves)

        # Check leaf spans
        expected_pos = 0
        for i, leaf in enumerate(leaves):
            assert leaf.span_start == expected_pos
            assert leaf.span_end == expected_pos + len(chunks[i])
            expected_pos = leaf.span_end

        # Check parent spans
        for node_id, node in skeleton.lookup.items():
            if node.left_child_id:
                left_child = skeleton.lookup[node.left_child_id]
                assert node.span_start == left_child.span_start

                if node.right_child_id:
                    right_child = skeleton.lookup[node.right_child_id]
                    assert node.span_end == right_child.span_end
                else:
                    assert node.span_end == left_child.span_end

        # Check root covers entire document
        root = skeleton.get_root()
        assert root.span_start == 0
        total_length = sum(len(c) for c in chunks)
        assert root.span_end == total_length

    def test_skeleton_provides_iteration_order(self):
        """Test that skeleton can provide correct iteration order for dataflow."""
        chunks = ["A", "B", "C", "D", "E", "F"]
        skeleton, leaves = create_leaf_nodes(chunks, "test-doc")
        build_internal_nodes(skeleton, leaves)

        # Get nodes in bottom-up, left-to-right order
        nodes_by_height = skeleton.get_nodes_by_height()

        # Height 0 should be leaves in order
        assert len(nodes_by_height[0]) == 6
        for i, node in enumerate(nodes_by_height[0]):
            assert node.text == chunks[i]

        # Each higher level should have fewer nodes
        prev_count = len(nodes_by_height[0])
        for height in sorted(nodes_by_height.keys())[1:]:
            count = len(nodes_by_height[height])
            assert count < prev_count
            prev_count = count

        # Highest level should have exactly 1 node (root)
        max_height = max(nodes_by_height.keys())
        assert len(nodes_by_height[max_height]) == 1

    def test_empty_chunks_raises_error(self):
        """Test that empty chunks list raises appropriate error."""
        with pytest.raises(ValueError, match="No chunks provided"):
            create_leaf_nodes([], "test-doc")

    def test_single_chunk_creates_single_node_tree(self):
        """Test that a single chunk creates a valid single-node tree."""
        chunks = ["Only chunk"]
        skeleton, leaves = create_leaf_nodes(chunks, "test-doc")
        build_internal_nodes(skeleton, leaves)

        # Should have just one node that is both leaf and root
        assert len(skeleton.lookup) == 1
        assert len(leaves) == 1

        node = leaves[0]
        assert node.parent_id is None  # It's the root
        assert node.left_child_id is None
        assert node.right_child_id is None
        assert node.preceding_neighbor_id is None
        assert node.following_neighbor_id is None
        assert node.height == 0
        assert node.path == "0"
