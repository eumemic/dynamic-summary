"""Tests for DP assembly functionality.

Tests for the assembler's ability to reconstruct text from tiling results.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.assemble import Assembler
from ragzoom.config import OperationalConfig, QueryConfig, SecretStr
from ragzoom.document_store import DocumentStore
from tests.utils import create_retriever


@pytest.mark.usefixtures("sqlite_backend")
class TestDPAssembly:
    """Test the DP assembly path and core DP algorithm.

    This class contains both assembly tests (using node IDs) and
    core DP algorithm tests.
    """

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("doc1")

    @pytest.fixture
    def assembler(self, doc_store: DocumentStore) -> Assembler:
        """Create assembler with SQLite document store."""
        return Assembler(doc_store)

    @pytest.fixture
    def mock_nodes(self, doc_store: DocumentStore) -> None:
        """Create mock nodes in the SQLite store."""
        # Create a simple tree structure
        # Root (depth=2)
        #  / \
        # L   R (depth=1)
        # /\ /\
        # 1 2 3 4 (depth=0)

        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Leaf nodes
            {
                "node_id": "leaf1",
                "text": "First chunk of text.",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 20,
                "document_id": "doc1",
                "token_count": 20,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf2",
                "text": "Second chunk of text.",
                "embedding": np.array([0.2] * 1536, dtype=np.float64),
                "span_start": 20,
                "span_end": 41,
                "document_id": "doc1",
                "token_count": 21,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf3",
                "text": "Third chunk of text.",
                "embedding": np.array([0.3] * 1536, dtype=np.float64),
                "span_start": 41,
                "span_end": 61,
                "document_id": "doc1",
                "token_count": 20,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf4",
                "text": "Fourth chunk of text.",
                "embedding": np.array([0.4] * 1536, dtype=np.float64),
                "span_start": 61,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 21,
                "height": 0,
                "level_index": 0,
            },
            # Internal nodes
            {
                "node_id": "left",
                "text": "Summary of first and second chunks.",
                "embedding": np.array([0.15] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 41,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
                "document_id": "doc1",
                "token_count": 35,
                "height": 1,
                "level_index": 0,
            },
            {
                "node_id": "right",
                "text": "Summary of third and fourth chunks.",
                "embedding": np.array([0.35] * 1536, dtype=np.float64),
                "span_start": 41,
                "span_end": 82,
                "left_child_id": "leaf3",
                "right_child_id": "leaf4",
                "document_id": "doc1",
                "token_count": 36,
                "height": 1,
                "level_index": 0,
            },
            {
                "node_id": "root",
                "text": "Overall document summary.",
                "embedding": np.array([0.25] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 82,
                "left_child_id": "left",
                "right_child_id": "right",
                "document_id": "doc1",
                "token_count": 26,
                "height": 2,
                "level_index": 0,
            },
        ]
        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("leaf1", "left"),
                ("leaf2", "left"),
                ("leaf3", "right"),
                ("leaf4", "right"),
                ("left", "root"),
                ("right", "root"),
            ]
        )

    def test_basic_dp_assembly(self, assembler: Assembler, mock_nodes: None) -> None:
        """Test basic DP assembly with leaf nodes."""
        # List of node IDs
        tiling = ["leaf1", "leaf2"]

        result = assembler.assemble_dp(tiling)

        # Leaf nodes return full text
        assert result == "First chunk of text.\n\nSecond chunk of text."

    def test_internal_node_assembly(
        self, assembler: Assembler, mock_nodes: None
    ) -> None:
        """Test assembly with internal nodes (atomic units)."""
        tiling = ["left", "leaf3"]

        result = assembler.assemble_dp(tiling)

        # Internal nodes return their full summary
        assert result == "Summary of first and second chunks.\n\nThird chunk of text."

    def test_mixed_nodes_assembly(self, assembler: Assembler, mock_nodes: None) -> None:
        """Test assembly with mix of leaf and internal nodes."""
        tiling = ["leaf1", "right"]

        result = assembler.assemble_dp(tiling)

        # Each node returns its full text
        assert result == "First chunk of text.\n\nSummary of third and fourth chunks."

    def test_all_internal_nodes(self, assembler: Assembler, mock_nodes: None) -> None:
        """Test assembly with only internal nodes."""
        tiling = ["left", "right", "root"]

        result = assembler.assemble_dp(tiling)

        expected = "Summary of first and second chunks.\n\nSummary of third and fourth chunks.\n\nOverall document summary."
        assert result == expected

    def test_all_leaf_nodes(self, assembler: Assembler, mock_nodes: None) -> None:
        """Test assembly with all leaf nodes."""
        tiling = ["leaf1", "leaf2", "leaf3", "leaf4"]

        result = assembler.assemble_dp(tiling)

        # Should get full text for all leaf nodes
        expected = "First chunk of text.\n\nSecond chunk of text.\n\nThird chunk of text.\n\nFourth chunk of text."
        assert result == expected

    def test_empty_tiling(self, assembler: Assembler, mock_nodes: None) -> None:
        """Test handling of empty tiling list."""
        tiling: list[str] = []

        result = assembler.assemble_dp(tiling)

        assert result == ""

    def test_missing_node(self, assembler: Assembler, mock_nodes: None) -> None:
        """Test handling when tiling references a missing node."""
        tiling = ["leaf1", "missing", "leaf3"]

        result = assembler.assemble_dp(tiling)

        # Should skip missing node
        assert result == "First chunk of text.\n\nThird chunk of text."

    def test_node_with_no_text(
        self, assembler: Assembler, mock_nodes: None, doc_store: DocumentStore
    ) -> None:
        """Test handling of nodes with empty text."""
        # Add a node with empty text
        empty_nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "empty",
                "text": "",
                "embedding": np.array([0.5] * 1536, dtype=np.float64),
                "span_start": 82,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 0,
                "height": 0,
                "level_index": 0,
            }
        ]
        doc_store.nodes.add_batch(empty_nodes)

        tiling = ["leaf1", "empty", "leaf3"]

        result = assembler.assemble_dp(tiling)

        # Should skip empty node
        assert result == "First chunk of text.\n\nThird chunk of text."

    def test_single_root_node(self, assembler: Assembler, mock_nodes: None) -> None:
        """Test assembly with just the root node."""
        tiling = ["root"]

        result = assembler.assemble_dp(tiling)

        # Should return root's full summary
        assert result == "Overall document summary."

    def test_complex_tiling_assembly(
        self, assembler: Assembler, mock_nodes: None
    ) -> None:
        """Test a complex tiling that resembles real DP output."""
        # Simulate a tiling that might come from DP algorithm
        # Mix of internal and leaf nodes
        tiling = ["left", "leaf3", "leaf4"]

        result = assembler.assemble_dp(tiling)

        expected = (
            "Summary of first and second chunks.\n\n"
            "Third chunk of text.\n\n"
            "Fourth chunk of text."
        )
        assert result == expected

    def test_ordering_preservation(
        self, assembler: Assembler, mock_nodes: None
    ) -> None:
        """Test that tiling order is preserved in output."""
        # Nodes in non-sequential order
        tiling = ["leaf3", "leaf1", "leaf4", "leaf2"]

        result = assembler.assemble_dp(tiling)

        # Output should follow tiling order, not span order
        expected = (
            "Third chunk of text.\n\n"
            "First chunk of text.\n\n"
            "Fourth chunk of text.\n\n"
            "Second chunk of text."
        )
        assert result == expected

    def test_dp_single_node_tree(self, doc_store: DocumentStore) -> None:
        """Test the DP algorithm on a tree with only a single node.

        This test focuses on the DP algorithm itself rather than assembly.
        """
        # Set up configuration
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test-key"),
            database_url="sqlite:///:memory:",
        )

        from ragzoom.vector_factory import create_vector_index

        vi = create_vector_index(
            "python", "sqlite:///:memory:", query_config.embedding_model
        )
        retriever = create_retriever(
            query_config=query_config,
            store=doc_store,
            api_key=operational_config.openai_api_key.get_secret_value(),
            vector_index=vi,
        )
        dp_generator = retriever.dp_generator

        # Manually create a single-node tree
        single_node: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            {
                "node_id": "root",
                "text": "single node",
                "embedding": np.array([0.1] * 1536, dtype=np.float64),
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc-single",
                "token_count": 11,
                "height": 0,
                "level_index": 0,
            }
        ]
        doc_store.nodes.add_batch(single_node)

        # Create coverage map
        coverage_map = {"root": True}

        # Load nodes from coverage map
        nodes = {}
        for node_id in coverage_map:
            node = doc_store.nodes.get_node(node_id)
            if node:
                nodes[node_id] = node

        # Find root node
        root_id = "root"

        # Test the DP algorithm
        dp_result = dp_generator.find_optimal_tiling(
            1000, {"root": 1.0}, nodes, root_id
        )
        tiling = dp_result.tiling

        assert tiling, "DP tiling should not be empty for single node tree"
        assert len(tiling.node_ids) == 1
        assert tiling.node_ids[0] == "root"
