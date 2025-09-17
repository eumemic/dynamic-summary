"""SQLite-based tests for span corruption bug in tree building.

These tests ensure that tree building handles odd numbers of nodes correctly
and prevents span corruption issues using
the real in-memory SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.document_store import DocumentStore
from ragzoom.index import TreeBuilder
from tests.conftest import BackwardCompatibilityConfig


@pytest.mark.usefixtures("sqlite_backend")
class TestSpanCorruptionSQLite:
    """Test span corruption issues in tree building."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc")

    def setup_system(
        self, doc_store: DocumentStore, vector_index: _VectorIndexProtocol
    ) -> tuple[BackwardCompatibilityConfig, DocumentStore, TreeBuilder, AsyncMock]:
        """Set up test system."""
        # Create separate configs
        index_config = IndexConfig.load(
            target_chunk_tokens=100,  # Small chunks to create many nodes
            preceding_context_tokens=10,  # Must be less than leaf_tokens
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test-key"),
        )

        # Create tree builder
        tree_builder = TreeBuilder(
            index_config,
            doc_store,
            vector_index,
            api_key=operational_config.openai_api_key.get_secret_value(),
        )

        # Mock API calls
        mock_client = AsyncMock()
        tree_builder.llm_service.client = mock_client

        # Create a config wrapper for backward compatibility
        config = BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )

        return config, doc_store, tree_builder, mock_client

    @pytest.mark.asyncio
    async def test_odd_nodes_create_invalid_spans(
        self, doc_store: DocumentStore, vector_index: _VectorIndexProtocol
    ) -> None:
        """Test that odd number of nodes creates span corruption."""
        config, doc_store, tree_builder, mock_client = self.setup_system(
            doc_store, vector_index
        )

        # Create text that will split into an odd number of chunks
        # Each chunk is ~100 tokens, so we need longer content
        chunk_text = (
            "This is a longer chunk of text that should be approximately one hundred tokens. "
            * 12
        )
        chunks = [f"Chunk {i}: {chunk_text}" for i in range(15)]
        text = " ".join(chunks)

        # Mock embeddings - return different embeddings for each text
        mock_client.embeddings.create = AsyncMock(
            side_effect=lambda **kwargs: Mock(
                data=[Mock(embedding=[0.1] * 1536)]
                * len(kwargs.get("input", [kwargs.get("input")]))
            )
        )

        # Mock summaries
        mock_client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[
                    Mock(message=Mock(content="Summary of left and right content"))
                ]
            )
        )

        # Index the document
        await tree_builder.add_document_async(text, show_progress=False)

        # Check for span corruption
        from collections.abc import Sequence
        from typing import cast

        nodes = cast(Sequence[TreeNode], doc_store.nodes.get_all())

        # Check for invalid spans
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
                # Zero-width spans for non-leaf nodes are also invalid
                corrupt_nodes.append(
                    {
                        "id": node.id,
                        "height": node_height,
                        "span_start": node.span_start,
                        "span_end": node.span_end,
                    }
                )

        # Report findings
        if corrupt_nodes:
            print(f"\nFound {len(corrupt_nodes)} corrupt nodes:")
            for corrupt_node in corrupt_nodes:
                print(
                    f"  Height {corrupt_node['height']}: span ({corrupt_node['span_start']}, {corrupt_node['span_end']})"
                )

        # This test SHOULD fail with the current implementation
        assert (
            len(corrupt_nodes) == 0
        ), f"Found {len(corrupt_nodes)} nodes with invalid spans"

    @pytest.mark.asyncio
    async def test_wraparound_pairing(
        self, doc_store: DocumentStore, vector_index: _VectorIndexProtocol
    ) -> None:
        """Test that demonstrates wraparound pairing issue."""
        config, doc_store, tree_builder, mock_client = self.setup_system(
            doc_store, vector_index
        )

        # Create 5 chunks that will split properly at 100 tokens each
        # Each chunk needs to be long enough to hit the token limit
        base_text = "The quick brown fox jumps over the lazy dog. " * 20  # ~100 tokens
        chunks = []
        for i in range(5):
            chunks.append(f"CHUNK_{i}_START {base_text} CHUNK_{i}_END")
        text = " ".join(chunks)

        # Mock embeddings
        mock_client.embeddings.create = AsyncMock(
            side_effect=lambda **kwargs: Mock(
                data=[Mock(embedding=[0.1] * 1536)]
                * len(kwargs.get("input", [kwargs.get("input")]))
            )
        )

        # Mock summaries
        mock_client.chat.completions.create = AsyncMock(
            return_value=Mock(
                choices=[Mock(message=Mock(content="Summary of the content"))]
            )
        )

        # Index the document
        await tree_builder.add_document_async(text, show_progress=False)

        # Verify tree structure
        from collections.abc import Sequence
        from typing import cast

        nodes = cast(Sequence[TreeNode], doc_store.nodes.get_all())

        # Group nodes by height
        nodes_by_height: dict[int, list[TreeNode]] = {}
        for node in nodes:
            node_height = node.height
            if node_height not in nodes_by_height:
                nodes_by_height[node_height] = []
            nodes_by_height[node_height].append(node)

        # Sort nodes by span_start within each height
        for height in nodes_by_height:
            nodes_by_height[height].sort(key=lambda n: n.span_start)

        print("\nTree structure:")
        for height in sorted(nodes_by_height.keys()):
            print(f"\nHeight {height}:")
            for node in nodes_by_height[height]:
                print(f"  Node: span ({node.span_start}, {node.span_end})")
                if node.left_child_id and node.right_child_id:
                    left = next(n for n in nodes if n.id == node.left_child_id)
                    right = next(n for n in nodes if n.id == node.right_child_id)
                    print(f"    Left child: span ({left.span_start}, {left.span_end})")
                    print(
                        f"    Right child: span ({right.span_start}, {right.span_end})"
                    )

                    # Check for wraparound
                    if left.span_end > right.span_start:
                        print(
                            f"    WARNING: Wraparound detected! Left end {left.span_end} > Right start {right.span_start}"
                        )

        # Check all parent nodes have valid spans
        for node in nodes:
            node_height = node.height
            if node_height > 0:
                assert (
                    node.span_end >= node.span_start
                ), f"Node at height {node_height} has invalid span: ({node.span_start}, {node.span_end})"

    def test_manual_span_corruption_scenario(self, doc_store: DocumentStore) -> None:
        """Test a manually constructed scenario that should expose span corruption."""
        # Seed nodes that simulate what happens with odd pairing in tree construction
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # 5 leaf nodes simulating odd count
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
            # Internal nodes that pair the leaves
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
            # Higher level internal node
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
            # Root
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

        # Verify no span corruption
        all_nodes = doc_store.nodes.get_all()
        corrupt_nodes = []

        for node in all_nodes:
            # Check basic span validity
            if node.span_end < node.span_start:
                corrupt_nodes.append(node)
            # Check that parent spans encompass child spans
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

        assert (
            len(corrupt_nodes) == 0
        ), f"Found {len(corrupt_nodes)} nodes with span corruption"
