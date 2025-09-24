"""Tests for span corruption bug in tree building.

These tests ensure that tree building handles odd numbers of nodes correctly
and prevents span corruption issues.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.index import TreeBuilder
from tests.conftest import BackwardCompatibilityConfig


class TestSpanCorruption:
    """Test span corruption issues in tree building."""

    def setup_system(
        self, storage_backend: StorageBackend, vector_index: _VectorIndexProtocol
    ) -> tuple[BackwardCompatibilityConfig, TreeBuilder, AsyncMock]:
        """Set up test system."""
        # Get document store first
        doc_store = storage_backend.for_document("test-doc")

        # Set up document metadata
        doc_store.set_metadata(
            file_path="test_span_corruption.txt",
            content_hash="span-corruption-test-hash",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

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

        return config, tree_builder, mock_client

    @pytest.mark.asyncio
    async def test_odd_nodes_create_invalid_spans(
        self, storage_backend: StorageBackend, vector_index: _VectorIndexProtocol
    ) -> None:
        """Test that odd number of nodes creates span corruption."""
        config, tree_builder, mock_client = self.setup_system(
            storage_backend, vector_index
        )
        doc_store = storage_backend.for_document("test-doc")

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
        self, storage_backend: StorageBackend, vector_index: _VectorIndexProtocol
    ) -> None:
        """Test that demonstrates wraparound pairing issue."""
        config, tree_builder, mock_client = self.setup_system(
            storage_backend, vector_index
        )
        doc_store = storage_backend.for_document("test-doc")

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
            height = node.height
            if height not in nodes_by_height:
                nodes_by_height[height] = []
            nodes_by_height[height].append(node)

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
            if node.height > 0:
                assert (
                    node.span_end >= node.span_start
                ), f"Node at height {node.height} has invalid span: ({node.span_start}, {node.span_end})"
