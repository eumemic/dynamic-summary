"""Test for span corruption bug in tree building."""

from unittest.mock import AsyncMock, Mock

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.index import TreeBuilder


class TestSpanCorruption:
    """Test span corruption issues in tree building."""

    @pytest.fixture
    def setup_system(self, store):
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

        # Create document-scoped store
        doc_store = store.for_document("test-doc")
        tree_builder = TreeBuilder(
            index_config,
            doc_store,
            api_key=operational_config.openai_api_key.get_secret_value(),
        )

        # Mock API calls
        mock_client = AsyncMock()
        tree_builder.llm_service.client = mock_client

        # Create a config wrapper for backward compatibility
        from tests.conftest import BackwardCompatibilityConfig

        config = BackwardCompatibilityConfig(
            index_config, query_config, operational_config
        )

        yield config, store, tree_builder, mock_client

    @pytest.mark.asyncio
    async def test_odd_nodes_create_invalid_spans(self, setup_system):
        """Test that odd number of nodes creates span corruption."""
        config, store, tree_builder, mock_client = setup_system

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
        doc_id = await tree_builder.add_document_async(text, show_progress=False)

        # Check for span corruption
        with store.SessionLocal() as session:
            from ragzoom.models import TreeNode

            # Get all nodes
            all_nodes = session.query(TreeNode).filter_by(document_id=doc_id).all()

            # Check for invalid spans
            corrupt_nodes = []
            for node in all_nodes:
                node_depth = store.get_node_depth(node.id)
                if node.span_end < node.span_start:
                    corrupt_nodes.append(
                        {
                            "id": node.id,
                            "depth": node_depth,
                            "span_start": node.span_start,
                            "span_end": node.span_end,
                        }
                    )
                elif node.span_start == node.span_end and node_depth > 0:
                    # Zero-width spans for non-leaf nodes are also invalid
                    corrupt_nodes.append(
                        {
                            "id": node.id,
                            "depth": node_depth,
                            "span_start": node.span_start,
                            "span_end": node.span_end,
                        }
                    )

            # Report findings
            if corrupt_nodes:
                print(f"\nFound {len(corrupt_nodes)} corrupt nodes:")
                for node in corrupt_nodes:
                    print(
                        f"  Depth {node['depth']}: span ({node['span_start']}, {node['span_end']})"
                    )

            # This test SHOULD fail with the current implementation
            assert (
                len(corrupt_nodes) == 0
            ), f"Found {len(corrupt_nodes)} nodes with invalid spans"

    @pytest.mark.asyncio
    async def test_wraparound_pairing(self, setup_system):
        """Test that demonstrates wraparound pairing issue."""
        config, store, tree_builder, mock_client = setup_system

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
        doc_id = await tree_builder.add_document_async(text, show_progress=False)

        # Verify tree structure
        with store.SessionLocal() as session:
            from ragzoom.models import TreeNode

            # Get nodes by depth
            nodes_by_depth = {}
            all_nodes = session.query(TreeNode).filter_by(document_id=doc_id).all()

            for node in all_nodes:
                node_depth = store.get_node_depth(node.id)
                if node_depth not in nodes_by_depth:
                    nodes_by_depth[node_depth] = []
                nodes_by_depth[node_depth].append(node)

            # Sort nodes by span_start within each depth
            for depth in nodes_by_depth:
                nodes_by_depth[depth].sort(key=lambda n: n.span_start)

            print("\nTree structure:")
            for depth in sorted(nodes_by_depth.keys()):
                print(f"\nDepth {depth}:")
                for node in nodes_by_depth[depth]:
                    print(f"  Node: span ({node.span_start}, {node.span_end})")
                    if node.left_child_id and node.right_child_id:
                        left = next(n for n in all_nodes if n.id == node.left_child_id)
                        right = next(
                            n for n in all_nodes if n.id == node.right_child_id
                        )
                        print(
                            f"    Left child: span ({left.span_start}, {left.span_end})"
                        )
                        print(
                            f"    Right child: span ({right.span_start}, {right.span_end})"
                        )

                        # Check for wraparound
                        if left.span_end > right.span_start:
                            print(
                                f"    WARNING: Wraparound detected! Left end {left.span_end} > Right start {right.span_start}"
                            )

            # Check all parent nodes have valid spans
            for node in all_nodes:
                node_depth = store.get_node_depth(node.id)
                if node_depth > 0:
                    assert (
                        node.span_end >= node.span_start
                    ), f"Node at depth {node_depth} has invalid span: ({node.span_start}, {node.span_end})"
