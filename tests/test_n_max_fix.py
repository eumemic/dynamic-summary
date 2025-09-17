"""SQLite-based test that verifies the num_seeds constraint fix works correctly.

These tests ensure that retrieve() only passes coverage tree nodes to DP,
preventing the algorithm from operating on nodes outside the coverage set.
Using the real in-memory SQLite backend.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend


class TestNumSeedsFixSQLite:
    """Test that the fix for num_seeds constraint works correctly."""

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> StorageBackend:
        return storage_backend

    def test_retrieve_respects_coverage_tree(self, doc_store: StorageBackend) -> None:
        """Test that retrieve() only passes coverage tree nodes to DP."""
        # Get document store for the specific document
        document_store = doc_store.for_document("doc1")
        document_store.set_metadata(
            file_path="test.txt",
            content_hash="test-hash",
            chunk_count=7,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Build a simple tree with explicit token counts for DP cost control
        nodes: list[
            dict[
                str,
                str | int | float | bool | list[float] | NDArray[np.float64] | None,
            ]
        ] = [
            # Leaf nodes
            {
                "node_id": "leaf1",
                "text": "Leaf 0 with dragon content",
                "embedding": np.array([0.9] * 1536),
                "span_start": 0,
                "span_end": 250,
                "document_id": "doc1",
                "token_count": 100,
                "height": 0,
                "parent_id": "nodeA",
                "path": "00",
            },
            {
                "node_id": "leaf2",
                "text": "Leaf 1 with dragon content",
                "embedding": np.array([0.9] * 1536),
                "span_start": 250,
                "span_end": 500,
                "document_id": "doc1",
                "token_count": 100,
                "height": 0,
                "parent_id": "nodeA",
                "path": "01",
            },
            {
                "node_id": "leaf3",
                "text": "Leaf 2 with dragon content",
                "embedding": np.array([0.9] * 1536),
                "span_start": 500,
                "span_end": 750,
                "document_id": "doc1",
                "token_count": 100,
                "height": 0,
                "parent_id": "nodeB",
                "path": "10",
            },
            {
                "node_id": "leaf4",
                "text": "Leaf 3 with dragon content",
                "embedding": np.array([0.9] * 1536),
                "span_start": 750,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 100,
                "height": 0,
                "parent_id": "nodeB",
                "path": "11",
            },
            # Internal nodes
            {
                "node_id": "nodeA",
                "text": "Node A content",
                "embedding": np.array([0.5] * 1536),
                "span_start": 0,
                "span_end": 500,
                "document_id": "doc1",
                "token_count": 150,
                "height": 1,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
                "parent_id": "root",
                "path": "0",
            },
            {
                "node_id": "nodeB",
                "text": "Node B content",
                "embedding": np.array([0.5] * 1536),
                "span_start": 500,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 150,
                "height": 1,
                "left_child_id": "leaf3",
                "right_child_id": "leaf4",
                "parent_id": "root",
                "path": "1",
            },
            # Root node
            {
                "node_id": "root",
                "text": "Root summary document",
                "embedding": np.array([0.5] * 1536),
                "span_start": 0,
                "span_end": 1000,
                "document_id": "doc1",
                "token_count": 200,
                "height": 2,
                "left_child_id": "nodeA",
                "right_child_id": "nodeB",
                "parent_id": None,
                "path": "",
            },
        ]
        document_store.nodes.add_batch(nodes)
        document_store.nodes.update_parent_references_batch(
            [
                ("leaf1", "nodeA"),
                ("leaf2", "nodeA"),
                ("leaf3", "nodeB"),
                ("leaf4", "nodeB"),
                ("nodeA", "root"),
                ("nodeB", "root"),
            ]
        )

        # Create config and retriever
        query_config = QueryConfig(budget_tokens=10000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Mock OpenAI client
        with patch("openai.OpenAI") as mock_client:
            mock_embeddings = Mock()
            mock_embeddings.create = Mock(
                return_value=Mock(data=[Mock(embedding=[0.9] * 1536)])
            )
            mock_instance = Mock()
            mock_instance.embeddings = mock_embeddings
            mock_client.return_value = mock_instance

            from ragzoom.vector_factory import create_vector_index
            from tests.utils import create_retriever

            vi = create_vector_index(
                "python", "sqlite:///:memory:", query_config.embedding_model
            )
            retriever = create_retriever(
                query_config=query_config,
                store=document_store,
                document_id="doc1",
                api_key=operational_config.openai_api_key.get_secret_value(),
                client=mock_instance,
                vector_index=vi,
            )

            # Patch vector index

            import numpy as _np
            from numpy.typing import NDArray as _NDArray

            from ragzoom.vector_api import Vector as _Vector

            def _mock_search_similar(
                query_embedding: list[float] | _NDArray[_np.float64],
                k: int,
                where: dict[str, str | int | float | bool | None] | None = None,
            ) -> list[_Vector]:
                import numpy as _np

                from ragzoom.vector_api import Vector

                return [
                    Vector(
                        "leaf1",
                        _np.ones(1536, dtype=_np.float32),
                        {
                            "document_id": "doc1",
                            "span_start": 0,
                            "span_end": 0,
                            "parent_id": "nodeA",
                            "is_leaf": 1,
                        },
                        "m",
                        3,
                    ),
                    Vector(
                        "leaf2",
                        _np.ones(1536, dtype=_np.float32),
                        {
                            "document_id": "doc1",
                            "span_start": 0,
                            "span_end": 0,
                            "parent_id": "nodeA",
                            "is_leaf": 1,
                        },
                        "m",
                        3,
                    ),
                    Vector(
                        "leaf3",
                        _np.ones(1536, dtype=_np.float32),
                        {
                            "document_id": "doc1",
                            "span_start": 0,
                            "span_end": 0,
                            "parent_id": "nodeB",
                            "is_leaf": 1,
                        },
                        "m",
                        3,
                    ),
                    Vector(
                        "leaf4",
                        _np.ones(1536, dtype=_np.float32),
                        {
                            "document_id": "doc1",
                            "span_start": 0,
                            "span_end": 0,
                            "parent_id": "nodeB",
                            "is_leaf": 1,
                        },
                        "m",
                        3,
                    ),
                ]

            retriever.vector_index.search_similar = _mock_search_similar  # type: ignore[method-assign]

            # Retrieve with num_seeds=1
            result = retriever.retrieve("dragon", num_seeds=1, document_id="doc1")

        # Verify selected nodes
        assert result.node_ids == ["leaf1"]

        # Verify coverage map contains selected + ancestors + siblings to maintain full binary tree
        # Since leaf1 is included and nodeA is its parent, leaf2 (sibling) must be included
        # Since nodeA is included and root is its parent, nodeB (sibling) must be included
        expected_coverage = {"leaf1", "leaf2", "nodeA", "nodeB", "root"}
        assert set(result.coverage_map.keys()) == expected_coverage

        # CRITICAL: Verify scores only contain nodes from coverage map
        assert set(result.scores.keys()).issubset(expected_coverage), (
            f"Scores contain nodes outside coverage map! "
            f"Scores: {set(result.scores.keys())}, "
            f"Coverage: {expected_coverage}"
        )

        # Verify tiling only uses nodes from coverage tree
        if result.tiling:
            tiling_nodes = set(result.tiling)  # tiling is now a list of node IDs
            assert tiling_nodes.issubset(expected_coverage), (
                f"Tiling contains nodes outside coverage tree! "
                f"Tiling: {tiling_nodes}, Coverage: {expected_coverage}"
            )

            # Count leaf nodes in tiling (using height == 0)
            leaf_count = 0
            for node_id in result.tiling:
                node = document_store.nodes.get_node(node_id)
                if node and node.height == 0:
                    leaf_count += 1
            # Since we have to include leaf2 to maintain coverage property, the DP algorithm
            # might choose to use both leaves instead of their parent
            assert leaf_count <= 2, f"Expected at most 2 leaf nodes, got {leaf_count}"
