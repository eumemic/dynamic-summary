"""Test that demonstrates the current bug in Retriever - creates incomplete coverage trees."""

import asyncio
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragzoom.retrieve import Retriever

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.vector_api import Vector


@pytest.mark.usefixtures("sqlite_backend")
class TestRetrieverBugSQLite:
    """Tests that show the current Retriever creates incomplete coverage trees."""

    @pytest.fixture
    def setup_tree_for_bug_demo(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> Generator[tuple[object, DocumentStore, "Retriever"], None, None]:
        """Set up a system with a tree structure to demonstrate the bug."""
        index_config = IndexConfig.load(
            target_chunk_tokens=100, preceding_context_tokens=50
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create document store
        doc_store = sqlite_store_factory("test-doc")

        # Create tree structure:
        #         root
        #        /    \
        #      P1      P2
        #     /  \    /  \
        #    L1  L2  L3  L4

        # Add all nodes using add_batch
        nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "L1",
                "text": "Chapter 1 content",
                "span_start": 0,
                "span_end": 20,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "path": "00",
                "parent_id": None,  # Will be set in update_parent_references_batch
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "L2",
                "text": "Chapter 2 content",
                "span_start": 20,
                "span_end": 40,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "path": "01",
                "parent_id": None,  # Will be set in update_parent_references_batch
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "L3",
                "text": "Chapter 3 content",
                "span_start": 40,
                "span_end": 60,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "path": "10",
                "parent_id": None,  # Will be set in update_parent_references_batch
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "L4",
                "text": "Chapter 4 content",
                "span_start": 60,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "path": "11",
                "parent_id": None,  # Will be set in update_parent_references_batch
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "P1",
                "text": "Summary of chapters 1-2",
                "span_start": 0,
                "span_end": 40,
                "document_id": "test-doc",
                "token_count": 75,
                "height": 1,
                "path": "0",
                "parent_id": None,  # Will be set in update_parent_references_batch
                "left_child_id": "L1",
                "right_child_id": "L2",
            },
            {
                "node_id": "P2",
                "text": "Summary of chapters 3-4",
                "span_start": 40,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 75,
                "height": 1,
                "path": "1",
                "parent_id": None,  # Will be set in update_parent_references_batch
                "left_child_id": "L3",
                "right_child_id": "L4",
            },
            {
                "node_id": "root",
                "text": "Full document summary",
                "span_start": 0,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 100,
                "height": 2,
                "path": "",
                "parent_id": None,
                "left_child_id": "P1",
                "right_child_id": "P2",
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Set parent references
        doc_store.nodes.update_parent_references_batch(
            [
                ("L1", "P1"),
                ("L2", "P1"),
                ("L3", "P2"),
                ("L4", "P2"),
                ("P1", "root"),
                ("P2", "root"),
            ]
        )

        from ragzoom.vector_factory import create_vector_index
        from tests.utils import create_retriever

        vi = create_vector_index(
            "python", "sqlite:///:memory:", query_config.embedding_model
        )
        retriever = create_retriever(
            query_config=query_config,
            store=doc_store,
            document_id="test-doc",
            api_key=operational_config.openai_api_key.get_secret_value(),
            vector_index=vi,
        )
        yield index_config, doc_store, retriever

    def test_retriever_bug_with_num_seeds_1(
        self,
        setup_tree_for_bug_demo: tuple[object, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that the retriever should build complete coverage trees but currently doesn't."""
        index_config, doc_store, retriever = setup_tree_for_bug_demo

        # Mock the search similar functionality to return only L3
        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            k: int,
            where: dict[str, str | int | float | bool | None] | None = None,
        ) -> list[Vector]:
            import numpy as _np

            return [
                Vector(
                    id="L3",
                    vec=_np.ones(1536, dtype=_np.float32),
                    meta={
                        "document_id": "test-doc",
                        "span_start": 0,
                        "span_end": 0,
                        "parent_id": "P2",
                        "is_leaf": 1,
                    },
                    model_id="text-embedding-3-small",
                    dim=1536,
                )
            ]

        retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]

        # We don't need vector embeddings since we're mocking search

        # Mock the query embedding generation to return something close to L3
        retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
            lambda query, document_id=None: [0.3] * 1536
        )

        # This SHOULD work without errors if the retriever built complete coverage trees
        # But it currently raises an error because of the bug
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
            )
        )

        # If we got here, the retriever should have produced a valid tiling
        assert result.tiling is not None
        assert len(result.tiling) > 0

    def test_retriever_builds_complete_coverage_trees(
        self,
        setup_tree_for_bug_demo: tuple[object, DocumentStore, "Retriever"],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that retriever should include siblings to build complete coverage trees."""
        index_config, doc_store, retriever = setup_tree_for_bug_demo

        # Mock the search similar functionality to return only L3
        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            k: int,
            where: dict[str, str | int | float | bool | None] | None = None,
        ) -> list[Vector]:
            import numpy as _np

            return [
                Vector(
                    id="L3",
                    vec=_np.ones(1536, dtype=_np.float32),
                    meta={
                        "document_id": "test-doc",
                        "span_start": 0,
                        "span_end": 0,
                        "parent_id": "P2",
                        "is_leaf": 1,
                    },
                    model_id="text-embedding-3-small",
                    dim=1536,
                )
            ]

        retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]

        # We don't need vector embeddings since we're mocking search

        # Mock the query embedding generation to return something close to L3
        retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
            lambda query, document_id=None: [0.3] * 1536
        )

        # Patch to capture what nodes the DP algorithm receives
        captured_nodes = {}
        original_find_optimal = retriever.dp_generator.find_optimal_tiling

        def capture_and_pass_through(
            budget_tokens: int,
            scores: dict[str, float],
            nodes: dict[str, TreeNode],
            root_id: str,
        ) -> object:
            captured_nodes.update(nodes)
            return original_find_optimal(budget_tokens, scores, nodes, root_id)

        retriever.dp_generator.find_optimal_tiling = capture_and_pass_through  # type: ignore[method-assign,assignment]

        # Run retriever - should work without errors
        result = asyncio.run(
            retriever.retrieve_async(
                query="test",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
            )
        )

        # Verify the retriever built a coverage tree with the right nodes
        # When L3 is selected, we need its sibling L4 to maintain completeness
        assert "L3" in captured_nodes  # Selected leaf
        assert (
            "L4" in captured_nodes
        )  # Sibling MUST be included (P2 needs both children)
        assert "P2" in captured_nodes  # Parent of L3 and L4
        assert (
            "P1" in captured_nodes
        )  # Sibling of P2 MUST be included (root needs both children)
        assert "root" in captured_nodes  # Root

        # P1's children (L1, L2) should NOT be included
        # P1 can be a leaf in the coverage tree
        assert "L1" not in captured_nodes
        assert "L2" not in captured_nodes

        # The result should be valid
        assert result.tiling is not None
        assert len(result.tiling) > 0
