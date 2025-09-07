"""SQLite-based tests that demonstrate the current bug in Retriever - creates incomplete coverage trees.

This file converts the core retriever bug tests from test_retriever_bug.py to use the
real in-memory SQLite backend instead of SimpleMockStore, providing higher fidelity
testing of the retrieval functionality.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.document_store import DocumentStore

if TYPE_CHECKING:
    from ragzoom.retrieve import Retriever


@pytest.mark.usefixtures("sqlite_backend")
class TestRetrieverBugSQLite:
    """Tests that show the current Retriever creates incomplete coverage trees with SQLite."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        """Create a document-scoped store for test-doc."""
        return sqlite_store_factory("test-doc")

    @pytest.fixture
    def setup_tree_for_bug_demo(
        self, sqlite_backend: SQLiteStorageBackend, doc_store: DocumentStore
    ) -> Generator[tuple[object, DocumentStore, Retriever], None, None]:
        """Set up a system with a tree structure to demonstrate the bug."""
        index_config = IndexConfig.load(
            target_chunk_tokens=100, preceding_context_tokens=50
        )
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create same tree structure as before:
        #         root
        #        /    \
        #      P1      P2
        #     /  \    /  \
        #    L1  L2  L3  L4

        nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            # Leaf nodes with proper path values for tree navigation
            {
                "node_id": "L1",
                "text": "Chapter 1 content",
                "embedding": [0.1] * 1536,
                "span_start": 0,
                "span_end": 20,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "path": "00",  # Left child of P1 (path "0")
            },
            {
                "node_id": "L2",
                "text": "Chapter 2 content",
                "embedding": [0.2] * 1536,
                "span_start": 20,
                "span_end": 40,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "path": "01",  # Right child of P1 (path "0")
            },
            {
                "node_id": "L3",
                "text": "Chapter 3 content",
                "embedding": [0.3] * 1536,
                "span_start": 40,
                "span_end": 60,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "path": "10",  # Left child of P2 (path "1")
            },
            {
                "node_id": "L4",
                "text": "Chapter 4 content",
                "embedding": [0.4] * 1536,
                "span_start": 60,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 5,
                "height": 0,
                "path": "11",  # Right child of P2 (path "1")
            },
            # Parent nodes
            {
                "node_id": "P1",
                "text": "Summary of chapters 1-2",
                "embedding": [0.15] * 1536,
                "span_start": 0,
                "span_end": 40,
                "document_id": "test-doc",
                "height": 1,
                "left_child_id": "L1",
                "right_child_id": "L2",
                "path": "0",  # Left child of root
            },
            {
                "node_id": "P2",
                "text": "Summary of chapters 3-4",
                "embedding": [0.35] * 1536,
                "span_start": 40,
                "span_end": 80,
                "document_id": "test-doc",
                "height": 1,
                "left_child_id": "L3",
                "right_child_id": "L4",
                "path": "1",  # Right child of root
            },
            # Root node
            {
                "node_id": "root",
                "text": "Full document summary",
                "embedding": [0.25] * 1536,
                "span_start": 0,
                "span_end": 80,
                "document_id": "test-doc",
                "height": 2,
                "left_child_id": "P1",
                "right_child_id": "P2",
                "path": "",  # Root has empty path
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Add parent references using the batch method
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

        # Upsert embeddings into the vector index for retrieval behavior
        sqlite_backend.vector_index.upsert(
            [
                (
                    "L1",
                    [0.1] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 20,
                        "parent_id": "P1",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "L2",
                    [0.2] * 1536,
                    {
                        "span_start": 20,
                        "span_end": 40,
                        "parent_id": "P1",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "L3",
                    [0.3] * 1536,
                    {
                        "span_start": 40,
                        "span_end": 60,
                        "parent_id": "P2",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "L4",
                    [0.4] * 1536,
                    {
                        "span_start": 60,
                        "span_end": 80,
                        "parent_id": "P2",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "P1",
                    [0.15] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 40,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
                (
                    "P2",
                    [0.35] * 1536,
                    {
                        "span_start": 40,
                        "span_end": 80,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
                (
                    "root",
                    [0.25] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 80,
                        "parent_id": "",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
            ]
        )

        from tests.utils import create_retriever

        retriever = create_retriever(
            query_config=query_config,
            store=doc_store,
            document_id="test-doc",  # Specify the document we're working with
            api_key=operational_config.openai_api_key.get_secret_value(),
        )
        yield index_config, doc_store, retriever

    def test_retriever_bug_with_num_seeds_1(
        self, setup_tree_for_bug_demo: tuple[object, DocumentStore, Retriever]
    ) -> None:
        """Test that the retriever should build complete coverage trees but currently doesn't."""
        index_config, doc_store, retriever = setup_tree_for_bug_demo

        # Mock the search service to return only L3
        # This simulates what happens with --num-seeds 1 when L3 is most relevant
        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            n_results: int,
        ) -> list[tuple[str, float, dict[str, str | int | float | bool | None]]]:
            return [("L3", 0.95, {})]

        # Mock search through the document store's search service
        original_similar = doc_store.search.similar
        doc_store.search.similar = mock_search_similar  # type: ignore[assignment, method-assign]

        # Also mock MMR to return L3 as selected
        def mock_compute_mmr_diverse_results(
            query_embedding: list[float] | NDArray[np.float64],
            candidates: list[
                tuple[str, float, dict[str, str | int | float | bool | None]]
            ],
            lambda_param: float,
            k: int,
        ) -> list[str]:
            return ["L3"]

        doc_store.search.compute_mmr_diverse_results = mock_compute_mmr_diverse_results  # type: ignore[assignment, method-assign]

        # Mock the query embedding generation using MagicMock
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding.return_value = [0.3] * 1536
        retriever.embedding_service = mock_embedding_service

        # This SHOULD work without errors if the retriever built complete coverage trees
        # But it currently raises an error because of the bug
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",  # Query text doesn't matter with our mock
                num_seeds=1,  # Only select 1 node
                budget_tokens=1000,
                document_id="test-doc",
            )
        )

        # If we got here, the retriever should have produced a valid tiling
        assert result.tiling is not None
        assert len(result.tiling) > 0

        # Restore original method
        doc_store.search.similar = original_similar  # type: ignore[method-assign]

    def test_retriever_builds_complete_coverage_trees(
        self, setup_tree_for_bug_demo: tuple[object, DocumentStore, Retriever]
    ) -> None:
        """Test that retriever should include siblings to build complete coverage trees."""
        index_config, doc_store, retriever = setup_tree_for_bug_demo

        # Mock search service to return only L3
        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            n_results: int,
        ) -> list[tuple[str, float, dict[str, str | int | float | bool | None]]]:
            return [("L3", 0.95, {})]

        # Mock search through the document store's search service
        original_similar = doc_store.search.similar
        doc_store.search.similar = mock_search_similar  # type: ignore[assignment, method-assign]

        # Also mock MMR to return L3 as selected
        def mock_compute_mmr_diverse_results(
            query_embedding: list[float] | NDArray[np.float64],
            candidates: list[
                tuple[str, float, dict[str, str | int | float | bool | None]]
            ],
            lambda_param: float,
            k: int,
        ) -> list[str]:
            return ["L3"]

        doc_store.search.compute_mmr_diverse_results = mock_compute_mmr_diverse_results  # type: ignore[assignment, method-assign]

        # Mock the query embedding generation using MagicMock
        mock_embedding_service = MagicMock()
        mock_embedding_service.get_query_embedding.return_value = [0.3] * 1536
        retriever.embedding_service = mock_embedding_service

        # Patch to capture what nodes the DP algorithm receives
        captured_nodes = {}
        original_find_optimal = retriever.dp_generator.find_optimal_tiling

        def capture_and_pass_through(
            budget_tokens: int,
            scores: dict[str, float],
            nodes: dict[str, object],  # Use object to avoid TreeNode import issues
            root_id: str,
        ) -> object:
            captured_nodes.update(nodes)
            # Use duck-typing - both TreeNode and SqliteTreeNode have the required attributes
            # The DP algorithm only uses common node attributes (id, token_count, child_ids, spans)
            # which both node types provide, so we can safely cast
            from typing import cast

            from ragzoom.models import TreeNode

            typed_nodes = cast(dict[str, TreeNode], nodes)
            return original_find_optimal(budget_tokens, scores, typed_nodes, root_id)

        # Use MagicMock to wrap the method
        mock_dp_generator = MagicMock(wraps=retriever.dp_generator)
        mock_dp_generator.find_optimal_tiling.side_effect = capture_and_pass_through
        retriever.dp_generator = mock_dp_generator

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

        # Restore original method
        doc_store.search.similar = original_similar  # type: ignore[method-assign]
