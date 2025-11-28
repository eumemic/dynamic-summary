"""SQLite-based budget guarantee tests.

SQLite-based tests for budget guarantees in retrieval and assembly
with the real in-memory SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.assemble import Assembler
from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import QueryConfig
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.validate import validate_tiling
from tests.utils import create_retriever, mock_openai_context


@pytest.mark.usefixtures("sqlite_backend")
class TestBudgetGuarantee:
    """Test that budget guarantees are enforced by construction."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc")

    @pytest.fixture
    def assembler(self, doc_store: DocumentStore) -> Assembler:
        return Assembler(doc_store)

    @pytest.mark.slow_threshold(3.0)
    def test_budget_never_exceeded_worst_case(
        self,
        doc_store: DocumentStore,
        assembler: Assembler,
        sqlite_backend: SQLiteStorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test that assembly never exceeds budget even in worst case."""
        # Create a multi-level tree with known token costs
        nodes: list[NodeDataDict] = [
            # Leaf nodes (level 0)
            {
                "node_id": "leaf1",
                "text": "This is test content. " * 40,  # ~200 tokens
                "span_start": 0,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 200,
                "height": 0,
                "level_index": 0,
                "parent_id": "mid1",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "leaf2",
                "text": "This is test content. " * 40,  # ~200 tokens
                "span_start": 200,
                "span_end": 400,
                "document_id": "test-doc",
                "token_count": 200,
                "height": 0,
                "level_index": 0,
                "parent_id": "mid1",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "leaf3",
                "text": "This is test content. " * 40,  # ~200 tokens
                "span_start": 400,
                "span_end": 600,
                "document_id": "test-doc",
                "token_count": 200,
                "height": 0,
                "level_index": 0,
                "parent_id": "mid2",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "leaf4",
                "text": "This is test content. " * 40,  # ~200 tokens
                "span_start": 600,
                "span_end": 800,
                "document_id": "test-doc",
                "token_count": 200,
                "height": 0,
                "level_index": 0,
                "parent_id": "mid2",
                "left_child_id": None,
                "right_child_id": None,
            },
            # Middle level nodes (level 1)
            {
                "node_id": "mid1",
                "text": "Summary of leaves 1-2",
                "span_start": 0,
                "span_end": 400,
                "document_id": "test-doc",
                "token_count": 100,
                "height": 1,
                "level_index": 0,
                "parent_id": "root",
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
            },
            {
                "node_id": "mid2",
                "text": "Summary of leaves 3-4",
                "span_start": 400,
                "span_end": 800,
                "document_id": "test-doc",
                "token_count": 100,
                "height": 1,
                "level_index": 0,
                "parent_id": "root",
                "left_child_id": "leaf3",
                "right_child_id": "leaf4",
            },
            # Root node (level 2)
            {
                "node_id": "root",
                "text": "Root summary of all content",
                "span_start": 0,
                "span_end": 800,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 2,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "mid1",
                "right_child_id": "mid2",
            },
        ]

        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("leaf1", "mid1"),
                ("leaf2", "mid1"),
                ("leaf3", "mid2"),
                ("leaf4", "mid2"),
                ("mid1", "root"),
                ("mid2", "root"),
            ]
        )

        # Add embeddings for vector search
        vector_index.upsert(
            [
                (
                    "leaf1",
                    [0.9] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 200,
                        "parent_id": "mid1",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "leaf2",
                    [0.8] * 1536,
                    {
                        "span_start": 200,
                        "span_end": 400,
                        "parent_id": "mid1",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "leaf3",
                    [0.7] * 1536,
                    {
                        "span_start": 400,
                        "span_end": 600,
                        "parent_id": "mid2",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "leaf4",
                    [0.6] * 1536,
                    {
                        "span_start": 600,
                        "span_end": 800,
                        "parent_id": "mid2",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "mid1",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 400,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
                (
                    "mid2",
                    [0.4] * 1536,
                    {
                        "span_start": 400,
                        "span_end": 800,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
                (
                    "root",
                    [0.3] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 800,
                        "parent_id": "",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
            ]
        )

        # Test with various budgets using real retriever
        query_config = QueryConfig(budget_tokens=1000)

        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            retriever = create_retriever(
                query_config,
                doc_store,
                document_id="test-doc",
                client=mock_retrieve,
                vector_index=vector_index,
            )

            test_queries = [
                "test content",
                "this is",
                "random query that might not match well",
                "test",
            ]

            for query in test_queries:
                # Test multiple strict budgets
                for budget in [250, 500, 750]:
                    result = retriever.retrieve(query, budget_tokens=budget)
                    assembled_text = assembler.assemble(result)
                    token_count = assembler.get_token_count(assembled_text)

                    # CRITICAL: Token count must NEVER exceed budget
                    assert (
                        token_count <= budget
                    ), f"Budget exceeded for query '{query}' with budget {budget}: {token_count} > {budget}"

                    # Also verify by encoding directly
                    actual_tokens = assembler.tokenizer.encode(assembled_text)
                    assert (
                        len(actual_tokens) <= budget
                    ), f"Actual token count exceeds budget: {len(actual_tokens)} > {budget}"

    def test_worst_case_parent_child_extraction(
        self,
        doc_store: DocumentStore,
        assembler: Assembler,
        sqlite_backend: SQLiteStorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test worst case where parent-child extraction could double content."""
        # Create a simple parent-child structure with precise token costs
        nodes: list[NodeDataDict] = [
            {
                "node_id": "expensive_leaf",
                "text": "Very long content. " * 60,  # ~300 tokens
                "span_start": 0,
                "span_end": 300,
                "document_id": "test-doc",
                "token_count": 300,
                "height": 0,
                "level_index": 0,
                "parent_id": "parent",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "parent",
                "text": "Summary of expensive leaf",
                "span_start": 0,
                "span_end": 300,
                "document_id": "test-doc",
                "token_count": 100,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "expensive_leaf",
                "right_child_id": None,
            },
        ]

        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch([("expensive_leaf", "parent")])

        # Add embeddings with high score for leaf
        vector_index.upsert(
            [
                (
                    "expensive_leaf",
                    [1.0] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 300,
                        "parent_id": "parent",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "parent",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 300,
                        "parent_id": "",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
            ]
        )

        query_config = QueryConfig(budget_tokens=500)

        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            retriever = create_retriever(
                query_config,
                doc_store,
                document_id="test-doc",
                client=mock_retrieve,
                vector_index=vector_index,
            )

            # Test that retriever respects budget even with parent + child preference
            result = retriever.retrieve(
                "expensive leaf", num_seeds=2, budget_tokens=250
            )
            assembled_text = assembler.assemble(result)
            token_count = assembler.get_token_count(assembled_text)

            assert (
                token_count <= 250
            ), f"Budget exceeded in worst case: {token_count} > 250"

            # Verify no content duplication - should pick parent over expensive child
            assert result.tiling is not None
            assert len(result.tiling) == 1
            assert "parent" in result.tiling

    @pytest.mark.slow_threshold(2.0)
    def test_conservative_num_seeds_calculation(
        self,
        doc_store: DocumentStore,
        assembler: Assembler,
        sqlite_backend: SQLiteStorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test that the conservative num_seeds calculation respects budget."""
        # Create multiple nodes with varying token costs
        nodes: list[NodeDataDict] = []

        # Create 10 leaf nodes with 100 tokens each
        for i in range(10):
            nodes.append(
                {
                    "node_id": f"leaf_{i}",
                    "text": f"Content {i}. " * 25,  # ~100 tokens each
                    "span_start": i * 100,
                    "span_end": (i + 1) * 100,
                    "document_id": "test-doc",
                    "token_count": 100,
                    "height": 0,
                    "level_index": 0,
                    "parent_id": "root",
                    "left_child_id": None,
                    "right_child_id": None,
                }
            )

        # Add root node
        nodes.append(
            {
                "node_id": "root",
                "text": "Root summary of all content",
                "span_start": 0,
                "span_end": 1000,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "leaf_0",
                "right_child_id": "leaf_1",
            }
        )

        doc_store.nodes.add_batch(nodes)

        # Update parent references for all leaves
        parent_refs = [(f"leaf_{i}", "root") for i in range(10)]
        doc_store.nodes.update_parent_references_batch(parent_refs)

        # Add embeddings for all nodes
        embeddings_data = []
        for i in range(10):
            embeddings_data.append(
                (
                    f"leaf_{i}",
                    [0.9 - i * 0.05] * 1536,  # Decreasing similarity scores
                    {
                        "span_start": i * 100,
                        "span_end": (i + 1) * 100,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                )
            )

        embeddings_data.append(
            (
                "root",
                [0.3] * 1536,
                {
                    "span_start": 0,
                    "span_end": 1000,
                    "parent_id": "",
                    "document_id": "test-doc",
                    "is_leaf": 0,
                },
            )
        )

        # Cast to precise type for upsert
        typed_entries: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = [(i, list(vec), meta) for (i, vec, meta) in embeddings_data]
        vector_index.upsert(typed_entries)

        query_config = QueryConfig(budget_tokens=1000)

        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            retriever = create_retriever(
                query_config,
                doc_store,
                document_id="test-doc",
                client=mock_retrieve,
                vector_index=vector_index,
            )

            # Test various budget sizes
            test_budgets = [250, 500, 750, 1200]

            for budget in test_budgets:
                # Calculate conservative num_seeds using the retriever's method
                conservative_num_seeds = (
                    retriever.budget_planner.calculate_conservative_num_seeds(
                        budget, "test-doc"
                    )
                )

                # Verify the calculation is at least 1
                assert (
                    conservative_num_seeds >= 1
                ), f"num_seeds calculation should be at least 1 for budget {budget}"

                # Test that using this num_seeds respects the budget
                result = retriever.retrieve(
                    "content",
                    num_seeds=conservative_num_seeds,
                    budget_tokens=budget,
                    document_id="test-doc",
                )

                assembled_text = assembler.assemble(result)
                final_token_count = assembler.get_token_count(assembled_text)

                assert (
                    final_token_count <= budget
                ), f"Budget {budget} exceeded with conservative_num_seeds={conservative_num_seeds}, final tokens={final_token_count}"

    def test_mixed_mode_budget_plus_num_seeds(
        self,
        doc_store: DocumentStore,
        assembler: Assembler,
        sqlite_backend: SQLiteStorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test mixed mode where both budget and num_seeds are specified."""
        # Create several nodes that could exceed budget if all selected
        nodes: list[NodeDataDict] = [
            {
                "node_id": "node1",
                "text": "Node 1 content. " * 30,  # ~120 tokens
                "span_start": 0,
                "span_end": 120,
                "document_id": "test-doc",
                "token_count": 120,
                "height": 0,
                "level_index": 0,
                "parent_id": "root",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "node2",
                "text": "Node 2 content. " * 30,  # ~120 tokens
                "span_start": 120,
                "span_end": 240,
                "document_id": "test-doc",
                "token_count": 120,
                "height": 0,
                "level_index": 0,
                "parent_id": "root",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "node3",
                "text": "Node 3 content. " * 30,  # ~120 tokens
                "span_start": 240,
                "span_end": 360,
                "document_id": "test-doc",
                "token_count": 120,
                "height": 0,
                "level_index": 0,
                "parent_id": "root",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "root",
                "text": "Root summary",
                "span_start": 0,
                "span_end": 360,
                "document_id": "test-doc",
                "token_count": 60,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "node1",
                "right_child_id": "node2",
            },
        ]

        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("node1", "root"), ("node2", "root"), ("node3", "root")]
        )

        # Add embeddings
        vector_index.upsert(
            [
                (
                    "node1",
                    [0.9] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 120,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "node2",
                    [0.8] * 1536,
                    {
                        "span_start": 120,
                        "span_end": 240,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "node3",
                    [0.7] * 1536,
                    {
                        "span_start": 240,
                        "span_end": 360,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "root",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 360,
                        "parent_id": "",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
            ]
        )

        query_config = QueryConfig(budget_tokens=800)

        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            retriever = create_retriever(
                query_config,
                doc_store,
                document_id="test-doc",
                client=mock_retrieve,
                vector_index=vector_index,
            )

            # Specify both budget and num_seeds
            budget = 200  # Should only allow 1-2 nodes
            num_seeds = 10  # Way more than budget allows

            result = retriever.retrieve(
                "test", num_seeds=num_seeds, budget_tokens=budget
            )

            # Should respect budget constraint via DP algorithm
            assert result.tiling is not None

            assembled_text = assembler.assemble(result)
            token_count = assembler.get_token_count(assembled_text)
            assert (
                token_count <= budget
            ), f"Budget exceeded with mixed mode: {token_count} > {budget}"

    def test_num_seeds_only_mode(
        self,
        doc_store: DocumentStore,
        assembler: Assembler,
        sqlite_backend: SQLiteStorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test num_seeds only mode (no budget enforcement)."""
        # Create a simple tree structure
        nodes: list[NodeDataDict] = [
            {
                "node_id": "leaf1",
                "text": "Test content. " * 50,  # ~200 tokens
                "span_start": 0,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 200,
                "height": 0,
                "level_index": 0,
                "parent_id": "root",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "leaf2",
                "text": "Other content. " * 50,  # ~200 tokens
                "span_start": 200,
                "span_end": 400,
                "document_id": "test-doc",
                "token_count": 200,
                "height": 0,
                "level_index": 0,
                "parent_id": "root",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "root",
                "text": "Root summary of the document",
                "span_start": 0,
                "span_end": 400,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
            },
        ]

        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("leaf1", "root"), ("leaf2", "root")]
        )

        # Add embeddings with high score for leaf1
        vector_index.upsert(
            [
                (
                    "leaf1",
                    [0.9] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 200,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "leaf2",
                    [0.2] * 1536,
                    {
                        "span_start": 200,
                        "span_end": 400,
                        "parent_id": "root",
                        "document_id": "test-doc",
                        "is_leaf": 1,
                    },
                ),
                (
                    "root",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 400,
                        "parent_id": "",
                        "document_id": "test-doc",
                        "is_leaf": 0,
                    },
                ),
            ]
        )

        query_config = QueryConfig(budget_tokens=1000)

        with mock_openai_context() as (mock_index, mock_retrieve, mock_assemble):
            retriever = create_retriever(
                query_config,
                doc_store,
                document_id="test-doc",
                client=mock_retrieve,
                vector_index=vector_index,
            )

            # Retrieve with only num_seeds (no budget)
            num_seeds = 5
            result = retriever.retrieve(
                "test", num_seeds=num_seeds, budget_tokens=None, document_id="test-doc"
            )

            # Should have nodes from DP algorithm
            assert result.tiling is not None
            assert len(result.tiling) > 0

            # Assembly should work without budget constraints
            assembled_text = assembler.assemble(result)
            assert len(assembled_text) > 0


@pytest.mark.usefixtures("sqlite_backend")
class TestBudgetValidation:
    """Test that budget validation catches overflows."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("test-doc")

    def test_budget_validation_catches_overflow(self, doc_store: DocumentStore) -> None:
        """Test that validation fails when tiling exceeds budget."""
        # Create some nodes with known token costs
        nodes: list[NodeDataDict] = [
            {
                "node_id": "node1",
                "text": "test " * 20,  # ~20 tokens
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 20,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "node2",
                "text": "test " * 30,  # ~30 tokens
                "span_start": 100,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 30,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Create tiling that would exceed a small budget
        tiling = ["node1", "node2"]  # ~20 + ~30 = ~50 tokens

        # Validate with budget that's too small
        error = validate_tiling(tiling, doc_store, budget_tokens=40)

        assert error is not None
        assert "exceeds budget" in error
        assert "> 40 budget" in error

    def test_budget_validation_passes_within_budget(
        self, doc_store: DocumentStore
    ) -> None:
        """Test that validation passes when tiling is within budget."""
        # Create a node
        nodes: list[NodeDataDict] = [
            {
                "node_id": "node1",
                "text": "test " * 10,  # ~10 tokens
                "span_start": 0,
                "span_end": 50,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
        ]

        doc_store.nodes.add_batch(nodes)

        # Create tiling within budget
        tiling = ["node1"]  # ~10 tokens

        # Validate with sufficient budget
        error = validate_tiling(tiling, doc_store, budget_tokens=100)

        assert error is None

    def test_budget_validation_with_parent_child(
        self, doc_store: DocumentStore
    ) -> None:
        """Test budget validation with parent and child nodes."""
        # Create parent-child structure
        nodes: list[NodeDataDict] = [
            {
                "node_id": "left_child",
                "text": "left part " * 10,  # ~10 tokens
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": "parent",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "right_child",
                "text": "right part " * 10,  # ~10 tokens
                "span_start": 100,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": "parent",
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "parent",
                "text": "Summary of left and right parts",
                "span_start": 0,
                "span_end": 200,
                "document_id": "test-doc",
                "token_count": 15,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "left_child",
                "right_child_id": "right_child",
            },
        ]

        doc_store.nodes.add_batch(nodes)
        doc_store.nodes.update_parent_references_batch(
            [("left_child", "parent"), ("right_child", "parent")]
        )

        # Create tiling with child nodes
        tiling = ["left_child", "right_child"]  # ~20 tokens total

        # Should pass with budget of 50
        error = validate_tiling(tiling, doc_store, budget_tokens=50)
        assert error is None

        # Should fail with budget of 15
        error = validate_tiling(tiling, doc_store, budget_tokens=15)
        assert error is not None
        assert "exceeds budget" in error
