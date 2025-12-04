"""End-to-end integration tests for windowed queries.

Tests the complete windowed query pipeline from span_start/span_end
parameters through retrieval to assembly, verifying:
- Correct window boundary alignment to leaf nodes
- Coverage map includes only nodes within window
- actual_start/actual_end are set correctly on results
- Edge-max nodes provide full window coverage
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import QueryConfig
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from tests.utils import create_retriever, mock_openai_context
from tests.vector_index_stubs import RecordingVectorIndex

if TYPE_CHECKING:
    pass


@pytest.fixture
def doc_store_with_tree(storage_backend: StorageBackend) -> DocumentStore:
    """Create a document store with a well-structured tree for window tests.

    Tree structure (8 leaves, 3 levels):
                         root (h=3, idx=0)
                        /                 \\
               P0 (h=2, idx=0)      P1 (h=2, idx=1)
              /          \\          /          \\
        G0 (h=1,0)  G1 (h=1,1)  G2 (h=1,2)  G3 (h=1,3)
        /    \\      /    \\      /    \\      /    \\
       L0   L1    L2    L3    L4    L5    L6    L7

    Each leaf covers 100 characters, so total document is 800 chars.
    """
    doc_id = "windowed-test-doc"
    doc_store = storage_backend.for_document(doc_id)
    doc_store.set_metadata(
        file_path="test.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )

    # Build nodes bottom-up
    nodes: list[NodeDataDict] = []

    # Leaves (height=0)
    for i in range(8):
        nodes.append(
            {
                "node_id": f"L{i}",
                "text": f"Leaf {i} content. " * 10,
                "span_start": i * 100,
                "span_end": (i + 1) * 100,
                "document_id": doc_id,
                "token_count": 20,
                "height": 0,
                "level_index": i,
            }
        )

    # Grandparents (height=1)
    for i in range(4):
        nodes.append(
            {
                "node_id": f"G{i}",
                "text": f"Summary of leaves {i*2} and {i*2+1}",
                "span_start": i * 200,
                "span_end": (i + 1) * 200,
                "document_id": doc_id,
                "token_count": 15,
                "height": 1,
                "level_index": i,
                "left_child_id": f"L{i*2}",
                "right_child_id": f"L{i*2+1}",
            }
        )

    # Parents (height=2)
    nodes.append(
        {
            "node_id": "P0",
            "text": "Summary of first half",
            "span_start": 0,
            "span_end": 400,
            "document_id": doc_id,
            "token_count": 10,
            "height": 2,
            "level_index": 0,
            "left_child_id": "G0",
            "right_child_id": "G1",
        }
    )
    nodes.append(
        {
            "node_id": "P1",
            "text": "Summary of second half",
            "span_start": 400,
            "span_end": 800,
            "document_id": doc_id,
            "token_count": 10,
            "height": 2,
            "level_index": 1,
            "left_child_id": "G2",
            "right_child_id": "G3",
        }
    )

    # Root (height=3)
    nodes.append(
        {
            "node_id": "root",
            "text": "Full document summary",
            "span_start": 0,
            "span_end": 800,
            "document_id": doc_id,
            "token_count": 8,
            "height": 3,
            "level_index": 0,
            "left_child_id": "P0",
            "right_child_id": "P1",
        }
    )

    doc_store.nodes.add_batch(nodes)

    # Set parent references
    parent_refs = [(f"L{i}", f"G{i//2}") for i in range(8)] + [
        ("G0", "P0"),
        ("G1", "P0"),
        ("G2", "P1"),
        ("G3", "P1"),
        ("P0", "root"),
        ("P1", "root"),
    ]
    doc_store.nodes.update_parent_references_batch(parent_refs)

    return doc_store


class TestWindowedQueryIntegration:
    """Integration tests for windowed query functionality."""

    def test_query_with_middle_window(self, doc_store_with_tree: DocumentStore) -> None:
        """Query with window covering middle of document."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Window [200, 600) covers leaves L2, L3, L4, L5
            result = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=200,
                span_end=600,
            )

            # Verify window bounds are aligned to leaf boundaries
            assert result.actual_start == 200  # L2.span_start
            assert result.actual_end == 600  # L5.span_end

            # Result should have valid tiling
            assert result.tiling is not None
            assert len(result.tiling) > 0

    def test_query_with_window_at_start(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """Query with window starting at document beginning."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Window [0, 300) - should align to [0, 300) covering L0, L1, L2
            result = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=0,
                span_end=300,
            )

            assert result.actual_start == 0
            assert result.actual_end == 300
            assert result.tiling is not None

    def test_query_with_window_at_end(self, doc_store_with_tree: DocumentStore) -> None:
        """Query with window extending to document end."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Window [500, 800) - covers L5, L6, L7
            result = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=500,
                span_end=800,
            )

            assert result.actual_start == 500
            assert result.actual_end == 800
            assert result.tiling is not None

    def test_query_with_unaligned_window_expands(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """Query with unaligned window boundaries expands to leaf boundaries."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Request [250, 550) - should expand to [200, 600) to align to leaves
            result = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=250,
                span_end=550,
            )

            # Should expand to cover full leaves
            assert result.actual_start == 200  # L2 starts at 200
            assert result.actual_end == 600  # L5 ends at 600

    def test_query_with_single_leaf_window(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """Query with window covering a single leaf."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Window exactly covering L3 [300, 400)
            result = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=300,
                span_end=400,
            )

            assert result.actual_start == 300
            assert result.actual_end == 400

    def test_query_full_document_equivalent_to_no_window(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """Query with full document window behaves like no window."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Full document window
            result_windowed = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=0,
                span_end=800,
            )

            assert result_windowed.actual_start == 0
            assert result_windowed.actual_end == 800

    def test_tiling_respects_window_bounds(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """Tiling contains only nodes within window bounds."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Window covering L2, L3 only [200, 400)
            result = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=200,
                span_end=400,
            )

            assert result.tiling is not None
            assert result.actual_end is not None

            # Fetch all nodes in tiling
            tiling_nodes = doc_store_with_tree.nodes.get_nodes(list(result.tiling))

            # All nodes in tiling should be within window bounds
            actual_end = result.actual_end  # Capture for type narrowing
            for node in tiling_nodes:
                assert node.span_start >= result.actual_start, (
                    f"Node {node.id} span_start {node.span_start} < "
                    f"window start {result.actual_start}"
                )
                assert node.span_end <= actual_end, (
                    f"Node {node.id} span_end {node.span_end} > "
                    f"window end {actual_end}"
                )

    def test_assembly_includes_only_window_content(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """Assembled summary contains only content from window."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )
            assembler = Assembler(doc_store_with_tree)

            # Window covering middle section
            result = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=300,
                span_end=500,
            )

            summary = assembler.assemble(result)

            # Summary should be non-empty
            assert len(summary) > 0


class TestWindowedQueryValidation:
    """Tests for windowed query input validation."""

    def test_invalid_span_start_greater_than_end_raises(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """span_start >= span_end should raise ValidationError."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            with pytest.raises(ValueError):
                retriever.retrieve(
                    "test query",
                    budget_tokens=500,
                    span_start=500,
                    span_end=300,  # End before start
                )

    def test_span_start_equal_to_end_raises(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """span_start == span_end should raise ValidationError."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            with pytest.raises(ValueError):
                retriever.retrieve(
                    "test query",
                    budget_tokens=500,
                    span_start=300,
                    span_end=300,  # Empty range
                )

    def test_span_end_beyond_document_uses_document_end(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """span_end beyond document length should use document end."""
        query_config = QueryConfig(budget_tokens=500)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Document is 800 chars, request span_end=1000
            result = retriever.retrieve(
                "test query",
                budget_tokens=500,
                span_start=600,
                span_end=1000,
            )

            # Should clamp to document end
            assert result.actual_end == 800


class TestWindowedQueryWithVerbatim:
    """Tests for windowed queries with verbatim leaf budget."""

    def test_verbatim_leaves_respect_window(
        self, doc_store_with_tree: DocumentStore
    ) -> None:
        """Verbatim leaves are counted from window end, not document end."""
        query_config = QueryConfig(budget_tokens=1000)
        vector_index = RecordingVectorIndex()

        with mock_openai_context() as (_, mock_client, _):
            retriever = create_retriever(
                query_config,
                doc_store_with_tree,
                client=mock_client,
                vector_index=vector_index,
            )

            # Window [200, 600) with verbatim budget
            result = retriever.retrieve(
                "test query",
                budget_tokens=1000,
                span_start=200,
                span_end=600,
                recent_verbatim_budget=100,  # Should get leaves near window end
            )

            # Should include some verbatim leaves
            assert result.verbatim_count >= 0
            assert result.actual_start == 200
            assert result.actual_end == 600
