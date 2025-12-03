"""Tests for retrieve_for_context() used during contextual indexing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Generator
from collections.abc import Sequence as Seq
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragzoom.retrieve import Retriever

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.vector_filter import SpanEndLtFilter, VectorFilter
from ragzoom.document_store import DocumentStore
from ragzoom.vector_api import Vector


@pytest.mark.usefixtures("sqlite_backend")
class TestContextRetrieval:
    """Tests for retrieve_for_context() method."""

    @pytest.fixture
    def setup_tree_with_vectors(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> Generator[tuple[DocumentStore, Retriever], None, None]:
        """Set up a tree with nodes indexed in the vector store."""
        query_config = QueryConfig(budget_tokens=500, tiling_strategy="greedy")
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        doc_store = sqlite_store_factory("test-doc-context")

        # Create tree structure:
        #         root (spans 0-100)
        #        /    \
        #      P1      P2
        #     /  \    /  \
        #    L1  L2  L3  L4
        # Spans: L1=0-25, L2=25-50, P1=0-50
        #        L3=50-75, L4=75-100, P2=50-100, root=0-100

        nodes: list[NodeDataDict] = [
            {
                "node_id": "L1",
                "text": "Chapter 1: Introduction to dragons.",
                "span_start": 0,
                "span_end": 25,
                "document_id": "test-doc-context",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "L2",
                "text": "Chapter 2: Dragon habitats and behavior.",
                "span_start": 25,
                "span_end": 50,
                "document_id": "test-doc-context",
                "token_count": 10,
                "height": 0,
                "level_index": 1,
            },
            {
                "node_id": "L3",
                "text": "Chapter 3: Wizard spells and magic.",
                "span_start": 50,
                "span_end": 75,
                "document_id": "test-doc-context",
                "token_count": 10,
                "height": 0,
                "level_index": 2,
            },
            {
                "node_id": "L4",
                "text": "Chapter 4: Advanced magical techniques.",
                "span_start": 75,
                "span_end": 100,
                "document_id": "test-doc-context",
                "token_count": 10,
                "height": 0,
                "level_index": 3,
            },
            {
                "node_id": "P1",
                "text": "Summary: Dragons introduction and habitats.",
                "span_start": 0,
                "span_end": 50,
                "document_id": "test-doc-context",
                "token_count": 15,
                "height": 1,
                "level_index": 0,
                "left_child_id": "L1",
                "right_child_id": "L2",
            },
            {
                "node_id": "P2",
                "text": "Summary: Wizard spells and advanced magic.",
                "span_start": 50,
                "span_end": 100,
                "document_id": "test-doc-context",
                "token_count": 15,
                "height": 1,
                "level_index": 1,
                "left_child_id": "L3",
                "right_child_id": "L4",
            },
            {
                "node_id": "root",
                "text": "Full summary: Dragons and wizard magic.",
                "span_start": 0,
                "span_end": 100,
                "document_id": "test-doc-context",
                "token_count": 20,
                "height": 2,
                "level_index": 0,
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
            document_id="test-doc-context",
            api_key=operational_config.openai_api_key.get_secret_value(),
            vector_index=vi,
        )
        yield doc_store, retriever

    def test_retrieve_for_context_returns_preceding_content(
        self,
        setup_tree_with_vectors: tuple[DocumentStore, Retriever],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that retrieve_for_context returns content from before span_end_limit."""
        doc_store, retriever = setup_tree_with_vectors

        # Mock search to return nodes from the first half of the document
        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            k: int,
            filters: Seq[VectorFilter] | None = None,
        ) -> list[Vector]:
            import numpy as _np

            # Extract span limit from filters
            span_limit = 100  # Default
            if filters:
                for f in filters:
                    if isinstance(f, SpanEndLtFilter):
                        span_limit = f.threshold
                        break

            # Return nodes that are before the span limit
            results: list[Vector] = []
            node_data = [
                ("L1", 0, 25, "P1"),
                ("L2", 25, 50, "P1"),
                ("P1", 0, 50, "root"),
            ]

            for node_id, span_start, span_end, parent_id in node_data:
                if span_end < span_limit:
                    results.append(
                        Vector(
                            id=node_id,
                            vec=_np.ones(1536, dtype=_np.float32),
                            meta={
                                "document_id": "test-doc-context",
                                "span_start": span_start,
                                "span_end": span_end,
                                "parent_id": parent_id,
                                "is_leaf": 1 if node_id.startswith("L") else 0,
                            },
                            model_id="text-embedding-3-small",
                            dim=1536,
                        )
                    )

            return results[:k] if results else []

        retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]
        retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
            lambda query, document_id=None: [0.5] * 1536
        )

        # Query for context before span 75 (should get content from first half)
        result = asyncio.run(
            retriever.retrieve_for_context(
                query_text="Wizard spells",
                span_end_limit=75,
                budget_tokens=200,
                document_id="test-doc-context",
            )
        )

        # Should get assembled text from preceding nodes
        assert result != ""
        # The result should contain content from nodes before span 75
        assert "dragon" in result.lower() or "summary" in result.lower()

    def test_retrieve_for_context_empty_when_no_preceding(
        self,
        setup_tree_with_vectors: tuple[DocumentStore, Retriever],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that retrieve_for_context returns empty string when no preceding content."""
        doc_store, retriever = setup_tree_with_vectors

        # Mock search to return nothing (simulating first node with no preceding content)
        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            k: int,
            filters: Seq[VectorFilter] | None = None,
        ) -> list[Vector]:
            # No content before span_start=0
            return []

        retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]
        retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
            lambda query, document_id=None: [0.5] * 1536
        )

        # Query for context at the very beginning
        result = asyncio.run(
            retriever.retrieve_for_context(
                query_text="Introduction",
                span_end_limit=0,
                budget_tokens=200,
                document_id="test-doc-context",
            )
        )

        # Should return empty string
        assert result == ""

    def test_retrieve_for_context_respects_span_filter(
        self,
        setup_tree_with_vectors: tuple[DocumentStore, Retriever],
        sqlite_backend: SQLiteStorageBackend,
    ) -> None:
        """Test that SpanEndLtFilter is properly applied."""
        doc_store, retriever = setup_tree_with_vectors

        captured_filters: list[VectorFilter] = []

        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            k: int,
            filters: Seq[VectorFilter] | None = None,
        ) -> list[Vector]:
            import numpy as _np

            # Capture filters for assertion
            if filters:
                captured_filters.extend(filters)

            # Return a simple result
            return [
                Vector(
                    id="L1",
                    vec=_np.ones(1536, dtype=_np.float32),
                    meta={
                        "document_id": "test-doc-context",
                        "span_start": 0,
                        "span_end": 25,
                        "parent_id": "P1",
                        "is_leaf": 1,
                    },
                    model_id="text-embedding-3-small",
                    dim=1536,
                )
            ]

        retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]
        retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
            lambda query, document_id=None: [0.5] * 1536
        )

        asyncio.run(
            retriever.retrieve_for_context(
                query_text="Test query",
                span_end_limit=50,
                budget_tokens=200,
                document_id="test-doc-context",
            )
        )

        # Verify SpanEndLtFilter was applied with correct threshold
        span_filters = [f for f in captured_filters if isinstance(f, SpanEndLtFilter)]
        assert len(span_filters) == 1
        assert span_filters[0].threshold == 50
