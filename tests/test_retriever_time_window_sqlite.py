"""Tests for Retriever time_start/time_end parameter acceptance."""

import asyncio
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ragzoom.retrieve import Retriever

from ragzoom.config import OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.vector_filter import VectorFilter
from ragzoom.document_store import DocumentStore
from ragzoom.vector_api import Vector


@pytest.mark.usefixtures("sqlite_backend")
class TestRetrieverAcceptsTimeWindow:
    """Tests that Retriever.retrieve_async() accepts time_start and time_end parameters."""

    @pytest.fixture
    def retriever_with_temporal_tree(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> "Retriever":
        """Set up a retriever with a tree structure containing temporal metadata."""
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create document store
        doc_store = sqlite_store_factory("test-doc")

        # Create a simple tree with temporal metadata:
        #     root
        #    /    \
        #   L1     L2
        # L1: time 1000-2000 (spans 0-40)
        # L2: time 2000-3000 (spans 40-80)

        nodes: list[NodeDataDict] = [
            {
                "node_id": "L1",
                "text": "Chapter 1 content",
                "span_start": 0,
                "span_end": 40,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
                "time_start": 1000.0,
                "time_end": 2000.0,
            },
            {
                "node_id": "L2",
                "text": "Chapter 2 content",
                "span_start": 40,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 50,
                "height": 0,
                "level_index": 1,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
                "time_start": 2000.0,
                "time_end": 3000.0,
            },
            {
                "node_id": "root",
                "text": "Full document summary",
                "span_start": 0,
                "span_end": 80,
                "document_id": "test-doc",
                "token_count": 100,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "L1",
                "right_child_id": "L2",
                "time_start": 1000.0,
                "time_end": 3000.0,
            },
        ]

        doc_store.nodes.add_batch(nodes)

        # Set parent references
        doc_store.nodes.update_parent_references_batch(
            [
                ("L1", "root"),
                ("L2", "root"),
            ]
        )

        # Mark document as temporal since time-windowed queries require it
        doc_store.set_metadata(
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc_store._doc_repo.set_document_is_temporal("test-doc", is_temporal=True)

        from ragzoom.vector_factory import create_vector_index
        from tests.utils import create_retriever

        vi = create_vector_index(
            "python", "sqlite:///:memory:", query_config.embedding_model
        )
        return create_retriever(
            query_config=query_config,
            store=doc_store,
            document_id="test-doc",
            api_key=operational_config.openai_api_key.get_secret_value(),
            vector_index=vi,
        )

    def _mock_retriever_for_test(self, retriever: "Retriever") -> None:
        """Mock the retriever's vector search to return a simple result."""

        def mock_search_similar(
            query_embedding: list[float] | NDArray[np.float64],
            k: int,
            filters: Sequence[VectorFilter] | None = None,
        ) -> list[Vector]:
            return [
                Vector(
                    id="L1",
                    vec=np.ones(1536, dtype=np.float32),
                    meta={
                        "document_id": "test-doc",
                        "span_start": 0,
                        "span_end": 40,
                        "parent_id": "root",
                        "is_leaf": 1,
                    },
                    model_id="text-embedding-3-small",
                    dim=1536,
                )
            ]

        retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]
        retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
            lambda query, document_id=None: [0.3] * 1536
        )

    def test_retrieve_async_accepts_time_start_parameter(
        self,
        retriever_with_temporal_tree: "Retriever",
    ) -> None:
        """Test that retrieve_async() accepts time_start parameter."""
        self._mock_retriever_for_test(retriever_with_temporal_tree)

        # Use timestamp within test data range (1000-3000)
        # Unix 1500 = 1970-01-01T00:25:00Z
        result = asyncio.run(
            retriever_with_temporal_tree.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
                time_start="1970-01-01T00:25:00Z",
            )
        )

        # Basic sanity check - retrieval completed
        assert result.tiling is not None

    def test_retrieve_async_accepts_time_end_parameter(
        self,
        retriever_with_temporal_tree: "Retriever",
    ) -> None:
        """Test that retrieve_async() accepts time_end parameter."""
        self._mock_retriever_for_test(retriever_with_temporal_tree)

        # Use timestamp within test data range (1000-3000)
        # Unix 2500 = 1970-01-01T00:41:40Z
        result = asyncio.run(
            retriever_with_temporal_tree.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
                time_end="1970-01-01T00:41:40Z",
            )
        )

        # Basic sanity check - retrieval completed
        assert result.tiling is not None

    def test_retrieve_async_accepts_both_time_parameters(
        self,
        retriever_with_temporal_tree: "Retriever",
    ) -> None:
        """Test that retrieve_async() accepts both time_start and time_end."""
        self._mock_retriever_for_test(retriever_with_temporal_tree)

        # Use timestamps within test data range (1000-3000)
        result = asyncio.run(
            retriever_with_temporal_tree.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id="test-doc",
                time_start="1970-01-01T00:20:00Z",  # Unix 1200
                time_end="1970-01-01T00:41:40Z",  # Unix 2500
            )
        )

        # Basic sanity check - retrieval completed
        assert result.tiling is not None

    def test_retrieve_sync_accepts_time_parameters(
        self,
        retriever_with_temporal_tree: "Retriever",
    ) -> None:
        """Test that synchronous retrieve() also accepts time parameters."""
        self._mock_retriever_for_test(retriever_with_temporal_tree)

        # Use timestamps within test data range (1000-3000)
        result = retriever_with_temporal_tree.retrieve(
            query="test query",
            num_seeds=1,
            budget_tokens=1000,
            document_id="test-doc",
            time_start="1970-01-01T00:20:00Z",  # Unix 1200
            time_end="1970-01-01T00:41:40Z",  # Unix 2500
        )

        # Basic sanity check - retrieval completed
        assert result.tiling is not None
