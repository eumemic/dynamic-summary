"""Tests for time→span mapping logic in Retriever.

These tests verify that time-windowed queries correctly map to span-windowed
queries using the get_leaf_at_time_position() repository method.
"""

import asyncio
from collections.abc import Callable

import numpy as np
import pytest

from ragzoom.config import OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.vector_filter import VectorFilter
from ragzoom.document_store import DocumentStore
from ragzoom.retrieve import Retriever
from ragzoom.vector_api import Vector


def _create_temporal_doc_tree(
    doc_store: DocumentStore,
    doc_id: str,
    *,
    mark_temporal: bool = False,
) -> None:
    """Create a temporal document with a simple tree for testing.

    Tree structure:
        root (spans 0-120, time 1000-4000)
       /    \\
      L1     L2
    L1: spans 0-60, time 1000-2000
    L2: spans 60-120, time 2000-4000
    """
    # Ensure document record exists before adding nodes
    doc_store.set_metadata(
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )

    # Mark temporal if requested
    if mark_temporal:
        doc_store._doc_repo.set_document_is_temporal(doc_id, is_temporal=True)

    nodes: list[NodeDataDict] = [
        {
            "node_id": "L1",
            "text": "First chunk content",
            "span_start": 0,
            "span_end": 60,
            "document_id": doc_id,
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
            "text": "Second chunk content",
            "span_start": 60,
            "span_end": 120,
            "document_id": doc_id,
            "token_count": 50,
            "height": 0,
            "level_index": 1,
            "parent_id": None,
            "left_child_id": None,
            "right_child_id": None,
            "time_start": 2000.0,
            "time_end": 4000.0,
        },
        {
            "node_id": "root",
            "text": "Summary of document",
            "span_start": 0,
            "span_end": 120,
            "document_id": doc_id,
            "token_count": 100,
            "height": 1,
            "level_index": 0,
            "parent_id": None,
            "left_child_id": "L1",
            "right_child_id": "L2",
            "time_start": 1000.0,
            "time_end": 4000.0,
        },
    ]

    doc_store.nodes.add_batch(nodes)
    doc_store.nodes.update_parent_references_batch(
        [
            ("L1", "root"),
            ("L2", "root"),
        ]
    )


def _create_retriever(
    doc_store: DocumentStore,
    doc_id: str,
) -> Retriever:
    """Create a retriever with mocked embedding service."""
    from ragzoom.vector_factory import create_vector_index
    from tests.utils import create_retriever

    query_config = QueryConfig(budget_tokens=1000)
    operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

    vi = create_vector_index(
        "python", "sqlite:///:memory:", query_config.embedding_model
    )
    return create_retriever(
        query_config=query_config,
        store=doc_store,
        document_id=doc_id,
        api_key=operational_config.openai_api_key.get_secret_value(),
        vector_index=vi,
    )


def _mock_retriever(retriever: Retriever, doc_id: str) -> None:
    """Mock the retriever's vector search and embedding service."""
    from collections.abc import Sequence

    from numpy.typing import NDArray

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
                    "document_id": doc_id,
                    "span_start": 0,
                    "span_end": 60,
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


@pytest.mark.usefixtures("sqlite_backend")
class TestTimeToSpanMapping:
    """Test time→span mapping in retrieve_async()."""

    def test_time_window_maps_to_span_window(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """Time window query should use leaf spans for the window bounds.

        Overlap semantics per spec:
        - "start" position: earliest leaf where query_time <= leaf.time_end
        - "end" position: latest leaf where leaf.time_start <= query_time

        With L1 (time 1000-2000) and L2 (time 2000-4000):
        - At time 1500 (mid L1): only L1 overlaps
        - At time 1999 (just before L2): only L1 overlaps
        """
        doc_id = "temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=True)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        # Query with time window covering L1 only (time 1000-1999)
        # At time 1999, L2 (time_start=2000) doesn't overlap yet
        # L1 has span 0-60
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id=doc_id,
                time_start="1970-01-01T00:16:40Z",  # Unix 1000
                time_end="1970-01-01T00:33:19Z",  # Unix 1999
            )
        )

        # The mapping should find L1 as the boundary leaf
        # actual_start should be L1.span_start (0)
        # actual_end should be L1.span_end (60)
        assert result.actual_start == 0
        assert result.actual_end == 60

    def test_time_window_spanning_multiple_leaves(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """Time window spanning multiple leaves should use outer boundaries."""
        doc_id = "temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=True)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        # Query with time window covering both L1 and L2 (time 1000-4000)
        # L1 has span 0-60, L2 has span 60-120
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id=doc_id,
                time_start="1970-01-01T00:16:40Z",  # Unix 1000
                time_end="1970-01-01T01:06:40Z",  # Unix 4000
            )
        )

        # actual_start should be L1.span_start (0)
        # actual_end should be L2.span_end (120)
        assert result.actual_start == 0
        assert result.actual_end == 120

    def test_time_start_only_maps_to_span_start(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """Only time_start provided should map to span_start, span_end defaults to doc end.

        With start position semantics: earliest leaf where time_start <= leaf.time_end
        At time 2001, L1 (time_end=2000) doesn't overlap, so L2 is selected.
        """
        doc_id = "temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=True)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        # Query with only time_start at Unix 2001 (just after L1 ends)
        # L1 (time_end=2000) doesn't overlap at 2001
        # Should find L2 as the start leaf (span_start=60)
        # span_end should default to document end (120)
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id=doc_id,
                time_start="1970-01-01T00:33:21Z",  # Unix 2001
            )
        )

        # actual_start should be L2.span_start (60)
        # actual_end should default to doc_span_end (120)
        assert result.actual_start == 60
        assert result.actual_end == 120

    def test_time_end_only_maps_to_span_end(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """Only time_end provided should default span_start to 0, map time_end to span_end.

        With end position semantics: latest leaf where leaf.time_start <= time_end
        At time 1999, L2 (time_start=2000) doesn't overlap, so only L1 matches.
        """
        doc_id = "temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=True)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        # Query with only time_end at Unix 1999 (just before L2 starts)
        # L2 (time_start=2000) doesn't overlap at 1999
        # Should find L1 as the end leaf (span_end=60)
        # span_start should default to 0
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id=doc_id,
                time_end="1970-01-01T00:33:19Z",  # Unix 1999
            )
        )

        # actual_start should default to 0
        # actual_end should be L1.span_end (60)
        assert result.actual_start == 0
        assert result.actual_end == 60

    def test_time_window_outside_document_returns_empty(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """Time window with no overlapping data should return empty result, not error.

        Document has data at time 1000-4000. Querying with time_end=500 (before
        all data) should return an empty tiling rather than raising ValueError.
        """
        doc_id = "temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=True)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        # Query with time_end=500, before all document data (starts at 1000)
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id=doc_id,
                time_end="1970-01-01T00:08:20Z",  # Unix 500
            )
        )

        # Should return empty result, not raise an error
        assert result.tiling == []
        assert result.nodes == {}
        assert result.actual_start == 0
        assert result.actual_end == 0

    def test_time_start_after_all_data_returns_empty(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """time_start after all document data should return empty result.

        Document has data at time 1000-4000. Querying with time_start=5000 (after
        all data) should return an empty tiling rather than raising ValueError.
        """
        doc_id = "temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=True)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        # Query with time_start=5000, after all document data (ends at 4000)
        result = asyncio.run(
            retriever.retrieve_async(
                query="test query",
                num_seeds=1,
                budget_tokens=1000,
                document_id=doc_id,
                time_start="1970-01-01T01:23:20Z",  # Unix 5000
            )
        )

        # Should return empty result, not raise an error
        assert result.tiling == []
        assert result.nodes == {}
        assert result.actual_start == 0
        assert result.actual_end == 0


@pytest.mark.usefixtures("sqlite_backend")
class TestTimeQueryValidation:
    """Test validation for time-windowed queries."""

    def test_time_query_on_non_temporal_raises_error(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """Time query on non-temporal document should raise clear error."""
        doc_id = "non-temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        # mark_temporal=False by default (is_temporal defaults to False)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=False)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        with pytest.raises(ValueError, match="non-temporal"):
            asyncio.run(
                retriever.retrieve_async(
                    query="test query",
                    num_seeds=1,
                    budget_tokens=1000,
                    document_id=doc_id,
                    time_start="1970-01-01T00:16:40Z",
                )
            )

    def test_time_end_less_than_time_start_raises_error(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """time_end < time_start in query should raise validation error."""
        doc_id = "temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=True)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        with pytest.raises(ValueError, match="time_end.*time_start"):
            asyncio.run(
                retriever.retrieve_async(
                    query="test query",
                    num_seeds=1,
                    budget_tokens=1000,
                    document_id=doc_id,
                    time_start="1970-01-01T01:00:00Z",  # Unix 3600
                    time_end="1970-01-01T00:30:00Z",  # Unix 1800
                )
            )

    def test_invalid_timestamp_format_raises_error(
        self,
        sqlite_store_factory: Callable[[str | None], DocumentStore],
    ) -> None:
        """Invalid ISO 8601 format should raise clear error."""
        doc_id = "temporal-doc"
        doc_store = sqlite_store_factory(doc_id)
        _create_temporal_doc_tree(doc_store, doc_id, mark_temporal=True)

        retriever = _create_retriever(doc_store, doc_id)
        _mock_retriever(retriever, doc_id)

        # Missing timezone info
        with pytest.raises(ValueError, match="timezone"):
            asyncio.run(
                retriever.retrieve_async(
                    query="test query",
                    num_seeds=1,
                    budget_tokens=1000,
                    document_id=doc_id,
                    time_start="1970-01-01T00:16:40",  # No timezone
                )
            )
