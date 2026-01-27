"""Tests for Temporal Document APIs.

Tests document-status and truncate_from_time APIs as specified in
specs/temporal-document-apis.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import pytest

from ragzoom.server.servicers import complete_forest_size

if TYPE_CHECKING:
    from ragzoom.contracts.node_repository import NodeDataDict
    from ragzoom.contracts.storage_backend import StorageBackend
    from ragzoom.document_store import DocumentStore


class TestCompleteForestSize:
    """Tests for the complete_forest_size helper function.

    The formula is: 2N - popcount(N)
    where popcount(N) is the number of 1-bits in N's binary representation.

    This represents the total nodes (leaves + inner) in a complete binary forest.
    """

    def test_complete_forest_size_zero(self) -> None:
        """Zero leaves means zero nodes."""
        assert complete_forest_size(0) == 0

    def test_complete_forest_size_negative(self) -> None:
        """Negative leaf count should return 0."""
        assert complete_forest_size(-1) == 0
        assert complete_forest_size(-100) == 0

    def test_complete_forest_size_powers_of_two(self) -> None:
        """Powers of two have popcount=1, so 2N - 1.

        These form a single perfect binary tree.
        """
        # 1 leaf (0b1): 2*1 - 1 = 1
        assert complete_forest_size(1) == 1
        # 2 leaves (0b10): 2*2 - 1 = 3 (2 leaves + 1 root)
        assert complete_forest_size(2) == 3
        # 4 leaves (0b100): 2*4 - 1 = 7 (perfect tree of depth 2)
        assert complete_forest_size(4) == 7
        # 8 leaves (0b1000): 2*8 - 1 = 15
        assert complete_forest_size(8) == 15
        # 16 leaves (0b10000): 2*16 - 1 = 31
        assert complete_forest_size(16) == 31

    def test_complete_forest_size_mixed(self) -> None:
        """Non-power-of-two counts have popcount > 1.

        These form a forest of multiple perfect binary trees.
        """
        # 3 leaves (0b11): popcount=2, 2*3 - 2 = 4
        # (tree of 2 + single leaf = 3 nodes + the root... wait, no)
        # Actually: 3 leaves, 1 inner node (pairs 2 leaves), so 3+1=4
        assert complete_forest_size(3) == 4

        # 5 leaves (0b101): popcount=2, 2*5 - 2 = 8
        # Tree of 4 (7 nodes) + 1 leaf = 8 nodes
        assert complete_forest_size(5) == 8

        # 6 leaves (0b110): popcount=2, 2*6 - 2 = 10
        # Tree of 4 + tree of 2 = 7 + 3 = 10
        assert complete_forest_size(6) == 10

        # 7 leaves (0b111): popcount=3, 2*7 - 3 = 11
        # Tree of 4 (7) + tree of 2 (3) + leaf (1) = 11
        assert complete_forest_size(7) == 11

        # 100 leaves (0b1100100): popcount=3, 2*100 - 3 = 197
        assert complete_forest_size(100) == 197

    def test_complete_forest_size_formula_correctness(self) -> None:
        """Verify formula against explicit calculation for small values."""
        # For N leaves, a complete binary forest has:
        # - N leaves
        # - N - popcount(N) inner nodes
        # Total = 2N - popcount(N)

        for n in range(1, 100):
            popcount = bin(n).count("1")
            expected = 2 * n - popcount
            actual = complete_forest_size(n)
            assert (
                actual == expected
            ), f"Failed for n={n}: expected {expected}, got {actual}"


class TestDocumentStoreNodeCount:
    """Tests for DocumentStore.get_node_count() method.

    Verifies that the document store correctly returns the total count
    of nodes (leaves + inner nodes) for a document.
    """

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> DocumentStore:
        """Create a document store with test metadata."""

        doc_store = storage_backend.for_document("test-doc")
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        return doc_store

    def test_document_store_node_count_empty(self, doc_store: DocumentStore) -> None:
        """Empty document returns 0 node count."""
        assert doc_store.get_node_count() == 0

    def test_document_store_node_count_leaves_only(
        self, doc_store: DocumentStore
    ) -> None:
        """Document with only leaves returns leaf count."""
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"leaf-{i}",
                "text": f"Leaf text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "token_count": 5,
                "height": 0,
                "level_index": i,
            }
            for i in range(5)
        ]
        doc_store.nodes.add_batch(nodes_data)

        assert doc_store.get_node_count() == 5

    def test_document_store_node_count_with_inner_nodes(
        self, doc_store: DocumentStore
    ) -> None:
        """Document with leaves and inner nodes returns total count."""
        # Create 4 leaves
        leaves: list[NodeDataDict] = [
            {
                "node_id": f"leaf-{i}",
                "text": f"Leaf text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "token_count": 5,
                "height": 0,
                "level_index": i,
            }
            for i in range(4)
        ]
        doc_store.nodes.add_batch(leaves)

        # Create 2 inner nodes at height 1
        inner_h1: list[NodeDataDict] = [
            {
                "node_id": "inner-0",
                "text": "Summary of leaves 0-1",
                "span_start": 0,
                "span_end": 20,
                "token_count": 10,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf-0",
                "right_child_id": "leaf-1",
            },
            {
                "node_id": "inner-1",
                "text": "Summary of leaves 2-3",
                "span_start": 20,
                "span_end": 40,
                "token_count": 10,
                "height": 1,
                "level_index": 1,
                "left_child_id": "leaf-2",
                "right_child_id": "leaf-3",
            },
        ]
        doc_store.nodes.add_batch(inner_h1)

        # Create 1 root at height 2
        root: list[NodeDataDict] = [
            {
                "node_id": "root",
                "text": "Summary of all",
                "span_start": 0,
                "span_end": 40,
                "token_count": 15,
                "height": 2,
                "level_index": 0,
                "left_child_id": "inner-0",
                "right_child_id": "inner-1",
            },
        ]
        doc_store.nodes.add_batch(root)

        # 4 leaves + 2 inner + 1 root = 7 total nodes
        assert doc_store.get_node_count() == 7


class TestDocumentStoreTemporalRange:
    """Tests for DocumentStore.get_temporal_range() method.

    Verifies that the document store correctly returns the temporal range
    (min time_start, max time_end) from leaf nodes only.
    """

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> DocumentStore:
        """Create a document store with test metadata."""
        doc_store = storage_backend.for_document("test-temporal-doc")
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        return doc_store

    def test_temporal_range_empty_document(self, doc_store: DocumentStore) -> None:
        """Empty document returns (None, None) for temporal range."""
        result = doc_store.get_temporal_range()
        assert result == (None, None)

    def test_temporal_range_non_temporal_document(
        self, doc_store: DocumentStore
    ) -> None:
        """Document with leaves but no timestamps returns (None, None)."""
        # Add leaves without timestamps
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"leaf-{i}",
                "text": f"Leaf text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "token_count": 5,
                "height": 0,
                "level_index": i,
            }
            for i in range(3)
        ]
        doc_store.nodes.add_batch(nodes_data)

        result = doc_store.get_temporal_range()
        assert result == (None, None)

    def test_temporal_range_leaves_only(self, doc_store: DocumentStore) -> None:
        """Document with temporal leaves returns correct min/max range."""
        # Create leaves with timestamps
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": "leaf-0",
                "text": "First message",
                "span_start": 0,
                "span_end": 10,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1100.0,
            },
            {
                "node_id": "leaf-1",
                "text": "Second message",
                "span_start": 10,
                "span_end": 20,
                "token_count": 5,
                "height": 0,
                "level_index": 1,
                "time_start": 1100.0,
                "time_end": 1200.0,
            },
            {
                "node_id": "leaf-2",
                "text": "Third message",
                "span_start": 20,
                "span_end": 30,
                "token_count": 5,
                "height": 0,
                "level_index": 2,
                "time_start": 1200.0,
                "time_end": 1300.0,
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        time_start, time_end = doc_store.get_temporal_range()
        assert time_start == 1000.0
        assert time_end == 1300.0

    def test_temporal_range_with_inner_nodes(self, doc_store: DocumentStore) -> None:
        """Temporal range only considers leaves, not inner nodes."""
        # Create leaves with timestamps
        leaves: list[NodeDataDict] = [
            {
                "node_id": "leaf-0",
                "text": "First",
                "span_start": 0,
                "span_end": 10,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1050.0,
            },
            {
                "node_id": "leaf-1",
                "text": "Second",
                "span_start": 10,
                "span_end": 20,
                "token_count": 5,
                "height": 0,
                "level_index": 1,
                "time_start": 1050.0,
                "time_end": 1100.0,
            },
        ]
        doc_store.nodes.add_batch(leaves)

        # Create inner node spanning both leaves (with wider time range)
        inner: list[NodeDataDict] = [
            {
                "node_id": "inner-0",
                "text": "Summary",
                "span_start": 0,
                "span_end": 20,
                "token_count": 8,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf-0",
                "right_child_id": "leaf-1",
                "time_start": 900.0,  # Wider range than leaves (shouldn't be used)
                "time_end": 1200.0,
            },
        ]
        doc_store.nodes.add_batch(inner)

        # Should only use leaf timestamps
        time_start, time_end = doc_store.get_temporal_range()
        assert time_start == 1000.0
        assert time_end == 1100.0

    def test_temporal_range_single_leaf(self, doc_store: DocumentStore) -> None:
        """Single temporal leaf returns that leaf's time range."""
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": "leaf-0",
                "text": "Only message",
                "span_start": 0,
                "span_end": 10,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1500.0,
                "time_end": 1600.0,
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        time_start, time_end = doc_store.get_temporal_range()
        assert time_start == 1500.0
        assert time_end == 1600.0

    def test_temporal_range_no_document_id(
        self, storage_backend: StorageBackend
    ) -> None:
        """Document store without document_id returns (None, None)."""
        doc_store = storage_backend.for_document(None)
        result = doc_store.get_temporal_range()
        assert result == (None, None)


class _MockServicerContext:
    """Mock gRPC servicer context for testing."""

    def invocation_metadata(self) -> list[tuple[str, str]] | None:
        return None

    async def abort(self, code: object, details: str) -> NoReturn:
        raise ValueError(f"Aborted with {code}: {details}")


class TestGetDocumentStatusServicer:
    """Tests for the GetDocumentStatus gRPC servicer method.

    Verifies that the servicer correctly returns document status including
    existence, completion metrics, and temporal range.
    """

    @pytest.fixture
    def doc_store(self, storage_backend: StorageBackend) -> DocumentStore:
        """Create a document store with test metadata."""
        doc_store = storage_backend.for_document("test-status-doc")
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        return doc_store

    @pytest.fixture
    def mock_state(self, storage_backend: StorageBackend) -> object:
        """Create a mock ServerState with the storage backend."""
        from unittest.mock import MagicMock

        state = MagicMock()
        state.store = storage_backend
        return state

    @pytest.mark.asyncio
    async def test_get_document_status_not_found(self, mock_state: object) -> None:
        """GetDocumentStatus returns exists=False for non-existent documents."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import WorkerServicer

        servicer = WorkerServicer(mock_state)  # type: ignore[arg-type]
        request_cls = getattr(pb2, "DocumentStatusRequest")
        request = request_cls(document_id="nonexistent-doc")
        context = _MockServicerContext()

        response = await servicer.GetDocumentStatus(request, context)

        assert getattr(response, "document_id") == "nonexistent-doc"
        assert getattr(response, "exists") is False
        assert getattr(response, "is_temporal") is False
        assert getattr(response, "leaf_count") == 0
        assert getattr(response, "node_count") == 0
        assert getattr(response, "complete_forest_size") == 0
        assert getattr(response, "completion_pct") == 0.0

    @pytest.mark.asyncio
    async def test_get_document_status_existing_document(
        self, mock_state: object, doc_store: DocumentStore
    ) -> None:
        """GetDocumentStatus returns correct metrics for existing documents."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import WorkerServicer

        # Add 4 leaves
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"leaf-{i}",
                "text": f"Leaf text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "token_count": 5,
                "height": 0,
                "level_index": i,
            }
            for i in range(4)
        ]
        doc_store.nodes.add_batch(nodes_data)

        servicer = WorkerServicer(mock_state)  # type: ignore[arg-type]
        request_cls = getattr(pb2, "DocumentStatusRequest")
        request = request_cls(document_id="test-status-doc")
        context = _MockServicerContext()

        response = await servicer.GetDocumentStatus(request, context)

        assert getattr(response, "document_id") == "test-status-doc"
        assert getattr(response, "exists") is True
        assert getattr(response, "is_temporal") is False  # No timestamps on leaves
        assert getattr(response, "leaf_count") == 4
        assert getattr(response, "node_count") == 4  # Only leaves, no inner nodes yet
        # complete_forest_size for 4 leaves: 2*4 - 1 = 7
        assert getattr(response, "complete_forest_size") == 7
        # completion: 4/7 * 100 ≈ 57.14%
        expected_pct = 4 / 7 * 100.0
        actual_pct: float = getattr(response, "completion_pct")
        assert abs(actual_pct - expected_pct) < 0.01

    @pytest.mark.asyncio
    async def test_get_document_status_completion_with_inner_nodes(
        self, mock_state: object, doc_store: DocumentStore
    ) -> None:
        """GetDocumentStatus shows higher completion when inner nodes exist."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import WorkerServicer

        # Add 4 leaves
        leaves: list[NodeDataDict] = [
            {
                "node_id": f"leaf-{i}",
                "text": f"Leaf text {i}",
                "span_start": i * 10,
                "span_end": (i + 1) * 10,
                "token_count": 5,
                "height": 0,
                "level_index": i,
            }
            for i in range(4)
        ]
        doc_store.nodes.add_batch(leaves)

        # Add 2 inner nodes at height 1
        inner_h1: list[NodeDataDict] = [
            {
                "node_id": "inner-0",
                "text": "Summary of leaves 0-1",
                "span_start": 0,
                "span_end": 20,
                "token_count": 10,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf-0",
                "right_child_id": "leaf-1",
            },
            {
                "node_id": "inner-1",
                "text": "Summary of leaves 2-3",
                "span_start": 20,
                "span_end": 40,
                "token_count": 10,
                "height": 1,
                "level_index": 1,
                "left_child_id": "leaf-2",
                "right_child_id": "leaf-3",
            },
        ]
        doc_store.nodes.add_batch(inner_h1)

        # Add root at height 2
        root: list[NodeDataDict] = [
            {
                "node_id": "root",
                "text": "Summary of all",
                "span_start": 0,
                "span_end": 40,
                "token_count": 15,
                "height": 2,
                "level_index": 0,
                "left_child_id": "inner-0",
                "right_child_id": "inner-1",
            },
        ]
        doc_store.nodes.add_batch(root)

        servicer = WorkerServicer(mock_state)  # type: ignore[arg-type]
        request_cls = getattr(pb2, "DocumentStatusRequest")
        request = request_cls(document_id="test-status-doc")
        context = _MockServicerContext()

        response = await servicer.GetDocumentStatus(request, context)

        assert getattr(response, "leaf_count") == 4
        assert getattr(response, "node_count") == 7  # 4 leaves + 2 inner + 1 root
        assert getattr(response, "complete_forest_size") == 7  # 2*4 - 1
        # 100% complete: 7/7
        assert getattr(response, "completion_pct") == 100.0

    @pytest.mark.asyncio
    async def test_get_document_status_temporal_range(
        self, mock_state: object, storage_backend: StorageBackend
    ) -> None:
        """GetDocumentStatus includes temporal range for temporal documents."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import WorkerServicer

        # Create a temporal document
        doc_store = storage_backend.for_document("test-temporal-status")
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        # Mark as temporal via the document repository
        doc_store._doc_repo.set_document_is_temporal(
            "test-temporal-status", is_temporal=True
        )

        # Add leaves with timestamps
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": "leaf-0",
                "text": "First message",
                "span_start": 0,
                "span_end": 10,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1737849600.0,  # 2025-01-26T00:00:00Z
                "time_end": 1737853200.0,  # 2025-01-26T01:00:00Z
            },
            {
                "node_id": "leaf-1",
                "text": "Second message",
                "span_start": 10,
                "span_end": 20,
                "token_count": 5,
                "height": 0,
                "level_index": 1,
                "time_start": 1737853200.0,  # 2025-01-26T01:00:00Z
                "time_end": 1737856800.0,  # 2025-01-26T02:00:00Z
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        servicer = WorkerServicer(mock_state)  # type: ignore[arg-type]
        request_cls = getattr(pb2, "DocumentStatusRequest")
        request = request_cls(document_id="test-temporal-status")
        context = _MockServicerContext()

        response = await servicer.GetDocumentStatus(request, context)

        assert getattr(response, "exists") is True
        assert getattr(response, "is_temporal") is True
        assert getattr(response, "time_start") == "2025-01-26T00:00:00Z"
        assert getattr(response, "time_end") == "2025-01-26T02:00:00Z"

    @pytest.mark.asyncio
    async def test_get_document_status_requires_document_id(
        self, mock_state: object
    ) -> None:
        """GetDocumentStatus returns error when document_id is empty."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import WorkerServicer

        servicer = WorkerServicer(mock_state)  # type: ignore[arg-type]
        request_cls = getattr(pb2, "DocumentStatusRequest")
        request = request_cls(document_id="")
        context = _MockServicerContext()

        with pytest.raises(ValueError, match="GetDocumentStatus requires"):
            await servicer.GetDocumentStatus(request, context)


class TestTruncateFromTimeVectors:
    """Tests for vector deletion during time-based truncation.

    Verifies that truncate_from_time properly removes vectors from the
    vector index for all deleted nodes.
    """

    @pytest.mark.asyncio
    async def test_truncate_from_time_removes_vectors(
        self, storage_backend: StorageBackend
    ) -> None:
        """Truncate from time should delete vectors for removed nodes.

        This test verifies acceptance criterion #7 from the spec:
        'truncate_from_time removes vectors for deleted nodes'
        """
        from typing import cast
        from unittest.mock import AsyncMock, MagicMock

        from ragzoom.config import IndexConfig
        from ragzoom.contracts.vector_index import VectorIndex
        from ragzoom.indexing.runtime import (
            IndexerRuntime,
            TruncateFromTimeResult,
        )
        from ragzoom.server.append_executor import AppendExecutor
        from ragzoom.server.indexing_engine import IndexingEngine

        doc_id = "test-truncate-vectors"

        # Create a temporal document with timestamped leaves
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc_store._doc_repo.set_document_is_temporal(doc_id, is_temporal=True)

        # Add leaves with timestamps spanning a time range
        # leaf-0: t=1000-1100, leaf-1: t=1100-1200, leaf-2: t=1200-1300
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"{doc_id}-leaf-0",
                "text": "First message content here",
                "span_start": 0,
                "span_end": 100,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1100.0,
            },
            {
                "node_id": f"{doc_id}-leaf-1",
                "text": "Second message content here",
                "span_start": 100,
                "span_end": 200,
                "token_count": 5,
                "height": 0,
                "level_index": 1,
                "time_start": 1100.0,
                "time_end": 1200.0,
            },
            {
                "node_id": f"{doc_id}-leaf-2",
                "text": "Third message content here",
                "span_start": 200,
                "span_end": 300,
                "token_count": 5,
                "height": 0,
                "level_index": 2,
                "time_start": 1200.0,
                "time_end": 1300.0,
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        # Track vector delete calls
        delete_calls: list[dict[str, object]] = []

        class VectorIndexStub:
            """Stub that tracks delete calls."""

            def delete(
                self,
                filter: dict[str, object] | None = None,
                ids: list[str] | None = None,
            ) -> int:
                delete_calls.append({"filter": filter, "ids": ids})
                return len(ids) if ids else 0

            def upsert(
                self,
                items: list[tuple[str, list[float], dict[str, object]]],
            ) -> None:
                pass

            def search_similar(
                self,
                query_embedding: object,
                k: int,
                filters: object = None,
            ) -> list[object]:
                return []

            def get_vectors(self, ids: list[str]) -> list[object]:
                return []

        def vector_factory(model: str) -> VectorIndexStub:
            return VectorIndexStub()

        # Create runtime with stub vector index
        index_config = IndexConfig.load()
        mock_append_executor = MagicMock(spec=AppendExecutor)
        mock_indexing_engine = MagicMock(spec=IndexingEngine)
        mock_indexing_engine.cancel_document = AsyncMock(return_value=None)

        runtime = IndexerRuntime(
            store=storage_backend,
            index_config=index_config,
            append_executor=cast(AppendExecutor, mock_append_executor),
            indexing_engine=cast(IndexingEngine, mock_indexing_engine),
            telemetry_manager=None,
            vector_index_factory=cast(VectorIndex, vector_factory),  # type: ignore[arg-type]
        )

        # Truncate from cutoff_time=1150.0 (should delete leaf-1 and leaf-2)
        # because their time_end > cutoff_time (1200 > 1150 and 1300 > 1150)
        session = runtime.get_session(doc_id)
        cutoff_time = 1150.0
        result = await session.truncate_from_time(cutoff_time)

        # Verify result type and contents
        assert isinstance(result, TruncateFromTimeResult)
        assert result.document_id == doc_id
        assert result.cutoff_time == cutoff_time
        # Should have deleted leaf-1 and leaf-2
        assert len(result.deleted_node_ids) == 2
        assert f"{doc_id}-leaf-1" in result.deleted_node_ids
        assert f"{doc_id}-leaf-2" in result.deleted_node_ids

        # Verify vector_index.delete was called with the deleted node IDs
        assert len(delete_calls) == 1, "Vector index delete should be called once"
        delete_call = delete_calls[0]
        # The delete should use filter with node_id $in list
        assert delete_call["filter"] is not None
        filter_dict = cast(dict[str, object], delete_call["filter"])
        assert "node_id" in filter_dict
        node_id_filter = cast(dict[str, object], filter_dict["node_id"])
        assert "$in" in node_id_filter
        deleted_ids = set(cast(list[str], node_id_filter["$in"]))
        assert deleted_ids == {f"{doc_id}-leaf-1", f"{doc_id}-leaf-2"}

        # Verify the kept leaf (leaf-0) still exists in storage
        remaining_leaves = list(doc_store.nodes.get_leaves())
        assert len(remaining_leaves) == 1
        assert remaining_leaves[0].id == f"{doc_id}-leaf-0"


class TestTruncateFromTimeOrphaning:
    """Tests for orphaning behavior during time-based truncation.

    Verifies acceptance criterion #6: truncate_from_time correctly orphans
    kept children by setting parent_id = NULL when their parents are deleted.
    """

    def test_truncate_from_time_orphans_kept_children(
        self, storage_backend: StorageBackend
    ) -> None:
        """Kept children whose parents are deleted get parent_id = NULL.

        This test verifies acceptance criterion #6 from the spec:
        'truncate_from_time correctly orphans kept children (NULL parent_id)'

        Setup:
        - 2 leaves with time_end before cutoff (should be kept)
        - 1 inner node (parent) with time_end after cutoff (should be deleted)
        - The leaves reference their parent via parent_id

        After truncation:
        - Inner node is deleted
        - Both leaves remain but with parent_id = NULL (orphaned)
        """
        doc_id = "test-truncate-orphan"

        # Create a temporal document
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc_store._doc_repo.set_document_is_temporal(doc_id, is_temporal=True)

        # Create structure:
        # - leaf-0: t=1000-1050 (before cutoff, kept)
        # - leaf-1: t=1050-1100 (before cutoff, kept)
        # - inner-0: t=1000-1200 (after cutoff, deleted) - parent of both leaves
        # Cutoff at t=1150: delete nodes where time_end > 1150

        inner_node_id = f"{doc_id}-inner-0"

        # Add leaves with parent_id referencing the inner node
        leaves: list[NodeDataDict] = [
            {
                "node_id": f"{doc_id}-leaf-0",
                "text": "First leaf content",
                "span_start": 0,
                "span_end": 100,
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1050.0,
                "parent_id": inner_node_id,
            },
            {
                "node_id": f"{doc_id}-leaf-1",
                "text": "Second leaf content",
                "span_start": 100,
                "span_end": 200,
                "token_count": 10,
                "height": 0,
                "level_index": 1,
                "time_start": 1050.0,
                "time_end": 1100.0,
                "parent_id": inner_node_id,
            },
        ]
        doc_store.nodes.add_batch(leaves)

        # Add parent node that spans both leaves but extends beyond cutoff
        inner: list[NodeDataDict] = [
            {
                "node_id": inner_node_id,
                "text": "Summary of both leaves",
                "span_start": 0,
                "span_end": 200,
                "token_count": 15,
                "height": 1,
                "level_index": 0,
                "left_child_id": f"{doc_id}-leaf-0",
                "right_child_id": f"{doc_id}-leaf-1",
                "time_start": 1000.0,
                "time_end": 1200.0,  # Extends beyond cutoff
            },
        ]
        doc_store.nodes.add_batch(inner)

        # Verify pre-condition: leaves have parent_id set
        all_nodes_before = {n.id: n for n in doc_store.nodes.get_all()}
        assert all_nodes_before[f"{doc_id}-leaf-0"].parent_id == inner_node_id
        assert all_nodes_before[f"{doc_id}-leaf-1"].parent_id == inner_node_id
        assert inner_node_id in all_nodes_before

        # Truncate at cutoff_time = 1150
        # This should delete inner-0 (time_end=1200 > 1150)
        # And orphan leaf-0 and leaf-1 (their parent was deleted)
        deleted_ids = storage_backend.delete_nodes_from_time(doc_id, cutoff_time=1150.0)

        # Verify the inner node was deleted
        assert inner_node_id in deleted_ids
        assert len(deleted_ids) == 1  # Only inner node deleted

        # Verify leaves still exist
        all_nodes_after = {n.id: n for n in doc_store.nodes.get_all()}
        assert f"{doc_id}-leaf-0" in all_nodes_after
        assert f"{doc_id}-leaf-1" in all_nodes_after
        assert inner_node_id not in all_nodes_after

        # KEY ASSERTION: Verify leaves are now orphaned (parent_id = NULL)
        assert (
            all_nodes_after[f"{doc_id}-leaf-0"].parent_id is None
        ), "Kept child should be orphaned (parent_id = NULL) when parent is deleted"
        assert (
            all_nodes_after[f"{doc_id}-leaf-1"].parent_id is None
        ), "Kept child should be orphaned (parent_id = NULL) when parent is deleted"


class TestTruncateFromTimeServicer:
    """Tests for the TruncateFromTime gRPC servicer method.

    Verifies that the servicer correctly validates requests and delegates
    to the runtime for time-based truncation of temporal documents.
    """

    @pytest.fixture
    def mock_state(self, storage_backend: StorageBackend) -> object:
        """Create a mock ServerState with the storage backend and runtime."""
        from typing import cast
        from unittest.mock import AsyncMock, MagicMock

        from ragzoom.config import IndexConfig
        from ragzoom.indexing.runtime import IndexerRuntime
        from ragzoom.server.append_executor import AppendExecutor
        from ragzoom.server.indexing_engine import IndexingEngine

        # Create a real runtime for proper integration testing
        index_config = IndexConfig.load()
        mock_append_executor = MagicMock(spec=AppendExecutor)
        mock_indexing_engine = MagicMock(spec=IndexingEngine)
        mock_indexing_engine.cancel_document = AsyncMock(return_value=None)

        def noop_vector_factory(model: str) -> object:
            """Vector factory that returns a stub with no-op methods."""
            stub = MagicMock()
            stub.delete = MagicMock(return_value=0)
            return stub

        runtime = IndexerRuntime(
            store=storage_backend,
            index_config=index_config,
            append_executor=cast(AppendExecutor, mock_append_executor),
            indexing_engine=cast(IndexingEngine, mock_indexing_engine),
            telemetry_manager=None,
            vector_index_factory=noop_vector_factory,  # type: ignore[arg-type]
        )

        state = MagicMock()
        state.store = storage_backend
        state.index_runtime = runtime
        return state

    @pytest.mark.asyncio
    async def test_truncate_from_time_servicer(
        self, mock_state: object, storage_backend: StorageBackend
    ) -> None:
        """TruncateFromTime servicer delegates to runtime and returns result."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import IndexerServicer

        doc_id = "test-truncate-servicer"

        # Create a temporal document with timestamped leaves
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc_store._doc_repo.set_document_is_temporal(doc_id, is_temporal=True)

        # Add leaves: t=1000-1100, t=1100-1200, t=1200-1300
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"{doc_id}-leaf-0",
                "text": "First message",
                "span_start": 0,
                "span_end": 100,
                "token_count": 5,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1100.0,
            },
            {
                "node_id": f"{doc_id}-leaf-1",
                "text": "Second message",
                "span_start": 100,
                "span_end": 200,
                "token_count": 5,
                "height": 0,
                "level_index": 1,
                "time_start": 1100.0,
                "time_end": 1200.0,
            },
            {
                "node_id": f"{doc_id}-leaf-2",
                "text": "Third message",
                "span_start": 200,
                "span_end": 300,
                "token_count": 5,
                "height": 0,
                "level_index": 2,
                "time_start": 1200.0,
                "time_end": 1300.0,
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        servicer = IndexerServicer(mock_state)  # type: ignore[arg-type]
        # Truncate at cutoff_time=1150 (1970-01-01T00:19:10Z)
        # Should delete leaf-1 and leaf-2 (time_end > 1150)
        request = pb2.TruncateFromTimeRequest(
            document_id=doc_id,
            cutoff_time="1970-01-01T00:19:10Z",  # Unix timestamp 1150
        )
        context = _MockServicerContext()

        response = await servicer.TruncateFromTime(request, context)

        assert getattr(response, "document_id") == doc_id
        assert getattr(response, "cutoff_time") == "1970-01-01T00:19:10Z"
        # Should have deleted leaf-1 and leaf-2
        deleted_ids = list(getattr(response, "deleted_node_ids"))
        assert len(deleted_ids) == 2
        assert f"{doc_id}-leaf-1" in deleted_ids
        assert f"{doc_id}-leaf-2" in deleted_ids

    @pytest.mark.asyncio
    async def test_truncate_from_time_not_found(self, mock_state: object) -> None:
        """TruncateFromTime returns NOT_FOUND for non-existent documents."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import IndexerServicer

        servicer = IndexerServicer(mock_state)  # type: ignore[arg-type]
        request = pb2.TruncateFromTimeRequest(
            document_id="nonexistent-doc",
            cutoff_time="2025-01-26T00:00:00Z",
        )
        context = _MockServicerContext()

        with pytest.raises(ValueError, match="NOT_FOUND"):
            await servicer.TruncateFromTime(request, context)

    @pytest.mark.asyncio
    async def test_truncate_from_time_non_temporal(
        self, mock_state: object, storage_backend: StorageBackend
    ) -> None:
        """TruncateFromTime returns INVALID_ARGUMENT for non-temporal documents."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import IndexerServicer

        doc_id = "test-non-temporal"

        # Create a NON-temporal document (no timestamps)
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        # is_temporal defaults to False, don't set it

        # Add a leaf without timestamps
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"{doc_id}-leaf-0",
                "text": "Non-temporal content",
                "span_start": 0,
                "span_end": 100,
                "token_count": 10,
                "height": 0,
                "level_index": 0,
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        servicer = IndexerServicer(mock_state)  # type: ignore[arg-type]
        request = pb2.TruncateFromTimeRequest(
            document_id=doc_id,
            cutoff_time="2025-01-26T00:00:00Z",
        )
        context = _MockServicerContext()

        with pytest.raises(ValueError, match="INVALID_ARGUMENT"):
            await servicer.TruncateFromTime(request, context)

    @pytest.mark.asyncio
    async def test_truncate_from_time_invalid_timestamp(
        self, mock_state: object, storage_backend: StorageBackend
    ) -> None:
        """TruncateFromTime returns INVALID_ARGUMENT for malformed timestamps."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import IndexerServicer

        doc_id = "test-invalid-timestamp"

        # Create a temporal document
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc_store._doc_repo.set_document_is_temporal(doc_id, is_temporal=True)

        # Add a leaf with timestamps
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"{doc_id}-leaf-0",
                "text": "Temporal content",
                "span_start": 0,
                "span_end": 100,
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1100.0,
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        servicer = IndexerServicer(mock_state)  # type: ignore[arg-type]
        # Use an invalid timestamp format
        request = pb2.TruncateFromTimeRequest(
            document_id=doc_id,
            cutoff_time="not-a-valid-timestamp",
        )
        context = _MockServicerContext()

        with pytest.raises(ValueError, match="INVALID_ARGUMENT"):
            await servicer.TruncateFromTime(request, context)

    @pytest.mark.asyncio
    async def test_truncate_from_time_requires_document_id(
        self, mock_state: object
    ) -> None:
        """TruncateFromTime returns error when document_id is empty."""
        from ragzoom.rpc import dynamic_summary_pb2 as pb2
        from ragzoom.server.servicers import IndexerServicer

        servicer = IndexerServicer(mock_state)  # type: ignore[arg-type]
        request = pb2.TruncateFromTimeRequest(
            document_id="",
            cutoff_time="2025-01-26T00:00:00Z",
        )
        context = _MockServicerContext()

        with pytest.raises(ValueError, match="TruncateFromTime requires"):
            await servicer.TruncateFromTime(request, context)


class TestTruncateFromTimeSpanGapBug:
    """Regression test ensuring span gaps are not created after temporal truncation.

    When truncate_from_time deletes a leaf from the middle of a document,
    subsequent leaves must have their span positions adjusted to eliminate gaps.
    This is required because the tree requires contiguous spans for summarization.

    The fix adjusts spans by shifting all remaining nodes after deleted leaves:
    - Before: A at [0,100], B at [100,200] (deleted), C at [200,300]
    - After: A at [0,100], C at [100,200] (shifted by 100)
    """

    def test_truncate_from_time_should_not_create_span_gaps(
        self, storage_backend: StorageBackend
    ) -> None:
        """After temporal truncation, remaining leaves should have contiguous spans.

        Verifies that when a middle leaf is deleted by truncate_from_time,
        remaining leaves have their span positions adjusted to eliminate gaps.
        """
        doc_id = "test-truncate-span-gap"

        # Create a temporal document
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="test.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc_store._doc_repo.set_document_is_temporal(doc_id, is_temporal=True)

        # Create 3 leaves at contiguous spans with different time ranges:
        # - leaf-0: t=1000-1050 (will be KEPT - time_end < cutoff)
        # - leaf-1: t=1050-1150 (will be DELETED - time_end > cutoff)
        # - leaf-2: t=1100-1120 (will be KEPT - time_end < cutoff)
        #
        # Cutoff at t=1125: delete nodes where time_end > 1125
        # This creates the scenario where a middle leaf is deleted.
        nodes_data: list[NodeDataDict] = [
            {
                "node_id": f"{doc_id}-leaf-0",
                "text": "A" * 100,  # 100 chars
                "span_start": 0,
                "span_end": 100,
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "time_start": 1000.0,
                "time_end": 1050.0,
            },
            {
                "node_id": f"{doc_id}-leaf-1",
                "text": "B" * 100,  # 100 chars
                "span_start": 100,
                "span_end": 200,
                "token_count": 10,
                "height": 0,
                "level_index": 1,
                "time_start": 1050.0,
                "time_end": 1150.0,  # > cutoff, will be deleted
            },
            {
                "node_id": f"{doc_id}-leaf-2",
                "text": "C" * 100,  # 100 chars
                "span_start": 200,
                "span_end": 300,
                "token_count": 10,
                "height": 0,
                "level_index": 2,
                "time_start": 1100.0,
                "time_end": 1120.0,  # < cutoff, will be kept
            },
        ]
        doc_store.nodes.add_batch(nodes_data)

        # Verify pre-condition: 3 leaves with contiguous spans
        leaves_before = sorted(doc_store.nodes.get_leaves(), key=lambda n: n.span_start)
        assert len(leaves_before) == 3
        assert leaves_before[0].span_end == leaves_before[1].span_start  # No gap
        assert leaves_before[1].span_end == leaves_before[2].span_start  # No gap

        # Truncate at cutoff_time = 1125
        # This should delete leaf-1 (time_end=1150 > 1125)
        # and keep leaf-0 and leaf-2
        deleted_ids = storage_backend.delete_nodes_from_time(doc_id, cutoff_time=1125.0)

        # Verify leaf-1 was deleted
        assert f"{doc_id}-leaf-1" in deleted_ids
        assert len(deleted_ids) == 1

        # Verify remaining leaves
        leaves_after = sorted(doc_store.nodes.get_leaves(), key=lambda n: n.span_start)
        assert len(leaves_after) == 2
        assert leaves_after[0].id == f"{doc_id}-leaf-0"
        assert leaves_after[1].id == f"{doc_id}-leaf-2"

        # Verify spans are contiguous after truncation:
        # - leaf-0: [0, 100] (unchanged)
        # - leaf-2: [100, 200] (shifted from [200, 300] to fill gap left by deleted leaf-1)
        assert leaves_after[0].span_end == leaves_after[1].span_start, (
            f"Gap detected after truncation: leaf-0 ends at {leaves_after[0].span_end}, "
            f"leaf-2 starts at {leaves_after[1].span_start}. "
            f"Gap size: {leaves_after[1].span_start - leaves_after[0].span_end} chars."
        )
