"""Tests for temporal metadata extraction in gRPC servicers.

Verifies that servicers correctly extract timestamps from proto messages
and pass them through to the underlying append operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import pytest

from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.server.servicers import IndexerServicer, RetrievalServicer, WorkerServicer

if TYPE_CHECKING:
    from unittest.mock import MagicMock


class MockServicerContext:
    """Mock gRPC servicer context for testing."""

    async def abort(self, code: object, details: str) -> NoReturn:
        raise ValueError(f"Aborted with {code}: {details}")


class TestServicerAppendExtractsTimestamp:
    """Test that AppendText servicer extracts timestamps from proto."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock IndexingSession."""
        from unittest.mock import AsyncMock, MagicMock

        session = MagicMock()
        session.append_text = AsyncMock(
            return_value=MagicMock(
                document_id="test_doc",
                chunks_created=1,
                tree_depth=0,
                span_start=0,
                span_end=10,
                mutated_nodes=1,
                resummarized_nodes=0,
                new_leaves=1,
                telemetry=None,
                telemetry_run_id=None,
            )
        )
        return session

    @pytest.fixture
    def mock_state(self, mock_session: MagicMock) -> MagicMock:
        """Create a mock ServerState with the session."""
        from unittest.mock import MagicMock

        state = MagicMock()
        state.index_runtime.get_session.return_value = mock_session
        return state

    @pytest.mark.asyncio
    async def test_append_without_timestamp(
        self, mock_state: MagicMock, mock_session: MagicMock
    ) -> None:
        """AppendText without timestamp passes None to append_text."""
        servicer = IndexerServicer(mock_state)
        request = pb2.AppendTextRequest(
            document_id="test_doc",
            content=b"hello world",
        )
        context = MockServicerContext()

        await servicer.AppendText(request, context)

        # Verify append_text was called with timestamp=None
        mock_session.append_text.assert_called_once()
        call_kwargs = mock_session.append_text.call_args.kwargs
        assert call_kwargs.get("timestamp") is None

    @pytest.mark.asyncio
    async def test_append_with_single_timestamp(
        self, mock_state: MagicMock, mock_session: MagicMock
    ) -> None:
        """AppendText with timestamp (time_start only) passes it through."""
        servicer = IndexerServicer(mock_state)
        timestamp = pb2.Timestamp(time_start="2024-01-21T14:30:00Z")
        request = pb2.AppendTextRequest(
            document_id="test_doc",
            content=b"hello world",
            timestamp=timestamp,
        )
        context = MockServicerContext()

        await servicer.AppendText(request, context)

        mock_session.append_text.assert_called_once()
        call_kwargs = mock_session.append_text.call_args.kwargs
        # When time_end is not set, it should be same as time_start
        assert call_kwargs.get("timestamp") == "2024-01-21T14:30:00Z"

    @pytest.mark.asyncio
    async def test_append_with_time_range_timestamp(
        self, mock_state: MagicMock, mock_session: MagicMock
    ) -> None:
        """AppendText with timestamp (time_start and time_end) passes tuple."""
        servicer = IndexerServicer(mock_state)
        timestamp = pb2.Timestamp(
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:05Z",
        )
        request = pb2.AppendTextRequest(
            document_id="test_doc",
            content=b"hello world",
            timestamp=timestamp,
        )
        context = MockServicerContext()

        await servicer.AppendText(request, context)

        mock_session.append_text.assert_called_once()
        call_kwargs = mock_session.append_text.call_args.kwargs
        assert call_kwargs.get("timestamp") == (
            "2024-01-21T14:30:00Z",
            "2024-01-21T14:30:05Z",
        )


class TestServicerBatchAppendExtractsTimestamps:
    """Test that BatchAppendText servicer extracts timestamps from proto."""

    @pytest.fixture
    def mock_session(self) -> MagicMock:
        """Create a mock IndexingSession."""
        from unittest.mock import AsyncMock, MagicMock

        session = MagicMock()
        session.batch_append_text = AsyncMock(
            return_value=MagicMock(
                document_id="test_doc",
                chunks_created=2,
                tree_depth=0,
                span_start=0,
                span_end=20,
                mutated_nodes=2,
                resummarized_nodes=0,
                new_leaves=2,
                telemetry=None,
                telemetry_run_id=None,
            )
        )
        return session

    @pytest.fixture
    def mock_state(self, mock_session: MagicMock) -> MagicMock:
        """Create a mock ServerState with the session."""
        from unittest.mock import MagicMock

        state = MagicMock()
        state.index_runtime.get_session.return_value = mock_session
        return state

    @pytest.mark.asyncio
    async def test_batch_append_without_timestamps(
        self, mock_state: MagicMock, mock_session: MagicMock
    ) -> None:
        """BatchAppendText without timestamps passes None to batch_append_text."""
        servicer = IndexerServicer(mock_state)
        unit1 = pb2.AppendUnit(content=b"chunk 1")
        unit2 = pb2.AppendUnit(content=b"chunk 2")
        request = pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[unit1, unit2],
        )
        context = MockServicerContext()

        await servicer.BatchAppendText(request, context)

        mock_session.batch_append_text.assert_called_once()
        call_kwargs = mock_session.batch_append_text.call_args.kwargs
        assert call_kwargs.get("timestamps") is None

    @pytest.mark.asyncio
    async def test_batch_append_with_timestamps(
        self, mock_state: MagicMock, mock_session: MagicMock
    ) -> None:
        """BatchAppendText with timestamps in AppendUnits passes them through."""
        servicer = IndexerServicer(mock_state)
        # Unit with only time_start (uses same value for both)
        unit1 = pb2.AppendUnit(
            content=b"chunk 1",
            time_start="2024-01-21T14:30:00Z",
        )
        # Unit with both time_start and time_end
        unit2 = pb2.AppendUnit(
            content=b"chunk 2",
            time_start="2024-01-21T14:30:05Z",
            time_end="2024-01-21T14:30:12Z",
        )
        request = pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[unit1, unit2],
        )
        context = MockServicerContext()

        await servicer.BatchAppendText(request, context)

        mock_session.batch_append_text.assert_called_once()
        call_kwargs = mock_session.batch_append_text.call_args.kwargs
        timestamps = call_kwargs.get("timestamps")
        assert timestamps is not None
        assert len(timestamps) == 2
        # unit1 has only time_start, so it's a single string
        assert timestamps[0] == "2024-01-21T14:30:00Z"
        # unit2 has both, so it's a tuple
        assert timestamps[1] == ("2024-01-21T14:30:05Z", "2024-01-21T14:30:12Z")

    @pytest.mark.asyncio
    async def test_batch_append_mixed_timestamps(
        self, mock_state: MagicMock, mock_session: MagicMock
    ) -> None:
        """BatchAppendText with some timestamped, some non-timestamped units."""
        servicer = IndexerServicer(mock_state)
        unit1 = pb2.AppendUnit(content=b"chunk 1")  # No timestamp
        unit2 = pb2.AppendUnit(
            content=b"chunk 2",
            time_start="2024-01-21T14:30:05Z",
        )
        unit3 = pb2.AppendUnit(content=b"chunk 3")  # No timestamp
        request = pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[unit1, unit2, unit3],
        )
        context = MockServicerContext()

        await servicer.BatchAppendText(request, context)

        call_kwargs = mock_session.batch_append_text.call_args.kwargs
        timestamps = call_kwargs.get("timestamps")
        assert timestamps is not None
        assert len(timestamps) == 3
        assert timestamps[0] is None
        assert timestamps[1] == "2024-01-21T14:30:05Z"
        assert timestamps[2] is None

    @pytest.mark.asyncio
    async def test_batch_append_time_end_without_time_start_rejected(
        self, mock_state: MagicMock, mock_session: MagicMock
    ) -> None:
        """BatchAppendText with time_end but no time_start raises error."""
        servicer = IndexerServicer(mock_state)
        # Unit with only time_end is invalid
        unit1 = pb2.AppendUnit(content=b"chunk 1")
        unit2 = pb2.AppendUnit(
            content=b"chunk 2",
            time_end="2024-01-21T14:30:12Z",  # No time_start
        )
        request = pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[unit1, unit2],
        )
        context = MockServicerContext()

        with pytest.raises(ValueError, match="time_end provided without time_start"):
            await servicer.BatchAppendText(request, context)

        # batch_append_text should not be called
        mock_session.batch_append_text.assert_not_called()


class TestServicerQueryExtractsTimeWindow:
    """Test that ExecuteQuery servicer extracts time window from proto."""

    @pytest.fixture
    def mock_retriever(self) -> MagicMock:
        """Create a mock Retriever."""
        from unittest.mock import AsyncMock, MagicMock

        retriever = MagicMock()
        retriever.retrieve_async = AsyncMock(
            return_value=MagicMock(
                node_ids=["node1"],
                tiling=["node1"],
                scores={"node1": 0.5},
                coverage_map={"node1": True},
                nodes={
                    "node1": MagicMock(
                        id="node1",
                        text="test content",
                        token_count=10,
                        span_start=0,
                        span_end=100,
                        parent_id=None,
                        left_child_id=None,
                        right_child_id=None,
                        height=0,
                    )
                },
                seed_count=1,
                verbatim_count=0,
                actual_start=0,
                actual_end=100,
            )
        )
        return retriever

    @pytest.fixture
    def mock_state(self) -> MagicMock:
        """Create a mock ServerState."""
        from unittest.mock import MagicMock

        state = MagicMock()
        state.query_config.budget_tokens = 1000
        state.query_config.embedding_model = "text-embedding-3-small"
        state.query_log.record_query = MagicMock(return_value="query123")
        return state

    @pytest.fixture
    def servicer(
        self,
        mock_state: MagicMock,
        mock_retriever: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> RetrievalServicer:
        """Create a RetrievalServicer with mocked _build_retriever."""
        from unittest.mock import MagicMock as Mock

        from ragzoom.server import servicers

        def patched_build_retriever(
            state: object, *, document_id: str, embedding_model: str | None = None
        ) -> tuple[MagicMock, Mock]:
            return mock_retriever, Mock()

        monkeypatch.setattr(servicers, "_build_retriever", patched_build_retriever)
        return servicers.RetrievalServicer(mock_state)

    @pytest.mark.asyncio
    async def test_query_without_time_window(
        self,
        servicer: RetrievalServicer,
        mock_retriever: MagicMock,
    ) -> None:
        """ExecuteQuery without time window passes None for time params."""
        request = pb2.ExecuteQueryRequest(
            document_id="test_doc",
            query="test query",
            budget_tokens=1000,
        )

        await servicer.ExecuteQuery(request, MockServicerContext())

        mock_retriever.retrieve_async.assert_called_once()
        call_kwargs = mock_retriever.retrieve_async.call_args.kwargs
        assert call_kwargs.get("time_start") is None
        assert call_kwargs.get("time_end") is None

    @pytest.mark.asyncio
    async def test_query_with_time_window(
        self,
        servicer: RetrievalServicer,
        mock_retriever: MagicMock,
    ) -> None:
        """ExecuteQuery with time window passes time_start and time_end to retriever."""
        request = pb2.ExecuteQueryRequest(
            document_id="test_doc",
            query="test query",
            budget_tokens=1000,
            time_start="2024-01-21T14:00:00Z",
            time_end="2024-01-21T15:00:00Z",
        )

        await servicer.ExecuteQuery(request, MockServicerContext())

        mock_retriever.retrieve_async.assert_called_once()
        call_kwargs = mock_retriever.retrieve_async.call_args.kwargs
        assert call_kwargs.get("time_start") == "2024-01-21T14:00:00Z"
        assert call_kwargs.get("time_end") == "2024-01-21T15:00:00Z"

    @pytest.mark.asyncio
    async def test_query_with_time_start_only(
        self,
        servicer: RetrievalServicer,
        mock_retriever: MagicMock,
    ) -> None:
        """ExecuteQuery with only time_start passes it to retriever."""
        request = pb2.ExecuteQueryRequest(
            document_id="test_doc",
            query="test query",
            budget_tokens=1000,
            time_start="2024-01-21T14:00:00Z",
        )

        await servicer.ExecuteQuery(request, MockServicerContext())

        mock_retriever.retrieve_async.assert_called_once()
        call_kwargs = mock_retriever.retrieve_async.call_args.kwargs
        assert call_kwargs.get("time_start") == "2024-01-21T14:00:00Z"
        assert call_kwargs.get("time_end") is None


class TestDocumentStatusIncludesIsTemporal:
    """Test that GetDocument response includes is_temporal field."""

    @pytest.fixture
    def mock_doc_repo(self) -> MagicMock:
        """Create a mock document repository."""
        from unittest.mock import MagicMock

        doc_repo = MagicMock()
        doc_repo.get_document_is_temporal = MagicMock(return_value=True)
        return doc_repo

    @pytest.fixture
    def mock_document_store(self, mock_doc_repo: MagicMock) -> MagicMock:
        """Create a mock DocumentStore."""
        from unittest.mock import MagicMock

        store = MagicMock()
        store._doc_repo = mock_doc_repo
        store.nodes.leaf_count.return_value = 5
        store.tree.get_root.return_value = MagicMock(height=2)
        return store

    @pytest.fixture
    def mock_state_for_get_document(self, mock_document_store: MagicMock) -> MagicMock:
        """Create a mock ServerState for GetDocument."""
        from unittest.mock import AsyncMock, MagicMock

        state = MagicMock()
        state.store.for_document.return_value = mock_document_store
        state.indexing_engine.status = AsyncMock(
            return_value=MagicMock(in_flight_by_document={})
        )
        return state

    @pytest.mark.asyncio
    async def test_document_status_includes_is_temporal_true(
        self,
        mock_state_for_get_document: MagicMock,
        mock_doc_repo: MagicMock,
    ) -> None:
        """GetDocument returns is_temporal=True for temporal documents."""
        mock_doc_repo.get_document_is_temporal.return_value = True
        servicer = WorkerServicer(mock_state_for_get_document)
        request = pb2.GetDocumentRequest(document_id="test_doc")
        context = MockServicerContext()

        response = await servicer.GetDocument(request, context)

        assert response.status.is_temporal is True
        mock_doc_repo.get_document_is_temporal.assert_called_once_with("test_doc")

    @pytest.mark.asyncio
    async def test_document_status_includes_is_temporal_false(
        self,
        mock_state_for_get_document: MagicMock,
        mock_doc_repo: MagicMock,
    ) -> None:
        """GetDocument returns is_temporal=False for non-temporal documents."""
        mock_doc_repo.get_document_is_temporal.return_value = False
        servicer = WorkerServicer(mock_state_for_get_document)
        request = pb2.GetDocumentRequest(document_id="test_doc")
        context = MockServicerContext()

        response = await servicer.GetDocument(request, context)

        assert response.status.is_temporal is False

    @pytest.mark.asyncio
    async def test_document_status_defaults_is_temporal_false_when_not_found(
        self,
        mock_state_for_get_document: MagicMock,
        mock_doc_repo: MagicMock,
    ) -> None:
        """GetDocument returns is_temporal=False when document not found."""
        mock_doc_repo.get_document_is_temporal.return_value = None
        servicer = WorkerServicer(mock_state_for_get_document)
        request = pb2.GetDocumentRequest(document_id="nonexistent_doc")
        context = MockServicerContext()

        response = await servicer.GetDocument(request, context)

        # None from repository should be treated as False
        assert response.status.is_temporal is False
