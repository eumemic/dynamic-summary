"""Tests for temporal metadata extraction in gRPC servicers.

Verifies that servicers correctly extract timestamps from proto messages
and pass them through to the underlying append operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import pytest

from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.server.servicers import IndexerServicer

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
        request = pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[b"chunk 1", b"chunk 2"],
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
        """BatchAppendText with timestamps passes them through."""
        servicer = IndexerServicer(mock_state)
        ts1 = pb2.Timestamp(time_start="2024-01-21T14:30:00Z")
        ts2 = pb2.Timestamp(
            time_start="2024-01-21T14:30:05Z",
            time_end="2024-01-21T14:30:12Z",
        )
        request = pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[b"chunk 1", b"chunk 2"],
            timestamps=[ts1, ts2],
        )
        context = MockServicerContext()

        await servicer.BatchAppendText(request, context)

        mock_session.batch_append_text.assert_called_once()
        call_kwargs = mock_session.batch_append_text.call_args.kwargs
        timestamps = call_kwargs.get("timestamps")
        assert timestamps is not None
        assert len(timestamps) == 2
        # ts1 has only time_start, so it's a single string
        assert timestamps[0] == "2024-01-21T14:30:00Z"
        # ts2 has both, so it's a tuple
        assert timestamps[1] == ("2024-01-21T14:30:05Z", "2024-01-21T14:30:12Z")

    @pytest.mark.asyncio
    async def test_batch_append_timestamps_length_mismatch_is_rejected(
        self, mock_state: MagicMock, mock_session: MagicMock
    ) -> None:
        """BatchAppendText with mismatched timestamps length raises error."""
        servicer = IndexerServicer(mock_state)
        ts1 = pb2.Timestamp(time_start="2024-01-21T14:30:00Z")
        request = pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[b"chunk 1", b"chunk 2"],
            timestamps=[ts1],  # Only 1 timestamp for 2 units
        )
        context = MockServicerContext()

        with pytest.raises(ValueError, match="timestamps length"):
            await servicer.BatchAppendText(request, context)

        # batch_append_text should not be called
        mock_session.batch_append_text.assert_not_called()
