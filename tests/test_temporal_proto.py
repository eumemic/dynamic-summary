"""Tests for temporal metadata gRPC protocol definitions.

Verifies that the proto definitions for temporal metadata are correctly generated
and available for use in the gRPC client/server.
"""

from __future__ import annotations

from ragzoom.rpc import dynamic_summary_pb2


class TestTimestampMessage:
    """Test the Timestamp message type in the proto definitions."""

    def test_timestamp_type_exists(self) -> None:
        """Timestamp message type should be importable from generated proto."""
        assert hasattr(dynamic_summary_pb2, "Timestamp")

    def test_timestamp_has_time_start_field(self) -> None:
        """Timestamp message should have a time_start field."""
        ts = dynamic_summary_pb2.Timestamp()
        ts.time_start = "2024-01-21T14:30:00Z"
        assert ts.time_start == "2024-01-21T14:30:00Z"

    def test_timestamp_has_optional_time_end_field(self) -> None:
        """Timestamp message should have an optional time_end field."""
        ts = dynamic_summary_pb2.Timestamp()

        # time_end defaults to empty string in proto3
        assert ts.time_end == ""

        ts.time_end = "2024-01-21T14:30:12Z"
        assert ts.time_end == "2024-01-21T14:30:12Z"

    def test_timestamp_can_be_created_with_both_fields(self) -> None:
        """Timestamp message can be created with both time_start and time_end."""
        ts = dynamic_summary_pb2.Timestamp(
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:12Z",
        )
        assert ts.time_start == "2024-01-21T14:30:00Z"
        assert ts.time_end == "2024-01-21T14:30:12Z"

    def test_timestamp_time_end_optional_semantics(self) -> None:
        """When time_end is not set, HasField returns False.

        The application layer should interpret unset time_end as equal to time_start.
        """
        ts = dynamic_summary_pb2.Timestamp(time_start="2024-01-21T14:30:00Z")

        assert ts.time_start == "2024-01-21T14:30:00Z"
        assert ts.time_end == ""
        assert ts.HasField("time_end") is False

        ts.time_end = "2024-01-21T14:30:12Z"
        assert ts.HasField("time_end") is True


class TestAppendTextRequestTimestamp:
    """Test that AppendTextRequest has optional Timestamp field."""

    def test_append_request_has_timestamp_field(self) -> None:
        """AppendTextRequest should have an optional timestamp field."""
        req = dynamic_summary_pb2.AppendTextRequest(
            document_id="test_doc",
            content=b"test content",
        )
        # Timestamp field should exist and be unset by default
        assert hasattr(req, "timestamp")
        assert req.HasField("timestamp") is False

    def test_append_request_accepts_timestamp(self) -> None:
        """AppendTextRequest can accept a Timestamp value."""
        ts = dynamic_summary_pb2.Timestamp(
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:12Z",
        )
        req = dynamic_summary_pb2.AppendTextRequest(
            document_id="test_doc",
            content=b"test content",
            timestamp=ts,
        )
        assert req.HasField("timestamp") is True
        assert req.timestamp.time_start == "2024-01-21T14:30:00Z"
        assert req.timestamp.time_end == "2024-01-21T14:30:12Z"

    def test_append_request_timestamp_is_field_number_5(self) -> None:
        """Timestamp field should be field number 5 per the spec."""
        # DESCRIPTOR gives us access to the proto field descriptors
        field = dynamic_summary_pb2.AppendTextRequest.DESCRIPTOR.fields_by_name.get(
            "timestamp"
        )
        assert field is not None
        assert field.number == 5
