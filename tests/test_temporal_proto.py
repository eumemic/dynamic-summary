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


class TestBatchAppendTextRequestUsesAppendUnit:
    """Test that BatchAppendTextRequest uses repeated AppendUnit field."""

    def test_batch_append_request_uses_append_unit(self) -> None:
        """BatchAppendTextRequest should use AppendUnit for units field."""
        unit1 = dynamic_summary_pb2.AppendUnit(content=b"chunk 1")
        unit2 = dynamic_summary_pb2.AppendUnit(content=b"chunk 2")
        req = dynamic_summary_pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[unit1, unit2],
        )
        assert len(req.units) == 2
        assert req.units[0].content == b"chunk 1"
        assert req.units[1].content == b"chunk 2"

    def test_batch_append_request_with_timestamps_in_units(self) -> None:
        """BatchAppendTextRequest AppendUnits can have timestamps bundled."""
        unit1 = dynamic_summary_pb2.AppendUnit(
            content=b"chunk 1",
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:00Z",
        )
        unit2 = dynamic_summary_pb2.AppendUnit(
            content=b"chunk 2",
            time_start="2024-01-21T14:30:05Z",
            time_end="2024-01-21T14:30:12Z",
        )
        req = dynamic_summary_pb2.BatchAppendTextRequest(
            document_id="test_doc",
            units=[unit1, unit2],
        )
        assert len(req.units) == 2
        assert req.units[0].time_start == "2024-01-21T14:30:00Z"
        assert req.units[1].time_start == "2024-01-21T14:30:05Z"
        assert req.units[1].time_end == "2024-01-21T14:30:12Z"

    def test_batch_append_request_units_is_field_number_2(self) -> None:
        """units field should be field number 2."""
        field = (
            dynamic_summary_pb2.BatchAppendTextRequest.DESCRIPTOR.fields_by_name.get(
                "units"
            )
        )
        assert field is not None
        assert field.number == 2


class TestExecuteQueryRequestTimeFields:
    """Test that ExecuteQueryRequest has optional time_start and time_end fields."""

    def test_query_request_has_time_start_field(self) -> None:
        """ExecuteQueryRequest should have an optional time_start field."""
        req = dynamic_summary_pb2.ExecuteQueryRequest(
            document_id="test_doc",
            query="test query",
        )
        # time_start field should exist and be unset by default
        assert hasattr(req, "time_start")
        assert req.HasField("time_start") is False

    def test_query_request_has_time_end_field(self) -> None:
        """ExecuteQueryRequest should have an optional time_end field."""
        req = dynamic_summary_pb2.ExecuteQueryRequest(
            document_id="test_doc",
            query="test query",
        )
        # time_end field should exist and be unset by default
        assert hasattr(req, "time_end")
        assert req.HasField("time_end") is False

    def test_query_request_accepts_time_fields(self) -> None:
        """ExecuteQueryRequest can accept time_start and time_end values."""
        req = dynamic_summary_pb2.ExecuteQueryRequest(
            document_id="test_doc",
            query="test query",
            time_start="2024-01-21T14:00:00Z",
            time_end="2024-01-21T15:00:00Z",
        )
        assert req.HasField("time_start") is True
        assert req.HasField("time_end") is True
        assert req.time_start == "2024-01-21T14:00:00Z"
        assert req.time_end == "2024-01-21T15:00:00Z"

    def test_query_request_time_fields_are_strings(self) -> None:
        """Time fields should be string type for ISO 8601 format."""
        req = dynamic_summary_pb2.ExecuteQueryRequest(
            document_id="test_doc",
            query="test query",
            time_start="2024-01-21T14:30:00.123456+00:00",
            time_end="2024-01-21T14:30:00-05:00",
        )
        assert isinstance(req.time_start, str)
        assert isinstance(req.time_end, str)
        assert req.time_start == "2024-01-21T14:30:00.123456+00:00"
        assert req.time_end == "2024-01-21T14:30:00-05:00"


class TestAppendUnitProtoMessage:
    """Test the AppendUnit message type for batch_append with bundled timestamps."""

    def test_append_unit_type_exists(self) -> None:
        """AppendUnit message type should be importable from generated proto."""
        assert hasattr(dynamic_summary_pb2, "AppendUnit")

    def test_append_unit_has_content_field(self) -> None:
        """AppendUnit message should have a content field for text."""
        unit = dynamic_summary_pb2.AppendUnit()
        unit.content = b"test content"
        assert unit.content == b"test content"

    def test_append_unit_has_optional_time_start_field(self) -> None:
        """AppendUnit message should have an optional time_start field."""
        unit = dynamic_summary_pb2.AppendUnit()

        # time_start defaults to empty string in proto3
        assert unit.time_start == ""
        assert unit.HasField("time_start") is False

        unit.time_start = "2024-01-21T14:30:00Z"
        assert unit.time_start == "2024-01-21T14:30:00Z"
        assert unit.HasField("time_start") is True

    def test_append_unit_has_optional_time_end_field(self) -> None:
        """AppendUnit message should have an optional time_end field."""
        unit = dynamic_summary_pb2.AppendUnit()

        # time_end defaults to empty string in proto3
        assert unit.time_end == ""
        assert unit.HasField("time_end") is False

        unit.time_end = "2024-01-21T14:30:12Z"
        assert unit.time_end == "2024-01-21T14:30:12Z"
        assert unit.HasField("time_end") is True

    def test_append_unit_can_be_created_with_all_fields(self) -> None:
        """AppendUnit message can be created with content and both timestamps."""
        unit = dynamic_summary_pb2.AppendUnit(
            content=b"test content",
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:12Z",
        )
        assert unit.content == b"test content"
        assert unit.time_start == "2024-01-21T14:30:00Z"
        assert unit.time_end == "2024-01-21T14:30:12Z"

    def test_append_unit_field_numbers(self) -> None:
        """AppendUnit fields should have correct field numbers per spec."""
        descriptor = dynamic_summary_pb2.AppendUnit.DESCRIPTOR
        content_field = descriptor.fields_by_name.get("content")
        time_start_field = descriptor.fields_by_name.get("time_start")
        time_end_field = descriptor.fields_by_name.get("time_end")

        assert content_field is not None
        assert content_field.number == 1
        assert time_start_field is not None
        assert time_start_field.number == 2
        assert time_end_field is not None
        assert time_end_field.number == 3


class TestDocumentStatusIsTemporalField:
    """Test that DocumentStatus has is_temporal field for temporal documents."""

    def test_document_status_has_is_temporal_field(self) -> None:
        """DocumentStatus should have an is_temporal boolean field."""
        status = dynamic_summary_pb2.DocumentStatus(
            document_id="test_doc",
            leaf_count=10,
            has_pending_work=False,
            tree_depth=3,
        )
        assert hasattr(status, "is_temporal")
        # Default value for bool in proto3 is False
        assert status.is_temporal is False

    def test_document_status_is_temporal_can_be_set_true(self) -> None:
        """DocumentStatus.is_temporal can be set to True."""
        status = dynamic_summary_pb2.DocumentStatus(
            document_id="test_doc",
            leaf_count=10,
            is_temporal=True,
        )
        assert status.is_temporal is True

    def test_document_status_is_temporal_is_field_number_5(self) -> None:
        """is_temporal field should be field number 5 per the spec."""
        field = dynamic_summary_pb2.DocumentStatus.DESCRIPTOR.fields_by_name.get(
            "is_temporal"
        )
        assert field is not None
        assert field.number == 5
