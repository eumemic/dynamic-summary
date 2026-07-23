"""Tests for JSON output mode proto definitions.

Verifies that the Node message has temporal fields needed for JSON output.
"""

from __future__ import annotations

from ragzoom.rpc import dynamic_summary_pb2


class TestNodeTemporalFields:
    """Test that Node message has time_start and time_end fields."""

    def test_node_has_temporal_fields(self) -> None:
        """Node message should have optional time_start and time_end fields."""
        node = dynamic_summary_pb2.Node(
            node_id="test_node",
            text="Test node text",
            token_count=10,
            span_start=0,
            span_end=100,
            height=1,
        )

        # time_start and time_end should exist
        assert hasattr(node, "time_start")
        assert hasattr(node, "time_end")

        # Should be unset by default
        assert node.HasField("time_start") is False
        assert node.HasField("time_end") is False

    def test_node_time_start_can_be_set(self) -> None:
        """Node time_start field can be set to ISO 8601 string."""
        node = dynamic_summary_pb2.Node(
            node_id="test_node",
            text="Test node text",
            time_start="2024-01-21T10:00:00Z",
        )
        assert node.HasField("time_start") is True
        assert node.time_start == "2024-01-21T10:00:00Z"

    def test_node_time_end_can_be_set(self) -> None:
        """Node time_end field can be set to ISO 8601 string."""
        node = dynamic_summary_pb2.Node(
            node_id="test_node",
            text="Test node text",
            time_end="2024-01-21T10:30:00Z",
        )
        assert node.HasField("time_end") is True
        assert node.time_end == "2024-01-21T10:30:00Z"

    def test_node_both_temporal_fields_can_be_set(self) -> None:
        """Node can have both time_start and time_end set."""
        node = dynamic_summary_pb2.Node(
            node_id="test_node",
            text="Test node text",
            token_count=10,
            span_start=0,
            span_end=100,
            time_start="2024-01-21T10:00:00Z",
            time_end="2024-01-21T10:30:00Z",
        )
        assert node.time_start == "2024-01-21T10:00:00Z"
        assert node.time_end == "2024-01-21T10:30:00Z"

    def test_node_temporal_fields_are_optional_strings(self) -> None:
        """Temporal fields should be optional string type per JSON output spec."""
        descriptor = dynamic_summary_pb2.Node.DESCRIPTOR
        time_start_field = descriptor.fields_by_name.get("time_start")
        time_end_field = descriptor.fields_by_name.get("time_end")

        assert time_start_field is not None
        assert time_end_field is not None

        # Field type 9 is TYPE_STRING in protobuf
        from google.protobuf.descriptor import FieldDescriptor

        assert time_start_field.type == FieldDescriptor.TYPE_STRING
        assert time_end_field.type == FieldDescriptor.TYPE_STRING

        # Both should be optional (proto3 with explicit optional keyword).
        # FieldDescriptor.label was removed from the protobuf runtime; the
        # modern equivalent of LABEL_OPTIONAL is "singular with presence".
        assert not time_start_field.is_repeated and time_start_field.has_presence
        assert not time_end_field.is_repeated and time_end_field.has_presence

    def test_node_temporal_field_numbers(self) -> None:
        """Temporal fields should have correct field numbers (10 and 11)."""
        descriptor = dynamic_summary_pb2.Node.DESCRIPTOR
        time_start_field = descriptor.fields_by_name.get("time_start")
        time_end_field = descriptor.fields_by_name.get("time_end")

        assert time_start_field is not None
        assert time_start_field.number == 10

        assert time_end_field is not None
        assert time_end_field.number == 11
