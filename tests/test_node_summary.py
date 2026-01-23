"""Tests for NodeSummary dataclass."""

from ragzoom.client.grpc_client import NodeSummary


class TestNodeSummaryTemporalFields:
    """Tests for temporal fields in NodeSummary."""

    def test_node_summary_has_temporal_fields(self) -> None:
        """NodeSummary should include time_start and time_end optional fields."""
        # Create NodeSummary with temporal fields
        node = NodeSummary(
            node_id="test-node",
            text="Test summary text",
            token_count=50,
            span_start=0,
            span_end=100,
            parent_id="parent-1",
            left_child_id="left-1",
            right_child_id="right-1",
            height=2,
            time_start="2024-01-21T10:00:00Z",
            time_end="2024-01-21T10:30:00Z",
        )

        assert node.time_start == "2024-01-21T10:00:00Z"
        assert node.time_end == "2024-01-21T10:30:00Z"

    def test_node_summary_temporal_fields_default_to_none(self) -> None:
        """Temporal fields should default to None for non-temporal documents."""
        node = NodeSummary(
            node_id="test-node",
            text="Test summary text",
            token_count=50,
            span_start=0,
            span_end=100,
            parent_id="",
            left_child_id="",
            right_child_id="",
            height=0,
        )

        assert node.time_start is None
        assert node.time_end is None

    def test_node_summary_partial_temporal_fields(self) -> None:
        """Allow setting only time_start without time_end."""
        node = NodeSummary(
            node_id="test-node",
            text="Test text",
            token_count=25,
            span_start=0,
            span_end=50,
            parent_id="",
            left_child_id="",
            right_child_id="",
            height=0,
            time_start="2024-01-21T10:00:00Z",
        )

        assert node.time_start == "2024-01-21T10:00:00Z"
        assert node.time_end is None
