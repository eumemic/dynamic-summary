"""Tests for Step dataclass in step-level transcript sync."""

from __future__ import annotations

from ragzoom_claude_code.transcript_sync import Step


class TestStepDataclass:
    """Tests for the Step dataclass."""

    def test_step_has_uuid_field(self) -> None:
        """Step should have a uuid field holding the message UUID."""
        step = Step(
            uuid="msg-abc-123",
            timestamp="2024-01-21T14:30:00Z",
        )
        assert step.uuid == "msg-abc-123"

    def test_step_has_timestamp_field(self) -> None:
        """Step should have a timestamp field with ISO 8601 timestamp."""
        step = Step(
            uuid="msg-abc-123",
            timestamp="2024-01-21T14:30:00Z",
        )
        assert step.timestamp == "2024-01-21T14:30:00Z"

    def test_step_timestamps_with_timezone_offset(self) -> None:
        """Step should accept timestamps with various timezone formats."""
        step = Step(
            uuid="msg1",
            timestamp="2024-01-21T14:30:00+05:30",
        )
        assert step.timestamp == "2024-01-21T14:30:00+05:30"

    def test_step_timestamps_with_microseconds(self) -> None:
        """Step should accept timestamps with microsecond precision."""
        step = Step(
            uuid="msg1",
            timestamp="2024-01-21T14:30:00.123456Z",
        )
        assert step.timestamp == "2024-01-21T14:30:00.123456Z"

    def test_step_is_dataclass(self) -> None:
        """Step should be a dataclass with expected fields."""
        from dataclasses import fields

        field_names = {f.name for f in fields(Step)}
        assert field_names == {"uuid", "timestamp"}
