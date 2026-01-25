"""Tests for Turn dataclass in timestamped transcript sync."""

from __future__ import annotations

from ragzoom_claude_code.transcript_sync import Turn


class TestTurnDataclass:
    """Tests for the Turn dataclass."""

    def test_turn_has_uuids_field(self) -> None:
        """Turn should have a uuids field holding message UUIDs."""
        turn = Turn(
            uuids=["msg1", "msg2", "msg3"],
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:15Z",
        )
        assert turn.uuids == ["msg1", "msg2", "msg3"]

    def test_turn_has_time_start_field(self) -> None:
        """Turn should have a time_start field with ISO 8601 timestamp."""
        turn = Turn(
            uuids=["msg1"],
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:05Z",
        )
        assert turn.time_start == "2024-01-21T14:30:00Z"

    def test_turn_has_time_end_field(self) -> None:
        """Turn should have a time_end field with ISO 8601 timestamp."""
        turn = Turn(
            uuids=["msg1"],
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:05Z",
        )
        assert turn.time_end == "2024-01-21T14:30:05Z"

    def test_turn_single_message_turn(self) -> None:
        """Turn with single message should have same start and end times."""
        turn = Turn(
            uuids=["msg1"],
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:00Z",
        )
        assert len(turn.uuids) == 1
        assert turn.time_start == turn.time_end

    def test_turn_multi_message_turn(self) -> None:
        """Turn with multiple messages spans their time range."""
        turn = Turn(
            uuids=["user1", "assistant1", "tool1", "assistant2"],
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:35:00Z",
        )
        assert len(turn.uuids) == 4
        assert turn.time_start == "2024-01-21T14:30:00Z"
        assert turn.time_end == "2024-01-21T14:35:00Z"

    def test_turn_timestamps_with_timezone_offset(self) -> None:
        """Turn should accept timestamps with various timezone formats."""
        turn = Turn(
            uuids=["msg1"],
            time_start="2024-01-21T14:30:00+05:30",
            time_end="2024-01-21T14:35:00-08:00",
        )
        assert turn.time_start == "2024-01-21T14:30:00+05:30"
        assert turn.time_end == "2024-01-21T14:35:00-08:00"

    def test_turn_timestamps_with_microseconds(self) -> None:
        """Turn should accept timestamps with microsecond precision."""
        turn = Turn(
            uuids=["msg1"],
            time_start="2024-01-21T14:30:00.123456Z",
            time_end="2024-01-21T14:30:00.999999Z",
        )
        assert turn.time_start == "2024-01-21T14:30:00.123456Z"
        assert turn.time_end == "2024-01-21T14:30:00.999999Z"

    def test_turn_is_dataclass(self) -> None:
        """Turn should be a dataclass with expected fields."""
        from dataclasses import fields

        field_names = {f.name for f in fields(Turn)}
        assert field_names == {"uuids", "time_start", "time_end"}
