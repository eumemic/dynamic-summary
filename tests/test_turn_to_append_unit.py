"""Tests for Turn to AppendUnit conversion in transcript sync."""

from __future__ import annotations

from ragzoom.claude_memory.transcript_sync import (
    Turn,
    turns_to_append_units,
)
from ragzoom.wrapper import AppendUnit


def _records(*items: dict[str, object]) -> dict[str, dict[str, object]]:
    """Helper to build records_by_uuid from a list of records."""
    result: dict[str, dict[str, object]] = {}
    for item in items:
        uuid = item.get("uuid")
        if isinstance(uuid, str):
            result[uuid] = item
    return result


class TestTurnsToAppendUnits:
    """Tests for turns_to_append_units function."""

    def test_single_turn_converts_to_single_append_unit(self) -> None:
        """A single turn becomes a single AppendUnit."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Hi there!"}]},
            },
        )
        turn = Turn(
            uuids=["msg1", "msg2"],
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:05Z",
        )

        result = turns_to_append_units([turn], records)

        assert len(result) == 1
        assert isinstance(result[0], AppendUnit)
        assert result[0].time_start == "2024-01-21T14:30:00Z"
        assert result[0].time_end == "2024-01-21T14:30:05Z"
        # Text should contain both user and assistant content
        assert "[USER]" in result[0].text
        assert "Hello" in result[0].text
        assert "[ASSISTANT]" in result[0].text
        assert "Hi there!" in result[0].text

    def test_multiple_turns_convert_to_multiple_append_units(self) -> None:
        """Multiple turns become multiple AppendUnits."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "First"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "One"}]},
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:31:00Z",
                "message": {"content": "Second"},
            },
            {
                "uuid": "msg4",
                "type": "assistant",
                "timestamp": "2024-01-21T14:31:05Z",
                "message": {"content": [{"type": "text", "text": "Two"}]},
            },
        )
        turns = [
            Turn(
                uuids=["msg1", "msg2"],
                time_start="2024-01-21T14:30:00Z",
                time_end="2024-01-21T14:30:05Z",
            ),
            Turn(
                uuids=["msg3", "msg4"],
                time_start="2024-01-21T14:31:00Z",
                time_end="2024-01-21T14:31:05Z",
            ),
        ]

        result = turns_to_append_units(turns, records)

        assert len(result) == 2
        assert result[0].time_start == "2024-01-21T14:30:00Z"
        assert result[0].time_end == "2024-01-21T14:30:05Z"
        assert "First" in result[0].text
        assert result[1].time_start == "2024-01-21T14:31:00Z"
        assert result[1].time_end == "2024-01-21T14:31:05Z"
        assert "Second" in result[1].text

    def test_empty_turns_returns_empty_list(self) -> None:
        """Empty turn list returns empty AppendUnit list."""
        result = turns_to_append_units([], {})
        assert result == []

    def test_turn_with_tool_calls_transcribes_correctly(self) -> None:
        """Turn with tool calls has proper transcription."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "What's in file.py?"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "file.py"},
                        }
                    ]
                },
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:30:03Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "def hello(): pass"},
            },
            {
                "uuid": "msg4",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {
                    "content": [
                        {"type": "text", "text": "It defines a hello function."}
                    ]
                },
            },
        )
        turn = Turn(
            uuids=["msg1", "msg2", "msg3", "msg4"],
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:05Z",
        )

        result = turns_to_append_units([turn], records)

        assert len(result) == 1
        # Tool results should be skipped in transcription
        assert "def hello()" not in result[0].text
        # But tool use should be mentioned
        assert "Read" in result[0].text

    def test_turn_preserves_timestamps_exactly(self) -> None:
        """AppendUnit preserves turn timestamps exactly as-is."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00.123456Z",
                "message": {"content": "Hello"},
            },
        )
        turn = Turn(
            uuids=["msg1"],
            time_start="2024-01-21T14:30:00.123456Z",
            time_end="2024-01-21T14:30:00.123456Z",
        )

        result = turns_to_append_units([turn], records)

        assert result[0].time_start == "2024-01-21T14:30:00.123456Z"
        assert result[0].time_end == "2024-01-21T14:30:00.123456Z"

    def test_append_units_are_temporal(self) -> None:
        """All returned AppendUnits have is_temporal=True."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
        )
        turn = Turn(
            uuids=["msg1"],
            time_start="2024-01-21T14:30:00Z",
            time_end="2024-01-21T14:30:00Z",
        )

        result = turns_to_append_units([turn], records)

        assert result[0].is_temporal is True
