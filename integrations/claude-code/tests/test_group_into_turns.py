"""Tests for group_into_turns() in timestamped transcript sync."""

from __future__ import annotations

import pytest

from ragzoom_claude_code.transcript_sync import group_into_turns


def _records(*items: dict[str, object]) -> dict[str, dict[str, object]]:
    """Helper to build records_by_uuid from a list of records."""
    result: dict[str, dict[str, object]] = {}
    for item in items:
        uuid = item.get("uuid")
        if isinstance(uuid, str):
            result[uuid] = item
    return result


class TestGroupIntoTurnsBasic:
    """Basic tests for group_into_turns function."""

    def test_empty_input_returns_empty_list(self) -> None:
        """Empty UUID list produces empty turn list."""
        result = group_into_turns([], {})
        assert result == []

    def test_single_user_message_creates_single_turn(self) -> None:
        """A single user message becomes one turn."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            }
        )
        result = group_into_turns(["msg1"], records)
        assert len(result) == 1
        assert result[0].uuids == ["msg1"]
        assert result[0].time_start == "2024-01-21T14:30:00Z"
        assert result[0].time_end == "2024-01-21T14:30:00Z"

    def test_user_then_assistant_creates_single_turn(self) -> None:
        """User followed by assistant response becomes one turn."""
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
        result = group_into_turns(["msg1", "msg2"], records)
        assert len(result) == 1
        assert result[0].uuids == ["msg1", "msg2"]
        assert result[0].time_start == "2024-01-21T14:30:00Z"
        assert result[0].time_end == "2024-01-21T14:30:05Z"

    def test_two_user_messages_create_two_turns(self) -> None:
        """Each user message starts a new turn."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "First question"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "First answer"}]},
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:31:00Z",
                "message": {"content": "Second question"},
            },
            {
                "uuid": "msg4",
                "type": "assistant",
                "timestamp": "2024-01-21T14:31:05Z",
                "message": {"content": [{"type": "text", "text": "Second answer"}]},
            },
        )
        result = group_into_turns(["msg1", "msg2", "msg3", "msg4"], records)
        assert len(result) == 2
        assert result[0].uuids == ["msg1", "msg2"]
        assert result[0].time_start == "2024-01-21T14:30:00Z"
        assert result[0].time_end == "2024-01-21T14:30:05Z"
        assert result[1].uuids == ["msg3", "msg4"]
        assert result[1].time_start == "2024-01-21T14:31:00Z"
        assert result[1].time_end == "2024-01-21T14:31:05Z"


class TestGroupIntoTurnsToolResults:
    """Tests for tool result handling in turn grouping."""

    def test_tool_result_stays_in_current_turn(self) -> None:
        """Tool results (user messages with toolUseResult) stay in current turn."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Run git status"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:02Z",
                "message": {
                    "content": [{"type": "tool_use", "name": "Bash", "input": {}}]
                },
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:30:03Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "On branch main..."},
            },
            {
                "uuid": "msg4",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "You're on main."}]},
            },
        )
        result = group_into_turns(["msg1", "msg2", "msg3", "msg4"], records)
        assert len(result) == 1
        assert result[0].uuids == ["msg1", "msg2", "msg3", "msg4"]
        assert result[0].time_start == "2024-01-21T14:30:00Z"
        assert result[0].time_end == "2024-01-21T14:30:05Z"

    def test_multiple_tool_calls_in_single_turn(self) -> None:
        """Multiple tool call/result cycles stay in one turn."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "What files changed?"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:02Z",
                "message": {
                    "content": [{"type": "tool_use", "name": "Bash", "input": {}}]
                },
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:30:03Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "file1.py"},
            },
            {
                "uuid": "msg4",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:04Z",
                "message": {
                    "content": [{"type": "tool_use", "name": "Read", "input": {}}]
                },
            },
            {
                "uuid": "msg5",
                "type": "user",
                "timestamp": "2024-01-21T14:30:05Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "def foo(): pass"},
            },
            {
                "uuid": "msg6",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:10Z",
                "message": {"content": [{"type": "text", "text": "Changed file1.py"}]},
            },
        )
        result = group_into_turns(
            ["msg1", "msg2", "msg3", "msg4", "msg5", "msg6"], records
        )
        assert len(result) == 1
        assert result[0].uuids == ["msg1", "msg2", "msg3", "msg4", "msg5", "msg6"]


class TestGroupIntoTurnsFiltering:
    """Tests for filtering compaction summaries and queue operations."""

    def test_filters_compaction_summaries(self) -> None:
        """Compaction summaries (isCompactSummary=true) are excluded."""
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
                "isCompactSummary": True,
                "message": {"content": [{"type": "text", "text": "Summary..."}]},
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:31:00Z",
                "message": {"content": "Next question"},
            },
        )
        result = group_into_turns(["msg1", "msg2", "msg3"], records)
        # msg2 is filtered, so msg1 is alone in first turn, msg3 starts second turn
        assert len(result) == 2
        assert result[0].uuids == ["msg1"]
        assert result[1].uuids == ["msg3"]

    def test_filters_queue_operations(self) -> None:
        """Queue operations (type=queue-operation) are excluded."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "type": "queue-operation",
                "timestamp": "2024-01-21T14:30:02Z",
            },
            {
                "uuid": "msg3",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Hi!"}]},
            },
        )
        result = group_into_turns(["msg1", "msg2", "msg3"], records)
        assert len(result) == 1
        assert result[0].uuids == ["msg1", "msg3"]

    def test_handles_records_not_in_map(self) -> None:
        """UUIDs not in records_by_uuid are skipped gracefully."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
        )
        # msg2 is not in the records map
        result = group_into_turns(["msg1", "msg2"], records)
        assert len(result) == 1
        assert result[0].uuids == ["msg1"]


class TestGroupIntoTurnsStandalone:
    """Tests for standalone user messages without assistant response."""

    def test_standalone_user_message_creates_valid_turn(self) -> None:
        """Single user message with no response creates a turn (e.g., /command)."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "/clear"},
            },
        )
        result = group_into_turns(["msg1"], records)
        assert len(result) == 1
        assert result[0].uuids == ["msg1"]
        assert result[0].time_start == "2024-01-21T14:30:00Z"
        assert result[0].time_end == "2024-01-21T14:30:00Z"

    def test_consecutive_user_messages_create_separate_turns(self) -> None:
        """Two user messages back-to-back create two single-message turns."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "/clear"},
            },
            {
                "uuid": "msg2",
                "type": "user",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": "Hello"},
            },
        )
        result = group_into_turns(["msg1", "msg2"], records)
        assert len(result) == 2
        assert result[0].uuids == ["msg1"]
        assert result[1].uuids == ["msg2"]


class TestGroupIntoTurnsToolOnly:
    """Tests for tool-only assistant messages."""

    def test_tool_only_assistant_batched_within_turn(self) -> None:
        """Assistant message with only tool calls (no text) stays in turn."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Read the file"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:02Z",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {"file": "a.py"}}
                    ]
                },
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:30:03Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "file contents"},
            },
            {
                "uuid": "msg4",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Read", "input": {"file": "b.py"}}
                    ]
                },
            },
            {
                "uuid": "msg5",
                "type": "user",
                "timestamp": "2024-01-21T14:30:06Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "more contents"},
            },
            {
                "uuid": "msg6",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:10Z",
                "message": {
                    "content": [{"type": "text", "text": "Here's what I found"}]
                },
            },
        )
        result = group_into_turns(
            ["msg1", "msg2", "msg3", "msg4", "msg5", "msg6"], records
        )
        assert len(result) == 1
        assert len(result[0].uuids) == 6


class TestGroupIntoTurnsMissingTimestamp:
    """Tests for records with missing timestamp field."""

    def test_record_without_timestamp_raises_error(self) -> None:
        """Records without timestamp field raise ValueError."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "message": {"content": "Hello"},
                # Note: no timestamp field
            },
        )
        with pytest.raises(ValueError, match="missing timestamp"):
            group_into_turns(["msg1"], records)
