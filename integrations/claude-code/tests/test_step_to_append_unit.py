"""Tests for Step to AppendUnit conversion in transcript sync."""

from __future__ import annotations

from ragzoom_claude_code.transcript_sync import Step, steps_to_append_units

from ragzoom.wrapper import AppendUnit


def _records(*items: dict[str, object]) -> dict[str, dict[str, object]]:
    """Helper to build records_by_uuid from a list of records."""
    result: dict[str, dict[str, object]] = {}
    for item in items:
        uuid = item.get("uuid")
        if isinstance(uuid, str):
            result[uuid] = item
    return result


class TestStepsToAppendUnits:
    """Tests for steps_to_append_units function."""

    def test_single_step_converts_to_single_append_unit(self) -> None:
        """A single step becomes a single AppendUnit."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
        )
        step = Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z")

        result = steps_to_append_units([step], records)

        assert len(result) == 1
        assert isinstance(result[0], AppendUnit)
        assert "Hello" in result[0].text

    def test_step_has_time_start_equals_time_end(self) -> None:
        """Step's AppendUnit has time_start == time_end (point-in-time)."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
        )
        step = Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z")

        result = steps_to_append_units([step], records)

        assert result[0].time_start == "2024-01-21T14:30:00Z"
        assert result[0].time_end == "2024-01-21T14:30:00Z"
        assert result[0].time_start == result[0].time_end

    def test_multiple_steps_convert_to_multiple_append_units(self) -> None:
        """Multiple steps become multiple AppendUnits, each with same timestamp."""
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
                "message": {"content": [{"type": "text", "text": "Reply"}]},
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:31:00Z",
                "message": {"content": "Second"},
            },
        )
        steps = [
            Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z"),
            Step(uuid="msg2", timestamp="2024-01-21T14:30:05Z"),
            Step(uuid="msg3", timestamp="2024-01-21T14:31:00Z"),
        ]

        result = steps_to_append_units(steps, records)

        assert len(result) == 3
        # Each step has point-in-time timestamp
        assert result[0].time_start == result[0].time_end == "2024-01-21T14:30:00Z"
        assert result[1].time_start == result[1].time_end == "2024-01-21T14:30:05Z"
        assert result[2].time_start == result[2].time_end == "2024-01-21T14:31:00Z"
        # Each has its own content
        assert "First" in result[0].text
        assert "Reply" in result[1].text
        assert "Second" in result[2].text

    def test_empty_steps_returns_empty_list(self) -> None:
        """Empty step list returns empty AppendUnit list."""
        result = steps_to_append_units([], {})
        assert result == []

    def test_step_with_missing_record_skipped(self) -> None:
        """Steps whose UUID is not in records are skipped."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
        )
        steps = [
            Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z"),
            Step(uuid="missing", timestamp="2024-01-21T14:30:05Z"),  # Not in records
        ]

        result = steps_to_append_units(steps, records)

        assert len(result) == 1
        assert "Hello" in result[0].text

    def test_step_with_empty_transcription_skipped(self) -> None:
        """Steps that transcribe to empty/whitespace are skipped."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": ""},  # Empty content
            },
            {
                "uuid": "msg2",
                "type": "user",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": "Real content"},
            },
        )
        steps = [
            Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z"),
            Step(uuid="msg2", timestamp="2024-01-21T14:30:05Z"),
        ]

        result = steps_to_append_units(steps, records)

        # Only the non-empty step should be included
        assert len(result) == 1
        assert "Real content" in result[0].text

    def test_step_preserves_timestamp_exactly(self) -> None:
        """AppendUnit preserves step timestamp exactly as-is."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00.123456Z",
                "message": {"content": "Hello"},
            },
        )
        step = Step(uuid="msg1", timestamp="2024-01-21T14:30:00.123456Z")

        result = steps_to_append_units([step], records)

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
        step = Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z")

        result = steps_to_append_units([step], records)

        assert result[0].is_temporal is True

    def test_step_with_tool_call_transcribes(self) -> None:
        """Tool call step is transcribed with tool information."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:00Z",
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
        )
        step = Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z")

        result = steps_to_append_units([step], records)

        assert len(result) == 1
        assert "Read" in result[0].text
        assert result[0].time_start == result[0].time_end

    def test_step_with_tool_result_transcribes(self) -> None:
        """Tool result step is transcribed as its own unit."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "def hello(): pass"},
            },
        )
        step = Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z")

        result = steps_to_append_units([step], records)

        assert len(result) == 1
        # Tool results transcribe with the result content
        assert result[0].time_start == result[0].time_end
