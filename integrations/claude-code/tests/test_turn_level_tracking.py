"""Tests for step-level tracking in transcript sync.

Each conversation step (message) should map to exactly one AppendUnit/leaf node.
Verifies that steps are correctly identified, filtered, and indexed.

These tests verify observable behavior through the sync result and client calls.

Note: This was previously turn-level tracking, but step-level chunking provides
finer granularity where each message becomes its own leaf node.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragzoom_claude_code.transcript_sync import execute_sync

from tests.conftest import FakeTranscriptClient


class TestStepLevelTracking:
    """Tests that each step (message) is processed separately."""

    def test_each_step_creates_one_append_unit(self, tmp_path: Path) -> None:
        """Each conversation step should create exactly one AppendUnit."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"

        # Create transcript with 5 steps:
        # Step 1: msg1 (user)
        # Step 2: msg2 (assistant)
        # Step 3: msg3 (user)
        # Step 4: msg4 (assistant)
        # Step 5: msg5 (user)
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "First question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "First answer"}]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {"content": "Second question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:03:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "Second answer"}]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg5",
                            "parentUuid": "msg4",
                            "type": "user",
                            "timestamp": "2024-01-01T10:04:00Z",
                            "message": {"content": "Third question"},
                        }
                    ),
                ]
            )
            + "\n"
        )

        client = FakeTranscriptClient()
        result = execute_sync(transcript_path, document_id, client)

        # Should have batch_append called once with 5 units (one per step)
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 5, f"Expected 5 units (one per step), got {len(units)}"

        # All steps should have been appended
        assert result.steps_appended == 5

    def test_step_units_have_point_in_time_timestamps(self, tmp_path: Path) -> None:
        """Each step's AppendUnit should have time_start == time_end (point-in-time)."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"

        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "Short"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {"content": [{"type": "text", "text": "Reply"}]},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {"content": "Another message here"},
                        }
                    ),
                ]
            )
            + "\n"
        )

        client = FakeTranscriptClient()
        execute_sync(transcript_path, document_id, client)

        # Should have 3 units (one per step)
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 3

        # Each step has point-in-time timestamps (time_start == time_end)
        assert units[0].time_start == "2024-01-01T10:00:00Z"
        assert units[0].time_end == "2024-01-01T10:00:00Z"

        assert units[1].time_start == "2024-01-01T10:01:00Z"
        assert units[1].time_end == "2024-01-01T10:01:00Z"

        assert units[2].time_start == "2024-01-01T10:02:00Z"
        assert units[2].time_end == "2024-01-01T10:02:00Z"

    def test_tool_results_are_separate_steps(self, tmp_path: Path) -> None:
        """Tool results should be their own separate steps."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"

        transcript_path.write_text(
            "\n".join(
                [
                    # Step 1: user message
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "Please help"},
                        }
                    ),
                    # Step 2: Assistant with tool use
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "name": "Read",
                                        "input": {"file_path": "/test.txt"},
                                    }
                                ]
                            },
                        }
                    ),
                    # Step 3: Tool result (user message with toolUseResult)
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:01:01Z",
                            "toolUseResult": {"content": "file contents"},
                            "message": {"content": "file contents"},
                        }
                    ),
                    # Step 4: Final assistant response
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {
                                "content": [
                                    {"type": "text", "text": "Here is the file"}
                                ]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        client = FakeTranscriptClient()
        result = execute_sync(transcript_path, document_id, client)

        # Should have 4 units (one per step, including tool result)
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 4

        # Four steps were appended
        assert result.steps_appended == 4

    def test_incremental_sync_adds_new_steps(self, tmp_path: Path) -> None:
        """Incremental sync should add new steps correctly."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"
        client = FakeTranscriptClient()

        # Initial sync with two steps
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "First question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "First answer"}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        result1 = execute_sync(transcript_path, document_id, client)

        # First sync should append 2 steps
        assert len(client.batch_append_calls) == 1
        assert result1.steps_appended == 2

        # Reset tracking
        client.batch_append_calls.clear()

        # Add two more steps and sync again
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "First question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "First answer"}]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {"content": "Second question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:03:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "Second answer"}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        result2 = execute_sync(transcript_path, document_id, client)

        # Second sync should append only the new steps
        assert len(client.batch_append_calls) == 1
        assert result2.steps_appended == 2


class TestTurnLevelTracking:
    """Legacy test class - renamed for backwards compatibility.

    These tests are kept for documentation purposes but now verify
    step-level behavior which replaced turn-level tracking.
    """

    def test_each_turn_creates_one_append_unit(self, tmp_path: Path) -> None:
        """Verify step-level behavior (each message = one unit)."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"

        # 5 messages = 5 steps (not 3 turns as before)
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "First question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "First answer"}]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {"content": "Second question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:03:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "Second answer"}]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg5",
                            "parentUuid": "msg4",
                            "type": "user",
                            "timestamp": "2024-01-01T10:04:00Z",
                            "message": {"content": "Third question"},
                        }
                    ),
                ]
            )
            + "\n"
        )

        client = FakeTranscriptClient()
        result = execute_sync(transcript_path, document_id, client)

        # With step-level chunking: 5 messages = 5 units
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 5, f"Expected 5 units (one per step), got {len(units)}"
        assert result.steps_appended == 5

    def test_turn_units_have_correct_timestamps(self, tmp_path: Path) -> None:
        """Verify step-level timestamps (point-in-time, not ranges)."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"

        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "Short"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {"content": [{"type": "text", "text": "Reply"}]},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {"content": "Another message here"},
                        }
                    ),
                ]
            )
            + "\n"
        )

        client = FakeTranscriptClient()
        execute_sync(transcript_path, document_id, client)

        # With step-level: 3 messages = 3 units
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 3

        # Each step has point-in-time timestamps (time_start == time_end)
        assert units[0].time_start == "2024-01-01T10:00:00Z"
        assert units[0].time_end == "2024-01-01T10:00:00Z"

        assert units[1].time_start == "2024-01-01T10:01:00Z"
        assert units[1].time_end == "2024-01-01T10:01:00Z"

        assert units[2].time_start == "2024-01-01T10:02:00Z"
        assert units[2].time_end == "2024-01-01T10:02:00Z"

    def test_tool_results_grouped_in_same_turn(self, tmp_path: Path) -> None:
        """Verify step-level behavior: tool results are their own steps."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"

        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "Please help"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {
                                "content": [
                                    {
                                        "type": "tool_use",
                                        "name": "Read",
                                        "input": {"file_path": "/test.txt"},
                                    }
                                ]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:01:01Z",
                            "toolUseResult": {"content": "file contents"},
                            "message": {"content": "file contents"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {
                                "content": [
                                    {"type": "text", "text": "Here is the file"}
                                ]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        client = FakeTranscriptClient()
        result = execute_sync(transcript_path, document_id, client)

        # With step-level: 4 messages = 4 units (tool results are separate)
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 4
        assert result.steps_appended == 4

    def test_incremental_sync_adds_new_turn(self, tmp_path: Path) -> None:
        """Verify incremental sync with step-level behavior."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"
        client = FakeTranscriptClient()

        # Initial sync with two steps
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "First question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "First answer"}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        result1 = execute_sync(transcript_path, document_id, client)

        # First sync should append 2 steps
        assert len(client.batch_append_calls) == 1
        assert result1.steps_appended == 2

        # Reset tracking
        client.batch_append_calls.clear()

        # Add two more steps and sync again
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "First question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "First answer"}]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {"content": "Second question"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:03:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "Second answer"}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        result2 = execute_sync(transcript_path, document_id, client)

        # Second sync should append only the new 2 steps
        assert len(client.batch_append_calls) == 1
        assert result2.steps_appended == 2
