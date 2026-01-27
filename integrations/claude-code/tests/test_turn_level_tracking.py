"""Tests for turn-level tracking in transcript sync.

Each conversation turn should map to exactly one AppendUnit/leaf node.
Verifies that turns are correctly identified, grouped, and indexed.

These tests verify observable behavior through the sync result and client calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragzoom_claude_code.transcript_sync import execute_sync

from tests.conftest import FakeTranscriptClient


class TestTurnLevelTracking:
    """Tests that each turn is processed separately."""

    def test_each_turn_creates_one_append_unit(self, tmp_path: Path) -> None:
        """Each conversation turn should create exactly one AppendUnit."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"

        # Create transcript with 3 turns:
        # Turn 1: msg1 (user) -> msg2 (assistant)
        # Turn 2: msg3 (user) -> msg4 (assistant)
        # Turn 3: msg5 (user) - standalone
        transcript_path.write_text(
            "\n".join(
                [
                    # Turn 1
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
                    # Turn 2
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
                    # Turn 3 (standalone user message)
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

        # Should have batch_append called once with 3 units (one per turn)
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 3, f"Expected 3 units (one per turn), got {len(units)}"

        # All turns should have been appended
        assert result.turns_appended == 3

    def test_turn_units_have_correct_timestamps(self, tmp_path: Path) -> None:
        """Each turn's AppendUnit should have correct timestamps."""
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

        # Should have 2 units (two turns)
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 2

        # First turn: msg1->msg2
        assert units[0].time_start == "2024-01-01T10:00:00Z"
        assert units[0].time_end == "2024-01-01T10:01:00Z"

        # Second turn: msg3 (standalone)
        assert units[1].time_start == "2024-01-01T10:02:00Z"
        assert units[1].time_end == "2024-01-01T10:02:00Z"

    def test_tool_results_grouped_in_same_turn(self, tmp_path: Path) -> None:
        """Tool results should be in the same turn as their user message."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"

        transcript_path.write_text(
            "\n".join(
                [
                    # Turn 1: user message
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "Please help"},
                        }
                    ),
                    # Assistant with tool use
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
                    # Tool result (user message with toolUseResult)
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
                    # Final assistant response
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

        # Should have only 1 unit (all msgs are one turn)
        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 1

        # One turn was appended
        assert result.turns_appended == 1

    def test_incremental_sync_adds_new_turn(self, tmp_path: Path) -> None:
        """Incremental sync should add new turns correctly."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"
        client = FakeTranscriptClient()

        # Initial sync with one turn
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

        # First sync should append 1 turn
        assert len(client.batch_append_calls) == 1
        assert result1.turns_appended == 1

        # Reset tracking
        client.batch_append_calls.clear()

        # Add another turn and sync again
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

        # Second sync should append only the new turn
        assert len(client.batch_append_calls) == 1
        assert result2.turns_appended == 1
