"""Tests for turn-level AppendEntry tracking in transcript sync.

Each conversation turn should map to exactly one leaf node, and
each turn should have its own AppendEntry in the append log.
This enables revert detection at turn granularity.

Spec: specs/timestamped-transcript-sync.md § AppendEntry Tracking
"""

from __future__ import annotations

import json
from pathlib import Path

from ragzoom.claude_memory.transcript_sync import (
    SessionState,
    execute_sync,
)
from tests.conftest import FakeTranscriptClient


class TestTurnLevelAppendEntryTracking:
    """Tests that each turn creates a separate AppendEntry."""

    def test_each_turn_creates_one_append_entry(self, tmp_path: Path) -> None:
        """Each conversation turn should create exactly one AppendEntry.

        The spec requires: "Each turn's AppendEntry.last_uuid = the last
        message UUID in that turn."
        """
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

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
        execute_sync(transcript_path, state_path, client)

        # Load state and check entries
        state = SessionState.load(state_path)
        assert state is not None

        # Should have 3 AppendEntries (one per turn)
        assert len(state.entries) == 3, (
            f"Expected 3 entries (one per turn), got {len(state.entries)}. "
            f"Entries: {[(e.last_uuid, e.span_end) for e in state.entries]}"
        )

        # Each entry should have the last UUID of its turn
        assert state.entries[0].last_uuid == "msg2", "Turn 1's last UUID should be msg2"
        assert state.entries[1].last_uuid == "msg4", "Turn 2's last UUID should be msg4"
        assert state.entries[2].last_uuid == "msg5", "Turn 3's last UUID should be msg5"

    def test_turn_entries_have_monotonic_span_ends(self, tmp_path: Path) -> None:
        """Each turn's span_end should be greater than the previous turn's."""
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

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
        execute_sync(transcript_path, state_path, client)

        state = SessionState.load(state_path)
        assert state is not None
        assert len(state.entries) == 2  # Two turns

        # span_end values should be monotonically increasing
        assert state.entries[0].span_end > 0
        assert state.entries[1].span_end > state.entries[0].span_end

    def test_tool_results_grouped_in_same_turn_entry(self, tmp_path: Path) -> None:
        """Tool results should be in the same turn as their user message."""
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

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
        execute_sync(transcript_path, state_path, client)

        state = SessionState.load(state_path)
        assert state is not None

        # Should have only 1 turn (msg1 through msg4 are all one turn)
        assert len(state.entries) == 1

        # The entry's last_uuid should be msg4 (last message in the turn)
        # Note: Tool result UUID (msg3) is excluded from appended_uuids but
        # the turn still ends with msg4
        assert state.entries[0].last_uuid == "msg4"

    def test_incremental_sync_adds_new_turn_entries(self, tmp_path: Path) -> None:
        """Incremental sync should add new entries for new turns."""
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"
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
        execute_sync(transcript_path, state_path, client)

        state = SessionState.load(state_path)
        assert state is not None
        assert len(state.entries) == 1
        first_entry = state.entries[0]
        assert first_entry.last_uuid == "msg2"

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
        execute_sync(transcript_path, state_path, client)

        state = SessionState.load(state_path)
        assert state is not None

        # Should have 2 entries now
        assert len(state.entries) == 2
        assert state.entries[0].last_uuid == "msg2"
        assert state.entries[1].last_uuid == "msg4"
        # Second entry's span_end should be greater
        assert state.entries[1].span_end > state.entries[0].span_end
