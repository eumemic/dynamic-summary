"""Tests for revert detection at turn granularity.

When using turn-level AppendEntry tracking, revert detection must understand
turn boundaries. If a common ancestor falls in the middle of a turn (not at
a turn boundary), we need to truncate to *before* that turn.

Spec: specs/timestamped-transcript-sync.md § AppendEntry Tracking:
"On revert detection:
- Common ancestor found in middle of a turn → truncate to before that turn
- Re-index from the turn boundary"
"""

from __future__ import annotations

import json
from pathlib import Path

from ragzoom.claude_memory.transcript_sync import (
    SessionState,
    execute_sync,
)
from tests.conftest import FakeTranscriptClient


class TestRevertDetectionAtTurnGranularity:
    """Tests for revert detection when AppendEntries track turn boundaries."""

    def test_revert_within_turn_truncates_to_before_turn(self, tmp_path: Path) -> None:
        """If user reverts to a message WITHIN an indexed turn, truncate to
        before that turn and re-index from the turn boundary.

        Scenario:
        - Indexed: Turn 1 (msg1->msg2), Turn 2 (msg3->msg4)
        - User reverts to msg3 (start of Turn 2) and continues differently
        - Common ancestor is msg3, which is within Turn 2
        - Should truncate to Turn 1's span_end and re-index from msg3
        """
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"
        client = FakeTranscriptClient()

        # Initial sync: 2 turns
        # Turn 1: msg1 (user) -> msg2 (assistant)
        # Turn 2: msg3 (user) -> msg4 (assistant)
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
        result1 = execute_sync(transcript_path, state_path, client)
        assert not result1.truncated, "Initial sync should not truncate"

        # Verify initial state: 2 entries (one per turn)
        state = SessionState.load(state_path)
        assert state is not None
        assert len(state.entries) == 2
        turn1_span = state.entries[0].span_end
        turn2_span = state.entries[1].span_end
        assert turn2_span > turn1_span

        # Reset client for tracking
        client.truncates.clear()
        client.appends.clear()

        # Now user reverts to msg3 (start of Turn 2) and continues differently
        # Claude Code transcripts are append-only, so old messages remain
        # The transcript now has: original messages + new branch from msg3
        # msg1 -> msg2 -> msg3 -> msg4 (original, now orphaned)
        #                      \-> msg4-alt (new branch, current head)
        with open(transcript_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "uuid": "msg4-alt",
                        "parentUuid": "msg3",
                        "type": "assistant",
                        "timestamp": "2024-01-01T10:03:30Z",
                        "message": {
                            "content": [
                                {"type": "text", "text": "Different second answer"}
                            ]
                        },
                    }
                )
                + "\n"
            )
        result2 = execute_sync(transcript_path, state_path, client)

        # Should have truncated because msg4-alt branches from msg3
        assert result2.truncated, (
            "Should detect revert and truncate. "
            f"Result: truncated={result2.truncated}, truncate_span={result2.truncate_span}"
        )

        # The common ancestor is msg3, which is the START of Turn 2
        # But Turn 2's AppendEntry has last_uuid=msg4
        # Since msg3 is WITHIN Turn 2, we should truncate to Turn 1's boundary
        assert result2.truncate_span == turn1_span, (
            f"Should truncate to Turn 1's span_end ({turn1_span}), "
            f"not Turn 2's ({turn2_span}). "
            f"Got truncate_span={result2.truncate_span}"
        )

        # Client should have been told to truncate to Turn 1's span
        assert len(client.truncates) == 1
        assert client.truncates[0][1] == turn1_span

        # After truncation, we re-index from Turn 2's start
        # So msg3 and msg4-alt should be in the appended content
        assert "msg3" in result2.appended_uuids
        # msg4-alt should be there since it replaces msg4
        assert "msg4-alt" in result2.appended_uuids

    def test_revert_at_turn_boundary_truncates_correctly(self, tmp_path: Path) -> None:
        """If user reverts to the END of a turn (turn boundary), truncate
        to that turn's span_end and continue from there.

        Scenario:
        - Indexed: Turn 1 (msg1->msg2), Turn 2 (msg3->msg4)
        - User reverts to msg2 (end of Turn 1) and starts new Turn 2
        - Common ancestor is msg2, which is at Turn 1's boundary
        - Should truncate to Turn 1's span_end and re-index new Turn 2
        """
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"
        client = FakeTranscriptClient()

        # Initial sync: 2 turns
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
        turn1_span = state.entries[0].span_end

        # Reset for tracking
        client.truncates.clear()
        client.appends.clear()

        # User reverts to msg2 (end of Turn 1) and starts new Turn 2
        # Claude Code transcripts are append-only, so old messages remain
        # msg1 -> msg2 -> msg3 -> msg4 (original, now orphaned)
        #              \-> msg3-alt -> msg4-alt (new branch, current head)
        with open(transcript_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "uuid": "msg3-alt",
                        "parentUuid": "msg2",
                        "type": "user",
                        "timestamp": "2024-01-01T10:02:30Z",
                        "message": {"content": "Different second question"},
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "uuid": "msg4-alt",
                        "parentUuid": "msg3-alt",
                        "type": "assistant",
                        "timestamp": "2024-01-01T10:03:30Z",
                        "message": {
                            "content": [
                                {"type": "text", "text": "Different second answer"}
                            ]
                        },
                    }
                )
                + "\n"
            )
        result2 = execute_sync(transcript_path, state_path, client)

        # Should truncate to Turn 1's span (msg2 is exactly at Turn 1 boundary)
        assert result2.truncated
        assert result2.truncate_span == turn1_span

        # Should re-index the new Turn 2
        assert "msg3-alt" in result2.appended_uuids
        assert "msg4-alt" in result2.appended_uuids

    def test_revert_preserves_untouched_turns(self, tmp_path: Path) -> None:
        """Turns before the revert point should remain indexed.

        Scenario:
        - Indexed: Turn 1, Turn 2, Turn 3
        - User reverts to end of Turn 1
        - Turn 1 should remain, Turns 2 & 3 get replaced
        """
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"
        client = FakeTranscriptClient()

        # Initial sync: 3 turns
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
                            "message": {"content": "Q1"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {"content": [{"type": "text", "text": "A1"}]},
                        }
                    ),
                    # Turn 2
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-01T10:02:00Z",
                            "message": {"content": "Q2"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:03:00Z",
                            "message": {"content": [{"type": "text", "text": "A2"}]},
                        }
                    ),
                    # Turn 3
                    json.dumps(
                        {
                            "uuid": "msg5",
                            "parentUuid": "msg4",
                            "type": "user",
                            "timestamp": "2024-01-01T10:04:00Z",
                            "message": {"content": "Q3"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg6",
                            "parentUuid": "msg5",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:05:00Z",
                            "message": {"content": [{"type": "text", "text": "A3"}]},
                        }
                    ),
                ]
            )
            + "\n"
        )
        execute_sync(transcript_path, state_path, client)

        state = SessionState.load(state_path)
        assert state is not None
        assert len(state.entries) == 3
        turn1_span = state.entries[0].span_end

        # Reset
        client.truncates.clear()

        # Revert to end of Turn 1, new Turn 2
        # Claude Code transcripts are append-only, so old messages remain
        # msg1 -> msg2 -> msg3 -> msg4 -> msg5 -> msg6 (original, now orphaned)
        #              \-> msg3-new -> msg4-new (new branch, current head)
        with open(transcript_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "uuid": "msg3-new",
                        "parentUuid": "msg2",
                        "type": "user",
                        "timestamp": "2024-01-01T11:00:00Z",
                        "message": {"content": "New Q2"},
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "uuid": "msg4-new",
                        "parentUuid": "msg3-new",
                        "type": "assistant",
                        "timestamp": "2024-01-01T11:01:00Z",
                        "message": {"content": [{"type": "text", "text": "New A2"}]},
                    }
                )
                + "\n"
            )
        result2 = execute_sync(transcript_path, state_path, client)

        assert result2.truncated
        assert result2.truncate_span == turn1_span

        # After sync, should have 2 entries: Turn 1 (preserved) + new Turn 2
        state2 = SessionState.load(state_path)
        assert state2 is not None
        assert len(state2.entries) == 2
        assert state2.entries[0].last_uuid == "msg2"  # Turn 1 preserved
        assert state2.entries[1].last_uuid == "msg4-new"  # New Turn 2

    def test_no_revert_continues_normally(self, tmp_path: Path) -> None:
        """When there's no revert, new turns are simply appended."""
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"
        client = FakeTranscriptClient()

        # Initial sync: 1 turn
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-01T10:00:00Z",
                            "message": {"content": "Q1"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-01T10:01:00Z",
                            "message": {"content": [{"type": "text", "text": "A1"}]},
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

        # Add another turn (no revert, linear continuation)
        # Append new messages to the existing transcript
        with open(transcript_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "uuid": "msg3",
                        "parentUuid": "msg2",
                        "type": "user",
                        "timestamp": "2024-01-01T10:02:00Z",
                        "message": {"content": "Q2"},
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "uuid": "msg4",
                        "parentUuid": "msg3",
                        "type": "assistant",
                        "timestamp": "2024-01-01T10:03:00Z",
                        "message": {"content": [{"type": "text", "text": "A2"}]},
                    }
                )
                + "\n"
            )
        result2 = execute_sync(transcript_path, state_path, client)

        # No truncation needed
        assert not result2.truncated
        assert result2.truncate_span is None

        # Should have added the new turn
        state2 = SessionState.load(state_path)
        assert state2 is not None
        assert len(state2.entries) == 2
        assert state2.entries[1].last_uuid == "msg4"
