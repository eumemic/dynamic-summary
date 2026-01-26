"""Tests for revert detection at turn granularity.

Tests that reverts are detected correctly and trigger appropriate truncation.
With stateless sync, truncation is time-based rather than span-based.

The sliding window algorithm in find_truncation_point() ensures we always
stop at turn boundaries, so mid-turn reverts are handled correctly.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragzoom_claude_code.transcript_sync import execute_sync

from tests.conftest import FakeTranscriptClient


class TestRevertDetectionAtTurnGranularity:
    """Tests for revert detection at turn boundaries using stateless sync.

    With stateless sync, reverts are detected by comparing connection point
    timestamps with indexed_time_end. Truncation uses time-based truncation
    instead of span-based.
    """

    def test_revert_within_turn_truncates_to_before_turn(self, tmp_path: Path) -> None:
        """If user reverts within an indexed turn, truncate to turn boundary.

        Scenario:
        - Indexed: Turn 1 (msg1->msg2), Turn 2 (msg3->msg4)
        - User reverts to msg3 (start of Turn 2) and continues differently
        - The sliding window finds connection at msg2 (turn boundary before msg3)
        - Should truncate and re-index from msg3
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
        assert len(client.batch_append_calls) == 1, "Should batch append 2 turns"

        # Reset client for tracking but preserve indexed state
        client.truncate_from_time_calls.clear()
        client.appends.clear()
        client.batch_append_calls.clear()

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
            f"Result: truncated={result2.truncated}"
        )

        # With stateless sync, truncation uses time-based (truncate_from_time)
        # The connection point is msg2 (end of Turn 1), so truncate to its timestamp
        assert len(client.truncate_from_time_calls) == 1
        truncate_doc, truncate_time = client.truncate_from_time_calls[0]
        assert truncate_doc == "transcript"
        assert truncate_time == "2024-01-01T10:01:00Z"  # msg2's timestamp

        # After truncation, we re-index from Turn 2's start
        # So msg3 and msg4-alt should be in the appended content
        assert "msg3" in result2.appended_uuids
        # msg4-alt should be there since it replaces msg4
        assert "msg4-alt" in result2.appended_uuids

    def test_revert_at_turn_boundary_truncates_correctly(self, tmp_path: Path) -> None:
        """If user reverts to the END of a turn (turn boundary), truncate
        and re-index from the new turn.

        Scenario:
        - Indexed: Turn 1 (msg1->msg2), Turn 2 (msg3->msg4)
        - User reverts to msg2 (end of Turn 1) and starts new Turn 2
        - Connection point is msg2 (Turn 1's end)
        - Should truncate and re-index new Turn 2
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

        # Reset for tracking but preserve indexed state
        client.truncate_from_time_calls.clear()
        client.appends.clear()
        client.batch_append_calls.clear()

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

        # Should truncate using time-based truncation
        assert result2.truncated
        assert len(client.truncate_from_time_calls) == 1
        truncate_doc, truncate_time = client.truncate_from_time_calls[0]
        assert truncate_doc == "transcript"
        # Truncate to msg2's timestamp (connection point)
        assert truncate_time == "2024-01-01T10:01:00Z"

        # Should re-index the new Turn 2
        assert "msg3-alt" in result2.appended_uuids
        assert "msg4-alt" in result2.appended_uuids

    def test_revert_preserves_untouched_turns(self, tmp_path: Path) -> None:
        """Turns before the revert point should remain indexed.

        Scenario:
        - Indexed: Turn 1, Turn 2, Turn 3
        - User reverts to end of Turn 1
        - Turn 1 should remain (truncation only removes content after)
        - Turns 2 & 3 get replaced
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

        # Reset for tracking
        client.truncate_from_time_calls.clear()
        client.batch_append_calls.clear()

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

        # Should truncate using time-based truncation
        assert result2.truncated
        assert len(client.truncate_from_time_calls) == 1
        truncate_doc, truncate_time = client.truncate_from_time_calls[0]
        assert truncate_doc == "transcript"
        # Truncate to msg2's timestamp (end of Turn 1)
        assert truncate_time == "2024-01-01T10:01:00Z"

        # New turn should be in appended content
        assert "msg3-new" in result2.appended_uuids
        assert "msg4-new" in result2.appended_uuids

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
        assert len(client.batch_append_calls) == 1

        # Reset for tracking but preserve indexed state
        client.truncate_from_time_calls.clear()
        client.batch_append_calls.clear()

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
        assert len(client.truncate_from_time_calls) == 0

        # Should have appended the new turn
        assert len(client.batch_append_calls) == 1
        assert "msg3" in result2.appended_uuids
        assert "msg4" in result2.appended_uuids
