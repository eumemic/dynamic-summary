"""Tests for revert detection with step-level granularity.

Tests that reverts are detected correctly and trigger appropriate truncation.
With stateless sync, truncation is time-based rather than span-based.

With step-level chunking, every record is a valid truncation point, so
truncation occurs at the exact connection point (no turn boundary constraints).
"""

from __future__ import annotations

import json
from pathlib import Path

from ragzoom_claude_code.transcript_sync import execute_sync

from tests.conftest import FakeTranscriptClient


class TestRevertDetectionAtStepGranularity:
    """Tests for revert detection using stateless sync with step-level granularity.

    With stateless sync, reverts are detected by comparing connection point
    timestamps with indexed_time_end. Truncation uses time-based truncation
    instead of span-based.

    With step-level chunking, truncation occurs at the exact connection point
    (the last message before the branch), not at turn boundaries.
    """

    def test_revert_with_timestamps_after_indexed_content(self, tmp_path: Path) -> None:
        """Revert where new content has timestamps AFTER indexed content.

        This is the typical revert case: user creates new content at a later time,
        branching from an earlier point. The new content's timestamps are all
        greater than the indexed content's timestamps.

        Scenario:
        - Indexed: msg1->msg2->msg3->msg4 (times 10:00 through 10:03)
        - User reverts to msg2 and adds new content at 11:00+ (after indexed)
        - Connection point is msg2 (the branch point)
        - Should truncate from msg2's timestamp and re-index new branch
        """
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"
        client = FakeTranscriptClient()

        # Initial sync: 4 steps
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
        result1 = execute_sync(transcript_path, document_id, client)
        assert not result1.truncated, "Initial sync should not truncate"
        assert len(client.batch_append_calls) == 1

        # Reset client for tracking but preserve indexed state
        client.truncate_from_time_calls.clear()
        client.appends.clear()
        client.batch_append_calls.clear()

        # User reverts to msg2 and starts new content AFTER indexed_time_end
        # This is the typical real-world case: user creates new content later
        # msg1 -> msg2 -> msg3 -> msg4 (original, now orphaned)
        #              \-> msg3-alt (user) -> msg4-alt (assistant)
        with open(transcript_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "uuid": "msg3-alt",
                        "parentUuid": "msg2",
                        "type": "user",
                        "timestamp": "2024-01-01T11:00:00Z",  # After indexed_time_end
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
                        "timestamp": "2024-01-01T11:01:00Z",  # After indexed_time_end
                        "message": {
                            "content": [
                                {"type": "text", "text": "Different second answer"}
                            ]
                        },
                    }
                )
                + "\n"
            )
        result2 = execute_sync(transcript_path, document_id, client)

        # Should have truncated because the new branch diverges from indexed content
        assert result2.truncated

        # With step-level granularity:
        # - Walking: msg4-alt (11:01) > 10:03, slide
        # - Walking: msg3-alt (11:00) > 10:03, slide
        # - Walking: msg2 (10:01) <= 10:03, stop. R=msg2, S=msg3-alt
        # Truncate to msg2's timestamp (the actual connection point)
        assert len(client.truncate_from_time_calls) == 1
        truncate_doc, truncate_time = client.truncate_from_time_calls[0]
        assert truncate_doc == "transcript"
        assert truncate_time == "2024-01-01T10:01:00Z"  # msg2's timestamp

        # After truncation, ancestry chain from msg4-alt to msg2 (exclusive):
        # [msg3-alt, msg4-alt]
        # group_into_turns creates 1 turn: msg3-alt (user) -> msg4-alt (assistant)
        assert result2.steps_appended >= 1

    def test_revert_with_mid_range_timestamps_detects_revert(
        self, tmp_path: Path
    ) -> None:
        """Revert where new content has timestamps within indexed time range.

        This edge case occurs when new content is created with timestamps that
        fall between existing indexed timestamps. The algorithm detects this as
        a revert because R.timestamp < indexed_time_end.

        Note: In this case, R is a NEW record (not actually indexed), but since
        its timestamp < indexed_time_end, the algorithm treats it as a connection
        point. This is a known limitation of the timestamp-based approach.

        Scenario:
        - Indexed: msg1->msg2->msg3->msg4 (times 10:00 through 10:03)
        - User reverts to msg2 and adds new content at 10:02:30 (within range!)
        - The algorithm finds msg3-alt as R (because 10:02:30 < 10:03)
        - Truncates at msg3-alt's timestamp
        """
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"
        client = FakeTranscriptClient()

        # Initial sync: 4 steps
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
        execute_sync(transcript_path, document_id, client)

        # Reset for tracking but preserve indexed state
        client.truncate_from_time_calls.clear()
        client.appends.clear()
        client.batch_append_calls.clear()

        # User reverts to msg2 and creates content with mid-range timestamps
        # This is an edge case where new content timestamp falls within indexed range
        with open(transcript_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "uuid": "msg3-alt",
                        "parentUuid": "msg2",
                        "type": "user",
                        "timestamp": "2024-01-01T10:02:30Z",  # Within indexed range!
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
        result2 = execute_sync(transcript_path, document_id, client)

        # Should truncate - the algorithm detects a revert because:
        # - Walking: msg4-alt (10:03:30 > 10:03), slide
        # - Walking: msg3-alt (10:02:30 < 10:03), stop. R=msg3-alt
        # - R.timestamp (10:02:30) < indexed_time_end (10:03) => revert detected
        assert result2.truncated
        assert len(client.truncate_from_time_calls) == 1
        truncate_doc, truncate_time = client.truncate_from_time_calls[0]
        assert truncate_doc == "transcript"
        assert truncate_time == "2024-01-01T10:02:30Z"

        # In this edge case, the ancestry chain from msg4-alt to msg3-alt (exclusive)
        # is just [msg4-alt]. With step-based functions, msg4-alt (assistant type)
        # becomes its own step and is appended.
        assert result2.steps_appended == 1

    def test_revert_preserves_untouched_content(self, tmp_path: Path) -> None:
        """Content before the connection point should remain indexed.

        Scenario:
        - Indexed: msg1 through msg6
        - User reverts to msg2 and continues differently
        - Content before msg2 should remain (truncation only removes content after)
        - Messages after msg2 get replaced
        """
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"
        client = FakeTranscriptClient()

        # Initial sync: 6 steps
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
        execute_sync(transcript_path, document_id, client)

        # Reset for tracking
        client.truncate_from_time_calls.clear()
        client.batch_append_calls.clear()

        # Revert to msg2, new branch
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
        result2 = execute_sync(transcript_path, document_id, client)

        # Should truncate using time-based truncation
        assert result2.truncated
        assert len(client.truncate_from_time_calls) == 1
        truncate_doc, truncate_time = client.truncate_from_time_calls[0]
        assert truncate_doc == "transcript"
        # Connection point is msg2, so truncate to msg2's timestamp
        assert truncate_time == "2024-01-01T10:01:00Z"

        # New steps should have been appended
        assert result2.steps_appended >= 1

    def test_no_revert_continues_normally(self, tmp_path: Path) -> None:
        """When there's no revert, new steps are simply appended."""
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "transcript"
        client = FakeTranscriptClient()

        # Initial sync: 2 steps
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
        execute_sync(transcript_path, document_id, client)
        assert len(client.batch_append_calls) == 1

        # Reset for tracking but preserve indexed state
        client.truncate_from_time_calls.clear()
        client.batch_append_calls.clear()

        # Add more steps (no revert, linear continuation)
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
        result2 = execute_sync(transcript_path, document_id, client)

        # No truncation needed
        assert not result2.truncated
        assert len(client.truncate_from_time_calls) == 0

        # Should have appended the new content
        # Note: Until execute_sync() is updated to use step-based functions,
        # this creates 1 turn (msg3 -> msg4) not 2 separate steps
        assert len(client.batch_append_calls) == 1
        assert result2.steps_appended >= 1
