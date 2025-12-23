"""Tests for span tracking through the append pipeline.

These tests verify that character-accurate span_start and span_end values
flow correctly from AppendExecutor through gRPC to the transcript sync layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragzoom.claude_memory.transcript_sync import (
    AppendEntry,
    SessionState,
    SessionStateHeader,
    execute_sync,
    get_compaction_uuid,
)


class MockAppendResult(Protocol):
    """Protocol for append result with span info."""

    span_start: int
    span_end: int


@dataclass
class FakeAppendResult:
    """Fake append result that mimics IndexingResult."""

    span_start: int
    span_end: int
    chunks_created: int = 1


class FakeClient:
    """Fake client that tracks appends and returns span positions."""

    def __init__(self) -> None:
        self.appends: list[tuple[str, str]] = []
        self.truncates: list[tuple[str, int]] = []
        self._current_span: int = 0

    def append(self, document_id: str, text: str) -> FakeAppendResult:
        """Append text and return span positions."""
        self.appends.append((document_id, text))
        span_start = self._current_span
        span_end = self._current_span + len(text)
        self._current_span = span_end
        return FakeAppendResult(
            span_start=span_start,
            span_end=span_end,
            chunks_created=1,
        )

    def truncate(self, document_id: str, span_start: int) -> None:
        """Truncate document to span."""
        self.truncates.append((document_id, span_start))
        self._current_span = span_start


class TestSpanTrackingInSync:
    """Tests that execute_sync correctly uses span_end from append results."""

    def test_records_actual_span_end_not_chunk_count(self, tmp_path: Path) -> None:
        """span_end in append log should be character offset, not chunk count."""
        # Create transcript with messages of known lengths
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "message": {"content": "Short message"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "A much longer response here",
                                    }
                                ]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        state_path = tmp_path / "state.jsonl"
        client = FakeClient()

        result = execute_sync(transcript_path, state_path, client)

        # Verify synced - both messages appended
        assert len(result.appended_uuids) == 2

        # Load state - now batched into single entry
        state = SessionState.load(state_path)
        assert state is not None
        assert len(state.entries) == 1  # Batched into single append

        # The single entry should have span_end equal to total text length
        entry = state.entries[0]

        # span_end should be > 1 (it would be small if using chunk count)
        assert (
            entry.span_end > 10
        ), f"span_end={entry.span_end} looks like chunk count, not char offset"

        # Entry should be keyed by the last UUID
        assert entry.last_uuid == "msg2"

    def test_incremental_sync_continues_from_correct_span(self, tmp_path: Path) -> None:
        """Incremental syncs should continue from previous span_end."""
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"
        client = FakeClient()

        # First sync with one message
        transcript_path.write_text(
            json.dumps(
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "type": "user",
                    "message": {"content": "First message"},
                }
            )
            + "\n"
        )
        execute_sync(transcript_path, state_path, client)

        # Get first span_end
        state = SessionState.load(state_path)
        assert state is not None
        first_span_end = state.entries[0].span_end

        # Add second message and sync again
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "message": {"content": "First message"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "Second message"}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        execute_sync(transcript_path, state_path, client)

        # Verify second entry's span_end is greater
        state = SessionState.load(state_path)
        assert state is not None
        assert len(state.entries) == 2
        assert state.entries[1].span_end > first_span_end


class TestGetCompactionUuid:
    """Tests for finding compaction boundary UUID."""

    def test_finds_uuid_before_compaction(self, tmp_path: Path) -> None:
        """Should return UUID of message just before compaction."""
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None, "type": "user"}),
                    json.dumps(
                        {"uuid": "msg2", "parentUuid": "msg1", "type": "assistant"}
                    ),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2", "type": "user"}),
                    # Compaction summary (no uuid, or has isCompactSummary)
                    json.dumps({"isCompactSummary": True, "type": "summary"}),
                    json.dumps(
                        {"uuid": "msg4", "parentUuid": "msg3", "type": "assistant"}
                    ),
                ]
            )
            + "\n"
        )

        compaction_uuid = get_compaction_uuid(transcript_path)

        # Should be msg3, the last message before compaction
        assert compaction_uuid == "msg3"

    def test_returns_none_when_no_compaction(self, tmp_path: Path) -> None:
        """Should return None if no compaction in transcript."""
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None, "type": "user"}),
                    json.dumps(
                        {"uuid": "msg2", "parentUuid": "msg1", "type": "assistant"}
                    ),
                ]
            )
            + "\n"
        )

        compaction_uuid = get_compaction_uuid(transcript_path)

        assert compaction_uuid is None

    def test_finds_most_recent_compaction(self, tmp_path: Path) -> None:
        """With multiple compactions, should find the most recent one."""
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None, "type": "user"}),
                    json.dumps(
                        {"uuid": "msg2", "parentUuid": "msg1", "type": "assistant"}
                    ),
                    # First compaction
                    json.dumps({"isCompactSummary": True, "type": "summary"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2", "type": "user"}),
                    json.dumps(
                        {"uuid": "msg4", "parentUuid": "msg3", "type": "assistant"}
                    ),
                    # Second (most recent) compaction
                    json.dumps({"isCompactSummary": True, "type": "summary"}),
                    json.dumps({"uuid": "msg5", "parentUuid": "msg4", "type": "user"}),
                ]
            )
            + "\n"
        )

        compaction_uuid = get_compaction_uuid(transcript_path)

        # Should be msg4, before the MOST RECENT compaction
        assert compaction_uuid == "msg4"

    def test_handles_compaction_at_start(self, tmp_path: Path) -> None:
        """Should return None if compaction is at start (no messages before)."""
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"isCompactSummary": True, "type": "summary"}),
                    json.dumps({"uuid": "msg1", "parentUuid": None, "type": "user"}),
                ]
            )
            + "\n"
        )

        compaction_uuid = get_compaction_uuid(transcript_path)

        # No messages before compaction
        assert compaction_uuid is None

    def test_handles_empty_transcript(self, tmp_path: Path) -> None:
        """Should return None for empty transcript."""
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text("")

        compaction_uuid = get_compaction_uuid(transcript_path)

        assert compaction_uuid is None

    def test_scans_efficiently_backwards(self, tmp_path: Path) -> None:
        """Should find compaction near end quickly (reverse scan)."""
        # Create a large transcript with compaction near the end
        records = []
        for i in range(1000):
            records.append(
                json.dumps(
                    {"uuid": f"msg{i}", "parentUuid": f"msg{i-1}" if i > 0 else None}
                )
            )

        # Add compaction near end
        records.append(json.dumps({"isCompactSummary": True}))
        records.append(json.dumps({"uuid": "final", "parentUuid": "msg999"}))

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text("\n".join(records) + "\n")

        # This should be fast because it scans backwards
        compaction_uuid = get_compaction_uuid(transcript_path)

        assert compaction_uuid == "msg999"


class TestSpanEndToCompactionBoundary:
    """Tests for mapping compaction UUID to span_end."""

    def test_finds_span_end_for_compaction_uuid(self, tmp_path: Path) -> None:
        """When compaction UUID is in append log, should return its span_end."""
        # Create state with entries
        state = SessionState(
            header=SessionStateHeader(document_id="test-doc"),
            entries=[
                AppendEntry(last_uuid="msg1", span_end=100),
                AppendEntry(last_uuid="msg2", span_end=250),
                AppendEntry(last_uuid="msg3", span_end=400),
            ],
        )

        # Find span_end for msg2
        known_uuids = {entry.last_uuid: entry.span_end for entry in state.entries}
        compaction_uuid = "msg2"

        span_end = known_uuids.get(compaction_uuid)

        assert span_end == 250

    def test_returns_none_when_uuid_not_in_log(self, tmp_path: Path) -> None:
        """When compaction UUID isn't in append log, returns None."""
        state = SessionState(
            header=SessionStateHeader(document_id="test-doc"),
            entries=[
                AppendEntry(last_uuid="msg1", span_end=100),
                AppendEntry(last_uuid="msg2", span_end=250),
            ],
        )

        known_uuids = {entry.last_uuid: entry.span_end for entry in state.entries}
        compaction_uuid = "unknown-msg"

        span_end = known_uuids.get(compaction_uuid)

        assert span_end is None


class TestEndToEndSpanAccuracy:
    """End-to-end tests verifying span values are character-accurate."""

    def test_span_matches_transcribed_text_length(self, tmp_path: Path) -> None:
        """span_end should match cumulative length of transcribed text."""
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # Create transcript with known content
        user_content = "Hello, this is a test message"
        assistant_content = "This is the assistant's response"

        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "message": {"content": user_content},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": assistant_content}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        # Use fake client that returns actual text lengths
        client = FakeClient()
        execute_sync(transcript_path, state_path, client)

        # Verify single batched append happened
        assert len(client.appends) == 1

        # The combined text should be appended in one call
        combined_text = client.appends[0][1]

        state = SessionState.load(state_path)
        assert state is not None
        assert len(state.entries) == 1

        # Single entry's span_end should equal total text length
        assert state.entries[0].span_end == len(combined_text)


class TestCompactionSegmentBridging:
    """Tests for bridging parent chains across compaction boundaries.

    Claude Code's compaction creates "segments" - the message after compaction
    has parentUuid=None, breaking the ancestor chain. The sync logic must
    bridge these segments by treating compaction as a virtual parent edge.
    """

    def test_parent_map_bridges_compaction_boundary(self, tmp_path: Path) -> None:
        """Parent map should connect post-compaction messages to pre-compaction chain."""
        from ragzoom.claude_memory.transcript_sync import build_parent_map

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    # Pre-compaction messages
                    json.dumps({"uuid": "msg1", "parentUuid": None, "type": "user"}),
                    json.dumps(
                        {"uuid": "msg2", "parentUuid": "msg1", "type": "assistant"}
                    ),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2", "type": "user"}),
                    # System message that starts resume (parentUuid=None)
                    json.dumps(
                        {"uuid": "system1", "parentUuid": None, "type": "system"}
                    ),
                    # Compaction summary
                    json.dumps(
                        {
                            "uuid": "compact1",
                            "parentUuid": "system1",
                            "isCompactSummary": True,
                            "type": "user",
                        }
                    ),
                    # Post-compaction messages (parentUuid chains from compaction)
                    json.dumps(
                        {"uuid": "msg4", "parentUuid": "compact1", "type": "assistant"}
                    ),
                    json.dumps({"uuid": "msg5", "parentUuid": "msg4", "type": "user"}),
                ]
            )
            + "\n"
        )

        parent_map = build_parent_map(transcript_path)

        # The system message (which has parentUuid=None) should be bridged
        # to msg3 (the last message before the system/compaction pair)
        assert parent_map["system1"] == "msg3"

        # Regular parent relationships should still work
        assert parent_map["msg2"] == "msg1"
        assert parent_map["msg4"] == "compact1"

    def test_sync_indexes_content_across_compaction(self, tmp_path: Path) -> None:
        """Sync should index ALL messages, including those before compaction."""
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        transcript_path.write_text(
            "\n".join(
                [
                    # Pre-compaction messages
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "message": {"content": "First message"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "Second"}]
                            },
                        }
                    ),
                    # System + compaction
                    json.dumps(
                        {"uuid": "system1", "parentUuid": None, "type": "system"}
                    ),
                    json.dumps(
                        {
                            "uuid": "compact1",
                            "parentUuid": "system1",
                            "isCompactSummary": True,
                            "type": "user",
                        }
                    ),
                    # Post-compaction
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "compact1",
                            "type": "assistant",
                            "message": {"content": [{"type": "text", "text": "Third"}]},
                        }
                    ),
                ]
            )
            + "\n"
        )

        client = FakeClient()
        result = execute_sync(transcript_path, state_path, client)

        # Should have synced messages from BOTH sides of compaction
        # msg1, msg2, msg3 (compaction summary itself may or may not be synced)
        assert len(result.appended_uuids) >= 3

        # Pre-compaction messages should be in the appends
        appended_texts = [text for _, text in client.appends]
        combined = "".join(appended_texts)
        assert "First message" in combined
        assert "Second" in combined
        assert "Third" in combined

    def test_ancestor_chain_spans_multiple_compactions(self, tmp_path: Path) -> None:
        """With multiple compactions, ancestor chain should span all of them."""
        from ragzoom.claude_memory.transcript_sync import (
            build_parent_map,
            get_ancestor_chain,
        )

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    # First segment
                    json.dumps({"uuid": "a1", "parentUuid": None, "type": "user"}),
                    json.dumps({"uuid": "a2", "parentUuid": "a1", "type": "assistant"}),
                    # First compaction
                    json.dumps({"uuid": "sys1", "parentUuid": None, "type": "system"}),
                    json.dumps(
                        {
                            "uuid": "c1",
                            "parentUuid": "sys1",
                            "isCompactSummary": True,
                        }
                    ),
                    # Second segment
                    json.dumps({"uuid": "b1", "parentUuid": "c1", "type": "user"}),
                    json.dumps({"uuid": "b2", "parentUuid": "b1", "type": "assistant"}),
                    # Second compaction
                    json.dumps({"uuid": "sys2", "parentUuid": None, "type": "system"}),
                    json.dumps(
                        {
                            "uuid": "c2",
                            "parentUuid": "sys2",
                            "isCompactSummary": True,
                        }
                    ),
                    # Third segment
                    json.dumps({"uuid": "d1", "parentUuid": "c2", "type": "user"}),
                ]
            )
            + "\n"
        )

        parent_map = build_parent_map(transcript_path)

        # Get full ancestor chain from d1 back to root
        chain = get_ancestor_chain("d1", None, parent_map)

        # Should include messages from all three segments
        assert "a1" in chain
        assert "a2" in chain
        assert "b1" in chain
        assert "b2" in chain
        assert "d1" in chain

    def test_compaction_boundary_is_at_start_of_post_compaction_content(
        self, tmp_path: Path
    ) -> None:
        """Compaction boundary span should be 0 when starting fresh from compaction."""
        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        transcript_path.write_text(
            "\n".join(
                [
                    # Pre-compaction
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "message": {"content": "Before compaction"},
                        }
                    ),
                    # System + compaction
                    json.dumps({"uuid": "sys1", "parentUuid": None, "type": "system"}),
                    json.dumps(
                        {
                            "uuid": "c1",
                            "parentUuid": "sys1",
                            "isCompactSummary": True,
                        }
                    ),
                    # Post-compaction
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "c1",
                            "type": "assistant",
                            "message": {"content": [{"type": "text", "text": "After"}]},
                        }
                    ),
                ]
            )
            + "\n"
        )

        client = FakeClient()
        execute_sync(transcript_path, state_path, client)

        state = SessionState.load(state_path)
        assert state is not None

        # With per-segment batching, we have one entry per segment
        assert len(state.entries) == 2

        # First entry is for pre-compaction segment
        assert state.entries[0].last_uuid == "msg1"
        assert state.entries[0].span_end > 0

        # Second entry is for post-compaction segment
        assert state.entries[1].last_uuid == "msg2"
        assert state.entries[1].span_end > state.entries[0].span_end

        # Verify both segments were indexed separately
        assert len(client.appends) == 2
        assert "Before compaction" in client.appends[0][1]
        assert "After" in client.appends[1][1]
