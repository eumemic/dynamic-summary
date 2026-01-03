"""Tests for server-side sync logic (bytes-based)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from memory_service.ingestion.claude.transcript_sync import (
    _build_parent_map_from_bytes,
    _build_records_map_from_bytes,
    _get_current_head_from_bytes,
    _iter_jsonl_from_bytes,
    execute_streaming_resync,
    prepare_delta_sync,
    stream_find_common_ancestor_from_bytes,
)
from memory_service.storage import SessionCursor


def _make_jsonl(*records: dict[str, object]) -> bytes:
    """Helper to create JSONL bytes from records."""
    return b"\n".join(json.dumps(r).encode() for r in records) + b"\n"


class TestIterJsonlFromBytes:
    """Tests for _iter_jsonl_from_bytes."""

    def test_parses_jsonl_lines(self) -> None:
        content = _make_jsonl(
            {"uuid": "msg1", "type": "user"},
            {"uuid": "msg2", "type": "assistant"},
        )

        results = list(_iter_jsonl_from_bytes(content))

        assert len(results) == 2
        record0, _offset0 = results[0]
        record1, _offset1 = results[1]
        assert record0["uuid"] == "msg1"
        assert record1["uuid"] == "msg2"

    def test_handles_empty_content(self) -> None:
        results = list(_iter_jsonl_from_bytes(b""))
        assert results == []

    def test_skips_empty_lines(self) -> None:
        content = b'{"uuid": "msg1"}\n\n{"uuid": "msg2"}\n'

        results = list(_iter_jsonl_from_bytes(content))

        assert len(results) == 2


class TestGetCurrentHeadFromBytes:
    """Tests for _get_current_head_from_bytes."""

    def test_returns_last_uuid(self) -> None:
        content = _make_jsonl(
            {"uuid": "msg1"},
            {"uuid": "msg2"},
            {"uuid": "msg3"},
        )

        head = _get_current_head_from_bytes(content)

        assert head == "msg3"

    def test_returns_none_for_empty(self) -> None:
        head = _get_current_head_from_bytes(b"")
        assert head is None

    def test_skips_records_without_uuid(self) -> None:
        content = _make_jsonl(
            {"uuid": "msg1"},
            {"type": "system"},  # No uuid
            {"uuid": "msg2"},
        )

        head = _get_current_head_from_bytes(content)

        assert head == "msg2"


class TestBuildParentMapFromBytes:
    """Tests for _build_parent_map_from_bytes."""

    def test_builds_map(self) -> None:
        content = _make_jsonl(
            {"uuid": "msg1", "parentUuid": None},
            {"uuid": "msg2", "parentUuid": "msg1"},
            {"uuid": "msg3", "parentUuid": "msg2"},
        )

        parent_map = _build_parent_map_from_bytes(content)

        assert parent_map == {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

    def test_handles_branched_transcript(self) -> None:
        content = _make_jsonl(
            {"uuid": "msg1", "parentUuid": None},
            {"uuid": "msg2", "parentUuid": "msg1"},
            {"uuid": "msg3", "parentUuid": "msg2"},
            {"uuid": "msg3-alt", "parentUuid": "msg2"},  # Branch
        )

        parent_map = _build_parent_map_from_bytes(content)

        assert parent_map["msg3"] == "msg2"
        assert parent_map["msg3-alt"] == "msg2"


class TestBuildRecordsMapFromBytes:
    """Tests for _build_records_map_from_bytes."""

    def test_filters_to_requested_uuids(self) -> None:
        content = _make_jsonl(
            {"uuid": "msg1", "text": "first"},
            {"uuid": "msg2", "text": "second"},
            {"uuid": "msg3", "text": "third"},
        )

        records_map = _build_records_map_from_bytes(content, {"msg1", "msg3"})

        assert set(records_map.keys()) == {"msg1", "msg3"}
        assert records_map["msg1"]["text"] == "first"
        assert records_map["msg3"]["text"] == "third"


@dataclass
class MockRagZoomClient:
    """Mock RagZoom client for testing sync."""

    appended: list[tuple[str, list[str]]] = field(default_factory=list)
    truncated: list[tuple[str, int]] = field(default_factory=list)
    span_counter: int = 0

    def batch_append(self, document_id: str, units: list[str]) -> MockBatchAppendResult:
        self.appended.append((document_id, units))
        self.span_counter += sum(len(u) for u in units)
        return MockBatchAppendResult(span_end=self.span_counter)

    def truncate(self, document_id: str, span_start: int) -> None:
        self.truncated.append((document_id, span_start))
        self.span_counter = span_start


@dataclass
class MockBatchAppendResult:
    span_end: int


class TestStreamFindCommonAncestorFromBytes:
    """Tests for stream_find_common_ancestor_from_bytes."""

    def test_finds_ancestor_linear_chain(self) -> None:
        """Should find common ancestor in a linear chain."""
        content = _make_jsonl(
            {"uuid": "msg1", "parentUuid": None},
            {"uuid": "msg2", "parentUuid": "msg1"},
            {"uuid": "msg3", "parentUuid": "msg2"},
        )

        result = stream_find_common_ancestor_from_bytes(
            jsonl_content=content,
            current_head="msg3",
            last_indexed="msg2",
        )

        assert result.common_ancestor == "msg2"
        # Only msg3 is cached - streaming stops early once common ancestor is found
        assert "msg3" in result.records_cache

    def test_finds_ancestor_with_branch(self) -> None:
        """Should find common ancestor when there's a branch."""
        content = _make_jsonl(
            {"uuid": "msg1", "parentUuid": None},
            {"uuid": "msg2", "parentUuid": "msg1"},
            {"uuid": "msg3", "parentUuid": "msg2"},  # Original branch
            {"uuid": "msg3-alt", "parentUuid": "msg2"},  # New branch (revert)
        )

        result = stream_find_common_ancestor_from_bytes(
            jsonl_content=content,
            current_head="msg3-alt",
            last_indexed="msg3",
        )

        # Common ancestor should be msg2 (parent of both branches)
        assert result.common_ancestor == "msg2"

    def test_disjoint_branches_returns_none(self) -> None:
        """Should return None for disjoint branches."""
        content = _make_jsonl(
            {"uuid": "msg1", "parentUuid": None},
            {"uuid": "msg2", "parentUuid": "msg1"},
        )

        result = stream_find_common_ancestor_from_bytes(
            jsonl_content=content,
            current_head="msg2",
            last_indexed="nonexistent",
        )

        assert result.common_ancestor is None

    def test_empty_content(self) -> None:
        """Should handle empty content."""
        result = stream_find_common_ancestor_from_bytes(
            jsonl_content=b"",
            current_head="msg1",
            last_indexed=None,
        )

        assert result.common_ancestor is None
        assert result.records_cache == {}


class TestExecuteStreamingResync:
    """Tests for execute_streaming_resync."""

    def test_empty_content_returns_empty_result(self) -> None:
        client = MockRagZoomClient()

        result = execute_streaming_resync(
            session_id="session1",
            jsonl_content=b"",
            last_synced_uuid=None,
            span_end=0,
            client=client,
        )

        assert result.document_id == "session1"
        assert result.appended_uuids == []
        assert result.truncated is False
        assert client.appended == []

    def test_syncs_new_messages_from_scratch(self) -> None:
        """First sync (last_synced_uuid=None) should sync all messages."""
        content = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hi there"}]},
            },
        )
        client = MockRagZoomClient()

        result = execute_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid=None,
            span_end=0,
            client=client,
        )

        assert result.document_id == "session1"
        assert len(result.appended_uuids) > 0
        assert len(client.appended) > 0

    def test_resync_after_revert_truncates_and_reindexes(self) -> None:
        """After a revert, should truncate and re-index from common ancestor."""
        # Content with a branch: msg1 -> msg2 -> msg3, then revert to msg2 -> msg3-alt
        content = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hi"}]},
            },
            {
                "uuid": "msg3-alt",
                "parentUuid": "msg2",
                "type": "user",
                "message": {"content": "New branch"},
            },
        )
        client = MockRagZoomClient()

        # Simulate we had synced msg3 (which is now gone), current head is msg3-alt
        result = execute_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid="msg3",  # This was the old head (now reverted)
            span_end=100,
            client=client,
        )

        # Should have truncated (because msg3 is not an ancestor of msg3-alt)
        assert result.truncated is True
        assert result.truncate_span == 0  # Full re-index
        # Should have re-indexed content
        assert len(client.appended) > 0


class TestPrepareStreamingResync:
    """Tests for prepare_streaming_resync."""

    def test_cursor_reset_triggers_truncate(self) -> None:
        """When cursor is reset (last_synced_uuid=None) but span_end > 0, trigger truncate."""
        from memory_service.ingestion.claude.transcript_sync import (
            prepare_streaming_resync,
        )

        content = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hi"}]},
            },
        )

        # Simulate cursor reset: last_synced_uuid=None but span_end > 0
        result = prepare_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid=None,
            span_end=100,  # Had indexed data, then cursor was reset
        )

        # Should detect this as cursor reset and set needs_truncate=True
        assert result.needs_truncate is True
        assert result.truncate_span == 0  # Truncate to beginning

    def test_fresh_sync_no_truncate(self) -> None:
        """Fresh sync (span_end=0, last_synced_uuid=None) should not truncate."""
        from memory_service.ingestion.claude.transcript_sync import (
            prepare_streaming_resync,
        )

        content = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Hello"},
            },
        )

        # Fresh sync - no previous indexed data
        result = prepare_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid=None,
            span_end=0,
        )

        # Should NOT truncate - this is a fresh sync
        assert result.needs_truncate is False

    def test_compact_summaries_excluded_from_segments(self) -> None:
        """Compaction summaries (isCompactSummary=true) should not be indexed."""
        from memory_service.ingestion.claude.transcript_sync import (
            prepare_streaming_resync,
        )

        # Simulate a conversation with a compaction summary in the middle
        # The compact summary has parentUuid=None but bridges to earlier content
        content = _make_jsonl(
            # Original messages before compaction
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "First message"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "First response"}]},
            },
            # Compaction summary - should NOT be indexed
            {
                "uuid": "compact1",
                "parentUuid": "msg2",
                "type": "user",
                "isCompactSummary": True,
                "message": {
                    "content": "This is a compaction summary that should be skipped"
                },
            },
            # Messages after compaction (child of compact summary)
            {
                "uuid": "msg3",
                "parentUuid": "compact1",
                "type": "user",
                "message": {"content": "Message after compaction"},
            },
            {
                "uuid": "msg4",
                "parentUuid": "msg3",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Response after"}]},
            },
        )

        result = prepare_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid=None,
            span_end=0,
        )

        # Concatenate all segment texts
        full_text = "".join(result.segment_texts)

        # The compaction summary text should NOT appear
        assert "compaction summary that should be skipped" not in full_text

        # But the real messages should appear
        assert "First message" in full_text
        assert "First response" in full_text
        assert "Message after compaction" in full_text
        assert "Response after" in full_text

    def test_compaction_boundary_computed(self) -> None:
        """Compaction boundary is the span_end just before post-compaction content."""
        from memory_service.ingestion.claude.transcript_sync import (
            prepare_streaming_resync,
        )

        content = _make_jsonl(
            # Pre-compaction: msg1 -> msg2
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Pre-compaction user message"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Pre-compaction response"}]
                },
            },
            # Compaction summary (skipped but marks boundary)
            {
                "uuid": "compact1",
                "parentUuid": "msg2",
                "type": "user",
                "isCompactSummary": True,
                "message": {"content": "Compaction summary"},
            },
            # Post-compaction: msg3 -> msg4
            {
                "uuid": "msg3",
                "parentUuid": "compact1",
                "type": "user",
                "message": {"content": "Post-compaction user message"},
            },
            {
                "uuid": "msg4",
                "parentUuid": "msg3",
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Post-compaction response"}]
                },
            },
        )

        result = prepare_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid=None,
            span_end=0,
        )

        # compaction_span_end should be the cumulative length of pre-compaction segments
        assert result.compaction_span_end is not None

        # The boundary should equal the length of pre-compaction content
        pre_compaction_len = sum(
            len(s) for s in result.segment_texts if "Pre-compaction" in s
        )
        assert result.compaction_span_end == pre_compaction_len

    def test_compaction_boundary_none_without_compaction(self) -> None:
        """Compaction boundary is None when there's no compaction summary."""
        from memory_service.ingestion.claude.transcript_sync import (
            prepare_streaming_resync,
        )

        content = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "First message"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Response"}]},
            },
        )

        result = prepare_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid=None,
            span_end=0,
        )

        # No compaction, so boundary should be None
        assert result.compaction_span_end is None


class TestPrepareDeltaSync:
    """Tests for prepare_delta_sync."""

    def test_cursor_reset_triggers_revert(self) -> None:
        """When cursor is reset (last_synced_uuid=None) but span_end > 0, trigger revert."""
        delta = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hi"}]},
            },
        )

        # Cursor with last_synced_uuid=None but span_end > 0 (had indexed data, then reset)
        cursor = SessionCursor(byte_offset=0, last_synced_uuid=None, span_end=100)

        result = prepare_delta_sync(
            session_id="session1",
            delta=delta,
            cursor=cursor,
        )

        # Should detect this as a revert/reset and set truncated=True
        assert result.truncated is True
        assert result.truncate_span == 100  # Truncate from the existing span_end

    def test_fresh_sync_no_revert(self) -> None:
        """Fresh sync (span_end=0, last_synced_uuid=None) should not trigger revert."""
        delta = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Hello"},
            },
        )

        # Fresh cursor - no previous data
        cursor = SessionCursor(byte_offset=0, last_synced_uuid=None, span_end=0)

        result = prepare_delta_sync(
            session_id="session1",
            delta=delta,
            cursor=cursor,
        )

        # Should NOT be truncated - this is a fresh sync
        assert result.truncated is False

    def test_normal_continuation_no_revert(self) -> None:
        """Normal continuation (delta continues from last_synced_uuid) should not revert."""
        delta = _make_jsonl(
            {
                "uuid": "msg3",
                "parentUuid": "msg2",  # Continues from last_synced_uuid
                "type": "user",
                "message": {"content": "Continuing"},
            },
        )

        # Cursor showing we synced up to msg2
        cursor = SessionCursor(byte_offset=100, last_synced_uuid="msg2", span_end=50)

        result = prepare_delta_sync(
            session_id="session1",
            delta=delta,
            cursor=cursor,
        )

        # Should NOT be truncated - normal continuation
        assert result.truncated is False

    def test_actual_revert_triggers_truncate(self) -> None:
        """Revert (delta doesn't continue from last_synced_uuid) should trigger truncate."""
        delta = _make_jsonl(
            {
                "uuid": "msg3-alt",
                "parentUuid": "msg1",  # Branches from msg1, not msg2
                "type": "user",
                "message": {"content": "Different branch"},
            },
        )

        # Cursor showing we synced up to msg2
        cursor = SessionCursor(byte_offset=100, last_synced_uuid="msg2", span_end=50)

        result = prepare_delta_sync(
            session_id="session1",
            delta=delta,
            cursor=cursor,
        )

        # Should be truncated - this is a revert
        assert result.truncated is True
        assert result.truncate_span == 50


class TestGranularRevertTruncation:
    """Tests for granular revert truncation with append entries."""

    def test_revert_truncates_to_divergence_point_not_zero(self) -> None:
        """Revert should truncate to the span at common ancestor, not to 0.

        This tests the core regression: the server was truncating to 0 on every
        revert, but it should truncate to the span_end of the valid prefix entry
        in the append log.

        Setup:
        - Transcript contains BOTH branches (Claude Code keeps full history):
          msg1 -> msg2 -> msg3 (indexed branch)
                       -> msg3_alt (new branch after revert)
        - We indexed up to msg3 (span_end=300)
        - User reverted to msg2 and continued with msg3_alt
        - Common ancestor is msg2
        - Should truncate to 200 (msg2's span_end), not 0
        """
        from memory_service.ingestion.claude.transcript_sync import (
            prepare_streaming_resync,
        )

        # Transcript with BOTH branches (this is how Claude Code works -
        # reverts don't delete old messages, they just branch off)
        content = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "First message"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "First response"}]},
            },
            {
                "uuid": "msg3",
                "parentUuid": "msg2",
                "type": "user",
                "message": {"content": "Original third message (will be reverted)"},
            },
            {
                "uuid": "msg3_alt",
                "parentUuid": "msg2",
                "type": "user",
                "message": {"content": "New branch after revert"},
            },
        )

        # Append entries representing what we indexed before the revert:
        # msg1 at span_end=100, msg2 at span_end=200, msg3 at span_end=300
        append_entries = [
            ("msg1", 100),  # Segment ending at msg1 -> span_end=100
            ("msg2", 200),  # Segment ending at msg2 -> span_end=200
            ("msg3", 300),  # Segment ending at msg3 -> span_end=300 (now reverted)
        ]

        result = prepare_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid="msg3",  # We had synced up to msg3
            span_end=300,
            append_entries=append_entries,
        )

        # Should truncate to msg2's span_end (200), not 0
        assert result.needs_truncate is True
        assert result.truncate_span == 200, (
            f"Expected truncate_span=200 (msg2's span_end), got {result.truncate_span}. "
            "The algorithm should find msg2 as the valid prefix and truncate to its span_end."
        )

    def test_disjoint_branches_truncate_to_zero(self) -> None:
        """Disjoint branches (no common ancestor) should truncate to 0."""
        from memory_service.ingestion.claude.transcript_sync import (
            prepare_streaming_resync,
        )

        # Completely new transcript with no relation to indexed content
        content = _make_jsonl(
            {
                "uuid": "new_msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Completely new conversation"},
            },
        )

        append_entries = [
            ("old_msg1", 100),
            ("old_msg2", 200),
        ]

        result = prepare_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid="old_msg2",
            span_end=200,
            append_entries=append_entries,
        )

        # Disjoint branches - must truncate to 0
        assert result.needs_truncate is True
        assert result.truncate_span == 0

    def test_no_append_entries_falls_back_to_zero(self) -> None:
        """Without append entries, revert should fall back to truncating to 0."""
        from memory_service.ingestion.claude.transcript_sync import (
            prepare_streaming_resync,
        )

        content = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Message"},
            },
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Response"}]},
            },
            {
                "uuid": "msg3_alt",
                "parentUuid": "msg2",
                "type": "user",
                "message": {"content": "After revert"},
            },
        )

        # No append entries provided (legacy/migration case)
        result = prepare_streaming_resync(
            session_id="session1",
            jsonl_content=content,
            last_synced_uuid="msg3",
            span_end=300,
            append_entries=None,  # No entries
        )

        # Should still work, falling back to truncate to 0
        assert result.needs_truncate is True
        assert result.truncate_span == 0
