"""Tests for execute_sync_from_bytes server-side sync logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from memory_service.ingestion.claude.transcript_sync import (
    _build_parent_map_from_bytes,
    _build_records_map_from_bytes,
    _get_current_head_from_bytes,
    _iter_jsonl_from_bytes,
    execute_sync_from_bytes,
)


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


class TestExecuteSyncFromBytes:
    """Tests for execute_sync_from_bytes."""

    def test_empty_content_returns_empty_result(self) -> None:
        client = MockRagZoomClient()

        result = execute_sync_from_bytes(
            session_id="session1",
            jsonl_content=b"",
            previous_byte_offset=0,
            client=client,
        )

        assert result.document_id == "session1"
        assert result.appended_uuids == []
        assert result.truncated is False
        assert client.appended == []

    def test_syncs_new_messages(self) -> None:
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
                "message": {"content": "Hi there"},
            },
        )
        client = MockRagZoomClient()

        result = execute_sync_from_bytes(
            session_id="session1",
            jsonl_content=content,
            previous_byte_offset=0,
            client=client,
        )

        assert result.document_id == "session1"
        assert len(result.appended_uuids) > 0
        assert result.truncated is False
        assert len(client.appended) > 0

    def test_incremental_sync_only_new_messages(self) -> None:
        """When previous_byte_offset > 0, should only sync new messages."""
        first_msg = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Hello"},
            },
        )
        full_content = first_msg + _make_jsonl(
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": {"content": "Hi"},
            },
        )
        client = MockRagZoomClient()

        # Simulate first sync already done
        execute_sync_from_bytes(
            session_id="session1",
            jsonl_content=first_msg,
            previous_byte_offset=0,
            client=client,
        )
        first_append_count = len(client.appended)

        # Now sync with new content
        result = execute_sync_from_bytes(
            session_id="session1",
            jsonl_content=full_content,
            previous_byte_offset=len(first_msg),
            client=client,
        )

        # Should have appended more content
        assert len(client.appended) > first_append_count
        assert "msg2" in result.appended_uuids

    def test_nothing_to_sync_when_already_current(self) -> None:
        """When already synced to current head, should do nothing."""
        content = _make_jsonl(
            {
                "uuid": "msg1",
                "parentUuid": None,
                "type": "user",
                "message": {"content": "Hello"},
            },
        )
        client = MockRagZoomClient()

        # First sync
        execute_sync_from_bytes(
            session_id="session1",
            jsonl_content=content,
            previous_byte_offset=0,
            client=client,
        )
        first_append_count = len(client.appended)

        # Second sync with same content and offset at end
        result = execute_sync_from_bytes(
            session_id="session1",
            jsonl_content=content,
            previous_byte_offset=len(content),
            client=client,
        )

        # Should not have appended anything new
        assert len(client.appended) == first_append_count
        assert result.appended_uuids == []
