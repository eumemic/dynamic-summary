"""Tests for metadata-based sync functions (memory-efficient navigation)."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import pytest

from ragzoom_claude_code.transcript_sync import (
    RecordMeta,
    _build_metadata_and_parent_map,
    _extract_metadata,
    _handle_revert_detection_from_meta,
    _should_include_meta,
    build_ancestry_chain_from_meta,
    filter_to_steps_from_meta,
    find_entries_after_time_from_meta,
    find_truncation_point_from_meta,
    load_records_for_uuids,
)


class TestExtractMetadata:
    """Tests for _extract_metadata()."""

    def test_extracts_all_fields(self) -> None:
        """Extracts all metadata fields from a record."""
        record: dict[str, object] = {
            "uuid": "msg1",
            "parentUuid": "msg0",
            "timestamp": "2024-01-01T10:00:00Z",
            "type": "user",
            "isCompactSummary": False,
            "isMeta": False,
            "message": {"content": "Hello"},  # Should be ignored
        }

        meta = _extract_metadata(record)

        assert meta is not None
        assert meta.uuid == "msg1"
        assert meta.parent_uuid == "msg0"
        assert meta.timestamp == "2024-01-01T10:00:00Z"
        assert meta.record_type == "user"
        assert meta.is_compact_summary is False
        assert meta.is_meta is False

    def test_returns_none_without_uuid(self) -> None:
        """Returns None if record has no uuid."""
        record: dict[str, object] = {
            "type": "user",
            "timestamp": "2024-01-01T10:00:00Z",
        }
        assert _extract_metadata(record) is None

    def test_handles_missing_optional_fields(self) -> None:
        """Handles missing optional fields gracefully."""
        record: dict[str, object] = {"uuid": "msg1"}

        meta = _extract_metadata(record)

        assert meta is not None
        assert meta.uuid == "msg1"
        assert meta.parent_uuid is None
        assert meta.timestamp is None
        assert meta.record_type is None
        assert meta.is_compact_summary is False
        assert meta.is_meta is False

    def test_handles_invalid_parent_uuid_type(self) -> None:
        """Treats non-string parentUuid as None."""
        record: dict[str, object] = {"uuid": "msg1", "parentUuid": 123}

        meta = _extract_metadata(record)

        assert meta is not None
        assert meta.parent_uuid is None


class TestShouldIncludeMeta:
    """Tests for _should_include_meta()."""

    def test_includes_user_message(self) -> None:
        """Includes user messages."""
        meta = RecordMeta(
            uuid="msg1",
            parent_uuid=None,
            timestamp="2024-01-01T10:00:00Z",
            record_type="user",
            is_compact_summary=False,
            is_meta=False,
        )
        assert _should_include_meta(meta) is True

    def test_includes_assistant_message(self) -> None:
        """Includes assistant messages."""
        meta = RecordMeta(
            uuid="msg1",
            parent_uuid=None,
            timestamp="2024-01-01T10:00:00Z",
            record_type="assistant",
            is_compact_summary=False,
            is_meta=False,
        )
        assert _should_include_meta(meta) is True

    def test_excludes_queue_operation(self) -> None:
        """Excludes queue operations."""
        meta = RecordMeta(
            uuid="msg1",
            parent_uuid=None,
            timestamp="2024-01-01T10:00:00Z",
            record_type="queue-operation",
            is_compact_summary=False,
            is_meta=False,
        )
        assert _should_include_meta(meta) is False

    def test_excludes_compact_summary(self) -> None:
        """Excludes compaction summaries even if type is user/assistant."""
        meta = RecordMeta(
            uuid="msg1",
            parent_uuid=None,
            timestamp="2024-01-01T10:00:00Z",
            record_type="user",
            is_compact_summary=True,
            is_meta=False,
        )
        assert _should_include_meta(meta) is False

    def test_excludes_meta_records(self) -> None:
        """Excludes meta records (skill expansions, etc)."""
        meta = RecordMeta(
            uuid="msg1",
            parent_uuid=None,
            timestamp="2024-01-01T10:00:00Z",
            record_type="user",
            is_compact_summary=False,
            is_meta=True,
        )
        assert _should_include_meta(meta) is False


class TestBuildMetadataAndParentMap:
    """Tests for _build_metadata_and_parent_map()."""

    def _write_jsonl(self, records: list[dict[str, object]]) -> Path:
        """Write records to a temp JSONL file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for record in records:
            tmp.write(json.dumps(record) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_builds_metadata_map(self) -> None:
        """Builds metadata map from JSONL file."""
        path = self._write_jsonl(
            [
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "type": "user",
                },
                {
                    "uuid": "msg2",
                    "parentUuid": "msg1",
                    "timestamp": "2024-01-01T10:01:00Z",
                    "type": "assistant",
                },
            ]
        )

        try:
            metadata, parent_map = _build_metadata_and_parent_map(path)

            assert len(metadata) == 2
            assert metadata["msg1"].uuid == "msg1"
            assert metadata["msg2"].parent_uuid == "msg1"
            assert parent_map["msg1"] is None
            assert parent_map["msg2"] == "msg1"
        finally:
            path.unlink()

    def test_handles_compaction_bridging(self) -> None:
        """Bridges compaction boundaries in parent map."""
        # Simulate: session1 -> compaction -> session2
        path = self._write_jsonl(
            [
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "type": "user",
                },
                {
                    "uuid": "msg2",
                    "parentUuid": "msg1",
                    "timestamp": "2024-01-01T10:01:00Z",
                    "type": "assistant",
                },
                # After compaction, new message has parentUuid=None but is followed by compaction
                {
                    "uuid": "msg3",
                    "parentUuid": None,
                    "timestamp": "2024-01-01T11:00:00Z",
                    "type": "user",
                },
                {
                    "uuid": "compact",
                    "parentUuid": "msg3",
                    "timestamp": "2024-01-01T11:00:01Z",
                    "type": "user",
                    "isCompactSummary": True,
                },
                {
                    "uuid": "msg4",
                    "parentUuid": "compact",
                    "timestamp": "2024-01-01T11:01:00Z",
                    "type": "assistant",
                },
            ]
        )

        try:
            metadata, parent_map = _build_metadata_and_parent_map(path)

            # msg3 should be bridged to msg2 (last regular before compaction)
            assert parent_map["msg3"] == "msg2"
        finally:
            path.unlink()


class TestLoadRecordsForUuids:
    """Tests for load_records_for_uuids()."""

    def _write_jsonl(self, records: list[dict[str, object]]) -> Path:
        """Write records to a temp JSONL file."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for record in records:
            tmp.write(json.dumps(record) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_loads_only_requested_uuids(self) -> None:
        """Only loads records for requested UUIDs."""
        path = self._write_jsonl(
            [
                {"uuid": "msg1", "message": {"content": "Hello"}},
                {"uuid": "msg2", "message": {"content": "World"}},
                {"uuid": "msg3", "message": {"content": "!"}},
            ]
        )

        try:
            records = load_records_for_uuids(path, {"msg1", "msg3"})

            assert len(records) == 2
            assert "msg1" in records
            assert "msg3" in records
            assert "msg2" not in records
        finally:
            path.unlink()

    def test_stops_early_when_all_found(self) -> None:
        """Stops scanning when all requested UUIDs are found."""
        # This is hard to test directly, but we can verify it works correctly
        path = self._write_jsonl(
            [
                {"uuid": "msg1", "message": {"content": "Hello"}},
                {"uuid": "msg2", "message": {"content": "World"}},
            ]
        )

        try:
            records = load_records_for_uuids(path, {"msg1"})
            assert len(records) == 1
            assert records["msg1"]["message"] == {"content": "Hello"}
        finally:
            path.unlink()

    def test_empty_set_returns_empty_dict(self) -> None:
        """Returns empty dict for empty UUID set."""
        path = self._write_jsonl([{"uuid": "msg1"}])

        try:
            records = load_records_for_uuids(path, set())
            assert records == {}
        finally:
            path.unlink()


class TestFindTruncationPointFromMeta:
    """Tests for find_truncation_point_from_meta()."""

    def _make_metadata(
        self, chain: list[tuple[str, str | None, str]]
    ) -> dict[str, RecordMeta]:
        """Build metadata dict from chain spec."""
        metadata: dict[str, RecordMeta] = {}
        for uuid, parent, ts in chain:
            metadata[uuid] = RecordMeta(
                uuid=uuid,
                parent_uuid=parent,
                timestamp=ts,
                record_type="user",
                is_compact_summary=False,
                is_meta=False,
            )
        return metadata

    def test_first_sync_returns_none_head(self) -> None:
        """First sync (indexed_time_end=None) returns (None, head_uuid)."""
        metadata = self._make_metadata(
            [
                ("msg1", None, "2024-01-01T10:00:00Z"),
                ("msg2", "msg1", "2024-01-01T10:01:00Z"),
            ]
        )

        r, s = find_truncation_point_from_meta("msg2", metadata, indexed_time_end=None)

        assert r is None
        assert s == "msg2"

    def test_normal_append(self) -> None:
        """Normal append finds connection point."""
        metadata = self._make_metadata(
            [
                ("msg1", None, "2024-01-01T10:00:00Z"),
                ("msg2", "msg1", "2024-01-01T10:01:00Z"),
                ("msg3", "msg2", "2024-01-01T10:02:00Z"),
            ]
        )

        indexed_time_end = datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point_from_meta("msg3", metadata, indexed_time_end)

        assert r == "msg2"
        assert s == "msg3"


class TestFilterToStepsFromMeta:
    """Tests for filter_to_steps_from_meta()."""

    def test_filters_to_user_assistant_only(self) -> None:
        """Filters to only user and assistant messages."""
        metadata: dict[str, RecordMeta] = {
            "msg1": RecordMeta(
                "msg1", None, "2024-01-01T10:00:00Z", "user", False, False
            ),
            "msg2": RecordMeta(
                "msg2", "msg1", "2024-01-01T10:01:00Z", "assistant", False, False
            ),
            "msg3": RecordMeta(
                "msg3", "msg2", "2024-01-01T10:02:00Z", "queue-operation", False, False
            ),
            "msg4": RecordMeta(
                "msg4", "msg3", "2024-01-01T10:03:00Z", "user", False, False
            ),
        }

        steps = filter_to_steps_from_meta(["msg1", "msg2", "msg3", "msg4"], metadata)

        assert len(steps) == 3
        assert [s.uuid for s in steps] == ["msg1", "msg2", "msg4"]

    def test_skips_records_without_timestamp(self) -> None:
        """Skips records that have no timestamp."""
        metadata: dict[str, RecordMeta] = {
            "msg1": RecordMeta(
                "msg1", None, "2024-01-01T10:00:00Z", "user", False, False
            ),
            "msg2": RecordMeta(
                "msg2", "msg1", None, "user", False, False
            ),  # No timestamp
        }

        steps = filter_to_steps_from_meta(["msg1", "msg2"], metadata)

        assert len(steps) == 1
        assert steps[0].uuid == "msg1"


class TestBuildAncestryChainFromMeta:
    """Tests for build_ancestry_chain_from_meta()."""

    def test_builds_chain_in_chronological_order(self) -> None:
        """Builds ancestry chain in oldest-first order."""
        metadata: dict[str, RecordMeta] = {
            "msg1": RecordMeta(
                "msg1", None, "2024-01-01T10:00:00Z", "user", False, False
            ),
            "msg2": RecordMeta(
                "msg2", "msg1", "2024-01-01T10:01:00Z", "assistant", False, False
            ),
            "msg3": RecordMeta(
                "msg3", "msg2", "2024-01-01T10:02:00Z", "user", False, False
            ),
        }
        parent_map = {"msg1": None, "msg2": "msg1", "msg3": "msg2"}

        chain = build_ancestry_chain_from_meta("msg3", None, metadata, parent_map)

        assert chain == ["msg1", "msg2", "msg3"]

    def test_stops_at_stop_uuid(self) -> None:
        """Stops at stop_uuid (exclusive)."""
        metadata: dict[str, RecordMeta] = {
            "msg1": RecordMeta(
                "msg1", None, "2024-01-01T10:00:00Z", "user", False, False
            ),
            "msg2": RecordMeta(
                "msg2", "msg1", "2024-01-01T10:01:00Z", "assistant", False, False
            ),
            "msg3": RecordMeta(
                "msg3", "msg2", "2024-01-01T10:02:00Z", "user", False, False
            ),
        }
        parent_map = {"msg1": None, "msg2": "msg1", "msg3": "msg2"}

        chain = build_ancestry_chain_from_meta("msg3", "msg1", metadata, parent_map)

        assert chain == ["msg2", "msg3"]


class TestFindEntriesAfterTimeFromMeta:
    """Tests for find_entries_after_time_from_meta()."""

    def test_finds_entries_after_cutoff(self) -> None:
        """Finds entries with timestamp > cutoff_time."""
        metadata: dict[str, RecordMeta] = {
            "msg1": RecordMeta(
                "msg1", None, "2024-01-01T10:00:00Z", "user", False, False
            ),
            "msg2": RecordMeta(
                "msg2", "msg1", "2024-01-01T10:01:00Z", "assistant", False, False
            ),
            "msg3": RecordMeta(
                "msg3", "msg2", "2024-01-01T10:02:00Z", "user", False, False
            ),
        }
        parent_map = {"msg1": None, "msg2": "msg1", "msg3": "msg2"}

        cutoff = datetime(2024, 1, 1, 10, 0, 30, tzinfo=timezone.utc)
        entries = find_entries_after_time_from_meta(
            "msg3", metadata, parent_map, cutoff
        )

        # msg1 is at 10:00:00, cutoff is 10:00:30, so only msg2 and msg3
        assert entries == ["msg2", "msg3"]


# =============================================================================
# BUG: Full revert not handled correctly
# =============================================================================


class MockSyncClient:
    """Mock client for testing sync operations."""

    def __init__(self) -> None:
        self.truncate_calls: list[tuple[str, str]] = []

    def truncate_from_time(self, document_id: str, cutoff_time: str) -> None:
        self.truncate_calls.append((document_id, cutoff_time))


class TestFullRevertBug:
    """Tests for the full revert bug.

    When a user reverts to before any indexed content:
    - r_uuid = None (no connection point found)
    - indexed_time_end is not None (we have indexed content)

    Currently, _handle_revert_detection_from_meta returns None (no truncation),
    but it SHOULD truncate everything and re-index.
    """

    
    def test_full_revert_should_truncate_everything(self) -> None:
        """Full revert should truncate all indexed content.

        Scenario:
        - Document has indexed content up to time T
        - User reverts to before the first indexed message
        - All new messages have timestamps > T but no chain connects to indexed content
        - Expected: truncate everything, re-index from scratch
        - Actual: no truncation occurs (bug!)
        """
        metadata: dict[str, RecordMeta] = {
            "new1": RecordMeta(
                "new1", None, "2024-01-02T10:00:00Z", "user", False, False
            ),
            "new2": RecordMeta(
                "new2", "new1", "2024-01-02T10:01:00Z", "assistant", False, False
            ),
        }

        # Indexed up to Jan 1st, but new messages start fresh on Jan 2nd
        # with no parent chain connecting to the old content
        indexed_time_end = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        client = MockSyncClient()

        # r_uuid is None because we walked the entire new chain without
        # finding any record with timestamp <= indexed_time_end
        result = _handle_revert_detection_from_meta(
            r_uuid=None,  # No connection point found
            indexed_time_end=indexed_time_end,
            metadata=metadata,
            client=client,
            document_id="test-doc",
        )

        # BUG: Currently returns None (no truncation)
        # EXPECTED: Should truncate everything
        assert result is not None, "Full revert should trigger truncation"
        assert len(client.truncate_calls) == 1, "Should have called truncate_from_time"

    def test_first_sync_does_not_truncate(self) -> None:
        """First sync (indexed_time_end=None) should not truncate.

        This is the correct behavior - just verifying it still works.
        """
        metadata: dict[str, RecordMeta] = {
            "msg1": RecordMeta(
                "msg1", None, "2024-01-01T10:00:00Z", "user", False, False
            ),
        }

        client = MockSyncClient()

        result = _handle_revert_detection_from_meta(
            r_uuid=None,
            indexed_time_end=None,  # First sync
            metadata=metadata,
            client=client,
            document_id="test-doc",
        )

        assert result is None
        assert len(client.truncate_calls) == 0
