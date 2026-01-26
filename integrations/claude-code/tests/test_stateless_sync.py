"""Tests for stateless transcript sync functions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ragzoom_claude_code.transcript_sync import (
    build_ancestry_chain,
    execute_sync,
    find_truncation_point,
    is_user_message,
)

from ragzoom.wrapper import AppendUnit


class TestIsUserMessage:
    """Tests for is_user_message() helper function."""

    def test_returns_true_for_user_type(self) -> None:
        """User messages with type='user' and no toolUseResult return True."""
        record = {
            "uuid": "msg1",
            "type": "user",
            "message": {"content": "Hello world"},
        }

        assert is_user_message(record) is True

    def test_returns_false_for_assistant_type(self) -> None:
        """Assistant messages return False."""
        record = {
            "uuid": "msg1",
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Response"}]},
        }

        assert is_user_message(record) is False

    def test_returns_false_for_tool_result(self) -> None:
        """User messages with toolUseResult return False."""
        record = {
            "uuid": "msg1",
            "type": "user",
            "toolUseResult": {"result": "tool output"},
            "message": {"content": "Tool result"},
        }

        assert is_user_message(record) is False

    def test_returns_false_for_command_output(self) -> None:
        """User messages with command output marker return False."""
        record = {
            "uuid": "msg1",
            "type": "user",
            "message": {
                "content": "<local-command-stdout>output here</local-command-stdout>"
            },
        }

        assert is_user_message(record) is False

    def test_returns_true_for_normal_user_with_dict_content(self) -> None:
        """Normal user messages with dict content return True."""
        record = {
            "uuid": "msg1",
            "type": "user",
            "message": {"content": {"text": "Hello", "type": "text"}},
        }

        assert is_user_message(record) is True


class TestFindTruncationPoint:
    """Tests for find_truncation_point() stateless sync algorithm."""

    def _make_records(
        self, chain: list[tuple[str, str | None, str, str]]
    ) -> dict[str, dict[str, object]]:
        """Build records dict from chain spec.

        Args:
            chain: List of (uuid, parent_uuid, timestamp, type) tuples

        Returns:
            UUID -> record mapping
        """
        records: dict[str, dict[str, object]] = {}
        for uuid, parent, ts, msg_type in chain:
            record: dict[str, object] = {
                "uuid": uuid,
                "parentUuid": parent,
                "timestamp": ts,
                "type": msg_type,
            }
            if msg_type == "user":
                record["message"] = {"content": f"User message {uuid}"}
            else:
                record["message"] = {
                    "content": [{"type": "text", "text": f"Response {uuid}"}]
                }
            records[uuid] = record
        return records

    def test_first_sync_returns_none_head(self) -> None:
        """First sync (indexed_time_end=None) returns (None, head_uuid)."""
        # Chain: msg1 -> msg2 -> msg3
        records = self._make_records(
            [
                ("msg1", None, "2024-01-01T10:00:00Z", "user"),
                ("msg2", "msg1", "2024-01-01T10:01:00Z", "assistant"),
                ("msg3", "msg2", "2024-01-01T10:02:00Z", "user"),
            ]
        )

        r, s = find_truncation_point("msg3", records, indexed_time_end=None)

        assert r is None
        assert s == "msg3"

    def test_normal_append_same_time(self) -> None:
        """Normal append: R.timestamp == indexed_time_end."""
        # Chain: msg1(user) -> msg2(asst) -> msg3(user) -> msg4(asst)
        # Indexed up to msg2's timestamp, should connect at msg2
        records = self._make_records(
            [
                ("msg1", None, "2024-01-01T10:00:00Z", "user"),
                ("msg2", "msg1", "2024-01-01T10:01:00Z", "assistant"),
                ("msg3", "msg2", "2024-01-01T10:02:00Z", "user"),
                ("msg4", "msg3", "2024-01-01T10:03:00Z", "assistant"),
            ]
        )

        indexed_time_end = datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg4", records, indexed_time_end)

        # Should connect at msg2 (last in indexed range) with S=msg3 (turn boundary)
        assert r == "msg2"
        assert s == "msg3"

    def test_slides_past_assistant_to_find_turn_boundary(self) -> None:
        """Sliding window continues past assistant messages to find turn boundary."""
        # Chain: msg1(user) -> msg2(asst) -> msg3(user) -> msg4(asst) -> msg5(asst)
        # When indexed_time_end is at msg4, S=msg5 (assistant), so we slide
        # to find msg3 (user) as the turn boundary
        records = self._make_records(
            [
                ("msg1", None, "2024-01-01T10:00:00Z", "user"),
                ("msg2", "msg1", "2024-01-01T10:01:00Z", "assistant"),
                ("msg3", "msg2", "2024-01-01T10:02:00Z", "user"),
                ("msg4", "msg3", "2024-01-01T10:03:00Z", "assistant"),
                ("msg5", "msg4", "2024-01-01T10:04:00Z", "assistant"),  # continuation
            ]
        )

        # Indexed up to 10:03 (includes msg4)
        indexed_time_end = datetime(2024, 1, 1, 10, 3, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg5", records, indexed_time_end)

        # Walking: msg5->msg4->msg3->msg2->msg1
        # r=msg5, s=None: msg5.ts=10:04 > 10:03, slide
        # r=msg4, s=msg5: msg4.ts=10:03 <= 10:03, s=msg5 is assistant (not boundary), slide
        # r=msg3, s=msg4: msg3.ts=10:02 <= 10:03, s=msg4 is assistant (not boundary), slide
        # r=msg2, s=msg3: msg2.ts=10:01 <= 10:03, s=msg3 is user (turn boundary!), stop
        assert r == "msg2"
        assert s == "msg3"

    def test_revert_to_turn_boundary(self) -> None:
        """Revert case: R.timestamp < indexed_time_end indicates orphaned content."""
        # Original: msg1(user) -> msg2(asst) -> msg3(user) -> msg4(asst)
        # User reverted to msg2 and continued differently
        # New branch: msg1 -> msg2 -> msg5(user) -> msg6(asst)
        records = self._make_records(
            [
                ("msg1", None, "2024-01-01T10:00:00Z", "user"),
                ("msg2", "msg1", "2024-01-01T10:01:00Z", "assistant"),
                ("msg5", "msg2", "2024-01-01T10:05:00Z", "user"),  # New branch
                ("msg6", "msg5", "2024-01-01T10:06:00Z", "assistant"),
            ]
        )

        # Indexed up to 10:04 (includes msg3, msg4 from old branch - now orphaned)
        indexed_time_end = datetime(2024, 1, 1, 10, 4, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg6", records, indexed_time_end)

        # Walking: msg6->msg5->msg2->msg1
        # msg6.ts=10:06 > 10:04, slide
        # msg5.ts=10:05 > 10:04, slide
        # msg2.ts=10:01 <= 10:04, S=msg5 is user (turn boundary), stop
        assert r == "msg2"
        assert s == "msg5"

    def test_revert_mid_turn_rounds_down(self) -> None:
        """Mid-turn revert continues to turn boundary."""
        # Turn 1: msg1(user) -> msg2(asst)
        # Turn 2: msg3(user) -> msg4(asst) -> msg5(asst) [assistant continuation]
        # Revert happens within turn 2, creates new branch from msg3
        # New: msg1 -> msg2 -> msg3 -> msg7(asst) [different response]
        records = self._make_records(
            [
                ("msg1", None, "2024-01-01T10:00:00Z", "user"),
                ("msg2", "msg1", "2024-01-01T10:01:00Z", "assistant"),
                ("msg3", "msg2", "2024-01-01T10:02:00Z", "user"),
                ("msg7", "msg3", "2024-01-01T10:07:00Z", "assistant"),  # New response
            ]
        )

        # Indexed up to 10:05 (would have included msg4, msg5 in old turn 2)
        indexed_time_end = datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg7", records, indexed_time_end)

        # Walking: msg7->msg3->msg2->msg1
        # msg7.ts=10:07 > 10:05, slide
        # msg3.ts=10:02 <= 10:05, S=msg7 is assistant (not boundary), slide
        # msg2.ts=10:01 <= 10:05, S=msg3 is user (turn boundary), stop
        # So we round down to msg2, returning msg3 as start of new content
        assert r == "msg2"
        assert s == "msg3"

    def test_complete_reindex_no_common_ancestor(self) -> None:
        """Complete reindex when entire chain is newer than indexed content."""
        # All records are newer than indexed content
        records = self._make_records(
            [
                ("msg1", None, "2024-01-01T12:00:00Z", "user"),
                ("msg2", "msg1", "2024-01-01T12:01:00Z", "assistant"),
            ]
        )

        # Indexed up to 10:00, but all records are at 12:00+
        indexed_time_end = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg2", records, indexed_time_end)

        # Walking: msg2->msg1->None
        # msg2.ts=12:01 > 10:00, slide
        # msg1.ts=12:00 > 10:00, slide
        # r=None, return (None, head_uuid)
        assert r is None
        assert s == "msg2"

    def test_handles_missing_timestamp(self) -> None:
        """Records without timestamps are skipped in comparison."""
        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "parentUuid": None,
                "timestamp": "2024-01-01T10:00:00Z",
                "type": "user",
                "message": {"content": "Hello"},
            },
            "msg2": {
                "uuid": "msg2",
                "parentUuid": "msg1",
                # No timestamp
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Response"}]},
            },
            "msg3": {
                "uuid": "msg3",
                "parentUuid": "msg2",
                "timestamp": "2024-01-01T10:02:00Z",
                "type": "user",
                "message": {"content": "Follow up"},
            },
        }

        indexed_time_end = datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg3", records, indexed_time_end)

        # Walking: msg3->msg2->msg1
        # msg3.ts=10:02 > 10:01, slide
        # msg2 has no timestamp, slide (s=msg3)
        # msg1.ts=10:00 <= 10:01, S=msg2 is assistant (not boundary)
        # Actually we need to check: the S at this point is msg2, not msg3
        # Let me trace again:
        # r=msg3, s=None: ts=10:02 > 10:01, slide -> r=msg2, s=msg3
        # r=msg2, s=msg3: no timestamp, slide -> r=msg1, s=msg2
        # r=msg1, s=msg2: ts=10:00 <= 10:01, is s=msg2 user? No (assistant)
        #   slide -> r=None, s=msg1
        # r=None, return (None, head_uuid="msg3")
        assert r is None
        assert s == "msg3"

    def test_head_already_indexed(self) -> None:
        """When head is already indexed, S is None (nothing to append)."""
        records = self._make_records(
            [
                ("msg1", None, "2024-01-01T10:00:00Z", "user"),
                ("msg2", "msg1", "2024-01-01T10:01:00Z", "assistant"),
            ]
        )

        # Indexed includes msg2
        indexed_time_end = datetime(2024, 1, 1, 10, 1, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg2", records, indexed_time_end)

        # Walking: msg2
        # r=msg2, s=None: ts=10:01 <= 10:01, s=None (valid boundary), stop
        assert r == "msg2"
        assert s is None

    def test_single_user_message_first_sync(self) -> None:
        """First sync with single user message."""
        records = self._make_records(
            [
                ("msg1", None, "2024-01-01T10:00:00Z", "user"),
            ]
        )

        r, s = find_truncation_point("msg1", records, indexed_time_end=None)

        assert r is None
        assert s == "msg1"

    def test_timezone_handling(self) -> None:
        """Timestamps with different timezone formats are handled."""
        # Test with +00:00 format
        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "parentUuid": None,
                "timestamp": "2024-01-01T10:00:00+00:00",  # +00:00 format
                "type": "user",
                "message": {"content": "Hello"},
            },
            "msg2": {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "timestamp": "2024-01-01T10:01:00Z",  # Z format
                "type": "user",
                "message": {"content": "Follow up"},
            },
        }

        indexed_time_end = datetime(2024, 1, 1, 10, 0, 30, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg2", records, indexed_time_end)

        # msg2.ts=10:01 > 10:00:30, slide
        # msg1.ts=10:00 <= 10:00:30, s=msg2 is user (boundary), stop
        assert r == "msg1"
        assert s == "msg2"

    def test_tool_result_not_turn_boundary(self) -> None:
        """Tool results are not turn boundaries even though they're user type."""
        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "parentUuid": None,
                "timestamp": "2024-01-01T10:00:00Z",
                "type": "user",
                "message": {"content": "Hello"},
            },
            "msg2": {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "timestamp": "2024-01-01T10:01:00Z",
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": "read"}]},
            },
            "msg3": {
                "uuid": "msg3",
                "parentUuid": "msg2",
                "timestamp": "2024-01-01T10:02:00Z",
                "type": "user",
                "toolUseResult": {"result": "file content"},  # Tool result
                "message": {"content": "Tool result"},
            },
            "msg4": {
                "uuid": "msg4",
                "parentUuid": "msg3",
                "timestamp": "2024-01-01T10:03:00Z",
                "type": "user",
                "message": {"content": "Real user message"},
            },
        }

        indexed_time_end = datetime(2024, 1, 1, 10, 2, 0, tzinfo=timezone.utc)
        r, s = find_truncation_point("msg4", records, indexed_time_end)

        # Walking: msg4->msg3->msg2->msg1
        # msg4.ts=10:03 > 10:02, slide
        # msg3.ts=10:02 <= 10:02, s=msg4 is user (real), boundary, stop
        assert r == "msg3"
        assert s == "msg4"


class TestBuildAncestryChain:
    """Tests for build_ancestry_chain() function."""

    def _make_records(
        self, chain: list[tuple[str, str | None]]
    ) -> dict[str, dict[str, object]]:
        """Build records dict from chain spec.

        Args:
            chain: List of (uuid, parent_uuid) tuples

        Returns:
            UUID -> record mapping
        """
        records: dict[str, dict[str, object]] = {}
        for uuid, parent in chain:
            records[uuid] = {
                "uuid": uuid,
                "parentUuid": parent,
            }
        return records

    def test_build_ancestry_chain_normal(self) -> None:
        """Builds chain from stop_uuid to head_uuid in chronological order."""
        # Chain: msg1 -> msg2 -> msg3 -> msg4
        records = self._make_records(
            [
                ("msg1", None),
                ("msg2", "msg1"),
                ("msg3", "msg2"),
                ("msg4", "msg3"),
            ]
        )

        # Get chain from msg1 to msg4 (excluding msg1)
        result = build_ancestry_chain("msg4", "msg1", records)

        # Should return [msg2, msg3, msg4] in chronological order
        assert result == ["msg2", "msg3", "msg4"]

    def test_build_ancestry_chain_from_root(self) -> None:
        """Builds entire chain when stop_uuid is None."""
        # Chain: msg1 -> msg2 -> msg3
        records = self._make_records(
            [
                ("msg1", None),
                ("msg2", "msg1"),
                ("msg3", "msg2"),
            ]
        )

        # Get entire chain from root (stop_uuid=None)
        result = build_ancestry_chain("msg3", None, records)

        # Should return [msg1, msg2, msg3] in chronological order
        assert result == ["msg1", "msg2", "msg3"]

    def test_build_ancestry_chain_adjacent(self) -> None:
        """Builds chain when stop_uuid is immediate parent of head."""
        # Chain: msg1 -> msg2
        records = self._make_records(
            [
                ("msg1", None),
                ("msg2", "msg1"),
            ]
        )

        # Get chain from msg1 to msg2 (excluding msg1)
        result = build_ancestry_chain("msg2", "msg1", records)

        # Should return [msg2]
        assert result == ["msg2"]

    def test_build_ancestry_chain_empty_when_same(self) -> None:
        """Returns empty list when stop_uuid equals head_uuid."""
        records = self._make_records(
            [
                ("msg1", None),
                ("msg2", "msg1"),
            ]
        )

        # Get chain from msg2 to msg2
        result = build_ancestry_chain("msg2", "msg2", records)

        # Should return empty list
        assert result == []

    def test_build_ancestry_chain_single_root(self) -> None:
        """Handles single message at root."""
        records = self._make_records(
            [
                ("msg1", None),
            ]
        )

        # Get entire chain
        result = build_ancestry_chain("msg1", None, records)

        # Should return [msg1]
        assert result == ["msg1"]

    def test_build_ancestry_chain_missing_parent_stops(self) -> None:
        """Stops at missing parent gracefully."""
        # Chain with missing intermediate record
        records: dict[str, dict[str, object]] = {
            "msg1": {"uuid": "msg1", "parentUuid": None},
            # msg2 is missing
            "msg3": {"uuid": "msg3", "parentUuid": "msg2"},  # Points to missing
        }

        # Get chain from msg1 to msg3 - but msg2 is missing
        # Should stop when it can't find msg2
        result = build_ancestry_chain("msg3", "msg1", records)

        # Should return [msg3] since it can't trace further back
        assert result == ["msg3"]

    def test_build_ancestry_chain_stop_not_ancestor(self) -> None:
        """Returns chain to root when stop_uuid is not an ancestor."""
        # Chain: msg1 -> msg2 -> msg3
        # msg99 is not in the ancestry chain
        records = self._make_records(
            [
                ("msg1", None),
                ("msg2", "msg1"),
                ("msg3", "msg2"),
                ("msg99", None),  # Separate root
            ]
        )

        # Try to get chain stopping at msg99 (not an ancestor of msg3)
        result = build_ancestry_chain("msg3", "msg99", records)

        # Should return entire chain since stop_uuid was never found
        assert result == ["msg1", "msg2", "msg3"]


# --- Helper classes and functions for execute_sync tests ---


def make_user_message(
    uuid: str,
    parent_uuid: str | None,
    timestamp: str,
    content: str,
) -> dict[str, object]:
    """Create a user transcript message record."""
    return {
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "type": "user",
        "timestamp": timestamp,
        "message": {"content": content},
    }


def make_assistant_message(
    uuid: str,
    parent_uuid: str | None,
    timestamp: str,
    content: str,
) -> dict[str, object]:
    """Create an assistant transcript message record."""
    return {
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "type": "assistant",
        "timestamp": timestamp,
        "message": {"content": [{"type": "text", "text": content}]},
    }


@dataclass
class MockDocumentStatus:
    """Mock document status for testing stateless sync."""

    document_id: str
    exists: bool = False
    is_temporal: bool = True
    leaf_count: int = 0
    node_count: int = 0
    complete_forest_size: int = 0
    completion_pct: float = 0.0
    time_start: str | None = None
    time_end: str | None = None


@dataclass
class BatchAppendResult:
    """Result type compatible with execute_sync expectations."""

    span_start: int
    span_end: int


@dataclass
class TruncateFromTimeResult:
    """Mock result from time-based truncation."""

    document_id: str
    deleted_node_ids: list[str]
    cutoff_time: str


@dataclass
class StatelessMockClient:
    """Mock client that tracks calls for testing stateless sync.

    This client provides get_document_status() for querying indexed state
    and truncate_from_time() for temporal truncation - the key APIs for
    stateless sync.
    """

    # Call tracking
    get_document_status_calls: list[str] = field(default_factory=list)
    truncate_from_time_calls: list[tuple[str, str]] = field(default_factory=list)
    batch_append_calls: list[tuple[str, list[AppendUnit]]] = field(default_factory=list)
    truncate_calls: list[tuple[str, int]] = field(default_factory=list)

    # Configurable return values
    _document_status: MockDocumentStatus | None = None
    _span_counter: int = field(default=0)

    def get_document_status(self, document_id: str) -> MockDocumentStatus:
        """Return document status for stateless sync."""
        self.get_document_status_calls.append(document_id)
        if self._document_status is not None:
            return self._document_status
        # Default: non-existent document
        return MockDocumentStatus(document_id=document_id, exists=False)

    def truncate_from_time(
        self, document_id: str, cutoff_time: str
    ) -> TruncateFromTimeResult:
        """Track time-based truncation calls."""
        self.truncate_from_time_calls.append((document_id, cutoff_time))
        return TruncateFromTimeResult(
            document_id=document_id,
            deleted_node_ids=[],
            cutoff_time=cutoff_time,
        )

    def batch_append(
        self,
        document_id: str,
        units: list[AppendUnit],
    ) -> BatchAppendResult:
        """Track batch append calls."""
        self.batch_append_calls.append((document_id, units))
        for unit in units:
            self._span_counter += len(unit.text)
        return BatchAppendResult(span_start=0, span_end=self._span_counter)

    def truncate(self, document_id: str, span_start: int) -> None:
        """Track span-based truncate calls (should not be used in stateless)."""
        self.truncate_calls.append((document_id, span_start))


class TestExecuteSyncStateless:
    """Tests for execute_sync using stateless document status approach."""

    def test_execute_sync_calls_get_document_status(self, tmp_path: Path) -> None:
        """execute_sync should call client.get_document_status() to get indexed state."""
        client = StatelessMockClient()

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # Create transcript with one turn
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        make_user_message("msg1", None, "2024-01-21T14:30:00Z", "Hello")
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg2", "msg1", "2024-01-21T14:30:05Z", "Hi!"
                        )
                    ),
                ]
            )
            + "\n"
        )

        execute_sync(transcript_path, state_path, client)

        # Should have called get_document_status to determine indexed state
        assert len(client.get_document_status_calls) == 1
        assert client.get_document_status_calls[0] == "transcript"
