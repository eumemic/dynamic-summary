"""Tests for transcript sync with revert detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memory_service.ingestion.claude.transcript_sync import (
    AppendEntry,
    AppendLog,
)


class TestStreamingParentMap:
    """Tests for streaming parent map building."""

    def test_builds_map_from_linear_transcript(self, tmp_path: Path) -> None:
        """Should map each uuid to its parent."""
        from memory_service.ingestion.claude.transcript_sync import (
            stream_find_common_ancestor_and_records,
        )

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2"}),
                ]
            )
            + "\n"
        )

        result = stream_find_common_ancestor_and_records(jsonl, "msg3", None)

        assert result.parent_of["msg1"] is None
        assert result.parent_of["msg2"] == "msg1"
        assert result.parent_of["msg3"] == "msg2"

    def test_builds_map_from_branched_transcript(self, tmp_path: Path) -> None:
        """Should include all branches in the map."""
        from memory_service.ingestion.claude.transcript_sync import (
            stream_find_common_ancestor_and_records,
        )

        # msg1 -> msg2 -> msg3 -> msg4
        #              \-> msg3' -> msg4'
        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2"}),
                    json.dumps({"uuid": "msg4", "parentUuid": "msg3"}),
                    # User reverted to msg2 and continued
                    json.dumps({"uuid": "msg3-alt", "parentUuid": "msg2"}),
                    json.dumps({"uuid": "msg4-alt", "parentUuid": "msg3-alt"}),
                ]
            )
            + "\n"
        )

        result = stream_find_common_ancestor_and_records(jsonl, "msg4-alt", None)

        assert result.parent_of["msg3-alt"] == "msg2"
        assert result.parent_of["msg4-alt"] == "msg3-alt"

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        """Should return empty map for empty file."""
        from memory_service.ingestion.claude.transcript_sync import (
            stream_find_common_ancestor_and_records,
        )

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text("")

        # With empty file, there's no valid head to pass
        # This is a degenerate case - in practice we'd check for empty first
        result = stream_find_common_ancestor_and_records(jsonl, "nonexistent", None)
        assert result.parent_of == {}

    def test_skips_records_without_uuid(self, tmp_path: Path) -> None:
        """Should skip non-message records."""
        from memory_service.ingestion.claude.transcript_sync import (
            stream_find_common_ancestor_and_records,
        )

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"type": "system", "data": "ignored"}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                ]
            )
            + "\n"
        )

        result = stream_find_common_ancestor_and_records(jsonl, "msg2", None)

        assert result.parent_of["msg1"] is None
        assert result.parent_of["msg2"] == "msg1"


class TestStreamingCommonAncestor:
    """Tests for finding common ancestor via streaming."""

    def test_x_is_ancestor_of_y(self, tmp_path: Path) -> None:
        """When X is on Y's branch, common ancestor is found."""
        from memory_service.ingestion.claude.transcript_sync import (
            stream_find_common_ancestor_and_records,
        )

        # msg1 -> msg2 -> msg3 -> msg4
        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2"}),
                    json.dumps({"uuid": "msg4", "parentUuid": "msg3"}),
                ]
            )
            + "\n"
        )

        result = stream_find_common_ancestor_and_records(jsonl, "msg4", "msg2")

        assert result.common_ancestor == "msg2"

    def test_finds_branch_point(self, tmp_path: Path) -> None:
        """Should find where branches diverged."""
        from memory_service.ingestion.claude.transcript_sync import (
            stream_find_common_ancestor_and_records,
        )

        # msg1 -> msg2 -> msg3 -> msg4 (current)
        #              \-> msg3' -> msg4' (last_indexed)
        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2"}),
                    json.dumps({"uuid": "msg4", "parentUuid": "msg3"}),
                    json.dumps({"uuid": "msg3-alt", "parentUuid": "msg2"}),
                    json.dumps({"uuid": "msg4-alt", "parentUuid": "msg3-alt"}),
                ]
            )
            + "\n"
        )

        result = stream_find_common_ancestor_and_records(jsonl, "msg4", "msg4-alt")

        assert result.common_ancestor == "msg2"

    def test_disjoint_branches_returns_none(self, tmp_path: Path) -> None:
        """When branches have no common ancestor, returns None."""
        from memory_service.ingestion.claude.transcript_sync import (
            stream_find_common_ancestor_and_records,
        )

        # msg1 -> msg2 (current)
        # alt1 -> alt2 (last_indexed)
        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "alt1", "parentUuid": None}),
                    json.dumps({"uuid": "alt2", "parentUuid": "alt1"}),
                ]
            )
            + "\n"
        )

        result = stream_find_common_ancestor_and_records(jsonl, "msg2", "alt2")

        assert result.common_ancestor is None


class TestAppendEntry:
    """Tests for AppendEntry dataclass."""

    def test_to_json(self) -> None:
        """Should serialize to JSON dict."""
        entry = AppendEntry(last_uuid="abc123", span_end=1523)

        assert entry.to_json() == {"last_uuid": "abc123", "span_end": 1523}

    def test_from_json(self) -> None:
        """Should deserialize from JSON dict."""
        data = {"last_uuid": "abc123", "span_end": 1523}

        entry = AppendEntry.from_json(data)

        assert entry.last_uuid == "abc123"
        assert entry.span_end == 1523


class TestAppendLog:
    """Tests for AppendLog class."""

    def test_append_and_iterate(self, tmp_path: Path) -> None:
        """Should append entries and iterate them."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))
        log.append(AppendEntry("msg3", 300))

        entries = list(log)
        assert len(entries) == 3
        assert entries[0].last_uuid == "msg1"
        assert entries[2].last_uuid == "msg3"

    def test_last_entry(self, tmp_path: Path) -> None:
        """Should return last entry efficiently."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))

        last = log.last_entry()
        assert last is not None
        assert last.last_uuid == "msg2"
        assert last.span_end == 200

    def test_last_entry_empty_log(self, tmp_path: Path) -> None:
        """Should return None for empty log."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        assert log.last_entry() is None

    def test_truncate_to(self, tmp_path: Path) -> None:
        """Should remove entries after the specified one."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))
        log.append(AppendEntry("msg3", 300))
        log.append(AppendEntry("msg4", 400))

        # Keep entries up to and including msg2
        log.truncate_to("msg2")

        entries = list(log)
        assert len(entries) == 2
        assert entries[-1].last_uuid == "msg2"

    def test_truncate_to_nonexistent_raises(self, tmp_path: Path) -> None:
        """Should raise if truncation uuid not found."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        log.append(AppendEntry("msg1", 100))

        with pytest.raises(ValueError, match="not found"):
            log.truncate_to("nonexistent")

    def test_persistence(self, tmp_path: Path) -> None:
        """Should persist entries across instances."""
        log_path = tmp_path / "append.log"

        log1 = AppendLog(log_path)
        log1.append(AppendEntry("msg1", 100))
        log1.append(AppendEntry("msg2", 200))

        log2 = AppendLog(log_path)
        entries = list(log2)

        assert len(entries) == 2
        assert entries[0].last_uuid == "msg1"
        assert entries[1].last_uuid == "msg2"


class TestGetAncestorChain:
    """Tests for getting ordered ancestor chain between two nodes."""

    def test_gets_chain_exclusive_of_ancestor(self) -> None:
        """Should return chain from ancestor to target, exclusive of ancestor."""
        from memory_service.ingestion.claude.transcript_sync import get_ancestor_chain

        # msg1 -> msg2 -> msg3 -> msg4
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "msg4": "msg3",
        }

        # Get chain from msg1 to msg4 (exclusive of msg1)
        chain = get_ancestor_chain("msg4", "msg1", parent_map)

        assert chain == ["msg2", "msg3", "msg4"]

    def test_gets_chain_to_root(self) -> None:
        """When ancestor is None, returns full chain from root."""
        from memory_service.ingestion.claude.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

        chain = get_ancestor_chain("msg3", None, parent_map)

        assert chain == ["msg1", "msg2", "msg3"]

    def test_immediate_child(self) -> None:
        """Chain from parent to child is just the child."""
        from memory_service.ingestion.claude.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
        }

        chain = get_ancestor_chain("msg2", "msg1", parent_map)

        assert chain == ["msg2"]

    def test_same_node_returns_empty(self) -> None:
        """When target equals ancestor, returns empty list."""
        from memory_service.ingestion.claude.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {"msg1": None, "msg2": "msg1"}

        chain = get_ancestor_chain("msg2", "msg2", parent_map)

        assert chain == []

    def test_raises_if_ancestor_not_in_chain(self) -> None:
        """Should raise if claimed ancestor isn't actually an ancestor."""
        from memory_service.ingestion.claude.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "other": None,
        }

        with pytest.raises(ValueError, match="not an ancestor"):
            get_ancestor_chain("msg2", "other", parent_map)


class TestComputeSyncPlan:
    """Tests for computing what sync operations are needed."""

    def test_no_op_when_already_synced(self, tmp_path: Path) -> None:
        """When transcript head matches last indexed, nothing to do."""
        from memory_service.ingestion.claude.transcript_sync import (
            AppendEntry,
            AppendLog,
            compute_sync_plan_streaming,
        )

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2"}),
                ]
            )
            + "\n"
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)
        log.append(AppendEntry("msg3", 300))

        plan = compute_sync_plan_streaming(
            transcript_path=transcript_path,
            current_head="msg3",
            append_log=log,
        )

        assert plan.uuids_to_transcribe == []
        assert plan.truncate_to_span is None

    def test_append_new_messages(self, tmp_path: Path) -> None:
        """When new messages added, transcribe them."""
        from memory_service.ingestion.claude.transcript_sync import (
            AppendEntry,
            AppendLog,
            compute_sync_plan_streaming,
        )

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2"}),
                    json.dumps({"uuid": "msg4", "parentUuid": "msg3"}),
                ]
            )
            + "\n"
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)
        log.append(AppendEntry("msg2", 200))

        plan = compute_sync_plan_streaming(
            transcript_path=transcript_path,
            current_head="msg4",
            append_log=log,
        )

        assert plan.uuids_to_transcribe == ["msg3", "msg4"]
        assert plan.truncate_to_span is None

    def test_revert_and_new_branch(self, tmp_path: Path) -> None:
        """When user reverted and continued, truncate and re-transcribe."""
        from memory_service.ingestion.claude.transcript_sync import (
            AppendEntry,
            AppendLog,
            compute_sync_plan_streaming,
        )

        # msg1 -> msg2 -> msg3 -> msg4 (indexed)
        #              \-> msg3' -> msg4' (current)
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2"}),
                    json.dumps({"uuid": "msg4", "parentUuid": "msg3"}),
                    json.dumps({"uuid": "msg3-alt", "parentUuid": "msg2"}),
                    json.dumps({"uuid": "msg4-alt", "parentUuid": "msg3-alt"}),
                ]
            )
            + "\n"
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)
        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))
        log.append(AppendEntry("msg3", 300))
        log.append(AppendEntry("msg4", 400))

        plan = compute_sync_plan_streaming(
            transcript_path=transcript_path,
            current_head="msg4-alt",
            append_log=log,
        )

        # Should truncate to msg2's span_end and transcribe the new branch
        assert plan.truncate_to_span == 200
        assert plan.truncate_to_uuid == "msg2"
        assert plan.uuids_to_transcribe == ["msg3-alt", "msg4-alt"]

    def test_empty_log_transcribes_full_chain(self, tmp_path: Path) -> None:
        """When append log is empty, transcribe from root."""
        from memory_service.ingestion.claude.transcript_sync import (
            AppendLog,
            compute_sync_plan_streaming,
        )

        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2"}),
                ]
            )
            + "\n"
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        plan = compute_sync_plan_streaming(
            transcript_path=transcript_path,
            current_head="msg3",
            append_log=log,
        )

        assert plan.uuids_to_transcribe == ["msg1", "msg2", "msg3"]
        assert plan.truncate_to_span is None

    def test_disjoint_branches_truncates_all(self, tmp_path: Path) -> None:
        """When branches are disjoint, truncate everything and start fresh."""
        from memory_service.ingestion.claude.transcript_sync import (
            AppendEntry,
            AppendLog,
            compute_sync_plan_streaming,
        )

        # Indexed: msg1 -> msg2
        # Current: alt1 -> alt2 (completely separate)
        transcript_path = tmp_path / "transcript.jsonl"
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None}),
                    json.dumps({"uuid": "msg2", "parentUuid": "msg1"}),
                    json.dumps({"uuid": "alt1", "parentUuid": None}),
                    json.dumps({"uuid": "alt2", "parentUuid": "alt1"}),
                ]
            )
            + "\n"
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)
        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))

        plan = compute_sync_plan_streaming(
            transcript_path=transcript_path,
            current_head="alt2",
            append_log=log,
        )

        # Should truncate from span 0 (delete everything) and transcribe new chain
        assert plan.truncate_to_span == 0
        assert plan.truncate_to_uuid is None
        assert plan.uuids_to_transcribe == ["alt1", "alt2"]


class TestSessionState:
    """Tests for SessionState JSONL format."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Should persist and restore state."""
        from memory_service.ingestion.claude.transcript_sync import (
            AppendEntry,
            SessionState,
            SessionStateHeader,
        )

        state_path = tmp_path / "session.jsonl"
        state = SessionState(
            header=SessionStateHeader(document_id="doc-123"),
            entries=[
                AppendEntry("msg1", 100),
                AppendEntry("msg2", 200),
            ],
        )

        state.save(state_path)
        loaded = SessionState.load(state_path)

        assert loaded is not None
        assert loaded.header.document_id == "doc-123"
        assert len(loaded.entries) == 2
        assert loaded.entries[0].last_uuid == "msg1"
        assert loaded.entries[1].span_end == 200

    def test_load_nonexistent_returns_none(self, tmp_path: Path) -> None:
        """Should return None for missing file."""
        from memory_service.ingestion.claude.transcript_sync import SessionState

        state = SessionState.load(tmp_path / "missing.jsonl")
        assert state is None

    def test_append_log_view(self, tmp_path: Path) -> None:
        """append_log() should return working AppendLog."""
        from memory_service.ingestion.claude.transcript_sync import (
            AppendEntry,
            SessionState,
            SessionStateHeader,
        )

        state = SessionState(
            header=SessionStateHeader(document_id="doc-123"),
            entries=[AppendEntry("msg1", 100)],
        )

        log = state.append_log()
        log.append(AppendEntry("msg2", 200))

        assert len(state.entries) == 2
        assert state.entries[1].last_uuid == "msg2"

        last = log.last_entry()
        assert last is not None
        assert last.last_uuid == "msg2"

    def test_append_log_truncate(self, tmp_path: Path) -> None:
        """append_log().truncate_to() should modify state entries."""
        from memory_service.ingestion.claude.transcript_sync import (
            AppendEntry,
            SessionState,
            SessionStateHeader,
        )

        state = SessionState(
            header=SessionStateHeader(document_id="doc-123"),
            entries=[
                AppendEntry("msg1", 100),
                AppendEntry("msg2", 200),
                AppendEntry("msg3", 300),
            ],
        )

        log = state.append_log()
        log.truncate_to("msg2")

        assert len(state.entries) == 2
        assert state.entries[-1].last_uuid == "msg2"


class TestGetCurrentHead:
    """Tests for getting current head UUID from transcript."""

    def test_gets_last_uuid(self, tmp_path: Path) -> None:
        """Should return the last UUID in the transcript."""
        from memory_service.ingestion.claude.transcript_sync import get_current_head

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "parentUuid": None, "type": "user"}),
                    json.dumps(
                        {"uuid": "msg2", "parentUuid": "msg1", "type": "assistant"}
                    ),
                    json.dumps({"uuid": "msg3", "parentUuid": "msg2", "type": "user"}),
                ]
            )
            + "\n"
        )

        head = get_current_head(jsonl)
        assert head == "msg3"

    def test_empty_transcript_returns_none(self, tmp_path: Path) -> None:
        """Should return None for empty transcript."""
        from memory_service.ingestion.claude.transcript_sync import get_current_head

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text("")

        head = get_current_head(jsonl)
        assert head is None

    def test_skips_records_without_uuid(self, tmp_path: Path) -> None:
        """Should skip non-message records."""
        from memory_service.ingestion.claude.transcript_sync import get_current_head

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps({"uuid": "msg1", "type": "user"}),
                    json.dumps({"type": "system", "data": "ignored"}),
                ]
            )
            + "\n"
        )

        head = get_current_head(jsonl)
        assert head == "msg1"


class TestTranscribeUuids:
    """Tests for transcribing specific UUIDs."""

    def test_transcribes_user_message(self) -> None:
        """Should transcribe user messages."""
        from memory_service.ingestion.claude.transcript_sync import (
            transcribe_uuids_from_map,
        )

        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "type": "user",
                "message": {"content": "Hello world"},
            }
        }

        text = transcribe_uuids_from_map(["msg1"], records)
        assert text == "[USER]\nHello world"

    def test_transcribes_assistant_message(self) -> None:
        """Should transcribe assistant messages with tool count."""
        from memory_service.ingestion.claude.transcript_sync import (
            transcribe_uuids_from_map,
        )

        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Here's my response"},
                        {"type": "tool_use", "name": "read"},
                        {"type": "tool_use", "name": "write"},
                    ]
                },
            }
        }

        text = transcribe_uuids_from_map(["msg1"], records)
        assert "[ASSISTANT]\nHere's my response" in text
        assert "[Used 2 tools: read, write]" in text

    def test_transcribes_multiple_in_order(self) -> None:
        """Should transcribe multiple UUIDs in specified order."""
        from memory_service.ingestion.claude.transcript_sync import (
            transcribe_uuids_from_map,
        )

        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "type": "user",
                "message": {"content": "First"},
            },
            "msg2": {
                "uuid": "msg2",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Second"}]},
            },
            "msg3": {
                "uuid": "msg3",
                "type": "user",
                "message": {"content": "Third"},
            },
        }

        text = transcribe_uuids_from_map(["msg1", "msg3"], records)
        assert "[USER]\nFirst" in text
        assert "[USER]\nThird" in text
        assert "Second" not in text
        # Verify order
        assert text.index("First") < text.index("Third")

    def test_empty_uuids_returns_empty(self) -> None:
        """Should return empty string for empty UUID list."""
        from memory_service.ingestion.claude.transcript_sync import (
            transcribe_uuids_from_map,
        )

        records: dict[str, dict[str, object]] = {
            "msg1": {"uuid": "msg1", "type": "user", "message": {"content": "Hello"}}
        }

        text = transcribe_uuids_from_map([], records)
        assert text == ""

    def test_skips_missing_uuids(self) -> None:
        """Should skip UUIDs not found in records."""
        from memory_service.ingestion.claude.transcript_sync import (
            transcribe_uuids_from_map,
        )

        records: dict[str, dict[str, object]] = {
            "msg1": {"uuid": "msg1", "type": "user", "message": {"content": "Hello"}}
        }

        text = transcribe_uuids_from_map(["msg1", "missing", "also-missing"], records)
        assert text == "[USER]\nHello"
