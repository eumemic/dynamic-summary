"""Tests for transcript sync with revert detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ragzoom.claude_memory.transcript_sync import (
    AppendEntry,
    AppendLog,
    build_parent_map,
    find_common_ancestor,
)


class TestBuildParentMap:
    """Tests for building uuid -> parentUuid map from transcript."""

    def test_builds_map_from_linear_transcript(self, tmp_path: Path) -> None:
        """Should map each uuid to its parent."""
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

        parent_map = build_parent_map(jsonl)

        assert parent_map == {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

    def test_builds_map_from_branched_transcript(self, tmp_path: Path) -> None:
        """Should include all branches in the map."""
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

        parent_map = build_parent_map(jsonl)

        assert parent_map["msg3"] == "msg2"
        assert parent_map["msg3-alt"] == "msg2"
        assert parent_map["msg4"] == "msg3"
        assert parent_map["msg4-alt"] == "msg3-alt"

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        """Should return empty map for empty file."""
        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text("")

        parent_map = build_parent_map(jsonl)

        assert parent_map == {}

    def test_skips_records_without_uuid(self, tmp_path: Path) -> None:
        """Should skip non-message records."""
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

        parent_map = build_parent_map(jsonl)

        assert parent_map == {"msg1": None, "msg2": "msg1"}


class TestFindCommonAncestor:
    """Tests for finding common ancestor of two uuids."""

    def test_x_is_ancestor_of_y(self) -> None:
        """When X is on Y's branch, common ancestor is X."""
        # msg1 -> msg2 -> msg3 -> msg4
        # X = msg2, Y = msg4
        parent_map = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "msg4": "msg3",
        }

        ancestor = find_common_ancestor("msg2", "msg4", parent_map)

        assert ancestor == "msg2"

    def test_y_is_ancestor_of_x(self) -> None:
        """When Y is on X's branch (unusual), common ancestor is Y."""
        parent_map = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "msg4": "msg3",
        }

        ancestor = find_common_ancestor("msg4", "msg2", parent_map)

        assert ancestor == "msg2"

    def test_x_equals_y(self) -> None:
        """When X and Y are the same, common ancestor is X."""
        parent_map = {"msg1": None, "msg2": "msg1"}

        ancestor = find_common_ancestor("msg2", "msg2", parent_map)

        assert ancestor == "msg2"

    def test_finds_branch_point(self) -> None:
        """Should find where branches diverged."""
        # msg1 -> msg2 -> msg3 -> msg4 (X = msg4)
        #              \-> msg3' -> msg4' (Y = msg4')
        parent_map = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "msg4": "msg3",
            "msg3-alt": "msg2",
            "msg4-alt": "msg3-alt",
        }

        ancestor = find_common_ancestor("msg4", "msg4-alt", parent_map)

        assert ancestor == "msg2"

    def test_finds_root_as_common_ancestor(self) -> None:
        """When branches diverge at root, common ancestor is root."""
        # msg1 -> msg2 (X = msg2)
        #     \-> msg2' (Y = msg2')
        parent_map = {
            "msg1": None,
            "msg2": "msg1",
            "msg2-alt": "msg1",
        }

        ancestor = find_common_ancestor("msg2", "msg2-alt", parent_map)

        assert ancestor == "msg1"

    def test_deep_branch_point(self) -> None:
        """Should handle deeply nested branch points."""
        # Long chain with branch near the end
        parent_map: dict[str, str | None] = {"msg1": None}
        for i in range(2, 101):
            parent_map[f"msg{i}"] = f"msg{i-1}"
        # Branch at msg98
        parent_map["msg99-alt"] = "msg98"
        parent_map["msg100-alt"] = "msg99-alt"

        ancestor = find_common_ancestor("msg100", "msg100-alt", parent_map)

        assert ancestor == "msg98"

    def test_disjoint_branches_returns_none(self) -> None:
        """Should return None when branches have no common ancestor."""
        # Two completely separate conversation trees
        # Tree 1: msg1 -> msg2 -> msg3
        # Tree 2: alt1 -> alt2 (started after reverting to before msg1)
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "alt1": None,
            "alt2": "alt1",
        }

        ancestor = find_common_ancestor("msg3", "alt2", parent_map)

        assert ancestor is None

    def test_raises_on_unknown_uuid(self) -> None:
        """Should raise if uuid not in parent map."""
        parent_map: dict[str, str | None] = {"msg1": None, "msg2": "msg1"}

        with pytest.raises(KeyError):
            find_common_ancestor("unknown", "msg2", parent_map)


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

    def test_find_valid_prefix(self, tmp_path: Path) -> None:
        """Should find last entry that's an ancestor of target."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))
        log.append(AppendEntry("msg3", 300))
        log.append(AppendEntry("msg4", 400))

        # msg1 -> msg2 -> msg3 -> msg4 (indexed)
        #              \-> msg3' -> msg4' (current branch)
        parent_map = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "msg4": "msg3",
            "msg3-alt": "msg2",
            "msg4-alt": "msg3-alt",
        }

        # Common ancestor of msg4 and msg4-alt is msg2
        valid_entry = log.find_valid_prefix("msg4-alt", parent_map)

        assert valid_entry is not None
        assert valid_entry.last_uuid == "msg2"
        assert valid_entry.span_end == 200

    def test_find_valid_prefix_no_revert(self, tmp_path: Path) -> None:
        """When no revert, should return last entry."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))

        # Linear chain, msg3 continues from msg2
        parent_map = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

        valid_entry = log.find_valid_prefix("msg3", parent_map)

        assert valid_entry is not None
        assert valid_entry.last_uuid == "msg2"

    def test_find_valid_prefix_empty_log(self, tmp_path: Path) -> None:
        """Empty log returns None, signaling 'transcribe from root to head'."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        # Transcript exists with messages, but we haven't indexed anything
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

        # None means: no valid prefix, caller should transcribe ancestor chain
        # from root (msg1) to current head (msg3)
        assert log.find_valid_prefix("msg3", parent_map) is None

    def test_find_valid_prefix_disjoint_branches(self, tmp_path: Path) -> None:
        """Disjoint branches return None, signaling 'transcribe from root'."""
        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        # We indexed msg1 -> msg2 -> msg3
        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))
        log.append(AppendEntry("msg3", 300))

        # User reverted to before msg1 and started completely fresh
        # alt1 -> alt2 (no shared ancestry with msg1-3)
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "alt1": None,
            "alt2": "alt1",
        }

        # None means: no valid prefix, caller should transcribe ancestor chain
        # from root (alt1) to current head (alt2), and truncate entire document
        assert log.find_valid_prefix("alt2", parent_map) is None


class TestGetAncestorChain:
    """Tests for getting ordered ancestor chain between two nodes."""

    def test_gets_chain_exclusive_of_ancestor(self) -> None:
        """Should return chain from ancestor to target, exclusive of ancestor."""
        from ragzoom.claude_memory.transcript_sync import get_ancestor_chain

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
        from ragzoom.claude_memory.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

        chain = get_ancestor_chain("msg3", None, parent_map)

        assert chain == ["msg1", "msg2", "msg3"]

    def test_immediate_child(self) -> None:
        """Chain from parent to child is just the child."""
        from ragzoom.claude_memory.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
        }

        chain = get_ancestor_chain("msg2", "msg1", parent_map)

        assert chain == ["msg2"]

    def test_same_node_returns_empty(self) -> None:
        """When target equals ancestor, returns empty list."""
        from ragzoom.claude_memory.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {"msg1": None, "msg2": "msg1"}

        chain = get_ancestor_chain("msg2", "msg2", parent_map)

        assert chain == []

    def test_raises_if_ancestor_not_in_chain(self) -> None:
        """Should raise if claimed ancestor isn't actually an ancestor."""
        from ragzoom.claude_memory.transcript_sync import get_ancestor_chain

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
        from ragzoom.claude_memory.transcript_sync import (
            AppendEntry,
            AppendLog,
            compute_sync_plan,
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)
        log.append(AppendEntry("msg3", 300))

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

        plan = compute_sync_plan(
            current_head="msg3",
            append_log=log,
            parent_map=parent_map,
        )

        assert plan.uuids_to_transcribe == []
        assert plan.truncate_to_span is None

    def test_append_new_messages(self, tmp_path: Path) -> None:
        """When new messages added, transcribe them."""
        from ragzoom.claude_memory.transcript_sync import (
            AppendEntry,
            AppendLog,
            compute_sync_plan,
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)
        log.append(AppendEntry("msg2", 200))

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "msg4": "msg3",
        }

        plan = compute_sync_plan(
            current_head="msg4",
            append_log=log,
            parent_map=parent_map,
        )

        assert plan.uuids_to_transcribe == ["msg3", "msg4"]
        assert plan.truncate_to_span is None

    def test_revert_and_new_branch(self, tmp_path: Path) -> None:
        """When user reverted and continued, truncate and re-transcribe."""
        from ragzoom.claude_memory.transcript_sync import (
            AppendEntry,
            AppendLog,
            compute_sync_plan,
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)
        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))
        log.append(AppendEntry("msg3", 300))
        log.append(AppendEntry("msg4", 400))

        # msg1 -> msg2 -> msg3 -> msg4 (indexed)
        #              \-> msg3' -> msg4' (current)
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "msg4": "msg3",
            "msg3-alt": "msg2",
            "msg4-alt": "msg3-alt",
        }

        plan = compute_sync_plan(
            current_head="msg4-alt",
            append_log=log,
            parent_map=parent_map,
        )

        # Should truncate to msg2's span_end and transcribe the new branch
        assert plan.truncate_to_span == 200
        assert plan.truncate_to_uuid == "msg2"
        assert plan.uuids_to_transcribe == ["msg3-alt", "msg4-alt"]

    def test_empty_log_transcribes_full_chain(self, tmp_path: Path) -> None:
        """When append log is empty, transcribe from root."""
        from ragzoom.claude_memory.transcript_sync import AppendLog, compute_sync_plan

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

        plan = compute_sync_plan(
            current_head="msg3",
            append_log=log,
            parent_map=parent_map,
        )

        assert plan.uuids_to_transcribe == ["msg1", "msg2", "msg3"]
        assert plan.truncate_to_span is None

    def test_disjoint_branches_truncates_all(self, tmp_path: Path) -> None:
        """When branches are disjoint, truncate everything and start fresh."""
        from ragzoom.claude_memory.transcript_sync import (
            AppendEntry,
            AppendLog,
            compute_sync_plan,
        )

        log_path = tmp_path / "append.log"
        log = AppendLog(log_path)
        log.append(AppendEntry("msg1", 100))
        log.append(AppendEntry("msg2", 200))

        # Completely separate conversation tree
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "alt1": None,
            "alt2": "alt1",
        }

        plan = compute_sync_plan(
            current_head="alt2",
            append_log=log,
            parent_map=parent_map,
        )

        # Should truncate from span 0 (delete everything) and transcribe new chain
        assert plan.truncate_to_span == 0
        assert plan.truncate_to_uuid is None
        assert plan.uuids_to_transcribe == ["alt1", "alt2"]


class TestSessionState:
    """Tests for SessionState JSONL format."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Should persist and restore state."""
        from ragzoom.claude_memory.transcript_sync import (
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
        from ragzoom.claude_memory.transcript_sync import SessionState

        state = SessionState.load(tmp_path / "missing.jsonl")
        assert state is None

    def test_append_log_view(self, tmp_path: Path) -> None:
        """append_log() should return working AppendLog."""
        from ragzoom.claude_memory.transcript_sync import (
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
        from ragzoom.claude_memory.transcript_sync import (
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
        from ragzoom.claude_memory.transcript_sync import get_current_head

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
        from ragzoom.claude_memory.transcript_sync import get_current_head

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text("")

        head = get_current_head(jsonl)
        assert head is None

    def test_skips_records_without_uuid(self, tmp_path: Path) -> None:
        """Should skip non-message records."""
        from ragzoom.claude_memory.transcript_sync import get_current_head

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

    def test_transcribes_user_message(self, tmp_path: Path) -> None:
        """Should transcribe user messages."""
        from ragzoom.claude_memory.transcript_sync import transcribe_uuids

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            json.dumps(
                {
                    "uuid": "msg1",
                    "type": "user",
                    "message": {"content": "Hello world"},
                }
            )
            + "\n"
        )

        text = transcribe_uuids(jsonl, ["msg1"])
        assert text == "[USER]\nHello world"

    def test_transcribes_assistant_message(self, tmp_path: Path) -> None:
        """Should transcribe assistant messages with tool count."""
        from ragzoom.claude_memory.transcript_sync import transcribe_uuids

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            json.dumps(
                {
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
            )
            + "\n"
        )

        text = transcribe_uuids(jsonl, ["msg1"])
        assert "[ASSISTANT]\nHere's my response" in text
        assert "[Used 2 tools: read, write]" in text

    def test_transcribes_multiple_in_order(self, tmp_path: Path) -> None:
        """Should transcribe multiple UUIDs in specified order."""
        from ragzoom.claude_memory.transcript_sync import transcribe_uuids

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "type": "user",
                            "message": {"content": "First"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "type": "assistant",
                            "message": {
                                "content": [{"type": "text", "text": "Second"}]
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "type": "user",
                            "message": {"content": "Third"},
                        }
                    ),
                ]
            )
            + "\n"
        )

        text = transcribe_uuids(jsonl, ["msg1", "msg3"])
        assert "[USER]\nFirst" in text
        assert "[USER]\nThird" in text
        assert "Second" not in text
        # Verify order
        assert text.index("First") < text.index("Third")

    def test_empty_uuids_returns_empty(self, tmp_path: Path) -> None:
        """Should return empty string for empty UUID list."""
        from ragzoom.claude_memory.transcript_sync import transcribe_uuids

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            json.dumps(
                {"uuid": "msg1", "type": "user", "message": {"content": "Hello"}}
            )
            + "\n"
        )

        text = transcribe_uuids(jsonl, [])
        assert text == ""

    def test_skips_missing_uuids(self, tmp_path: Path) -> None:
        """Should skip UUIDs not found in transcript."""
        from ragzoom.claude_memory.transcript_sync import transcribe_uuids

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text(
            json.dumps(
                {"uuid": "msg1", "type": "user", "message": {"content": "Hello"}}
            )
            + "\n"
        )

        text = transcribe_uuids(jsonl, ["msg1", "missing", "also-missing"])
        assert text == "[USER]\nHello"
