"""Tests for transcript sync with revert detection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ragzoom_claude_code.transcript_sync import (
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


class TestGetAncestorChain:
    """Tests for getting ordered ancestor chain between two nodes."""

    def test_gets_chain_exclusive_of_ancestor(self) -> None:
        """Should return chain from ancestor to target, exclusive of ancestor."""
        from ragzoom_claude_code.transcript_sync import get_ancestor_chain

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
        from ragzoom_claude_code.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }

        chain = get_ancestor_chain("msg3", None, parent_map)

        assert chain == ["msg1", "msg2", "msg3"]

    def test_immediate_child(self) -> None:
        """Chain from parent to child is just the child."""
        from ragzoom_claude_code.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
        }

        chain = get_ancestor_chain("msg2", "msg1", parent_map)

        assert chain == ["msg2"]

    def test_same_node_returns_empty(self) -> None:
        """When target equals ancestor, returns empty list."""
        from ragzoom_claude_code.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {"msg1": None, "msg2": "msg1"}

        chain = get_ancestor_chain("msg2", "msg2", parent_map)

        assert chain == []

    def test_raises_if_ancestor_not_in_chain(self) -> None:
        """Should raise if claimed ancestor isn't actually an ancestor."""
        from ragzoom_claude_code.transcript_sync import get_ancestor_chain

        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "other": None,
        }

        with pytest.raises(ValueError, match="not an ancestor"):
            get_ancestor_chain("msg2", "other", parent_map)


class TestGetCurrentHead:
    """Tests for getting current head UUID from transcript."""

    def test_gets_last_uuid(self, tmp_path: Path) -> None:
        """Should return the last UUID in the transcript."""
        from ragzoom_claude_code.transcript_sync import get_current_head

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
        from ragzoom_claude_code.transcript_sync import get_current_head

        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text("")

        head = get_current_head(jsonl)
        assert head is None

    def test_skips_records_without_uuid(self, tmp_path: Path) -> None:
        """Should skip non-message records."""
        from ragzoom_claude_code.transcript_sync import get_current_head

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


class TestTranscribeUuidsFromMap:
    """Tests for transcribing UUIDs using claude-transcriber."""

    def test_transcribes_user_message(self) -> None:
        """Should transcribe user messages using claude-transcriber format."""
        from ragzoom_claude_code.transcript_sync import transcribe_uuids_from_map

        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "type": "user",
                "message": {"content": "Hello world"},
            }
        }

        text = transcribe_uuids_from_map(["msg1"], records)
        # claude-transcriber uses ❯ prefix for user messages
        assert "Hello world" in text

    def test_transcribes_assistant_message(self) -> None:
        """Should transcribe assistant messages with tools."""
        from ragzoom_claude_code.transcript_sync import transcribe_uuids_from_map

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
        # claude-transcriber uses ⏺ prefix for assistant messages
        assert "Here's my response" in text
        # Tool uses are formatted individually
        assert "read" in text
        assert "write" in text

    def test_transcribes_multiple_in_order(self) -> None:
        """Should transcribe multiple UUIDs in specified order."""
        from ragzoom_claude_code.transcript_sync import transcribe_uuids_from_map

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
        assert "First" in text
        assert "Third" in text
        assert "Second" not in text
        # Verify order
        assert text.index("First") < text.index("Third")

    def test_empty_uuids_returns_empty(self) -> None:
        """Should return empty string for empty UUID list."""
        from ragzoom_claude_code.transcript_sync import transcribe_uuids_from_map

        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "type": "user",
                "message": {"content": "Hello"},
            }
        }

        text = transcribe_uuids_from_map([], records)
        assert text == ""

    def test_skips_missing_uuids(self) -> None:
        """Should skip UUIDs not found in records map."""
        from ragzoom_claude_code.transcript_sync import transcribe_uuids_from_map

        records: dict[str, dict[str, object]] = {
            "msg1": {
                "uuid": "msg1",
                "type": "user",
                "message": {"content": "Hello"},
            }
        }

        text = transcribe_uuids_from_map(["msg1", "missing", "also-missing"], records)
        assert "Hello" in text
