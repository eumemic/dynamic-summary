"""Tests for transcript sync with revert detection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from ragzoom_claude_code.transcript_sync import (
    build_parent_map,
    find_common_ancestor,
    find_entries_after_time,
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


class TestConversationSummarizationGuidance:
    """Tests for CONVERSATION_SUMMARIZATION_GUIDANCE constant."""

    def test_conversation_guidance_constant_defined(self) -> None:
        """Should have guidance constant for conversation transcripts."""
        from ragzoom_claude_code.transcript_sync import (
            CONVERSATION_SUMMARIZATION_GUIDANCE,
        )

        # Constant should be a non-empty string
        assert isinstance(CONVERSATION_SUMMARIZATION_GUIDANCE, str)
        assert len(CONVERSATION_SUMMARIZATION_GUIDANCE) > 0

    def test_guidance_preserves_key_aspects(self) -> None:
        """Guidance should mention identity, decisions, causality, and chronology."""
        from ragzoom_claude_code.transcript_sync import (
            CONVERSATION_SUMMARIZATION_GUIDANCE,
        )

        guidance = CONVERSATION_SUMMARIZATION_GUIDANCE.lower()

        # Key aspects from spec
        assert "identity" in guidance
        assert "decision" in guidance
        assert "cause" in guidance or "why" in guidance
        assert "chronolog" in guidance or "temporal" in guidance

    def test_guidance_mentions_technical_preservation(self) -> None:
        """Guidance should instruct to preserve technical terms."""
        from ragzoom_claude_code.transcript_sync import (
            CONVERSATION_SUMMARIZATION_GUIDANCE,
        )

        guidance = CONVERSATION_SUMMARIZATION_GUIDANCE.lower()

        # Should preserve exact technical details
        assert "file path" in guidance or "function name" in guidance


class TestFindEntriesAfterTime:
    """Tests for finding entries after a cutoff time."""

    def test_finds_entries_after_cutoff(self) -> None:
        """Should return entries with timestamp > cutoff_time."""
        records: dict[str, dict[str, object]] = {
            "msg1": {"uuid": "msg1", "timestamp": "2024-01-10T10:00:00Z"},
            "msg2": {"uuid": "msg2", "timestamp": "2024-01-10T11:00:00Z"},
            "msg3": {"uuid": "msg3", "timestamp": "2024-01-10T12:00:00Z"},
        }
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }
        cutoff = datetime(2024, 1, 10, 10, 30, tzinfo=timezone.utc)

        result = find_entries_after_time("msg3", records, parent_map, cutoff)

        # Should include msg2 and msg3 (both after 10:30)
        assert result == ["msg2", "msg3"]

    def test_returns_empty_when_all_before_cutoff(self) -> None:
        """Should return empty list when all entries are at or before cutoff."""
        records: dict[str, dict[str, object]] = {
            "msg1": {"uuid": "msg1", "timestamp": "2024-01-10T10:00:00Z"},
            "msg2": {"uuid": "msg2", "timestamp": "2024-01-10T11:00:00Z"},
        }
        parent_map: dict[str, str | None] = {"msg1": None, "msg2": "msg1"}
        cutoff = datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc)

        result = find_entries_after_time("msg2", records, parent_map, cutoff)

        assert result == []

    def test_returns_all_when_all_after_cutoff(self) -> None:
        """Should return all entries when all are after cutoff."""
        records: dict[str, dict[str, object]] = {
            "msg1": {"uuid": "msg1", "timestamp": "2024-01-10T10:00:00Z"},
            "msg2": {"uuid": "msg2", "timestamp": "2024-01-10T11:00:00Z"},
        }
        parent_map: dict[str, str | None] = {"msg1": None, "msg2": "msg1"}
        cutoff = datetime(2024, 1, 10, 9, 0, tzinfo=timezone.utc)

        result = find_entries_after_time("msg2", records, parent_map, cutoff)

        assert result == ["msg1", "msg2"]

    def test_returns_chronological_order(self) -> None:
        """Should return entries in oldest-first order."""
        records: dict[str, dict[str, object]] = {
            "msg1": {"uuid": "msg1", "timestamp": "2024-01-10T10:00:00Z"},
            "msg2": {"uuid": "msg2", "timestamp": "2024-01-10T11:00:00Z"},
            "msg3": {"uuid": "msg3", "timestamp": "2024-01-10T12:00:00Z"},
            "msg4": {"uuid": "msg4", "timestamp": "2024-01-10T13:00:00Z"},
        }
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
            "msg4": "msg3",
        }
        cutoff = datetime(2024, 1, 10, 10, 30, tzinfo=timezone.utc)

        result = find_entries_after_time("msg4", records, parent_map, cutoff)

        # Should be chronological: msg2 -> msg3 -> msg4
        assert result == ["msg2", "msg3", "msg4"]

    def test_includes_entries_without_timestamp(self) -> None:
        """Should include entries without timestamps in the chain."""
        records: dict[str, dict[str, object]] = {
            "msg1": {"uuid": "msg1", "timestamp": "2024-01-10T10:00:00Z"},
            "msg2": {"uuid": "msg2"},  # No timestamp
            "msg3": {"uuid": "msg3", "timestamp": "2024-01-10T12:00:00Z"},
        }
        parent_map: dict[str, str | None] = {
            "msg1": None,
            "msg2": "msg1",
            "msg3": "msg2",
        }
        cutoff = datetime(2024, 1, 10, 10, 30, tzinfo=timezone.utc)

        result = find_entries_after_time("msg3", records, parent_map, cutoff)

        # msg2 has no timestamp so we can't compare, but msg3 is after
        # and msg2 is in the chain, so both should be included
        assert result == ["msg2", "msg3"]


class TestAppendOnlySync:
    """Tests for append-only sync mode."""

    def _make_transcript(
        self, tmp_path: Path, records: list[dict[str, object]]
    ) -> Path:
        """Helper to create a JSONL transcript file."""
        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return jsonl

    def _make_mock_client(
        self, exists: bool = False, time_end: str | None = None
    ) -> MagicMock:
        """Create a mock RagZoom client."""
        client = MagicMock()
        doc_status = MagicMock()
        doc_status.exists = exists
        doc_status.time_end = time_end
        client.get_document_status.return_value = doc_status
        client.batch_append.return_value = None
        client.truncate_from_time.return_value = None
        return client

    def test_append_only_first_sync(self, tmp_path: Path) -> None:
        """First sync in append-only mode should index everything."""
        from ragzoom_claude_code.transcript_sync import execute_sync

        jsonl = self._make_transcript(
            tmp_path,
            [
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "type": "user",
                    "timestamp": "2024-01-10T10:00:00Z",
                    "message": {"content": "Hello"},
                },
                {
                    "uuid": "msg2",
                    "parentUuid": "msg1",
                    "type": "assistant",
                    "timestamp": "2024-01-10T10:01:00Z",
                    "message": {"content": [{"type": "text", "text": "Hi there"}]},
                },
            ],
        )
        client = self._make_mock_client(exists=False)

        result = execute_sync(jsonl, "test-doc", client, append_only=True)

        assert result.steps_appended == 2
        assert result.truncated is False
        assert result.truncate_cutoff_time is None
        client.batch_append.assert_called_once()

    def test_append_only_normal_append(self, tmp_path: Path) -> None:
        """Append-only should add entries after time_end."""
        from ragzoom_claude_code.transcript_sync import execute_sync

        jsonl = self._make_transcript(
            tmp_path,
            [
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "type": "user",
                    "timestamp": "2024-01-10T10:00:00Z",
                    "message": {"content": "Hello"},
                },
                {
                    "uuid": "msg2",
                    "parentUuid": "msg1",
                    "type": "assistant",
                    "timestamp": "2024-01-10T10:01:00Z",
                    "message": {"content": [{"type": "text", "text": "Hi there"}]},
                },
                {
                    "uuid": "msg3",
                    "parentUuid": "msg2",
                    "type": "user",
                    "timestamp": "2024-01-10T10:02:00Z",
                    "message": {"content": "New message"},
                },
            ],
        )
        # Document already has msg1 and msg2 indexed
        client = self._make_mock_client(exists=True, time_end="2024-01-10T10:01:00Z")

        result = execute_sync(jsonl, "test-doc", client, append_only=True)

        # Should only append msg3
        assert result.steps_appended == 1
        assert result.truncated is False
        client.truncate_from_time.assert_not_called()

    def test_append_only_ignores_revert(self, tmp_path: Path) -> None:
        """Append-only should NOT detect or handle reverts."""
        from ragzoom_claude_code.transcript_sync import execute_sync

        # Simulates a transcript where user reverted: msg1 -> msg2 -> msg3
        # then reverted to msg1 and continued with msg2-alt
        # In normal mode this would truncate. In append-only, we just look for
        # entries after indexed time_end.
        jsonl = self._make_transcript(
            tmp_path,
            [
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "type": "user",
                    "timestamp": "2024-01-10T10:00:00Z",
                    "message": {"content": "Hello"},
                },
                {
                    "uuid": "msg2",
                    "parentUuid": "msg1",
                    "type": "assistant",
                    "timestamp": "2024-01-10T10:01:00Z",
                    "message": {"content": [{"type": "text", "text": "Response"}]},
                },
                {
                    "uuid": "msg3",
                    "parentUuid": "msg2",
                    "type": "user",
                    "timestamp": "2024-01-10T10:02:00Z",
                    "message": {"content": "Original continuation"},
                },
                # User reverted to msg1 and continued differently
                {
                    "uuid": "msg2-alt",
                    "parentUuid": "msg1",
                    "type": "assistant",
                    "timestamp": "2024-01-10T10:05:00Z",
                    "message": {"content": [{"type": "text", "text": "Alt response"}]},
                },
            ],
        )
        # Document has up to msg3 indexed (time_end = 10:02)
        # The current head is msg2-alt which is on a different branch
        client = self._make_mock_client(exists=True, time_end="2024-01-10T10:02:00Z")

        result = execute_sync(jsonl, "test-doc", client, append_only=True)

        # In append-only mode, we just look for entries > time_end
        # msg2-alt has timestamp 10:05 which is > 10:02, so it gets appended
        # No truncation should occur
        assert result.truncated is False
        assert result.truncate_cutoff_time is None
        client.truncate_from_time.assert_not_called()
        # Should append msg2-alt (timestamp 10:05 > 10:02)
        assert result.steps_appended == 1

    def test_append_only_no_new_content(self, tmp_path: Path) -> None:
        """Append-only should return 0 steps when nothing new."""
        from ragzoom_claude_code.transcript_sync import execute_sync

        jsonl = self._make_transcript(
            tmp_path,
            [
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "type": "user",
                    "timestamp": "2024-01-10T10:00:00Z",
                    "message": {"content": "Hello"},
                },
            ],
        )
        # Document already has everything indexed
        client = self._make_mock_client(exists=True, time_end="2024-01-10T10:00:00Z")

        result = execute_sync(jsonl, "test-doc", client, append_only=True)

        assert result.steps_appended == 0
        assert result.truncated is False
        client.batch_append.assert_not_called()


class TestAppendOnlyCli:
    """Tests for append-only CLI flag and environment variable."""

    def test_append_only_flag_in_sync_command(self) -> None:
        """CLI sync command should accept --append-only flag."""
        from click.testing import CliRunner
        from ragzoom_claude_code.cli import sync_cmd

        runner = CliRunner()
        # Just verify the option is accepted (will fail on missing file, that's ok)
        result = runner.invoke(sync_cmd, ["--help"])

        assert result.exit_code == 0
        assert "--append-only" in result.output
        assert "RAGZOOM_APPEND_ONLY" in result.output

    def test_append_only_env_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI should pick up RAGZOOM_APPEND_ONLY environment variable."""
        from unittest.mock import patch

        from click.testing import CliRunner
        from ragzoom_claude_code.cli import sync_cmd

        # Create a minimal transcript
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(
            json.dumps(
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "type": "user",
                    "timestamp": "2024-01-10T10:00:00Z",
                    "message": {"content": "Hello"},
                }
            )
            + "\n"
        )

        # Mock execute_sync to capture the append_only argument
        with patch("ragzoom_claude_code.cli.execute_sync") as mock_sync:
            mock_result = MagicMock()
            mock_result.truncated = False
            mock_result.steps_appended = 1
            mock_result.document_id = "test"
            mock_sync.return_value = mock_result

            runner = CliRunner(env={"RAGZOOM_APPEND_ONLY": "1"})
            result = runner.invoke(sync_cmd, [str(jsonl)])

            # Should have called execute_sync with append_only=True
            assert result.exit_code == 0
            mock_sync.assert_called_once()
            call_kwargs = mock_sync.call_args[1]
            assert call_kwargs.get("append_only") is True
