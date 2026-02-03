"""Tests for saved_hook_context cycle detection and exclusion.

The Claude Code SDK writes saved_hook_context records during tool execution.
These records form intentional parentUuid cycles in the JSONL:

  assistant → last_hook → ... → first_hook → assistant

Without filtering, build_ancestry_chain loops infinitely and OOMs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from ragzoom_claude_code.transcript_sync import (
    RecordMeta,
    _build_metadata_and_parent_map,
    _build_records_and_parent_map,
    build_ancestry_chain,
    build_ancestry_chain_from_meta,
)


def _write_jsonl(records: list[dict[str, object]]) -> Path:
    """Write records to a temp JSONL file."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
    for record in records:
        tmp.write(json.dumps(record) + "\n")
    tmp.close()
    return Path(tmp.name)


class TestBuildAncestryChainCycleDetection:
    """build_ancestry_chain_from_meta must terminate on parentUuid cycles."""

    @pytest.mark.timeout(2)
    def test_cycle_terminates_and_returns_prefix(self) -> None:
        """A 3-node cycle terminates, returning the chain up to the revisit."""
        # Cycle: A -> B -> C -> A
        metadata: dict[str, RecordMeta] = {
            "A": RecordMeta(
                "A", "C", "2024-01-01T10:00:00Z", "assistant", False, False
            ),
            "B": RecordMeta("B", "A", "2024-01-01T10:00:01Z", "user", False, False),
            "C": RecordMeta(
                "C", "B", "2024-01-01T10:00:02Z", "assistant", False, False
            ),
        }
        parent_map = {"A": "C", "B": "A", "C": "B"}

        chain = build_ancestry_chain_from_meta("C", None, metadata, parent_map)

        # Must terminate (not hang/OOM). The chain walks C -> B -> A -> C(revisit),
        # so we get [C, B, A] reversed = [A, B, C].
        assert len(chain) == 3
        assert "A" in chain
        assert "B" in chain
        assert "C" in chain

    @pytest.mark.timeout(2)
    def test_cycle_with_linear_prefix(self) -> None:
        """Linear chain leading into a cycle: only the linear part is returned."""
        # Linear: D -> C -> B -> A -> C (cycle at A->C)
        metadata: dict[str, RecordMeta] = {
            "A": RecordMeta("A", "C", "2024-01-01T10:00:00Z", "user", False, False),
            "B": RecordMeta(
                "B", "A", "2024-01-01T10:00:01Z", "assistant", False, False
            ),
            "C": RecordMeta("C", "B", "2024-01-01T10:00:02Z", "user", False, False),
            "D": RecordMeta(
                "D", "C", "2024-01-01T10:00:03Z", "assistant", False, False
            ),
        }
        parent_map = {"A": "C", "B": "A", "C": "B", "D": "C"}

        chain = build_ancestry_chain_from_meta("D", None, metadata, parent_map)

        # Walks D -> C -> B -> A -> C(revisit, stop).
        # Collected [D, C, B, A], reversed = [A, B, C, D]
        assert len(chain) == 4

    def test_no_cycle_works_normally(self) -> None:
        """Linear chains still work correctly with cycle detection."""
        metadata: dict[str, RecordMeta] = {
            "A": RecordMeta("A", None, "2024-01-01T10:00:00Z", "user", False, False),
            "B": RecordMeta(
                "B", "A", "2024-01-01T10:01:00Z", "assistant", False, False
            ),
            "C": RecordMeta("C", "B", "2024-01-01T10:02:00Z", "user", False, False),
        }
        parent_map: dict[str, str | None] = {"A": None, "B": "A", "C": "B"}

        chain = build_ancestry_chain_from_meta("C", None, metadata, parent_map)

        assert chain == ["A", "B", "C"]


class TestBuildAncestryChainDeprecatedCycleDetection:
    """build_ancestry_chain (deprecated full-record path) must also handle cycles."""

    @pytest.mark.timeout(2)
    def test_cycle_terminates(self) -> None:
        """A 3-node cycle terminates in the deprecated code path."""
        records: dict[str, dict[str, object]] = {
            "A": {"uuid": "A", "parentUuid": "C", "type": "assistant"},
            "B": {"uuid": "B", "parentUuid": "A", "type": "user"},
            "C": {"uuid": "C", "parentUuid": "B", "type": "assistant"},
        }
        parent_map = {"A": "C", "B": "A", "C": "B"}

        chain = build_ancestry_chain("C", None, records, parent_map)

        assert len(chain) == 3


class TestBuildMetadataExcludesHookContext:
    """_build_metadata_and_parent_map must skip saved_hook_context records."""

    def test_excludes_saved_hook_context_from_metadata(self) -> None:
        """saved_hook_context records are excluded from the metadata dict."""
        path = _write_jsonl(
            [
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "type": "user",
                },
                {
                    "uuid": "hook1",
                    "parentUuid": "msg1",
                    "timestamp": "2024-01-01T10:00:01Z",
                    "type": "saved_hook_context",
                    "isSidechain": False,
                    "isCompactSummary": False,
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

            assert (
                "hook1" not in metadata
            ), "saved_hook_context should be excluded from metadata"
            assert (
                "hook1" not in parent_map
            ), "saved_hook_context should be excluded from parent_map"
            assert "msg1" in metadata
            assert "msg2" in metadata
        finally:
            path.unlink()

    def test_excludes_hook_cycle_from_parent_map(self) -> None:
        """A realistic hook cycle is excluded from the parent map entirely."""
        # Realistic structure: assistant (A) -> hook3 -> hook2 -> hook1 -> A (cycle)
        path = _write_jsonl(
            [
                {
                    "uuid": "user1",
                    "parentUuid": None,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "type": "user",
                },
                {
                    "uuid": "asst1",
                    "parentUuid": "user1",
                    "timestamp": "2024-01-01T10:01:00Z",
                    "type": "assistant",
                },
                {
                    "uuid": "hook1",
                    "parentUuid": "asst1",
                    "timestamp": "2024-01-01T10:01:01Z",
                    "type": "saved_hook_context",
                },
                {
                    "uuid": "hook2",
                    "parentUuid": "hook1",
                    "timestamp": "2024-01-01T10:01:02Z",
                    "type": "saved_hook_context",
                },
                {
                    "uuid": "hook3",
                    "parentUuid": "hook2",
                    "timestamp": "2024-01-01T10:01:03Z",
                    "type": "saved_hook_context",
                },
                {
                    "uuid": "user2",
                    "parentUuid": "asst1",
                    "timestamp": "2024-01-01T10:02:00Z",
                    "type": "user",
                },
            ]
        )

        try:
            metadata, parent_map = _build_metadata_and_parent_map(path)

            # No hook records in either map
            for hook_id in ("hook1", "hook2", "hook3"):
                assert hook_id not in metadata
                assert hook_id not in parent_map

            # Conversation records are present and connected
            assert parent_map["user2"] == "asst1"
            assert parent_map["asst1"] == "user1"
        finally:
            path.unlink()


class TestBuildRecordsAndParentMapExcludesHookContext:
    """_build_records_and_parent_map (deprecated) must also skip saved_hook_context."""

    def test_excludes_saved_hook_context(self) -> None:
        """saved_hook_context records are excluded from both records and parent_map."""
        path = _write_jsonl(
            [
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "timestamp": "2024-01-01T10:00:00Z",
                    "type": "user",
                },
                {
                    "uuid": "hook1",
                    "parentUuid": "msg1",
                    "timestamp": "2024-01-01T10:00:01Z",
                    "type": "saved_hook_context",
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
            records, parent_map = _build_records_and_parent_map(path)

            assert (
                "hook1" not in records
            ), "saved_hook_context should be excluded from records"
            assert (
                "hook1" not in parent_map
            ), "saved_hook_context should be excluded from parent_map"
            assert "msg1" in records
            assert "msg2" in records
        finally:
            path.unlink()
