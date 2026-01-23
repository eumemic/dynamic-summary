"""Tests for execute_sync using batch_append instead of individual append calls.

Verifies that execute_sync() uses batch_append() with AppendUnits for efficient
temporal indexing of conversation turns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ragzoom.claude_memory.transcript_sync import (
    execute_sync,
)
from ragzoom.wrapper import AppendUnit


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
class BatchAppendResult:
    """Result type compatible with execute_sync expectations."""

    span_start: int
    span_end: int


@dataclass
class MockClient:
    """Mock client that tracks method calls for testing."""

    append_calls: list[tuple[str, str, str | tuple[str, str] | None]] = field(
        default_factory=list
    )
    batch_append_calls: list[tuple[str, list[AppendUnit]]] = field(default_factory=list)
    truncate_calls: list[tuple[str, int]] = field(default_factory=list)

    _span_counter: int = field(default=0)

    def append(
        self,
        document_id: str,
        text: str,
        timestamp: str | tuple[str, str] | None = None,
    ) -> BatchAppendResult:
        """Track individual append calls."""
        self.append_calls.append((document_id, text, timestamp))
        self._span_counter += len(text)
        return BatchAppendResult(span_start=0, span_end=self._span_counter)

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
        """Track truncate calls."""
        self.truncate_calls.append((document_id, span_start))


class TestExecuteSyncUsesBatchAppend:
    """Tests that execute_sync uses batch_append instead of individual appends."""

    def test_execute_sync_calls_batch_append_not_append(self, tmp_path: Path) -> None:
        """execute_sync should call batch_append, not individual append calls."""
        client = MockClient()

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # Create transcript with two turns
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        make_user_message(
                            "msg1",
                            None,
                            "2024-01-21T14:30:00Z",
                            "Hello, can you help me?",
                        )
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg2",
                            "msg1",
                            "2024-01-21T14:30:05Z",
                            "Of course! How can I help?",
                        )
                    ),
                    json.dumps(
                        make_user_message(
                            "msg3", "msg2", "2024-01-21T14:31:00Z", "What is Python?"
                        )
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg4",
                            "msg3",
                            "2024-01-21T14:31:10Z",
                            "Python is a programming language.",
                        )
                    ),
                ]
            )
            + "\n"
        )

        execute_sync(transcript_path, state_path, client)

        assert len(client.append_calls) == 0, "Should not call individual append()"
        assert len(client.batch_append_calls) == 1, "Should call batch_append() once"

    def test_batch_append_receives_append_units_with_timestamps(
        self, tmp_path: Path
    ) -> None:
        """batch_append should receive AppendUnit objects with correct timestamps."""
        client = MockClient()

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        make_user_message("msg1", None, "2024-01-21T14:30:00Z", "Hello")
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg2", "msg1", "2024-01-21T14:30:05Z", "Hi there!"
                        )
                    ),
                ]
            )
            + "\n"
        )

        execute_sync(transcript_path, state_path, client)

        # Verify batch_append was called with AppendUnits
        assert len(client.batch_append_calls) == 1
        document_id, units = client.batch_append_calls[0]
        assert document_id == "transcript"
        assert len(units) >= 1

        # Verify units are AppendUnit objects with timestamps
        for unit in units:
            assert isinstance(unit, AppendUnit)
            assert unit.time_start is not None, "AppendUnit should have time_start"
            assert unit.time_end is not None, "AppendUnit should have time_end"
            assert unit.is_temporal is True

    def test_each_turn_becomes_one_append_unit(self, tmp_path: Path) -> None:
        """Each conversation turn should become exactly one AppendUnit."""
        client = MockClient()

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        transcript_path.write_text(
            "\n".join(
                [
                    # Turn 1
                    json.dumps(
                        make_user_message(
                            "msg1", None, "2024-01-21T14:30:00Z", "First question"
                        )
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg2", "msg1", "2024-01-21T14:30:05Z", "First answer"
                        )
                    ),
                    # Turn 2
                    json.dumps(
                        make_user_message(
                            "msg3", "msg2", "2024-01-21T14:31:00Z", "Second question"
                        )
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg4", "msg3", "2024-01-21T14:31:10Z", "Second answer"
                        )
                    ),
                ]
            )
            + "\n"
        )

        execute_sync(transcript_path, state_path, client)

        assert len(client.batch_append_calls) == 1
        _, units = client.batch_append_calls[0]
        assert len(units) == 2, "Each turn should become one AppendUnit"

        assert units[0].time_start == "2024-01-21T14:30:00Z"
        assert units[0].time_end == "2024-01-21T14:30:05Z"
        assert units[1].time_start == "2024-01-21T14:31:00Z"
        assert units[1].time_end == "2024-01-21T14:31:10Z"

    def test_turn_content_is_correctly_transcribed(self, tmp_path: Path) -> None:
        """Each turn's AppendUnit should contain transcribed content from all messages."""
        client = MockClient()

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        make_user_message(
                            "msg1",
                            None,
                            "2024-01-21T14:30:00Z",
                            "Unique user content XYZ123",
                        )
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg2",
                            "msg1",
                            "2024-01-21T14:30:05Z",
                            "Unique assistant content ABC456",
                        )
                    ),
                ]
            )
            + "\n"
        )

        execute_sync(transcript_path, state_path, client)

        _, units = client.batch_append_calls[0]
        assert len(units) == 1
        assert "XYZ123" in units[0].text, "User content should be in turn text"
        assert "ABC456" in units[0].text, "Assistant content should be in turn text"

    def test_incremental_sync_uses_batch_append(self, tmp_path: Path) -> None:
        """Incremental syncs should also use batch_append."""
        client = MockClient()

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # First sync with one turn
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        make_user_message("msg1", None, "2024-01-21T14:30:00Z", "First")
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg2", "msg1", "2024-01-21T14:30:05Z", "One"
                        )
                    ),
                ]
            )
            + "\n"
        )
        execute_sync(transcript_path, state_path, client)

        # Reset mock tracking
        client.batch_append_calls = []
        client.append_calls = []

        # Add new turn and sync again
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        make_user_message("msg1", None, "2024-01-21T14:30:00Z", "First")
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg2", "msg1", "2024-01-21T14:30:05Z", "One"
                        )
                    ),
                    json.dumps(
                        make_user_message(
                            "msg3", "msg2", "2024-01-21T14:31:00Z", "Second"
                        )
                    ),
                    json.dumps(
                        make_assistant_message(
                            "msg4", "msg3", "2024-01-21T14:31:05Z", "Two"
                        )
                    ),
                ]
            )
            + "\n"
        )
        execute_sync(transcript_path, state_path, client)

        assert len(client.append_calls) == 0
        assert len(client.batch_append_calls) == 1

    def test_empty_transcript_does_not_call_batch_append(self, tmp_path: Path) -> None:
        """Empty transcript should not call batch_append."""
        client = MockClient()

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"
        transcript_path.write_text("")

        execute_sync(transcript_path, state_path, client)

        assert len(client.batch_append_calls) == 0
        assert len(client.append_calls) == 0

    def test_already_synced_does_not_call_batch_append(self, tmp_path: Path) -> None:
        """Already synced transcript should not call batch_append again."""
        client = MockClient()

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        transcript_path.write_text(
            json.dumps(make_user_message("msg1", None, "2024-01-21T14:30:00Z", "Hello"))
            + "\n"
        )

        # First sync
        execute_sync(transcript_path, state_path, client)
        assert len(client.batch_append_calls) == 1

        # Reset tracking
        client.batch_append_calls = []

        # Second sync with no changes
        execute_sync(transcript_path, state_path, client)
        assert len(client.batch_append_calls) == 0
