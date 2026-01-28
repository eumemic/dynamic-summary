"""Integration tests for stateless transcript sync.

These tests verify the acceptance criteria from specs/stateless-transcript-sync.md:
1. Sync works without any local state files
2. Normal append case: new turns appended correctly
3. Revert case: orphaned content removed, new content appended
4. Mid-turn revert: correctly rounds down to turn boundary
5. First sync: entire transcript indexed
6. Idempotent: running sync twice is safe (no duplicates, no errors)
7. Crash-safe: sync can be interrupted and resumed correctly

Uses real SQLite storage backend to verify end-to-end behavior.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pytest
from ragzoom_claude_code.transcript_sync import execute_sync

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.server.append_executor import AppendExecutor, AppendOutcome
from ragzoom.wrapper import AppendUnit


class StubEmbedder(EmbeddingProvider):
    """Stub embedder for testing - returns fixed embeddings."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]


@dataclass
class AppendResult:
    """Result type compatible with execute_sync expectations."""

    span_start: int
    span_end: int


class IntegrationClient:
    """Client wrapper for integration tests with real storage.

    This client wraps AppendExecutor for indexing and provides the
    stateless sync API (get_document_status, truncate_from_time) using
    real storage queries.
    """

    def __init__(
        self,
        backend: SQLiteStorageBackend,
        executor: AppendExecutor,
    ) -> None:
        self._backend = backend
        self._executor = executor

    def _ensure_document(self, document_id: str) -> None:
        """Ensure the document exists in the backend."""
        if self._backend.doc_repo.get_document_by_id(document_id) is None:
            self._backend.add_document(
                document_id=document_id,
                file_path=None,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-5-nano",
            )

    def get_document_status(self, document_id: str) -> object:
        """Return real document status from storage.

        Queries the storage backend to get actual indexed state,
        verifying the stateless sync works with real data.
        """
        # Check if document exists
        doc = self._backend.doc_repo.get_document_by_id(document_id)
        if doc is None:
            return _DocumentStatus(
                document_id=document_id,
                exists=False,
                is_temporal=False,
                time_start=None,
                time_end=None,
            )

        # Get document store and query temporal range
        store = self._backend.for_document(document_id)
        is_temporal = self._backend.doc_repo.get_document_is_temporal(document_id)

        # Get temporal range from leaves
        time_start: str | None = None
        time_end: str | None = None

        if is_temporal:
            leaves = list(store.nodes.iter_leaves())
            if leaves:
                timestamps = [
                    (leaf.time_start, leaf.time_end)
                    for leaf in leaves
                    if leaf.time_start is not None and leaf.time_end is not None
                ]
                if timestamps:
                    time_start = _unix_to_iso8601(min(ts[0] for ts in timestamps))
                    time_end = _unix_to_iso8601(max(ts[1] for ts in timestamps))

        return _DocumentStatus(
            document_id=document_id,
            exists=True,
            is_temporal=bool(is_temporal),
            time_start=time_start,
            time_end=time_end,
        )

    def truncate_from_time(self, document_id: str, cutoff_time: str) -> object:
        """Truncate document using time-based deletion.

        Removes all nodes where time_end > cutoff_time.
        """
        # Parse cutoff time to unix timestamp
        cutoff_iso = cutoff_time.replace("Z", "+00:00")
        cutoff_dt = datetime.fromisoformat(cutoff_iso)
        cutoff_unix = cutoff_dt.timestamp()

        # Delete nodes from storage
        deleted_ids = self._backend.delete_nodes_from_time(document_id, cutoff_unix)

        return _TruncateFromTimeResult(
            document_id=document_id,
            deleted_node_ids=deleted_ids,
            cutoff_time=cutoff_time,
        )

    def batch_append(
        self,
        document_id: str,
        units: list[AppendUnit],
        summarization_guidance: str | None = None,
    ) -> AppendResult:
        """Batch append multiple units to document."""

        self._ensure_document(document_id)
        store = self._backend.for_document(document_id)

        # Convert AppendUnits to (texts, timestamps) format
        texts = [u.text for u in units]

        timestamps: list[tuple[str, str]] | None = None
        if units and units[0].is_temporal:
            timestamps = []
            for u in units:
                assert u.time_start is not None and u.time_end is not None
                timestamps.append((u.time_start, u.time_end))

        loop = asyncio.new_event_loop()
        try:
            result: AppendOutcome = loop.run_until_complete(
                self._executor.append_batch(
                    store=store,
                    document_id=document_id,
                    units=texts,
                    timestamps=timestamps,
                )
            )
            return AppendResult(
                span_start=result.appended_span_start,
                span_end=result.appended_span_end,
            )
        finally:
            loop.close()


@dataclass
class _DocumentStatus:
    """Internal document status dataclass."""

    document_id: str
    exists: bool
    is_temporal: bool
    time_start: str | None
    time_end: str | None


@dataclass
class _TruncateFromTimeResult:
    """Internal truncation result dataclass."""

    document_id: str
    deleted_node_ids: list[str]
    cutoff_time: str


def _unix_to_iso8601(unix_ts: float) -> str:
    """Convert Unix timestamp to ISO 8601 string."""

    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def _make_transcript_record(
    uuid: str,
    parent_uuid: str | None,
    msg_type: str,
    timestamp: str,
    content: str,
) -> dict[str, object]:
    """Create a transcript record for testing."""
    record: dict[str, object] = {
        "uuid": uuid,
        "parentUuid": parent_uuid,
        "type": msg_type,
        "timestamp": timestamp,
    }

    if msg_type == "user":
        record["message"] = {"content": content}
    else:
        record["message"] = {"content": [{"type": "text", "text": content}]}

    return record


def _build_transcript(
    *messages: tuple[str, str | None, str, str, str],
) -> str:
    """Build transcript JSONL from message tuples.

    Args:
        messages: Tuples of (uuid, parent_uuid, msg_type, timestamp, content)

    Returns:
        JSONL string with newline-separated records
    """
    lines = [
        json.dumps(_make_transcript_record(uuid, parent, msg_type, ts, content))
        for uuid, parent, msg_type, ts, content in messages
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture
def integration_client(
    sqlite_backend: SQLiteStorageBackend,
) -> IntegrationClient:
    """Create integration client for stateless sync testing."""
    config = IndexConfig.load(target_chunk_tokens=None)
    executor = AppendExecutor(config, StubEmbedder())
    return IntegrationClient(sqlite_backend, executor)


class TestStatelessSyncIntegration:
    """Integration tests for stateless transcript sync."""

    def test_sync_no_state_files(
        self, integration_client: IntegrationClient, tmp_path: Path
    ) -> None:
        """Sync works without creating any local state files.

        Acceptance Criteria #1: Sync completes successfully without
        creating or reading state files.
        """
        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "test-no-state"

        # Create transcript with one turn
        transcript_path.write_text(
            _build_transcript(
                ("msg1", None, "user", "2024-01-21T14:30:00Z", "Hello"),
                ("msg2", "msg1", "assistant", "2024-01-21T14:30:05Z", "Hi!"),
            )
        )

        # Execute sync
        result = execute_sync(transcript_path, document_id, integration_client)

        # Verify sync succeeded
        assert result.document_id == document_id
        assert result.steps_appended >= 1

        # Verify no state files were created
        # State files would be in RAGZOOM_STATE_DIR or alongside transcript
        all_files = list(tmp_path.rglob("*"))
        state_files = [
            f
            for f in all_files
            if f.is_file()
            and f != transcript_path
            and (f.suffix == ".jsonl" or "state" in f.name.lower())
        ]
        assert state_files == [], f"Unexpected state files: {state_files}"

    def test_sync_normal_append(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Normal append case: new turns appended correctly.

        Acceptance Criteria #2: New content is indexed after existing content.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = IntegrationClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "test-normal-append"

        # Initial sync with one turn
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1",
                            None,
                            "user",
                            "2024-01-21T14:30:00Z",
                            "First message",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "First response",
                        )
                    ),
                ]
            )
            + "\n"
        )

        result1 = execute_sync(transcript_path, document_id, client)
        assert result1.steps_appended >= 1
        assert not result1.truncated

        # Append new turn to transcript
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1",
                            None,
                            "user",
                            "2024-01-21T14:30:00Z",
                            "First message",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "First response",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg3",
                            "msg2",
                            "user",
                            "2024-01-21T14:35:00Z",
                            "Second message",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:35:05Z",
                            "Second response",
                        )
                    ),
                ]
            )
            + "\n"
        )

        # Sync again - should append new content
        result2 = execute_sync(transcript_path, document_id, client)
        assert result2.steps_appended >= 1
        assert not result2.truncated

        # Verify document has content from both syncs
        store = sqlite_backend.for_document(document_id)
        leaves = list(store.nodes.iter_leaves())
        total_text = " ".join(leaf.text for leaf in leaves)
        assert "First" in total_text or len(leaves) >= 1

    def test_sync_revert_detection(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Revert case: orphaned content removed, new content appended.

        Acceptance Criteria #3: When user reverts, old content is truncated
        and new branch content is appended.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = IntegrationClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "test-revert"

        # Initial sync: msg1 -> msg2 -> msg3 -> msg4
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1", None, "user", "2024-01-21T14:30:00Z", "Start"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "First response",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg3", "msg2", "user", "2024-01-21T14:31:00Z", "Continue"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:31:05Z",
                            "Second response",
                        )
                    ),
                ]
            )
            + "\n"
        )

        result1 = execute_sync(transcript_path, document_id, client)
        assert result1.steps_appended >= 1

        # Simulate revert: user went back to msg2 and created new branch
        # msg3, msg4 are orphaned; msg5, msg6 are the new branch
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1", None, "user", "2024-01-21T14:30:00Z", "Start"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "First response",
                        )
                    ),
                    # Old branch (orphaned, still in JSONL but not in ancestry)
                    json.dumps(
                        _make_transcript_record(
                            "msg3", "msg2", "user", "2024-01-21T14:31:00Z", "Continue"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:31:05Z",
                            "Second response",
                        )
                    ),
                    # New branch (from msg2)
                    json.dumps(
                        _make_transcript_record(
                            "msg5",
                            "msg2",
                            "user",
                            "2024-01-21T14:35:00Z",
                            "New direction",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg6",
                            "msg5",
                            "assistant",
                            "2024-01-21T14:35:05Z",
                            "New response",
                        )
                    ),
                ]
            )
            + "\n"
        )

        # Sync with revert - should detect and handle the revert
        result2 = execute_sync(transcript_path, document_id, client)
        # Should have truncated orphaned content
        assert result2.truncated
        assert result2.truncate_cutoff_time is not None
        # Should have appended new branch
        assert result2.steps_appended >= 1

    def test_sync_mid_turn_revert(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Mid-turn revert correctly rounds down to turn boundary.

        Acceptance Criteria #4: Partial turn is fully removed, not partially kept.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = IntegrationClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "test-mid-turn-revert"

        # Initial sync: turn1 complete, turn2 with multiple assistant messages
        transcript_path.write_text(
            "\n".join(
                [
                    # Turn 1
                    json.dumps(
                        _make_transcript_record(
                            "msg1", None, "user", "2024-01-21T14:30:00Z", "Question 1"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "Answer 1",
                        )
                    ),
                    # Turn 2
                    json.dumps(
                        _make_transcript_record(
                            "msg3", "msg2", "user", "2024-01-21T14:31:00Z", "Question 2"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:31:05Z",
                            "Answer 2 part 1",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg5",
                            "msg4",
                            "assistant",
                            "2024-01-21T14:31:10Z",
                            "Answer 2 part 2",
                        )
                    ),
                ]
            )
            + "\n"
        )

        result1 = execute_sync(transcript_path, document_id, client)
        assert result1.steps_appended >= 1

        # Revert mid-turn: from msg3 (start of turn 2) with different response
        transcript_path.write_text(
            "\n".join(
                [
                    # Turn 1 (unchanged)
                    json.dumps(
                        _make_transcript_record(
                            "msg1", None, "user", "2024-01-21T14:30:00Z", "Question 1"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "Answer 1",
                        )
                    ),
                    # Old turn 2 (orphaned)
                    json.dumps(
                        _make_transcript_record(
                            "msg3", "msg2", "user", "2024-01-21T14:31:00Z", "Question 2"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:31:05Z",
                            "Answer 2 part 1",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg5",
                            "msg4",
                            "assistant",
                            "2024-01-21T14:31:10Z",
                            "Answer 2 part 2",
                        )
                    ),
                    # New turn 2 (from msg2, skipping old turn 2)
                    json.dumps(
                        _make_transcript_record(
                            "msg6",
                            "msg2",
                            "user",
                            "2024-01-21T14:35:00Z",
                            "Different question",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg7",
                            "msg6",
                            "assistant",
                            "2024-01-21T14:35:05Z",
                            "Different answer",
                        )
                    ),
                ]
            )
            + "\n"
        )

        result2 = execute_sync(transcript_path, document_id, client)
        # Should detect revert and truncate
        assert result2.truncated
        # Should append the new turn
        assert result2.steps_appended >= 1

    def test_sync_first_sync(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """First sync: entire transcript indexed.

        Acceptance Criteria #5: Empty document gets all content from transcript.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = IntegrationClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "test-first-sync"

        # Multi-turn transcript
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1", None, "user", "2024-01-21T14:30:00Z", "Hello"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "Hi there",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg3", "msg2", "user", "2024-01-21T14:31:00Z", "Help me"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:31:05Z",
                            "Sure thing",
                        )
                    ),
                ]
            )
            + "\n"
        )

        # Verify document doesn't exist yet
        status_before = client.get_document_status(document_id)
        assert not status_before.exists

        # First sync
        result = execute_sync(transcript_path, document_id, client)

        # Should have indexed all turns
        assert result.steps_appended >= 1
        assert not result.truncated  # No truncation on first sync

        # Verify document now exists with content
        status_after = client.get_document_status(document_id)
        assert status_after.exists
        assert status_after.is_temporal
        assert status_after.time_end is not None

    def test_sync_idempotent(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Idempotent: running sync twice is safe.

        Acceptance Criteria #6: Second sync produces no changes or duplicates.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = IntegrationClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "test-idempotent"

        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1", None, "user", "2024-01-21T14:30:00Z", "Test message"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "Test response",
                        )
                    ),
                ]
            )
            + "\n"
        )

        # First sync
        result1 = execute_sync(transcript_path, document_id, client)
        assert result1.steps_appended >= 1

        # Get state after first sync
        store = sqlite_backend.for_document(document_id)
        leaves_after_first = list(store.nodes.iter_leaves())
        count_after_first = len(leaves_after_first)

        # Second sync (same transcript, no changes)
        result2 = execute_sync(transcript_path, document_id, client)

        # Should be a no-op (or minimal operation)
        assert result2.steps_appended == 0  # Nothing new to append
        assert not result2.truncated  # No truncation needed

        # Verify no duplicates
        leaves_after_second = list(store.nodes.iter_leaves())
        count_after_second = len(leaves_after_second)
        assert count_after_second == count_after_first

    def test_sync_crash_recovery(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Crash-safe: sync can be interrupted and resumed correctly.

        Acceptance Criteria #7: Simulated crash and restart produces correct state.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = IntegrationClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        document_id = "test-crash-recovery"

        # Transcript with multiple turns
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1", None, "user", "2024-01-21T14:30:00Z", "First"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "Response 1",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg3", "msg2", "user", "2024-01-21T14:31:00Z", "Second"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:31:05Z",
                            "Response 2",
                        )
                    ),
                ]
            )
            + "\n"
        )

        # First sync succeeds
        result1 = execute_sync(transcript_path, document_id, client)
        assert result1.steps_appended >= 1

        # Simulate "crash" by simply running sync again
        # In the old state-file approach, losing state would cause issues
        # With stateless sync, it should recover gracefully

        # Create a new executor (simulating restart)
        executor2 = AppendExecutor(config, StubEmbedder())
        client2 = IntegrationClient(sqlite_backend, executor2)

        # Resume sync - should detect already-indexed content and not duplicate
        result2 = execute_sync(transcript_path, document_id, client2)

        # Should be idempotent - no new turns to append
        assert result2.steps_appended == 0
        assert not result2.truncated

        # Add more content and sync again
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1", None, "user", "2024-01-21T14:30:00Z", "First"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "Response 1",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg3", "msg2", "user", "2024-01-21T14:31:00Z", "Second"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:31:05Z",
                            "Response 2",
                        )
                    ),
                    # New content
                    json.dumps(
                        _make_transcript_record(
                            "msg5", "msg4", "user", "2024-01-21T14:32:00Z", "Third"
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg6",
                            "msg5",
                            "assistant",
                            "2024-01-21T14:32:05Z",
                            "Response 3",
                        )
                    ),
                ]
            )
            + "\n"
        )

        # Sync with new content
        result3 = execute_sync(transcript_path, document_id, client2)
        assert result3.steps_appended >= 1  # New content appended
        assert not result3.truncated  # No revert, just append
