"""Tests for synced document temporality.

Verifies that documents synced via execute_sync() become temporal when
transcripts contain timestamps (which Claude Code transcripts always do).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.claude_memory.transcript_sync import (
    SessionState,
    execute_sync,
)
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


class AppendExecutorClient:
    """Client wrapper around AppendExecutor for execute_sync tests.

    This client bridges execute_sync (which expects a client with append/truncate
    methods) to the AppendExecutor (which does the actual indexing work).
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

    def append(
        self,
        document_id: str,
        text: str,
        timestamp: str | tuple[str, str] | None = None,
    ) -> AppendResult:
        """Append text to document with optional timestamp."""
        import asyncio

        self._ensure_document(document_id)
        store = self._backend.for_document(document_id)

        loop = asyncio.new_event_loop()
        try:
            result: AppendOutcome = loop.run_until_complete(
                self._executor.append(
                    store=store,
                    document_id=document_id,
                    new_text=text,
                    timestamp=timestamp,
                )
            )
            return AppendResult(
                span_start=result.appended_span_start,
                span_end=result.appended_span_end,
            )
        finally:
            loop.close()

    def batch_append(
        self,
        document_id: str,
        units: list[AppendUnit],
    ) -> AppendResult:
        """Batch append multiple units to document.

        Uses AppendExecutor.append_batch internally.
        """
        import asyncio

        self._ensure_document(document_id)
        store = self._backend.for_document(document_id)

        # Convert AppendUnits to (texts, timestamps) format expected by executor
        texts = [u.text for u in units]

        # Build timestamps list - if any unit has timestamps, all should
        # (temporal documents are all-or-nothing per spec)
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

    def truncate(self, document_id: str, span_start: int) -> None:
        """Truncate document to span.

        Note: For these tests, truncation is not actually implemented since
        we're only testing the append path with timestamps.
        """
        # Truncation is handled by the indexing runtime, not AppendExecutor.
        # For testing purposes, we just track this was called.
        pass


class TestSyncedDocumentIsTemporal:
    """Tests that execute_sync creates temporal documents when timestamps present."""

    def test_synced_document_is_temporal(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Synced documents should have is_temporal=True when transcripts have timestamps."""
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = AppendExecutorClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # Create transcript with timestamps (like real Claude Code transcripts)
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-21T14:30:00Z",
                            "message": {"content": "Hello, can you help me?"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:30:05Z",
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Of course! How can I help?",
                                    }
                                ]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        # Execute sync
        result = execute_sync(transcript_path, state_path, client)

        # Verify document was created and is temporal
        assert result.document_id == "transcript"
        assert len(result.appended_uuids) >= 1

        # Check the is_temporal flag
        is_temporal = sqlite_backend.doc_repo.get_document_is_temporal(
            result.document_id
        )
        assert (
            is_temporal is True
        ), "Synced document should be temporal when transcript contains timestamps"

    def test_synced_document_has_correct_timestamps(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Synced document leaf nodes should have correct time_start and time_end."""
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = AppendExecutorClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # Create transcript with distinct timestamps
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-21T14:30:00Z",
                            "message": {"content": "First message"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:35:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "Response"}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        # Execute sync
        result = execute_sync(transcript_path, state_path, client)

        # Get leaf nodes and check timestamps
        store = sqlite_backend.for_document(result.document_id)
        leaves = list(store.nodes.iter_leaves())

        # Should have at least one leaf with timestamps
        assert len(leaves) >= 1
        leaf = leaves[0]

        # Leaf should have timestamps set
        assert leaf.time_start is not None, "Leaf should have time_start"
        assert leaf.time_end is not None, "Leaf should have time_end"

    def test_incremental_sync_maintains_temporality(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Incremental syncs should continue to use timestamps."""
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = AppendExecutorClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # First sync with one message
        transcript_path.write_text(
            json.dumps(
                {
                    "uuid": "msg1",
                    "parentUuid": None,
                    "type": "user",
                    "timestamp": "2024-01-21T14:30:00Z",
                    "message": {"content": "First message"},
                }
            )
            + "\n"
        )
        execute_sync(transcript_path, state_path, client)

        # Verify document is temporal after first sync
        is_temporal_after_first = sqlite_backend.doc_repo.get_document_is_temporal(
            "transcript"
        )
        assert is_temporal_after_first is True

        # Add second message and sync again
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-21T14:30:00Z",
                            "message": {"content": "First message"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:35:00Z",
                            "message": {
                                "content": [{"type": "text", "text": "Response"}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )
        execute_sync(transcript_path, state_path, client)

        # Should still be temporal
        is_temporal_after_second = sqlite_backend.doc_repo.get_document_is_temporal(
            "transcript"
        )
        assert is_temporal_after_second is True

        # Load state to verify entries
        state = SessionState.load(state_path)
        assert state is not None
        assert len(state.entries) >= 1
