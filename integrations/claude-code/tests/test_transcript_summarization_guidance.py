"""Integration tests for transcript summarization guidance.

Acceptance tests verifying that execute_sync() passes conversation-specific guidance
from specs/transcript-summarization-guidance.md § Acceptance Criteria #5.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from ragzoom_claude_code.transcript_sync import (
    CONVERSATION_SUMMARIZATION_GUIDANCE,
    execute_sync,
)

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.server.append_executor import AppendExecutor, AppendOutcome
from ragzoom.wrapper import AppendUnit

if TYPE_CHECKING:
    pass


class StubEmbedder(EmbeddingProvider):
    """Stub embedder for testing - returns fixed embeddings."""

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(i + 1)] * 4 for i, _ in enumerate(texts)]


@dataclass
class AppendResult:
    """Result type compatible with execute_sync expectations."""

    span_start: int
    span_end: int


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


class GuidanceCapturingClient:
    """Client that captures summarization_guidance for verification.

    This client extends the integration test pattern to track the guidance
    passed to batch_append(), and also stores it on the document.
    """

    def __init__(
        self,
        backend: SQLiteStorageBackend,
        executor: AppendExecutor,
    ) -> None:
        self._backend = backend
        self._executor = executor
        # Track the guidance that was passed to batch_append
        self.captured_guidance: str | None = None

    def _ensure_document(
        self, document_id: str, summarization_guidance: str | None
    ) -> None:
        """Ensure the document exists, creating with guidance if needed."""
        if self._backend.doc_repo.get_document_by_id(document_id) is None:
            self._backend.add_document(
                document_id=document_id,
                file_path=None,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-5-nano",
                summarization_guidance=summarization_guidance,
            )

    def get_document_status(self, document_id: str) -> _DocumentStatus:
        """Return real document status from storage."""
        doc = self._backend.doc_repo.get_document_by_id(document_id)
        if doc is None:
            return _DocumentStatus(
                document_id=document_id,
                exists=False,
                is_temporal=False,
                time_start=None,
                time_end=None,
            )

        store = self._backend.for_document(document_id)
        is_temporal = self._backend.doc_repo.get_document_is_temporal(document_id)

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

    def truncate_from_time(
        self, document_id: str, cutoff_time: str
    ) -> _TruncateFromTimeResult:
        """Truncate document using time-based deletion."""
        cutoff_iso = cutoff_time.replace("Z", "+00:00")
        cutoff_dt = datetime.fromisoformat(cutoff_iso)
        cutoff_unix = cutoff_dt.timestamp()

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
        """Batch append with guidance capture and storage."""
        # Capture the guidance for test verification
        self.captured_guidance = summarization_guidance

        # Ensure document with guidance
        self._ensure_document(document_id, summarization_guidance)
        store = self._backend.for_document(document_id)

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


class TestExecuteSyncSetsConversationGuidance:
    """Acceptance Criteria #5: execute_sync() passes conversation-specific guidance.

    Spec: specs/transcript-summarization-guidance.md § Acceptance Criteria #5
    """

    def test_execute_sync_passes_guidance_to_batch_append(
        self,
        sqlite_backend: SQLiteStorageBackend,
        tmp_path: Path,
    ) -> None:
        """execute_sync should pass CONVERSATION_SUMMARIZATION_GUIDANCE to batch_append.

        This verifies the guidance is correctly threaded from execute_sync
        to the client's batch_append method.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = GuidanceCapturingClient(sqlite_backend, executor)

        transcript_path = tmp_path / "test-transcript.jsonl"
        document_id = "guidance-test-doc"

        # Create a simple transcript
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
                            "Hi there!",
                        )
                    ),
                ]
            )
            + "\n"
        )

        # Execute sync
        result = execute_sync(transcript_path, document_id, client)

        # Verify sync completed
        assert result.turns_appended >= 1

        # Verify guidance was passed to batch_append
        assert client.captured_guidance == CONVERSATION_SUMMARIZATION_GUIDANCE

    def test_execute_sync_stores_guidance_on_document(
        self,
        sqlite_backend: SQLiteStorageBackend,
        tmp_path: Path,
    ) -> None:
        """execute_sync should store CONVERSATION_SUMMARIZATION_GUIDANCE on the document.

        This is an end-to-end integration test verifying the guidance is
        persisted to the document record in storage.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = GuidanceCapturingClient(sqlite_backend, executor)

        transcript_path = tmp_path / "test-transcript.jsonl"
        document_id = "guidance-storage-test"

        # Create transcript with multiple turns
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        _make_transcript_record(
                            "msg1",
                            None,
                            "user",
                            "2024-01-21T14:30:00Z",
                            "First question",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg2",
                            "msg1",
                            "assistant",
                            "2024-01-21T14:30:05Z",
                            "First answer",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg3",
                            "msg2",
                            "user",
                            "2024-01-21T14:31:00Z",
                            "Second question",
                        )
                    ),
                    json.dumps(
                        _make_transcript_record(
                            "msg4",
                            "msg3",
                            "assistant",
                            "2024-01-21T14:31:05Z",
                            "Second answer",
                        )
                    ),
                ]
            )
            + "\n"
        )

        # Execute sync
        result = execute_sync(transcript_path, document_id, client)
        assert result.turns_appended >= 1

        # Verify document was created with guidance stored
        doc = sqlite_backend.doc_repo.get_document_by_id(document_id)
        assert doc is not None, "Document should be created"
        assert doc.summarization_guidance == CONVERSATION_SUMMARIZATION_GUIDANCE

    def test_guidance_contains_conversation_preservation_instructions(self) -> None:
        """CONVERSATION_SUMMARIZATION_GUIDANCE should contain key preservation instructions.

        This verifies the constant is properly defined with the spec-required
        guidance elements.
        """
        # Verify identity and agency preservation
        assert "Identity and agency" in CONVERSATION_SUMMARIZATION_GUIDANCE
        assert "Who said what" in CONVERSATION_SUMMARIZATION_GUIDANCE

        # Verify decision preservation
        assert "Decisions and outcomes" in CONVERSATION_SUMMARIZATION_GUIDANCE

        # Verify causality preservation
        assert "Cause and effect" in CONVERSATION_SUMMARIZATION_GUIDANCE

        # Verify chronological flow
        assert "Chronological flow" in CONVERSATION_SUMMARIZATION_GUIDANCE

        # Verify technical terms are preserved
        assert "technical terms" in CONVERSATION_SUMMARIZATION_GUIDANCE
