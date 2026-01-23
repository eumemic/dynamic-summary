"""Tests for time-windowed queries on synced transcripts.

Verifies that time-windowed queries work correctly on documents synced via
execute_sync(), returning the correct turns based on their timestamp ranges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.claude_memory.transcript_sync import execute_sync
from ragzoom.config import IndexConfig, QueryConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.retrieve import Retriever
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
    """Client wrapper around AppendExecutor for execute_sync tests."""

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
        """Batch append multiple units to document."""
        import asyncio

        self._ensure_document(document_id)
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

    def truncate(self, document_id: str, span_start: int) -> None:
        """Truncate document to span.

        Note: For these tests, truncation is not actually implemented since
        we're only testing the time-windowed query path. Truncation is handled
        by the indexing runtime, not AppendExecutor.
        """
        pass


def _create_retriever(
    backend: SQLiteStorageBackend,
    doc_id: str,
) -> Retriever:
    """Create a retriever for testing time-windowed queries."""
    from ragzoom.vector_factory import create_vector_index
    from tests.utils import create_retriever

    query_config = QueryConfig(budget_tokens=1000)

    vi = create_vector_index(
        "python", "sqlite:///:memory:", query_config.embedding_model
    )
    doc_store = backend.for_document(doc_id)
    return create_retriever(
        query_config=query_config,
        store=doc_store,
        document_id=doc_id,
        api_key="test-key",
        vector_index=vi,
    )


def _mock_retriever_for_query(
    retriever: Retriever, doc_id: str, leaves: list[tuple[str, int, int]]
) -> None:
    """Mock the retriever's vector search and embedding service.

    Args:
        retriever: Retriever to mock
        doc_id: Document ID
        leaves: List of (node_id, span_start, span_end) tuples for mock vectors
    """
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ragzoom.contracts.vector_filter import VectorFilter
    from ragzoom.vector_api import Vector

    def mock_search_similar(
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        filters: Sequence[VectorFilter] | None = None,
    ) -> list[Vector]:
        vectors = []
        for node_id, span_start, span_end in leaves:
            vectors.append(
                Vector(
                    id=node_id,
                    vec=np.ones(1536, dtype=np.float32),
                    meta={
                        "document_id": doc_id,
                        "span_start": span_start,
                        "span_end": span_end,
                        "parent_id": None,
                        "is_leaf": 1,
                    },
                    model_id="text-embedding-3-small",
                    dim=1536,
                )
            )
        return vectors

    retriever.vector_index.search_similar = mock_search_similar  # type: ignore[method-assign]
    retriever.embedding_service.get_query_embedding = (  # type: ignore[method-assign]
        lambda query, document_id=None: [0.3] * 1536
    )


@pytest.mark.slow_threshold(3.0)
class TestTimeWindowedQueryOnSyncedTranscript:
    """Tests that time-windowed queries work on synced transcripts."""

    def test_time_windowed_query_returns_correct_turns(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Synced transcripts should support time-windowed queries.

        This test verifies the full flow:
        1. Sync a transcript with timestamped turns
        2. Query using a time window
        3. Verify the time→span mapping correctly identifies the turns in range
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = AppendExecutorClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # Create a transcript with 3 distinct turns at different times:
        # Turn 1: 14:00:00 - 14:00:30 (asks about breakfast)
        # Turn 2: 14:05:00 - 14:05:45 (asks about weather)
        # Turn 3: 14:10:00 - 14:10:20 (says goodbye)
        transcript_path.write_text(
            "\n".join(
                [
                    # Turn 1: breakfast discussion
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-21T14:00:00Z",
                            "message": {"content": "What should I have for breakfast?"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:00:30Z",
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "I recommend eggs and toast!",
                                    }
                                ]
                            },
                        }
                    ),
                    # Turn 2: weather discussion
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-21T14:05:00Z",
                            "message": {"content": "What's the weather like today?"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:05:45Z",
                            "message": {
                                "content": [{"type": "text", "text": "Sunny and warm!"}]
                            },
                        }
                    ),
                    # Turn 3: goodbye
                    json.dumps(
                        {
                            "uuid": "msg5",
                            "parentUuid": "msg4",
                            "type": "user",
                            "timestamp": "2024-01-21T14:10:00Z",
                            "message": {"content": "Thanks, goodbye!"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg6",
                            "parentUuid": "msg5",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:10:20Z",
                            "message": {
                                "content": [{"type": "text", "text": "Goodbye!"}]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        # Execute sync
        result = execute_sync(transcript_path, state_path, client)

        # Verify document was created
        assert result.document_id == "transcript"
        assert len(result.appended_uuids) >= 3

        # Verify document is temporal
        is_temporal = sqlite_backend.doc_repo.get_document_is_temporal(
            result.document_id
        )
        assert is_temporal is True

        # Get leaf nodes to understand the spans
        doc_store = sqlite_backend.for_document(result.document_id)
        leaves = list(doc_store.nodes.iter_leaves())
        assert len(leaves) >= 1

        # Each turn should be one leaf with its timestamp range
        # Verify leaves have temporal metadata
        for leaf in leaves:
            assert leaf.time_start is not None, f"Leaf {leaf.id} missing time_start"
            assert leaf.time_end is not None, f"Leaf {leaf.id} missing time_end"

        # Create retriever and test time-windowed query
        retriever = _create_retriever(sqlite_backend, result.document_id)

        # Prepare leaf data for mock
        leaf_data = [(leaf.id, leaf.span_start, leaf.span_end) for leaf in leaves]
        _mock_retriever_for_query(retriever, result.document_id, leaf_data)

        # Query for weather discussion timeframe (14:04 - 14:06)
        # This should map to the span covering Turn 2
        import asyncio

        query_result = asyncio.run(
            retriever.retrieve_async(
                query="weather",
                num_seeds=1,
                budget_tokens=1000,
                document_id=result.document_id,
                time_start="2024-01-21T14:04:00Z",
                time_end="2024-01-21T14:06:00Z",
            )
        )

        # The time window should have been mapped to a span window
        # actual_start and actual_end should be set based on leaf lookup
        assert query_result.actual_start >= 0
        assert query_result.actual_end is not None
        assert query_result.actual_end > query_result.actual_start

        # The span window should cover the weather turn (Turn 2)
        # Find the leaf that contains the weather discussion
        weather_leaf = None
        for leaf in leaves:
            if leaf.time_start is not None and leaf.time_end is not None:
                # Unix timestamps for 14:05:00Z and 14:05:45Z
                t_start = 1705845900.0  # 2024-01-21T14:05:00Z
                t_end = 1705845945.0  # 2024-01-21T14:05:45Z
                # Check if this leaf's time range overlaps with Turn 2's time
                if leaf.time_start <= t_end and leaf.time_end >= t_start:
                    weather_leaf = leaf
                    break

        if weather_leaf is not None:
            # The returned span window should include this leaf's span
            assert query_result.actual_start <= weather_leaf.span_start
            assert query_result.actual_end >= weather_leaf.span_end

    def test_time_query_on_non_temporal_synced_doc_raises_error(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Time query on a non-temporal document should raise a clear error.

        This test creates a document without timestamps (simulating non-transcript
        content) and verifies the error handling.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        doc_id = "non-temporal-doc"

        # Create document without timestamps
        if sqlite_backend.doc_repo.get_document_by_id(doc_id) is None:
            sqlite_backend.add_document(
                document_id=doc_id,
                file_path=None,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-5-nano",
            )

        import asyncio

        store = sqlite_backend.for_document(doc_id)
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                executor.append(
                    store=store,
                    document_id=doc_id,
                    new_text="Non-temporal content without timestamps",
                    timestamp=None,
                )
            )
        finally:
            loop.close()

        # Verify document is NOT temporal
        is_temporal = sqlite_backend.doc_repo.get_document_is_temporal(doc_id)
        assert is_temporal is False

        # Create retriever
        retriever = _create_retriever(sqlite_backend, doc_id)

        # Try time-windowed query on non-temporal document
        with pytest.raises(ValueError, match="non-temporal"):
            asyncio.run(
                retriever.retrieve_async(
                    query="test",
                    num_seeds=1,
                    budget_tokens=1000,
                    document_id=doc_id,
                    time_start="2024-01-21T14:00:00Z",
                    time_end="2024-01-21T14:30:00Z",
                )
            )

    def test_partial_time_overlap_includes_turn(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """A time window that partially overlaps a turn should include that turn.

        Per spec: overlap semantics include any leaf whose time range overlaps
        the query window.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = AppendExecutorClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # Create a transcript with one turn spanning 14:00 - 14:05
        transcript_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-21T14:00:00Z",
                            "message": {"content": "Tell me a story"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:05:00Z",
                            "message": {
                                "content": [
                                    {"type": "text", "text": "Once upon a time..."}
                                ]
                            },
                        }
                    ),
                ]
            )
            + "\n"
        )

        result = execute_sync(transcript_path, state_path, client)

        # Get leaves for mock setup
        doc_store = sqlite_backend.for_document(result.document_id)
        leaves = list(doc_store.nodes.iter_leaves())
        assert len(leaves) >= 1

        # Create retriever
        retriever = _create_retriever(sqlite_backend, result.document_id)
        leaf_data = [(leaf.id, leaf.span_start, leaf.span_end) for leaf in leaves]
        _mock_retriever_for_query(retriever, result.document_id, leaf_data)

        # Query with time window that only partially overlaps (14:03 - 14:10)
        # The turn spans 14:00 - 14:05, so this overlaps at 14:03-14:05
        import asyncio

        query_result = asyncio.run(
            retriever.retrieve_async(
                query="story",
                num_seeds=1,
                budget_tokens=1000,
                document_id=result.document_id,
                time_start="2024-01-21T14:03:00Z",
                time_end="2024-01-21T14:10:00Z",
            )
        )

        # The query should succeed and include the turn
        assert query_result.actual_start is not None
        assert query_result.actual_end is not None

        # The span should cover the leaf (since it overlaps the time window)
        leaf = leaves[0]
        assert query_result.actual_start <= leaf.span_start
        assert query_result.actual_end >= leaf.span_end

    def test_compaction_summaries_not_indexed(
        self, sqlite_backend: SQLiteStorageBackend, tmp_path: Path
    ) -> None:
        """Compaction summaries (isCompactSummary=true) should not be indexed.

        Verifies that compaction summary messages are filtered out during
        turn grouping and do not appear in the indexed document content.
        """
        config = IndexConfig.load(target_chunk_tokens=None)
        executor = AppendExecutor(config, StubEmbedder())
        client = AppendExecutorClient(sqlite_backend, executor)

        transcript_path = tmp_path / "transcript.jsonl"
        state_path = tmp_path / "state.jsonl"

        # Create a transcript with a compaction summary in the middle
        transcript_path.write_text(
            "\n".join(
                [
                    # Turn 1: normal user/assistant exchange
                    json.dumps(
                        {
                            "uuid": "msg1",
                            "parentUuid": None,
                            "type": "user",
                            "timestamp": "2024-01-21T14:00:00Z",
                            "message": {"content": "Hello, I need help with my code"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg2",
                            "parentUuid": "msg1",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:00:30Z",
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Sure, I can help with your code!",
                                    }
                                ]
                            },
                        }
                    ),
                    # Compaction summary - should NOT be indexed
                    json.dumps(
                        {
                            "uuid": "compact1",
                            "parentUuid": "msg2",
                            "type": "user",
                            "timestamp": "2024-01-21T14:00:35Z",
                            "isCompactSummary": True,
                            "message": {
                                "content": "SECRET_COMPACTION_MARKER_XYZ123 - this summary should never appear in indexed content"
                            },
                        }
                    ),
                    # Turn 2: after compaction, normal exchange
                    json.dumps(
                        {
                            "uuid": "msg3",
                            "parentUuid": "compact1",
                            "type": "user",
                            "timestamp": "2024-01-21T14:05:00Z",
                            "message": {"content": "Can you fix this bug?"},
                        }
                    ),
                    json.dumps(
                        {
                            "uuid": "msg4",
                            "parentUuid": "msg3",
                            "type": "assistant",
                            "timestamp": "2024-01-21T14:05:45Z",
                            "message": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "I found and fixed the bug!",
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

        # Verify document was created
        assert result.document_id == "transcript"

        # Get all leaf nodes and their text content
        doc_store = sqlite_backend.for_document(result.document_id)
        leaves = list(doc_store.nodes.iter_leaves())
        assert len(leaves) >= 1

        # Collect all indexed text
        all_indexed_text = " ".join(
            leaf.text for leaf in leaves if leaf.text is not None
        )

        # The compaction summary marker should NOT appear in any indexed content
        assert "SECRET_COMPACTION_MARKER_XYZ123" not in all_indexed_text

        # But real user/assistant messages SHOULD appear
        assert "help with my code" in all_indexed_text
        assert "fix this bug" in all_indexed_text

        # Verify we have exactly 2 turns (msg1+msg2) and (msg3+msg4), not 3
        assert len(leaves) == 2, f"Expected 2 leaves (2 turns), got {len(leaves)}"
