"""Test that documents don't become zombies when parallelism slots are exhausted.

Regression test for a bug where concurrent trigger_work calls could starve
documents: when available_slots <= 0, _find_and_start_jobs returned without
re-queuing the document, leaving it marked active but with no workers.

Also tests that the fix doesn't introduce a busy loop: when slots are full,
the scheduler must NOT spin re-checking the same document until a slot frees.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, SecretStr
from ragzoom.server.indexing_engine import IndexingEngine
from ragzoom.services.llm_service import LLMService
from tests.test_builders import make_node_data


def _mock_sync_embeddings_create(**kwargs: object) -> Mock:
    """Mock for sync OpenAI client embeddings used by EmbeddingService."""
    input_texts = cast(list[str] | str, kwargs.get("input", []))
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    return Mock(data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_texts])


def _make_engine(
    storage_backend: SQLiteStorageBackend,
    max_parallelism: int = 2,
) -> tuple[IndexingEngine, LLMService]:
    """Create an IndexingEngine with mocked LLM services."""
    index_config = IndexConfig.load(target_chunk_tokens=50)
    llm_service = LLMService(index_config, api_key=SecretStr("test-key"))

    mock_async_client = AsyncMock()
    mock_chat_response = Mock(
        choices=[Mock(message=Mock(content="Summary of content"))],
        usage=SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            prompt_tokens_details=None,
        ),
    )
    mock_async_client.chat.completions.create = AsyncMock(
        return_value=mock_chat_response
    )

    async def mock_async_embeddings(**kwargs: object) -> Mock:
        input_data = cast(list[str] | str, kwargs.get("input", []))
        if isinstance(input_data, str):
            input_data = [input_data]
        return Mock(
            data=[SimpleNamespace(embedding=[0.1] * 1536) for _ in input_data],
            usage=SimpleNamespace(
                prompt_tokens=len(input_data) * 10,
                total_tokens=len(input_data) * 10,
            ),
        )

    mock_async_client.embeddings.create = AsyncMock(side_effect=mock_async_embeddings)
    llm_service.client = mock_async_client

    mock_openai_client = Mock()
    mock_openai_client.embeddings.create = Mock(
        side_effect=_mock_sync_embeddings_create
    )

    from ragzoom.vector_factory import create_vector_index

    engine = IndexingEngine(
        store=storage_backend,
        llm_service=llm_service,
        index_config=index_config,
        openai_client=mock_openai_client,
        vector_index_factory=lambda model_id: create_vector_index(
            "python", "sqlite:///:memory:", model_id
        ),
        max_parallelism=max_parallelism,
    )
    return engine, llm_service


def _add_document_leaves(
    storage_backend: SQLiteStorageBackend,
    doc_id: str,
    num_leaves: int = 2,
) -> None:
    """Add leaf nodes to a document in the storage backend."""
    doc_store = storage_backend.for_document(doc_id)
    doc_store.set_metadata(
        file_path=f"{doc_id}.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )
    doc_store.nodes.add_batch(
        [
            make_node_data(
                node_id=f"{doc_id}-leaf-{i}",
                text=f"Content for {doc_id} leaf {i}. Some text here.",
                span_start=i * 50,
                span_end=(i + 1) * 50,
                document_id=doc_id,
                token_count=25,
                height=0,
                level_index=i,
            )
            for i in range(num_leaves)
        ]
    )


class TestParallelismStarvation:
    """Verify all documents complete when more docs than parallelism slots."""

    @pytest.mark.asyncio
    @pytest.mark.slow_threshold(15.0)
    async def test_all_documents_complete_with_limited_parallelism(self) -> None:
        """4 documents with 2 leaves each, max_parallelism=2.

        First 2 docs grab both slots. Docs 3 and 4 must not become zombies —
        they should complete once slots free up.
        """
        storage_backend = SQLiteStorageBackend("sqlite:///:memory:")
        engine, _ = _make_engine(storage_backend, max_parallelism=2)

        doc_ids = [f"doc-{i}" for i in range(4)]
        for doc_id in doc_ids:
            _add_document_leaves(storage_backend, doc_id)

        for doc_id in doc_ids:
            await engine.trigger_work(doc_id)

        await engine.wait_until_idle(timeout=10)

        assert (
            engine._active_documents == set()
        ), f"Zombie documents remain active: {engine._active_documents}"
        assert (
            engine._active_jobs == set()
        ), f"Active jobs remain: {engine._active_jobs}"

        await engine.shutdown()
        storage_backend.close()

    @pytest.mark.asyncio
    async def test_no_busy_loop_when_slots_exhausted(self) -> None:
        """Scheduler must not spin when all parallelism slots are full.

        Reproduces the bug: when _find_and_start_jobs finds no available slots
        and re-queues the document as dirty, the scheduler re-processes it
        immediately in a tight loop, pegging the CPU at 100%.

        Strategy: patch _find_and_start_jobs to count calls, fill all slots
        with fake jobs, then trigger scheduling for a new document. If the
        scheduler busy-loops, the call count explodes within 100ms. A correct
        implementation calls _find_and_start_jobs at most once per document
        per scheduling round.
        """
        storage_backend = SQLiteStorageBackend("sqlite:///:memory:")
        engine, _ = _make_engine(storage_backend, max_parallelism=2)

        _add_document_leaves(storage_backend, "doc-blocked")

        # Manually fill all parallelism slots with fake active jobs
        # so _find_and_start_jobs will hit the "no slots" early return
        fake_job_1 = Mock()
        fake_job_1.document_id = "other-doc"
        fake_job_2 = Mock()
        fake_job_2.document_id = "other-doc"
        engine._active_jobs = {fake_job_1, fake_job_2}
        engine._active_documents.add("doc-blocked")

        # Track calls to the real _find_and_start_jobs
        original_find = engine._find_and_start_jobs
        call_count = 0

        async def counting_find(doc_id: str) -> None:
            nonlocal call_count
            call_count += 1
            await original_find(doc_id)

        with patch.object(engine, "_find_and_start_jobs", side_effect=counting_find):
            # Request scheduling — this should NOT busy-loop
            engine._request_scheduling("doc-blocked")

            # Give the scheduler time to misbehave if it's going to.
            # A busy loop would call _find_and_start_jobs thousands of times
            # in 100ms. A correct implementation calls it exactly once.
            await asyncio.sleep(0.1)

        # With a busy loop, call_count would be in the thousands.
        # Correct behavior: exactly 1 call (the initial scheduling attempt).
        assert call_count <= 2, (
            f"Scheduler busy-looped: _find_and_start_jobs called {call_count} times "
            f"in 100ms with no available slots. Expected at most 2."
        )

        # Clean up fake state
        engine._active_jobs.clear()
        engine._active_documents.clear()
        storage_backend.close()
