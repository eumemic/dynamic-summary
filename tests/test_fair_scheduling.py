"""Tests for greedy job scheduling across documents in IndexingEngine.

Verifies that documents greedily claim all available parallelism slots
without per-document caps. The min(available, remaining) check prevents
overcommitting, while the coalescing scheduler gives every document a
chance each round.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from ragzoom.server.indexing_engine import (
    EmbeddingJob,
    IndexingEngine,
    IndexingJob,
)


def _make_engine(max_parallelism: int = 10) -> IndexingEngine:
    """Create an IndexingEngine with mocked dependencies for scheduling tests."""
    return IndexingEngine(
        store=MagicMock(),
        llm_service=MagicMock(),
        index_config=MagicMock(target_chunk_tokens=512),
        openai_client=MagicMock(),
        max_parallelism=max_parallelism,
    )


class TestGreedyScheduling:
    """Test greedy distribution of parallelism slots across documents."""

    @pytest.mark.asyncio
    async def test_two_docs_greedy_scheduling(self) -> None:
        """With greedy scheduling, the pool should be fully saturated (10/10 slots)."""
        engine = _make_engine(max_parallelism=10)

        job_complete_events: list[asyncio.Event] = []

        async def mock_run_job(job: EmbeddingJob) -> None:
            event = asyncio.Event()
            job_complete_events.append(event)
            await event.wait()

        # Each doc has 20 embedding jobs (far more than max_parallelism)
        remaining_a = [
            EmbeddingJob(document_id="doc-a", leaf_id=f"leaf-{i}") for i in range(20)
        ]
        remaining_b = [
            EmbeddingJob(document_id="doc-b", leaf_id=f"leaf-{i}") for i in range(20)
        ]

        def mock_find_next_n_jobs(
            document_id: str,
            active_jobs: set[IndexingJob],
            ctx: object,
            max_jobs: int,
        ) -> list[EmbeddingJob]:
            remaining = remaining_a if document_id == "doc-a" else remaining_b
            found: list[EmbeddingJob] = []
            for job in list(remaining):
                if job not in active_jobs and len(found) < max_jobs:
                    found.append(job)
                    remaining.remove(job)
            return found

        engine._active_documents = {"doc-a", "doc-b"}
        engine._document_contexts = {
            "doc-a": MagicMock(cancelled=False),
            "doc-b": MagicMock(cancelled=False),
        }

        with (
            patch.object(
                engine, "_find_next_n_jobs", side_effect=mock_find_next_n_jobs
            ),
            patch.object(engine, "_run_job", side_effect=mock_run_job),
        ):
            # Trigger scheduling for both documents
            task_a = asyncio.create_task(engine._find_and_start_jobs("doc-a"))
            task_b = asyncio.create_task(engine._find_and_start_jobs("doc-b"))
            await asyncio.sleep(0.05)

            # Greedy: pool should be fully saturated regardless of per-doc distribution
            total_inflight = len(engine._active_jobs)
            assert (
                total_inflight == 10
            ), f"Pool should be saturated at 10, got {total_inflight}"

            # Clean up
            for event in job_complete_events:
                event.set()
            await asyncio.sleep(0.05)
            task_a.cancel()
            task_b.cancel()
            await asyncio.gather(task_a, task_b, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_single_doc_gets_full_throughput(self) -> None:
        """With only one active document, it should claim all 10 slots."""
        engine = _make_engine(max_parallelism=10)

        jobs_started = 0
        job_complete_events: list[asyncio.Event] = []

        async def mock_run_job(job: EmbeddingJob) -> None:
            nonlocal jobs_started
            jobs_started += 1
            event = asyncio.Event()
            job_complete_events.append(event)
            await event.wait()

        doc_jobs = [
            EmbeddingJob(document_id="doc-a", leaf_id=f"leaf-{i}") for i in range(20)
        ]
        remaining = list(doc_jobs)

        def mock_find_next_n_jobs(
            document_id: str,
            active_jobs: set[IndexingJob],
            ctx: object,
            max_jobs: int,
        ) -> list[EmbeddingJob]:
            found: list[EmbeddingJob] = []
            for job in list(remaining):
                if job not in active_jobs and len(found) < max_jobs:
                    found.append(job)
                    remaining.remove(job)
            return found

        engine._active_documents = {"doc-a"}
        engine._document_contexts = {"doc-a": MagicMock(cancelled=False)}

        with (
            patch.object(
                engine, "_find_next_n_jobs", side_effect=mock_find_next_n_jobs
            ),
            patch.object(engine, "_run_job", side_effect=mock_run_job),
        ):
            task = asyncio.create_task(engine._find_and_start_jobs("doc-a"))
            await asyncio.sleep(0.05)

            # Single doc should get all 10 slots
            inflight = sum(1 for j in engine._active_jobs if j.document_id == "doc-a")
            assert inflight == 10, f"Expected 10 inflight, got {inflight}"

            for event in job_complete_events:
                event.set()
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_unused_slots_filled_immediately(self) -> None:
        """Doc B fills remaining slots immediately — no redistribution pass needed.

        Doc A has only 2 jobs. With greedy scheduling, doc B claims all
        remaining 8 slots in the same scheduling round.
        """
        engine = _make_engine(max_parallelism=10)

        job_events: dict[str, asyncio.Event] = {}

        async def mock_run_job(job: EmbeddingJob) -> None:
            event = asyncio.Event()
            key = f"{job.document_id}:{job.leaf_id}"
            job_events[key] = event
            await event.wait()

        remaining_a = [
            EmbeddingJob(document_id="doc-a", leaf_id=f"leaf-{i}") for i in range(2)
        ]
        remaining_b = [
            EmbeddingJob(document_id="doc-b", leaf_id=f"leaf-{i}") for i in range(20)
        ]

        def mock_find_next_n_jobs(
            document_id: str,
            active_jobs: set[IndexingJob],
            ctx: object,
            max_jobs: int,
        ) -> list[EmbeddingJob]:
            remaining = remaining_a if document_id == "doc-a" else remaining_b
            found: list[EmbeddingJob] = []
            for job in list(remaining):
                if job not in active_jobs and len(found) < max_jobs:
                    found.append(job)
                    remaining.remove(job)
            return found

        engine._active_documents = {"doc-a", "doc-b"}
        engine._document_contexts = {
            "doc-a": MagicMock(cancelled=False, leaves_at_last_idle=0),
            "doc-b": MagicMock(cancelled=False, leaves_at_last_idle=0),
        }

        with (
            patch.object(
                engine, "_find_next_n_jobs", side_effect=mock_find_next_n_jobs
            ),
            patch.object(engine, "_run_job", side_effect=mock_run_job),
            patch.object(engine, "_maybe_complete_runs", return_value=None),
            patch.object(engine, "_safe_on_document_idle", return_value=None),
        ):
            # Schedule doc A first (2 jobs), then doc B fills the rest
            task_a = asyncio.create_task(engine._find_and_start_jobs("doc-a"))
            await asyncio.sleep(0.01)
            task_b = asyncio.create_task(engine._find_and_start_jobs("doc-b"))
            await asyncio.sleep(0.05)

            inflight_a = sum(1 for j in engine._active_jobs if j.document_id == "doc-a")
            inflight_b = sum(1 for j in engine._active_jobs if j.document_id == "doc-b")
            total = len(engine._active_jobs)

            assert inflight_a == 2, f"doc-a should have 2 inflight, got {inflight_a}"
            assert inflight_b == 8, f"doc-b should fill remaining 8, got {inflight_b}"
            assert total == 10, f"Pool should be saturated at 10, got {total}"

            # Clean up
            for event in job_events.values():
                event.set()
            await asyncio.sleep(0.05)
            for t in [task_a, task_b]:
                t.cancel()
            await asyncio.gather(task_a, task_b, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_idle_doc_frees_slots(self) -> None:
        """After doc A goes idle, doc B claims the freed slots.

        Doc A takes 3 slots, doc B fills remaining 7. When doc A's jobs
        complete and it goes idle, doc B re-schedules and fills to 10.
        """
        engine = _make_engine(max_parallelism=10)

        job_events: dict[str, asyncio.Event] = {}

        async def mock_run_job(job: EmbeddingJob) -> None:
            event = asyncio.Event()
            key = f"{job.document_id}:{job.leaf_id}"
            job_events[key] = event
            await event.wait()

        remaining_a = [
            EmbeddingJob(document_id="doc-a", leaf_id=f"leaf-{i}") for i in range(3)
        ]
        remaining_b = [
            EmbeddingJob(document_id="doc-b", leaf_id=f"leaf-{i}") for i in range(20)
        ]

        def mock_find_next_n_jobs(
            document_id: str,
            active_jobs: set[IndexingJob],
            ctx: object,
            max_jobs: int,
        ) -> list[EmbeddingJob]:
            remaining = remaining_a if document_id == "doc-a" else remaining_b
            found: list[EmbeddingJob] = []
            for job in list(remaining):
                if job not in active_jobs and len(found) < max_jobs:
                    found.append(job)
                    remaining.remove(job)
            return found

        engine._active_documents = {"doc-a", "doc-b"}
        engine._document_contexts = {
            "doc-a": MagicMock(cancelled=False, leaves_at_last_idle=0),
            "doc-b": MagicMock(cancelled=False, leaves_at_last_idle=0),
        }

        with (
            patch.object(
                engine, "_find_next_n_jobs", side_effect=mock_find_next_n_jobs
            ),
            patch.object(engine, "_run_job", side_effect=mock_run_job),
            patch.object(engine, "_maybe_complete_runs", return_value=None),
            patch.object(engine, "_safe_on_document_idle", return_value=None),
        ):
            # Phase 1: doc A takes 3, doc B fills remaining 7
            task_a = asyncio.create_task(engine._find_and_start_jobs("doc-a"))
            await asyncio.sleep(0.01)
            task_b = asyncio.create_task(engine._find_and_start_jobs("doc-b"))
            await asyncio.sleep(0.05)

            total = len(engine._active_jobs)
            assert total == 10, f"Pool should be saturated at 10, got {total}"

            # Phase 2: doc A completes and goes idle
            doc_a_jobs = {j for j in engine._active_jobs if j.document_id == "doc-a"}
            engine._active_jobs -= doc_a_jobs
            engine._active_documents.discard("doc-a")
            for key, event in list(job_events.items()):
                if key.startswith("doc-a:"):
                    event.set()
            await asyncio.sleep(0.05)

            # Phase 3: re-trigger doc B — claims freed slots
            task_b2 = asyncio.create_task(engine._find_and_start_jobs("doc-b"))
            await asyncio.sleep(0.05)

            inflight_b = sum(1 for j in engine._active_jobs if j.document_id == "doc-b")
            assert (
                inflight_b == 10
            ), f"After doc-a idle, doc-b should have 10 inflight, got {inflight_b}"

            # Clean up
            for event in job_events.values():
                event.set()
            await asyncio.sleep(0.05)
            for t in [task_a, task_b, task_b2]:
                t.cancel()
            await asyncio.gather(task_a, task_b, task_b2, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_scheduler_fills_pool_in_first_round(self) -> None:
        """Scheduler fills all slots in the first round with greedy scheduling.

        With max_parallelism=10 and 2 dirty docs, the scheduler iterates
        them sequentially. With greedy budgets, the pool is fully saturated
        after one round. Per-doc distribution depends on iteration order
        (set ordering), so we only assert pool saturation.
        """
        engine = _make_engine(max_parallelism=10)

        job_events: dict[str, asyncio.Event] = {}

        async def mock_run_job(job: EmbeddingJob) -> None:
            event = asyncio.Event()
            key = f"{job.document_id}:{job.leaf_id}"
            job_events[key] = event
            await event.wait()

        remaining_a = [
            EmbeddingJob(document_id="doc-a", leaf_id=f"leaf-{i}") for i in range(2)
        ]
        remaining_b = [
            EmbeddingJob(document_id="doc-b", leaf_id=f"leaf-{i}") for i in range(20)
        ]

        def mock_find_next_n_jobs(
            document_id: str,
            active_jobs: set[IndexingJob],
            ctx: object,
            max_jobs: int,
        ) -> list[EmbeddingJob]:
            remaining = remaining_a if document_id == "doc-a" else remaining_b
            found: list[EmbeddingJob] = []
            for job in list(remaining):
                if job not in active_jobs and len(found) < max_jobs:
                    found.append(job)
                    remaining.remove(job)
            return found

        engine._active_documents = {"doc-a", "doc-b"}
        engine._document_contexts = {
            "doc-a": MagicMock(cancelled=False, leaves_at_last_idle=0),
            "doc-b": MagicMock(cancelled=False, leaves_at_last_idle=0),
        }

        engine._dirty_documents = {"doc-a", "doc-b"}

        with (
            patch.object(
                engine, "_find_next_n_jobs", side_effect=mock_find_next_n_jobs
            ),
            patch.object(engine, "_run_job", side_effect=mock_run_job),
            patch.object(engine, "_maybe_complete_runs", return_value=None),
            patch.object(engine, "_safe_on_document_idle", return_value=None),
        ):
            await engine._run_scheduler()

            total = len(engine._active_jobs)
            assert total == 10, f"All 10 slots should be filled, got {total}"

            # Clean up
            for event in job_events.values():
                event.set()
            await asyncio.sleep(0.05)
