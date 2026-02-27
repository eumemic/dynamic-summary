"""Tests for fair job distribution across documents in IndexingEngine.

Verifies that documents get fair access to the parallelism pool without
hard caps that waste slots. Documents that can't use their share (due to
tree dependencies) should have their unused capacity redistributed.
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


class TestFairScheduling:
    """Test fair distribution of parallelism slots across documents."""

    @pytest.mark.asyncio
    async def test_two_docs_get_fair_share(self) -> None:
        """With max_parallelism=10 and 2 active docs, neither should hold >5 inflight."""
        engine = _make_engine(max_parallelism=10)

        # Track peak inflight per document
        peak_inflight: dict[str, int] = {"doc-a": 0, "doc-b": 0}
        jobs_started: dict[str, int] = {"doc-a": 0, "doc-b": 0}
        job_complete_events: list[asyncio.Event] = []

        async def mock_run_job(job: EmbeddingJob) -> None:
            """Mock job that waits until signaled to complete."""
            doc_id = job.document_id
            jobs_started[doc_id] = jobs_started.get(doc_id, 0) + 1

            # Record peak inflight
            inflight = sum(1 for j in engine._active_jobs if j.document_id == doc_id)
            peak_inflight[doc_id] = max(peak_inflight[doc_id], inflight)

            # Wait for completion signal
            event = asyncio.Event()
            job_complete_events.append(event)
            await event.wait()

        # Each doc has 20 embedding jobs (far more than max_parallelism)
        doc_a_jobs = [
            EmbeddingJob(document_id="doc-a", leaf_id=f"leaf-{i}") for i in range(20)
        ]
        doc_b_jobs = [
            EmbeddingJob(document_id="doc-b", leaf_id=f"leaf-{i}") for i in range(20)
        ]
        remaining_a = list(doc_a_jobs)
        remaining_b = list(doc_b_jobs)

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

        # Both documents are active
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

            # Let scheduling run
            await asyncio.sleep(0.05)

            # Check: neither document should exceed fair share (5 slots each)
            assert (
                peak_inflight["doc-a"] <= 5
            ), f"doc-a peaked at {peak_inflight['doc-a']} inflight, expected <= 5"
            assert (
                peak_inflight["doc-b"] <= 5
            ), f"doc-b peaked at {peak_inflight['doc-b']} inflight, expected <= 5"

            # Clean up: signal all jobs to complete
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
    async def test_unused_slots_redistribute(self) -> None:
        """If doc A has only 2 jobs, doc B eventually holds up to 10 after A goes idle.

        The flow mirrors the real scheduler:
        1. Both docs get fair share (5 each), doc A only uses 2
        2. Doc A's jobs complete → goes idle → removed from _active_documents
        3. Doc B re-schedules → fair_share becomes 10 → claims remaining slots
        """
        engine = _make_engine(max_parallelism=10)

        job_events: dict[str, asyncio.Event] = {}

        async def mock_run_job(job: EmbeddingJob) -> None:
            event = asyncio.Event()
            key = f"{job.document_id}:{job.leaf_id}"
            job_events[key] = event
            await event.wait()

        # Doc A: only 2 jobs. Doc B: many jobs.
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
            # Phase 1: initial scheduling — each doc gets fair share (5)
            task_a = asyncio.create_task(engine._find_and_start_jobs("doc-a"))
            task_b = asyncio.create_task(engine._find_and_start_jobs("doc-b"))
            await asyncio.sleep(0.05)

            inflight_a = sum(1 for j in engine._active_jobs if j.document_id == "doc-a")
            inflight_b = sum(1 for j in engine._active_jobs if j.document_id == "doc-b")
            assert inflight_a == 2, f"doc-a should have 2 inflight, got {inflight_a}"
            assert inflight_b == 5, f"doc-b should have 5 inflight, got {inflight_b}"

            # Phase 2: simulate doc A jobs completing (remove from active_jobs)
            # and doc A going idle (remove from _active_documents)
            doc_a_jobs = {j for j in engine._active_jobs if j.document_id == "doc-a"}
            engine._active_jobs -= doc_a_jobs
            engine._active_documents.discard("doc-a")
            for key, event in list(job_events.items()):
                if key.startswith("doc-a:"):
                    event.set()
            await asyncio.sleep(0.05)

            # Phase 3: re-trigger doc B scheduling — fair share now 10 (sole active doc)
            task_b2 = asyncio.create_task(engine._find_and_start_jobs("doc-b"))
            await asyncio.sleep(0.05)

            inflight_b_after = sum(
                1 for j in engine._active_jobs if j.document_id == "doc-b"
            )
            assert (
                inflight_b_after == 10
            ), f"After doc-a idle, doc-b should have 10 inflight, got {inflight_b_after}"

            # Clean up
            for event in job_events.values():
                event.set()
            await asyncio.sleep(0.05)
            for t in [task_a, task_b, task_b2]:
                t.cancel()
            await asyncio.gather(task_a, task_b, task_b2, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_idle_doc_frees_share(self) -> None:
        """When doc A finishes all work, doc B's fair share grows to full parallelism.

        Unlike test_unused_slots_redistribute which verifies the end-state,
        this test verifies the transition: doc B's inflight count increases
        from its initial fair share after doc A goes idle.
        """
        engine = _make_engine(max_parallelism=10)

        job_events: dict[str, asyncio.Event] = {}

        async def mock_run_job(job: EmbeddingJob) -> None:
            event = asyncio.Event()
            key = f"{job.document_id}:{job.leaf_id}"
            job_events[key] = event
            await event.wait()

        # Doc A: 3 jobs, Doc B: 20 jobs
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
            # Phase 1: both docs scheduling — fair share = ceil(10/2) = 5
            task_a = asyncio.create_task(engine._find_and_start_jobs("doc-a"))
            task_b = asyncio.create_task(engine._find_and_start_jobs("doc-b"))
            await asyncio.sleep(0.05)

            inflight_a = sum(1 for j in engine._active_jobs if j.document_id == "doc-a")
            inflight_b_initial = sum(
                1 for j in engine._active_jobs if j.document_id == "doc-b"
            )
            assert inflight_a == 3, f"doc-a should use all 3 jobs, got {inflight_a}"
            assert (
                inflight_b_initial == 5
            ), f"doc-b fair share should be 5, got {inflight_b_initial}"

            # Phase 2: simulate doc A completing and going idle
            doc_a_jobs = {j for j in engine._active_jobs if j.document_id == "doc-a"}
            engine._active_jobs -= doc_a_jobs
            engine._active_documents.discard("doc-a")
            for key, event in list(job_events.items()):
                if key.startswith("doc-a:"):
                    event.set()
            await asyncio.sleep(0.05)

            # Phase 3: re-trigger doc B — now sole active doc, gets full 10
            task_b2 = asyncio.create_task(engine._find_and_start_jobs("doc-b"))
            await asyncio.sleep(0.05)

            inflight_b_after = sum(
                1 for j in engine._active_jobs if j.document_id == "doc-b"
            )
            assert (
                inflight_b_after == 10
            ), f"After doc-a idle, doc-b should have 10 inflight, got {inflight_b_after}"
            # Confirm growth from initial fair share
            assert inflight_b_after > inflight_b_initial

            # Clean up
            for event in job_events.values():
                event.set()
            await asyncio.sleep(0.05)
            for t in [task_a, task_b, task_b2]:
                t.cancel()
            await asyncio.gather(task_a, task_b, task_b2, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_scheduler_redistributes_unused_capacity(self) -> None:
        """Scheduler redistributes slots when some docs can't use their share.

        With max_parallelism=10, 2 active docs, doc A can only find 2 jobs
        (tree deps) but doc B has 20. The scheduler should redistribute
        A's unused slots to B, filling all 10 slots.
        """
        engine = _make_engine(max_parallelism=10)

        job_events: dict[str, asyncio.Event] = {}

        async def mock_run_job(job: EmbeddingJob) -> None:
            event = asyncio.Event()
            key = f"{job.document_id}:{job.leaf_id}"
            job_events[key] = event
            await event.wait()

        # Doc A: only 2 jobs (simulating tree dependency bottleneck)
        # Doc B: many jobs
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

        # Mark both documents as dirty (as _run_job would do)
        engine._dirty_documents = {"doc-a", "doc-b"}

        with (
            patch.object(
                engine, "_find_next_n_jobs", side_effect=mock_find_next_n_jobs
            ),
            patch.object(engine, "_run_job", side_effect=mock_run_job),
            patch.object(engine, "_maybe_complete_runs", return_value=None),
            patch.object(engine, "_safe_on_document_idle", return_value=None),
        ):
            # Run the scheduler — it should distribute fairly then redistribute
            await engine._run_scheduler()

            inflight_a = sum(1 for j in engine._active_jobs if j.document_id == "doc-a")
            inflight_b = sum(1 for j in engine._active_jobs if j.document_id == "doc-b")
            total = len(engine._active_jobs)

            assert inflight_a == 2, f"doc-a has 2 jobs total, got {inflight_a} inflight"
            assert total == 10, (
                f"All 10 slots should be filled (a={inflight_a}, b={inflight_b}, "
                f"total={total})"
            )
            assert (
                inflight_b == 8
            ), f"doc-b should absorb doc-a's unused slots, got {inflight_b}"

            # Clean up
            for event in job_events.values():
                event.set()
            await asyncio.sleep(0.05)
