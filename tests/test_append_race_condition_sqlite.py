"""Regression tests for concurrent append race condition.

These tests verify that concurrent append calls do NOT create
duplicate (height, level_index) coordinates.

Historical bug (fixed):
1. Request A reads rightmost leaf with level_index=N
2. Request B reads rightmost leaf with level_index=N (same leaf!)
3. Request A creates leaves starting at level_index=N+1
4. Request B creates leaves starting at level_index=N+1 (DUPLICATES!)

The fix: DocumentIndexSession (and grpc_servicer.py) hold locks through
batch_append, ensuring concurrent sessions are serialized when appending
to the same document.

These tests verify the storage-layer locking prevents concurrent access.
For full integration tests against the gRPC servicer with PostgreSQL,
see tests/memory_service/test_concurrent_ingest_duplicates.py.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from ragzoom.contracts.storage_backend import StorageBackend


@pytest.fixture
def storage_backend_for_locking(
    storage_backend: StorageBackend,
) -> StorageBackend:
    """Provide the storage backend from the standard fixture."""
    return storage_backend


def test_lock_document_rejects_concurrent_access(
    storage_backend_for_locking: StorageBackend,
) -> None:
    """Verify lock_document rejects concurrent access to the same document.

    The FileDocumentLock is non-blocking by design - it fails fast if the
    document is already being modified. This prevents race conditions by
    ensuring only one operation at a time can modify a document.

    This test verifies that:
    1. The first thread acquires the lock successfully
    2. The second thread gets rejected with RuntimeError
    """
    document_id = f"test-doc-{uuid.uuid4()}"
    store = storage_backend_for_locking

    # Track outcomes
    results: list[str] = []
    barrier = threading.Barrier(2)
    lock_acquired_event = threading.Event()

    def access_with_lock(thread_name: str, should_wait: bool) -> None:
        """Acquire lock, hold for a moment, record outcome."""
        barrier.wait(timeout=5.0)  # Synchronize start

        if should_wait:
            # Thread B waits for thread A to acquire lock first
            lock_acquired_event.wait(timeout=5.0)

        lock_cm = store.lock_document(document_id)
        try:
            with lock_cm:
                if not should_wait:
                    # Thread A signals it has the lock
                    lock_acquired_event.set()
                time.sleep(0.2)  # Hold lock briefly
                results.append(f"{thread_name}:success")
        except RuntimeError as e:
            if "currently being modified" in str(e):
                results.append(f"{thread_name}:rejected")
            else:
                raise

    # Run two threads concurrently trying to lock the same document
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Thread A will acquire lock first (doesn't wait)
        # Thread B waits for lock_acquired_event then tries to acquire
        future_a = executor.submit(access_with_lock, "A", False)
        future_b = executor.submit(access_with_lock, "B", True)
        future_a.result(timeout=10)
        future_b.result(timeout=10)

    # Verify outcomes
    assert len(results) == 2
    assert "A:success" in results, f"Thread A should succeed, got: {results}"
    assert "B:rejected" in results, f"Thread B should be rejected, got: {results}"


def test_lock_document_allows_different_documents(
    storage_backend_for_locking: StorageBackend,
) -> None:
    """Verify locks on different documents don't block each other.

    This ensures the locking is per-document, not global.
    """
    doc_a = f"test-doc-a-{uuid.uuid4()}"
    doc_b = f"test-doc-b-{uuid.uuid4()}"
    store = storage_backend_for_locking

    # Track when each thread holds the lock
    access_times: list[tuple[str, float, float]] = []
    barrier = threading.Barrier(2)

    def access_with_lock(doc_id: str, name: str) -> None:
        """Acquire lock, hold for a moment, record times."""
        barrier.wait(timeout=5.0)

        lock_cm = store.lock_document(doc_id)
        with lock_cm:
            start = time.perf_counter()
            time.sleep(0.1)
            end = time.perf_counter()
            access_times.append((name, start, end))

    # Run two threads locking different documents
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_a = executor.submit(access_with_lock, doc_a, "A")
        future_b = executor.submit(access_with_lock, doc_b, "B")
        future_a.result(timeout=10)
        future_b.result(timeout=10)

    # Verify both ran (different docs should allow concurrent access)
    assert len(access_times) == 2
    times_a = next(t for t in access_times if t[0] == "A")
    times_b = next(t for t in access_times if t[0] == "B")

    a_start, a_end = times_a[1], times_a[2]
    b_start, b_end = times_b[1], times_b[2]

    # They SHOULD overlap (or nearly overlap) since they're locking different docs
    # Note: this might not always overlap perfectly due to thread scheduling,
    # but at minimum they should complete in roughly the same time frame
    # (not 2x the single-lock time)
    total_time = max(a_end, b_end) - min(a_start, b_start)
    single_lock_time = 0.1  # How long each holds the lock

    # If truly serialized, total_time would be ~0.2s. If concurrent, ~0.1s
    # Allow some margin for thread scheduling
    assert total_time < 0.18, (
        f"Different documents should lock concurrently but took {total_time:.4f}s "
        f"(expected ~{single_lock_time}s if concurrent, ~{2*single_lock_time}s if serialized)"
    )


def test_lock_document_allows_sequential_access(
    storage_backend_for_locking: StorageBackend,
) -> None:
    """Verify sequential lock acquisition works correctly.

    After the first lock is released, a second acquisition should succeed.
    """
    document_id = f"test-doc-{uuid.uuid4()}"
    store = storage_backend_for_locking

    # First acquisition
    lock_cm1 = store.lock_document(document_id)
    with lock_cm1:
        pass  # Lock acquired and released

    # Second acquisition (should succeed now that first is released)
    lock_cm2 = store.lock_document(document_id)
    with lock_cm2:
        pass  # Lock acquired and released

    # If we get here without exception, the test passes
