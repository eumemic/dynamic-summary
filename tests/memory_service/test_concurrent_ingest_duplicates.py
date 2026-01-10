"""Regression test for concurrent IngestSession handling.

This test verifies that concurrent IngestSession calls on the same document
do NOT create duplicate tree coordinates (height, level_index).

Historical context: A bug was observed in production where concurrent calls
could create duplicate coordinates. The root cause was identified as a race
condition in Phase 2 (batch_append) after the advisory lock was released.
However, the current implementation correctly serializes concurrent calls
through pg_advisory_lock, preventing duplicates.

To run this test, you need the devstack running:
    ./scripts/devstack start

Then run with integration tests:
    RAGZOOM_DATABASE_URL="postgresql://ragzoom:ragzoom@localhost:5433/ragzoom" \
    pytest tests/memory_service/test_concurrent_ingest_duplicates.py -v --use-real-store
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Generator
from typing import TYPE_CHECKING

import grpc
import pytest
from sqlalchemy import create_engine, text

from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc

if TYPE_CHECKING:
    from ragzoom.backends.postgres_backend import PostgresStorageBackend


def _make_jsonl_message(msg_uuid: str, parent_uuid: str | None, content: str) -> bytes:
    """Create a JSONL message record."""
    record = {
        "uuid": msg_uuid,
        "parentUuid": parent_uuid,
        "type": "user",
        "message": {"content": content},
    }
    return json.dumps(record).encode() + b"\n"


def _make_transcript(
    num_messages: int, content_size: int = 100, starting_parent: str | None = None
) -> tuple[bytes, str]:
    """Create a transcript with multiple messages.

    Returns:
        Tuple of (transcript_bytes, last_message_uuid) so callers can chain transcripts.
    """
    lines = []
    parent = starting_parent
    last_uuid = ""
    for i in range(num_messages):
        msg_id = str(uuid.uuid4())
        content = f"Message {i}: " + "x" * content_size
        lines.append(_make_jsonl_message(msg_id, parent, content))
        parent = msg_id
        last_uuid = msg_id
    return b"".join(lines), last_uuid


@pytest.fixture
def devstack_db_url() -> str:
    """Get the devstack database URL, skipping if not available."""
    db_url = os.environ.get(
        "RAGZOOM_DATABASE_URL",
        "postgresql://ragzoom:ragzoom@localhost:5433/ragzoom",
    )

    # Skip if we can't connect to postgres
    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        pytest.skip(f"PostgreSQL devstack not available: {e}")

    return db_url


@pytest.fixture
def grpc_address() -> str:
    """Get the devstack gRPC address."""
    return os.environ.get("RAGZOOM_GRPC_ADDRESS", "localhost:50051")


@pytest.fixture
def postgres_backend(
    devstack_db_url: str,
) -> Generator[PostgresStorageBackend, None, None]:
    """Create a PostgreSQL backend for checking results.

    We avoid using OperationalConfig directly because it applies worktree
    isolation transformations to URLs that contain 'ragzoom'. Instead, we
    patch get_worktree_id to return None, which bypasses the transformation.
    """
    from unittest.mock import patch

    from ragzoom.backends.postgres_backend import PostgresStorageBackend
    from ragzoom.config import OperationalConfig

    # Bypass worktree URL transformation by making get_worktree_id return None
    with patch("ragzoom.worktree_utils.get_worktree_id", return_value=None):
        config = OperationalConfig(
            backend="postgres",
            database_url=devstack_db_url,
        )
    backend = PostgresStorageBackend(config)
    yield backend
    backend.close()


def _ingest_session_sync(
    stub: pb2_grpc.SessionIngestionServiceStub,
    session_id: str,
    jsonl_delta: bytes,
    user_id: str,
) -> pb2.IngestSessionResponse:
    """Make a synchronous IngestSession call."""
    request = pb2.IngestSessionRequest(
        session_id=session_id,
        jsonl_delta=jsonl_delta,
    )
    metadata = [("user_id", user_id)]
    return stub.IngestSession(request, metadata=metadata)


@pytest.mark.integration
@pytest.mark.slow_threshold(60)  # Integration test needs time for workers
def test_concurrent_ingest_no_duplicate_coordinates(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Concurrent IngestSession calls must NOT create duplicate coordinates.

    This regression test verifies that the advisory lock mechanism correctly
    serializes concurrent ingest calls, preventing duplicate tree coordinates.
    """
    import concurrent.futures
    import threading

    session_id = f"test-concurrent-{uuid.uuid4()}"
    user_id = "test-user"

    # Check if gRPC server is available
    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        # Quick test call
        grpc.channel_ready_future(channel).result(timeout=2)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        # Create initial content so there's a rightmost leaf to race on
        initial_transcript, last_uuid = _make_transcript(10, content_size=1000)
        _ingest_session_sync(stub, session_id, initial_transcript, user_id)

        # Give workers time to process initial content
        import time

        time.sleep(2)

        # Create two deltas that chain from the same parent
        # Note: When both deltas claim the same parent, the second one will detect
        # the first's append as a "revert" (parent mismatch) and truncate to re-sync.
        # This is EXPECTED behavior - you can't have two branches.
        # We verify that this process doesn't create duplicate coordinates.
        delta_a, last_uuid_a = _make_transcript(
            5, content_size=1000, starting_parent=last_uuid
        )
        delta_b, _ = _make_transcript(5, content_size=1000, starting_parent=last_uuid)

        # Use a barrier to synchronize the start
        barrier = threading.Barrier(2)
        results: list[pb2.IngestSessionResponse | Exception] = [None, None]  # type: ignore[list-item]

        def ingest_with_barrier(delta: bytes, result_idx: int) -> None:
            try:
                # Create a new channel for this thread
                thread_channel = grpc.insecure_channel(grpc_address)
                thread_stub = pb2_grpc.SessionIngestionServiceStub(thread_channel)

                # Wait for both threads to be ready
                barrier.wait()

                # Both fire simultaneously
                result = _ingest_session_sync(thread_stub, session_id, delta, user_id)
                results[result_idx] = result
                thread_channel.close()
            except Exception as e:
                results[result_idx] = e

        # Launch both in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_a = executor.submit(ingest_with_barrier, delta_a, 0)
            future_b = executor.submit(ingest_with_barrier, delta_b, 1)
            concurrent.futures.wait([future_a, future_b])

        # Give workers time to process
        time.sleep(3)

        # Check for duplicate coordinates
        doc_store = postgres_backend.for_document(session_id)
        all_leaves = list(doc_store.nodes.iter_leaves())

        coord_counts: dict[tuple[int, int], int] = {}
        for leaf in all_leaves:
            coord = (leaf.height, leaf.level_index)
            coord_counts[coord] = coord_counts.get(coord, 0) + 1

        duplicates = {
            coord: count for coord, count in coord_counts.items() if count > 1
        }

        # Verify no duplicate coordinates (regression test)
        assert not duplicates, (
            f"Concurrent ingest created duplicate coordinates: {duplicates}. "
            f"Total leaves: {len(all_leaves)}, "
            f"Coords: {sorted(coord_counts.keys())}"
        )

        # Verify we got a reasonable number of leaves.
        # Due to revert detection (when both deltas claim same parent, one wins),
        # the exact count varies by timing. We expect at least 10 (initial) + 5 (one delta).
        assert len(all_leaves) >= 15, (
            f"Expected at least 15 leaves (10 initial + 5 from one delta), "
            f"got {len(all_leaves)}"
        )

    finally:
        channel.close()
        # Cleanup: delete the test document's nodes
        try:
            doc_store = postgres_backend.for_document(session_id)
            # Use SQL to delete directly
            engine = create_engine(devstack_db_url)
            with engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM tree_nodes WHERE document_id = :doc_id"),
                    {"doc_id": session_id},
                )
                conn.execute(
                    text("DELETE FROM documents WHERE id = :doc_id"),
                    {"doc_id": session_id},
                )
                conn.commit()
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow_threshold(60)  # Integration test needs time for workers
def test_sequential_ingest_no_duplicates(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Sequential IngestSession calls should never create duplicate coordinates.

    This is the baseline test showing that sequential operation is correct.
    """
    import time

    session_id = f"test-sequential-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=2)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        # Sequential ingests with delays between them
        last_uuid: str | None = None
        for i in range(3):
            transcript, last_uuid = _make_transcript(
                5, content_size=1000, starting_parent=last_uuid
            )
            _ingest_session_sync(stub, session_id, transcript, user_id)
            time.sleep(2)  # Wait for workers to complete

        # Final wait for any remaining work
        time.sleep(2)

        # Check for duplicate coordinates
        doc_store = postgres_backend.for_document(session_id)
        all_leaves = list(doc_store.nodes.iter_leaves())

        coord_counts: dict[tuple[int, int], int] = {}
        for leaf in all_leaves:
            coord = (leaf.height, leaf.level_index)
            coord_counts[coord] = coord_counts.get(coord, 0) + 1

        duplicates = {
            coord: count for coord, count in coord_counts.items() if count > 1
        }

        assert (
            not duplicates
        ), f"Sequential ingest should not create duplicates: {duplicates}"

    finally:
        channel.close()
        try:
            engine = create_engine(devstack_db_url)
            with engine.connect() as conn:
                conn.execute(
                    text("DELETE FROM tree_nodes WHERE document_id = :doc_id"),
                    {"doc_id": session_id},
                )
                conn.execute(
                    text("DELETE FROM documents WHERE id = :doc_id"),
                    {"doc_id": session_id},
                )
                conn.commit()
        except Exception:
            pass
