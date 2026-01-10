"""Regression test for progressive append during active indexing.

This test simulates realistic Claude Code usage where new transcript data
arrives every 5-10 seconds while background indexing (embeddings + summaries)
is still in-flight from previous appends.

Historical context: Production usage revealed issues with concurrent indexing
jobs racing against new appends, potentially causing duplicate coordinates,
span ordering issues, or orphaned nodes.

To run this test, you need the devstack running:
    ./scripts/devstack start

Then run with integration tests:
    RAGZOOM_DATABASE_URL="postgresql://ragzoom:ragzoom@localhost:5433/ragzoom" \
    pytest tests/memory_service/test_progressive_append.py -xvs
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from collections.abc import Generator
from typing import TYPE_CHECKING

import grpc
import pytest
from sqlalchemy import create_engine, text

from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
from ragzoom.validation.tree import ValidationReport, validate_document

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


def _make_transcript_with_uuids(
    num_messages: int, content_size: int = 100, starting_parent: str | None = None
) -> tuple[bytes, str, list[str]]:
    """Create a transcript and return all generated UUIDs.

    Returns:
        Tuple of (transcript_bytes, last_message_uuid, all_uuids_in_order)
        so callers can revert to specific points.
    """
    lines = []
    all_uuids: list[str] = []
    parent = starting_parent
    last_uuid = ""
    for i in range(num_messages):
        msg_id = str(uuid.uuid4())
        content = f"Message {i}: " + "x" * content_size
        lines.append(_make_jsonl_message(msg_id, parent, content))
        all_uuids.append(msg_id)
        parent = msg_id
        last_uuid = msg_id
    return b"".join(lines), last_uuid, all_uuids


def _make_revert_transcript(
    revert_to_uuid: str,
    num_messages: int,
    content_size: int = 100,
    branch_marker: str = "BRANCH",
) -> tuple[bytes, str, list[str]]:
    """Create a transcript that branches from a specific UUID (simulates revert).

    This simulates a user reverting their conversation and continuing
    from an earlier point with different content.

    Returns:
        Tuple of (transcript_bytes, last_message_uuid, all_uuids_in_order)
    """
    lines = []
    all_uuids: list[str] = []
    parent = revert_to_uuid  # Start from the revert point
    last_uuid = ""
    for i in range(num_messages):
        msg_id = str(uuid.uuid4())
        # Use different content to distinguish from original branch
        content = f"{branch_marker} Message {i}: " + "y" * content_size
        lines.append(_make_jsonl_message(msg_id, parent, content))
        all_uuids.append(msg_id)
        parent = msg_id
        last_uuid = msg_id
    return b"".join(lines), last_uuid, all_uuids


def _make_assistant_message(
    msg_uuid: str, parent_uuid: str | None, content: str
) -> bytes:
    """Create a JSONL assistant message record."""
    record = {
        "uuid": msg_uuid,
        "parentUuid": parent_uuid,
        "type": "assistant",
        "message": {"content": content},
    }
    return json.dumps(record).encode() + b"\n"


def _make_multiturn_transcript(
    num_turns: int,
    content_size: int = 100,
    starting_parent: str | None = None,
) -> tuple[bytes, str, list[str], list[str]]:
    """Create a realistic multi-turn transcript with user/assistant alternation.

    Each turn = one user message + one assistant message.
    This creates proper segment boundaries (new segment starts at each user message
    following an assistant message).

    Returns:
        Tuple of (transcript_bytes, last_uuid, all_uuids, user_uuids_only)
        - all_uuids: all message UUIDs in order (user + assistant)
        - user_uuids_only: just the user message UUIDs (useful for revert targets)
    """
    lines = []
    all_uuids: list[str] = []
    user_uuids: list[str] = []
    parent = starting_parent
    last_uuid = ""

    for i in range(num_turns):
        # User message
        user_id = str(uuid.uuid4())
        user_content = f"User turn {i}: " + "u" * content_size
        lines.append(_make_jsonl_message(user_id, parent, user_content))
        all_uuids.append(user_id)
        user_uuids.append(user_id)
        parent = user_id

        # Assistant message
        assistant_id = str(uuid.uuid4())
        assistant_content = f"Assistant turn {i}: " + "a" * content_size
        lines.append(_make_assistant_message(assistant_id, parent, assistant_content))
        all_uuids.append(assistant_id)
        parent = assistant_id
        last_uuid = assistant_id

    return b"".join(lines), last_uuid, all_uuids, user_uuids


def _make_multiturn_revert_transcript(
    revert_to_uuid: str,
    num_turns: int,
    content_size: int = 100,
    branch_marker: str = "BRANCH",
) -> tuple[bytes, str, list[str], list[str]]:
    """Create a multi-turn transcript that branches from a specific UUID.

    Returns:
        Tuple of (transcript_bytes, last_uuid, all_uuids, user_uuids_only)
    """
    lines = []
    all_uuids: list[str] = []
    user_uuids: list[str] = []
    parent = revert_to_uuid
    last_uuid = ""

    for i in range(num_turns):
        # User message
        user_id = str(uuid.uuid4())
        user_content = f"{branch_marker} User turn {i}: " + "u" * content_size
        lines.append(_make_jsonl_message(user_id, parent, user_content))
        all_uuids.append(user_id)
        user_uuids.append(user_id)
        parent = user_id

        # Assistant message
        assistant_id = str(uuid.uuid4())
        assistant_content = f"{branch_marker} Assistant turn {i}: " + "a" * content_size
        lines.append(_make_assistant_message(assistant_id, parent, assistant_content))
        all_uuids.append(assistant_id)
        parent = assistant_id
        last_uuid = assistant_id

    return b"".join(lines), last_uuid, all_uuids, user_uuids


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
    return stub.IngestSession(request, metadata=metadata, timeout=300)


def _validate_tree_invariants(
    backend: PostgresStorageBackend,
    session_id: str,
    allow_incomplete: bool,
) -> ValidationReport:
    """Run comprehensive tree validation using existing ragzoom validators.

    Args:
        backend: Storage backend
        session_id: Document ID to validate
        allow_incomplete: If True, skip completeness checks (for mid-indexing validation)

    Returns:
        ValidationReport for inspection

    Raises:
        AssertionError: If critical invariants are violated
    """
    report: ValidationReport = validate_document(
        document_id=session_id,
        store=backend,
        require_complete=not allow_incomplete,
    )

    # These invariants must ALWAYS hold, even during indexing
    critical_codes = {
        "level_neighbors.duplicate_level_index",  # Duplicate coordinates
        "parent.missing",  # Orphaned nodes
        "parent.mismatch",  # Parent-child mismatch
        "leaf.gap",  # Gap in leaf spans
        "leaf.span_start",  # Leaf doesn't start at 0
        "leaf.span_end",  # Leaf span invalid
        "tree.left_only",  # Missing right child
        "tree.right_only",  # Missing left child
        "coord.leaf_height",  # Leaf with non-zero height
        "coord.height_mismatch",  # Parent height not child+1
        "coord.level_index_mismatch",  # Wrong level_index
        "span.union_mismatch",  # Parent span != union of children
    }

    # Completeness codes - only errors when require_complete=True
    completeness_codes = {
        "forest.unpaired_siblings",
        "forest.missing_embedding",
    }

    critical_errors = []
    for finding in report.findings:
        if finding.severity != "error":
            continue
        if finding.code in critical_codes:
            critical_errors.append(f"{finding.code}: {finding.message}")
        elif finding.code in completeness_codes and not allow_incomplete:
            critical_errors.append(f"{finding.code}: {finding.message}")

    if critical_errors:
        raise AssertionError(
            f"Tree invariant violations for {session_id}:\n"
            + "\n".join(f"  - {e}" for e in critical_errors)
        )

    return report


def _get_indexing_progress(db_url: str, session_id: str) -> dict[str, int]:
    """Return current indexing progress."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
            SELECT
                COUNT(*) FILTER (WHERE height = 0) as leaves,
                COUNT(*) FILTER (WHERE height = 0 AND embedding IS NOT NULL) as embedded,
                COUNT(*) FILTER (WHERE height > 0) as summaries
            FROM tree_nodes WHERE document_id = :doc_id
        """
            ),
            {"doc_id": session_id},
        ).one()
    return {
        "leaves": result.leaves,
        "embedded": result.embedded,
        "summaries": result.summaries,
    }


def _count_unpaired_siblings(db_url: str, session_id: str) -> int:
    """Count sibling pairs at the same height that could be merged but haven't been."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        # For a complete forest, nodes at the same height should either:
        # 1. Both have a parent (already merged)
        # 2. Be at different level_index parity (not adjacent siblings)
        # Adjacent siblings (even/odd consecutive level_index) without parents = incomplete
        result = conn.execute(
            text(
                """
            WITH nodes AS (
                SELECT height, level_index, parent_id
                FROM tree_nodes
                WHERE document_id = :doc_id
            )
            SELECT COUNT(*) as unpaired
            FROM nodes n1
            JOIN nodes n2 ON n1.height = n2.height
                AND n1.level_index = n2.level_index - 1
                AND n1.level_index % 2 = 0
            WHERE n1.parent_id IS NULL AND n2.parent_id IS NULL
        """
            ),
            {"doc_id": session_id},
        ).scalar()
    return result or 0


def _wait_for_indexing_complete(
    db_url: str, session_id: str, timeout: int, check_interval: float = 2.0
) -> None:
    """Poll until indexing is complete (all leaves embedded AND tree fully merged)."""
    start = time.time()
    last_status: str | None = None
    while time.time() - start < timeout:
        progress = _get_indexing_progress(db_url, session_id)
        unpaired = _count_unpaired_siblings(db_url, session_id)

        status = (
            f"leaves={progress['leaves']}, embedded={progress['embedded']}, "
            f"summaries={progress['summaries']}, unpaired={unpaired}"
        )

        # Complete when all leaves have embeddings AND no mergeable pairs remain
        if (
            progress["embedded"] == progress["leaves"]
            and progress["leaves"] > 0
            and unpaired == 0
        ):
            print(f"    Complete: {status}")
            return

        if status != last_status:
            print(f"    Waiting: {status}")
            last_status = status

        time.sleep(check_interval)

    raise TimeoutError(f"Indexing not complete after {timeout}s: {status}")


def _cleanup_test_session(db_url: str, session_id: str) -> None:
    """Delete test session data from database."""
    engine = create_engine(db_url)
    with engine.connect() as conn:
        conn.execute(
            text("DELETE FROM tree_nodes WHERE document_id = :doc_id"),
            {"doc_id": session_id},
        )
        conn.execute(
            text("DELETE FROM documents WHERE id = :doc_id"),
            {"doc_id": session_id},
        )
        conn.execute(
            text("DELETE FROM session_raw_data WHERE session_id = :session_id"),
            {"session_id": session_id},
        )
        conn.commit()


@pytest.mark.integration
@pytest.mark.slow_threshold(180)  # Long-running test
def test_progressive_append_during_active_indexing(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Progressive appends while indexing jobs are in-flight.

    Simulates realistic Claude Code usage where new transcript data
    arrives every 5-10 seconds while background indexing is ongoing.
    """
    session_id = f"repro-progressive-{uuid.uuid4()}"
    user_id = "test-user"

    # Check if gRPC server is available
    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        last_uuid: str | None = None

        # Phase 1: Large initial chunk (creates many indexing jobs)
        print(f"\n  Session: {session_id}")
        print("  Phase 1: Ingesting initial chunk (50 messages)...")
        chunk, last_uuid = _make_transcript(
            num_messages=50,  # ~10-15 leaves
            content_size=1000,
            starting_parent=last_uuid,
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    Initial: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Phase 2: Progressive appends at realistic intervals
        # Don't wait for completion - start appending immediately
        print("  Phase 2: Progressive appends (10 rounds, 5-10s intervals)...")

        for i in range(10):
            # Realistic delay between user interactions
            delay = random.uniform(5, 10)
            time.sleep(delay)

            chunk, last_uuid = _make_transcript(
                num_messages=20,  # ~4-6 leaves per chunk
                content_size=1000,
                starting_parent=last_uuid,
            )
            _ingest_session_sync(stub, session_id, chunk, user_id)

            # Validate tree invariants DURING indexing (allow incomplete)
            report = _validate_tree_invariants(
                postgres_backend, session_id, allow_incomplete=True
            )

            # Log progress for debugging
            progress = _get_indexing_progress(devstack_db_url, session_id)
            print(
                f"    Append {i + 1}/10: leaves={progress['leaves']}, "
                f"embedded={progress['embedded']}, summaries={progress['summaries']}, "
                f"validation={report.status}"
            )

        # Phase 3: Wait for completion and final validation
        print("  Phase 3: Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=180)

        # Final full validation (tree must be complete)
        print("  Phase 4: Final validation (require_complete=True)...")
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        print(
            f"    Final: nodes={final_report.metrics.get('node_count')}, "
            f"leaves={final_report.metrics.get('leaf_count')}, "
            f"status={final_report.status}"
        )

        # Additional sanity checks
        progress = _get_indexing_progress(devstack_db_url, session_id)

        # We expect at least 50 + 10*20 = 250 messages, which at ~5 msg/leaf = ~50 leaves
        assert (
            progress["leaves"] >= 40
        ), f"Expected at least 40 leaves from 250 messages, got {progress['leaves']}"

        # All leaves should have embeddings
        assert (
            progress["embedded"] == progress["leaves"]
        ), f"Not all leaves have embeddings: {progress['embedded']}/{progress['leaves']}"

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass  # Best-effort cleanup


@pytest.mark.integration
@pytest.mark.slow_threshold(120)
def test_rapid_fire_appends_no_delay(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Rapid-fire appends with minimal delay to stress-test race conditions.

    This is a more aggressive version that fires appends as fast as possible,
    maximizing the chance of hitting race conditions between append and indexing.
    """
    session_id = f"repro-rapid-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        last_uuid: str | None = None
        print(f"\n  Session: {session_id}")

        # Fire 20 rapid appends with minimal delay
        print("  Rapid-fire: 20 appends with 0.5s delay...")
        for i in range(20):
            chunk, last_uuid = _make_transcript(
                num_messages=15,
                content_size=500,
                starting_parent=last_uuid,
            )
            _ingest_session_sync(stub, session_id, chunk, user_id)
            time.sleep(0.5)  # Minimal delay

            # Validate after each append
            _validate_tree_invariants(
                postgres_backend, session_id, allow_incomplete=True
            )

            if (i + 1) % 5 == 0:
                progress = _get_indexing_progress(devstack_db_url, session_id)
                print(
                    f"    Append {i + 1}/20: leaves={progress['leaves']}, "
                    f"embedded={progress['embedded']}"
                )

        # Wait for completion
        print("  Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=120)

        # Final validation
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        print(f"    Final: status={final_report.status}")
        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass


# =============================================================================
# REVERT TESTS - Simulate user reverting conversation while indexing in-flight
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow_threshold(180)
def test_revert_while_indexing_in_flight(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Revert conversation while summary jobs are still running.

    This tests the most dangerous race condition: truncating the tree while
    background indexing jobs are in-flight, potentially creating orphaned
    nodes or duplicate coordinates.

    Timeline:
      t=0:   Ingest 50 messages (creates ~15 leaves, many summary jobs queued)
      t=2:   While summaries still running, ingest REVERT to msg 25
      t=5:   Validate tree invariants
      t=60:  Wait for completion, final validation
    """
    session_id = f"repro-revert-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        print(f"\n  Session: {session_id}")

        # Phase 1: Initial large ingest (creates many indexing jobs)
        print("  Phase 1: Ingesting initial 50 messages...")
        chunk, last_uuid, all_uuids = _make_transcript_with_uuids(
            num_messages=50, content_size=1000
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    After initial: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # DON'T wait - let indexing run in background
        # Brief pause to let some jobs start but not complete
        time.sleep(2)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    Before revert: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Phase 2: Revert to midpoint while jobs are in-flight
        revert_index = 25
        revert_point = all_uuids[revert_index]
        print(
            f"  Phase 2: REVERTING to message {revert_index} "
            f"(uuid={revert_point[:8]}...)"
        )

        revert_chunk, new_last, new_uuids = _make_revert_transcript(
            revert_to_uuid=revert_point,
            num_messages=20,
            content_size=1000,
            branch_marker="REVERT_A",
        )
        _ingest_session_sync(stub, session_id, revert_chunk, user_id)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    After revert: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Validate immediately after revert (critical check)
        print("  Phase 3: Validating tree invariants after revert...")
        report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=True
        )
        print(f"    Validation: {report.status}")

        # Wait for completion
        print("  Phase 4: Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=180)

        # Final full validation
        print("  Phase 5: Final validation (require_complete=True)...")
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        print(
            f"    Final: nodes={final_report.metrics.get('node_count')}, "
            f"leaves={final_report.metrics.get('leaf_count')}, "
            f"status={final_report.status}"
        )

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow_threshold(180)
def test_rapid_append_revert_cycles(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Rapid append-revert-append cycles to stress-test tree mutation.

    Timeline:
      t=0:   Ingest initial 30 messages
      t=2:   Append 10 more (msg31-40)
      t=3:   REVERT to msg20, append new branch
      t=4:   Append more to new branch
      t=5:   REVERT again to msg15 of new branch
      Validate after each operation
    """
    session_id = f"repro-cycles-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        print(f"\n  Session: {session_id}")

        # Phase 1: Initial ingest
        print("  Phase 1: Ingesting initial 30 messages...")
        chunk, last_uuid, uuids_v1 = _make_transcript_with_uuids(
            num_messages=30, content_size=800
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)
        time.sleep(1)

        # Phase 2: Append more
        print("  Phase 2: Appending 10 more messages...")
        chunk, last_uuid, uuids_v1_ext = _make_transcript_with_uuids(
            num_messages=10, content_size=800, starting_parent=last_uuid
        )
        uuids_v1.extend(uuids_v1_ext)
        _ingest_session_sync(stub, session_id, chunk, user_id)
        time.sleep(1)

        _validate_tree_invariants(postgres_backend, session_id, allow_incomplete=True)
        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(f"    After append: leaves={progress['leaves']}")

        # Phase 3: REVERT to msg 20
        revert_point = uuids_v1[19]  # 0-indexed, so this is message 20
        print("  Phase 3: REVERTING to message 20...")
        chunk, last_uuid, uuids_v2 = _make_revert_transcript(
            revert_to_uuid=revert_point,
            num_messages=15,
            content_size=800,
            branch_marker="BRANCH_V2",
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)
        time.sleep(1)

        _validate_tree_invariants(postgres_backend, session_id, allow_incomplete=True)
        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(f"    After revert 1: leaves={progress['leaves']}")

        # Phase 4: Append more to new branch
        print("  Phase 4: Appending 10 more to new branch...")
        chunk, last_uuid, uuids_v2_ext = _make_transcript_with_uuids(
            num_messages=10, content_size=800, starting_parent=last_uuid
        )
        uuids_v2.extend(uuids_v2_ext)
        _ingest_session_sync(stub, session_id, chunk, user_id)
        time.sleep(1)

        _validate_tree_invariants(postgres_backend, session_id, allow_incomplete=True)
        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(f"    After append 2: leaves={progress['leaves']}")

        # Phase 5: REVERT again within the new branch
        revert_point_v2 = uuids_v2[10]  # Revert to message 11 of v2 branch
        print("  Phase 5: REVERTING again within new branch...")
        chunk, last_uuid, uuids_v3 = _make_revert_transcript(
            revert_to_uuid=revert_point_v2,
            num_messages=10,
            content_size=800,
            branch_marker="BRANCH_V3",
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)

        _validate_tree_invariants(postgres_backend, session_id, allow_incomplete=True)
        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(f"    After revert 2: leaves={progress['leaves']}")

        # Wait for completion
        print("  Phase 6: Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=180)

        # Final validation
        print("  Phase 7: Final validation...")
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        print(f"    Final: status={final_report.status}")

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow_threshold(180)
def test_revert_to_very_early_point(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Revert to near the beginning after building a large tree.

    This stress tests the truncation logic by deleting almost the entire tree.

    Timeline:
      t=0:   Ingest 100 messages (~30 leaves, deep tree)
      t=5:   Wait for partial indexing (~50% complete)
      t=6:   REVERT to msg 5 (deletes almost everything)
      t=7:   Append new branch
      Validate: No orphaned nodes, no dangling references
    """
    session_id = f"repro-early-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        print(f"\n  Session: {session_id}")

        # Phase 1: Build a large tree
        print("  Phase 1: Ingesting 100 messages (large tree)...")
        chunk, last_uuid, all_uuids = _make_transcript_with_uuids(
            num_messages=100, content_size=800
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    After initial: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Phase 2: Wait for partial indexing (let tree build up)
        print("  Phase 2: Waiting for partial indexing...")
        time.sleep(5)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    Before revert: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Phase 3: REVERT to very early point (message 5)
        revert_index = 4  # 0-indexed
        revert_point = all_uuids[revert_index]
        print(
            f"  Phase 3: REVERTING to message {revert_index + 1} "
            "(deletes ~95% of tree)..."
        )

        revert_chunk, new_last, new_uuids = _make_revert_transcript(
            revert_to_uuid=revert_point,
            num_messages=30,
            content_size=800,
            branch_marker="EARLY_REVERT",
        )
        _ingest_session_sync(stub, session_id, revert_chunk, user_id)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    After revert: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Validate immediately (this is when orphaned nodes would appear)
        print("  Phase 4: Validating after massive truncation...")
        report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=True
        )
        print(f"    Validation: {report.status}")

        # Wait for completion
        print("  Phase 5: Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=180)

        # Final validation
        print("  Phase 6: Final validation...")
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        print(
            f"    Final: nodes={final_report.metrics.get('node_count')}, "
            f"leaves={final_report.metrics.get('leaf_count')}, "
            f"status={final_report.status}"
        )

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass


def _get_append_entry_count(db_url: str, session_id: str) -> int:
    """Count append log entries for a session.

    Uses a join through session_raw_data since append entries use a FK.
    """
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT COUNT(*)
                FROM session_append_entries sae
                JOIN session_raw_data srd ON sae.session_raw_data_id = srd.id
                WHERE srd.session_id = :sid
            """
            ),
            {"sid": session_id},
        ).scalar()
    return result or 0


def _get_all_leaf_content(
    postgres_backend: PostgresStorageBackend, session_id: str
) -> str:
    """Get concatenated content of all leaves in span order."""
    doc_store = postgres_backend.for_document(session_id)
    leaves = sorted(doc_store.nodes.iter_leaves(), key=lambda n: n.span_start)
    return "".join(leaf.text for leaf in leaves)


def _verify_content_contains_markers(
    content: str, expected_markers: list[str], marker_name: str
) -> None:
    """Verify content contains all expected markers.

    Args:
        content: The full leaf content to search
        expected_markers: List of strings that must appear in content
        marker_name: Description of what we're checking (for error messages)
    """
    missing = [m for m in expected_markers if m not in content]
    if missing:
        # Show first few missing for debugging
        sample = missing[:3]
        raise AssertionError(
            f"Bug 1 regression: {len(missing)} {marker_name} missing from indexed content. "
            f"First few missing: {sample}. "
            f"This indicates the ancestor chain was incomplete during re-indexing."
        )


@pytest.mark.integration
@pytest.mark.slow_threshold(120)
def test_full_revert_multiturn(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Full revert requiring complete ancestor chain reconstruction.

    This tests Bug 1: the incomplete parent_of map issue where
    stream_find_common_ancestor stops too early.

    Scenario:
    - Create 10 turns (10 segments = 10 append log entries)
    - Revert to turn 0 (before any segments) with new content
    - This requires walking the full chain to root

    Expected: Tree should contain only the new content after revert.
    Bug behavior: May fail to build full chain, causing incorrect indexing.
    """
    session_id = f"repro-full-revert-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        print(f"\n  Session: {session_id}")

        # Phase 1: Create 10 turns (should create 10 append log entries)
        print("  Phase 1: Ingesting 10 turns...")
        chunk, last_uuid, all_uuids, user_uuids = _make_multiturn_transcript(
            num_turns=10, content_size=500
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)

        entry_count = _get_append_entry_count(devstack_db_url, session_id)
        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    After initial: append_entries={entry_count}, "
            f"leaves={progress['leaves']}"
        )

        # Verify we have multiple append log entries
        assert entry_count >= 5, (
            f"Expected multiple append log entries from multi-turn transcript, "
            f"got {entry_count}"
        )

        # Wait for initial indexing to complete
        print("  Phase 2: Waiting for initial indexing...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=120)

        initial_leaves = _get_indexing_progress(devstack_db_url, session_id)["leaves"]
        print(f"    Initial indexing complete: {initial_leaves} leaves")

        # Phase 3: Full revert - branch from the very first user message
        # This means the common ancestor is at the root, requiring full chain walk
        first_user_uuid = user_uuids[0]
        print(
            f"  Phase 3: FULL REVERT to first user message "
            f"(uuid={first_user_uuid[:8]}...)"
        )

        revert_chunk, new_last, new_all, new_users = _make_multiturn_revert_transcript(
            revert_to_uuid=first_user_uuid,
            num_turns=5,
            content_size=500,
            branch_marker="FULL_REVERT",
        )
        response = _ingest_session_sync(stub, session_id, revert_chunk, user_id)
        print(
            f"    Response: truncated={response.truncated}, "
            f"truncate_span={response.truncate_span}"
        )

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(f"    After revert: leaves={progress['leaves']}")

        # Validate immediately
        print("  Phase 4: Validating after full revert...")
        report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=True
        )
        print(f"    Validation: {report.status}")

        # Wait for completion
        print("  Phase 5: Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=120)

        # Final validation
        print("  Phase 6: Final validation...")
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        final_leaves = final_report.metrics.get("leaf_count", 0)
        print(
            f"    Final: nodes={final_report.metrics.get('node_count')}, "
            f"leaves={final_leaves}, status={final_report.status}"
        )

        # The final tree should have significantly fewer leaves than the initial
        # since we reverted most of the content
        # Initial: 10 turns, Final: 1 kept turn + 5 new turns = 6 turns worth
        # Allow some margin for chunking differences
        assert final_leaves < initial_leaves, (
            f"Expected fewer leaves after full revert. "
            f"Initial: {initial_leaves}, Final: {final_leaves}. "
            f"This may indicate truncation did not work correctly."
        )

        # Phase 7: Verify ALL expected content is indexed (Bug 1 regression check)
        # The indexed content should contain:
        # - Turn 0 user message from original (the revert point)
        # - All 5 turns from the revert branch (FULL_REVERT markers)
        print("  Phase 7: Verifying all expected content is indexed...")
        all_content = _get_all_leaf_content(postgres_backend, session_id)

        # Check for FULL_REVERT markers from all 5 new turns
        expected_markers = [f"FULL_REVERT User turn {i}" for i in range(5)]
        expected_markers += [f"FULL_REVERT Assistant turn {i}" for i in range(5)]
        _verify_content_contains_markers(all_content, expected_markers, "turn markers")

        # Also verify original turn 0 is present
        assert (
            "User turn 0" in all_content
        ), "Original turn 0 user message should be present (the revert point)"

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow_threshold(120)
@pytest.mark.xfail(
    reason="Bug 3 blocks this test: FK violation on partial revert. "
    "This test is intended to verify Bug 1 (incomplete parent_of map) "
    "but hits Bug 3 first because granular truncation is attempted. "
    "Revisit after Bug 3 is fixed.",
    strict=True,
)
def test_revert_to_recent_point_bug1(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Revert to a RECENT point to test Bug 1 (incomplete parent_of map).

    Bug 1 hypothesis: stream_find_common_ancestor_from_bytes stops scanning
    as soon as it finds the common ancestor. If the common ancestor is near
    the END of the file (recent), the parent_of map won't have entries for
    messages earlier in the conversation.

    When we then need to walk the full chain to root (for truncate_span=0),
    get_ancestor_chain silently stops at the first missing entry.

    Scenario:
    - Create 10 turns
    - Revert to turn 8 (near the end) with new content
    - The common ancestor (turn 8) is found quickly
    - But we need full chain back to turn 0 for re-indexing
    - If parent_of is incomplete, early turns will be missing from index

    NOTE: This test currently hits Bug 3 (FK violation) before it can
    test Bug 1, because the append entries allow granular truncation.
    """
    session_id = f"repro-bug1-recent-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        print(f"\n  Session: {session_id}")

        # Phase 1: Create 10 turns
        print("  Phase 1: Ingesting 10 turns...")
        chunk, last_uuid, all_uuids, user_uuids = _make_multiturn_transcript(
            num_turns=10, content_size=500
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)

        # Wait for initial indexing
        print("  Phase 2: Waiting for initial indexing...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=120)

        initial_leaves = _get_indexing_progress(devstack_db_url, session_id)["leaves"]
        print(f"    Initial indexing complete: {initial_leaves} leaves")

        # Phase 3: Revert to turn 8 (NEAR THE END - this is the key difference)
        # Turn 8's assistant message is at all_uuids[8*2 + 1] = all_uuids[17]
        revert_point = all_uuids[17]  # End of turn 8
        print(f"  Phase 3: REVERT to turn 8 (near end) (uuid={revert_point[:8]}...)")

        revert_chunk, new_last, new_all, new_users = _make_multiturn_revert_transcript(
            revert_to_uuid=revert_point,
            num_turns=3,  # Add 3 new turns
            content_size=500,
            branch_marker="RECENT_REVERT",
        )
        response = _ingest_session_sync(stub, session_id, revert_chunk, user_id)
        print(
            f"    Response: truncated={response.truncated}, "
            f"truncate_span={response.truncate_span}"
        )

        # Wait for indexing
        print("  Phase 4: Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=120)

        # Phase 5: Verify ALL expected content is indexed
        # Should have: turns 0-8 from original + 3 new turns from revert
        print("  Phase 5: Verifying all expected content is indexed...")
        all_content = _get_all_leaf_content(postgres_backend, session_id)

        # Check original turns 0-8 are present (the kept content)
        # These are the messages BEFORE the revert point
        original_markers = []
        for i in range(9):  # Turns 0-8
            original_markers.append(f"User turn {i}")
            original_markers.append(f"Assistant turn {i}")
        _verify_content_contains_markers(
            all_content, original_markers, "original turn markers (0-8)"
        )

        # Check new revert turns are present
        revert_markers = [f"RECENT_REVERT User turn {i}" for i in range(3)]
        revert_markers += [f"RECENT_REVERT Assistant turn {i}" for i in range(3)]
        _verify_content_contains_markers(
            all_content, revert_markers, "revert turn markers"
        )

        # Verify turn 9 is NOT present (it was on the discarded branch)
        assert (
            "User turn 9" not in all_content
        ), "Turn 9 should have been truncated (on discarded branch)"

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass


# =============================================================================
# RESET TESTS - Simulate admin reset while indexing in-flight
# =============================================================================


def _reset_session_cursor(db_url: str, session_id: str) -> None:
    """Reset a session's cursor to force full re-sync.

    This mimics `python -m memory_service.admin reset <session-id>`.
    """
    engine = create_engine(db_url)
    with engine.connect() as conn:
        # Clear the sync cursor but preserve span_end (like admin reset does)
        conn.execute(
            text(
                """
                UPDATE session_raw_data
                SET last_synced_uuid = NULL, original_file_offset = 0
                WHERE session_id = :session_id
            """
            ),
            {"session_id": session_id},
        )
        # Clear append entries
        conn.execute(
            text(
                """
                DELETE FROM session_append_entries
                WHERE session_raw_data_id IN (
                    SELECT id FROM session_raw_data WHERE session_id = :session_id
                )
            """
            ),
            {"session_id": session_id},
        )
        conn.commit()


def _check_for_duplicate_coords(db_url: str, session_id: str) -> list[tuple[int, int]]:
    """Check for duplicate (height, level_index) coordinates.

    Returns list of (height, level_index) pairs that have duplicates.
    """
    engine = create_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                """
                SELECT height, level_index, COUNT(*) as cnt
                FROM tree_nodes
                WHERE document_id = :doc_id
                GROUP BY height, level_index
                HAVING COUNT(*) > 1
            """
            ),
            {"doc_id": session_id},
        )
        return [(row.height, row.level_index) for row in result]


@pytest.mark.integration
@pytest.mark.slow_threshold(180)
def test_reset_and_reingest_during_active_indexing(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Reset session cursor and re-ingest while background jobs are in-flight.

    This tests the interaction between:
    1. Ingestion (creates leaves and queues embedding/summary jobs)
    2. Reset (clears sync cursor but NOT the tree nodes)
    3. Re-ingestion (triggers full re-index, should cancel in-flight jobs)

    Timeline:
      t=0:   Ingest 80 messages (creates ~20 leaves, many jobs queued)
      t=2:   While jobs still running, RESET the session cursor
      t=3:   Re-ingest the SAME content (triggers full re-index from span=0)
      t=5:   Check for duplicate coordinates (the production bug symptom)
      t=60:  Wait for completion, final validation

    Expected: No duplicate coordinates, tree is valid after re-index.
    Bug symptom: Duplicate (height, level_index) coords at height > 0.
    """
    session_id = f"repro-reset-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        print(f"\n  Session: {session_id}")

        # Phase 1: Large initial ingest (creates many indexing jobs)
        print("  Phase 1: Ingesting 80 messages (creates ~20 leaves)...")
        chunk, last_uuid = _make_transcript(
            num_messages=80,
            content_size=800,  # ~200 tokens per message
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    After initial: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Phase 2: Brief pause to let some jobs START but not complete
        time.sleep(2)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    Before reset: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Phase 3: RESET while jobs are in-flight
        print("  Phase 3: RESETTING session cursor...")
        _reset_session_cursor(devstack_db_url, session_id)

        # Phase 4: Re-ingest the SAME content (triggers full re-index)
        # This is what happens when someone does `admin reset` then re-syncs
        print("  Phase 4: Re-ingesting same content (triggers full re-index)...")
        _ingest_session_sync(stub, session_id, chunk, user_id)

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(
            f"    After re-ingest: leaves={progress['leaves']}, "
            f"embedded={progress['embedded']}, summaries={progress['summaries']}"
        )

        # Phase 5: CHECK FOR DUPLICATE COORDINATES (the production bug symptom)
        print("  Phase 5: Checking for duplicate coordinates...")
        duplicates = _check_for_duplicate_coords(devstack_db_url, session_id)
        if duplicates:
            print(f"    ❌ FOUND {len(duplicates)} duplicate coordinates!")
            for h, li in duplicates[:5]:  # Show first 5
                print(f"       height={h}, level_index={li}")
            # This is the production bug - fail the test
            raise AssertionError(
                f"Duplicate coordinates found after reset + re-ingest: {duplicates[:5]}. "
                f"This indicates in-flight jobs from the first ingest weren't properly "
                f"cancelled before the re-index started."
            )
        else:
            print("    ✅ No duplicate coordinates found")

        # Phase 6: Validate tree invariants
        print("  Phase 6: Validating tree invariants...")
        report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=True
        )
        print(f"    Validation: {report.status}")

        # Phase 7: Wait for completion
        print("  Phase 7: Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=180)

        # Final check for duplicates after completion
        duplicates = _check_for_duplicate_coords(devstack_db_url, session_id)
        if duplicates:
            raise AssertionError(
                f"Duplicate coordinates found after indexing complete: {duplicates[:5]}"
            )

        # Final validation
        print("  Phase 8: Final validation...")
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        print(
            f"    Final: nodes={final_report.metrics.get('node_count')}, "
            f"leaves={final_report.metrics.get('leaf_count')}, "
            f"status={final_report.status}"
        )

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow_threshold(300)
def test_multiple_resets_during_indexing(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Multiple rapid reset + re-ingest cycles to stress-test cancellation.

    This is a more aggressive version that does multiple reset cycles
    to maximize the chance of race conditions between job cancellation
    and re-indexing.

    Timeline:
      Round 1-5:
        - Ingest content
        - Wait 1-2s (jobs partially complete)
        - Reset cursor
        - Re-ingest
        - Check for duplicates
    """
    session_id = f"repro-multi-reset-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        print(f"\n  Session: {session_id}")

        # Create the transcript once, reuse for all rounds
        chunk, last_uuid = _make_transcript(
            num_messages=60,
            content_size=600,
        )

        for round_num in range(5):
            print(f"\n  Round {round_num + 1}/5:")

            # Ingest
            print("    Ingesting...")
            _ingest_session_sync(stub, session_id, chunk, user_id)

            # Brief pause
            delay = random.uniform(1, 3)
            time.sleep(delay)

            progress = _get_indexing_progress(devstack_db_url, session_id)
            print(
                f"    Progress: leaves={progress['leaves']}, "
                f"embedded={progress['embedded']}, summaries={progress['summaries']}"
            )

            # Reset
            print("    Resetting...")
            _reset_session_cursor(devstack_db_url, session_id)

            # Check for duplicates BEFORE re-ingest
            duplicates = _check_for_duplicate_coords(devstack_db_url, session_id)
            if duplicates:
                raise AssertionError(
                    f"Round {round_num + 1}: Duplicates found after reset: {duplicates[:5]}"
                )

            # Validate
            _validate_tree_invariants(
                postgres_backend, session_id, allow_incomplete=True
            )

        # Final ingest and wait for completion
        print("\n  Final phase: Re-ingest and wait for completion...")
        _ingest_session_sync(stub, session_id, chunk, user_id)
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=180)

        # Final duplicate check
        duplicates = _check_for_duplicate_coords(devstack_db_url, session_id)
        if duplicates:
            raise AssertionError(f"Final duplicates found: {duplicates[:5]}")

        # Final validation
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        print(
            f"  Final: nodes={final_report.metrics.get('node_count')}, "
            f"leaves={final_report.metrics.get('leaf_count')}, "
            f"status={final_report.status}"
        )

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.slow_threshold(120)
@pytest.mark.xfail(
    reason="Bug 3: FK violation on partial revert - truncation logic doesn't handle "
    "tree hierarchy. Internal nodes spanning the truncation boundary get deleted "
    "while their children on the 'kept' side remain with dangling parent_id refs. "
    "See plan file: snug-finding-waterfall.md",
    strict=True,
)
def test_partial_revert_multiturn(
    devstack_db_url: str,
    grpc_address: str,
    postgres_backend: PostgresStorageBackend,
) -> None:
    """Partial revert to mid-point in append log.

    Scenario:
    - Create 10 turns (10 segments = 10 append log entries)
    - Revert to turn 5 (middle of the append log)
    - This should truncate to the span_end of an earlier append log entry

    Expected: Tree should contain turns 0-5 from original + new revert content.
    """
    session_id = f"repro-partial-revert-{uuid.uuid4()}"
    user_id = "test-user"

    try:
        channel = grpc.insecure_channel(grpc_address)
        stub = pb2_grpc.SessionIngestionServiceStub(channel)
        grpc.channel_ready_future(channel).result(timeout=5)
    except Exception as e:
        pytest.skip(f"gRPC server not available at {grpc_address}: {e}")

    try:
        print(f"\n  Session: {session_id}")

        # Phase 1: Create 10 turns
        print("  Phase 1: Ingesting 10 turns...")
        chunk, last_uuid, all_uuids, user_uuids = _make_multiturn_transcript(
            num_turns=10, content_size=500
        )
        _ingest_session_sync(stub, session_id, chunk, user_id)

        entry_count = _get_append_entry_count(devstack_db_url, session_id)
        print(f"    After initial: append_entries={entry_count}")

        # Wait for initial indexing
        print("  Phase 2: Waiting for initial indexing...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=120)

        initial_leaves = _get_indexing_progress(devstack_db_url, session_id)["leaves"]
        print(f"    Initial indexing complete: {initial_leaves} leaves")

        # Phase 3: Partial revert - branch from turn 5's assistant message
        # This is mid-way through the append log
        # user_uuids[5] is the user message of turn 5
        # The assistant message before it is all_uuids[2*5 - 1] = all_uuids[9]
        # Actually, let's revert from the end of turn 4 (assistant of turn 4)
        # Turn 4 = user_uuids[4], assistant is all_uuids[4*2 + 1] = all_uuids[9]
        revert_point = all_uuids[9]  # End of turn 4
        print(
            f"  Phase 3: PARTIAL REVERT to end of turn 4 (uuid={revert_point[:8]}...)"
        )

        revert_chunk, new_last, new_all, new_users = _make_multiturn_revert_transcript(
            revert_to_uuid=revert_point,
            num_turns=3,
            content_size=500,
            branch_marker="PARTIAL_REVERT",
        )
        response = _ingest_session_sync(stub, session_id, revert_chunk, user_id)
        print(
            f"    Response: truncated={response.truncated}, "
            f"truncate_span={response.truncate_span}"
        )

        progress = _get_indexing_progress(devstack_db_url, session_id)
        print(f"    After revert: leaves={progress['leaves']}")

        # Validate immediately
        print("  Phase 4: Validating after partial revert...")
        report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=True
        )
        print(f"    Validation: {report.status}")

        # Wait for completion
        print("  Phase 5: Waiting for indexing to complete...")
        _wait_for_indexing_complete(devstack_db_url, session_id, timeout=120)

        # Final validation
        print("  Phase 6: Final validation...")
        final_report = _validate_tree_invariants(
            postgres_backend, session_id, allow_incomplete=False
        )
        final_leaves = final_report.metrics.get("leaf_count", 0)
        print(
            f"    Final: nodes={final_report.metrics.get('node_count')}, "
            f"leaves={final_leaves}, status={final_report.status}"
        )

        # For partial revert: we keep turns 0-4 (5 turns) and add 3 new turns
        # Total: 8 turns worth of content
        # This should be less than the original 10 turns
        assert final_leaves < initial_leaves, (
            f"Expected fewer leaves after partial revert. "
            f"Initial: {initial_leaves}, Final: {final_leaves}. "
            f"This may indicate truncation did not work correctly."
        )

        print("  ✅ Test passed!")

    finally:
        channel.close()
        try:
            _cleanup_test_session(devstack_db_url, session_id)
        except Exception:
            pass
