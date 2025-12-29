"""gRPC servicer for Claude Code session ingestion."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import grpc
from sqlalchemy import text

from memory_service.storage import SessionStorage
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from ragzoom.wrapper import RagZoom

logger = logging.getLogger(__name__)


def _session_lock_id(user_id: str, session_id: str) -> int:
    """Generate a stable lock ID for a user/session pair.

    Uses hash to convert string IDs to a 32-bit integer for pg_advisory_lock.
    """
    key = f"{user_id}:{session_id}"
    # Use first 8 hex chars (32 bits) of MD5 hash
    hash_hex = hashlib.md5(key.encode(), usedforsecurity=False).hexdigest()[:8]
    return int(hash_hex, 16)


async def _validate_request(
    request: pb2.GetSessionCursorRequest | pb2.IngestSessionRequest,
    context: pb2_grpc.ServicerContext,
) -> tuple[str, str]:
    """Validate request and extract user_id and session_id.

    Raises gRPC errors via context.abort if validation fails.
    Returns (user_id, session_id) on success.
    """
    user_id = _get_user_id_from_context(context)
    if not user_id:
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing user_id")

    session_id = request.session_id
    if not session_id:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "session_id is required")

    return user_id, session_id


class SessionIngestionServicer(pb2_grpc.SessionIngestionServiceServicer):
    """Servicer for Claude Code session ingestion.

    Handles:
    - GetSessionCursor: Returns byte offset for incremental sync
    - IngestSession: Accepts JSONL delta, processes and indexes
    """

    def __init__(
        self,
        get_db_session: Callable[[], DbSession],
        get_ragzoom_client: Callable[[str], RagZoom],
    ) -> None:
        """Initialize the servicer.

        Args:
            get_db_session: Factory to get a database session
            get_ragzoom_client: Factory to get a RagZoom client for a user
        """
        self._get_db_session = get_db_session
        self._get_ragzoom_client = get_ragzoom_client

    async def GetSessionCursor(  # noqa: N802
        self,
        request: pb2.GetSessionCursorRequest,
        context: pb2_grpc.ServicerContext,
    ) -> pb2.GetSessionCursorResponse:
        """Get the current byte offset for a session."""
        user_id, session_id = await _validate_request(request, context)

        db_session = self._get_db_session()
        try:
            storage = SessionStorage(db_session, user_id)
            cursor = storage.get_cursor(session_id)
            return pb2.GetSessionCursorResponse(byte_offset=cursor.byte_offset)
        finally:
            db_session.close()

    async def IngestSession(  # noqa: N802
        self,
        request: pb2.IngestSessionRequest,
        context: pb2_grpc.ServicerContext,
    ) -> pb2.IngestSessionResponse:
        """Ingest JSONL delta for a session.

        Uses advisory lock only for the critical section (cursor read/write),
        then releases it before the long-running embedding process. This allows
        subsequent syncs to proceed while embeddings run in the background.
        """
        from memory_service.ingestion.claude.transcript_sync import (
            execute_sync_from_bytes,
        )

        t0 = time.perf_counter()
        user_id, session_id = await _validate_request(request, context)

        delta = request.jsonl_delta
        if not delta:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "jsonl_delta is required"
            )

        logger.info(
            "[TIMING] IngestSession start: session=%s delta_bytes=%d",
            session_id[:8],
            len(delta),
        )

        lock_id = _session_lock_id(user_id, session_id)
        db_session = self._get_db_session()

        # Phase 1: Under lock - read existing content and append delta
        t1 = time.perf_counter()
        try:
            db_session.execute(
                text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": lock_id}
            )
            t_lock = time.perf_counter()

            storage = SessionStorage(db_session, user_id)
            existing_content = storage.get_content(session_id)
            t_read = time.perf_counter()

            full_content = existing_content + delta
            previous_byte_offset = len(existing_content)

            # Append content immediately (before processing)
            # This updates the cursor so subsequent syncs don't re-send this delta
            new_offset = storage.append_content(session_id, delta)
            db_session.commit()
            t_commit = time.perf_counter()

            logger.info(
                "[TIMING] Phase1 complete: lock=%.3fs read=%.3fs commit=%.3fs "
                "total=%.3fs existing_bytes=%d",
                t_lock - t1,
                t_read - t_lock,
                t_commit - t_read,
                t_commit - t1,
                len(existing_content),
            )

        finally:
            # Release lock before the slow embedding phase
            try:
                db_session.execute(
                    text("SELECT pg_advisory_unlock(:lock_id)"), {"lock_id": lock_id}
                )
            except Exception:
                pass
            db_session.close()

        # Phase 2: Without lock - run embedding/indexing (slow)
        t2 = time.perf_counter()
        ragzoom_client = self._get_ragzoom_client(user_id)
        try:
            result = await asyncio.to_thread(
                execute_sync_from_bytes,
                session_id=session_id,
                jsonl_content=full_content,
                previous_byte_offset=previous_byte_offset,
                client=ragzoom_client,
            )
            t3 = time.perf_counter()
            logger.info(
                "[TIMING] Phase2 complete: execute_sync=%.3fs appended=%d truncated=%s",
                t3 - t2,
                len(result.appended_uuids),
                result.truncated,
            )

            # Handle truncation if detected (requires re-acquiring lock)
            if result.truncated and result.truncate_byte_offset is not None:
                db_session = self._get_db_session()
                try:
                    db_session.execute(
                        text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": lock_id}
                    )
                    storage = SessionStorage(db_session, user_id)
                    storage.truncate_content(session_id, result.truncate_byte_offset)
                    new_offset = storage.append_content(session_id, delta)
                    db_session.commit()
                finally:
                    try:
                        db_session.execute(
                            text("SELECT pg_advisory_unlock(:lock_id)"),
                            {"lock_id": lock_id},
                        )
                    except Exception:
                        pass
                    db_session.close()

            t_end = time.perf_counter()
            logger.info(
                "[TIMING] IngestSession complete: total=%.3fs session=%s",
                t_end - t0,
                session_id[:8],
            )
            return pb2.IngestSessionResponse(
                new_byte_offset=new_offset,
                messages_processed=len(result.appended_uuids),
                truncated=result.truncated,
                truncate_span=result.truncate_span or 0,
            )
        except Exception as e:
            logger.exception("Error ingesting session %s", session_id)
            await context.abort(grpc.StatusCode.INTERNAL, f"Ingestion failed: {e}")


def _get_user_id_from_context(context: pb2_grpc.ServicerContext) -> str | None:
    """Extract user_id from gRPC context metadata."""
    metadata = context.invocation_metadata()
    if metadata is None:
        return None

    for key, value in metadata:
        if key == "user_id":
            return str(value)

    return None
