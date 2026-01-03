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

    from ragzoom.wrapper import AsyncRagZoom

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
    request: (
        pb2.GetSessionCursorRequest
        | pb2.IngestSessionRequest
        | pb2.GetCompactionBoundaryRequest
        | pb2.ResetSessionCursorRequest
    ),
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
        get_async_ragzoom_client: Callable[[str], AsyncRagZoom],
    ) -> None:
        """Initialize the servicer.

        Args:
            get_db_session: Factory to get a database session
            get_async_ragzoom_client: Factory to get an AsyncRagZoom client for a user
        """
        self._get_db_session = get_db_session
        self._get_async_ragzoom_client = get_async_ragzoom_client

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

        Memory-efficient: only loads full content on revert (rare).
        Normal appends process only the delta without loading history.
        """
        from memory_service.ingestion.claude.transcript_sync import (
            SyncResult,
            prepare_delta_sync,
            prepare_streaming_resync,
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

        # Phase 1: Under lock - get cursor and append delta (NO content read)
        t1 = time.perf_counter()
        try:
            db_session.execute(
                text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": lock_id}
            )
            t_lock = time.perf_counter()

            storage = SessionStorage(db_session, user_id)
            cursor = storage.get_cursor(session_id)  # Uses LENGTH(), no content load
            t_read = time.perf_counter()

            # Append delta immediately (before processing)
            new_offset = storage.append_content(session_id, delta)
            db_session.commit()
            t_commit = time.perf_counter()

            logger.info(
                "[TIMING] Phase1 complete: lock=%.3fs cursor=%.3fs commit=%.3fs "
                "total=%.3fs prev_offset=%d",
                t_lock - t1,
                t_read - t_lock,
                t_commit - t_read,
                t_commit - t1,
                cursor.byte_offset,
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

        # Phase 2: Without lock - process delta or handle revert
        t2 = time.perf_counter()
        async_client = self._get_async_ragzoom_client(user_id)
        try:
            # Prepare delta sync in thread (CPU-bound parsing)
            prepared = await asyncio.to_thread(
                prepare_delta_sync,
                session_id=session_id,
                delta=delta,
                cursor=cursor,
            )
            t3 = time.perf_counter()

            if prepared.truncated:
                # Revert detected - need to load full content and re-sync
                logger.warning(
                    "[TIMING] Revert detected, loading full content for re-sync"
                )
                db_session = self._get_db_session()
                try:
                    db_session.execute(
                        text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": lock_id}
                    )
                    storage = SessionStorage(db_session, user_id)
                    full_content = storage.get_content(session_id)
                    # Load append entries for granular revert truncation
                    append_entries = storage.get_append_entries(session_id)
                finally:
                    try:
                        db_session.execute(
                            text("SELECT pg_advisory_unlock(:lock_id)"),
                            {"lock_id": lock_id},
                        )
                    except Exception:
                        pass
                    db_session.close()

                # Prepare streaming resync in thread (CPU-bound parsing)
                prepared_resync = await asyncio.to_thread(
                    prepare_streaming_resync,
                    session_id=session_id,
                    jsonl_content=full_content,
                    last_synced_uuid=cursor.last_synced_uuid,
                    span_end=cursor.span_end,
                    append_entries=append_entries,
                )

                # Execute truncate if needed (async, on main loop)
                if (
                    prepared_resync.needs_truncate
                    and prepared_resync.truncate_span is not None
                ):
                    await async_client.truncate(
                        prepared_resync.document_id, prepared_resync.truncate_span
                    )

                # Execute batch_append (async, on main loop - jobs run on this loop)
                new_span_end = prepared_resync.span_end
                if prepared_resync.segment_texts:
                    append_result = await async_client.batch_append(
                        prepared_resync.document_id, prepared_resync.segment_texts
                    )
                    new_span_end = append_result.span_end

                result = SyncResult(
                    document_id=prepared_resync.document_id,
                    truncated=prepared_resync.needs_truncate,
                    truncate_span=prepared_resync.truncate_span,
                    appended_uuids=prepared_resync.appended_uuids,
                    new_span_end=new_span_end,
                    segment_last_uuids=prepared_resync.segment_last_uuids,
                    valid_prefix_uuid=prepared_resync.valid_prefix_uuid,
                )
                t3 = time.perf_counter()
            else:
                # Normal delta sync - execute batch_append (async, on main loop)
                new_span_end = prepared.span_end
                if prepared.segment_texts:
                    append_result = await async_client.batch_append(
                        prepared.document_id, prepared.segment_texts
                    )
                    new_span_end = append_result.span_end

                result = SyncResult(
                    document_id=prepared.document_id,
                    truncated=False,
                    truncate_span=None,
                    appended_uuids=prepared.appended_uuids,
                    new_span_end=new_span_end,
                    segment_last_uuids=prepared.segment_last_uuids,
                )
                t3 = time.perf_counter()

            logger.info(
                "[TIMING] Phase2 complete: sync=%.3fs appended=%d truncated=%s",
                t3 - t2,
                len(result.appended_uuids),
                result.truncated,
            )

            # Phase 3: Update sync state and append log
            if result.appended_uuids or result.truncated:
                db_session = self._get_db_session()
                try:
                    db_session.execute(
                        text("SELECT pg_advisory_lock(:lock_id)"), {"lock_id": lock_id}
                    )
                    storage = SessionStorage(db_session, user_id)

                    # Handle append log truncation on revert
                    if result.truncated:
                        if result.valid_prefix_uuid is not None:
                            # Truncate to valid prefix
                            storage.truncate_entries_after(
                                session_id, result.valid_prefix_uuid
                            )
                        else:
                            # No valid prefix - clear all entries
                            storage.clear_append_entries(session_id)

                    # Add new append entries for each segment
                    # Calculate cumulative span positions starting from truncate point or previous span_end
                    if result.segment_last_uuids:
                        span_cursor = result.truncate_span or cursor.span_end
                        segment_texts_list = (
                            prepared_resync.segment_texts
                            if result.truncated
                            else prepared.segment_texts
                        )
                        for segment_text, last_uuid in zip(
                            segment_texts_list, result.segment_last_uuids
                        ):
                            span_cursor += len(segment_text)
                            storage.append_entry(session_id, last_uuid, span_cursor)

                    # Update sync state
                    if result.appended_uuids:
                        storage.update_sync_state(
                            session_id,
                            last_synced_uuid=result.appended_uuids[-1],
                            span_end=result.new_span_end,
                        )
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

    async def GetCompactionBoundary(  # noqa: N802
        self,
        request: pb2.GetCompactionBoundaryRequest,
        context: pb2_grpc.ServicerContext,
    ) -> pb2.GetCompactionBoundaryResponse:
        """Get the compaction boundary span_end for a session.

        Computes the boundary DYNAMICALLY by scanning the stored transcript
        to find the most recent compaction. This ensures the boundary is always
        correct even after multiple compactions occur.

        Returns has_boundary=False if no compaction has occurred,
        otherwise returns the span_end just before post-compaction content.
        """
        from memory_service.ingestion.claude.transcript_sync import (
            compute_compaction_boundary_from_bytes,
        )

        user_id, session_id = await _validate_request(request, context)

        db_session = self._get_db_session()
        try:
            storage = SessionStorage(db_session, user_id)
            content = storage.get_content(session_id)

            if not content:
                return pb2.GetCompactionBoundaryResponse(has_boundary=False, span_end=0)

            boundary = compute_compaction_boundary_from_bytes(content)
            if boundary is None:
                return pb2.GetCompactionBoundaryResponse(has_boundary=False, span_end=0)

            return pb2.GetCompactionBoundaryResponse(
                has_boundary=True, span_end=boundary
            )
        finally:
            db_session.close()

    async def ResetSessionCursor(  # noqa: N802
        self,
        request: pb2.ResetSessionCursorRequest,
        context: pb2_grpc.ServicerContext,
    ) -> pb2.ResetSessionCursorResponse:
        """Reset a session's cursor to force full re-sync.

        This clears the last_synced_uuid and byte offset, causing the next
        sync to re-process the entire transcript.
        """
        user_id, session_id = await _validate_request(request, context)

        db_session = self._get_db_session()
        try:
            storage = SessionStorage(db_session, user_id)
            storage.reset_cursor(session_id)
            db_session.commit()
            return pb2.ResetSessionCursorResponse(
                success=True, message=f"Reset cursor for session {session_id}"
            )
        except Exception as e:
            logger.exception("Error resetting session cursor %s", session_id)
            return pb2.ResetSessionCursorResponse(
                success=False, message=f"Failed to reset cursor: {e}"
            )
        finally:
            db_session.close()


def _get_user_id_from_context(context: pb2_grpc.ServicerContext) -> str | None:
    """Extract user_id from gRPC context metadata."""
    metadata = context.invocation_metadata()
    if metadata is None:
        return None

    for key, value in metadata:
        if key == "user_id":
            return str(value)

    return None
