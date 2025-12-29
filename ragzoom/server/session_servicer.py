"""gRPC servicer for Claude Code session ingestion."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, NoReturn, Protocol

import grpc

from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.rpc import dynamic_summary_pb2_grpc as pb2_grpc
from ragzoom.server.session_storage import SessionStorage

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from ragzoom.wrapper import RagZoom

logger = logging.getLogger(__name__)


class ServicerContextProto(Protocol):
    """Protocol for gRPC servicer context."""

    def invocation_metadata(self) -> list[tuple[str, str]] | None: ...

    async def abort(self, code: object, details: str) -> NoReturn: ...


async def _validate_request(
    request: object,
    context: ServicerContextProto,
) -> tuple[str, str]:
    """Validate request and extract user_id and session_id.

    Raises gRPC errors via context.abort if validation fails.
    Returns (user_id, session_id) on success.
    """
    user_id = _get_user_id_from_context(context)
    if not user_id:
        await context.abort(grpc.StatusCode.UNAUTHENTICATED, "Missing user_id")

    session_id = getattr(request, "session_id", "")
    if not session_id:
        await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "session_id is required")

    return user_id, session_id


class SessionIngestionServicer(pb2_grpc.SessionIngestionServiceServicer):  # type: ignore[misc,name-defined]
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
        request: object,
        context: ServicerContextProto,
    ) -> object:
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
        request: object,
        context: ServicerContextProto,
    ) -> object:
        """Ingest JSONL delta for a session."""
        from ragzoom.claude_memory.transcript_sync import execute_sync_from_bytes

        user_id, session_id = await _validate_request(request, context)

        delta = getattr(request, "jsonl_delta", b"")
        if not delta:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "jsonl_delta is required"
            )

        db_session = self._get_db_session()
        try:
            storage = SessionStorage(db_session, user_id)

            # Get existing content + append delta
            existing_content = storage.get_content(session_id)
            full_content = existing_content + delta

            # Get RagZoom client for this user
            ragzoom_client = self._get_ragzoom_client(user_id)

            # Process the full JSONL content using sync logic
            result = execute_sync_from_bytes(
                session_id=session_id,
                jsonl_content=full_content,
                previous_byte_offset=len(existing_content),
                client=ragzoom_client,
            )

            # Update stored content based on result
            if result.truncated and result.truncate_byte_offset is not None:
                # Revert detected - truncate stored content
                storage.truncate_content(session_id, result.truncate_byte_offset)
                # Re-append the delta (which includes new content after revert point)
                new_offset = storage.append_content(session_id, delta)
            else:
                # Normal append
                new_offset = storage.append_content(session_id, delta)

            db_session.commit()

            return pb2.IngestSessionResponse(
                new_byte_offset=new_offset,
                messages_processed=len(result.appended_uuids),
                truncated=result.truncated,
                truncate_span=result.truncate_span or 0,
            )
        except Exception as e:
            db_session.rollback()
            logger.exception("Error ingesting session %s", session_id)
            await context.abort(grpc.StatusCode.INTERNAL, f"Ingestion failed: {e}")
        finally:
            db_session.close()


def _get_user_id_from_context(context: ServicerContextProto) -> str | None:
    """Extract user_id from gRPC context metadata."""
    metadata = context.invocation_metadata()
    if metadata is None:
        return None

    for key, value in metadata:
        if key == "user_id":
            return value

    return None
