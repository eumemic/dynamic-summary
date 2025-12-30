"""Storage for raw Claude Code session JSONL files.

Stores complete JSONL content in the database, enabling:
- Cursor calculation (byte offset = length of stored content)
- Reindexing from raw data
- Debugging by inspecting original transcripts

Also tracks sync state (last_synced_uuid, span_end) for memory-efficient
incremental syncing without loading full content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import BigInteger, Index, Integer, LargeBinary, String, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from ragzoom.models import Base

logger = logging.getLogger(__name__)


class SessionRawData(Base):
    """Stores raw JSONL content and sync state for Claude Code sessions."""

    __tablename__ = "session_raw_data"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    jsonl_content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Sync state for memory-efficient incremental syncing
    last_synced_uuid: Mapped[str | None] = mapped_column(String(255), nullable=True)
    span_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_session_raw_data_user_session", "user_id", "session_id", unique=True),
    )


@dataclass
class SessionCursor:
    """Cursor position and sync state for a session."""

    byte_offset: int
    last_synced_uuid: str | None = None
    span_end: int = 0


class SessionStorage:
    """Storage interface for raw session JSONL data."""

    def __init__(self, db_session: Session, user_id: str) -> None:
        self._db = db_session
        self._user_id = user_id

    def get_cursor(self, session_id: str) -> SessionCursor:
        """Get the current cursor and sync state for a session.

        Memory-efficient: uses LENGTH() to get byte offset without loading content.
        Returns byte_offset=0 if session doesn't exist yet.
        """
        # Use LENGTH() to get byte offset without loading content into memory
        stmt = select(
            func.length(SessionRawData.jsonl_content),
            SessionRawData.last_synced_uuid,
            SessionRawData.span_end,
        ).where(
            SessionRawData.user_id == self._user_id,
            SessionRawData.session_id == session_id,
        )
        result = self._db.execute(stmt).one_or_none()

        if result is None:
            return SessionCursor(byte_offset=0)

        content_length, last_uuid, span_end = result
        return SessionCursor(
            byte_offset=content_length or 0,
            last_synced_uuid=last_uuid,
            span_end=span_end or 0,
        )

    def get_content(self, session_id: str) -> bytes:
        """Get the full JSONL content for a session.

        Returns empty bytes if session doesn't exist.
        """
        stmt = select(SessionRawData.jsonl_content).where(
            SessionRawData.user_id == self._user_id,
            SessionRawData.session_id == session_id,
        )
        result = self._db.execute(stmt).scalar_one_or_none()
        return result if result is not None else b""

    def append_content(self, session_id: str, delta: bytes) -> int:
        """Append JSONL bytes to a session.

        Creates the session if it doesn't exist.
        Returns the new byte offset (total length after append).
        """
        stmt = select(SessionRawData).where(
            SessionRawData.user_id == self._user_id,
            SessionRawData.session_id == session_id,
        )
        row = self._db.execute(stmt).scalar_one_or_none()

        if row is None:
            # Create new session
            row = SessionRawData(
                user_id=self._user_id,
                session_id=session_id,
                jsonl_content=delta,
            )
            self._db.add(row)
            self._db.flush()
            return len(delta)

        # Append to existing
        new_content = row.jsonl_content + delta
        row.jsonl_content = new_content
        self._db.flush()
        return len(new_content)

    def truncate_content(self, session_id: str, byte_offset: int) -> None:
        """Truncate session content to the given byte offset.

        Used when handling reverts in the transcript.
        Also resets sync state since we need to re-sync.
        """
        stmt = select(SessionRawData).where(
            SessionRawData.user_id == self._user_id,
            SessionRawData.session_id == session_id,
        )
        row = self._db.execute(stmt).scalar_one_or_none()

        if row is not None and len(row.jsonl_content) > byte_offset:
            row.jsonl_content = row.jsonl_content[:byte_offset]
            row.last_synced_uuid = None
            row.span_end = 0
            self._db.flush()

    def update_sync_state(
        self, session_id: str, last_synced_uuid: str, span_end: int
    ) -> None:
        """Update the sync state after successful processing.

        This is called after content is already appended and processed,
        to record what UUID we last synced and the document span position.
        """
        stmt = select(SessionRawData).where(
            SessionRawData.user_id == self._user_id,
            SessionRawData.session_id == session_id,
        )
        row = self._db.execute(stmt).scalar_one_or_none()

        if row is not None:
            row.last_synced_uuid = last_synced_uuid
            row.span_end = span_end
            self._db.flush()
