"""Storage for Claude Code session sync state.

Stores only the sync cursor and state, NOT the full JSONL content.
This enables memory-efficient incremental syncing where only deltas are processed.

The cursor tracks:
- byte_offset: Where the client should resume sending from
- last_uuid: UUID of the last synced message (for revert detection)
- span_end: Document span position after last sync
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import BigInteger, Index, Integer, String, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from ragzoom.models import Base

logger = logging.getLogger(__name__)


class SessionSyncState(Base):
    """Stores sync state for Claude Code sessions.

    This is a lightweight record that tracks sync progress without storing
    the full JSONL content. The client is responsible for sending deltas
    from the byte_offset position.
    """

    __tablename__ = "session_sync_state"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Cursor position - where client should resume sending from
    byte_offset: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    # Last synced message UUID - for revert detection
    last_uuid: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Document span position after last sync
    span_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index(
            "ix_session_sync_state_user_session", "user_id", "session_id", unique=True
        ),
    )


@dataclass
class SessionCursor:
    """Cursor position and sync state for a session."""

    byte_offset: int
    last_uuid: str | None = None
    span_end: int = 0


class SessionStorage:
    """Storage interface for session sync state.

    This is a lightweight storage that tracks only the sync cursor and state,
    NOT the full JSONL content. This enables memory-efficient incremental syncing.
    """

    def __init__(self, db_session: Session, user_id: str) -> None:
        self._db = db_session
        self._user_id = user_id

    def get_cursor(self, session_id: str) -> SessionCursor:
        """Get the current sync state for a session.

        Returns byte_offset=0 if session doesn't exist yet.
        """
        stmt = select(SessionSyncState).where(
            SessionSyncState.user_id == self._user_id,
            SessionSyncState.session_id == session_id,
        )
        row = self._db.execute(stmt).scalar_one_or_none()

        if row is None:
            return SessionCursor(byte_offset=0)

        return SessionCursor(
            byte_offset=row.byte_offset,
            last_uuid=row.last_uuid,
            span_end=row.span_end,
        )

    def update_cursor(
        self,
        session_id: str,
        byte_offset: int,
        last_uuid: str | None,
        span_end: int,
    ) -> None:
        """Update the sync state for a session.

        Creates the session if it doesn't exist.
        """
        stmt = select(SessionSyncState).where(
            SessionSyncState.user_id == self._user_id,
            SessionSyncState.session_id == session_id,
        )
        row = self._db.execute(stmt).scalar_one_or_none()

        if row is None:
            row = SessionSyncState(
                user_id=self._user_id,
                session_id=session_id,
                byte_offset=byte_offset,
                last_uuid=last_uuid,
                span_end=span_end,
            )
            self._db.add(row)
        else:
            row.byte_offset = byte_offset
            row.last_uuid = last_uuid
            row.span_end = span_end

        self._db.flush()

    def reset_cursor(self, session_id: str) -> None:
        """Reset the sync state for a session (used on revert).

        Deletes the session state entirely, forcing a full re-sync.
        """
        stmt = select(SessionSyncState).where(
            SessionSyncState.user_id == self._user_id,
            SessionSyncState.session_id == session_id,
        )
        row = self._db.execute(stmt).scalar_one_or_none()

        if row is not None:
            self._db.delete(row)
            self._db.flush()
