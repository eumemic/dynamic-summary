"""Storage for raw Claude Code session JSONL files.

Stores complete JSONL content in the database, enabling:
- Cursor calculation (byte offset = length of stored content)
- Reindexing from raw data
- Debugging by inspecting original transcripts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import Index, LargeBinary, String, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from ragzoom.models import Base

logger = logging.getLogger(__name__)


class SessionRawData(Base):
    """Stores raw JSONL content for Claude Code sessions."""

    __tablename__ = "session_raw_data"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    jsonl_content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    __table_args__ = (
        Index("ix_session_raw_data_user_session", "user_id", "session_id", unique=True),
    )


@dataclass
class SessionCursor:
    """Cursor position for a session."""

    byte_offset: int


class SessionStorage:
    """Storage interface for raw session JSONL data."""

    def __init__(self, db_session: Session, user_id: str) -> None:
        self._db = db_session
        self._user_id = user_id

    def get_cursor(self, session_id: str) -> SessionCursor:
        """Get the current cursor (byte offset) for a session.

        Returns byte_offset=0 if session doesn't exist yet.
        """
        stmt = select(SessionRawData.jsonl_content).where(
            SessionRawData.user_id == self._user_id,
            SessionRawData.session_id == session_id,
        )
        result = self._db.execute(stmt).scalar_one_or_none()

        if result is None:
            return SessionCursor(byte_offset=0)

        return SessionCursor(byte_offset=len(result))

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
        """
        stmt = select(SessionRawData).where(
            SessionRawData.user_id == self._user_id,
            SessionRawData.session_id == session_id,
        )
        row = self._db.execute(stmt).scalar_one_or_none()

        if row is not None and len(row.jsonl_content) > byte_offset:
            row.jsonl_content = row.jsonl_content[:byte_offset]
            self._db.flush()
