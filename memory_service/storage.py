"""Storage for raw Claude Code session JSONL files.

Stores stripped JSONL content in the database, enabling:
- Cursor calculation (byte offset = length of stored content)
- Reindexing from raw data
- Debugging by inspecting original transcripts

The stored JSONL has tool_result content stripped to reduce storage:
- Records with type="user" and toolUseResult are stripped to essential fields
- This removes ~80% of raw JSONL size (tool outputs we don't transcribe)

Also tracks sync state (last_synced_uuid, span_end) for memory-efficient
incremental syncing without loading full content.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sqlalchemy import BigInteger, Index, Integer, LargeBinary, String, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column

from ragzoom.models import Base

logger = logging.getLogger(__name__)


def strip_tool_results(jsonl_bytes: bytes) -> bytes:
    """Strip tool result content from JSONL bytes to reduce storage.

    Tool results (type="user" with toolUseResult field) contain the full output
    of tool executions, which can be megabytes of file contents, command output,
    etc. We don't need this content for transcription - we only need the record
    structure for sync/branch detection.

    For tool result records, we keep only:
    - uuid, parentUuid, type (required for sync algorithm)
    - isCompactSummary (if present, for compaction bridging)

    All other records are preserved unchanged.

    Args:
        jsonl_bytes: Raw JSONL bytes (newline-delimited JSON records)

    Returns:
        Stripped JSONL bytes with tool result content removed
    """
    if not jsonl_bytes:
        return jsonl_bytes

    lines = jsonl_bytes.split(b"\n")
    stripped_lines: list[bytes] = []

    for line in lines:
        if not line.strip():
            stripped_lines.append(line)
            continue

        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Keep malformed lines as-is (shouldn't happen, but be safe)
            stripped_lines.append(line)
            continue

        # Only strip tool result records
        if record.get("type") == "user" and "toolUseResult" in record:
            # Keep only essential fields for sync
            stripped_record: dict[str, object] = {
                "uuid": record.get("uuid"),
                "type": "user",
                "toolUseResult": "[stripped]",  # Marker so we know this was a tool result
            }
            if "parentUuid" in record:
                stripped_record["parentUuid"] = record["parentUuid"]
            if "isCompactSummary" in record:
                stripped_record["isCompactSummary"] = record["isCompactSummary"]

            stripped_lines.append(json.dumps(stripped_record).encode())
        else:
            # Preserve other records unchanged
            stripped_lines.append(line)

    return b"\n".join(stripped_lines)


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

    # Original file offset tracking (different from stored content length due to stripping)
    # This tracks the position in the original unstripped file that corresponds to
    # the end of stored content. Used by clients to know where to seek in the file.
    original_file_offset: Mapped[int] = mapped_column(
        BigInteger, nullable=False, default=0
    )

    __table_args__ = (
        Index("ix_session_raw_data_user_session", "user_id", "session_id", unique=True),
    )


@dataclass
class SessionCursor:
    """Cursor position and sync state for a session."""

    byte_offset: int  # Original file position (for client seeking)
    last_synced_uuid: str | None = None
    span_end: int = 0


class SessionStorage:
    """Storage interface for raw session JSONL data."""

    def __init__(self, db_session: Session, user_id: str) -> None:
        self._db = db_session
        self._user_id = user_id

    def get_cursor(self, session_id: str) -> SessionCursor:
        """Get the current cursor and sync state for a session.

        Memory-efficient: doesn't load content, just reads metadata.
        Returns byte_offset=0 if session doesn't exist yet.

        Note: byte_offset is the original file position, not the stored content
        length. This is because tool results are stripped before storage, so the
        stored content is shorter than the original file.
        """
        stmt = select(
            SessionRawData.original_file_offset,
            SessionRawData.last_synced_uuid,
            SessionRawData.span_end,
        ).where(
            SessionRawData.user_id == self._user_id,
            SessionRawData.session_id == session_id,
        )
        result = self._db.execute(stmt).one_or_none()

        if result is None:
            return SessionCursor(byte_offset=0)

        file_offset, last_uuid, span_end = result
        return SessionCursor(
            byte_offset=file_offset or 0,
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

        The delta is stripped of tool result content before storage to reduce
        disk usage by ~80%. Creates the session if it doesn't exist.
        Returns the new original file offset (for client seeking).
        """
        # Strip tool result content before storing
        stripped_delta = strip_tool_results(delta)
        original_delta_size = len(delta)

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
                jsonl_content=stripped_delta,
                original_file_offset=original_delta_size,
            )
            self._db.add(row)
            self._db.flush()
            return original_delta_size

        # Append to existing
        new_content = row.jsonl_content + stripped_delta
        row.jsonl_content = new_content
        row.original_file_offset = row.original_file_offset + original_delta_size
        self._db.flush()
        return row.original_file_offset

    def truncate_content(self, session_id: str, byte_offset: int) -> None:
        """Truncate session content to the given byte offset.

        Used when handling reverts in the transcript.
        Resets sync state and original_file_offset since we need to re-sync.

        Note: byte_offset here refers to stored content position, not original
        file position. After truncation, original_file_offset is set to 0 to
        force a full re-sync from the client.
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
            row.original_file_offset = 0  # Force full re-sync from client
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
