"""In-memory session registry for search continuations."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class SessionState:
    """Server-side state for a search session.

    The backend owns the conversation state; this registry only tracks
    which document a session belongs to for routing follow-up queries.
    """

    session_id: str
    document_id: str
    last_accessed_at: float
    turn_count: int = 1


class SessionRegistry:
    """In-memory session registry with TTL-based expiry.

    Maps backend-generated session IDs to document IDs.  The backends
    own conversation state; this registry is a lightweight routing table.
    """

    def __init__(self, ttl_seconds: float = 1800.0) -> None:
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, SessionState] = {}

    def create(self, session_id: str, document_id: str) -> None:
        """Register a new session."""
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            document_id=document_id,
            last_accessed_at=time.monotonic(),
        )

    def get(self, session_id: str) -> SessionState | None:
        """Look up a session, returning None if expired or missing."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        if time.monotonic() - session.last_accessed_at > self._ttl_seconds:
            del self._sessions[session_id]
            return None
        session.last_accessed_at = time.monotonic()
        return session

    def update(self, session_id: str) -> None:
        """Refresh timestamp and increment turn count after a follow-up."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        session.turn_count += 1
        session.last_accessed_at = time.monotonic()

    def cleanup_expired(self) -> int:
        """Remove all expired sessions. Returns count of removed sessions."""
        now = time.monotonic()
        expired = [
            sid
            for sid, state in self._sessions.items()
            if now - state.last_accessed_at > self._ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)
