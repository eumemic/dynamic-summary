"""In-memory session store for search continuations."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from ragzoom.agent.protocol import MessageHistory


@dataclass
class SessionState:
    """Server-side state for a search session."""

    session_id: str
    document_id: str
    history: MessageHistory
    last_accessed_at: float
    turn_count: int = 1


class SessionStore:
    """In-memory session store with TTL-based expiry.

    Sessions are keyed by a UUID and expire after ``ttl_seconds`` of inactivity.
    """

    def __init__(self, ttl_seconds: float = 1800.0) -> None:
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, SessionState] = {}

    def create(self, document_id: str, history: MessageHistory) -> str:
        """Create a new session and return its ID."""
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = SessionState(
            session_id=session_id,
            document_id=document_id,
            history=history,
            last_accessed_at=time.monotonic(),
        )
        return session_id

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

    def update(self, session_id: str, history: MessageHistory) -> None:
        """Update session history after a follow-up turn."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        session.history = history
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
