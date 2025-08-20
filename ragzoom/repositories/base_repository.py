"""Base repository class for common database operations."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sqlalchemy.orm import Session, sessionmaker


class BaseRepository:
    """Base class for repository implementations with common session management."""

    SessionLocal: "sessionmaker[Session]"

    def _get_session(
        self, session: Optional["Session"] = None
    ) -> tuple["Session", bool]:
        """Get session for database operations.

        Args:
            session: Optional existing session to use

        Returns:
            Tuple of (session, should_commit) where should_commit indicates
            if this method should handle commit/rollback
        """
        if session is not None:
            return session, False  # Don't commit - caller manages lifecycle
        else:
            return self.SessionLocal(), True  # We manage lifecycle
