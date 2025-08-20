"""Base repository class for common database operations."""

from collections.abc import Generator
from contextlib import contextmanager
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

    @contextmanager
    def _session_scope(
        self, session: Optional["Session"] = None
    ) -> Generator["Session", None, None]:
        """Context manager for safe session handling with proper rollback.

        Args:
            session: Optional existing session to use

        Yields:
            SQLAlchemy session

        Handles:
            - Automatic commit when managing session lifecycle
            - Automatic rollback on exceptions when managing session lifecycle
            - Proper session cleanup
        """
        db_session, should_commit = self._get_session(session)
        try:
            yield db_session
            if should_commit:
                db_session.commit()
        except Exception:
            if should_commit:
                db_session.rollback()
            raise
        finally:
            if should_commit:
                db_session.close()
