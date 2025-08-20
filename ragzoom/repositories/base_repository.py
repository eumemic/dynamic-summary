"""Base repository class for common database operations."""


class BaseRepository:
    """Base class for repository implementations with common session management."""

    def _get_session(self, session=None):
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