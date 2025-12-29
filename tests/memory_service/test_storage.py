"""Tests for memory_service.storage module."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from memory_service.storage import SessionCursor, SessionStorage
from ragzoom.models import Base


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """Create an in-memory SQLite database with the schema."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory: sessionmaker[Session] = sessionmaker(bind=engine)
    session = session_factory()
    yield session
    session.close()


class TestSessionStorage:
    """Tests for SessionStorage class."""

    def test_get_cursor_returns_zero_for_new_session(self, db_session: Session) -> None:
        """Should return byte_offset=0 for non-existent session."""
        storage = SessionStorage(db_session, user_id="user1")

        cursor = storage.get_cursor("session1")

        assert cursor == SessionCursor(byte_offset=0, last_uuid=None, span_end=0)

    def test_get_cursor_returns_stored_state(self, db_session: Session) -> None:
        """Should return the stored cursor state."""
        storage = SessionStorage(db_session, user_id="user1")
        storage.update_cursor(
            session_id="session1",
            byte_offset=1234,
            last_uuid="uuid-abc",
            span_end=5678,
        )
        db_session.commit()

        cursor = storage.get_cursor("session1")

        assert cursor == SessionCursor(
            byte_offset=1234, last_uuid="uuid-abc", span_end=5678
        )

    def test_update_cursor_creates_new_session(self, db_session: Session) -> None:
        """Should create a new session record on first update."""
        storage = SessionStorage(db_session, user_id="user1")

        storage.update_cursor(
            session_id="session1",
            byte_offset=100,
            last_uuid="uuid-1",
            span_end=200,
        )
        db_session.commit()

        cursor = storage.get_cursor("session1")
        assert cursor.byte_offset == 100
        assert cursor.last_uuid == "uuid-1"
        assert cursor.span_end == 200

    def test_update_cursor_updates_existing(self, db_session: Session) -> None:
        """Should update existing cursor state."""
        storage = SessionStorage(db_session, user_id="user1")
        storage.update_cursor(
            session_id="session1",
            byte_offset=100,
            last_uuid="uuid-1",
            span_end=200,
        )
        db_session.commit()

        storage.update_cursor(
            session_id="session1",
            byte_offset=500,
            last_uuid="uuid-5",
            span_end=1000,
        )
        db_session.commit()

        cursor = storage.get_cursor("session1")
        assert cursor.byte_offset == 500
        assert cursor.last_uuid == "uuid-5"
        assert cursor.span_end == 1000

    def test_reset_cursor_deletes_state(self, db_session: Session) -> None:
        """Should delete the cursor state, returning to initial state."""
        storage = SessionStorage(db_session, user_id="user1")
        storage.update_cursor(
            session_id="session1",
            byte_offset=100,
            last_uuid="uuid-1",
            span_end=200,
        )
        db_session.commit()

        storage.reset_cursor("session1")
        db_session.commit()

        cursor = storage.get_cursor("session1")
        assert cursor == SessionCursor(byte_offset=0, last_uuid=None, span_end=0)

    def test_reset_cursor_noop_if_session_missing(self, db_session: Session) -> None:
        """Should do nothing if session doesn't exist."""
        storage = SessionStorage(db_session, user_id="user1")

        # Should not raise
        storage.reset_cursor("nonexistent")

    def test_user_isolation(self, db_session: Session) -> None:
        """Different users should have isolated sessions."""
        storage1 = SessionStorage(db_session, user_id="user1")
        storage2 = SessionStorage(db_session, user_id="user2")

        storage1.update_cursor(
            session_id="session1",
            byte_offset=100,
            last_uuid="uuid-user1",
            span_end=200,
        )
        storage2.update_cursor(
            session_id="session1",
            byte_offset=300,
            last_uuid="uuid-user2",
            span_end=400,
        )
        db_session.commit()

        cursor1 = storage1.get_cursor("session1")
        cursor2 = storage2.get_cursor("session1")

        assert cursor1.byte_offset == 100
        assert cursor1.last_uuid == "uuid-user1"
        assert cursor2.byte_offset == 300
        assert cursor2.last_uuid == "uuid-user2"

    def test_session_isolation(self, db_session: Session) -> None:
        """Different sessions for the same user should be isolated."""
        storage = SessionStorage(db_session, user_id="user1")

        storage.update_cursor(
            session_id="session_a",
            byte_offset=100,
            last_uuid="uuid-a",
            span_end=200,
        )
        storage.update_cursor(
            session_id="session_b",
            byte_offset=300,
            last_uuid="uuid-b",
            span_end=400,
        )
        db_session.commit()

        cursor_a = storage.get_cursor("session_a")
        cursor_b = storage.get_cursor("session_b")

        assert cursor_a.byte_offset == 100
        assert cursor_a.last_uuid == "uuid-a"
        assert cursor_b.byte_offset == 300
        assert cursor_b.last_uuid == "uuid-b"

    def test_update_cursor_handles_none_last_uuid(self, db_session: Session) -> None:
        """Should handle None as last_uuid (initial state)."""
        storage = SessionStorage(db_session, user_id="user1")

        storage.update_cursor(
            session_id="session1",
            byte_offset=0,
            last_uuid=None,
            span_end=0,
        )
        db_session.commit()

        cursor = storage.get_cursor("session1")
        assert cursor.last_uuid is None
