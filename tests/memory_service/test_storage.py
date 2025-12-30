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

        assert cursor == SessionCursor(byte_offset=0)

    def test_get_cursor_returns_content_length(self, db_session: Session) -> None:
        """Should return byte_offset equal to stored content length."""
        storage = SessionStorage(db_session, user_id="user1")
        content = b'{"uuid": "msg1"}\n{"uuid": "msg2"}\n'
        storage.append_content("session1", content)
        db_session.commit()

        cursor = storage.get_cursor("session1")

        assert cursor == SessionCursor(byte_offset=len(content))

    def test_get_content_returns_empty_for_new_session(
        self, db_session: Session
    ) -> None:
        """Should return empty bytes for non-existent session."""
        storage = SessionStorage(db_session, user_id="user1")

        content = storage.get_content("session1")

        assert content == b""

    def test_get_content_returns_stored_content(self, db_session: Session) -> None:
        """Should return the full stored JSONL content."""
        storage = SessionStorage(db_session, user_id="user1")
        expected = b'{"uuid": "msg1"}\n{"uuid": "msg2"}\n'
        storage.append_content("session1", expected)
        db_session.commit()

        content = storage.get_content("session1")

        assert content == expected

    def test_append_content_creates_new_session(self, db_session: Session) -> None:
        """Should create a new session record on first append."""
        storage = SessionStorage(db_session, user_id="user1")
        content = b'{"uuid": "msg1"}\n'

        new_offset = storage.append_content("session1", content)
        db_session.commit()

        assert new_offset == len(content)
        assert storage.get_content("session1") == content

    def test_append_content_appends_to_existing(self, db_session: Session) -> None:
        """Should append to existing content."""
        storage = SessionStorage(db_session, user_id="user1")
        first = b'{"uuid": "msg1"}\n'
        second = b'{"uuid": "msg2"}\n'
        storage.append_content("session1", first)
        db_session.commit()

        new_offset = storage.append_content("session1", second)
        db_session.commit()

        assert new_offset == len(first) + len(second)
        assert storage.get_content("session1") == first + second

    def test_truncate_content_removes_after_offset(self, db_session: Session) -> None:
        """Should truncate content at the given byte offset."""
        storage = SessionStorage(db_session, user_id="user1")
        first = b'{"uuid": "msg1"}\n'
        second = b'{"uuid": "msg2"}\n'
        storage.append_content("session1", first + second)
        db_session.commit()

        storage.truncate_content("session1", len(first))
        db_session.commit()

        assert storage.get_content("session1") == first
        assert storage.get_cursor("session1") == SessionCursor(byte_offset=len(first))

    def test_truncate_content_noop_if_offset_beyond_content(
        self, db_session: Session
    ) -> None:
        """Should do nothing if offset is beyond content length."""
        storage = SessionStorage(db_session, user_id="user1")
        content = b'{"uuid": "msg1"}\n'
        storage.append_content("session1", content)
        db_session.commit()

        storage.truncate_content("session1", len(content) + 100)
        db_session.commit()

        assert storage.get_content("session1") == content

    def test_truncate_content_noop_if_session_missing(
        self, db_session: Session
    ) -> None:
        """Should do nothing if session doesn't exist."""
        storage = SessionStorage(db_session, user_id="user1")

        # Should not raise
        storage.truncate_content("nonexistent", 0)

    def test_user_isolation(self, db_session: Session) -> None:
        """Different users should have isolated sessions."""
        storage1 = SessionStorage(db_session, user_id="user1")
        storage2 = SessionStorage(db_session, user_id="user2")
        content1 = b'{"user": "1"}\n'
        content2 = b'{"user": "2"}\n'

        storage1.append_content("session1", content1)
        storage2.append_content("session1", content2)
        db_session.commit()

        assert storage1.get_content("session1") == content1
        assert storage2.get_content("session1") == content2

    def test_session_isolation(self, db_session: Session) -> None:
        """Different sessions for the same user should be isolated."""
        storage = SessionStorage(db_session, user_id="user1")
        content_a = b'{"session": "a"}\n'
        content_b = b'{"session": "b"}\n'

        storage.append_content("session_a", content_a)
        storage.append_content("session_b", content_b)
        db_session.commit()

        assert storage.get_content("session_a") == content_a
        assert storage.get_content("session_b") == content_b

    def test_update_sync_state(self, db_session: Session) -> None:
        """Should update sync state after successful processing."""
        storage = SessionStorage(db_session, user_id="user1")
        storage.append_content("session1", b'{"uuid": "msg1"}\n')
        db_session.commit()

        storage.update_sync_state("session1", last_synced_uuid="msg1", span_end=100)
        db_session.commit()

        cursor = storage.get_cursor("session1")
        assert cursor.last_synced_uuid == "msg1"
        assert cursor.span_end == 100

    def test_get_cursor_returns_sync_state(self, db_session: Session) -> None:
        """Should return sync state alongside byte offset."""
        storage = SessionStorage(db_session, user_id="user1")
        content = b'{"uuid": "msg1"}\n{"uuid": "msg2"}\n'
        storage.append_content("session1", content)
        storage.update_sync_state("session1", last_synced_uuid="msg2", span_end=200)
        db_session.commit()

        cursor = storage.get_cursor("session1")

        assert cursor.byte_offset == len(content)
        assert cursor.last_synced_uuid == "msg2"
        assert cursor.span_end == 200

    def test_truncate_content_resets_sync_state(self, db_session: Session) -> None:
        """Truncation should reset sync state for re-sync."""
        storage = SessionStorage(db_session, user_id="user1")
        content = b'{"uuid": "msg1"}\n{"uuid": "msg2"}\n'
        storage.append_content("session1", content)
        storage.update_sync_state("session1", last_synced_uuid="msg2", span_end=200)
        db_session.commit()

        # Truncate to just the first message
        storage.truncate_content("session1", 17)  # len('{"uuid": "msg1"}\n')
        db_session.commit()

        cursor = storage.get_cursor("session1")
        assert cursor.byte_offset == 17
        assert cursor.last_synced_uuid is None
        assert cursor.span_end == 0
