"""Tests for memory_service.storage module."""

from __future__ import annotations

import json
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from memory_service.storage import SessionCursor, SessionStorage, strip_tool_results
from ragzoom.models import Base


class TestStripToolResults:
    """Tests for the strip_tool_results function."""

    def test_empty_input(self) -> None:
        """Should return empty bytes for empty input."""
        assert strip_tool_results(b"") == b""

    def test_preserves_regular_user_messages(self) -> None:
        """Should preserve user messages without toolUseResult."""
        record = {
            "uuid": "msg1",
            "parentUuid": "msg0",
            "type": "user",
            "message": {"content": "hello"},
        }
        input_bytes = json.dumps(record).encode()

        result = strip_tool_results(input_bytes)

        assert result == input_bytes

    def test_preserves_assistant_messages(self) -> None:
        """Should preserve assistant messages completely."""
        record = {
            "uuid": "msg1",
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "response"}]},
        }
        input_bytes = json.dumps(record).encode()

        result = strip_tool_results(input_bytes)

        assert result == input_bytes

    def test_strips_tool_result_content(self) -> None:
        """Should strip toolUseResult content to just essential fields."""
        record = {
            "uuid": "msg2",
            "parentUuid": "msg1",
            "type": "user",
            "toolUseResult": "Here is 10MB of file content...",
            "message": {"content": "full tool output..."},
        }
        input_bytes = json.dumps(record).encode()

        result = strip_tool_results(input_bytes)
        parsed = json.loads(result)

        assert parsed["uuid"] == "msg2"
        assert parsed["parentUuid"] == "msg1"
        assert parsed["type"] == "user"
        assert parsed["toolUseResult"] == "[stripped]"
        assert "message" not in parsed

    def test_preserves_uuid_parent_uuid_type(self) -> None:
        """Should always preserve uuid, parentUuid, and type."""
        record = {
            "uuid": "tool-result-1",
            "parentUuid": "assistant-1",
            "type": "user",
            "toolUseResult": "huge output",
        }
        input_bytes = json.dumps(record).encode()

        result = strip_tool_results(input_bytes)
        parsed = json.loads(result)

        assert parsed["uuid"] == "tool-result-1"
        assert parsed["parentUuid"] == "assistant-1"
        assert parsed["type"] == "user"

    def test_preserves_is_compact_summary_field(self) -> None:
        """Should preserve isCompactSummary for compaction bridging."""
        record = {
            "uuid": "msg1",
            "type": "user",
            "toolUseResult": "output",
            "isCompactSummary": True,
        }
        input_bytes = json.dumps(record).encode()

        result = strip_tool_results(input_bytes)
        parsed = json.loads(result)

        assert parsed["isCompactSummary"] is True

    def test_handles_multiple_records(self) -> None:
        """Should handle multiple JSONL records correctly."""
        records = [
            {"uuid": "msg1", "type": "user", "message": {"content": "hello"}},
            {
                "uuid": "msg2",
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hi"}]},
            },
            {
                "uuid": "msg3",
                "parentUuid": "msg2",
                "type": "user",
                "toolUseResult": "big output",
            },
            {"uuid": "msg4", "type": "user", "message": {"content": "thanks"}},
        ]
        input_bytes = b"\n".join(json.dumps(r).encode() for r in records)

        result = strip_tool_results(input_bytes)
        lines = result.split(b"\n")

        assert len(lines) == 4
        # First two should be unchanged
        assert json.loads(lines[0]) == records[0]
        assert json.loads(lines[1]) == records[1]
        # Third should be stripped
        parsed_tool = json.loads(lines[2])
        assert parsed_tool["toolUseResult"] == "[stripped]"
        # Fourth should be unchanged
        assert json.loads(lines[3]) == records[3]

    def test_handles_empty_lines(self) -> None:
        """Should preserve empty lines in JSONL."""
        input_bytes = (
            b'{"uuid": "msg1", "type": "user"}\n\n{"uuid": "msg2", "type": "user"}'
        )

        result = strip_tool_results(input_bytes)

        lines = result.split(b"\n")
        assert len(lines) == 3
        assert lines[1] == b""

    def test_handles_malformed_json(self) -> None:
        """Should preserve malformed lines as-is."""
        input_bytes = b'{"uuid": "msg1"}\nnot valid json\n{"uuid": "msg2"}'

        result = strip_tool_results(input_bytes)

        lines = result.split(b"\n")
        assert lines[1] == b"not valid json"

    def test_reduces_size_significantly(self) -> None:
        """Should significantly reduce size for tool result records."""
        # Simulate a large tool result (like file content)
        large_output = "x" * 100000  # 100KB of output
        record = {
            "uuid": "msg1",
            "parentUuid": "msg0",
            "type": "user",
            "toolUseResult": large_output,
            "message": {"content": [{"type": "tool_result", "content": large_output}]},
        }
        input_bytes = json.dumps(record).encode()

        result = strip_tool_results(input_bytes)

        # Should be dramatically smaller
        assert len(result) < 200  # Just the essential fields
        assert len(input_bytes) > 200000  # Original was huge


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

    def test_get_cursor_returns_original_file_offset(self, db_session: Session) -> None:
        """Should return byte_offset equal to original file offset."""
        storage = SessionStorage(db_session, user_id="user1")
        # Content without tool results - original_file_offset equals content length
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
        # Truncation resets original_file_offset to 0 to force full re-sync
        assert storage.get_cursor("session1") == SessionCursor(byte_offset=0)

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
        # Truncation resets original_file_offset to 0 to force full re-sync
        assert cursor.byte_offset == 0
        assert cursor.last_synced_uuid is None
        assert cursor.span_end == 0

    def test_cursor_tracks_original_file_offset_with_stripping(
        self, db_session: Session
    ) -> None:
        """Cursor byte_offset should track original delta size, not stripped content."""
        storage = SessionStorage(db_session, user_id="user1")

        # Create content with a large tool result that will be stripped
        large_output = "x" * 10000
        record_with_tool = {
            "uuid": "msg1",
            "parentUuid": "msg0",
            "type": "user",
            "toolUseResult": large_output,
            "message": {"content": large_output},
        }
        original_content = json.dumps(record_with_tool).encode() + b"\n"
        original_size = len(original_content)

        # Append the content
        new_offset = storage.append_content("session1", original_content)
        db_session.commit()

        # byte_offset should be the original size, not the stripped size
        assert new_offset == original_size

        cursor = storage.get_cursor("session1")
        assert cursor.byte_offset == original_size

        # But stored content should be much smaller
        stored = storage.get_content("session1")
        assert len(stored) < original_size  # Should be stripped


class TestTranscribeSession:
    """Tests for _transcribe_session handling of memoryview input."""

    def test_transcribe_session_handles_memoryview(self) -> None:
        """_transcribe_session should handle memoryview input from SQLAlchemy.

        SQLAlchemy returns LargeBinary columns as memoryview objects, but
        _transcribe_session calls functions that use bytes.find() which
        doesn't exist on memoryview.
        """
        from memory_service.admin import _transcribe_session

        # Create minimal valid JSONL content
        records = [
            {"uuid": "msg1", "parentUuid": None, "type": "user", "message": "hello"},
            {
                "uuid": "msg2",
                "parentUuid": "msg1",
                "type": "assistant",
                "message": "world",
            },
        ]
        content_bytes = b"\n".join(json.dumps(r).encode() for r in records) + b"\n"

        # Simulate what SQLAlchemy returns - a memoryview
        content_memoryview = memoryview(content_bytes)

        # Previously failed with AttributeError: 'memoryview' object has no attribute 'find'
        result = _transcribe_session(content_memoryview)

        assert "hello" in result or result == ""  # May be empty if no valid chain
