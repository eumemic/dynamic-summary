"""Tests for the in-memory SessionStore."""

from __future__ import annotations

import time

import pytest

from ragzoom.agent.protocol import AssistantTurn, MessageHistory
from ragzoom.search.session import SessionStore


def _sample_history() -> MessageHistory:
    return ("Hello", AssistantTurn(text="Hi there"))


class TestSessionCreate:
    def test_create_returns_session_id(self) -> None:
        store = SessionStore()
        sid = store.create("doc-1", _sample_history())
        assert isinstance(sid, str)
        assert len(sid) > 0

    def test_create_unique_ids(self) -> None:
        store = SessionStore()
        sid1 = store.create("doc-1", _sample_history())
        sid2 = store.create("doc-1", _sample_history())
        assert sid1 != sid2


class TestSessionGet:
    def test_get_returns_session(self) -> None:
        store = SessionStore()
        history = _sample_history()
        sid = store.create("doc-1", history)

        session = store.get(sid)

        assert session is not None
        assert session.session_id == sid
        assert session.document_id == "doc-1"
        assert session.history == history
        assert session.turn_count == 1

    def test_get_missing_returns_none(self) -> None:
        store = SessionStore()
        assert store.get("nonexistent") is None

    def test_get_expired_returns_none(self) -> None:
        store = SessionStore(ttl_seconds=0.01)
        sid = store.create("doc-1", _sample_history())
        time.sleep(0.02)
        assert store.get(sid) is None

    def test_get_refreshes_ttl(self) -> None:
        store = SessionStore(ttl_seconds=0.1)
        sid = store.create("doc-1", _sample_history())
        # Access before expiry to refresh
        time.sleep(0.06)
        assert store.get(sid) is not None
        # Access again — still alive because TTL was refreshed
        time.sleep(0.06)
        assert store.get(sid) is not None


class TestSessionUpdate:
    def test_update_replaces_history(self) -> None:
        store = SessionStore()
        sid = store.create("doc-1", _sample_history())

        new_history: MessageHistory = ("Follow-up", AssistantTurn(text="Sure"))
        store.update(sid, new_history)

        session = store.get(sid)
        assert session is not None
        assert session.history == new_history
        assert session.turn_count == 2

    def test_update_missing_raises(self) -> None:
        store = SessionStore()
        with pytest.raises(KeyError, match="not found"):
            store.update("nonexistent", _sample_history())


class TestCleanupExpired:
    def test_cleanup_removes_expired(self) -> None:
        store = SessionStore(ttl_seconds=0.01)
        store.create("doc-1", _sample_history())
        store.create("doc-2", _sample_history())
        time.sleep(0.02)

        removed = store.cleanup_expired()

        assert removed == 2

    def test_cleanup_preserves_active(self) -> None:
        store = SessionStore(ttl_seconds=10.0)
        sid = store.create("doc-1", _sample_history())

        removed = store.cleanup_expired()

        assert removed == 0
        assert store.get(sid) is not None
