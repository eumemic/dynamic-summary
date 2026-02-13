"""Tests for the in-memory SessionRegistry."""

from __future__ import annotations

import time

import pytest

from ragzoom.search.session import SessionRegistry


class TestSessionCreate:
    def test_create_stores_session(self) -> None:
        registry = SessionRegistry()
        registry.create("sess-1", "doc-1")

        session = registry.get("sess-1")

        assert session is not None
        assert session.session_id == "sess-1"
        assert session.document_id == "doc-1"
        assert session.turn_count == 1


class TestSessionGet:
    def test_get_returns_session(self) -> None:
        registry = SessionRegistry()
        registry.create("sess-1", "doc-1")

        session = registry.get("sess-1")

        assert session is not None
        assert session.session_id == "sess-1"
        assert session.document_id == "doc-1"
        assert session.turn_count == 1

    def test_get_missing_returns_none(self) -> None:
        registry = SessionRegistry()
        assert registry.get("nonexistent") is None

    def test_get_expired_returns_none(self) -> None:
        registry = SessionRegistry(ttl_seconds=0.01)
        registry.create("sess-1", "doc-1")
        time.sleep(0.02)
        assert registry.get("sess-1") is None

    def test_get_refreshes_ttl(self) -> None:
        registry = SessionRegistry(ttl_seconds=0.1)
        registry.create("sess-1", "doc-1")
        # Access before expiry to refresh
        time.sleep(0.06)
        assert registry.get("sess-1") is not None
        # Access again — still alive because TTL was refreshed
        time.sleep(0.06)
        assert registry.get("sess-1") is not None


class TestSessionUpdate:
    def test_update_increments_turn_count(self) -> None:
        registry = SessionRegistry()
        registry.create("sess-1", "doc-1")

        registry.update("sess-1")

        session = registry.get("sess-1")
        assert session is not None
        assert session.turn_count == 2

    def test_update_missing_raises(self) -> None:
        registry = SessionRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.update("nonexistent")


class TestCleanupExpired:
    def test_cleanup_removes_expired(self) -> None:
        registry = SessionRegistry(ttl_seconds=0.01)
        registry.create("sess-1", "doc-1")
        registry.create("sess-2", "doc-2")
        time.sleep(0.02)

        removed = registry.cleanup_expired()

        assert removed == 2

    def test_cleanup_preserves_active(self) -> None:
        registry = SessionRegistry(ttl_seconds=10.0)
        registry.create("sess-1", "doc-1")

        removed = registry.cleanup_expired()

        assert removed == 0
        assert registry.get("sess-1") is not None
