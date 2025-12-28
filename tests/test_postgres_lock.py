from __future__ import annotations

from typing import cast

import pytest
from sqlalchemy.engine import Engine

from ragzoom.backends.postgres_backend import _AdvisoryLock, _hash_document_lock


class DummyConn:
    def __init__(self, recorder: list[tuple[str, dict[str, object] | None]]):
        self._recorder = recorder

    def execute(
        self,
        statement: object,
        params: dict[str, object] | None = None,
    ) -> DummyConn:
        self._recorder.append((str(statement), params))
        return self

    def close(self) -> None:  # noqa: D401 - trivial helper
        self._recorder.append(("close", None))

    def fetchall(self) -> list[object]:  # pragma: no cover - compatibility
        return []


class DummyEngine:
    def __init__(self, recorder: list[tuple[str, dict[str, object] | None]]):
        self._recorder = recorder

    def connect(self) -> DummyConn:  # noqa: D401 - trivial helper
        self._recorder.append(("connect", None))
        return DummyConn(self._recorder)


def test_hash_document_lock_ranges() -> None:
    lock_key = _hash_document_lock("example-document")
    assert -(2**63) <= lock_key < 2**63


def test_advisory_lock_acquires_and_releases() -> None:
    calls: list[tuple[str, dict[str, object] | None]] = []
    engine = DummyEngine(calls)
    lock = _AdvisoryLock(cast(Engine, engine), 12345)

    with lock:
        calls.append(("inside", None))

    assert calls[0][0] == "connect"
    assert "pg_advisory_lock" in calls[1][0]
    assert calls[-2][0].startswith("SELECT pg_advisory_unlock")
    assert calls[-1][0] == "close"


def test_advisory_lock_unlocks_on_exception() -> None:
    calls: list[tuple[str, dict[str, object] | None]] = []
    engine = DummyEngine(calls)
    lock = _AdvisoryLock(cast(Engine, engine), 67890)

    with pytest.raises(RuntimeError, match="boom"):
        with lock:
            raise RuntimeError("boom")

    assert calls[0][0] == "connect"
    assert "pg_advisory_unlock" in calls[-2][0]
    assert calls[-1][0] == "close"
