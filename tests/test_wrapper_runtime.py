"""Tests for RagZoom wrappers when using the local indexing runtime."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ragzoom.indexing import ClearedDocumentResult
from ragzoom.indexing.runtime import TruncateResult
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.wrapper import AsyncRagZoom, RagZoom, _SessionProtocol


class _StubSession:
    def __init__(self, result: IndexingResult) -> None:
        self._result = result
        self.append_calls: list[dict[str, object]] = []
        self.clear_calls = 0
        self.document_id: str | None = None

    async def append_text(
        self,
        text: str,
        *,
        replace_existing: bool,
        collect_telemetry: bool,
    ) -> IndexingResult:
        self.append_calls.append(
            {
                "text": text,
                "replace_existing": replace_existing,
                "collect_telemetry": collect_telemetry,
            }
        )
        return self._result

    async def clear(self) -> ClearedDocumentResult:
        self.clear_calls += 1
        return ClearedDocumentResult(
            document_id=self.document_id or "unknown",
            deleted_nodes=0,
            document_existed=True,
        )

    async def truncate_from_span(self, span_start: int) -> TruncateResult:
        return TruncateResult(
            document_id=self.document_id or "unknown",
            deleted_node_ids=[],
            span_start=span_start,
        )


class _StubRuntime:
    def __init__(self, session: _StubSession) -> None:
        self._session = session
        self.requests: list[tuple[str, str | None]] = []

    def get_session(
        self, document_id: str, *, file_path: str | None = None
    ) -> _SessionProtocol:
        self.requests.append((document_id, file_path))
        self._session.document_id = document_id
        return self._session


def test_ragzoom_uses_runtime_for_append_and_clear() -> None:
    expected = IndexingResult(
        document_id="doc",
        chunks_created=3,
        tree_depth=2,
    )
    session = _StubSession(expected)
    runtime = _StubRuntime(session)

    with patch("ragzoom.wrapper.GrpcRagzoomClient") as client_mock:
        wrapper = RagZoom(runtime=runtime)
        result = wrapper.append("doc", "hello world")
        wrapper.clear("doc")

    assert result == expected
    assert not client_mock.called
    assert runtime.requests == [("doc", None), ("doc", None)]
    assert session.append_calls[0]["replace_existing"] is False
    assert session.append_calls[0]["collect_telemetry"] is False
    assert session.clear_calls == 1


def test_ragzoom_index_sets_replace_existing() -> None:
    expected = IndexingResult(
        document_id="doc",
        chunks_created=1,
        tree_depth=0,
    )
    session = _StubSession(expected)
    runtime = _StubRuntime(session)

    with patch("ragzoom.wrapper.GrpcRagzoomClient") as client_mock:
        wrapper = RagZoom(runtime=runtime)
        result = wrapper.index("doc", "body")

    assert result == expected
    assert not client_mock.called
    assert session.append_calls[0]["replace_existing"] is True


def test_ragzoom_query_requires_address_without_client() -> None:
    session = _StubSession(
        IndexingResult(document_id="doc", chunks_created=0, tree_depth=0)
    )
    runtime = _StubRuntime(session)
    wrapper = RagZoom(runtime=runtime)

    with pytest.raises(RuntimeError):
        wrapper.query("doc", "query")


@pytest.mark.asyncio
async def test_async_ragzoom_uses_runtime() -> None:
    expected = IndexingResult(
        document_id="doc",
        chunks_created=2,
        tree_depth=1,
    )
    session = _StubSession(expected)
    runtime = _StubRuntime(session)

    with patch("ragzoom.wrapper.GrpcRagzoomClient") as client_mock:
        wrapper = AsyncRagZoom(runtime=runtime)
        result = await wrapper.append("doc", "async body")

    assert result == expected
    assert not client_mock.called
    assert session.append_calls[0]["replace_existing"] is False
