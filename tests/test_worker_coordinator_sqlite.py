import asyncio
from collections.abc import Generator

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.server.worker_coordinator import (
    ReadyParentCandidate,
    WorkerCoordinator,
    compute_ready_parent_candidates,
)
from ragzoom.vector_api import Vector


@pytest.fixture()
def sqlite_store() -> Generator[DocumentStore, None, None]:
    backend = SQLiteStorageBackend()
    try:
        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-5-nano",
        )
        yield backend.for_document("doc")
    finally:
        backend.close()


NodePayloadValue = str | int | float | bool | list[float] | NDArray[np.float64] | None
NodePayload = dict[str, NodePayloadValue]


def _leaf_payload(
    node_id: str,
    span_start: int,
    span_end: int,
    **kwargs: NodePayloadValue,
) -> NodePayload:
    payload: NodePayload = {
        "node_id": node_id,
        "text": f"text-{node_id}",
        "span_start": span_start,
        "span_end": span_end,
        "document_id": "doc",
        "token_count": span_end - span_start,
        "height": 0,
        "parent_id": None,
        "left_child_id": None,
        "right_child_id": None,
    }
    for key, value in kwargs.items():
        payload[key] = value
    return payload


def _add_batch(store: DocumentStore, *payloads: NodePayload) -> None:
    batch: list[
        dict[str, str | int | float | bool | list[float] | NDArray[np.float64] | None]
    ] = [dict(payload) for payload in payloads]
    store.nodes.add_batch(batch)


def test_detects_leaf_pairs(sqlite_store: DocumentStore) -> None:
    _add_batch(
        sqlite_store,
        _leaf_payload("L", 0, 10, following_neighbor_id="R"),
        _leaf_payload("R", 10, 20, preceding_neighbor_id="L"),
    )

    candidates = compute_ready_parent_candidates(sqlite_store)

    assert candidates == [
        ReadyParentCandidate(
            document_id="doc",
            left_child_id="L",
            right_child_id="R",
            height=0,
            span_start=0,
            span_end=20,
        )
    ]


def test_skips_nodes_with_existing_parents(sqlite_store: DocumentStore) -> None:
    _add_batch(
        sqlite_store,
        _leaf_payload("a", 0, 5, following_neighbor_id="b", parent_id="p"),
        _leaf_payload("b", 5, 10, preceding_neighbor_id="a", parent_id="p"),
    )

    assert compute_ready_parent_candidates(sqlite_store) == []


def test_returns_single_candidate_for_full_span_root(
    sqlite_store: DocumentStore,
) -> None:
    _add_batch(sqlite_store, _leaf_payload("root", 0, 12))

    candidates = compute_ready_parent_candidates(sqlite_store)
    assert candidates == [
        ReadyParentCandidate(
            document_id="doc",
            left_child_id="root",
            right_child_id=None,
            height=0,
            span_start=0,
            span_end=12,
        )
    ]


def test_requires_bidirectional_neighbor_link(sqlite_store: DocumentStore) -> None:
    _add_batch(
        sqlite_store,
        _leaf_payload("left", 0, 10, following_neighbor_id="right"),
        _leaf_payload("right", 10, 20),
    )

    assert compute_ready_parent_candidates(sqlite_store) == []


class StubLLMService:
    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object | None = None,
        prev_context: str | None = None,
        left_token_count: int | None = None,
        right_token_count: int | None = None,
    ) -> tuple[str, int, int]:
        pid = parent_id or "parent"
        summary = f"summary-{pid[:8]}"
        return summary, 0, len(summary.split())

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


class StubVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self.upserts: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = []

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[Vector]:  # pragma: no cover - unused
        raise NotImplementedError

    def get_vectors(self, ids: list[str]) -> list[Vector]:  # pragma: no cover - unused
        return []

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        self.upserts.extend(items)

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:  # pragma: no cover - unused
        return 0


class BlockingLLMService(StubLLMService):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self._release = asyncio.Event()

    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: object | None = None,
        prev_context: str | None = None,
        left_token_count: int | None = None,
        right_token_count: int | None = None,
    ) -> tuple[str, int, int]:
        self.started.set()
        await self._release.wait()
        return await super()._summarize_text(
            left_text,
            right_text,
            target_tokens,
            parent_id=parent_id,
            reporter=reporter,
            prev_context=prev_context,
            left_token_count=left_token_count,
            right_token_count=right_token_count,
        )

    def release(self) -> None:
        self._release.set()


@pytest.mark.asyncio
async def test_worker_coordinator_builds_parent() -> None:
    backend = SQLiteStorageBackend()
    try:
        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-5-nano",
        )
        store = backend.for_document("doc")

        _add_batch(
            store,
            _leaf_payload("L", 0, 10, following_neighbor_id="R"),
            _leaf_payload("R", 10, 20, preceding_neighbor_id="L"),
        )

        llm = StubLLMService()
        vector_index = StubVectorIndex()

        coordinator = WorkerCoordinator(
            store=backend,
            index_config=IndexConfig.load(),
            operational_config=OperationalConfig(
                openai_api_key=SecretStr("test"),
                vector_backend="python",
                database_url="sqlite:///:memory:",
            ),
            llm_service=llm,
            vector_index_factory=lambda _doc: vector_index,
            worker_count=1,
        )

        await coordinator.start()
        await coordinator.enqueue_document("doc")
        await coordinator.wait_until_idle("doc")

        parents = [node for node in store.nodes.get_all() if node.height == 1]
        assert len(parents) == 1
        parent = parents[0]
        assert parent.left_child_id == "L"
        assert parent.right_child_id == "R"
        assert parent.text
        left_node = store.nodes.get("L")
        right_node = store.nodes.get("R")
        assert left_node is not None
        assert right_node is not None
        assert left_node.parent_id == parent.id
        assert right_node.parent_id == parent.id

        assert any(item[0] == parent.id for item in vector_index.upserts)

        await coordinator.shutdown()
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_status_tracks_inflight_work() -> None:
    backend = SQLiteStorageBackend()
    try:
        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-5-nano",
        )
        store = backend.for_document("doc")

        _add_batch(
            store,
            _leaf_payload("L", 0, 10, following_neighbor_id="R"),
            _leaf_payload("R", 10, 20, preceding_neighbor_id="L"),
        )

        llm = BlockingLLMService()
        vector_index = StubVectorIndex()

        coordinator = WorkerCoordinator(
            store=backend,
            index_config=IndexConfig.load(),
            operational_config=OperationalConfig(
                openai_api_key=SecretStr("test"),
                vector_backend="python",
                database_url="sqlite:///:memory:",
            ),
            llm_service=llm,
            vector_index_factory=lambda _doc: vector_index,
            worker_count=1,
        )

        await coordinator.start()
        await coordinator.enqueue_document("doc")
        await asyncio.wait_for(llm.started.wait(), timeout=2)

        status_during = await coordinator.status()
        assert status_during.in_flight == 1
        assert status_during.inflight_by_document.get("doc") == 1

        llm.release()
        await coordinator.wait_until_idle("doc")

        status_final = await coordinator.status()
        assert status_final.queue_depth == 0
        assert status_final.in_flight == 0

        await coordinator.shutdown()
    finally:
        backend.close()
