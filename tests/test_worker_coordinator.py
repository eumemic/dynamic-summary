"""Backend-agnostic worker coordinator tests using real storage backend fixtures."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import numpy as np
import pytest

# from numpy.typing import NDArray
from numpy.typing import NDArray

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.server.worker_coordinator import (
    ReadyParentCandidate,
    WorkerCoordinator,
    compute_ready_parent_candidates,
)
from ragzoom.vector_api import Vector

DocStoreFixture = tuple[str, DocumentStore]
NodePayloadValue = str | int | float | bool | list[float] | NDArray[np.float64] | None
NodePayload = dict[str, NodePayloadValue]


@pytest.fixture()
def doc_store(
    storage_backend: StorageBackend,
) -> Generator[DocStoreFixture, None, None]:
    document_id = "worker-coordinator-doc"
    storage_backend.clear_document(document_id)
    store: DocumentStore = storage_backend.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )
    try:
        yield document_id, store
    finally:
        storage_backend.clear_document(document_id)


def _leaf_payload(
    node_id: str,
    span_start: int,
    span_end: int,
    *,
    level_index: int,
    document_id: str,
    preceding: str | None = None,
    following: str | None = None,
) -> NodePayload:
    return {
        "node_id": node_id,
        "text": node_id,
        "span_start": span_start,
        "span_end": span_end,
        "parent_id": None,
        "left_child_id": None,
        "right_child_id": None,
        "document_id": document_id,
        "token_count": span_end - span_start,
        "height": 0,
        "preceding_neighbor_id": preceding,
        "following_neighbor_id": following,
        "level_index": level_index,
    }


def test_compute_ready_parent_candidates_pairs_nodes(
    doc_store: DocStoreFixture,
) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "L", 0, 10, level_index=0, document_id=document_id, following="R"
            ),
            _leaf_payload(
                "R", 10, 20, level_index=1, document_id=document_id, preceding="L"
            ),
        ]
    )

    candidates = compute_ready_parent_candidates(store)
    assert candidates == [
        ReadyParentCandidate(document_id=document_id, left_child_id="L")
    ]


class StubVectorIndex(VectorIndex):
    def __init__(self) -> None:
        self.upserts: list[
            tuple[str, NDArray[np.float64] | list[float], dict[str, object]]
        ] = []
        self.deletions: list[str] = []

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None:
        self.upserts.extend(items)

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int:
        if ids:
            self.deletions.extend(ids)
        return len(ids or [])

    def get_vectors(self, ids: list[str]) -> list[Vector]:  # pragma: no cover - unused
        return []

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[Vector]:  # pragma: no cover - unused
        return []


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
        text = f"summary({left_text}|{right_text})"
        return text, 0, len(text.split())

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [np.ones(4, dtype=np.float64).tolist() for _ in texts]


@pytest.fixture()
def index_config() -> IndexConfig:
    return IndexConfig.load()


def _fetch_parent(store: DocumentStore, height: int) -> list[TreeNode]:
    nodes = store.nodes.get_parentless_nodes()
    return [node for node in nodes if int(node.height) == height]


@pytest.mark.asyncio
async def test_worker_coordinator_builds_parent(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "L", 0, 10, level_index=0, document_id=document_id, following="R"
            ),
            _leaf_payload(
                "R", 10, 20, level_index=1, document_id=document_id, preceding="L"
            ),
        ]
    )

    vector_index = StubVectorIndex()

    def _vector_factory(_: str) -> VectorIndex:
        return vector_index

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=StubLLMService(),
        vector_index_factory=_vector_factory,
        worker_count=2,
    )

    await coordinator.start()
    try:
        await coordinator.enqueue_document(document_id)
        await coordinator.wait_until_idle(document_id)
    finally:
        await coordinator.shutdown()

    parents = _fetch_parent(store, 1)
    assert len(parents) == 1
    parent = parents[0]
    assert parent.left_child_id == "L"
    assert parent.right_child_id == "R"
    left_node = store.nodes.get("L")
    right_node = store.nodes.get("R")
    assert left_node is not None
    assert right_node is not None
    assert left_node.parent_id == parent.id
    assert right_node.parent_id == parent.id


@pytest.mark.asyncio
async def test_worker_status_tracks_queue_and_inflight(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "L", 0, 10, level_index=0, document_id=document_id, following="R"
            ),
            _leaf_payload(
                "R", 10, 20, level_index=1, document_id=document_id, preceding="L"
            ),
        ]
    )

    class BlockingLLM(StubLLMService):
        def __init__(self) -> None:
            self.started = asyncio.Event()

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
            await asyncio.sleep(0.01)
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

    llm = BlockingLLM()
    vector_index = StubVectorIndex()

    def _vector_factory(_: str) -> VectorIndex:
        return vector_index

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=llm,
        vector_index_factory=_vector_factory,
        worker_count=1,
    )

    await coordinator.start()
    try:
        await coordinator.enqueue_document(document_id)
        await asyncio.wait_for(llm.started.wait(), timeout=1)
        status = await coordinator.status()
        assert status.in_flight == 1
        assert status.inflight_by_document.get(document_id) == 1
    finally:
        await coordinator.shutdown()
