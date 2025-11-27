"""Backend-agnostic worker coordinator tests using real storage backend fixtures."""

from __future__ import annotations

import asyncio
import types
from collections.abc import Callable, Generator, Sequence
from pathlib import Path
from typing import cast

import numpy as np
import pytest

# from numpy.typing import NDArray
from numpy.typing import NDArray

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_filter import VectorFilter
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.server.append_executor import AppendExecutor, EmbeddingProvider
from ragzoom.server.worker_coordinator import (
    DocumentState,
    ReadyParentCandidate,
    WorkerCoordinator,
    compute_ready_parent_candidates,
)
from ragzoom.splitter import TextSplitter
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


def test_parentless_nodes_sorted_by_height_and_level(
    doc_store: DocStoreFixture,
) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "leaf-right",
                span_start=10,
                span_end=20,
                level_index=1,
                document_id=document_id,
                preceding="leaf-left",
            ),
            _leaf_payload(
                "leaf-left",
                span_start=0,
                span_end=10,
                level_index=0,
                document_id=document_id,
                following="leaf-right",
            ),
            {
                "node_id": "intermediate",
                "text": "intermediate",
                "span_start": 0,
                "span_end": 20,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
                "document_id": document_id,
                "token_count": 20,
                "height": 1,
                "preceding_neighbor_id": None,
                "following_neighbor_id": None,
                "level_index": 0,
            },
        ]
    )

    ordered_ids = [node.id for node in store.nodes.get_parentless_nodes()]
    assert ordered_ids == ["leaf-left", "leaf-right", "intermediate"]


def test_parentless_nodes_respects_document_scope(
    doc_store: DocStoreFixture, storage_backend: StorageBackend
) -> None:
    document_id, store = doc_store
    other_store = storage_backend.add_document(
        document_id="other-doc",
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )
    try:
        other_store.nodes.add_batch(
            [
                _leaf_payload(
                    "foreign",
                    span_start=0,
                    span_end=5,
                    level_index=0,
                    document_id="other-doc",
                )
            ]
        )

        store.nodes.add_batch(
            [
                _leaf_payload(
                    "doc-node",
                    span_start=0,
                    span_end=5,
                    level_index=0,
                    document_id=document_id,
                )
            ]
        )

        ids = {node.id for node in store.nodes.get_parentless_nodes()}
        assert ids == {"doc-node"}
    finally:
        storage_backend.clear_document("other-doc")


def test_ready_left_children_returns_ids(doc_store: DocStoreFixture) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "left",
                span_start=0,
                span_end=10,
                level_index=0,
                document_id=document_id,
                following="right",
            ),
            _leaf_payload(
                "right",
                span_start=10,
                span_end=20,
                level_index=1,
                document_id=document_id,
                preceding="left",
            ),
        ]
    )

    assert store.nodes.get_ready_left_children() == ["left"]


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
    node = store.nodes.get("L")
    assert node is not None
    assert candidates == [
        ReadyParentCandidate(
            document_id=document_id,
            left_child_id="L",
            height=int(getattr(node, "height", 0)),
            level_index=int(getattr(node, "level_index", 0)),
            span_start=int(getattr(node, "span_start", 0)),
        )
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
        filters: Sequence[VectorFilter] | None = None,
    ) -> list[Vector]:  # pragma: no cover - unused
        return []


class FixedWidthSplitter:
    """Deterministic splitter that slices text into equal-sized chunks."""

    def __init__(self, width: int) -> None:
        self._width = width

    def split_text(self, text: str) -> list[str]:
        return [
            text[i : i + self._width]
            for i in range(0, len(text), self._width)
            if text[i : i + self._width]
        ]


class SimpleEmbedder(EmbeddingProvider):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(index + 1)] * 4 for index, _ in enumerate(texts)]


class StubLLMService:
    def __init__(self) -> None:
        self._on_summarize: Callable[[str, str, str | None], None] | None = None

    def set_hook(
        self,
        hook: Callable[[str, str, str | None], None],
    ) -> None:
        self._on_summarize = hook

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
        hook = self._on_summarize
        if hook is not None:
            hook(left_text, right_text, parent_id)
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


@pytest.mark.asyncio
async def test_worker_skips_candidate_when_dependencies_change(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store

    left_leaf_id = "leaf-left"
    right_leaf_id = "leaf-right"

    store.nodes.add_batch(
        [
            _leaf_payload(
                left_leaf_id,
                span_start=0,
                span_end=10,
                level_index=0,
                document_id=document_id,
                following=right_leaf_id,
            ),
            _leaf_payload(
                right_leaf_id,
                span_start=10,
                span_end=20,
                level_index=1,
                document_id=document_id,
                preceding=left_leaf_id,
            ),
        ]
    )

    vector_index = StubVectorIndex()

    def _vector_factory(_: str) -> VectorIndex:
        return vector_index

    llm_service = StubLLMService()

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=llm_service,
        vector_index_factory=_vector_factory,
        worker_count=1,
    )

    await coordinator.start()

    mutated = False

    def _on_summary(_: str, __: str, ___: str | None) -> None:
        nonlocal mutated
        if mutated:
            return
        mutated = True
        store.nodes.delete_nodes([right_leaf_id], session=None)
        store.nodes.add_batch(
            [
                _leaf_payload(
                    "replacement-right",
                    span_start=10,
                    span_end=20,
                    level_index=1,
                    document_id=document_id,
                    preceding=left_leaf_id,
                )
            ]
        )
        store.nodes.update_neighbors_batch(
            [
                (left_leaf_id, None, "replacement-right"),
                ("replacement-right", left_leaf_id, None),
            ],
            session=None,
        )

    llm_service.set_hook(_on_summary)

    await coordinator.enqueue_document(document_id, new_root_ids=[left_leaf_id])
    await asyncio.wait_for(coordinator.wait_until_idle(document_id), timeout=5)

    leaf_after = store.nodes.get(left_leaf_id)
    assert leaf_after is not None
    parent_id = leaf_after.parent_id
    assert parent_id is not None, "Coordinator left leaf without rebuilding parent"

    parent_node = store.nodes.get(parent_id)
    assert parent_node is not None
    assert parent_node.left_child_id == left_leaf_id
    assert parent_node.right_child_id in {right_leaf_id, "replacement-right"}

    await coordinator.shutdown()


@pytest.mark.asyncio
async def test_append_after_parent_reference_cleared_removes_ancestor(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store

    splitter = cast(TextSplitter, FixedWidthSplitter(width=4))
    embedder = SimpleEmbedder()
    executor = AppendExecutor(index_config, embedder, splitter=splitter)
    vector_index = StubVectorIndex()

    def _vector_factory(_: str) -> VectorIndex:
        return StubVectorIndex()

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
        # Seed the document with two leaves so the coordinator builds an initial parent.
        initial_outcome = await executor.append(
            store=store,
            vector_index=vector_index,
            document_id=document_id,
            new_text="AAAABBBB",
        )
        await coordinator.enqueue_document(
            document_id,
            deleted_node_ids=initial_outcome.deleted_node_ids,
            new_root_ids=initial_outcome.new_leaf_ids,
        )
        await asyncio.wait_for(coordinator.wait_until_idle(document_id), timeout=5)

        right_leaf_id = initial_outcome.new_leaf_ids[-1]
        right_leaf = store.nodes.get(right_leaf_id)
        assert right_leaf is not None
        parent_id = getattr(right_leaf, "parent_id", None)
        assert parent_id is not None

        # Simulate the worker clearing the parent reference before the append executor runs.
        store.nodes.update_parent_references_batch([(right_leaf_id, None)])

        append_outcome = await executor.append(
            store=store,
            vector_index=vector_index,
            document_id=document_id,
            new_text="CCCC",
        )
        await coordinator.enqueue_document(
            document_id,
            deleted_node_ids=append_outcome.deleted_node_ids,
            new_root_ids=append_outcome.new_leaf_ids,
        )

        await asyncio.wait_for(coordinator.wait_until_idle(document_id), timeout=5)

        parentless = store.nodes.get_parentless_nodes()
        leaf_parentless = [
            node.id for node in parentless if int(getattr(node, "height", 0)) == 0
        ]

        assert (
            not leaf_parentless
        ), f"Leaves missing parents after append: {leaf_parentless}"
        # A well-formed tree should have exactly one parentless node (the root).
        assert len(parentless) == 1
    finally:
        await coordinator.shutdown()


@pytest.mark.asyncio
@pytest.mark.slow_threshold(3.0)
async def test_worker_coordinator_converges_to_single_root(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store
    leaf_payloads: list[NodePayload] = []
    for idx in range(8):
        span_start = idx * 100
        span_end = (idx + 1) * 100
        preceding = None if idx == 0 else f"leaf-{idx - 1}"
        following = None if idx == 7 else f"leaf-{idx + 1}"
        leaf_payloads.append(
            _leaf_payload(
                f"leaf-{idx}",
                span_start,
                span_end,
                level_index=idx,
                document_id=document_id,
                preceding=preceding,
                following=following,
            )
        )
    store.nodes.add_batch(leaf_payloads)

    class CoordinatedLLM(StubLLMService):
        def __init__(self, expected: int) -> None:
            super().__init__()
            self._expected = expected
            self._seen = 0
            self._lock = asyncio.Lock()
            self._gate = asyncio.Event()

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
            async with self._lock:
                self._seen += 1
                if self._seen == self._expected:
                    self._gate.set()
            await self._gate.wait()
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

    llm = CoordinatedLLM(expected=4)
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
        worker_count=4,
    )

    await coordinator.start()
    try:
        await coordinator.enqueue_document(document_id)
        await coordinator.wait_until_idle(document_id)
    finally:
        await coordinator.shutdown()

    parentless = store.nodes.get_parentless_nodes()
    assert len(parentless) == 1
    assert int(parentless[0].height) >= 2


@pytest.mark.asyncio
@pytest.mark.slow_threshold(3.0)
async def test_worker_coordinator_converges_with_odd_parent_count(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store
    leaf_payloads: list[NodePayload] = []
    for idx in range(10):
        span_start = idx * 100
        span_end = (idx + 1) * 100
        preceding = None if idx == 0 else f"leaf-{idx - 1}"
        following = None if idx == 9 else f"leaf-{idx + 1}"
        leaf_payloads.append(
            _leaf_payload(
                f"leaf-{idx}",
                span_start,
                span_end,
                level_index=idx,
                document_id=document_id,
                preceding=preceding,
                following=following,
            )
        )
    store.nodes.add_batch(leaf_payloads)

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
        worker_count=4,
    )

    await coordinator.start()
    try:
        await coordinator.enqueue_document(document_id)
        await coordinator.wait_until_idle(document_id)
    finally:
        await coordinator.shutdown()

    parentless = store.nodes.get_parentless_nodes()
    assert len(parentless) == 1
    assert parentless[0].span_end == 1000


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


@pytest.mark.asyncio
@pytest.mark.slow_threshold(20)
async def test_worker_coordinator_preserves_preceding_parent_links(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store

    leaves = [
        _leaf_payload(
            "leaf-0",
            span_start=0,
            span_end=100,
            level_index=0,
            document_id=document_id,
            preceding=None,
            following="leaf-1",
        ),
        _leaf_payload(
            "leaf-1",
            span_start=100,
            span_end=200,
            level_index=1,
            document_id=document_id,
            preceding="leaf-0",
            following="leaf-2",
        ),
        _leaf_payload(
            "leaf-2",
            span_start=200,
            span_end=300,
            level_index=2,
            document_id=document_id,
            preceding="leaf-1",
            following="leaf-3",
        ),
        _leaf_payload(
            "leaf-3",
            span_start=300,
            span_end=400,
            level_index=3,
            document_id=document_id,
            preceding="leaf-2",
            following=None,
        ),
    ]
    store.nodes.add_batch(leaves)

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

    left_ready = asyncio.Event()
    left_resume = asyncio.Event()
    orig_process = WorkerCoordinator._process_candidate

    async def _instrumented_process(
        self: WorkerCoordinator,
        candidate: ReadyParentCandidate,
        state: DocumentState,
    ) -> tuple[list[str], bool]:
        if candidate.document_id != document_id or candidate.left_child_id != "leaf-0":
            return await orig_process(self, candidate, state)

        if not left_ready.is_set():
            left_ready.set()
            await left_resume.wait()

        return await orig_process(self, candidate, state)

    WorkerCoordinator._process_candidate = types.MethodType(  # type: ignore[method-assign]
        _instrumented_process, coordinator
    )

    doc_state = coordinator._get_or_create_document_state(document_id)

    left_node_meta = store.nodes.get("leaf-0")
    right_node_meta = store.nodes.get("leaf-2")
    assert left_node_meta is not None
    assert right_node_meta is not None

    left_candidate = ReadyParentCandidate(
        document_id=document_id,
        left_child_id="leaf-0",
        height=int(getattr(left_node_meta, "height", 0)),
        level_index=int(getattr(left_node_meta, "level_index", 0)),
        span_start=int(getattr(left_node_meta, "span_start", 0)),
    )
    right_candidate = ReadyParentCandidate(
        document_id=document_id,
        left_child_id="leaf-2",
        height=int(getattr(right_node_meta, "height", 0)),
        level_index=int(getattr(right_node_meta, "level_index", 0)),
        span_start=int(getattr(right_node_meta, "span_start", 0)),
    )

    left_task = asyncio.create_task(
        coordinator._process_candidate(left_candidate, doc_state)
    )
    await left_ready.wait()
    await coordinator._process_candidate(right_candidate, doc_state)
    left_resume.set()
    await left_task

    # Restore to avoid side-effects for later tests
    WorkerCoordinator._process_candidate = orig_process  # type: ignore[method-assign]

    left_node = store.nodes.get("leaf-0")
    right_node = store.nodes.get("leaf-2")

    assert left_node is not None
    assert right_node is not None

    left_parent_id = left_node.parent_id
    right_parent_id = right_node.parent_id

    assert left_parent_id is not None
    assert right_parent_id is not None

    left_parent = store.nodes.get(left_parent_id)
    right_parent = store.nodes.get(right_parent_id)

    assert left_parent is not None
    assert right_parent is not None

    # Parent nodes must be linked bidirectionally once both exist
    assert left_parent.following_neighbor_id == right_parent.id
    assert right_parent.preceding_neighbor_id == left_parent.id


@pytest.mark.asyncio
async def test_worker_coordinator_prioritises_lower_height_candidates(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store

    initial_leaves = [
        _leaf_payload(
            "leaf-0",
            span_start=0,
            span_end=100,
            level_index=0,
            document_id=document_id,
            preceding=None,
            following="leaf-1",
        ),
        _leaf_payload(
            "leaf-1",
            span_start=100,
            span_end=200,
            level_index=1,
            document_id=document_id,
            preceding="leaf-0",
            following="leaf-2",
        ),
        _leaf_payload(
            "leaf-2",
            span_start=200,
            span_end=300,
            level_index=2,
            document_id=document_id,
            preceding="leaf-1",
            following=None,
        ),
    ]
    store.nodes.add_batch(initial_leaves)

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=StubLLMService(),
        worker_count=1,
    )

    await coordinator._scan_document(document_id)
    doc_state = coordinator._get_or_create_document_state(document_id)

    oldest_priority, oldest_candidate = await coordinator._queue.get()
    assert oldest_candidate.left_child_id == "leaf-0"
    doc_state.queued.discard(oldest_candidate.left_child_id)
    coordinator._queue.task_done()
    store.update_parent_reference("leaf-0", "parent-x")
    store.nodes.add_batch(
        [
            {
                "node_id": "parent-x",
                "text": "parent-x",
                "span_start": 0,
                "span_end": 200,
                "parent_id": None,
                "left_child_id": "leaf-0",
                "right_child_id": "leaf-1",
                "document_id": document_id,
                "token_count": 200,
                "height": 1,
                "preceding_neighbor_id": None,
                "following_neighbor_id": None,
                "level_index": 0,
            }
        ]
    )

    new_leaf = _leaf_payload(
        "leaf-new",
        span_start=300,
        span_end=400,
        level_index=0,
        document_id=document_id,
        preceding="leaf-2",
        following=None,
    )
    store.nodes.add_batch([new_leaf])
    store.nodes.update_neighbors_batch([("leaf-2", "leaf-1", "leaf-new")])

    await coordinator._scan_document(document_id)

    next_priority, next_candidate = await coordinator._queue.get()
    assert next_candidate.left_child_id == "leaf-new"
    coordinator._queue.task_done()


@pytest.mark.asyncio
async def test_process_new_root_unlocks_adjacent_spans(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store

    store.nodes.add_batch(
        [
            _leaf_payload(
                "leaf-0",
                span_start=0,
                span_end=100,
                level_index=0,
                document_id=document_id,
                preceding=None,
                following="leaf-1",
            ),
            _leaf_payload(
                "leaf-1",
                span_start=100,
                span_end=200,
                level_index=1,
                document_id=document_id,
                preceding="leaf-0",
                following="leaf-2",
            ),
            _leaf_payload(
                "leaf-2",
                span_start=200,
                span_end=300,
                level_index=2,
                document_id=document_id,
                preceding="leaf-1",
                following=None,
            ),
        ]
    )

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=StubLLMService(),
        worker_count=1,
    )

    state = coordinator._get_or_create_document_state(document_id)
    doc_span_end = coordinator._document_span_end(document_id, store)
    await coordinator._process_new_root(
        document_id,
        "leaf-1",
        state,
        store,
        doc_span_end,
    )

    jobs: list[str] = []
    while not coordinator._queue.empty():
        _, job = coordinator._queue.get_nowait()
        jobs.append(job.left_child_id)
        coordinator._queue.task_done()

    assert jobs == ["leaf-0", "leaf-2"]


@pytest.mark.asyncio
async def test_enqueue_document_drops_deleted_jobs(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "leaf-0",
                span_start=0,
                span_end=100,
                level_index=0,
                document_id=document_id,
                preceding=None,
                following="leaf-1",
            ),
            _leaf_payload(
                "leaf-1",
                span_start=100,
                span_end=200,
                level_index=1,
                document_id=document_id,
                preceding="leaf-0",
                following=None,
            ),
        ]
    )

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=StubLLMService(),
        worker_count=1,
    )

    await coordinator.enqueue_document(document_id)
    assert coordinator.queue_depth(document_id) == 1

    await coordinator.enqueue_document(document_id, deleted_node_ids=["leaf-0"])
    assert coordinator.queue_depth(document_id) == 0


@pytest.mark.asyncio
async def test_dependency_check_detects_missing_right_child(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "leaf-0",
                span_start=0,
                span_end=100,
                level_index=0,
                document_id=document_id,
                preceding=None,
                following="leaf-1",
            ),
            _leaf_payload(
                "leaf-1",
                span_start=100,
                span_end=200,
                level_index=1,
                document_id=document_id,
                preceding="leaf-0",
                following=None,
            ),
        ]
    )

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=StubLLMService(),
        worker_count=1,
    )

    state = coordinator._get_or_create_document_state(document_id)
    ready, snapshot = coordinator._check_dependencies_still_valid(
        document_id, "leaf-0", state
    )
    assert ready
    assert snapshot.right is not None

    with store.transaction() as session:
        store.nodes.delete_nodes(["leaf-1"], session=session)

    ready_after, snapshot_after = coordinator._check_dependencies_still_valid(
        document_id, "leaf-0", state
    )
    assert not ready_after
    assert snapshot_after.right is None


@pytest.mark.asyncio
async def test_possibly_enqueue_requires_preceding_neighbor(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "leaf-0",
                span_start=0,
                span_end=10,
                level_index=0,
                document_id=document_id,
                following="leaf-1",
            ),
            _leaf_payload(
                "leaf-1",
                span_start=10,
                span_end=20,
                level_index=2,
                document_id=document_id,
                preceding=None,
                following=None,
            ),
        ]
    )

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=StubLLMService(),
        vector_index_factory=lambda _: StubVectorIndex(),
        worker_count=1,
    )

    state = coordinator._get_or_create_document_state(document_id)
    left_node = store.nodes.get("leaf-1")
    assert left_node is not None
    doc_span_end = coordinator._document_span_end(document_id, store)

    ready, snapshot = coordinator._resolve_dependencies(
        document_id,
        left_node,
        state,
        store,
        doc_span_end,
    )

    assert not ready
    assert snapshot.left is left_node

    await coordinator._possibly_enqueue_left_child(
        document_id,
        left_node,
        state,
        store,
        doc_span_end,
    )

    assert state.queued == set()
    assert coordinator._queue.empty()


@pytest.mark.asyncio
async def test_possibly_enqueue_skips_root_span(
    doc_store: DocStoreFixture,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id, store = doc_store
    store.nodes.add_batch(
        [
            _leaf_payload(
                "root-span",
                span_start=0,
                span_end=50,
                level_index=0,
                document_id=document_id,
                preceding=None,
                following=None,
            )
        ]
    )

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        ),
        llm_service=StubLLMService(),
        vector_index_factory=lambda _: StubVectorIndex(),
        worker_count=1,
    )

    state = coordinator._get_or_create_document_state(document_id)
    left_node = store.nodes.get("root-span")
    assert left_node is not None
    doc_span_end = coordinator._document_span_end(document_id, store)

    ready, snapshot = coordinator._resolve_dependencies(
        document_id,
        left_node,
        state,
        store,
        doc_span_end,
    )

    assert not ready
    assert snapshot.left is left_node
    assert snapshot.preceding is None
    assert snapshot.right is None

    await coordinator._possibly_enqueue_left_child(
        document_id,
        left_node,
        state,
        store,
        doc_span_end,
    )

    assert state.queued == set()
    assert coordinator._queue.empty()


class _DeterministicEmbedder:
    _provider_max_embedding_batch_size = 100

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]


@pytest.mark.asyncio
@pytest.mark.slow_threshold(10)
async def test_worker_coordinator_rolls_up_after_reappend(
    storage_backend: StorageBackend,
    index_config: IndexConfig,
) -> None:
    document_id = "worker-coordinator-reappend"
    storage_backend.clear_document(document_id)
    store = storage_backend.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )

    operational_config = OperationalConfig(
        openai_api_key=SecretStr("test"),
        vector_backend="python",
        database_url="sqlite:///:memory:",
    )

    append_executor = AppendExecutor(index_config, _DeterministicEmbedder())

    coordinator = WorkerCoordinator(
        store=storage_backend,
        index_config=index_config,
        operational_config=operational_config,
        llm_service=StubLLMService(),
        vector_index_factory=lambda _: StubVectorIndex(),
        worker_count=8,
    )

    source_text = Path("test_data/the_hobbit_chapter_1.txt").read_text(
        encoding="utf-8"
    )[:5000]

    async def _append_once() -> None:
        await append_executor.append(
            store=store,
            vector_index=StubVectorIndex(),
            document_id=document_id,
            new_text=source_text,
            reporter=None,
        )

    async def _run_workers() -> None:
        await coordinator.start()
        try:
            await coordinator.enqueue_document(document_id)
            await asyncio.wait_for(coordinator.wait_until_idle(document_id), timeout=5)
        finally:
            await coordinator.shutdown()

    await _append_once()
    await _run_workers()
    first_parentless = store.nodes.get_parentless_nodes()
    first_roots = [node for node in first_parentless if node.height > 0]
    assert len(first_roots) == 1
    assert len(first_parentless) == len(first_roots)

    await _append_once()
    await _run_workers()
    second_parentless = store.nodes.get_parentless_nodes()
    second_roots = [node for node in second_parentless if node.height > 0]
    assert len(second_roots) == 1
    assert len(second_parentless) == len(second_roots)
