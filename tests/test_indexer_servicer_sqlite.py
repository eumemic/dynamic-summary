from types import SimpleNamespace
from typing import NoReturn, cast
from unittest.mock import patch

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.server.append_executor import AppendOutcome
from ragzoom.server.run_manager import TelemetryRunManager
from ragzoom.server.servicers import IndexerServicer
from ragzoom.server.state import ServerState


class StubAppendExecutor:
    def __init__(self, outcome: AppendOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[DocumentStore, VectorIndex, str, str, object | None]] = (
            []
        )

    async def append(
        self,
        *,
        store: DocumentStore,
        vector_index: VectorIndex,
        document_id: str,
        new_text: str,
        reporter: object | None = None,
    ) -> AppendOutcome:
        self.calls.append((store, vector_index, document_id, new_text, reporter))
        return self.outcome


class StubWorkerCoordinator:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.deleted: list[list[str] | None] = []
        self.new_roots: list[list[str] | None] = []
        self.cancelled: list[str] = []
        self.attached_runs: list[object] = []
        self.detached: list[str] = []

    async def enqueue_document(
        self,
        document_id: str,
        *,
        deleted_node_ids: list[str] | None = None,
        new_root_ids: list[str] | None = None,
    ) -> None:
        self.enqueued.append(document_id)
        self.deleted.append(deleted_node_ids)
        self.new_roots.append(new_root_ids)

    async def attach_run(self, context: object) -> None:
        self.attached_runs.append(context)

    async def detach_run(self, document_id: str) -> None:
        self.detached.append(document_id)

    async def cancel_document(self, document_id: str) -> None:
        self.cancelled.append(document_id)

    async def wait_until_idle(self, document_id: str | None = None) -> None:
        return None


class StubContext:
    async def abort(self, code: object, details: str) -> NoReturn:
        raise AssertionError(f"Unexpected abort ({code}): {details}")


@pytest.mark.asyncio
async def test_append_text_uses_append_executor() -> None:
    backend = SQLiteStorageBackend()
    try:
        index_config = IndexConfig.load()
        query_config = QueryConfig()
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        )

        outcome = AppendOutcome(
            document_id="doc",
            appended_span_start=0,
            appended_span_end=5,
            new_leaf_ids=["leaf-1", "leaf-2"],
            deleted_node_ids=["old"],
            total_leaves=10,
        )

        append_executor = StubAppendExecutor(outcome)
        worker_coordinator = StubWorkerCoordinator()

        state = cast(
            ServerState,
            SimpleNamespace(
                index_config=index_config,
                query_config=query_config,
                operational_config=operational_config,
                store=backend,
                indexing_service=None,
                query_service=None,
                llm_service=None,
                telemetry_run_manager=TelemetryRunManager(index_config),
                append_executor=append_executor,
                worker_coordinator=worker_coordinator,
            ),
        )

        servicer = IndexerServicer(state)

        request = pb2.AppendTextRequest(
            document_id="doc",
            content=b"hello world",
            collect_telemetry=False,
        )

        response = await servicer.AppendText(request, StubContext())

        assert append_executor.calls, "Append executor was not invoked"
        call = append_executor.calls[0]
        assert call[2] == "doc"
        assert call[3] == "hello world"

        assert backend.get_document_by_id("doc") is not None
        assert worker_coordinator.enqueued == ["doc"]
        assert worker_coordinator.deleted == [outcome.deleted_node_ids]
        assert worker_coordinator.new_roots == [outcome.new_leaf_ids]

        stats = response.stats
        assert stats.document_id == "doc"
        assert stats.chunks_created == outcome.total_leaves
        assert stats.total_leaves == outcome.total_leaves
        expected_mutations = len(outcome.new_leaf_ids) + len(outcome.deleted_node_ids)
        assert stats.mutated_nodes == expected_mutations
        assert stats.new_leaves == len(outcome.new_leaf_ids)
        assert stats.resummarized_nodes == 0
        assert getattr(response, "telemetry_run_id", "") == ""
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_append_text_with_replace_existing_sets_flag() -> None:
    backend = SQLiteStorageBackend()
    try:
        index_config = IndexConfig.load()
        query_config = QueryConfig()
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        )

        document_id = "doc-replace"

        backend.add_document(
            document_id,
            file_path=None,
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )

        outcome = AppendOutcome(
            document_id=document_id,
            appended_span_start=0,
            appended_span_end=5,
            new_leaf_ids=["leaf-1"],
            deleted_node_ids=[],
            total_leaves=1,
        )

        append_executor = StubAppendExecutor(outcome)
        worker_coordinator = StubWorkerCoordinator()

        state = cast(
            ServerState,
            SimpleNamespace(
                index_config=index_config,
                query_config=query_config,
                operational_config=operational_config,
                store=backend,
                indexing_service=None,
                query_service=None,
                llm_service=None,
                telemetry_run_manager=TelemetryRunManager(index_config),
                append_executor=append_executor,
                worker_coordinator=worker_coordinator,
            ),
        )

        servicer = IndexerServicer(state)

        request = pb2.AppendTextRequest(
            document_id=document_id,
            content=b"hello world",
            collect_telemetry=False,
        )
        setattr(request, "replace_existing", True)

        delete_calls: list[bool] = []

        class VectorIndexStub:
            def delete(
                self,
                filter: dict[str, object] | None = None,
                ids: list[str] | None = None,
            ) -> int:
                delete_calls.append(True)
                return 0

            def upsert(
                self,
                items: list[tuple[str, list[float], dict[str, object]]],
            ) -> None:  # pragma: no cover - not used in this test
                return None

            def search_similar(
                self,
                query_embedding: object,
                k: int,
                where: dict[str, object] | None = None,
            ) -> list[object]:  # pragma: no cover
                return []

            def get_vectors(self, ids: list[str]) -> list[object]:  # pragma: no cover
                return []

        def vector_factory(*args: object, **kwargs: object) -> VectorIndexStub:
            return VectorIndexStub()

        with (
            patch.object(
                backend, "clear_document", wraps=backend.clear_document
            ) as clear_mock,
            patch(
                "ragzoom.server.servicers.create_vector_index",
                side_effect=vector_factory,
            ),
        ):
            await servicer.AppendText(request, StubContext())

        assert clear_mock.called
        assert delete_calls, "Vector index delete was not invoked"
        assert append_executor.calls
        assert worker_coordinator.cancelled == [document_id]
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_index_document_clears_then_appends() -> None:
    backend = SQLiteStorageBackend()
    try:
        index_config = IndexConfig.load()
        query_config = QueryConfig()
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        )

        document_id = "doc-index"
        backend.add_document(
            document_id,
            file_path="/tmp/original.txt",
            embedding_model=index_config.embedding_model,
            summary_model=index_config.summary_model,
        )

        outcome = AppendOutcome(
            document_id=document_id,
            appended_span_start=0,
            appended_span_end=11,
            new_leaf_ids=["leaf-1"],
            deleted_node_ids=[],
            total_leaves=1,
        )

        append_executor = StubAppendExecutor(outcome)
        worker_coordinator = StubWorkerCoordinator()

        state = cast(
            ServerState,
            SimpleNamespace(
                index_config=index_config,
                query_config=query_config,
                operational_config=operational_config,
                store=backend,
                indexing_service=None,
                query_service=None,
                llm_service=None,
                telemetry_run_manager=TelemetryRunManager(index_config),
                append_executor=append_executor,
                worker_coordinator=worker_coordinator,
            ),
        )

        servicer = IndexerServicer(state)

        request = pb2.IndexDocumentRequest(
            document_id=document_id,
            content=b"hello world",
            collect_telemetry=False,
        )

        delete_calls: list[bool] = []

        class VectorIndexStub:
            def delete(
                self,
                filter: dict[str, object] | None = None,
                ids: list[str] | None = None,
            ) -> int:
                delete_calls.append(True)
                return 0

            def upsert(
                self,
                items: list[tuple[str, list[float], dict[str, object]]],
            ) -> None:  # pragma: no cover - not used
                return None

            def get_vectors(
                self,
                ids: list[str],
            ) -> list[object]:  # pragma: no cover
                return []

        def vector_factory(*args: object, **kwargs: object) -> VectorIndexStub:
            return VectorIndexStub()

        with (
            patch.object(
                backend, "clear_document", wraps=backend.clear_document
            ) as clear_mock,
            patch(
                "ragzoom.server.servicers.create_vector_index",
                side_effect=vector_factory,
            ),
        ):
            response = await servicer.IndexDocument(request, StubContext())

        assert clear_mock.called
        assert delete_calls, "Vector index delete was not invoked"
        assert append_executor.calls
        assert worker_coordinator.cancelled == [document_id]
        assert worker_coordinator.enqueued == [document_id]

        stats = response.stats
        assert stats.document_id == document_id
        assert stats.chunks_created == outcome.total_leaves
        assert stats.new_leaves == len(outcome.new_leaf_ids)
        assert stats.resummarized_nodes == 0
    finally:
        backend.close()
