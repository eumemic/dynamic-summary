from types import SimpleNamespace
from typing import NoReturn, cast

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.server.append_executor import AppendOutcome
from ragzoom.server.servicers import IndexerServicer
from ragzoom.server.state import ServerState


class StubAppendExecutor:
    def __init__(self, outcome: AppendOutcome) -> None:
        self.outcome = outcome
        self.calls: list[tuple[DocumentStore, VectorIndex, str, str]] = []

    async def append(
        self,
        *,
        store: DocumentStore,
        vector_index: VectorIndex,
        document_id: str,
        new_text: str,
    ) -> AppendOutcome:
        self.calls.append((store, vector_index, document_id, new_text))
        return self.outcome


class StubWorkerCoordinator:
    def __init__(self) -> None:
        self.enqueued: list[str] = []

    async def enqueue_document(self, document_id: str) -> None:
        self.enqueued.append(document_id)


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

        stats = response.stats
        assert stats.document_id == "doc"
        assert stats.chunks_created == outcome.total_leaves
        assert stats.total_leaves == outcome.total_leaves
        expected_mutations = len(outcome.new_leaf_ids) + len(outcome.deleted_node_ids)
        assert stats.mutated_nodes == expected_mutations
        assert stats.new_leaves == len(outcome.new_leaf_ids)
        assert stats.resummarized_nodes == 0
    finally:
        backend.close()
