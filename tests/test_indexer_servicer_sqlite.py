from collections.abc import Callable
from types import SimpleNamespace
from typing import NoReturn, cast
from unittest.mock import patch

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.indexing import IndexerRuntime
from ragzoom.rpc import dynamic_summary_pb2 as pb2
from ragzoom.server.append_executor import AppendExecutor, AppendOutcome
from ragzoom.server.indexing_engine import IndexingEngine, IndexingStatus
from ragzoom.server.run_manager import TelemetryRunManager
from ragzoom.server.servicers import IndexerServicer
from ragzoom.server.state import ServerState


class StubAppendExecutor:
    def __init__(self, outcome: AppendOutcome) -> None:
        self.outcome = outcome
        self.calls: list[
            tuple[
                DocumentStore,
                str,
                str,
                object | None,
                object | None,
                object | None,
            ]
        ] = []

    async def append(
        self,
        *,
        store: DocumentStore,
        document_id: str,
        new_text: str,
        timestamp: str | tuple[str, str] | None = None,
        reporter: object | None = None,
        run_context: object | None = None,
        telemetry_manager: object | None = None,
    ) -> AppendOutcome:
        self.calls.append(
            (
                store,
                document_id,
                new_text,
                reporter,
                run_context,
                telemetry_manager,
            )
        )
        return self.outcome


class StubIndexingEngine:
    """Stub IndexingEngine for testing servicers without real indexing."""

    def __init__(self) -> None:
        self.triggered: list[str] = []
        self.cancelled: list[str] = []

    async def trigger_work(
        self,
        document_id: str,
        *,
        telemetry_collector: object | None = None,
    ) -> None:
        self.triggered.append(document_id)

    async def cancel_document(self, document_id: str) -> None:
        self.cancelled.append(document_id)

    async def wait_until_idle(self, document_id: str | None = None) -> None:
        return None

    def update_chars_per_token_after_append(self, document_id: str) -> None:
        """Stub for chars_per_token update (no-op in tests)."""
        pass

    async def status(self) -> IndexingStatus:
        return IndexingStatus(
            in_flight=0,
            in_flight_by_document={},
            completed_by_document={},
            expected_total_by_document={},
        )

    async def shutdown(self) -> None:
        pass


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
        indexing_engine = StubIndexingEngine()
        telemetry_manager = TelemetryRunManager(index_config)

        class VectorIndexStub:
            def delete(
                self,
                filter: dict[str, object] | None = None,
                ids: list[str] | None = None,
            ) -> int:
                return 0

            def upsert(
                self,
                items: list[tuple[str, list[float], dict[str, object]]],
            ) -> None:
                return None

            def search_similar(
                self,
                query_embedding: object,
                k: int,
                where: dict[str, object] | None = None,
            ) -> list[object]:
                return []

            def get_vectors(self, ids: list[str]) -> list[object]:
                return []

        def vector_factory(model: str) -> VectorIndexStub:
            return VectorIndexStub()

        runtime = IndexerRuntime(
            store=backend,
            index_config=index_config,
            append_executor=cast(AppendExecutor, append_executor),
            indexing_engine=cast(IndexingEngine, indexing_engine),
            telemetry_manager=telemetry_manager,
            vector_index_factory=cast(Callable[[str], VectorIndex], vector_factory),
        )

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
                telemetry_run_manager=telemetry_manager,
                append_executor=append_executor,
                indexing_engine=indexing_engine,
                index_runtime=runtime,
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
        assert call[1] == "doc"
        assert call[2] == "hello world"

        assert backend.get_document_by_id("doc") is not None
        assert indexing_engine.triggered == ["doc"]

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
        indexing_engine = StubIndexingEngine()
        telemetry_manager = TelemetryRunManager(index_config)

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
                telemetry_run_manager=telemetry_manager,
                append_executor=append_executor,
                indexing_engine=indexing_engine,
            ),
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
            ) -> None:
                return None

            def search_similar(
                self,
                query_embedding: object,
                k: int,
                where: dict[str, object] | None = None,
            ) -> list[object]:
                return []

            def get_vectors(self, ids: list[str]) -> list[object]:
                return []

        def vector_factory(model: str) -> VectorIndexStub:
            return VectorIndexStub()

        runtime = IndexerRuntime(
            store=backend,
            index_config=index_config,
            append_executor=cast(AppendExecutor, append_executor),
            indexing_engine=cast(IndexingEngine, indexing_engine),
            telemetry_manager=telemetry_manager,
            vector_index_factory=cast(Callable[[str], VectorIndex], vector_factory),
        )

        setattr(state, "index_runtime", runtime)

        servicer = IndexerServicer(state)

        request = pb2.AppendTextRequest(
            document_id=document_id,
            content=b"hello world",
            collect_telemetry=False,
        )
        setattr(request, "replace_existing", True)

        with (
            patch.object(
                backend, "clear_document", wraps=backend.clear_document
            ) as clear_mock,
        ):
            await servicer.AppendText(request, StubContext())

        assert clear_mock.called
        assert delete_calls, "Vector index delete was not invoked"
        assert append_executor.calls
        assert indexing_engine.cancelled == [document_id]
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
        indexing_engine = StubIndexingEngine()
        telemetry_manager = TelemetryRunManager(index_config)

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
                telemetry_run_manager=telemetry_manager,
                append_executor=append_executor,
                indexing_engine=indexing_engine,
            ),
        )

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

        def vector_factory(model: str) -> VectorIndexStub:
            return VectorIndexStub()

        runtime = IndexerRuntime(
            store=backend,
            index_config=index_config,
            append_executor=cast(AppendExecutor, append_executor),
            indexing_engine=cast(IndexingEngine, indexing_engine),
            telemetry_manager=telemetry_manager,
            vector_index_factory=cast(Callable[[str], VectorIndex], vector_factory),
        )

        setattr(state, "index_runtime", runtime)

        servicer = IndexerServicer(state)

        with (
            patch.object(
                backend, "clear_document", wraps=backend.clear_document
            ) as clear_mock,
        ):
            response = await servicer.IndexDocument(request, StubContext())

        assert clear_mock.called
        assert delete_calls, "Vector index delete was not invoked"
        assert append_executor.calls
        assert indexing_engine.cancelled == [document_id]
        assert indexing_engine.triggered == [document_id]

        stats = response.stats
        assert stats.document_id == document_id
        assert stats.chunks_created == outcome.total_leaves
        assert stats.new_leaves == len(outcome.new_leaf_ids)
        assert stats.resummarized_nodes == 0
    finally:
        backend.close()


@pytest.mark.asyncio
async def test_servicer_stores_summary_system_prompt() -> None:
    """Verify servicer extracts summary_system_prompt from request and stores per-document.

    Spec: specs/custom-prompt-config.md § CLI Override
    Test: tests/test_indexer_servicer_sqlite.py::test_servicer_stores_summary_system_prompt
    """
    backend = SQLiteStorageBackend()
    try:
        index_config = IndexConfig.load()
        query_config = QueryConfig()
        operational_config = OperationalConfig(
            openai_api_key=SecretStr("test"),
            vector_backend="python",
            database_url="sqlite:///:memory:",
        )

        document_id = "doc-custom-prompt"
        custom_prompt = (
            "You are a legal document summarizer. Output ONLY compressed text."
        )

        outcome = AppendOutcome(
            document_id=document_id,
            appended_span_start=0,
            appended_span_end=10,
            new_leaf_ids=["leaf-1"],
            deleted_node_ids=[],
            total_leaves=1,
        )

        append_executor = StubAppendExecutor(outcome)
        indexing_engine = StubIndexingEngine()
        telemetry_manager = TelemetryRunManager(index_config)

        class VectorIndexStub:
            def delete(
                self,
                filter: dict[str, object] | None = None,
                ids: list[str] | None = None,
            ) -> int:
                return 0

            def upsert(
                self,
                items: list[tuple[str, list[float], dict[str, object]]],
            ) -> None:
                return None

            def search_similar(
                self,
                query_embedding: object,
                k: int,
                where: dict[str, object] | None = None,
            ) -> list[object]:
                return []

            def get_vectors(self, ids: list[str]) -> list[object]:
                return []

        def vector_factory(model: str) -> VectorIndexStub:
            return VectorIndexStub()

        runtime = IndexerRuntime(
            store=backend,
            index_config=index_config,
            append_executor=cast(AppendExecutor, append_executor),
            indexing_engine=cast(IndexingEngine, indexing_engine),
            telemetry_manager=telemetry_manager,
            vector_index_factory=cast(Callable[[str], VectorIndex], vector_factory),
        )

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
                telemetry_run_manager=telemetry_manager,
                append_executor=append_executor,
                indexing_engine=indexing_engine,
                index_runtime=runtime,
            ),
        )

        servicer = IndexerServicer(state)

        # Create request with custom system prompt
        request = pb2.AppendTextRequest(
            document_id=document_id,
            content=b"Legal contract text here.",
            collect_telemetry=False,
            summary_system_prompt=custom_prompt,
        )

        await servicer.AppendText(request, StubContext())

        # Verify document was created with custom prompt stored
        doc = backend.get_document_by_id(document_id)
        assert doc is not None, "Document should be created"
        assert (
            doc.summary_system_prompt == custom_prompt
        ), f"Expected custom prompt stored, got: {doc.summary_system_prompt!r}"

    finally:
        backend.close()
