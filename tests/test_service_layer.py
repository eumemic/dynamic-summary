"""Tests for the service layer implementation."""

import asyncio
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.query_log import QueryLog
from ragzoom.services.document_service import (
    DocumentInfo,
    DocumentService,
    SystemStatus,
)
from ragzoom.services.indexing_service import IndexingResult, IndexingService
from ragzoom.services.query_service import QueryResult, QueryService


class TestDocumentService:
    """Test the DocumentService."""

    def test_list_documents(self, storage_backend: StorageBackend) -> None:
        """Test listing documents returns formatted results using backend."""
        # Setup a document with nodes
        doc_id = "test-doc"
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path="/path/to/file.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        nodes = [
            {
                "node_id": f"n{i}",
                "text": f"content {i}",
                "span_start": i * 10,
                "span_end": i * 10 + 5,
                "document_id": doc_id,
                "token_count": 1,
                "height": 0,
            }
            for i in range(10)
        ]
        doc_store.nodes.add_batch(nodes)  # type: ignore[arg-type]

        # DocumentService expects a Store-like object; using backend directly works here
        service = DocumentService(storage_backend)
        documents = service.list_documents()

        assert len(documents) == 1
        assert isinstance(documents[0], DocumentInfo)
        assert documents[0].document_id == doc_id
        assert documents[0].file_path == "/path/to/file.txt"
        assert not hasattr(documents[0], "chunk_count")
        assert documents[0].node_count == 10

    def test_get_system_status(self, storage_backend: StorageBackend) -> None:
        """Test getting system status with real backend."""
        # Create two docs with nodes
        for d in ("doc-a", "doc-b"):
            ds = storage_backend.for_document(d)
            ds.set_metadata(
                file_path=None,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
            )
            nodes = [
                {
                    "node_id": f"{d}-leaf-{i}",
                    "text": f"t{i}",
                    "span_start": i * 10,
                    "span_end": i * 10 + 5,
                    "document_id": d,
                    "token_count": 1,
                    "height": 0,
                }
                for i in range(5)
            ]
            ds.nodes.add_batch(nodes)  # type: ignore[arg-type]
        # Pin a node
        storage_backend.node_repo.pin_node("doc-a-leaf-0")  # type: ignore[attr-defined]

        service = DocumentService(storage_backend)
        status = service.get_system_status()

        assert isinstance(status, SystemStatus)
        assert status.total_nodes >= 10
        assert status.leaf_nodes >= 10
        assert status.tree_depth >= 0
        assert status.pinned_nodes >= 1

    def test_clear_document(self) -> None:
        """Test clearing a document."""
        mock_store = Mock()
        mock_store.clear_document.return_value = 15

        service = DocumentService(mock_store)
        deleted_count = service.clear_document("test-doc")

        assert deleted_count == 15
        mock_store.clear_document.assert_called_once_with("test-doc")

    def test_get_nodes_in_span_orders_by_height(
        self, storage_backend: StorageBackend
    ) -> None:
        doc_id = "span-doc"
        doc_store = storage_backend.for_document(doc_id)
        doc_store.set_metadata(
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        nodes_payload = [
            {
                "node_id": "root",
                "text": "root summary",
                "span_start": 0,
                "span_end": 120,
                "document_id": doc_id,
                "token_count": 12,
                "height": 2,
                "level_index": 0,
            },
            {
                "node_id": "parent-left",
                "text": "left summary",
                "span_start": 0,
                "span_end": 60,
                "document_id": doc_id,
                "token_count": 6,
                "height": 1,
                "level_index": 0,
            },
            {
                "node_id": "parent-right",
                "text": "right summary",
                "span_start": 60,
                "span_end": 120,
                "document_id": doc_id,
                "token_count": 6,
                "height": 1,
                "level_index": 1,
            },
            {
                "node_id": "leaf-a",
                "text": "leaf a",
                "span_start": 0,
                "span_end": 30,
                "document_id": doc_id,
                "token_count": 3,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf-b",
                "text": "leaf b",
                "span_start": 30,
                "span_end": 60,
                "document_id": doc_id,
                "token_count": 3,
                "height": 0,
                "level_index": 1,
            },
        ]
        doc_store.nodes.add_batch(nodes_payload)  # type: ignore[arg-type]

        service = DocumentService(storage_backend)
        result = service.get_nodes_in_span(
            doc_id,
            span_start=0,
            span_end=120,
            limit=3,
        )

        assert result.total_matching == 5
        assert [node.node_id for node in result.nodes] == [
            "root",
            "parent-left",
            "parent-right",
        ]

    def test_get_nodes_by_ids_filters_document(
        self, storage_backend: StorageBackend
    ) -> None:
        doc_id = "batch-doc"
        other_doc_id = "other-doc"
        for target in (doc_id, other_doc_id):
            ds = storage_backend.for_document(target)
            ds.set_metadata(
                file_path=None,
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
            )
            payload = [
                {
                    "node_id": f"{target}-n{i}",
                    "text": f"text {target} {i}",
                    "span_start": i * 10,
                    "span_end": i * 10 + 5,
                    "document_id": target,
                    "token_count": 1,
                    "height": 0,
                    "level_index": i,
                }
                for i in range(3)
            ]
            ds.nodes.add_batch(payload)  # type: ignore[arg-type]

        service = DocumentService(storage_backend)
        nodes = service.get_nodes_by_ids(
            doc_id,
            [f"{doc_id}-n0", f"{other_doc_id}-n1", f"{doc_id}-n2"],
        )

        node_ids = {node.node_id for node in nodes}
        assert node_ids == {f"{doc_id}-n0", f"{doc_id}-n2"}


class _StubSession:
    def __init__(self, result: IndexingResult) -> None:
        self._result = result
        self.calls: list[dict[str, object]] = []

    async def append_text(
        self,
        text: str,
        *,
        replace_existing: bool,
        collect_telemetry: bool,
    ) -> IndexingResult:
        self.calls.append(
            {
                "text": text,
                "replace_existing": replace_existing,
                "collect_telemetry": collect_telemetry,
            }
        )
        return self._result


class _StubRuntime:
    def __init__(self, session: _StubSession) -> None:
        self._session = session
        self.requests: list[tuple[str, str | None]] = []

    def get_session(
        self, document_id: str, *, file_path: str | None = None
    ) -> _StubSession:
        self.requests.append((document_id, file_path))
        return self._session


class TestIndexingService:
    """Tests for the gRPC-backed IndexingService."""

    def _make_service(self) -> tuple[IndexingService, MagicMock]:
        store = MagicMock()
        client_mock = MagicMock()
        client_mock.__enter__.return_value = client_mock
        client_mock.__exit__.return_value = None
        service = IndexingService(
            store,
            IndexConfig.load(),
            OperationalConfig(openai_api_key=SecretStr("test-key")),
            client_factory=lambda _address: client_mock,
        )
        return service, client_mock

    def test_index_document_calls_grpc(self) -> None:
        service, client_mock = self._make_service()
        expected = IndexingResult(
            document_id="doc-1",
            chunks_created=4,
            tree_depth=2,
            mutated_nodes=4,
            resummarized_nodes=0,
            new_leaves=1,
            telemetry=None,
        )
        client_mock.index_document.return_value = expected

        result = service.index_document(
            "Test content",
            document_id="doc-1",
            collect_telemetry=True,
        )

        assert result == expected
        client_mock.index_document.assert_called_once_with(
            document_id="doc-1",
            content=b"Test content",
            collect_telemetry=True,
        )

    def test_index_document_uses_filename_when_missing_id(self) -> None:
        service, client_mock = self._make_service()
        expected = IndexingResult(
            document_id="file.txt",
            chunks_created=1,
            tree_depth=0,
        )
        client_mock.index_document.return_value = expected

        result = service.index_document(
            "content",
            document_id=None,
            file_path="/tmp/file.txt",
        )

        assert result.document_id == "file.txt"
        client_mock.index_document.assert_called_once_with(
            document_id="file.txt",
            content=b"content",
            collect_telemetry=False,
        )

    def test_index_document_requires_identifier(self) -> None:
        service, _ = self._make_service()
        with pytest.raises(ValueError):
            service.index_document("content")

    def test_append_to_document_calls_grpc(self) -> None:
        service, client_mock = self._make_service()
        expected = IndexingResult(
            document_id="doc-1",
            chunks_created=5,
            tree_depth=3,
        )
        client_mock.append_text.return_value = expected

        result = service.append_to_document("doc-1", "more", collect_telemetry=True)

        assert result == expected
        client_mock.append_text.assert_called_once_with(
            document_id="doc-1",
            content=b"more",
            collect_telemetry=True,
            replace_existing=False,
        )

    def test_index_document_uses_runtime_when_available(self) -> None:
        store = MagicMock()
        runtime_result = IndexingResult(
            document_id="doc-1",
            chunks_created=2,
            tree_depth=1,
            mutated_nodes=2,
        )
        session = _StubSession(runtime_result)
        runtime = _StubRuntime(session)

        service = IndexingService(
            store,
            IndexConfig.load(),
            OperationalConfig(openai_api_key=SecretStr("test-key")),
            index_runtime=runtime,
        )
        assert service._index_runtime is runtime
        assert service._client_factory is None

        result = service.index_document("content", document_id="doc-1")

        assert result == runtime_result
        assert runtime.requests == [("doc-1", None)]
        assert session.calls[0]["replace_existing"] is True

    def test_append_to_document_uses_runtime_when_available(self) -> None:
        store = MagicMock()
        runtime_result = IndexingResult(
            document_id="doc-1",
            chunks_created=3,
            tree_depth=2,
        )
        session = _StubSession(runtime_result)
        runtime = _StubRuntime(session)

        service = IndexingService(
            store,
            IndexConfig.load(),
            OperationalConfig(openai_api_key=SecretStr("test-key")),
            index_runtime=runtime,
        )
        assert service._client_factory is None

        with patch.object(
            service, "_append_via_grpc", wraps=service._append_via_grpc
        ) as grpc_stub:
            result = service.append_to_document("doc-1", "more text")

        assert result == runtime_result
        assert not grpc_stub.called
        assert session.calls[0]["replace_existing"] is False
        assert runtime.requests == [("doc-1", None)]

    @pytest.mark.asyncio
    async def test_index_document_async_uses_runtime(self) -> None:
        store = MagicMock()
        runtime_result = IndexingResult(
            document_id="doc-async",
            chunks_created=5,
            tree_depth=3,
        )
        session = _StubSession(runtime_result)
        runtime = _StubRuntime(session)

        service = IndexingService(
            store,
            IndexConfig.load(),
            OperationalConfig(openai_api_key=SecretStr("test-key")),
            index_runtime=runtime,
        )
        assert service._client_factory is None

        result = await service.index_document_async("body", document_id="doc-async")

        assert result == runtime_result
        assert runtime.requests == [("doc-async", None)]
        assert session.calls[0]["replace_existing"] is True

    def test_append_to_document_async(self) -> None:
        service, client_mock = self._make_service()
        expected = IndexingResult(
            document_id="doc-async",
            chunks_created=3,
            tree_depth=1,
        )
        client_mock.append_text.return_value = expected

        result = asyncio.run(
            service.append_to_document_async("doc-async", "chunk", show_progress=False)
        )

        assert result == expected
        client_mock.append_text.assert_called_once()


class TestQueryService:
    """Test the QueryService."""

    @patch("ragzoom.services.query_service.Retriever")
    @patch("ragzoom.services.query_service.Assembler")
    def test_execute_query(
        self, mock_assembler_class: object, mock_retriever_class: object
    ) -> None:
        """Test executing a query."""
        # Mock dependencies
        mock_store = Mock()

        # Mock Retriever
        mock_retriever = Mock()
        mock_retrieval_result = Mock()
        mock_retrieval_result.node_ids = ["node1", "node2"]
        mock_retrieval_result.tiling = ["node1", "node3", "node2"]
        mock_retrieval_result.scores = {"node1": 0.7, "node2": 0.5, "node3": 0.1}
        mock_retriever.retrieve.return_value = mock_retrieval_result
        cast(MagicMock, mock_retriever_class).return_value = mock_retriever

        # Mock Assembler
        mock_assembler = Mock()
        mock_assembler.assemble.return_value = "This is the summary"
        mock_assembler.get_token_count.return_value = 50
        cast(MagicMock, mock_assembler_class).return_value = mock_assembler

        # Create configs
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            query_log = QueryLog(QueryLog.default_path(Path(tmp_dir)))
            service = QueryService(
                mock_store, query_config, operational_config, query_log
            )
            result = service.execute_query("test query", "doc-123")

            assert isinstance(result, QueryResult)
            assert result.summary == "This is the summary"
            assert result.token_count == 50
            assert result.nodes_retrieved == 2
            assert result.tiling_size == 3

            mock_retriever.retrieve.assert_called_once_with(
                "test query",
                budget_tokens=1000,
                document_id="doc-123",
                num_seeds=None,
                recent_verbatim_budget=None,
            )
            mock_assembler.assemble.assert_called_once_with(mock_retrieval_result)

    @patch("ragzoom.services.query_service.Retriever")
    def test_update_config(self, mock_retriever_class: object) -> None:
        """Test updating query configuration."""
        mock_store = Mock()

        # Mock original retriever
        mock_original_retriever = Mock()
        cast(MagicMock, mock_retriever_class).return_value = mock_original_retriever

        # Create configs
        query_config = QueryConfig(budget_tokens=1000, mmr_lambda=0.7)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create service
        with tempfile.TemporaryDirectory() as tmp_dir:
            query_log = QueryLog(QueryLog.default_path(Path(tmp_dir)))
            service = QueryService(
                mock_store, query_config, operational_config, query_log
            )

        # Mock new retriever for updated config
        mock_new_retriever = Mock()
        cast(MagicMock, mock_retriever_class).return_value = mock_new_retriever

        # Update config
        service.update_config(budget_tokens=2000, mmr_lambda=0.8)

        # Verify config was updated
        assert service.query_config.budget_tokens == 2000
        assert service.query_config.mmr_lambda == 0.8

        # Note: Retriever is now created per-request with DocumentStore
        # so we don't check for service.retriever attribute
