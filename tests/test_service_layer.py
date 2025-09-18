"""Tests for the service layer implementation."""

import asyncio
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
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
            content_hash="hash",
            chunk_count=5,
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
        assert documents[0].chunk_count == 5
        assert documents[0].node_count == 10

    def test_get_system_status(self, storage_backend: StorageBackend) -> None:
        """Test getting system status with real backend."""
        # Create two docs with nodes
        for d in ("doc-a", "doc-b"):
            ds = storage_backend.for_document(d)
            ds.set_metadata(
                file_path=None,
                content_hash="h",
                chunk_count=0,
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


class TestIndexingService:
    """Test the IndexingService."""

    def test_index_document(self, storage_backend: StorageBackend) -> None:
        """Test indexing a document."""
        # Create configs
        index_config = IndexConfig.load()
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))
        # Patch OpenAI client
        mock_async_client = MagicMock()

        async def mock_embeddings(*args: object, **kwargs: object) -> object:
            from typing import cast

            input_texts = cast(list[str] | str, kwargs.get("input", []))
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            return MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts]
            )

        mock_async_client.embeddings.create = mock_embeddings
        mock_async_client.chat.completions.create = MagicMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content="Summary of left and right content")
                    )
                ]
            )
        )

        with patch(
            "ragzoom.services.llm_service.AsyncOpenAI", return_value=mock_async_client
        ):
            # Ensure any stale lock file is removed
            import os
            from pathlib import Path

            lock_path = Path("data/.ragzoom/locks/test-doc.lock")
            try:
                if lock_path.exists():
                    os.remove(lock_path)
            except Exception:
                pass

            service = IndexingService(storage_backend, index_config, operational_config)
            result = service.index_document("test text", document_id="test-doc")

            assert isinstance(result, IndexingResult)
            assert result.document_id == "test-doc"
            assert result.chunks_created >= 1
            assert result.tree_depth >= 0
            assert result.telemetry is None

    def test_append_requires_schema_version(
        self,
        storage_backend: StorageBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Ensure append fails cleanly when schema is missing version column."""

        index_config = IndexConfig.load()
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))
        monkeypatch.setenv("RAGZOOM_ENABLE_INCREMENTAL", "1")

        mock_async_client = MagicMock()

        async def mock_embeddings(*args: object, **kwargs: object) -> object:
            from typing import cast

            input_texts = cast(list[str] | str, kwargs.get("input", []))
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            return MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts]
            )

        mock_async_client.embeddings.create = mock_embeddings
        mock_async_client.chat.completions.create = MagicMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content="Summary of left and right content")
                    )
                ]
            )
        )

        with patch(
            "ragzoom.services.llm_service.AsyncOpenAI", return_value=mock_async_client
        ):
            service = IndexingService(storage_backend, index_config, operational_config)
            service.index_document("seed text", document_id="doc-append")

            doc = service.store.get_document_by_id("doc-append")
            assert doc is not None
            doc.version = None  # type: ignore[assignment]

            monkeypatch.setattr(
                service.store,
                "get_document_by_id",
                lambda _doc_id: doc,
            )

            async def attempt() -> None:
                await service.append_to_document_async(
                    document_id="doc-append",
                    new_text=" more text",
                    show_progress=False,
                )

            with pytest.raises(RuntimeError, match="documents.version"):
                asyncio.run(attempt())


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

        # Create service and test
        service = QueryService(mock_store, query_config, operational_config)
        result = service.execute_query("test query", "doc-123")

        assert isinstance(result, QueryResult)
        assert result.summary == "This is the summary"
        assert result.token_count == 50
        assert result.nodes_retrieved == 2
        assert result.tiling_size == 3

        mock_retriever.retrieve.assert_called_once_with(
            "test query", budget_tokens=1000, document_id="doc-123", num_seeds=None
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
        service = QueryService(mock_store, query_config, operational_config)

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
