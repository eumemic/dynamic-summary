from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock, patch

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.services.indexing_service import IndexingService


@pytest.mark.usefixtures("sqlite_backend")
class TestClearDeletesVectorsSQLite:
    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        return sqlite_store_factory("docC")

    def test_indexing_service_clears_vectors_before_reindex(
        self, sqlite_backend: StorageBackend
    ) -> None:
        # Set up configs
        idx_cfg = IndexConfig.load(target_chunk_tokens=50, preceding_context_tokens=25)
        op_cfg = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Mock OpenAI client used by LLMService via Indexing path
        with patch("openai.AsyncOpenAI") as mock_async_client:
            # minimal stubs for embeddings and chat
            async def _emb_create(**kwargs: object) -> object:
                inp = kwargs.get("input", [])
                if isinstance(inp, list):
                    n = len(inp)
                else:
                    n = 1
                return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in range(n)])

            mock_client = Mock()
            mock_client.embeddings.create = _emb_create
            mock_client.chat = Mock()
            mock_client.chat.completions = Mock()

            async def _chat_create(**kwargs: object) -> object:
                return Mock(choices=[Mock(message=Mock(content="summary"))])

            mock_client.chat.completions.create = _chat_create
            mock_async_client.return_value = mock_client

            # Spy vector index factory to assert delete(where=...) is called
            with patch(
                "ragzoom.services.indexing_service.create_vector_index"
            ) as mock_factory:
                mock_vi = Mock()
                mock_vi.delete = Mock(return_value=0)
                mock_factory.return_value = mock_vi

                service = IndexingService(sqlite_backend, idx_cfg, op_cfg)

                # First index
                service.index_document(
                    "Hello world", document_id="docC", show_progress=False
                )
                # Re-index same doc; should clear vectors before reindex
                service.index_document(
                    "Hello world again", document_id="docC", show_progress=False
                )

                # Assert delete called with document_id filter at least once
                filters = [
                    call.kwargs.get("filter") for call in mock_vi.delete.call_args_list
                ]
                assert any(
                    isinstance(f, dict) and f.get("document_id") == "docC"
                    for f in filters
                ), "VectorIndex.delete was not called with document_id filter"
