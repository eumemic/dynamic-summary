"""Tests for chars_per_token computation on server startup."""

from __future__ import annotations

import os

import pytest
from openai import OpenAI

from ragzoom.config import IndexConfig
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.server.indexing_engine import IndexingEngine
from ragzoom.services.llm_service import LLMService
from ragzoom.vector_factory import create_vector_index


def _make_leaf(
    document_id: str, leaf_id: str, span_start: int, span_end: int, token_count: int
) -> NodeDataDict:
    """Create a leaf node dict with predictable chars_per_token ratio."""
    return {
        "node_id": leaf_id,
        "document_id": document_id,
        "height": 0,
        "span_start": span_start,
        "span_end": span_end,
        "token_count": token_count,
        "text": f"Leaf {leaf_id} text",
        "level_index": int(leaf_id.replace("leaf", "")) - 1,
    }


def _create_vector_factory() -> type[VectorIndex]:
    """Create a vector index factory for testing."""

    def factory(model_id: str) -> VectorIndex:
        backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        db_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")
        return create_vector_index(backend, db_url, model_id)

    return factory  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_chars_per_token_computed_on_startup(
    storage_backend: StorageBackend,
) -> None:
    """Test chars_per_token is computed on startup for existing documents.

    Spec: specs/client-managed-chunking.md § chars_per_token Tracking
    """
    index_config = IndexConfig.load(
        target_chunk_tokens=None,
        target_embedding_context_tokens=200,
    )

    # Set up document with leaves that have 4.0 chars/token ratio
    # 600 total chars / 150 total tokens = 4.0
    document_id = "test-startup-chars-per-token"
    storage_backend.clear_document(document_id)
    store = storage_backend.for_document(document_id)
    store.set_metadata(
        file_path="test.txt",
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
    )

    store.nodes.add_batch(
        [
            _make_leaf(document_id, "leaf1", 0, 100, 25),  # 100 chars / 25 tokens
            _make_leaf(document_id, "leaf2", 100, 300, 50),  # 200 chars / 50 tokens
            _make_leaf(document_id, "leaf3", 300, 600, 75),  # 300 chars / 75 tokens
        ]
    )

    # Sanity check: verify leaves are in DB with correct ratio
    assert store.nodes.get_avg_chars_per_token(document_id) == 4.0

    # Create new engine (simulates server startup)
    engine = IndexingEngine(
        store=storage_backend,
        llm_service=LLMService(index_config, api_key="test-key"),
        index_config=index_config,
        openai_client=OpenAI(api_key="test-key"),
        vector_index_factory=_create_vector_factory(),
        max_parallelism=1,
    )

    # Verify chars_per_token was computed and cached on startup
    assert engine._document_chars_per_token.get(document_id) == 4.0

    # Verify get_chars_per_token returns cached value
    assert engine.get_chars_per_token(document_id) == 4.0
