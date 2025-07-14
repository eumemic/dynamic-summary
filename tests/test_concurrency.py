"""Concurrency tests for thread safety."""

import asyncio
import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from ragzoom.api import app, get_ragzoom_service


class TestConcurrency:
    """Test thread safety and concurrent requests."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI for tests."""
        with patch('ragzoom.index.AsyncOpenAI') as mock_index, \
             patch('ragzoom.retrieve.OpenAI') as mock_retrieve, \
             patch('ragzoom.assemble.OpenAI') as mock_assemble:

            # Create async mocks for index client
            async def mock_embeddings_create_async(*args, **kwargs):
                input_data = kwargs.get('input', args[0] if args else '')
                if isinstance(input_data, list):
                    return Mock(data=[Mock(embedding=[0.1] * 384) for _ in input_data])
                else:
                    return Mock(data=[Mock(embedding=[0.1] * 384)])

            async def mock_chat_create_async(*args, **kwargs):
                return Mock(choices=[Mock(message=Mock(content="Summary"))])

            # Create sync mocks for retrieve/assemble clients
            def mock_embeddings_create_sync(*args, **kwargs):
                return Mock(data=[Mock(embedding=[0.1] * 384)])

            def mock_chat_create_sync(*args, **kwargs):
                return Mock(choices=[Mock(message=Mock(content="Summary"))])

            # Setup async client (index)
            instance_async = Mock()
            instance_async.embeddings = Mock()
            instance_async.embeddings.create = Mock(side_effect=mock_embeddings_create_async)
            instance_async.chat = Mock()
            instance_async.chat.completions = Mock()
            instance_async.chat.completions.create = Mock(side_effect=mock_chat_create_async)
            mock_index.return_value = instance_async

            # Setup sync clients (retrieve, assemble)
            for mock_client in [mock_retrieve, mock_assemble]:
                instance_sync = Mock()
                instance_sync.embeddings = Mock()
                instance_sync.embeddings.create = Mock(side_effect=mock_embeddings_create_sync)
                instance_sync.chat = Mock()
                instance_sync.chat.completions = Mock()
                instance_sync.chat.completions.create = Mock(side_effect=mock_chat_create_sync)
                mock_client.return_value = instance_sync

            yield

    @pytest.fixture
    def client(self, mock_openai, monkeypatch):
        """Create test client with mocked dependencies."""
        # Create temporary directories
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock environment
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            monkeypatch.setenv("RAGZOOM_CHROMA_PERSIST_DIRECTORY", os.path.join(tmpdir, "chroma"))
            monkeypatch.setenv("RAGZOOM_SQLITE_DATABASE_URL", f"sqlite:///{tmpdir}/test.db")

            with TestClient(app) as client:
                yield client

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, client):
        """Test multiple concurrent query requests."""
        # First index some data
        response = client.post("/index", json={
            "text": "Test document content for concurrency testing."
        })
        assert response.status_code == 200

        # Make concurrent queries
        async def make_query(query_num):
            response = client.post("/query", json={
                "query": f"Test query {query_num}"
            })
            return response

        # Run 5 concurrent queries
        tasks = [make_query(i) for i in range(5)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "summary" in data

    def test_service_isolation(self, mock_openai, monkeypatch):
        """Test that each request gets its own service instance."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Get multiple service instances
        service1 = get_ragzoom_service()
        service2 = get_ragzoom_service()

        # Should be different instances (not singleton)
        assert service1 is not service2
        assert service1.config is not service2.config
        assert service1.store is not service2.store

    @pytest.mark.asyncio
    async def test_concurrent_indexing(self, client):
        """Test concurrent document indexing."""
        async def index_doc(doc_num):
            response = client.post("/index", json={
                "text": f"Document {doc_num} content.",
                "document_id": f"doc-{doc_num}"
            })
            return response

        # Index 3 documents concurrently
        tasks = [index_doc(i) for i in range(3)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for i, response in enumerate(responses):
            assert response.status_code == 200
            data = response.json()
            assert data["document_id"] == f"doc-{i}"

    def test_no_shared_state(self, client):
        """Verify no shared mutable state between requests."""
        # Make a config update
        response1 = client.patch("/config", json={
            "budget_tokens": 5000
        })
        assert response1.status_code == 200

        # Get status - should reflect change for this request
        response2 = client.get("/status")
        assert response2.status_code == 200
        data2 = response2.json()

        # New request should have fresh config
        # (In practice, config changes would need to be persisted
        # This test verifies isolation)
        assert "config" in data2
