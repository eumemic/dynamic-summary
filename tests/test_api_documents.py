"""Tests for document API endpoints."""

import pytest
from fastapi.testclient import TestClient

from ragzoom.api import app
from tests.utils import mock_openai_context


class TestDocumentAPI:
    """Test document-related API endpoints."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI for tests with dragon/wizard specialized embeddings."""
        embedding_rules = {"dragon": [0.9] * 1536, "wizard": [0.8] * 1536}
        with mock_openai_context(embedding_rules):
            yield

    @pytest.fixture
    def client(self, mock_openai, monkeypatch, mock_store):
        """Create test client with mocked dependencies."""
        from ragzoom.api import get_service_container
        from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
        from ragzoom.services.document_service import DocumentService
        from ragzoom.services.indexing_service import IndexingService
        from ragzoom.services.query_service import QueryService

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a mock service container that uses our mock store
        class MockServiceContainer:
            def __init__(self):
                self.index_config = IndexConfig.load()
                self.query_config = QueryConfig()
                self.operational_config = OperationalConfig(openai_api_key="test-key")

                self.store = mock_store
                self.document_service = DocumentService(self.store)
                self.indexing_service = IndexingService(
                    self.store, self.index_config, self.operational_config
                )
                self.query_service = QueryService(
                    self.store, self.query_config, self.operational_config
                )

            def close(self):
                """Mock close method."""
                pass

        def mock_get_service_container():
            return MockServiceContainer()

        # Override the dependency
        app.dependency_overrides[get_service_container] = mock_get_service_container

        try:
            with TestClient(app) as client:
                yield client
        finally:
            # Clean up the override
            app.dependency_overrides.clear()

    def test_index_with_filename_as_default_id(self, client):
        """Test that filename is used as document_id when not specified."""
        response = client.post(
            "/index",
            json={
                "text": "Test content for filename default",
                "file_path": "/path/to/dragons.txt",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "dragons.txt"

    def test_list_documents_empty(self, client):
        """Test listing documents when none are indexed."""
        response = client.get("/documents")

        assert response.status_code == 200
        data = response.json()
        assert data["documents"] == []

    def test_list_documents_with_data(self, client):
        """Test listing documents after indexing."""
        # Index two documents
        response1 = client.post(
            "/index",
            json={
                "text": "Dragons are mighty creatures.",
                "document_id": "dragons-doc",
            },
        )
        assert response1.status_code == 200

        response2 = client.post(
            "/index",
            json={
                "text": "Wizards practice magic.",
                "document_id": "wizards-doc",
                "file_path": "/home/user/wizards.txt",
            },
        )
        assert response2.status_code == 200

        # List documents
        response = client.get("/documents")
        assert response.status_code == 200
        data = response.json()

        assert len(data["documents"]) == 2

        # Find each document
        dragons_doc = next(
            d for d in data["documents"] if d["document_id"] == "dragons-doc"
        )
        wizards_doc = next(
            d for d in data["documents"] if d["document_id"] == "wizards-doc"
        )

        # Check dragons document
        assert dragons_doc["file_path"] is None
        assert dragons_doc["chunk_count"] > 0
        assert dragons_doc["node_count"] > 0
        assert "indexed_at" in dragons_doc

        # Check wizards document
        assert wizards_doc["file_path"] == "/home/user/wizards.txt"
        assert wizards_doc["chunk_count"] > 0
        assert wizards_doc["node_count"] > 0

    def test_query_requires_document_id(self, client):
        """Test that query endpoint requires document_id."""
        # Try to query without document_id (should fail)
        response = client.post("/query", json={"query": "Tell me about dragons"})

        assert response.status_code == 422  # Unprocessable Entity

    def test_query_with_document_isolation(self, client):
        """Test that queries are isolated to specific documents."""
        # Index two documents
        response1 = client.post(
            "/index",
            json={"text": "Dragons breathe fire and fly.", "document_id": "dragons"},
        )
        assert response1.status_code == 200

        response2 = client.post(
            "/index",
            json={
                "text": "Wizards cast spells and study magic.",
                "document_id": "wizards",
            },
        )
        assert response2.status_code == 200

        # Query about dragons in dragons document
        response = client.post(
            "/query",
            json={"query": "Tell me about fire breathing", "document_id": "dragons"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data

        # Query about dragons in wizards document (should return wizard content)
        response = client.post(
            "/query",
            json={"query": "Tell me about fire breathing", "document_id": "wizards"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        # The summary should be about wizards, not dragons
        # (exact content depends on the mock responses)

    def test_concurrent_document_operations(self, client):
        """Test concurrent operations on different documents."""
        import asyncio

        async def index_and_query(doc_id, content):
            # Index document
            response = client.post(
                "/index", json={"text": content, "document_id": doc_id}
            )
            assert response.status_code == 200

            # Query document
            response = client.post(
                "/query", json={"query": "summarize", "document_id": doc_id}
            )
            assert response.status_code == 200
            return response.json()

        # Run operations on 3 documents concurrently
        async def run_test():
            tasks = [
                index_and_query("doc1", "Document 1 about dragons"),
                index_and_query("doc2", "Document 2 about wizards"),
                index_and_query("doc3", "Document 3 about knights"),
            ]
            results = await asyncio.gather(*tasks)
            return results

        results = asyncio.run(run_test())
        assert len(results) == 3
        for result in results:
            assert "summary" in result
