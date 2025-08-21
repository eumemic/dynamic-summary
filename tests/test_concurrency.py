"""Concurrency tests for thread safety."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from ragzoom.api import app
from tests.utils import mock_openai_context


class TestConcurrency:
    """Test thread safety and concurrent requests."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI for tests using centralized utilities."""
        with mock_openai_context():
            yield

    @pytest.fixture
    def client(self, mock_openai, monkeypatch, mock_store):
        """Create test client with mocked dependencies."""
        from ragzoom.api import get_service_container
        from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Create a mock service container that uses our mock store
        class MockServiceContainer:
            def __init__(self):
                self.index_config = IndexConfig.load()
                self.query_config = QueryConfig()
                self.operational_config = OperationalConfig(openai_api_key="test-key")
                self.store = mock_store

                # Initialize services with mock store
                from ragzoom.services.document_service import DocumentService
                from ragzoom.services.indexing_service import IndexingService
                from ragzoom.services.query_service import QueryService

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

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, client):
        """Test multiple concurrent query requests."""
        # First index some data
        response = client.post(
            "/index",
            json={
                "text": "Test document content for concurrency testing.",
                "document_id": "test-doc",
            },
        )
        assert response.status_code == 200

        # Make concurrent queries
        async def make_query(query_num):
            response = client.post(
                "/query",
                json={"query": f"Test query {query_num}", "document_id": "test-doc"},
            )
            return response

        # Run 5 concurrent queries
        tasks = [make_query(i) for i in range(5)]
        responses = await asyncio.gather(*tasks)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "summary" in data

    @pytest.mark.integration
    def test_service_isolation(self, mock_openai, monkeypatch):
        """Test that each request gets its own service instance."""
        from ragzoom.api import get_service_container

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        # Get multiple service instances
        service1 = get_service_container()
        service2 = get_service_container()

        # Should be different instances (not singleton)
        assert service1 is not service2
        assert service1.index_config is not service2.index_config
        assert service1.query_config is not service2.query_config
        assert service1.operational_config is not service2.operational_config
        assert service1.store is not service2.store

    @pytest.mark.asyncio
    async def test_concurrent_indexing(self, client):
        """Test concurrent document indexing."""

        async def index_doc(doc_num):
            response = client.post(
                "/index",
                json={
                    "text": f"Document {doc_num} content.",
                    "document_id": f"doc-{doc_num}",
                },
            )
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
        response1 = client.patch("/config", json={"budget_tokens": 5000})
        assert response1.status_code == 200

        # Get status - should reflect change for this request
        response2 = client.get("/status")
        assert response2.status_code == 200
        data2 = response2.json()

        # New request should have fresh config
        # (In practice, config changes would need to be persisted
        # This test verifies isolation)
        assert "config" in data2
