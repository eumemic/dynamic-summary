"""Tests for document API endpoints."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from ragzoom.api import app


class TestDocumentAPI:
    """Test document-related API endpoints."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI for tests."""
        with (
            patch("ragzoom.index.AsyncOpenAI") as mock_index,
            patch("ragzoom.retrieve.OpenAI") as mock_retrieve,
        ):

            # Create async mocks for index client
            async def mock_embeddings_create_async(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    embeddings = []
                    for text in input_data:
                        if "dragon" in text.lower():
                            embeddings.append(Mock(embedding=[0.9] * 1536))
                        elif "wizard" in text.lower():
                            embeddings.append(Mock(embedding=[0.8] * 1536))
                        else:
                            embeddings.append(Mock(embedding=[0.5] * 1536))
                    return Mock(data=embeddings)
                else:
                    if "dragon" in input_data.lower():
                        return Mock(data=[Mock(embedding=[0.9] * 1536)])
                    elif "wizard" in input_data.lower():
                        return Mock(data=[Mock(embedding=[0.8] * 1536)])
                    else:
                        return Mock(data=[Mock(embedding=[0.5] * 1536)])

            async def mock_chat_create_async(*args, **kwargs):
                return Mock(
                    choices=[
                        Mock(message=Mock(content="Summary of left and right content"))
                    ]
                )

            # Create sync mocks
            def mock_embeddings_create_sync(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if "dragon" in input_data.lower():
                    return Mock(data=[Mock(embedding=[0.9] * 1536)])
                elif "wizard" in input_data.lower():
                    return Mock(data=[Mock(embedding=[0.8] * 1536)])
                else:
                    return Mock(data=[Mock(embedding=[0.5] * 1536)])

            # Setup async client
            instance_async = Mock()
            instance_async.embeddings = Mock()
            instance_async.embeddings.create = Mock(
                side_effect=mock_embeddings_create_async
            )
            instance_async.chat = Mock()
            instance_async.chat.completions = Mock()
            instance_async.chat.completions.create = Mock(
                side_effect=mock_chat_create_async
            )
            mock_index.return_value = instance_async

            # Setup sync clients
            instance_sync = Mock()
            instance_sync.embeddings = Mock()
            instance_sync.embeddings.create = Mock(
                side_effect=mock_embeddings_create_sync
            )
            mock_retrieve.return_value = instance_sync

            yield

    @pytest.fixture
    def client(self, mock_openai, monkeypatch):
        """Create test client with mocked dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("OPENAI_API_KEY", "test-key")
            monkeypatch.setenv(
                "RAGZOOM_DATABASE_URL", f"postgresql:///{tmpdir}/test.db"
            )

            with TestClient(app) as client:
                yield client

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
