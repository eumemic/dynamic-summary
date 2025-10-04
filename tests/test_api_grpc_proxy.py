"""Tests for FastAPI endpoints proxying to the gRPC server."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from ragzoom.api import app, get_service_container
from ragzoom.client.grpc_client import ClearedDocumentResult
from ragzoom.services.indexing_service import IndexingResult


class _StubContainer:
    def __init__(self, address: str) -> None:
        self.grpc_address = address
        # Attributes required by other endpoints; set to simple mocks to avoid AttributeError
        self.document_service = MagicMock()
        self.query_service = MagicMock()
        self.index_config = MagicMock()
        self.query_config = MagicMock()
        self.operational_config = MagicMock()


def _with_container(container: _StubContainer) -> Callable[[], _StubContainer]:
    def _factory() -> _StubContainer:
        return container

    return _factory


def test_index_endpoint_appends_via_grpc() -> None:
    container = _StubContainer("grpc-address:9000")
    app.dependency_overrides[get_service_container] = _with_container(container)

    indexing_result = IndexingResult(
        document_id="doc-1",
        chunks_created=4,
        tree_depth=2,
        mutated_nodes=4,
        resummarized_nodes=0,
        new_leaves=1,
        telemetry=None,
    )

    client_mock = MagicMock()
    client_mock.__enter__.return_value = client_mock
    client_mock.__exit__.return_value = None
    client_mock.append_text.return_value = indexing_result

    with patch("ragzoom.api.GrpcRagzoomClient", return_value=client_mock) as factory:
        response = TestClient(app).post(
            "/index",
            json={"document_id": "doc-1", "text": "hello"},
        )

    app.dependency_overrides.pop(get_service_container, None)

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-1",
        "chunks_created": 4,
        "tree_depth": 2,
    }
    factory.assert_called_once_with("grpc-address:9000")
    client_mock.append_text.assert_called_once_with(
        document_id="doc-1",
        content=b"hello",
        collect_telemetry=False,
        replace_existing=False,
    )


def test_clear_endpoint_invokes_grpc_clear_document() -> None:
    container = _StubContainer("grpc-address:9000")
    app.dependency_overrides[get_service_container] = _with_container(container)

    cleared = ClearedDocumentResult(
        document_id="doc-1", deleted_nodes=3, document_existed=True
    )

    client_mock = MagicMock()
    client_mock.__enter__.return_value = client_mock
    client_mock.__exit__.return_value = None
    client_mock.clear_document.return_value = cleared

    with patch("ragzoom.api.GrpcRagzoomClient", return_value=client_mock):
        response = TestClient(app).post(
            "/clear",
            json={"document_id": "doc-1"},
        )

    app.dependency_overrides.pop(get_service_container, None)

    assert response.status_code == 200
    assert response.json() == {
        "document_id": "doc-1",
        "deleted_nodes": 3,
        "document_existed": True,
    }
    client_mock.clear_document.assert_called_once_with("doc-1")


def test_update_config_rejects_leaf_token_changes() -> None:
    container = _StubContainer("grpc-address:9000")
    app.dependency_overrides[get_service_container] = _with_container(container)

    response = TestClient(app).patch(
        "/config",
        json={"leaf_tokens": 2048},
    )

    app.dependency_overrides.pop(get_service_container, None)

    assert response.status_code == 400
    assert "leaf_tokens" in response.json()["detail"]
