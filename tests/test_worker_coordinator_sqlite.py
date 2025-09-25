from collections.abc import Generator
from typing import cast

import pytest

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.server.worker_coordinator import (
    ReadyParentCandidate,
    compute_ready_parent_candidates,
)


@pytest.fixture()
def sqlite_store() -> Generator[DocumentStore, None, None]:
    backend = SQLiteStorageBackend()
    try:
        backend.add_document(
            document_id="doc",
            file_path=None,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-5-nano",
        )
        yield backend.for_document("doc")
    finally:
        backend.close()


NodePayloadValue = str | int | float | bool | list[float] | None
NodePayload = dict[str, NodePayloadValue]


def _leaf_payload(
    node_id: str,
    span_start: int,
    span_end: int,
    **kwargs: NodePayloadValue,
) -> NodePayload:
    payload: NodePayload = {
        "node_id": node_id,
        "text": f"text-{node_id}",
        "span_start": span_start,
        "span_end": span_end,
        "document_id": "doc",
        "token_count": span_end - span_start,
        "height": 0,
        "parent_id": None,
        "left_child_id": None,
        "right_child_id": None,
    }
    for key, value in kwargs.items():
        payload[key] = value
    return payload


def _add_batch(store: DocumentStore, *payloads: NodePayload) -> None:
    store.nodes.add_batch(
        cast(list[dict[str, NodePayloadValue]], list(payloads))  # type: ignore[arg-type]
    )


def test_detects_leaf_pairs(sqlite_store: DocumentStore) -> None:
    _add_batch(
        sqlite_store,
        _leaf_payload("L", 0, 10, following_neighbor_id="R"),
        _leaf_payload("R", 10, 20, preceding_neighbor_id="L"),
    )

    candidates = compute_ready_parent_candidates(sqlite_store)

    assert candidates == [
        ReadyParentCandidate(
            document_id="doc",
            left_child_id="L",
            right_child_id="R",
            height=0,
            span_start=0,
            span_end=20,
        )
    ]


def test_skips_nodes_with_existing_parents(sqlite_store: DocumentStore) -> None:
    _add_batch(
        sqlite_store,
        _leaf_payload("a", 0, 5, following_neighbor_id="b", parent_id="p"),
        _leaf_payload("b", 5, 10, preceding_neighbor_id="a", parent_id="p"),
    )

    assert compute_ready_parent_candidates(sqlite_store) == []


def test_returns_single_candidate_for_full_span_root(
    sqlite_store: DocumentStore,
) -> None:
    _add_batch(sqlite_store, _leaf_payload("root", 0, 12))

    candidates = compute_ready_parent_candidates(sqlite_store)
    assert candidates == [
        ReadyParentCandidate(
            document_id="doc",
            left_child_id="root",
            right_child_id=None,
            height=0,
            span_start=0,
            span_end=12,
        )
    ]


def test_requires_bidirectional_neighbor_link(sqlite_store: DocumentStore) -> None:
    _add_batch(
        sqlite_store,
        _leaf_payload("left", 0, 10, following_neighbor_id="right"),
        _leaf_payload("right", 10, 20),
    )

    assert compute_ready_parent_candidates(sqlite_store) == []
