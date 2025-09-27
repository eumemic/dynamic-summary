from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from ragzoom.config import OperationalConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.validation import validate_document
from ragzoom.vector_api import Vector


@pytest.fixture()
def validator_store(storage_backend: StorageBackend) -> StorageBackend:
    return storage_backend


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _add_leaf(
    store: DocumentStore,
    *,
    node_id: str,
    start: int,
    end: int,
    level_index: int,
) -> None:
    store.nodes.add_node(
        node_id=node_id,
        text=node_id,
        embedding=[0.0],
        span_start=start,
        span_end=end,
        token_count=end - start,
        height=0,
        level_index=level_index,
    )


def _add_parent(
    store: DocumentStore,
    *,
    node_id: str,
    start: int,
    end: int,
    left: str,
    right: str | None,
    height: int,
    level_index: int,
) -> None:
    store.nodes.add_node(
        node_id=node_id,
        text=node_id,
        embedding=[0.0],
        span_start=start,
        span_end=end,
        left_child_id=left,
        right_child_id=right,
        token_count=end - start,
        height=height,
        level_index=level_index,
    )


def _build_four_leaf_tree(store: DocumentStore, document_id: str) -> None:
    leaf_ids = ["leaf-0", "leaf-1", "leaf-2", "leaf-3"]
    spans = [(0, 100), (100, 200), (200, 300), (300, 400)]

    for idx, (node_id, (start, end)) in enumerate(zip(leaf_ids, spans)):
        _add_leaf(store, node_id=node_id, start=start, end=end, level_index=idx)

    neighbor_updates = []
    for idx, node_id in enumerate(leaf_ids):
        preceding = leaf_ids[idx - 1] if idx > 0 else None
        following = leaf_ids[idx + 1] if idx + 1 < len(leaf_ids) else None
        neighbor_updates.append((node_id, preceding, following))
    store.nodes.update_neighbors_batch(neighbor_updates)

    _add_parent(
        store,
        node_id="parent-left",
        start=0,
        end=200,
        left="leaf-0",
        right="leaf-1",
        height=1,
        level_index=0,
    )
    _add_parent(
        store,
        node_id="parent-right",
        start=200,
        end=400,
        left="leaf-2",
        right="leaf-3",
        height=1,
        level_index=1,
    )

    store.nodes.update_parent_references_batch(
        [
            ("leaf-0", "parent-left"),
            ("leaf-1", "parent-left"),
            ("leaf-2", "parent-right"),
            ("leaf-3", "parent-right"),
        ]
    )
    store.nodes.update_neighbors_batch(
        [
            ("parent-left", None, "parent-right"),
            ("parent-right", "parent-left", None),
        ]
    )

    _add_parent(
        store,
        node_id="root",
        start=0,
        end=400,
        left="parent-left",
        right="parent-right",
        height=2,
        level_index=0,
    )
    store.nodes.update_parent_references_batch(
        [("parent-left", "root"), ("parent-right", "root")]
    )


def _build_misordered_parent_level(store: DocumentStore, document_id: str) -> None:
    leaf_ids = [f"leaf-{idx}" for idx in range(6)]
    spans = [(idx * 100, (idx + 1) * 100) for idx in range(6)]

    for idx, (node_id, (start, end)) in enumerate(zip(leaf_ids, spans, strict=True)):
        _add_leaf(store, node_id=node_id, start=start, end=end, level_index=idx)

    neighbor_updates = []
    for idx, node_id in enumerate(leaf_ids):
        preceding = leaf_ids[idx - 1] if idx > 0 else None
        following = leaf_ids[idx + 1] if idx + 1 < len(leaf_ids) else None
        neighbor_updates.append((node_id, preceding, following))
    store.nodes.update_neighbors_batch(neighbor_updates)

    parents = [
        ("parent-a", 0, 200, "leaf-0", "leaf-1", 1, 0),
        ("parent-b", 200, 400, "leaf-2", "leaf-3", 1, 1),
        ("parent-c", 400, 600, "leaf-4", "leaf-5", 1, 2),
    ]
    for node_id, start, end, left, right, height, level_index in parents:
        _add_parent(
            store,
            node_id=node_id,
            start=start,
            end=end,
            left=left,
            right=right,
            height=height,
            level_index=level_index,
        )

    parent_updates = [
        ("leaf-0", "parent-a"),
        ("leaf-1", "parent-a"),
        ("leaf-2", "parent-b"),
        ("leaf-3", "parent-b"),
        ("leaf-4", "parent-c"),
        ("leaf-5", "parent-c"),
    ]
    store.nodes.update_parent_references_batch(parent_updates)

    store.nodes.update_neighbors_batch(
        [
            ("parent-a", None, "parent-c"),
            ("parent-b", None, None),
            ("parent-c", "parent-a", None),
        ]
    )


def test_tree_validator_accepts_complete_tree(validator_store: StorageBackend) -> None:
    document_id = "validate-complete"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )
    _build_four_leaf_tree(store, document_id)

    report = validate_document(
        document_id=document_id, store=validator_store, require_complete=True
    )
    assert report.status == "ok"
    assert report.metrics["leaf_count"] == 4

    validator_store.clear_document(document_id)


def test_tree_validator_flags_incomplete_tree_when_required(
    validator_store: StorageBackend,
) -> None:
    document_id = "validate-incomplete"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )

    _add_leaf(store, node_id="leaf-0", start=0, end=100, level_index=0)
    _add_leaf(store, node_id="leaf-1", start=100, end=200, level_index=1)
    _add_leaf(store, node_id="leaf-2", start=200, end=300, level_index=2)
    neighbor_updates = [
        ("leaf-0", None, "leaf-1"),
        ("leaf-1", "leaf-0", "leaf-2"),
        ("leaf-2", "leaf-1", None),
    ]
    store.nodes.update_neighbors_batch(neighbor_updates)

    _add_parent(
        store,
        node_id="parent-left",
        start=0,
        end=200,
        left="leaf-0",
        right="leaf-1",
        height=1,
        level_index=0,
    )
    store.nodes.update_parent_references_batch(
        [("leaf-0", "parent-left"), ("leaf-1", "parent-left")]
    )

    report = validate_document(
        document_id=document_id, store=validator_store, require_complete=True
    )
    assert report.status == "failed"
    assert any(f.code == "tree.multiple_roots" for f in report.errors)

    validator_store.clear_document(document_id)


def test_tree_validator_detects_misordered_parent_neighbors(
    validator_store: StorageBackend,
) -> None:
    document_id = "validate-neighbor-chain"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )

    _build_misordered_parent_level(store, document_id)

    report = validate_document(document_id=document_id, store=validator_store)

    mismatch_codes = {
        (finding.code, finding.node_id)
        for finding in report.findings
        if finding.code.startswith("level_neighbors.")
    }

    assert ("level_neighbors.following_mismatch", "parent-a") in mismatch_codes
    assert ("level_neighbors.preceding_mismatch", "parent-b") in mismatch_codes
    assert ("level_neighbors.following_mismatch", "parent-b") in mismatch_codes
    assert ("level_neighbors.preceding_mismatch", "parent-c") in mismatch_codes

    validator_store.clear_document(document_id)


def test_cli_validate_command(
    monkeypatch: pytest.MonkeyPatch,
    validator_store: StorageBackend,
    runner: CliRunner,
) -> None:
    document_id = "cli-validate"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )
    _build_four_leaf_tree(store, document_id)

    from ragzoom import cli as cli_module

    class DummyVectorIndex:
        def __init__(self, ids: list[str]) -> None:
            self._ids = set(ids)

        def get_vectors(self, ids: list[str]) -> list[Vector]:
            vectors: list[Vector] = []
            for node_id in ids:
                if node_id in self._ids:
                    vectors.append(
                        Vector(
                            id=node_id,
                            vec=np.asarray([1.0], dtype=np.float32),
                            meta={"document_id": document_id},
                            model_id="dummy",
                            dim=1,
                        )
                    )
            return vectors

        def list_ids(self) -> list[str]:
            return list(self._ids)

        def search_similar(
            self,
            query_embedding: list[float],
            k: int,
            where: dict[str, str | int | float | bool | None] | None = None,
        ) -> list[Vector]:  # pragma: no cover - unused
            return []

        def upsert(
            self,
            items: list[tuple[str, list[float], dict[str, object]]],
        ) -> None:  # pragma: no cover - unused
            for node_id, _vec, _meta in items:
                self._ids.add(node_id)

        def delete(
            self,
            filter: dict[str, object] | None = None,
            ids: list[str] | None = None,
        ) -> int:  # pragma: no cover - unused
            return 0

    def fake_create_store(
        config: OperationalConfig, embedding_model: str
    ) -> StorageBackend:
        return validator_store

    def fake_create_vector_index(
        backend: str, database_url: str, embedding_model: str
    ) -> DummyVectorIndex:
        node_ids = [node.id for node in store.nodes.get_all()]
        return DummyVectorIndex(node_ids)

    monkeypatch.setattr(cli_module, "create_store_with_docker", fake_create_store)
    monkeypatch.setattr(cli_module, "create_vector_index", fake_create_vector_index)

    result = runner.invoke(cli_module.cli, ["validate", document_id, "--complete"])
    assert result.exit_code == 0
    assert "✅" in result.output

    validator_store.clear_document(document_id)
