from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.telemetry_types import NodeTelemetryDict, TelemetryDataDict
from ragzoom.validation import validate_document


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
    # Set preceding_context: empty for span_start=0, placeholder for others
    preceding_context = "" if start == 0 else "preceding context"
    store.nodes._repo.update_preceding_context(node_id, preceding_context)


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
    preceding_tiling_ids: list[str] | None = None,
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
    # Set preceding_context as JSON array of node IDs

    if start == 0:
        preceding_context = "[]"
    elif preceding_tiling_ids is not None:
        preceding_context = json.dumps(preceding_tiling_ids)
    else:
        # No valid tiling provided - this will fail validation
        preceding_context = "[]"
    store.nodes._repo.update_preceding_context(node_id, preceding_context)


def _build_two_leaf_tree(store: DocumentStore, document_id: str) -> None:
    _add_leaf(
        store,
        node_id="leaf-left",
        start=0,
        end=100,
        level_index=0,
    )
    _add_leaf(
        store,
        node_id="leaf-right",
        start=100,
        end=200,
        level_index=1,
    )
    store.nodes.update_neighbors_batch(
        [
            ("leaf-left", None, "leaf-right"),
            ("leaf-right", "leaf-left", None),
        ]
    )

    _add_parent(
        store,
        node_id="root-node",
        start=0,
        end=200,
        left="leaf-left",
        right="leaf-right",
        height=1,
        level_index=0,
        preceding_tiling_ids=[],  # start=0, so empty tiling
    )
    store.nodes.update_parent_references_batch(
        [
            ("leaf-left", "root-node"),
            ("leaf-right", "root-node"),
        ]
    )


def _make_telemetry_payload(
    document_id: str, nodes: list[TreeNode]
) -> TelemetryDataDict:
    sorted_nodes = sorted(
        nodes, key=lambda node: (int(node.height), int(node.span_start))
    )
    telemetry_nodes: list[NodeTelemetryDict] = []
    for index, node in enumerate(sorted_nodes):
        node_payload: NodeTelemetryDict = {
            "node_id": node.id,
            "height": int(node.height),
            "created_at": float(index),
            "span": (int(node.span_start), int(node.span_end)),
        }
        telemetry_nodes.append(node_payload)

    return {
        "format_version": "4.3",
        "document_id": document_id,
        "source_document_tokens": sum(int(node.token_count) for node in nodes),
        "indexed_at": 0.0,
        "config": {
            "target_chunk_tokens": 200,
            "summary_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
        },
        "model_metadata": {},
        "system_prompts": {},
        "runtime_info": {
            "python_version": "3.11.0",
            "platform": "test",
            "ragzoom_version": "test",
        },
        "nodes": telemetry_nodes,
    }


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
        preceding_tiling_ids=[],  # start=0, so empty tiling
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
        preceding_tiling_ids=["parent-left"],  # covers [0, 200)
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
    """Test that forest completeness fails when adjacent siblings lack a parent.

    Forest completeness requires: for every pair of adjacent nodes at the same
    height (L with even level_index, R with odd level_index where
    L.level_index + 1 = R.level_index), they should have a common parent.

    This test creates 4 leaves where leaves 0,1 are paired but leaves 2,3 are NOT,
    even though they are adjacent siblings that should have a parent.
    """
    document_id = "validate-incomplete"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )

    # Create 4 leaves
    _add_leaf(store, node_id="leaf-0", start=0, end=100, level_index=0)
    _add_leaf(store, node_id="leaf-1", start=100, end=200, level_index=1)
    _add_leaf(store, node_id="leaf-2", start=200, end=300, level_index=2)
    _add_leaf(store, node_id="leaf-3", start=300, end=400, level_index=3)
    neighbor_updates = [
        ("leaf-0", None, "leaf-1"),
        ("leaf-1", "leaf-0", "leaf-2"),
        ("leaf-2", "leaf-1", "leaf-3"),
        ("leaf-3", "leaf-2", None),
    ]
    store.nodes.update_neighbors_batch(neighbor_updates)

    # Only create parent for leaves 0,1 - leaves 2,3 remain unpaired
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
    # Leaves 2 (even) and 3 (odd) are adjacent siblings without a parent
    assert any(f.code == "forest.unpaired_siblings" for f in report.errors)

    validator_store.clear_document(document_id)


def test_validate_document_telemetry_alignment(
    validator_store: StorageBackend,
) -> None:
    document_id = "telemetry-ok"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )

    _build_two_leaf_tree(store, document_id)
    nodes = list(store.nodes.get_all())
    telemetry = _make_telemetry_payload(document_id, nodes)

    report = validate_document(
        document_id=document_id,
        store=validator_store,
        telemetry=telemetry,
    )

    assert report.status == "ok"
    validator_store.clear_document(document_id)


def test_validate_document_reports_missing_telemetry_nodes(
    validator_store: StorageBackend,
) -> None:
    document_id = "telemetry-missing"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )

    _build_two_leaf_tree(store, document_id)
    nodes = list(store.nodes.get_all())
    telemetry = _make_telemetry_payload(document_id, nodes)
    telemetry_nodes = telemetry["nodes"]
    idx = 0
    while idx < len(telemetry_nodes):
        if telemetry_nodes[idx]["node_id"] == "leaf-right":
            del telemetry_nodes[idx]
        else:
            idx += 1

    report = validate_document(
        document_id=document_id,
        store=validator_store,
        telemetry=telemetry,
    )

    missing_codes = {
        finding.code for finding in report.findings if finding.node_id == "leaf-right"
    }
    assert "telemetry.node.missing" in missing_codes
    assert report.status == "failed"
    validator_store.clear_document(document_id)


def test_validate_document_reports_extra_telemetry_nodes(
    validator_store: StorageBackend,
) -> None:
    document_id = "telemetry-extra"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )

    _build_two_leaf_tree(store, document_id)
    nodes = list(store.nodes.get_all())
    telemetry = _make_telemetry_payload(document_id, nodes)
    phantom_node: NodeTelemetryDict = {
        "node_id": "phantom",
        "height": 0,
        "created_at": 99.0,
        "span": (0, 10),
    }
    telemetry["nodes"].append(phantom_node)

    report = validate_document(
        document_id=document_id,
        store=validator_store,
        telemetry=telemetry,
    )

    extra_codes = {
        finding.code for finding in report.findings if finding.node_id == "phantom"
    }
    assert "telemetry.node.unexpected" in extra_codes
    assert report.status == "failed"
    validator_store.clear_document(document_id)


def test_validate_document_reports_span_mismatch(
    validator_store: StorageBackend,
) -> None:
    document_id = "telemetry-span"
    store: DocumentStore = validator_store.add_document(
        document_id=document_id,
        file_path=None,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-5-mini",
    )

    _build_two_leaf_tree(store, document_id)
    nodes = list(store.nodes.get_all())
    telemetry = _make_telemetry_payload(document_id, nodes)
    for node in telemetry["nodes"]:
        if node["node_id"] == "leaf-left":
            node["span"] = (0, 150)
            break

    report = validate_document(
        document_id=document_id,
        store=validator_store,
        telemetry=telemetry,
    )

    mismatch_codes = {
        finding.code for finding in report.findings if finding.node_id == "leaf-left"
    }
    assert "telemetry.mismatch.span" in mismatch_codes
    assert report.status == "failed"
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


# NOTE: CLI validate tests moved to tests/test_cli.py since the validate command
# now uses gRPC. The tests below were removed as part of the gRPC migration:
# - test_cli_validate_command: tested --complete and --telemetry-file options
# - test_cli_validate_command_reports_telemetry_mismatch: tested telemetry mismatch
#
# These features are no longer exposed via CLI; the gRPC validate command only
# supports fast (SQL-only) validation. See tests/test_cli.py for gRPC-based tests:
# - test_cli_validate_uses_grpc
# - test_cli_validate_shows_errors_on_failure
# - test_cli_validate_handles_not_found
# - test_cli_validate_server_option


# --- Fast (SQL-based) Validation Tests ---


def test_fast_validation_accepts_complete_tree(validator_store: StorageBackend) -> None:
    """Test that fast validation passes for a valid complete tree."""
    document_id = "test-fast-validation-complete"
    doc_store = validator_store.for_document(document_id)
    _build_two_leaf_tree(doc_store, document_id)

    # Run fast validation
    report = validate_document(
        document_id=document_id,
        store=validator_store,
        fast=True,
    )

    assert report.status == "ok", f"Unexpected errors: {report.errors}"
    assert report.metrics["node_count"] == 3  # 2 leaves + 1 parent
    assert report.metrics["leaf_count"] == 2
    assert report.metrics["root_count"] == 1

    validator_store.clear_document(document_id)


def test_fast_validation_detects_leaf_gap(validator_store: StorageBackend) -> None:
    """Test that fast validation detects gaps between leaves."""
    document_id = "test-fast-validation-gap"
    doc_store = validator_store.for_document(document_id)

    # Create leaves with a gap (0-100, 150-200 instead of 100-150)
    _add_leaf(doc_store, node_id="leaf-0", start=0, end=100, level_index=0)
    _add_leaf(doc_store, node_id="leaf-1", start=150, end=200, level_index=1)  # Gap!

    report = validate_document(
        document_id=document_id,
        store=validator_store,
        fast=True,
    )

    assert report.status == "failed"
    gap_errors = [e for e in report.errors if e.code == "leaf.gap"]
    assert len(gap_errors) > 0, f"Expected leaf.gap error, got: {report.errors}"

    validator_store.clear_document(document_id)


def test_fast_validation_detects_broken_parent_ref(
    validator_store: StorageBackend,
) -> None:
    """Test that fast validation detects broken parent references."""
    document_id = "test-fast-validation-broken-parent"
    doc_store = validator_store.for_document(document_id)

    # Create leaf with a non-existent parent
    doc_store.nodes.add_node(
        node_id="orphan-leaf",
        text="orphan",
        embedding=[0.0],
        span_start=0,
        span_end=100,
        parent_id="non-existent-parent",  # Broken reference
        token_count=100,
        height=0,
        level_index=0,
    )

    report = validate_document(
        document_id=document_id,
        store=validator_store,
        fast=True,
    )

    assert report.status == "failed"
    # Full validation uses 'parent.mismatch' for this case
    broken_refs = [
        e
        for e in report.errors
        if "parent" in e.code.lower() or "broken" in e.code.lower()
    ]
    assert len(broken_refs) > 0, f"Expected parent ref error, got: {report.errors}"

    validator_store.clear_document(document_id)


def test_fast_validation_detects_non_binary_tree(
    validator_store: StorageBackend,
) -> None:
    """Test that fast validation detects nodes with only one child."""
    document_id = "test-fast-validation-non-binary"
    doc_store = validator_store.for_document(document_id)

    # Create a leaf
    _add_leaf(doc_store, node_id="only-child", start=0, end=100, level_index=0)

    # Create parent with only left child (violates perfect binary tree)
    doc_store.nodes.add_node(
        node_id="bad-parent",
        text="parent",
        embedding=[0.0],
        span_start=0,
        span_end=100,
        left_child_id="only-child",
        right_child_id=None,  # Missing right child!
        token_count=100,
        height=1,
        level_index=0,
    )
    doc_store.nodes._repo.update_parent_references_batch([("only-child", "bad-parent")])

    report = validate_document(
        document_id=document_id,
        store=validator_store,
        fast=True,
    )

    assert report.status == "failed"
    # Full validation uses 'tree.one_child', SQL validation uses 'tree.not_binary'
    binary_errors = [
        e
        for e in report.errors
        if "binary" in e.code.lower() or "child" in e.code.lower()
    ]
    assert len(binary_errors) > 0, f"Expected binary tree error, got: {report.errors}"

    validator_store.clear_document(document_id)


def test_fast_validation_same_results_as_full(validator_store: StorageBackend) -> None:
    """Test that fast validation finds same structural issues as full validation."""
    document_id = "test-fast-vs-full"
    doc_store = validator_store.for_document(document_id)
    _build_two_leaf_tree(doc_store, document_id)

    # Run both validation modes
    fast_report = validate_document(
        document_id=document_id,
        store=validator_store,
        fast=True,
    )
    full_report = validate_document(
        document_id=document_id,
        store=validator_store,
        fast=False,
    )

    # Both should pass for a valid tree
    assert fast_report.status == "ok"
    assert full_report.status == "ok"

    # Metrics should match
    assert fast_report.metrics["node_count"] == full_report.metrics["node_count"]
    assert fast_report.metrics["leaf_count"] == full_report.metrics["leaf_count"]
    assert fast_report.metrics["root_count"] == full_report.metrics["root_count"]

    validator_store.clear_document(document_id)
