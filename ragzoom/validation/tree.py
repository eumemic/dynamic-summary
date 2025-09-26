"""Tree validation entry points and invariants."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore

Severity = Literal["error", "warning"]


@dataclass(slots=True)
class ValidationFinding:
    """Individual validation result."""

    code: str
    message: str
    severity: Severity = "error"
    node_id: str | None = None


@dataclass(slots=True)
class ValidationReport:
    """Aggregated findings for a single validation invocation."""

    document_id: str
    findings: list[ValidationFinding] = field(default_factory=list)
    metrics: dict[str, int] = field(default_factory=dict)

    @property
    def status(self) -> Literal["ok", "failed"]:
        return "failed" if any(f.severity == "error" for f in self.findings) else "ok"

    @property
    def errors(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity == "error"]

    @property
    def warnings(self) -> list[ValidationFinding]:
        return [f for f in self.findings if f.severity == "warning"]


@dataclass(slots=True)
class DocumentSnapshot:
    """In-memory snapshot of a document's tree state."""

    document_id: str
    store: DocumentStore
    nodes: list[TreeNode]
    leaves: list[TreeNode]
    parentless: list[TreeNode]
    _node_lookup: dict[str, TreeNode] | None = field(
        default=None, init=False, repr=False
    )

    @property
    def node_lookup(self) -> dict[str, TreeNode]:
        if self._node_lookup is None:
            self._node_lookup = {node.id: node for node in self.nodes}
        return self._node_lookup


class Invariant(Protocol):
    """Protocol for validation invariant callables."""

    def __call__(self, snapshot: DocumentSnapshot) -> list[ValidationFinding]: ...


def validate_document(
    *,
    document_id: str,
    store: StorageBackend,
    vector_index: VectorIndex | None = None,
    require_complete: bool = False,
) -> ValidationReport:
    """Validate invariants for a single document.

    Args:
        document_id: Target document identifier
        store: Storage backend providing document access
        vector_index: Optional vector index (reserved for future checks)
        require_complete: Whether to assert the tree has converged to a single root

    Returns:
        ValidationReport capturing errors and warnings
    """

    doc_store = store.for_document(document_id)
    snapshot = _build_snapshot(document_id, doc_store)

    invariants: list[Invariant] = [
        _nodes_exist,
        _leaf_spans_are_ordered,
        _parent_child_relationships,
        _neighbor_consistency,
    ]

    findings: list[ValidationFinding] = []
    for invariant in invariants:
        findings.extend(invariant(snapshot))

    findings.extend(_completeness_check(snapshot, require_complete=require_complete))

    metrics = {
        "node_count": len(snapshot.nodes),
        "leaf_count": len(snapshot.leaves),
        "parentless_count": len(snapshot.parentless),
    }

    # vector_index currently unused but kept to avoid unused-argument lint
    _ = vector_index

    return ValidationReport(document_id=document_id, findings=findings, metrics=metrics)


def _build_snapshot(document_id: str, doc_store: DocumentStore) -> DocumentSnapshot:
    nodes = doc_store.nodes.get_all()
    leaves = doc_store.nodes.get_leaves()
    parentless = doc_store.nodes.get_parentless_nodes()
    return DocumentSnapshot(
        document_id=document_id,
        store=doc_store,
        nodes=nodes,
        leaves=leaves,
        parentless=parentless,
    )


def _nodes_exist(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    if snapshot.nodes:
        return []
    return [
        ValidationFinding(
            code="tree.empty",
            message="Document has no nodes",
        )
    ]


def _leaf_spans_are_ordered(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    leaves = sorted(snapshot.leaves, key=lambda node: node.span_start)
    findings: list[ValidationFinding] = []

    if not leaves:
        return findings

    first_leaf = leaves[0]
    if first_leaf.span_start != 0:
        findings.append(
            ValidationFinding(
                code="leaf.span_start",
                message=f"First leaf starts at {first_leaf.span_start} instead of 0",
                node_id=first_leaf.id,
            )
        )

    for previous, current in zip(leaves, leaves[1:]):
        if current.span_start != previous.span_end:
            findings.append(
                ValidationFinding(
                    code="leaf.gap",
                    message=(
                        "Leaf spans are not contiguous: "
                        f"{previous.id} ends at {previous.span_end}, "
                        f"{current.id} starts at {current.span_start}"
                    ),
                    node_id=current.id,
                )
            )
    return findings


def _parent_child_relationships(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    lookup = snapshot.node_lookup
    findings: list[ValidationFinding] = []

    for node in snapshot.nodes:
        parent_id = getattr(node, "parent_id", None)
        if parent_id:
            parent = lookup.get(parent_id)
            if parent is None:
                findings.append(
                    ValidationFinding(
                        code="parent.missing",
                        message=f"Parent {parent_id} referenced by {node.id} not found",
                        node_id=node.id,
                    )
                )
                continue
            if parent.left_child_id != node.id and parent.right_child_id != node.id:
                findings.append(
                    ValidationFinding(
                        code="parent.mismatch",
                        message=(
                            f"Parent {parent.id} does not reference {node.id} as a child"
                        ),
                        node_id=node.id,
                    )
                )

        # ensure children know parent if defined
        for label, child_id in ("left", node.left_child_id), (
            "right",
            node.right_child_id,
        ):
            if not child_id:
                continue
            child = lookup.get(child_id)
            if child is None:
                findings.append(
                    ValidationFinding(
                        code="child.missing",
                        message=f"{label.title()} child {child_id} missing for parent {node.id}",
                        node_id=node.id,
                    )
                )
                continue
            if child.parent_id != node.id:
                findings.append(
                    ValidationFinding(
                        code="child.parent_mismatch",
                        message=(
                            f"Child {child.id} reports parent {child.parent_id} "
                            f"but expected {node.id}"
                        ),
                        node_id=child.id,
                    )
                )
    return findings


def _neighbor_consistency(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    lookup = snapshot.node_lookup
    findings: list[ValidationFinding] = []

    for node in snapshot.nodes:
        prev_id = getattr(node, "preceding_neighbor_id", None)
        next_id = getattr(node, "following_neighbor_id", None)

        if prev_id:
            prev_node = lookup.get(prev_id)
            if prev_node is None:
                findings.append(
                    ValidationFinding(
                        code="neighbor.preceding_missing",
                        message=f"Preceding neighbor {prev_id} for {node.id} not found",
                        node_id=node.id,
                    )
                )
            elif getattr(prev_node, "following_neighbor_id", None) != node.id:
                findings.append(
                    ValidationFinding(
                        code="neighbor.preceding_backlink",
                        message=(
                            f"Preceding neighbor {prev_id} of {node.id} does not point back"
                        ),
                        node_id=node.id,
                    )
                )

        if next_id:
            next_node = lookup.get(next_id)
            if next_node is None:
                findings.append(
                    ValidationFinding(
                        code="neighbor.following_missing",
                        message=f"Following neighbor {next_id} for {node.id} not found",
                        node_id=node.id,
                    )
                )
            elif getattr(next_node, "preceding_neighbor_id", None) != node.id:
                findings.append(
                    ValidationFinding(
                        code="neighbor.following_backlink",
                        message=(
                            f"Following neighbor {next_id} of {node.id} does not point back"
                        ),
                        node_id=node.id,
                    )
                )
    return findings


def _completeness_check(
    snapshot: DocumentSnapshot, *, require_complete: bool
) -> list[ValidationFinding]:
    if not require_complete:
        return []

    parentless = snapshot.parentless
    if len(parentless) != 1:
        return [
            ValidationFinding(
                code="tree.multiple_roots",
                message=f"Expected single root, found {len(parentless)} parentless nodes",
            )
        ]

    root = parentless[0]
    leaves = sorted(snapshot.leaves, key=lambda node: node.span_start)
    if leaves:
        final_span = leaves[-1].span_end
        if root.span_start != 0 or root.span_end != final_span:
            return [
                ValidationFinding(
                    code="tree.root_span",
                    message=(
                        f"Root span [{root.span_start}, {root.span_end}) "
                        f"does not cover leaves ending at {final_span}"
                    ),
                    node_id=root.id,
                )
            ]
    return []
