"""Tree validation entry points and invariants."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import (
    TreeNode,
)
from ragzoom.contracts.vector_index import VectorIndex
from ragzoom.document_store import DocumentStore
from ragzoom.telemetry_types import TelemetryDataDict

Severity = Literal["error", "warning"]


@dataclass
class ValidationFinding:
    """Individual validation result."""

    code: str
    message: str
    severity: Severity = "error"
    node_id: str | None = None


@dataclass
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


@dataclass
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
    target_chunk_tokens: int | None = None,
    chunk_tolerance: float = 0.2,
    telemetry: TelemetryDataDict | None = None,
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
        _level_neighbor_chains,
        _left_balanced,
        _parent_span_union,
        _per_tree_leaf_depth,
    ]

    findings: list[ValidationFinding] = []
    for invariant in invariants:
        findings.extend(invariant(snapshot))

    findings.extend(_leaf_chunk_size(snapshot, target_chunk_tokens, chunk_tolerance))
    findings.extend(
        _vector_index_consistency(snapshot, vector_index) if vector_index else []
    )
    findings.extend(_completeness_check(snapshot, require_complete=require_complete))
    if telemetry is not None:
        findings.extend(_telemetry_consistency(snapshot, telemetry))

    metrics = {
        "node_count": len(snapshot.nodes),
        "leaf_count": len(snapshot.leaves),
        "parentless_count": len(snapshot.parentless),
    }

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


def _telemetry_consistency(
    snapshot: DocumentSnapshot, telemetry: TelemetryDataDict
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []

    telemetry_document_id = telemetry.get("document_id")
    if (
        isinstance(telemetry_document_id, str)
        and telemetry_document_id != snapshot.document_id
    ):
        findings.append(
            ValidationFinding(
                code="telemetry.document_mismatch",
                message=(
                    "Telemetry document_id does not match target document: "
                    f"{telemetry_document_id} != {snapshot.document_id}"
                ),
            )
        )

    raw_nodes = telemetry.get("nodes")
    if not isinstance(raw_nodes, list):
        findings.append(
            ValidationFinding(
                code="telemetry.nodes.invalid",
                message="Telemetry payload is missing a valid 'nodes' list",
            )
        )
        return findings

    telemetry_nodes: dict[str, Mapping[str, object]] = {}
    for index, entry in enumerate(raw_nodes):
        if not isinstance(entry, dict):
            findings.append(
                ValidationFinding(
                    code="telemetry.node.invalid",
                    message=(
                        "Telemetry entry at index " f"{index} is not a JSON object"
                    ),
                )
            )
            continue

        raw_node_id = entry.get("node_id")
        if not isinstance(raw_node_id, str) or not raw_node_id:
            findings.append(
                ValidationFinding(
                    code="telemetry.node.missing_id",
                    message=(
                        "Telemetry entry at index "
                        f"{index} is missing a valid node_id"
                    ),
                )
            )
            continue

        if raw_node_id in telemetry_nodes:
            findings.append(
                ValidationFinding(
                    code="telemetry.node.duplicate",
                    message=(
                        "Telemetry contains multiple entries for node " f"{raw_node_id}"
                    ),
                    node_id=raw_node_id,
                )
            )
            continue

        telemetry_nodes[raw_node_id] = entry

    node_lookup = snapshot.node_lookup

    for node_id, node in node_lookup.items():
        if node_id not in telemetry_nodes:
            findings.append(
                ValidationFinding(
                    code="telemetry.node.missing",
                    message=("Database node is missing from telemetry payload"),
                    node_id=node_id,
                )
            )

    for node_id in telemetry_nodes:
        if node_id not in node_lookup:
            findings.append(
                ValidationFinding(
                    code="telemetry.node.unexpected",
                    message="Telemetry references a node not present in storage",
                    node_id=node_id,
                )
            )

    for node_id, node in node_lookup.items():
        telemetry_entry = telemetry_nodes.get(node_id)
        if telemetry_entry is None:
            continue

        telemetry_height = telemetry_entry.get("height")
        if telemetry_height is None:
            findings.append(
                ValidationFinding(
                    code="telemetry.missing.height",
                    message="Telemetry entry is missing height",
                    node_id=node_id,
                )
            )
        else:
            coerced_height = _coerce_int(telemetry_height)
            if coerced_height is None:
                findings.append(
                    ValidationFinding(
                        code="telemetry.invalid.height",
                        message="Telemetry height is not an integer",
                        node_id=node_id,
                    )
                )
            elif coerced_height != node.height:
                findings.append(
                    ValidationFinding(
                        code="telemetry.mismatch.height",
                        message=(
                            "Telemetry height does not match stored height: "
                            f"{coerced_height} != {node.height}"
                        ),
                        node_id=node_id,
                    )
                )

        if "span" in telemetry_entry:
            telemetry_span = _coerce_span(telemetry_entry.get("span"))
            if telemetry_span is None:
                findings.append(
                    ValidationFinding(
                        code="telemetry.invalid.span",
                        message="Telemetry span is not a two-element integer sequence",
                        node_id=node_id,
                    )
                )
            else:
                expected_span = (int(node.span_start), int(node.span_end))
                if telemetry_span != expected_span:
                    findings.append(
                        ValidationFinding(
                            code="telemetry.mismatch.span",
                            message=(
                                "Telemetry span does not match stored span: "
                                f"{telemetry_span} != {expected_span}"
                            ),
                            node_id=node_id,
                        )
                    )

    return findings


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):  # bool is subclass of int; reject explicitly
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_span(value: object) -> tuple[int, int] | None:
    if isinstance(value, list | tuple) and len(value) == 2:
        start = _coerce_int(value[0])
        end = _coerce_int(value[1])
        if start is None or end is None:
            return None
        return (start, end)
    return None


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
    last_leaf = leaves[-1]
    document_end = max((leaf.span_end for leaf in leaves), default=0)
    if last_leaf.span_end != document_end:
        findings.append(
            ValidationFinding(
                code="leaf.span_end",
                message=(
                    f"Last leaf ends at {last_leaf.span_end}, expected {document_end}"
                ),
                node_id=last_leaf.id,
                severity="warning",
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


def _level_neighbor_chains(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    """Ensure neighbors at each height form a contiguous chain."""

    findings: list[ValidationFinding] = []
    by_height: dict[int, dict[int, TreeNode]] = defaultdict(dict)
    for node in snapshot.nodes:
        height = int(getattr(node, "height", 0))
        if height < 0:
            continue
        raw_index = getattr(node, "level_index", None)
        if raw_index is None:
            findings.append(
                ValidationFinding(
                    code="level_neighbors.missing_level_index",
                    message=f"Node {node.id} at height {height} is missing a level_index",
                    node_id=node.id,
                )
            )
            continue
        try:
            level_index = int(raw_index)
        except (TypeError, ValueError):
            findings.append(
                ValidationFinding(
                    code="level_neighbors.invalid_level_index",
                    message=(
                        f"Node {node.id} at height {height} has non-integer level_index "
                        f"{raw_index!r}"
                    ),
                    node_id=node.id,
                )
            )
            continue

        existing = by_height[height].get(level_index)
        if existing is not None:
            findings.append(
                ValidationFinding(
                    code="level_neighbors.duplicate_level_index",
                    message=(
                        f"Nodes {existing.id} and {node.id} at height {height} share level_index "
                        f"{level_index}"
                    ),
                    node_id=node.id,
                )
            )
            continue

        by_height[height][level_index] = node

    for height, nodes_by_index in by_height.items():
        if len(nodes_by_index) <= 1:
            continue
        for level_index, node in nodes_by_index.items():
            expected_prev = nodes_by_index.get(level_index - 1)
            expected_next = nodes_by_index.get(level_index + 1)

            prev_id = getattr(node, "preceding_neighbor_id", None)
            next_id = getattr(node, "following_neighbor_id", None)

            if expected_prev is None:
                if prev_id:
                    findings.append(
                        ValidationFinding(
                            code="level_neighbors.leading_unexpected",
                            message=(
                                f"Node {node.id} at height {height} (level_index {level_index}) "
                                f"has preceding neighbor {prev_id} despite no level_index {level_index - 1}"
                            ),
                            node_id=node.id,
                        )
                    )
            else:
                if prev_id != expected_prev.id:
                    findings.append(
                        ValidationFinding(
                            code="level_neighbors.preceding_mismatch",
                            message=(
                                f"Node {node.id} at height {height} (level_index {level_index}) should "
                                f"reference {expected_prev.id} as preceding neighbor but found "
                                f"{prev_id or 'None'}"
                            ),
                            node_id=node.id,
                        )
                    )

            if expected_next is None:
                if next_id:
                    findings.append(
                        ValidationFinding(
                            code="level_neighbors.trailing_unexpected",
                            message=(
                                f"Node {node.id} at height {height} (level_index {level_index}) "
                                f"has following neighbor {next_id} despite no level_index {level_index + 1}"
                            ),
                            node_id=node.id,
                        )
                    )
            else:
                if next_id != expected_next.id:
                    findings.append(
                        ValidationFinding(
                            code="level_neighbors.following_mismatch",
                            message=(
                                f"Node {node.id} at height {height} (level_index {level_index}) should "
                                f"reference {expected_next.id} as following neighbor but found "
                                f"{next_id or 'None'}"
                            ),
                            node_id=node.id,
                        )
                    )

    return findings


def _left_balanced(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for node in snapshot.nodes:
        if getattr(node, "right_child_id", None) and not getattr(
            node, "left_child_id", None
        ):
            findings.append(
                ValidationFinding(
                    code="tree.right_without_left",
                    message=(
                        f"Node {node.id} has a right child but no left child, violating left-balanced invariant"
                    ),
                    node_id=node.id,
                )
            )
    return findings


def _parent_span_union(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    lookup = snapshot.node_lookup
    findings: list[ValidationFinding] = []

    for node in snapshot.nodes:
        left_id = getattr(node, "left_child_id", None)
        right_id = getattr(node, "right_child_id", None)
        if not left_id and not right_id:
            continue

        left_node = lookup.get(left_id) if left_id else None
        right_node = lookup.get(right_id) if right_id else None

        if left_node is None:
            findings.append(
                ValidationFinding(
                    code="span.left_missing",
                    message=f"Parent {node.id} references missing left child {left_id}",
                    node_id=node.id,
                )
            )
            continue

        expected_start = left_node.span_start
        expected_end = left_node.span_end
        if right_node is not None:
            expected_end = right_node.span_end
            if left_node.span_end != right_node.span_start:
                findings.append(
                    ValidationFinding(
                        code="span.child_gap",
                        message=(
                            f"Children of {node.id} have non-adjacent spans: "
                            f"left ends {left_node.span_end}, right starts {right_node.span_start}"
                        ),
                        node_id=node.id,
                    )
                )
        if node.span_start != expected_start or node.span_end != expected_end:
            findings.append(
                ValidationFinding(
                    code="span.union_mismatch",
                    message=(
                        f"Parent {node.id} span [{node.span_start}, {node.span_end}) "
                        f"does not match union of child spans [{expected_start}, {expected_end})"
                    ),
                    node_id=node.id,
                )
            )

    return findings


def _leaf_chunk_size(
    snapshot: DocumentSnapshot,
    target_tokens: int | None,
    tolerance: float,
) -> list[ValidationFinding]:
    if target_tokens is None or target_tokens <= 0 or not snapshot.leaves:
        return []

    upper = int(target_tokens * (1 + tolerance))
    leaves = sorted(snapshot.leaves, key=lambda node: node.span_start)
    findings: list[ValidationFinding] = []

    for idx, leaf in enumerate(leaves):
        tokens = int(getattr(leaf, "token_count", 0))
        if tokens <= 0:
            continue
        if tokens > upper:
            findings.append(
                ValidationFinding(
                    code="leaf.tokens_over",
                    message=(
                        f"Leaf {leaf.id} has {tokens} tokens, above upper bound {upper}"
                    ),
                    node_id=leaf.id,
                    severity="warning",
                )
            )

    return findings


def _vector_index_consistency(
    snapshot: DocumentSnapshot, vector_index: VectorIndex
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    node_ids = list(snapshot.node_lookup.keys())
    batch_size = 256

    for start in range(0, len(node_ids), batch_size):
        chunk = node_ids[start : start + batch_size]
        try:
            vectors = vector_index.get_vectors(chunk)
        except Exception as exc:  # pragma: no cover - defensive
            for node_id in chunk:
                findings.append(
                    ValidationFinding(
                        code="vector.fetch_error",
                        message=f"Failed to fetch vector for {node_id}: {exc}",
                        node_id=node_id,
                    )
                )
            continue

        returned = {vec.id for vec in vectors}
        for node_id in chunk:
            if node_id not in returned:
                findings.append(
                    ValidationFinding(
                        code="vector.missing",
                        message=f"Embedding missing for node {node_id}",
                        node_id=node_id,
                    )
                )

    extra_ids = _vector_index_ids(vector_index)
    if extra_ids is not None:
        for vec_id in extra_ids:
            if vec_id not in snapshot.node_lookup:
                findings.append(
                    ValidationFinding(
                        code="vector.orphan",
                        message=f"Vector index contains id {vec_id} with no matching node",
                        node_id=vec_id,
                        severity="warning",
                    )
                )

    return findings


def _vector_index_ids(vector_index: VectorIndex) -> list[str] | None:
    candidates = [
        getattr(vector_index, "list_ids", None),
        getattr(vector_index, "ids", None),
    ]
    for candidate in candidates:
        if callable(candidate):
            try:
                ids = candidate()
                return list(ids)
            except Exception:  # pragma: no cover - best effort
                continue
    inner = getattr(vector_index, "_idx", None)
    if inner is not None and hasattr(inner, "_ids"):
        try:
            ids_attr = getattr(inner, "_ids")
            return list(ids_attr)
        except Exception:  # pragma: no cover
            return None
    return None


def _per_tree_leaf_depth(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    lookup = snapshot.node_lookup
    children: dict[str, list[str]] = {}
    for node in snapshot.nodes:
        if node.parent_id and node.parent_id in lookup:
            children.setdefault(node.parent_id, []).append(node.id)

    roots = [
        node
        for node in snapshot.nodes
        if node.parent_id is None or node.parent_id not in lookup
    ]
    findings: list[ValidationFinding] = []
    for root in roots:
        stack: list[tuple[str, int]] = [(root.id, 0)]
        visited: set[str] = set()
        leaf_depths: list[tuple[str, int]] = []
        while stack:
            node_id, depth = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            current = lookup.get(node_id)
            if current is None:
                continue
            child_ids = children.get(node_id, [])
            actual_children = [cid for cid in child_ids if cid in lookup]
            if not actual_children:
                leaf_depths.append((node_id, depth))
            else:
                for cid in actual_children:
                    stack.append((cid, depth + 1))
        depths = {depth for _, depth in leaf_depths}
        if len(depths) > 1:
            sample = ", ".join(
                f"{leaf_id}:{depth}" for leaf_id, depth in leaf_depths[:4]
            )
            findings.append(
                ValidationFinding(
                    code="leaf.depth",
                    message=(
                        f"Leaves under root {root.id} occur at multiple depths: {sorted(depths)} (sample: {sample})"
                    ),
                    node_id=root.id,
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
