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
        _perfect_binary_trees,
        _node_coordinates,
        _parent_span_union,
        _per_tree_leaf_depth,
    ]

    findings: list[ValidationFinding] = []
    for invariant in invariants:
        findings.extend(invariant(snapshot))

    findings.extend(_leaf_chunk_size(snapshot, target_chunk_tokens, chunk_tolerance))
    findings.extend(
        _vector_index_consistency(snapshot, vector_index, require_complete)
        if vector_index
        else []
    )
    findings.extend(
        _completeness_check(
            snapshot,
            require_complete=require_complete,
            vector_index=vector_index,
        )
    )
    findings.extend(_preceding_context_check(snapshot))
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


def _perfect_binary_trees(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    """Validate that every tree in the forest is a perfect binary tree.

    In a perfect binary tree:
    - Every internal node has exactly 2 children (both left and right)
    - Leaves have no children
    """
    findings: list[ValidationFinding] = []
    for node in snapshot.nodes:
        has_left = node.left_child_id is not None
        has_right = node.right_child_id is not None

        if has_left != has_right:
            # One child but not both - violates perfect binary tree
            if has_left:
                findings.append(
                    ValidationFinding(
                        code="tree.left_only",
                        message=(
                            f"Node {node.id} has only a left child, "
                            "violating perfect binary tree invariant"
                        ),
                        node_id=node.id,
                    )
                )
            else:
                findings.append(
                    ValidationFinding(
                        code="tree.right_only",
                        message=(
                            f"Node {node.id} has only a right child, "
                            "violating perfect binary tree invariant"
                        ),
                        node_id=node.id,
                    )
                )
    return findings


def _node_coordinates(snapshot: DocumentSnapshot) -> list[ValidationFinding]:
    """Validate that node coordinates (height, level_index) are correct.

    Coordinate invariants:
    - height: 0 for leaves, parent.height = child.height + 1
    - level_index: Left child has even index, right child = left + 1,
                   parent.level_index = left_child.level_index // 2
    """
    lookup = snapshot.node_lookup
    findings: list[ValidationFinding] = []

    for node in snapshot.nodes:
        height = node.height
        level_index = node.level_index

        # Check leaf height
        is_leaf = node.left_child_id is None and node.right_child_id is None
        if is_leaf and height != 0:
            findings.append(
                ValidationFinding(
                    code="coord.leaf_height",
                    message=f"Leaf node has height {height}, expected 0",
                    node_id=node.id,
                )
            )

        # Check parent-child height relationship
        if node.left_child_id:
            left_child = lookup.get(node.left_child_id)
            if left_child and height != left_child.height + 1:
                findings.append(
                    ValidationFinding(
                        code="coord.height_mismatch",
                        message=(
                            f"Node height {height} != left child height {left_child.height} + 1"
                        ),
                        node_id=node.id,
                    )
                )

            # Check level_index relationship: parent = left_child // 2
            if left_child:
                expected_parent_index = left_child.level_index // 2
                if level_index != expected_parent_index:
                    findings.append(
                        ValidationFinding(
                            code="coord.level_index_mismatch",
                            message=(
                                f"Node level_index {level_index} != "
                                f"left_child.level_index // 2 ({left_child.level_index} // 2 = {expected_parent_index})"
                            ),
                            node_id=node.id,
                        )
                    )

                # Left child should have even level_index
                if left_child.level_index % 2 != 0:
                    findings.append(
                        ValidationFinding(
                            code="coord.left_child_odd",
                            message=(
                                f"Left child {left_child.id} has odd level_index {left_child.level_index}"
                            ),
                            node_id=node.id,
                        )
                    )

        if node.right_child_id:
            right_child = lookup.get(node.right_child_id)
            if right_child and height != right_child.height + 1:
                findings.append(
                    ValidationFinding(
                        code="coord.height_mismatch",
                        message=(
                            f"Node height {height} != right child height {right_child.height} + 1"
                        ),
                        node_id=node.id,
                    )
                )

            # Right child should have odd level_index
            if right_child and right_child.level_index % 2 != 1:
                findings.append(
                    ValidationFinding(
                        code="coord.right_child_even",
                        message=(
                            f"Right child {right_child.id} has even level_index {right_child.level_index}"
                        ),
                        node_id=node.id,
                    )
                )

        # Check sibling relationship: right = left + 1
        if node.left_child_id and node.right_child_id:
            left_child = lookup.get(node.left_child_id)
            right_child = lookup.get(node.right_child_id)
            if left_child and right_child:
                if right_child.level_index != left_child.level_index + 1:
                    findings.append(
                        ValidationFinding(
                            code="coord.sibling_index",
                            message=(
                                f"Right child level_index {right_child.level_index} != "
                                f"left child level_index {left_child.level_index} + 1"
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
    snapshot: DocumentSnapshot, vector_index: VectorIndex, require_complete: bool
) -> list[ValidationFinding]:
    """Check vector index consistency.

    Note: Missing embeddings are NOT checked here. When --complete is passed,
    missing embeddings are checked by _completeness_check. This function only
    checks for orphan vectors (vectors with no matching node).
    """
    findings: list[ValidationFinding] = []
    # Suppress unused parameter warning - kept for API consistency
    _ = require_complete

    # Check for orphan vectors (vectors with no matching node)
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
    snapshot: DocumentSnapshot,
    *,
    require_complete: bool,
    vector_index: VectorIndex | None = None,
) -> list[ValidationFinding]:
    """Check forest completeness when require_complete is True.

    Forest completeness means:
    1. Every adjacent pair of nodes at the same height has a parent.
       (L with even level_index, R with odd level_index where L.level_index + 1 = R.level_index)
    2. All leaves have embeddings in the vector index.

    This replaces the old single-root requirement, allowing forests as valid quiescent states.
    """
    if not require_complete:
        return []

    findings: list[ValidationFinding] = []

    # Group nodes by height
    by_height: dict[int, dict[int, TreeNode]] = defaultdict(dict)
    for node in snapshot.nodes:
        height = int(getattr(node, "height", 0))
        level_index = getattr(node, "level_index", None)
        if level_index is not None:
            by_height[height][int(level_index)] = node

    # Check that adjacent pairs have parents
    for height, nodes_by_index in by_height.items():
        level_indices = sorted(nodes_by_index.keys())
        for level_index in level_indices:
            # Only check left children (even level_index)
            if level_index % 2 != 0:
                continue

            left = nodes_by_index.get(level_index)
            right = nodes_by_index.get(level_index + 1)

            if left is None:
                continue

            # If there's a right sibling, both should have the same parent
            if right is not None:
                left_parent = getattr(left, "parent_id", None)
                right_parent = getattr(right, "parent_id", None)

                if left_parent is None and right_parent is None:
                    findings.append(
                        ValidationFinding(
                            code="forest.unpaired_siblings",
                            message=(
                                f"Adjacent siblings at height {height} "
                                f"(level_index {level_index}, {level_index + 1}) "
                                f"have no parent"
                            ),
                            node_id=left.id,
                        )
                    )
                elif left_parent != right_parent:
                    findings.append(
                        ValidationFinding(
                            code="forest.parent_mismatch",
                            message=(
                                f"Adjacent siblings at height {height} "
                                f"have different parents: {left_parent} vs {right_parent}"
                            ),
                            node_id=left.id,
                        )
                    )

    # Check all leaves have embeddings (only when --complete)
    if vector_index is not None:
        leaf_ids = [leaf.id for leaf in snapshot.leaves]
        if leaf_ids:
            try:
                vectors = vector_index.get_vectors(leaf_ids)
                returned_ids = {vec.id for vec in vectors}
                for leaf_id in leaf_ids:
                    if leaf_id not in returned_ids:
                        findings.append(
                            ValidationFinding(
                                code="forest.missing_embedding",
                                message=f"Leaf {leaf_id} has no embedding",
                                node_id=leaf_id,
                            )
                        )
            except Exception as exc:
                findings.append(
                    ValidationFinding(
                        code="forest.embedding_check_failed",
                        message=f"Failed to check embeddings: {exc}",
                    )
                )

    # Check preceding_context for leaves (only when --complete)
    # Inner nodes are checked separately by _preceding_context_check
    for node in snapshot.leaves:
        span_start = int(getattr(node, "span_start", 0))
        preceding_context = getattr(node, "preceding_context", None)

        if preceding_context is None:
            findings.append(
                ValidationFinding(
                    code="forest.missing_preceding_context",
                    message=(
                        f"Leaf (span_start={span_start}) has no preceding_context"
                    ),
                    node_id=node.id,
                )
            )
        elif span_start > 0 and preceding_context == "":
            findings.append(
                ValidationFinding(
                    code="forest.empty_preceding_context",
                    message=(
                        f"Leaf (span_start={span_start}) has empty preceding_context "
                        f"but span_start > 0"
                    ),
                    node_id=node.id,
                )
            )

    return findings


def _preceding_context_check(
    snapshot: DocumentSnapshot,
) -> list[ValidationFinding]:
    """Check that all nodes have valid preceding_context tilings.

    The preceding_context field stores a JSON array of node IDs representing
    the tiling that covers [0, span_start). For each node with preceding_context:
    - Parse the JSON array of node IDs
    - Validate it's a valid tiling (no gaps between spans)
    - Validate coverage: first node starts at 0, last node ends at span_start
    """
    import json

    findings: list[ValidationFinding] = []
    node_lookup = snapshot.node_lookup

    for node in snapshot.nodes:
        height = int(getattr(node, "height", 0))
        span_start = int(getattr(node, "span_start", 0))
        preceding_context = getattr(node, "preceding_context", None)

        # Skip leaves - they're checked by _completeness_check when --complete
        if height == 0:
            continue

        if preceding_context is None:
            findings.append(
                ValidationFinding(
                    code="forest.missing_preceding_context",
                    message=(
                        f"Inner node at height {height} (span_start={span_start}) "
                        f"has no preceding_context"
                    ),
                    node_id=node.id,
                )
            )
            continue

        # Nodes at span_start=0 should have empty tiling
        if span_start == 0:
            if preceding_context not in ("", "[]"):
                findings.append(
                    ValidationFinding(
                        code="preceding_context.nonempty_at_zero",
                        message=(
                            f"Node at span_start=0 has non-empty preceding_context: "
                            f"{preceding_context[:50]}..."
                        ),
                        node_id=node.id,
                    )
                )
            continue

        # Nodes at span_start>0 should have valid tiling
        if preceding_context in ("", "[]"):
            findings.append(
                ValidationFinding(
                    code="forest.empty_preceding_context",
                    message=(
                        f"Inner node at height {height} (span_start={span_start}) "
                        f"has empty preceding_context but span_start > 0"
                    ),
                    node_id=node.id,
                )
            )
            continue

        # Parse JSON array of node IDs
        try:
            tiling_ids = json.loads(preceding_context)
        except json.JSONDecodeError as e:
            findings.append(
                ValidationFinding(
                    code="preceding_context.invalid_json",
                    message=f"preceding_context is not valid JSON: {e}",
                    node_id=node.id,
                )
            )
            continue

        if not isinstance(tiling_ids, list):
            findings.append(
                ValidationFinding(
                    code="preceding_context.not_array",
                    message=f"preceding_context is not a JSON array: {type(tiling_ids).__name__}",
                    node_id=node.id,
                )
            )
            continue

        if not tiling_ids:
            findings.append(
                ValidationFinding(
                    code="preceding_context.empty_array",
                    message=(
                        f"Node at span_start={span_start} has empty tiling array "
                        f"but should cover [0, {span_start})"
                    ),
                    node_id=node.id,
                )
            )
            continue

        # Validate each node ID exists and collect spans
        tiling_spans: list[tuple[str, int, int]] = []
        for tiling_node_id in tiling_ids:
            if not isinstance(tiling_node_id, str):
                findings.append(
                    ValidationFinding(
                        code="preceding_context.invalid_id",
                        message=f"Tiling contains non-string ID: {tiling_node_id!r}",
                        node_id=node.id,
                    )
                )
                continue

            tiling_node = node_lookup.get(tiling_node_id)
            if tiling_node is None:
                findings.append(
                    ValidationFinding(
                        code="preceding_context.missing_node",
                        message=f"Tiling references missing node: {tiling_node_id}",
                        node_id=node.id,
                    )
                )
                continue

            tiling_spans.append(
                (tiling_node_id, int(tiling_node.span_start), int(tiling_node.span_end))
            )

        if not tiling_spans:
            continue  # Already reported errors above

        # Sort by span_start
        tiling_spans.sort(key=lambda x: x[1])

        # Check first node starts at 0
        first_id, first_start, first_end = tiling_spans[0]
        if first_start != 0:
            findings.append(
                ValidationFinding(
                    code="preceding_context.incomplete_start",
                    message=(
                        f"Tiling does not start at 0: first node {first_id} "
                        f"starts at {first_start}"
                    ),
                    node_id=node.id,
                )
            )

        # Check last node ends at span_start
        last_id, last_start, last_end = tiling_spans[-1]
        if last_end != span_start:
            findings.append(
                ValidationFinding(
                    code="preceding_context.incomplete_end",
                    message=(
                        f"Tiling does not end at span_start={span_start}: "
                        f"last node {last_id} ends at {last_end}"
                    ),
                    node_id=node.id,
                )
            )

        # Check for gaps between spans
        for i in range(len(tiling_spans) - 1):
            curr_id, curr_start, curr_end = tiling_spans[i]
            next_id, next_start, next_end = tiling_spans[i + 1]
            if curr_end != next_start:
                findings.append(
                    ValidationFinding(
                        code="preceding_context.gap",
                        message=(
                            f"Gap in tiling: {curr_id} ends at {curr_end}, "
                            f"{next_id} starts at {next_start}"
                        ),
                        node_id=node.id,
                    )
                )

    return findings
