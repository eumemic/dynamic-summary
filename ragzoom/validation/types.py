"""Types for SQL-pushdown validation.

These types represent validation results from SQL queries,
enabling fast validation without loading all nodes into memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SQLViolation:
    """A single validation violation from SQL query."""

    code: str
    message: str
    node_id: str | None = None
    severity: Literal["error", "warning"] = "error"


@dataclass
class SQLValidationMetrics:
    """Metrics computed from SQL queries."""

    node_count: int
    leaf_count: int
    root_count: int
    max_height: int
    embedded_count: int
    mergeable_pairs: int


@dataclass
class SQLValidationResult:
    """Result of SQL-pushdown validation queries.

    Each check field contains:
    - list[SQLViolation]: Violations found (empty list = check passed)
    - None: Check was not run
    """

    document_id: str
    metrics: SQLValidationMetrics

    # Structural checks
    empty_document: list[SQLViolation] | None = None
    leaf_gaps: list[SQLViolation] | None = None
    broken_parent_refs: list[SQLViolation] | None = None
    broken_child_refs: list[SQLViolation] | None = None
    neighbor_backlinks: list[SQLViolation] | None = None
    level_neighbor_chains: list[SQLViolation] | None = None
    perfect_binary_tree: list[SQLViolation] | None = None
    node_coordinates: list[SQLViolation] | None = None
    parent_span_union: list[SQLViolation] | None = None
    duplicate_coordinates: list[SQLViolation] | None = None

    # Optional checks
    leaf_chunk_size: list[SQLViolation] | None = None

    # Track which checks were run
    checks_run: list[str] = field(default_factory=list)

    @property
    def all_violations(self) -> list[SQLViolation]:
        """Flatten all violations into a single list."""
        result: list[SQLViolation] = []
        check_fields = [
            self.empty_document,
            self.leaf_gaps,
            self.broken_parent_refs,
            self.broken_child_refs,
            self.neighbor_backlinks,
            self.level_neighbor_chains,
            self.perfect_binary_tree,
            self.node_coordinates,
            self.parent_span_union,
            self.duplicate_coordinates,
            self.leaf_chunk_size,
        ]
        for check_result in check_fields:
            if check_result is not None:
                result.extend(check_result)
        return result

    @property
    def has_errors(self) -> bool:
        """Check if any error-severity violations exist."""
        return any(v.severity == "error" for v in self.all_violations)

    @property
    def error_count(self) -> int:
        """Count of error-severity violations."""
        return sum(1 for v in self.all_violations if v.severity == "error")

    @property
    def warning_count(self) -> int:
        """Count of warning-severity violations."""
        return sum(1 for v in self.all_violations if v.severity == "warning")
