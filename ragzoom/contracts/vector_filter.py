"""Typed filter definitions for vector index searches.

Filters represent conditions that constrain which vectors are returned
from similarity searches. Each backend must implement translation of
all filter types to its native query language, or raise UnsupportedFilterError.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VectorFilter(ABC):
    """Base class for all vector index filters.

    Filters are immutable value objects. Each subclass represents
    a specific type of filter that backends must explicitly support.
    """

    pass


@dataclass(frozen=True, slots=True)
class DocumentIdFilter(VectorFilter):
    """Filter vectors by exact document_id match."""

    value: str


@dataclass(frozen=True, slots=True)
class SpanEndLtFilter(VectorFilter):
    """Filter vectors where span_end < threshold.

    Used to exclude vectors from the "verbatim region" during retrieval,
    ensuring seeds come only from before the verbatim horizon.
    """

    threshold: int


@dataclass(frozen=True, slots=True)
class SpanOverlapsFilter(VectorFilter):
    """Filter vectors whose span overlaps with [start, end).

    A node overlaps if: node.span_start < end AND node.span_end > start

    Used for windowed queries to restrict seeds to a specific document region.
    """

    start: int
    end: int
