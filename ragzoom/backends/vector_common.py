"""Shared helpers for VectorIndex adapter implementations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.vector_api import MetaDict

VectorUpsertItem: TypeAlias = tuple[
    str,
    list[float] | NDArray[np.float64],
    dict[str, object],
]

# We return the same structural type after normalization, but the helper ensures
# the embedding is a plain list[float] and the metadata dict is copied so the
# adapters may mutate it freely without surprising callers.
NormalizedUpsertItem: TypeAlias = tuple[
    str,
    list[float] | NDArray[np.float64],
    dict[str, object],
]


def coerce_float(value: object) -> float | None:
    """Coerce a value to float, returning None for invalid/missing values.

    Handles bools (True→1.0, False→0.0), ints, floats, and numeric strings.
    Returns None for None, empty strings, or non-convertible values.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def coerce_int(value: object) -> int:
    """Coerce a value to an integer, handling bools, numbers, and digit strings."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return 0


def coerce_str(value: object) -> str:
    """Coerce a value to string, handling None gracefully."""
    if value is None:
        return ""
    return str(value)


def normalize_metadata_from_dict(meta: dict[str, object]) -> MetaDict:
    """Normalize dict-based metadata to canonical MetaDict.

    Ensures all 8 standard metadata fields are present with correct types:
    - span_start, span_end, is_leaf, height, level_index, coord_version -> int
    - parent_id, document_id -> str
    """
    return {
        "span_start": coerce_int(meta.get("span_start", 0)),
        "span_end": coerce_int(meta.get("span_end", 0)),
        "parent_id": coerce_str(meta.get("parent_id", "")),
        "document_id": coerce_str(meta.get("document_id", "")),
        "is_leaf": coerce_int(meta.get("is_leaf", 0)),
        "height": coerce_int(meta.get("height", 0)),
        "level_index": coerce_int(meta.get("level_index", 0)),
        "coord_version": coerce_int(meta.get("coord_version", 0)),
    }


def normalize_metadata_from_object(meta: object) -> MetaDict:
    """Normalize object-based metadata (with attributes) to canonical MetaDict.

    Used when metadata comes from an object with attributes rather than a dict.
    """
    return {
        "span_start": coerce_int(getattr(meta, "span_start", 0)),
        "span_end": coerce_int(getattr(meta, "span_end", 0)),
        "parent_id": coerce_str(getattr(meta, "parent_id", "")),
        "document_id": coerce_str(getattr(meta, "document_id", "")),
        "is_leaf": coerce_int(getattr(meta, "is_leaf", 0)),
        "height": coerce_int(getattr(meta, "height", 0)),
        "level_index": coerce_int(getattr(meta, "level_index", 0)),
        "coord_version": coerce_int(getattr(meta, "coord_version", 0)),
    }


def normalize_upsert_items(
    items: Iterable[VectorUpsertItem],
) -> list[NormalizedUpsertItem]:
    """Coerce embedding tuples into a stable list-of-floats + JSON meta payload."""

    normalized: list[NormalizedUpsertItem] = []
    for node_id, embedding, meta in items:
        if isinstance(embedding, np.ndarray):
            dense = [float(x) for x in embedding.tolist()]
        else:
            dense = [float(x) for x in cast(Sequence[float], embedding)]

        meta_copy = {k: v for k, v in meta.items()}
        meta_copy.setdefault("height", 0)
        meta_copy.setdefault("level_index", 0)
        meta_copy.setdefault("coord_version", 0)

        normalized.append((str(node_id), dense, meta_copy))

    return normalized
