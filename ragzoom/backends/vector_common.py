"""Shared helpers for VectorIndex adapter implementations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

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


def coerce_int(value: object) -> int:
    """Coerce a value to an integer, handling bools, numbers, and digit strings."""
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return 0


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
