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

NormalizedUpsertItem: TypeAlias = tuple[
    str,
    list[float] | NDArray[np.float64],
    dict[str, object],
]


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

        normalized.append(
            (
                str(node_id),
                dense,
                {k: v for k, v in meta.items()},
            )
        )

    return normalized
