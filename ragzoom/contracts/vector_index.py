"""New VectorIndex protocol returning canonical Vector objects.

Backends implement this interface to provide search and vector retrieval.
The core consumes only this API (no backend-native types).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ragzoom.vector_api import Vector


@runtime_checkable
class VectorIndex(Protocol):
    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        k: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[Vector]: ...

    def get_vectors(self, ids: list[str]) -> list[Vector]: ...

    def upsert(
        self,
        items: list[tuple[str, list[float] | NDArray[np.float64], dict[str, object]]],
    ) -> None: ...

    def delete(
        self,
        filter: dict[str, object] | None = None,
        ids: list[str] | None = None,
    ) -> int: ...
