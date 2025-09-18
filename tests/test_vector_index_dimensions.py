"""Tests for VectorIndex dimension handling and normalization behavior.

These tests replace legacy model-level embedding dimension tests with
VectorIndex-layer guarantees per the storage/vector decoupling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.vector_factory import create_vector_index


def _meta(
    doc: str, ss: int = 0, se: int = 0, pid: str = "", leaf: int = 1
) -> dict[str, object]:
    return {
        "document_id": doc,
        "span_start": ss,
        "span_end": se,
        "parent_id": pid,
        "is_leaf": leaf,
        "doc_version": 1,
    }


class TestVectorIndexDimensions:
    def test_upsert_consistent_dimension_succeeds(self) -> None:
        """Upserting vectors of a consistent dimension should succeed and normalize."""
        vi = create_vector_index(
            "python", "sqlite:///:memory:", "text-embedding-3-small"
        )

        # Two 3D vectors
        items: list[
            tuple[str, list[float] | NDArray[np.float64], dict[str, object]]
        ] = [
            ("a", [1.0, 2.0, 2.0], _meta("doc")),
            ("b", [0.0, 3.0, 4.0], _meta("doc")),
        ]
        vi.upsert(items)

        # Search with a 3D query
        res = vi.search_similar([1.0, 0.0, 0.0], 2, {"document_id": "doc"})
        assert len(res) == 2
        for v in res:
            # Normalized float32
            assert v.vec.dtype == np.float32
            norm = float(np.linalg.norm(v.vec))
            assert math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3)
            assert v.dim == v.vec.shape[0] == 3

        # get_vectors respects order and returns same ids
        ids = [r.id for r in res]
        got = vi.get_vectors(ids)
        assert [g.id for g in got] == ids

    def test_upsert_mismatched_dimension_raises(self) -> None:
        """Upserting a vector with a different dimensionality should error at the index layer."""
        vi = create_vector_index(
            "python", "sqlite:///:memory:", "text-embedding-3-small"
        )
        vi.upsert([("a", [1.0, 2.0, 3.0], _meta("doc"))])

        # Mismatched dim (4 instead of 3) should raise from numpy stacking
        with pytest.raises(Exception):
            vi.upsert([("b", [1.0, 2.0, 3.0, 4.0], _meta("doc"))])
