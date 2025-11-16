"""Unit tests for the summarization fidelity analyzer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from ragzoom.analyze import FidelityAnalyzerSettings, SummarizationFidelityAnalyzer
from ragzoom.document_store import DocumentStore
from ragzoom.retrieval.embedding_service import EmbeddingService


class MappingEmbeddingService:
    """Simple embedding service backed by a precomputed mapping."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping

    def get_query_embedding(
        self, text: str, document_id: str | None = None
    ) -> list[float]:
        return list(self.mapping[text])

    def embed_texts(
        self, texts: list[str], document_id: str | None = None
    ) -> list[list[float]]:
        return [list(self.mapping[text]) for text in texts]


@dataclass
class FakeNode:
    id: str
    text: str
    span_start: int
    span_end: int
    document_id: str = "doc-test"
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    height: int = 0


class FakeNodes:
    def __init__(self, nodes: list[FakeNode]) -> None:
        self._nodes = nodes

    def get_all(self) -> list[FakeNode]:
        return list(self._nodes)


class FakeTree:
    def __init__(self, root: FakeNode) -> None:
        self._root = root

    def get_root(self) -> FakeNode:
        return self._root


class FakeDocumentStore:
    def __init__(
        self, nodes: list[FakeNode], root: FakeNode, embedding_model: str
    ) -> None:
        self.document_id = root.document_id
        self.nodes = FakeNodes(nodes)
        self.tree = FakeTree(root)
        self._embedding_model = embedding_model

    def get_embedding_model(self) -> str:
        return self._embedding_model


def test_fidelity_analyzer_computes_basic_stats() -> None:
    """Analyzer should compute per-merge fidelities and aggregate stats."""

    leaf_a = FakeNode(
        id="leaf-a", text="Leaf A raw", span_start=0, span_end=10, height=0
    )
    leaf_b = FakeNode(
        id="leaf-b", text="Leaf B raw", span_start=10, span_end=20, height=0
    )
    mid1 = FakeNode(
        id="mid-1",
        text="Mid 1 summary",
        span_start=0,
        span_end=20,
        left_child_id=leaf_a.id,
        right_child_id=leaf_b.id,
        height=1,
    )
    leaf_a.parent_id = mid1.id
    leaf_b.parent_id = mid1.id

    mid2 = FakeNode(
        id="mid-2",
        text="Mid 2 summary",
        span_start=20,
        span_end=30,
        height=1,
    )
    mid2.parent_id = "root"

    root = FakeNode(
        id="root",
        text="Root summary",
        span_start=0,
        span_end=30,
        left_child_id=mid1.id,
        right_child_id=mid2.id,
        height=2,
    )
    mid1.parent_id = root.id

    nodes = [root, mid1, mid2, leaf_a, leaf_b]
    store = FakeDocumentStore(nodes, root, embedding_model="text-embedding-3-small")
    child_concat_mid1 = f"{leaf_a.text}\n{leaf_b.text}"
    child_concat_root = f"{mid1.text}\n{mid2.text}"
    mapping = {
        root.text: [1.0, 0.0],
        mid1.text: [0.6, 0.8],
        child_concat_mid1: [0.6, 0.8],
        child_concat_root: [0.4, 0.9],
    }
    embedding_service = MappingEmbeddingService(mapping)
    analyzer = SummarizationFidelityAnalyzer(
        cast(DocumentStore, store),
        cast(EmbeddingService, embedding_service),
        embedding_model="text-embedding-3-small",
        settings=FidelityAnalyzerSettings(
            top_k_worst=2,
            histogram_start=0.0,
            histogram_bucket_size=0.5,
            histogram_buckets=2,
        ),
        vector_dim=2,
    )

    result = analyzer.analyze()

    assert result.stats.count == 2
    metrics = {metric.node_id: metric for metric in result.metrics}
    assert set(metrics.keys()) == {"root", "mid-1"}
    mid1_metric = metrics["mid-1"]
    root_metric = metrics["root"]

    assert mid1_metric.fidelity == pytest.approx(1.0, rel=1e-9)
    expected_root_cos = np.dot(
        np.asarray(mapping[root.text]), np.asarray(mapping[child_concat_root])
    ) / (
        np.linalg.norm(mapping[root.text]) * np.linalg.norm(mapping[child_concat_root])
    )
    assert root_metric.fidelity == pytest.approx(expected_root_cos, rel=1e-9)
    assert result.stats.mean == pytest.approx((1.0 + expected_root_cos) / 2, rel=1e-9)
    assert result.worst_nodes[0].node_id == "root"


def test_histogram_and_extremes_reported() -> None:
    """Buckets should capture underflow/overflow counts."""

    leaf_a = FakeNode(id="leaf-a", text="Leaf", span_start=0, span_end=5, height=0)
    leaf_b = FakeNode(id="leaf-b", text="Leaf B", span_start=5, span_end=10, height=0)
    mid = FakeNode(
        id="mid",
        text="Mid summary",
        span_start=0,
        span_end=10,
        left_child_id=leaf_a.id,
        right_child_id=leaf_b.id,
        height=1,
    )
    leaf_a.parent_id = mid.id
    leaf_b.parent_id = mid.id
    sibling = FakeNode(
        id="sibling",
        text="Sibling summary",
        span_start=10,
        span_end=15,
        height=1,
    )
    sibling.parent_id = "root"

    root = FakeNode(
        id="root",
        text="Root summary",
        span_start=0,
        span_end=15,
        left_child_id=mid.id,
        right_child_id=sibling.id,
        height=2,
    )
    mid.parent_id = root.id

    child_concat_mid = f"{leaf_a.text}\n{leaf_b.text}"
    child_concat_root = f"{mid.text}\n{sibling.text}"
    low = math.sqrt(1 - 0.2**2)
    high = math.sqrt(1 - 0.95**2)
    mapping = {
        mid.text: [1.0, 0.0],
        child_concat_mid: [0.2, low],
        root.text: [1.0, 0.0],
        child_concat_root: [0.95, high],
    }
    store = FakeDocumentStore(
        [root, mid, sibling, leaf_a, leaf_b], root, "text-embedding-3-small"
    )
    analyzer = SummarizationFidelityAnalyzer(
        cast(DocumentStore, store),
        cast(EmbeddingService, MappingEmbeddingService(mapping)),
        embedding_model="text-embedding-3-small",
        settings=FidelityAnalyzerSettings(
            top_k_worst=2,
            histogram_start=0.3,
            histogram_bucket_size=0.3,
            histogram_buckets=2,
        ),
        vector_dim=2,
    )

    result = analyzer.analyze()

    # Expect both metrics present
    assert len(result.metrics) == 2
    assert result.histogram_underflow == 1  # fidelity 0.2 < 0.3
    assert result.histogram_overflow == 1  # fidelity 0.95 > 0.9 upper bound
    assert all(bucket.count == 0 for bucket in result.histogram)
