"""Unit tests for the drift analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from ragzoom.analyze import DriftAnalyzer, DriftAnalyzerSettings
from ragzoom.document_store import DocumentStore
from ragzoom.retrieval.embedding_service import EmbeddingService


def _fake_embed(text: str, dim: int) -> list[float]:
    """Deterministic embedding used for tests."""

    vector = np.zeros(dim, dtype=float)
    payload = text.encode("utf-8")
    for idx, byte in enumerate(payload):
        vector[idx % dim] += (byte % 11) + 1
    return cast(list[float], vector.tolist())


class FakeEmbeddingService:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_query_embedding(
        self, text: str, document_id: str | None = None
    ) -> list[float]:
        return _fake_embed(text, self.dim)

    def embed_texts(
        self, texts: list[str], document_id: str | None = None
    ) -> list[list[float]]:
        return [_fake_embed(text, self.dim) for text in texts]


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


def test_drift_analyzer_recovers_weighted_delta() -> None:
    """Analyzer should surface the weighted delta contributed by a single frontier."""

    # Build simple tree: root -> mid nodes -> leaves
    leaf1 = FakeNode(
        id="leaf-1", text="alpha alpha alpha alpha", span_start=0, span_end=5, height=0
    )
    leaf2 = FakeNode(
        id="leaf-2",
        text="beta beta beta beta beta",
        span_start=5,
        span_end=15,
        height=0,
    )
    leaf3 = FakeNode(
        id="leaf-3",
        text="gamma gamma gamma gamma",
        span_start=15,
        span_end=25,
        height=0,
    )

    mid1 = FakeNode(
        id="mid-1",
        text="alpha beta necromancer",
        span_start=0,
        span_end=15,
        left_child_id=leaf1.id,
        right_child_id=leaf2.id,
        height=1,
    )
    mid2 = FakeNode(
        id="mid-2",
        text=leaf3.text,
        span_start=15,
        span_end=25,
        left_child_id=leaf3.id,
        right_child_id=None,
        height=1,
    )
    leaf1.parent_id = mid1.id
    leaf2.parent_id = mid1.id
    leaf3.parent_id = mid2.id

    root = FakeNode(
        id="root",
        text=f"{mid1.text}\n{mid2.text}",
        span_start=0,
        span_end=25,
        left_child_id=mid1.id,
        right_child_id=mid2.id,
        height=2,
    )
    mid1.parent_id = root.id
    mid2.parent_id = root.id

    nodes = [root, mid1, mid2, leaf1, leaf2, leaf3]
    store = FakeDocumentStore(nodes, root, embedding_model="text-embedding-3-small")
    document_store = cast(DocumentStore, store)

    dim = 4
    embedding_service_impl = FakeEmbeddingService(dim=dim)
    embedding_service = cast(EmbeddingService, embedding_service_impl)
    settings = DriftAnalyzerSettings(
        top_k_terms=3,
        max_frontier_report=5,
        max_vocab_terms=100,
        center_embeddings=False,
        frequency_correction=False,
    )
    analyzer = DriftAnalyzer(
        document_store,
        embedding_service,
        embedding_model="text-embedding-3-small",
        settings=settings,
        max_frontier_tokens=10,
        vector_dim=dim,
    )

    result = analyzer.analyze()

    # Expected delta: only mid-1 introduces drift, scaled by span weighting
    baseline_mid1 = "\n".join([leaf1.text, leaf2.text])
    delta_mid1 = np.asarray(_fake_embed(mid1.text, dim)) - np.asarray(
        _fake_embed(baseline_mid1, dim)
    )
    expected = 0.2 * delta_mid1  # derived from recursive weighting described in design

    np.testing.assert_allclose(result.drift_vector[:dim], expected, atol=1e-9)
    assert result.document_id == store.document_id
    assert result.root_frontier_ids == [mid1.id, mid2.id]


def test_frequency_correction_reduces_name_bias() -> None:
    """Frequency correction should down-weight name inflation in summaries."""

    leaf1 = FakeNode(
        id="leaf-1",
        text="Bilbo journeys with friends",
        span_start=0,
        span_end=10,
        height=0,
    )
    leaf2 = FakeNode(
        id="leaf-2",
        text="Necromancer whispers afar",
        span_start=10,
        span_end=20,
        height=0,
    )
    leaf1.parent_id = "root"
    leaf2.parent_id = "root"

    root = FakeNode(
        id="root",
        text="Bilbo Bilbo Bilbo Bilbo",
        span_start=0,
        span_end=20,
        left_child_id=leaf1.id,
        right_child_id=leaf2.id,
        height=1,
    )

    nodes = [root, leaf1, leaf2]
    store = FakeDocumentStore(nodes, root, embedding_model="text-embedding-3-small")
    document_store = cast(DocumentStore, store)

    dim = 4
    embedding_service = cast(EmbeddingService, FakeEmbeddingService(dim=dim))
    settings = DriftAnalyzerSettings(
        top_k_terms=3,
        max_frontier_report=5,
        max_vocab_terms=100,
        center_embeddings=False,
        frequency_correction=True,
        freq_calibration_pairs=2,
    )
    analyzer = DriftAnalyzer(
        document_store,
        embedding_service,
        embedding_model="text-embedding-3-small",
        settings=settings,
        max_frontier_tokens=50,
        vector_dim=dim,
    )

    result = analyzer.analyze()

    root_metric = next(
        metric for metric in result.node_metrics if metric.node_id == root.id
    )
    assert root_metric.explained_ratio is not None and root_metric.explained_ratio > 0
    assert root_metric.residual_angle_degrees is not None
    assert root_metric.local_angle_degrees is not None
    assert root_metric.residual_angle_degrees < root_metric.local_angle_degrees
    assert result.frequency_correction is True
    assert result.frequency_lambda == 1.0
    assert result.root_frontier_leaf_ratio == 1.0
    assert result.root_leaf_angle_degrees is not None
    assert result.lambda_sweep and len(result.lambda_sweep) >= 1
    assert result.vocab_sweep and len(result.vocab_sweep) == 3
    assert result.root_js_divergence is not None
    assert result.root_lift_terms
    assert result.root_node_id == root.id
    assert root_metric.term_outlier_score is not None
    assert root_metric.entity_added >= 0
