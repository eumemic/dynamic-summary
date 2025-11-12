"""Frontier-aware semantic drift analyzer."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.model_info import ModelInfo
from ragzoom.retrieval.embedding_service import EmbeddingService
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)

Vector = NDArray[np.float64]

# Conservative per-model embedding limits (tokens). Default falls back to 8000.
EMBEDDING_CONTEXT_LIMITS: dict[str, int] = {
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8191,
}
DEFAULT_EMBED_LIMIT = 8000

WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z'\-]*")


@dataclass
class DriftAnalyzerSettings:
    """Runtime tuning knobs for the analyzer."""

    top_k_terms: int = 5
    max_frontier_report: int = 10
    max_vocab_terms: int = 512
    center_embeddings: bool = False
    max_vocab_ngram: int = 3
    min_term_length: int = 3


@dataclass
class NodeDriftMetric:
    """Human- and JSON-friendly per-node diagnostics."""

    node_id: str
    span_chars: int
    frontier_node_ids: list[str]
    summary_tokens: int
    frontier_tokens: int
    local_cosine: float | None
    local_angle_degrees: float | None
    compression_ratio: float | None


@dataclass
class DriftTerm:
    """Nearest-neighbour term for delta interpretation."""

    text: str
    score: float


@dataclass
class DriftAnalysisResult:
    """Complete result of a drift analysis run."""

    document_id: str
    embedding_model: str
    embedding_dim: int
    config_hash: str
    max_frontier_tokens: int
    fidelity: float | None
    fidelity_angle_degrees: float | None
    compression_ratio: float | None
    root_frontier_ids: list[str]
    drift_vector: list[float]
    drift_vector_norm: float
    drift_vector_preview: list[float]
    amplified_terms: list[DriftTerm]
    deemphasized_terms: list[DriftTerm]
    worst_nodes: list[NodeDriftMetric]
    node_metrics: list[NodeDriftMetric] = field(default_factory=list)
    root_direct_cosine: float | None = None
    root_direct_angle_degrees: float | None = None
    center_embeddings: bool = False

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""

        return {
            "document_id": self.document_id,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "config_hash": self.config_hash,
            "max_frontier_tokens": self.max_frontier_tokens,
            "center_embeddings": self.center_embeddings,
            "fidelity": self.fidelity,
            "fidelity_angle_degrees": self.fidelity_angle_degrees,
            "compression_ratio": self.compression_ratio,
            "root_direct_cosine": self.root_direct_cosine,
            "root_direct_angle_degrees": self.root_direct_angle_degrees,
            "root_frontier_ids": self.root_frontier_ids,
            "drift_vector": self.drift_vector,
            "drift_vector_norm": self.drift_vector_norm,
            "drift_vector_preview": self.drift_vector_preview,
            "amplified_terms": [
                {"text": t.text, "score": t.score} for t in self.amplified_terms
            ],
            "deemphasized_terms": [
                {"text": t.text, "score": t.score} for t in self.deemphasized_terms
            ],
            "worst_nodes": [metric.__dict__ for metric in self.worst_nodes],
            "node_metrics": [metric.__dict__ for metric in self.node_metrics],
        }


@dataclass
class _NodeComputation:
    node_id: str
    span_chars: int
    frontier_ids: list[str]
    summary_tokens: int
    frontier_tokens: int
    summary_vec: Vector
    frontier_vec: Vector
    local_delta: Vector
    expanded: bool


@dataclass
class _VocabularyTerm:
    text: str
    count: int
    vector: Vector
    norm: float
    idf: float


class DriftAnalyzer:
    """Scope-aligned frontier drift analyzer."""

    def __init__(
        self,
        document_store: DocumentStore,
        embedding_service: EmbeddingService,
        *,
        embedding_model: str,
        settings: DriftAnalyzerSettings | None = None,
        max_frontier_tokens: int | None = None,
        vector_dim: int | None = None,
    ) -> None:
        self.document_store = document_store
        self.embedding_service = embedding_service
        self.embedding_model = embedding_model
        self.settings = settings or DriftAnalyzerSettings()
        self.max_frontier_tokens = max_frontier_tokens or EMBEDDING_CONTEXT_LIMITS.get(
            embedding_model, DEFAULT_EMBED_LIMIT
        )
        model_info = ModelInfo()
        self.vector_dim = vector_dim or model_info.get_embedding_dimensions(
            embedding_model
        )
        self._zero = cast(Vector, np.zeros(self.vector_dim, dtype=np.float64))
        self._embedding_cache: dict[str, Vector] = {}
        self._node_records: dict[str, _NodeComputation] = {}
        self._vector_bank: list[Vector] = []
        self._vocabulary: list[_VocabularyTerm] = []
        self._doc_id = self.document_store.document_id or ""

    def analyze(self) -> DriftAnalysisResult:
        """Run analysis for the document attached to this store."""

        root = self.document_store.tree.get_root()
        if root is None:
            raise ValueError("Document has no root node; cannot analyze drift")

        self._doc_id = root.document_id or self.document_store.document_id or ""

        doc_model = self.document_store.get_embedding_model()
        if doc_model and doc_model != self.embedding_model:
            raise ValueError(
                f"Document stored embeddings with '{doc_model}' but analyzer was configured for '{self.embedding_model}'"
            )

        node_map = {node.id: node for node in self.document_store.nodes.get_all()}
        if root.id not in node_map:
            node_map[root.id] = root

        root_delta, _ = self._compute_drift(root, node_map)

        root_record = self._node_records.get(root.id)
        if not root_record:
            raise RuntimeError("Root metrics missing after analysis")

        root_vec = root_record.summary_vec
        frontiers_vec = root_record.frontier_vec

        drift_vector = root_delta
        drift_norm = float(np.linalg.norm(drift_vector))
        drift_preview = drift_vector[: min(16, drift_vector.size)].tolist()

        root_frontier_ids = root_record.frontier_ids

        center_mean: Vector | None = None
        if self.settings.center_embeddings and self._vector_bank:
            stacked = np.vstack(self._vector_bank)
            center_mean = cast(Vector, stacked.mean(axis=0))

        estimated_full = root_vec - drift_vector

        fidelity, fidelity_angle = self._cosine_pair(
            root_vec, estimated_full, center_mean
        )
        compression_ratio = self._compression_ratio(root_vec, estimated_full)
        root_direct_cos, root_direct_angle = self._cosine_pair(
            root_vec, frontiers_vec, center_mean
        )

        node_metrics = self._build_node_metrics(center_mean)
        worst_nodes = node_metrics[: self.settings.max_frontier_report]

        vocab_terms = self._build_vocabulary(list(node_map.values()))
        amplified, deprived = self._project_terms(drift_vector, vocab_terms)

        config_hash = self._compute_config_hash()

        return DriftAnalysisResult(
            document_id=self._doc_id,
            embedding_model=self.embedding_model,
            embedding_dim=self.vector_dim,
            config_hash=config_hash,
            max_frontier_tokens=self.max_frontier_tokens,
            fidelity=fidelity,
            fidelity_angle_degrees=fidelity_angle,
            compression_ratio=compression_ratio,
            root_frontier_ids=root_frontier_ids,
            drift_vector=drift_vector.tolist(),
            drift_vector_norm=drift_norm,
            drift_vector_preview=drift_preview,
            amplified_terms=amplified[: self.settings.top_k_terms],
            deemphasized_terms=deprived[: self.settings.top_k_terms],
            worst_nodes=worst_nodes,
            node_metrics=node_metrics,
            root_direct_cosine=root_direct_cos,
            root_direct_angle_degrees=root_direct_angle,
            center_embeddings=self.settings.center_embeddings,
        )

    def _compute_drift(
        self, node: TreeNode, node_map: dict[str, TreeNode]
    ) -> tuple[Vector, float]:
        span_chars = self._span_chars(node)
        summary_text = self._clean_text(node.text)
        if span_chars <= 0 or not summary_text:
            logger.debug("Skipping node %s due to empty span/text", node.id)
            return cast(Vector, self._zero.copy()), 0.0

        frontier, expanded = self._frontier_for(node, node_map)
        frontier_ids = [child.id for child in frontier]
        frontier_texts = [self._clean_text(child.text) for child in frontier]
        frontier_texts = [txt for txt in frontier_texts if txt]
        if not frontier_texts:
            logger.debug(
                "Frontier for node %s has no embeddable text; skipping", node.id
            )
            return cast(Vector, self._zero.copy()), 0.0

        baseline_text = "\n".join(frontier_texts)

        summary_vec = self._embed(summary_text)
        frontier_vec = self._embed(baseline_text)

        local_delta = summary_vec - frontier_vec
        frontier_span = sum(self._span_chars(child) for child in frontier)
        if frontier_span <= 0:
            frontier_span = span_chars

        summary_tokens = tokenizer.count_tokens(summary_text)
        frontier_tokens = tokenizer.count_tokens(baseline_text)

        record = _NodeComputation(
            node_id=node.id,
            span_chars=span_chars,
            frontier_ids=frontier_ids,
            summary_tokens=summary_tokens,
            frontier_tokens=frontier_tokens,
            summary_vec=summary_vec,
            frontier_vec=frontier_vec,
            local_delta=local_delta,
            expanded=expanded,
        )
        self._node_records[node.id] = record

        if not expanded:
            return local_delta, float(frontier_span)

        total_weight = float(frontier_span)
        weighted_delta = local_delta * total_weight

        for child in frontier:
            child_delta, child_weight = self._compute_drift(child, node_map)
            if child_weight <= 0:
                continue
            weighted_delta = weighted_delta + child_delta * child_weight
            total_weight += child_weight

        if total_weight <= 0:
            return cast(Vector, self._zero.copy()), 0.0

        return weighted_delta / total_weight, total_weight

    def _frontier_for(
        self, node: TreeNode, node_map: dict[str, TreeNode]
    ) -> tuple[list[TreeNode], bool]:
        layer = [node]
        expanded = False

        while True:
            next_layer: list[TreeNode] = []
            for candidate in layer:
                left_id = getattr(candidate, "left_child_id", None)
                right_id = getattr(candidate, "right_child_id", None)
                if left_id and left_id in node_map:
                    next_layer.append(node_map[left_id])
                if right_id and right_id in node_map:
                    next_layer.append(node_map[right_id])

            if not next_layer:
                break

            token_total = sum(self._repr_token_count(child) for child in next_layer)
            if token_total <= 0 or token_total > self.max_frontier_tokens:
                break

            layer = next_layer
            expanded = True

        return layer, expanded

    def _repr_token_count(self, node: TreeNode) -> int:
        text = self._clean_text(node.text)
        return tokenizer.count_tokens(text) if text else 0

    def _embed(self, text: str) -> Vector:
        cached = self._embedding_cache.get(text)
        if cached is not None:
            return cached

        vector = np.asarray(
            self.embedding_service.get_query_embedding(text, self._doc_id),
            dtype=np.float64,
        )
        if vector.shape[0] != self.vector_dim:
            raise ValueError(
                f"Embedding dimension mismatch ({vector.shape[0]} vs expected {self.vector_dim})"
            )
        self._embedding_cache[text] = vector
        if self.settings.center_embeddings:
            self._vector_bank.append(vector)
        return vector

    @staticmethod
    def _span_chars(node: TreeNode) -> int:
        start = int(getattr(node, "span_start", 0) or 0)
        end = int(getattr(node, "span_end", start) or start)
        return max(0, end - start)

    @staticmethod
    def _clean_text(text: str | None) -> str:
        return (text or "").strip()

    def _build_node_metrics(self, center_mean: Vector | None) -> list[NodeDriftMetric]:
        metrics: list[NodeDriftMetric] = []

        def adjusted(vec: Vector) -> Vector:
            if center_mean is None:
                return vec
            return vec - center_mean

        for record in self._node_records.values():
            summary_vec = adjusted(record.summary_vec)
            frontier_vec = adjusted(record.frontier_vec)
            cosine, angle = self._cosine_pair(summary_vec, frontier_vec)
            compression = self._compression_ratio(
                record.summary_vec, record.frontier_vec
            )
            metrics.append(
                NodeDriftMetric(
                    node_id=record.node_id,
                    span_chars=record.span_chars,
                    frontier_node_ids=record.frontier_ids,
                    summary_tokens=record.summary_tokens,
                    frontier_tokens=record.frontier_tokens,
                    local_cosine=cosine,
                    local_angle_degrees=angle,
                    compression_ratio=compression,
                )
            )

        metrics.sort(
            key=lambda m: (
                m.local_angle_degrees if m.local_angle_degrees is not None else -1.0
            ),
            reverse=True,
        )
        return metrics

    def _cosine_pair(
        self,
        vec_a: Vector,
        vec_b: Vector,
        center_mean: Vector | None = None,
    ) -> tuple[float | None, float | None]:
        if center_mean is not None:
            vec_a = vec_a - center_mean
            vec_b = vec_b - center_mean
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return None, None
        cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        cosine = max(-1.0, min(1.0, cosine))
        angle = math.degrees(math.acos(cosine))
        return cosine, angle

    @staticmethod
    def _compression_ratio(vec_a: Vector, vec_b: Vector) -> float | None:
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_b == 0.0:
            return None
        return norm_a / norm_b if norm_b else None

    def _build_vocabulary(self, nodes: Sequence[TreeNode]) -> list[_VocabularyTerm]:
        candidates: dict[str, tuple[str, int]] = {}

        def register(term: str) -> None:
            key = term.lower()
            if len(key) < self.settings.min_term_length:
                return
            display, count = candidates.get(key, (term, 0))
            candidates[key] = (display, count + 1)

        for node in nodes:
            text = self._clean_text(node.text)
            if not text:
                continue
            tokens = WORD_PATTERN.findall(text)
            if not tokens:
                continue
            for idx in range(len(tokens)):
                register(tokens[idx])
                if self.settings.max_vocab_ngram >= 2 and idx + 1 < len(tokens):
                    register(f"{tokens[idx]} {tokens[idx + 1]}")
                if self.settings.max_vocab_ngram >= 3 and idx + 2 < len(tokens):
                    register(f"{tokens[idx]} {tokens[idx + 1]} {tokens[idx + 2]}")

        if not candidates:
            self._vocabulary = []
            return []

        sorted_terms = sorted(
            candidates.items(),
            key=lambda item: (item[1][1], len(item[1][0])),
            reverse=True,
        )
        limited = sorted_terms[: self.settings.max_vocab_terms]
        total_terms = sum(count for _, (_, count) in limited) or 1

        vocab: list[_VocabularyTerm] = []
        for _, (display, count) in limited:
            embedding = self._embed(display)
            norm = float(np.linalg.norm(embedding))
            if norm == 0.0:
                continue
            idf = math.log(1.0 + (total_terms / max(1.0, count)))
            vocab.append(
                _VocabularyTerm(
                    text=display,
                    count=count,
                    vector=embedding,
                    norm=norm,
                    idf=idf,
                )
            )

        self._vocabulary = vocab
        return vocab

    def _project_terms(
        self, delta: Vector, vocab_terms: Sequence[_VocabularyTerm]
    ) -> tuple[list[DriftTerm], list[DriftTerm]]:
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm == 0.0 or not vocab_terms:
            return [], []

        unit = delta / delta_norm
        amplified: list[DriftTerm] = []
        diminished: list[DriftTerm] = []
        for term in vocab_terms:
            term_unit = term.vector / term.norm
            projection = float(np.dot(unit, term_unit))
            score = projection * term.idf
            if score > 0:
                amplified.append(DriftTerm(text=term.text, score=score))
            elif score < 0:
                diminished.append(DriftTerm(text=term.text, score=abs(score)))

        amplified.sort(key=lambda term: term.score, reverse=True)
        diminished.sort(key=lambda term: term.score, reverse=True)
        return amplified, diminished

    def _compute_config_hash(self) -> str:
        payload = {
            "embedding_model": self.embedding_model,
            "max_frontier_tokens": self.max_frontier_tokens,
            "top_k_terms": self.settings.top_k_terms,
            "max_frontier_report": self.settings.max_frontier_report,
            "max_vocab_terms": self.settings.max_vocab_terms,
            "center_embeddings": self.settings.center_embeddings,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]
