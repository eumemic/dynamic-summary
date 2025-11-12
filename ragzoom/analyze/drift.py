"""Frontier-aware semantic drift analyzer."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
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
HIGH_VALENCE_TERMS = {
    "necromancer",
    "dol guldur",
    "sauron",
    "smaug",
    "arkenstone",
    "thrain",
    "mirkwood",
    "lake-town",
    "durin",
}
STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "in",
    "for",
    "on",
    "a",
    "with",
    "said",
    "as",
    "at",
    "be",
    "by",
    "that",
    "had",
    "it",
    "but",
    "or",
    "while",
    "from",
    "into",
}


@dataclass
class DriftAnalyzerSettings:
    """Runtime tuning knobs for the analyzer."""

    top_k_terms: int = 5
    max_frontier_report: int = 10
    max_vocab_terms: int = 512
    center_embeddings: bool = True
    max_vocab_ngram: int = 3
    min_term_length: int = 3
    max_batch_items: int = 16
    max_batch_tokens: int = 6000
    leaf_baseline_report: int = 0


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
    root_direct_angle_uncentered: float | None = None
    fidelity_uncentered: float | None = None
    fidelity_uncentered_angle_degrees: float | None = None
    center_embeddings: bool = False
    alignment_coefficient: float | None = None
    high_valence_nodes: list[NodeDriftMetric] = field(default_factory=list)
    leaf_baseline: list[dict[str, object]] = field(default_factory=list)

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
            "root_direct_angle_uncentered": self.root_direct_angle_uncentered,
            "fidelity_uncentered": self.fidelity_uncentered,
            "fidelity_uncentered_angle_degrees": self.fidelity_uncentered_angle_degrees,
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
            "alignment_coefficient": self.alignment_coefficient,
            "high_valence_nodes": [
                metric.__dict__ for metric in self.high_valence_nodes
            ],
            "leaf_baseline": self.leaf_baseline,
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


@dataclass
class NodePlan:
    """Precomputed summary/frontier metadata for each node."""

    node_id: str
    span_chars: int
    summary_text: str
    frontier_text: str
    frontier_ids: list[str]
    frontier_span_chars: int
    expanded: bool
    contains_high_valence: bool


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
        vector_index: VectorIndex | None = None,
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
        self.vector_index = vector_index

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

        plan, summary_texts, frontier_texts = self._build_plan(root, node_map)
        vocab_candidates = self._collect_vocab_candidates(list(node_map.values()))
        vocab_texts = [text for text, _ in vocab_candidates]

        self._ensure_embeddings(summary_texts, frontier_texts, vocab_texts)
        self._materialize_records(plan)

        root_delta, _ = self._compute_plan_drift(root.id, plan)

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
        fidelity_uncentered, fidelity_uncentered_angle = self._cosine_pair(
            root_vec, estimated_full, None
        )
        compression_ratio = self._compression_ratio(root_vec, estimated_full)
        root_direct_cos, root_direct_angle = self._cosine_pair(
            root_vec, frontiers_vec, center_mean
        )
        _, root_direct_angle_uncentered = self._cosine_pair(
            root_vec, frontiers_vec, None
        )

        node_metrics = self._build_node_metrics(center_mean)
        worst_nodes = node_metrics[: self.settings.max_frontier_report]

        vocab_terms = self._build_vocabulary(vocab_candidates)
        amplified, deprived = self._project_terms(drift_vector, vocab_terms)
        alignment = self._compute_alignment_coefficient()
        high_valence_metrics = self._collect_high_valence_metrics(plan, node_metrics)
        leaf_baseline = self._build_leaf_baseline_report(node_metrics, plan, node_map)

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
            root_direct_angle_uncentered=root_direct_angle_uncentered,
            fidelity_uncentered=fidelity_uncentered,
            fidelity_uncentered_angle_degrees=fidelity_uncentered_angle,
            center_embeddings=self.settings.center_embeddings,
            alignment_coefficient=alignment,
            high_valence_nodes=high_valence_metrics,
            leaf_baseline=leaf_baseline,
        )

    def _compute_plan_drift(
        self, node_id: str, plan: dict[str, NodePlan]
    ) -> tuple[Vector, float]:
        plan_entry = plan.get(node_id)
        record = self._node_records.get(node_id)
        if plan_entry is None or record is None:
            return cast(Vector, self._zero.copy()), 0.0

        local_delta = record.local_delta
        frontier_span = float(plan_entry.frontier_span_chars)

        if not plan_entry.expanded:
            return local_delta, frontier_span

        total_weight = frontier_span
        weighted_delta = local_delta * total_weight

        for child_id in plan_entry.frontier_ids:
            child_delta, child_weight = self._compute_plan_drift(child_id, plan)
            if child_weight <= 0:
                continue
            weighted_delta = weighted_delta + child_delta * child_weight
            total_weight += child_weight

        if total_weight <= 0:
            return cast(Vector, self._zero.copy()), 0.0

        return weighted_delta / total_weight, total_weight

    @staticmethod
    def _contains_high_valence(text: str) -> bool:
        lower = text.lower()
        return any(term in lower for term in HIGH_VALENCE_TERMS)

    def _build_plan(
        self, node: TreeNode, node_map: dict[str, TreeNode]
    ) -> tuple[dict[str, NodePlan], dict[str, str], list[str]]:
        plan: dict[str, NodePlan] = {}
        summary_texts: dict[str, str] = {}
        frontier_strings: OrderedDict[str, None] = OrderedDict()
        visited: set[str] = set()

        def visit(current: TreeNode) -> None:
            if current.id in visited:
                return
            visited.add(current.id)

            span_chars = self._span_chars(current)
            summary_text = self._clean_text(current.text)
            if span_chars <= 0 or not summary_text:
                logger.debug(
                    "Skipping node %s due to missing span/text during planning",
                    current.id,
                )
                return

            frontier, expanded = self._frontier_for(current, node_map)
            frontier_texts = [
                self._clean_text(child.text)
                for child in frontier
                if self._clean_text(child.text)
            ]
            if not frontier_texts:
                logger.debug(
                    "Frontier for node %s yielded no embeddable text during planning",
                    current.id,
                )
                return

            baseline_text = "\n".join(frontier_texts)
            frontier_span = sum(self._span_chars(child) for child in frontier)
            if frontier_span <= 0:
                frontier_span = span_chars

            contains_valence = self._contains_high_valence(
                summary_text
            ) or self._contains_high_valence(baseline_text)

            plan[current.id] = NodePlan(
                node_id=current.id,
                span_chars=span_chars,
                summary_text=summary_text,
                frontier_text=baseline_text,
                frontier_ids=[child.id for child in frontier],
                frontier_span_chars=frontier_span,
                expanded=expanded,
                contains_high_valence=contains_valence,
            )
            summary_texts[current.id] = summary_text
            frontier_strings.setdefault(baseline_text, None)

            if expanded:
                for child in frontier:
                    visit(child)

        visit(node)

        if node.id not in plan:
            raise ValueError("Root node has no analyzable frontier; aborting analysis")

        return plan, summary_texts, list(frontier_strings.keys())

    def _collect_vocab_candidates(
        self, nodes: Sequence[TreeNode]
    ) -> list[tuple[str, int]]:
        candidates: dict[str, tuple[str, int]] = {}

        def register(term: str) -> None:
            key = term.lower()
            if len(key) < self.settings.min_term_length:
                return
            if key in STOPWORDS:
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
            return []

        sorted_terms = sorted(
            candidates.items(),
            key=lambda item: (item[1][1], len(item[1][0])),
            reverse=True,
        )
        limited = sorted_terms[: self.settings.max_vocab_terms]
        return [(display, count) for _, (display, count) in limited]

    def _ensure_embeddings(
        self,
        summary_texts: dict[str, str],
        frontier_texts: list[str],
        vocab_texts: list[str],
    ) -> None:
        texts_to_embed: OrderedDict[str, None] = OrderedDict()

        stored_vectors = self._fetch_vectors_from_index(list(summary_texts.keys()))
        for node_id, vector in stored_vectors.items():
            summary_text = summary_texts.get(node_id)
            if not summary_text:
                continue
            self._cache_embedding(summary_text, vector)

        for node_id, summary_text in summary_texts.items():
            if summary_text and summary_text not in self._embedding_cache:
                texts_to_embed.setdefault(summary_text, None)

        for text in frontier_texts:
            if text and text not in self._embedding_cache:
                texts_to_embed.setdefault(text, None)

        for text in vocab_texts:
            if text and text not in self._embedding_cache:
                texts_to_embed.setdefault(text, None)

        if texts_to_embed:
            self._embed_text_batches(list(texts_to_embed.keys()))

    def _materialize_records(self, plan: dict[str, NodePlan]) -> None:
        self._node_records = {}
        for entry in plan.values():
            summary_vec = self._embed(entry.summary_text)
            frontier_vec = self._embed(entry.frontier_text)
            summary_tokens = tokenizer.count_tokens(entry.summary_text)
            frontier_tokens = tokenizer.count_tokens(entry.frontier_text)
            record = _NodeComputation(
                node_id=entry.node_id,
                span_chars=entry.span_chars,
                frontier_ids=entry.frontier_ids,
                summary_tokens=summary_tokens,
                frontier_tokens=frontier_tokens,
                summary_vec=summary_vec,
                frontier_vec=frontier_vec,
                local_delta=summary_vec - frontier_vec,
                expanded=entry.expanded,
            )
            self._node_records[entry.node_id] = record

    def _fetch_vectors_from_index(self, node_ids: Sequence[str]) -> dict[str, Vector]:
        if not self.vector_index:
            return {}
        if not node_ids:
            return {}
        try:
            vectors = self.vector_index.get_vectors(list(node_ids))
        except Exception as exc:  # pragma: no cover - backend failures logged
            logger.warning("Failed to fetch stored embeddings: %s", exc)
            return {}

        result: dict[str, Vector] = {}
        for vector in vectors:
            if vector.id not in node_ids:
                continue
            if vector.model_id and vector.model_id != self.embedding_model:
                logger.debug(
                    "Skipping vector %s due to model mismatch (%s != %s)",
                    vector.id,
                    vector.model_id,
                    self.embedding_model,
                )
                continue
            vec_array = np.asarray(vector.vec, dtype=np.float64)
            if vec_array.shape[0] != self.vector_dim:
                logger.debug(
                    "Skipping vector %s due to dimension mismatch (%s != %s)",
                    vector.id,
                    vec_array.shape[0],
                    self.vector_dim,
                )
                continue
            result[vector.id] = vec_array
        return result

    def _cache_embedding(self, text: str, vector: Vector) -> None:
        if text in self._embedding_cache:
            return
        self._embedding_cache[text] = vector
        if self.settings.center_embeddings:
            self._vector_bank.append(vector)

    def _embed_text_batches(self, texts: Sequence[str]) -> None:
        if not texts:
            return
        for batch in self._batch_texts(texts):
            if not batch:
                continue
            embedded = self.embedding_service.embed_texts(
                batch, document_id=self._doc_id
            )
            if len(embedded) != len(batch):
                raise RuntimeError(
                    f"Embedding batch returned {len(embedded)} results for {len(batch)} inputs"
                )
            for text, values in zip(batch, embedded):
                vector = np.asarray(values, dtype=np.float64)
                self._cache_embedding(text, vector)

    def _batch_texts(self, texts: Sequence[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current_batch: list[str] = []
        current_tokens = 0

        for text in texts:
            token_count = tokenizer.count_tokens(text)
            would_exceed_items = len(current_batch) >= self.settings.max_batch_items
            would_exceed_tokens = (
                current_batch
                and current_tokens + token_count > self.settings.max_batch_tokens
            )

            if current_batch and (would_exceed_items or would_exceed_tokens):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += token_count

        if current_batch:
            batches.append(current_batch)

        return batches

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

    def _compute_alignment_coefficient(self) -> float | None:
        accum: list[Vector] = []
        for record in self._node_records.values():
            norm = float(np.linalg.norm(record.local_delta))
            if norm == 0.0:
                continue
            accum.append(record.local_delta / norm)
        if not accum:
            return None
        stacked = np.vstack(accum)
        total = stacked.sum(axis=0)
        return float(np.linalg.norm(total) / len(accum))

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

    def _build_vocabulary(
        self, candidates: Sequence[tuple[str, int]]
    ) -> list[_VocabularyTerm]:
        if not candidates:
            self._vocabulary = []
            return []

        total_terms = sum(count for _, count in candidates) or 1

        vocab: list[_VocabularyTerm] = []
        for display, count in candidates:
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

    def _collect_high_valence_metrics(
        self, plan: dict[str, NodePlan], metrics: Sequence[NodeDriftMetric]
    ) -> list[NodeDriftMetric]:
        flagged: list[NodeDriftMetric] = []
        for metric in metrics:
            entry = plan.get(metric.node_id)
            if entry and entry.contains_high_valence:
                flagged.append(metric)
        return flagged[: self.settings.max_frontier_report]

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

    def _build_leaf_baseline_report(
        self,
        metrics: Sequence[NodeDriftMetric],
        plan: dict[str, NodePlan],
        node_map: dict[str, TreeNode],
    ) -> list[dict[str, object]]:
        count = min(self.settings.leaf_baseline_report, len(metrics))
        if count <= 0:
            return []

        report: list[dict[str, object]] = []
        for metric in metrics[:count]:
            plan_entry = plan.get(metric.node_id)
            record = self._node_records.get(metric.node_id)
            if not plan_entry or not record:
                continue
            leaf_text = self._gather_leaf_text(metric.node_id, node_map)
            if not leaf_text:
                continue
            leaf_vec = self._embed(leaf_text)
            cosine, angle = self._cosine_pair(record.summary_vec, leaf_vec, None)
            compression = self._compression_ratio(record.summary_vec, leaf_vec)
            report.append(
                {
                    "node_id": metric.node_id,
                    "span_chars": metric.span_chars,
                    "leaf_angle_degrees": angle,
                    "leaf_cosine": cosine,
                    "leaf_compression": compression,
                }
            )
        return report

    def _gather_leaf_text(self, node_id: str, node_map: dict[str, TreeNode]) -> str:
        node = node_map.get(node_id)
        if node is None:
            return ""

        left_id = getattr(node, "left_child_id", None)
        right_id = getattr(node, "right_child_id", None)
        if not left_id and not right_id:
            return self._clean_text(node.text)

        parts: list[str] = []
        if left_id:
            text = self._gather_leaf_text(left_id, node_map)
            if text:
                parts.append(text)
        if right_id:
            text = self._gather_leaf_text(right_id, node_map)
            if text:
                parts.append(text)
        if parts:
            return "\n".join(parts)
        return self._clean_text(node.text)

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
