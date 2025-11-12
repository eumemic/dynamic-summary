"""Frontier-aware semantic drift analyzer."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import statistics
from collections import Counter, OrderedDict
from collections.abc import Collection, Mapping, Sequence
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
ENTITY_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
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
    max_vocab_terms: int = 1024
    center_embeddings: bool = True
    max_vocab_ngram: int = 3
    min_term_length: int = 3
    max_batch_items: int = 16
    max_batch_tokens: int = 6000
    leaf_baseline_report: int = 0
    frequency_correction: bool = True
    frequency_shift_clip: float = 0.05
    freq_calibration_pairs: int = 8


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
    residual_cosine: float | None = None
    residual_angle_degrees: float | None = None
    explained_ratio: float | None = None
    frontier_leaf_ratio: float | None = None
    js_divergence: float | None = None
    term_outlier_score: float | None = None
    top_lift_terms: list[str] = field(default_factory=list)
    entity_added: int = 0
    entity_dropped: int = 0
    entity_moved: int = 0


@dataclass
class DriftTerm:
    """Nearest-neighbour term for delta interpretation."""

    text: str
    score: float


@dataclass
class DriftAnalysisResult:
    """Complete result of a drift analysis run."""

    document_id: str
    root_node_id: str
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
    fidelity_raw: float | None = None
    fidelity_raw_angle_degrees: float | None = None
    compression_ratio_raw: float | None = None
    root_direct_cosine: float | None = None
    root_direct_angle_degrees: float | None = None
    root_direct_angle_uncentered: float | None = None
    root_adjusted_cosine: float | None = None
    root_adjusted_angle_degrees: float | None = None
    fidelity_uncentered: float | None = None
    fidelity_uncentered_angle_degrees: float | None = None
    center_embeddings: bool = False
    alignment_coefficient: float | None = None
    leaf_baseline: list[dict[str, object]] = field(default_factory=list)
    frequency_correction: bool = True
    frequency_lambda: float | None = None
    root_frontier_leaf_ratio: float | None = None
    root_leaf_angle_degrees: float | None = None
    root_leaf_span_chars: int | None = None
    lambda_sweep: list[dict[str, float]] = field(default_factory=list)
    vocab_sweep: list[dict[str, float]] = field(default_factory=list)
    root_js_divergence: float | None = None
    root_lift_terms: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """JSON-friendly view."""

        return {
            "document_id": self.document_id,
            "root_node_id": self.root_node_id,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "config_hash": self.config_hash,
            "max_frontier_tokens": self.max_frontier_tokens,
            "center_embeddings": self.center_embeddings,
            "fidelity": self.fidelity,
            "fidelity_angle_degrees": self.fidelity_angle_degrees,
            "fidelity_raw": self.fidelity_raw,
            "fidelity_raw_angle_degrees": self.fidelity_raw_angle_degrees,
            "compression_ratio": self.compression_ratio,
            "compression_ratio_raw": self.compression_ratio_raw,
            "root_direct_cosine": self.root_direct_cosine,
            "root_direct_angle_degrees": self.root_direct_angle_degrees,
            "root_direct_angle_uncentered": self.root_direct_angle_uncentered,
            "root_adjusted_cosine": self.root_adjusted_cosine,
            "root_adjusted_angle_degrees": self.root_adjusted_angle_degrees,
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
            "leaf_baseline": self.leaf_baseline,
            "frequency_correction": self.frequency_correction,
            "frequency_lambda": self.frequency_lambda,
            "root_frontier_leaf_ratio": self.root_frontier_leaf_ratio,
            "root_leaf_angle_degrees": self.root_leaf_angle_degrees,
            "root_leaf_span_chars": self.root_leaf_span_chars,
            "lambda_sweep": self.lambda_sweep,
            "vocab_sweep": self.vocab_sweep,
            "root_js_divergence": self.root_js_divergence,
            "root_lift_terms": self.root_lift_terms,
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
    raw_delta: Vector
    freq_delta: Vector
    residual_delta: Vector
    frontier_span_chars: int
    frontier_leaf_chars: int
    summary_freq: dict[str, float]
    frontier_freq: dict[str, float]
    summary_entities: dict[str, int]
    frontier_entities: dict[str, int]
    expanded: bool


@dataclass
class _VocabularyTerm:
    text: str
    count: int
    vector: Vector
    norm: float
    idf: float
    token_weight: int


@dataclass
class _LiftInfo:
    text: str
    score: float
    direction: str
    diff: float


@dataclass
class _CalibrationSample:
    node_id: str
    summary_vec: Vector
    frontier_vec: Vector
    freq_delta: Vector
    leaf_vec: Vector | None


@dataclass
class NodePlan:
    """Precomputed summary/frontier metadata for each node."""

    node_id: str
    span_chars: int
    summary_text: str
    frontier_text: str
    frontier_ids: list[str]
    frontier_span_chars: int
    frontier_leaf_chars: int
    expanded: bool


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
        self._vocab_lookup: dict[str, _VocabularyTerm] = {}
        self._term_frequency_cache: dict[str, dict[str, float]] = {}
        self._leaf_vector_cache: dict[str, Vector] = {}
        self._lambda_scale: float = 0.0
        self._doc_id = self.document_store.document_id or ""
        self.vector_index = vector_index

    def analyze(self) -> DriftAnalysisResult:
        """Run analysis for the document attached to this store."""

        root = self.document_store.tree.get_root()
        if root is None:
            raise ValueError("Document has no root node; cannot analyze drift")

        self._doc_id = root.document_id or self.document_store.document_id or ""
        self._lambda_scale = 0.0
        self._leaf_vector_cache = {}
        self._term_frequency_cache = {}

        doc_model = self.document_store.get_embedding_model()
        if doc_model and doc_model != self.embedding_model:
            raise ValueError(
                f"Document stored embeddings with '{doc_model}' but analyzer was configured for '{self.embedding_model}'"
            )

        node_map = {node.id: node for node in self.document_store.nodes.get_all()}
        if root.id not in node_map:
            node_map[root.id] = root

        plan, summary_texts, frontier_texts = self._build_plan(root, node_map)
        leaf_nodes = [
            node for node in node_map.values() if not self._has_children(node)
        ]
        vocab_candidates = self._collect_vocab_candidates(leaf_nodes)
        vocab_texts = list(OrderedDict((text, None) for text, _ in vocab_candidates))

        self._ensure_embeddings(summary_texts, frontier_texts, vocab_texts)
        vocab_terms = self._build_vocabulary(vocab_candidates)
        self._materialize_records(plan)

        raw_root_delta, _ = self._compute_plan_drift(root.id, plan, lambda_value=None)

        calibration_samples: list[_CalibrationSample] = []
        if self.settings.frequency_correction and vocab_terms:
            calibration_samples = self._build_calibration_samples(
                plan, node_map, self.settings.freq_calibration_pairs
            )
            self._lambda_scale = self._calibrate_frequency_scale(calibration_samples)
            logger.info(
                "Frequency correction lambda=%.3f (n=%d)",
                self._lambda_scale,
                len(calibration_samples),
            )
        else:
            self._lambda_scale = 0.0

        self._apply_frequency_correction()

        lambda_for_run = (
            self._lambda_scale if self.settings.frequency_correction else None
        )
        root_delta, _ = self._compute_plan_drift(
            root.id, plan, lambda_value=lambda_for_run
        )

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
        estimated_full_raw = root_vec - raw_root_delta

        fidelity, fidelity_angle = self._cosine_pair(
            root_vec, estimated_full, center_mean
        )
        fidelity_raw, fidelity_raw_angle = self._cosine_pair(
            root_vec, estimated_full_raw, center_mean
        )
        fidelity_uncentered, fidelity_uncentered_angle = self._cosine_pair(
            root_vec, estimated_full, None
        )
        compression_ratio = self._compression_ratio(root_vec, estimated_full)
        compression_ratio_raw = self._compression_ratio(root_vec, estimated_full_raw)
        root_direct_cos, root_direct_angle = self._cosine_pair(
            root_vec, frontiers_vec, center_mean
        )
        _, root_direct_angle_uncentered = self._cosine_pair(
            root_vec, frontiers_vec, None
        )
        root_adjusted_cos = None
        root_adjusted_angle = None
        if self.settings.frequency_correction and self._lambda_scale != 0.0:
            adjusted_frontier = frontiers_vec + (
                self._lambda_scale * root_record.freq_delta
            )
            root_adjusted_cos, root_adjusted_angle = self._cosine_pair(
                root_vec, adjusted_frontier, center_mean
            )

        node_metrics = self._build_node_metrics(plan, center_mean)
        worst_nodes = node_metrics[: self.settings.max_frontier_report]

        amplified, deprived = self._project_terms(drift_vector, self._vocabulary)
        alignment = self._compute_alignment_coefficient()
        leaf_baseline, root_leaf_entry = self._build_leaf_baseline_report(
            node_metrics, plan, node_map, root.id
        )

        root_leaf_angle = None
        root_leaf_span = None
        if root_leaf_entry is not None:
            root_leaf_angle = cast(
                float | None, root_leaf_entry.get("leaf_angle_degrees")
            )
            root_leaf_span = cast(int | None, root_leaf_entry.get("span_chars"))

        root_plan_entry = plan[root.id]
        root_leaf_ratio = self._coverage_ratio(
            root_plan_entry.frontier_span_chars, root_plan_entry.frontier_leaf_chars
        )
        lambda_sweep = self._lambda_sweep(root.id, plan, root_vec, center_mean)
        vocab_sweep = self._vocab_sweep(root_plan_entry, raw_root_delta)
        root_metric_info = next(
            (metric for metric in node_metrics if metric.node_id == root.id), None
        )
        root_js = root_metric_info.js_divergence if root_metric_info else None
        root_lifts = root_metric_info.top_lift_terms if root_metric_info else []

        config_hash = self._compute_config_hash()

        return DriftAnalysisResult(
            document_id=self._doc_id,
            root_node_id=root.id,
            embedding_model=self.embedding_model,
            embedding_dim=self.vector_dim,
            config_hash=config_hash,
            max_frontier_tokens=self.max_frontier_tokens,
            fidelity=fidelity,
            fidelity_angle_degrees=fidelity_angle,
            fidelity_raw=fidelity_raw,
            fidelity_raw_angle_degrees=fidelity_raw_angle,
            compression_ratio=compression_ratio,
            compression_ratio_raw=compression_ratio_raw,
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
            root_adjusted_cosine=root_adjusted_cos,
            root_adjusted_angle_degrees=root_adjusted_angle,
            fidelity_uncentered=fidelity_uncentered,
            fidelity_uncentered_angle_degrees=fidelity_uncentered_angle,
            center_embeddings=self.settings.center_embeddings,
            alignment_coefficient=alignment,
            leaf_baseline=leaf_baseline,
            frequency_correction=self.settings.frequency_correction,
            frequency_lambda=(
                self._lambda_scale if self.settings.frequency_correction else None
            ),
            root_frontier_leaf_ratio=root_leaf_ratio,
            root_leaf_angle_degrees=root_leaf_angle,
            root_leaf_span_chars=root_leaf_span,
            lambda_sweep=lambda_sweep,
            vocab_sweep=vocab_sweep,
            root_js_divergence=root_js,
            root_lift_terms=root_lifts,
        )

    def _compute_plan_drift(
        self, node_id: str, plan: dict[str, NodePlan], *, lambda_value: float | None
    ) -> tuple[Vector, float]:
        plan_entry = plan.get(node_id)
        record = self._node_records.get(node_id)
        if plan_entry is None or record is None:
            return cast(Vector, self._zero.copy()), 0.0

        local_delta = self._delta_for_lambda(record, lambda_value)
        frontier_span = float(plan_entry.frontier_span_chars)

        if not plan_entry.expanded:
            return local_delta, frontier_span

        total_weight = frontier_span
        weighted_delta = local_delta * total_weight

        for child_id in plan_entry.frontier_ids:
            child_delta, child_weight = self._compute_plan_drift(
                child_id, plan, lambda_value=lambda_value
            )
            if child_weight <= 0:
                continue
            weighted_delta = weighted_delta + child_delta * child_weight
            total_weight += child_weight

        if total_weight <= 0:
            return cast(Vector, self._zero.copy()), 0.0

        return weighted_delta / total_weight, total_weight

    def _delta_for_lambda(
        self, record: _NodeComputation, lambda_value: float | None
    ) -> Vector:
        if lambda_value is None:
            return record.raw_delta
        if lambda_value == self._lambda_scale and self.settings.frequency_correction:
            return record.residual_delta
        return record.raw_delta - (lambda_value * record.freq_delta)

    @staticmethod
    def _coverage_ratio(span_chars: int, leaf_chars: int) -> float | None:
        if span_chars <= 0:
            return None
        ratio = leaf_chars / span_chars
        return max(0.0, min(1.0, ratio))

    @staticmethod
    def _has_children(node: TreeNode) -> bool:
        return bool(
            getattr(node, "left_child_id", None)
            or getattr(node, "right_child_id", None)
        )

    def _iter_terms_with_display(self, text: str) -> list[tuple[str, str]]:
        cleaned = self._clean_text(text)
        if not cleaned:
            return []
        raw_tokens = WORD_PATTERN.findall(cleaned)
        if not raw_tokens:
            return []

        normalized = []
        for token in raw_tokens:
            lowered = token.lower()
            if len(lowered) < self.settings.min_term_length:
                continue
            normalized.append((token, lowered))

        if not normalized:
            return []

        terms: list[tuple[str, str]] = []
        limit = len(normalized)

        def all_stop(words: Sequence[str]) -> bool:
            return all(word in STOPWORDS for word in words)

        for idx, (display, key) in enumerate(normalized):
            if key not in STOPWORDS:
                terms.append((display, key))

            if self.settings.max_vocab_ngram >= 2 and idx + 1 < limit:
                next_pair = normalized[idx : idx + 2]
                pair_keys = [word[1] for word in next_pair]
                if all_stop(pair_keys):
                    continue
                pair_display = f"{next_pair[0][0]} {next_pair[1][0]}"
                pair_key = f"{pair_keys[0]} {pair_keys[1]}"
                terms.append((pair_display, pair_key))

            if self.settings.max_vocab_ngram >= 3 and idx + 2 < limit:
                trio = normalized[idx : idx + 3]
                trio_keys = [word[1] for word in trio]
                if all_stop(trio_keys):
                    continue
                trio_display = " ".join(word[0] for word in trio)
                trio_key = " ".join(trio_keys)
                terms.append((trio_display, trio_key))

        return terms

    def _term_keys(self, text: str) -> list[str]:
        return [key for _, key in self._iter_terms_with_display(text)]

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
            frontier_leaf_chars = sum(
                self._span_chars(child)
                for child in frontier
                if not self._has_children(child)
            )

            plan[current.id] = NodePlan(
                node_id=current.id,
                span_chars=span_chars,
                summary_text=summary_text,
                frontier_text=baseline_text,
                frontier_ids=[child.id for child in frontier],
                frontier_span_chars=frontier_span,
                frontier_leaf_chars=frontier_leaf_chars,
                expanded=expanded,
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

        def register(display: str, key: str) -> None:
            if not key:
                return
            display_value, count = candidates.get(key, (display, 0))
            candidates[key] = (display_value, count + 1)

        for node in nodes:
            text = self._clean_text(node.text)
            if not text:
                continue
            for display, key in self._iter_terms_with_display(text):
                register(display, key)

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
            summary_freq = dict(self._term_frequencies(entry.summary_text))
            frontier_freq = dict(self._term_frequencies(entry.frontier_text))
            freq_delta = self._frequency_delta_from_freqs(summary_freq, frontier_freq)
            summary_entities = self._extract_entities(entry.summary_text)
            frontier_entities = self._extract_entities(entry.frontier_text)
            raw_delta = summary_vec - frontier_vec
            record = _NodeComputation(
                node_id=entry.node_id,
                span_chars=entry.span_chars,
                frontier_ids=entry.frontier_ids,
                summary_tokens=summary_tokens,
                frontier_tokens=frontier_tokens,
                summary_vec=summary_vec,
                frontier_vec=frontier_vec,
                raw_delta=raw_delta,
                freq_delta=freq_delta,
                residual_delta=raw_delta.copy(),
                frontier_span_chars=entry.frontier_span_chars,
                frontier_leaf_chars=entry.frontier_leaf_chars,
                summary_freq=summary_freq,
                frontier_freq=frontier_freq,
                summary_entities=summary_entities,
                frontier_entities=frontier_entities,
                expanded=entry.expanded,
            )
            self._node_records[entry.node_id] = record

    def _frequency_delta(
        self,
        summary_text: str,
        frontier_text: str,
        allowed_terms: Collection[str] | None = None,
    ) -> Vector:
        if not self._vocab_lookup:
            return cast(Vector, np.zeros(self.vector_dim, dtype=np.float64))

        summary_freq = self._term_frequencies(summary_text)
        frontier_freq = self._term_frequencies(frontier_text)
        return self._frequency_delta_from_freqs(
            summary_freq, frontier_freq, allowed_terms
        )

    def _frequency_delta_from_freqs(
        self,
        summary_freq: Mapping[str, float],
        frontier_freq: Mapping[str, float],
        allowed_terms: Collection[str] | None = None,
    ) -> Vector:
        if not summary_freq and not frontier_freq:
            return cast(Vector, np.zeros(self.vector_dim, dtype=np.float64))
        clip = abs(self.settings.frequency_shift_clip)
        delta_vec = np.zeros(self.vector_dim, dtype=np.float64)
        keys = set(summary_freq.keys()) | set(frontier_freq.keys())
        for key in keys:
            vocab_term = self._vocab_lookup.get(key)
            if not vocab_term:
                continue
            if allowed_terms is not None and key not in allowed_terms:
                continue
            diff = summary_freq.get(key, 0.0) - frontier_freq.get(key, 0.0)
            if diff == 0.0:
                continue
            clipped = min(max(diff, -clip), clip)
            coeff = vocab_term.idf * clipped
            unit = vocab_term.vector / vocab_term.norm
            delta_vec = delta_vec + (coeff * unit)

        return cast(Vector, delta_vec)

    def _term_frequencies(self, text: str) -> dict[str, float]:
        cached = self._term_frequency_cache.get(text)
        if cached is not None:
            return cached

        keys = self._term_keys(text)
        if not keys:
            self._term_frequency_cache[text] = {}
            return {}

        counts = Counter(keys)
        total_tokens = max(1, tokenizer.count_tokens(text))
        freq: dict[str, float] = {}
        for key, count in counts.items():
            vocab = self._vocab_lookup.get(key)
            if not vocab:
                continue
            weight = (count * max(1, vocab.token_weight)) / total_tokens
            freq[key] = weight

        self._term_frequency_cache[text] = freq
        return freq

    @staticmethod
    def _extract_entities(text: str) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for match in ENTITY_PATTERN.finditer(text or ""):
            entity = match.group(0).strip()
            if not entity:
                continue
            if entity.lower() in STOPWORDS:
                continue
            counts[entity] += 1
        return dict(counts)

    def _apply_frequency_correction(self) -> None:
        if not self._node_records:
            return

        if not self.settings.frequency_correction or not self._vocab_lookup:
            for record in self._node_records.values():
                record.residual_delta = record.raw_delta.copy()
            return

        for record in self._node_records.values():
            correction = self._lambda_scale * record.freq_delta
            record.residual_delta = record.raw_delta - correction

    def _build_calibration_samples(
        self,
        plan: dict[str, NodePlan],
        node_map: dict[str, TreeNode],
        limit: int,
    ) -> list[_CalibrationSample]:
        if limit <= 0:
            return []

        ordered = sorted(
            plan.values(), key=lambda entry: entry.span_chars, reverse=True
        )
        samples: list[_CalibrationSample] = []
        for entry in ordered:
            if len(samples) >= limit:
                break
            record = self._node_records.get(entry.node_id)
            if not record:
                continue
            if float(np.linalg.norm(record.freq_delta)) == 0.0:
                continue
            leaf_text = self._gather_leaf_text(entry.node_id, node_map)
            if not leaf_text:
                continue
            if tokenizer.count_tokens(leaf_text) > self.max_frontier_tokens:
                continue
            leaf_vec = self._leaf_vector_cache.get(entry.node_id)
            if leaf_vec is None:
                leaf_vec = self._embed(leaf_text)
                self._leaf_vector_cache[entry.node_id] = leaf_vec
            samples.append(
                _CalibrationSample(
                    node_id=entry.node_id,
                    summary_vec=record.summary_vec,
                    frontier_vec=record.frontier_vec,
                    freq_delta=record.freq_delta,
                    leaf_vec=leaf_vec,
                )
            )

        return samples

    def _calibrate_frequency_scale(
        self, samples: Sequence[_CalibrationSample]
    ) -> float:
        usable = [
            sample
            for sample in samples
            if float(np.linalg.norm(sample.freq_delta)) > 0.0
        ]
        if len(usable) < 5:
            return 1.0

        lambda_candidates = np.linspace(0.0, 3.0, 31)
        best_lambda = 1.0
        best_score = math.inf

        for candidate in lambda_candidates:
            angles: list[float] = []
            for sample in usable:
                adjusted = sample.frontier_vec + (candidate * sample.freq_delta)
                _, angle = self._cosine_pair(sample.summary_vec, adjusted, None)
                if angle is None:
                    continue
                angles.append(angle)
            if not angles:
                continue
            score = statistics.median(angles)
            if score < best_score:
                best_score = score
                best_lambda = candidate

        return best_lambda

    def _lambda_sweep(
        self,
        root_id: str,
        plan: dict[str, NodePlan],
        root_vec: Vector,
        center_mean: Vector | None,
    ) -> list[dict[str, float]]:
        candidates = [0.0, 0.05, 0.1, 0.2]
        if self.settings.frequency_correction:
            candidates.append(self._lambda_scale)
        sweep: list[dict[str, float]] = []
        seen: set[float] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            delta, _ = self._compute_plan_drift(root_id, plan, lambda_value=candidate)
            estimated = root_vec - delta
            fidelity, angle = self._cosine_pair(root_vec, estimated, center_mean)
            if fidelity is None or angle is None:
                continue
            sweep.append(
                {
                    "lambda": candidate,
                    "fidelity": fidelity,
                    "angle_degrees": angle,
                }
            )
        sweep.sort(key=lambda entry: entry["lambda"])
        return sweep

    def _vocab_sweep(
        self, root_plan: NodePlan, raw_root_delta: Vector
    ) -> list[dict[str, float]]:
        if not self._vocabulary:
            return []
        raw_norm = float(np.linalg.norm(raw_root_delta))
        if raw_norm == 0.0:
            return []
        sizes = [250, 500, 1000]
        total_vocab = len(self._vocabulary)
        sweep: list[dict[str, float]] = []
        for size in sizes:
            actual = min(size, total_vocab)
            allowed_terms = {term.text.lower() for term in self._vocabulary[:actual]}
            if not allowed_terms:
                ratio = 0.0
            else:
                freq_delta = self._frequency_delta(
                    root_plan.summary_text,
                    root_plan.frontier_text,
                    allowed_terms=allowed_terms,
                )
                ratio = float(np.linalg.norm(freq_delta) / raw_norm)
            sweep.append({"size": actual, "explained_ratio": ratio})
        return sweep

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

    def _build_node_metrics(
        self, plan: dict[str, NodePlan], center_mean: Vector | None
    ) -> list[NodeDriftMetric]:
        metrics: list[NodeDriftMetric] = []

        lambda_scale = self._lambda_scale if self.settings.frequency_correction else 0.0

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
            residual_cos = cosine
            residual_angle = angle
            explained_ratio: float | None = None
            if lambda_scale != 0.0:
                adjusted_frontier = adjusted(
                    record.frontier_vec + (lambda_scale * record.freq_delta)
                )
                residual_cos, residual_angle = self._cosine_pair(
                    summary_vec, adjusted_frontier
                )
                raw_norm = float(np.linalg.norm(record.raw_delta))
                explained_norm = float(np.linalg.norm(lambda_scale * record.freq_delta))
                if raw_norm > 0:
                    explained_ratio = explained_norm / raw_norm
            plan_entry = plan.get(record.node_id)
            leaf_ratio = None
            if plan_entry is not None:
                leaf_ratio = self._coverage_ratio(
                    plan_entry.frontier_span_chars, plan_entry.frontier_leaf_chars
                )
            js_divergence = self._js_divergence(
                record.summary_freq, record.frontier_freq
            )
            term_lifts = self._term_lifts(record.summary_freq, record.frontier_freq)
            top_terms = [
                f"{entry.direction} {entry.text} ({entry.score:.3f})"
                for entry in term_lifts[: self.settings.top_k_terms]
            ]
            term_outlier_score = term_lifts[0].score if term_lifts else None
            entity_added, entity_dropped, entity_moved = self._entity_delta(
                record.summary_entities, record.frontier_entities
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
                    residual_cosine=residual_cos,
                    residual_angle_degrees=residual_angle,
                    compression_ratio=compression,
                    explained_ratio=explained_ratio,
                    frontier_leaf_ratio=leaf_ratio,
                    js_divergence=js_divergence,
                    term_outlier_score=term_outlier_score,
                    top_lift_terms=top_terms,
                    entity_added=entity_added,
                    entity_dropped=entity_dropped,
                    entity_moved=entity_moved,
                )
            )

        metrics.sort(
            key=lambda m: (
                m.residual_angle_degrees
                if m.residual_angle_degrees is not None
                else (
                    m.local_angle_degrees if m.local_angle_degrees is not None else -1.0
                )
            ),
            reverse=True,
        )
        return metrics

    @staticmethod
    def _normalize_distribution(dist: Mapping[str, float]) -> dict[str, float]:
        total = sum(dist.values())
        if total <= 0:
            return {}
        return {key: value / total for key, value in dist.items() if value > 0.0}

    def _js_divergence(
        self, dist_a: Mapping[str, float], dist_b: Mapping[str, float]
    ) -> float | None:
        norm_a = self._normalize_distribution(dist_a)
        norm_b = self._normalize_distribution(dist_b)
        support = set(norm_a.keys()) | set(norm_b.keys())
        if not support:
            return None
        mean: dict[str, float] = {}
        for key in support:
            mean[key] = 0.5 * (norm_a.get(key, 0.0) + norm_b.get(key, 0.0))

        def kl(p: Mapping[str, float], m: Mapping[str, float]) -> float:
            value = 0.0
            for key, prob in p.items():
                if prob == 0.0:
                    continue
                mid = m.get(key, 0.0)
                if mid == 0.0:
                    continue
                value += prob * math.log(prob / mid, 2)
            return value

        divergence = 0.5 * (kl(norm_a, mean) + kl(norm_b, mean))
        return divergence

    def _term_lifts(
        self,
        summary_freq: Mapping[str, float],
        frontier_freq: Mapping[str, float],
    ) -> list[_LiftInfo]:
        lifts: list[_LiftInfo] = []
        keys = set(summary_freq.keys()) | set(frontier_freq.keys())
        for key in keys:
            vocab_term = self._vocab_lookup.get(key)
            if not vocab_term:
                continue
            diff = summary_freq.get(key, 0.0) - frontier_freq.get(key, 0.0)
            if diff == 0.0:
                continue
            score = abs(diff) * vocab_term.idf
            direction = "↑" if diff > 0 else "↓"
            lifts.append(
                _LiftInfo(
                    text=vocab_term.text, score=score, direction=direction, diff=diff
                )
            )
        lifts.sort(key=lambda entry: entry.score, reverse=True)
        return lifts

    @staticmethod
    def _entity_delta(
        summary_entities: Mapping[str, int],
        frontier_entities: Mapping[str, int],
    ) -> tuple[int, int, int]:
        added = sum(
            count
            for entity, count in summary_entities.items()
            if entity not in frontier_entities
        )
        dropped = sum(
            count
            for entity, count in frontier_entities.items()
            if entity not in summary_entities
        )
        moved = 0
        if summary_entities and frontier_entities:
            summary_rank = DriftAnalyzer._rank_entities(summary_entities)
            frontier_rank = DriftAnalyzer._rank_entities(frontier_entities)
            for entity in set(summary_rank.keys()) & set(frontier_rank.keys()):
                if summary_rank[entity] != frontier_rank[entity]:
                    moved += 1
        return added, dropped, moved

    @staticmethod
    def _rank_entities(counts: Mapping[str, int]) -> dict[str, int]:
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return {entity: idx for idx, (entity, _) in enumerate(ordered)}

    def _compute_alignment_coefficient(self) -> float | None:
        accum: list[Vector] = []
        for record in self._node_records.values():
            norm = float(np.linalg.norm(record.residual_delta))
            if norm == 0.0:
                continue
            accum.append(record.residual_delta / norm)
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
            self._vocab_lookup = {}
            self._term_frequency_cache.clear()
            return []

        total_terms = sum(count for _, count in candidates) or 1

        vocab: list[_VocabularyTerm] = []
        for display, count in candidates:
            embedding = self._embed(display)
            norm = float(np.linalg.norm(embedding))
            if norm == 0.0:
                continue
            idf = math.log(1.0 + (total_terms / max(1.0, count)))
            token_weight = max(1, tokenizer.count_tokens(display))
            vocab.append(
                _VocabularyTerm(
                    text=display,
                    count=count,
                    vector=embedding,
                    norm=norm,
                    idf=idf,
                    token_weight=token_weight,
                )
            )

        self._vocabulary = vocab
        self._vocab_lookup = {term.text.lower(): term for term in vocab}
        self._term_frequency_cache.clear()
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

    def _build_leaf_baseline_report(
        self,
        metrics: Sequence[NodeDriftMetric],
        plan: dict[str, NodePlan],
        node_map: dict[str, TreeNode],
        root_id: str,
    ) -> tuple[list[dict[str, object]], dict[str, object] | None]:
        count = min(self.settings.leaf_baseline_report, len(metrics))
        report: list[dict[str, object]] = []
        if count > 0:
            for metric in metrics[:count]:
                entry = self._build_leaf_baseline_entry(metric.node_id, plan, node_map)
                if entry:
                    report.append(entry)
        root_entry = self._build_leaf_baseline_entry(root_id, plan, node_map)
        return report, root_entry

    def _build_leaf_baseline_entry(
        self, node_id: str, plan: dict[str, NodePlan], node_map: dict[str, TreeNode]
    ) -> dict[str, object] | None:
        plan_entry = plan.get(node_id)
        record = self._node_records.get(node_id)
        if not plan_entry or not record:
            return None
        leaf_text = self._gather_leaf_text(node_id, node_map)
        if not leaf_text:
            return None
        if tokenizer.count_tokens(leaf_text) > self.max_frontier_tokens:
            return None
        leaf_vec = self._leaf_vector_cache.get(node_id)
        if leaf_vec is None:
            leaf_vec = self._embed(leaf_text)
            self._leaf_vector_cache[node_id] = leaf_vec
        cosine, angle = self._cosine_pair(record.summary_vec, leaf_vec, None)
        compression = self._compression_ratio(record.summary_vec, leaf_vec)
        return {
            "node_id": node_id,
            "span_chars": plan_entry.span_chars,
            "leaf_angle_degrees": angle,
            "leaf_cosine": cosine,
            "leaf_compression": compression,
        }

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
            "frequency_correction": self.settings.frequency_correction,
            "frequency_shift_clip": self.settings.frequency_shift_clip,
            "freq_calibration_pairs": self.settings.freq_calibration_pairs,
        }
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]
