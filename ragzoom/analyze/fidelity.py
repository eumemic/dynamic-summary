"""Local summarization fidelity analyzer."""

from __future__ import annotations

import math
import statistics
from collections import OrderedDict
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

Vector = NDArray[np.float64]

# Conservative per-model embedding limits (tokens). Default falls back to 8000.
EMBEDDING_CONTEXT_LIMITS: dict[str, int] = {
    "text-embedding-3-small": 8192,
    "text-embedding-3-large": 8192,
    "text-embedding-ada-002": 8191,
}
DEFAULT_EMBED_LIMIT = 8000


@dataclass
class FidelityAnalyzerSettings:
    """Runtime configuration for fidelity analysis."""

    top_k_worst: int = 10
    histogram_start: float = 0.6
    histogram_bucket_size: float = 0.05
    histogram_buckets: int = 5


@dataclass
class NodeFidelityMetric:
    """Per-merge fidelity information."""

    node_id: str
    child_ids: list[str]
    fidelity: float | None
    angle_degrees: float | None
    parent_tokens: int
    children_tokens: int
    parent_span_chars: int
    children_span_chars: int
    parent_preview: str
    children_preview: str


@dataclass
class FidelityHistogramBucket:
    """Discrete range of fidelity values."""

    start: float
    end: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {"start": self.start, "end": self.end, "count": self.count}


@dataclass
class SummarizationFidelityStats:
    """Aggregate statistics for fidelity measurements."""

    count: int
    mean: float | None
    median: float | None
    minimum: float | None
    maximum: float | None
    stddev: float | None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "stddev": self.stddev,
        }


@dataclass
class SummarizationFidelityResult:
    """Complete result of a fidelity analysis run."""

    document_id: str
    embedding_model: str
    embedding_dim: int
    metrics: list[NodeFidelityMetric]
    stats: SummarizationFidelityStats
    histogram: list[FidelityHistogramBucket] = field(default_factory=list)
    histogram_underflow: int = 0
    histogram_overflow: int = 0
    worst_nodes: list[NodeFidelityMetric] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "stats": self.stats.to_dict(),
            "metrics": [metric.__dict__ for metric in self.metrics],
            "worst_nodes": [metric.__dict__ for metric in self.worst_nodes],
            "histogram": [bucket.to_dict() for bucket in self.histogram],
            "histogram_underflow": self.histogram_underflow,
            "histogram_overflow": self.histogram_overflow,
        }


@dataclass
class _NodePlan:
    node_id: str
    child_ids: list[str]
    parent_text: str
    children_text: str
    parent_tokens: int
    children_tokens: int
    parent_span_chars: int
    children_span_chars: int


class SummarizationFidelityAnalyzer:
    """Analyzer that measures how well each merge preserves semantics."""

    def __init__(
        self,
        document_store: DocumentStore,
        embedding_service: EmbeddingService,
        *,
        embedding_model: str,
        settings: FidelityAnalyzerSettings | None = None,
        vector_dim: int | None = None,
    ) -> None:
        self.document_store = document_store
        self.embedding_service = embedding_service
        self.embedding_model = embedding_model
        self.settings = settings or FidelityAnalyzerSettings()
        model_info = ModelInfo()
        self.vector_dim = vector_dim or model_info.get_embedding_dimensions(
            embedding_model
        )
        self._embedding_cache: dict[str, Vector] = {}
        self._doc_id: str = ""
        self._max_embed_tokens = EMBEDDING_CONTEXT_LIMITS.get(
            embedding_model, DEFAULT_EMBED_LIMIT
        )

    def analyze(self) -> SummarizationFidelityResult:
        """Compute fidelity for every non-leaf merge in the document."""

        root = self.document_store.tree.get_root()
        if root is None:
            raise ValueError("Document has no root node; cannot analyze fidelity")

        self._doc_id = root.document_id or self.document_store.document_id or ""
        nodes = self.document_store.nodes.get_all()
        node_map = {node.id: node for node in nodes}
        node_map.setdefault(root.id, root)

        plans = self._build_plan(node_map)
        if not plans:
            stats = SummarizationFidelityStats(
                count=0, mean=None, median=None, minimum=None, maximum=None, stddev=None
            )
            return SummarizationFidelityResult(
                document_id=self._doc_id,
                embedding_model=self.embedding_model,
                embedding_dim=self.vector_dim,
                metrics=[],
                stats=stats,
            )

        unique_texts: OrderedDict[str, None] = OrderedDict()
        for plan in plans:
            if plan.parent_text:
                unique_texts.setdefault(plan.parent_text, None)
            if plan.children_text:
                unique_texts.setdefault(plan.children_text, None)

        self._ensure_embeddings(list(unique_texts.keys()))

        metrics: list[NodeFidelityMetric] = []
        for plan in plans:
            parent_vec = (
                self._embedding_cache.get(plan.parent_text)
                if plan.parent_text
                else None
            )
            children_vec = (
                self._embedding_cache.get(plan.children_text)
                if plan.children_text
                else None
            )
            fidelity = angle = None
            if parent_vec is not None and children_vec is not None:
                fidelity, angle = self._cosine(parent_vec, children_vec)

            metrics.append(
                NodeFidelityMetric(
                    node_id=plan.node_id,
                    child_ids=list(plan.child_ids),
                    fidelity=fidelity,
                    angle_degrees=angle,
                    parent_tokens=plan.parent_tokens,
                    children_tokens=plan.children_tokens,
                    parent_span_chars=plan.parent_span_chars,
                    children_span_chars=plan.children_span_chars,
                    parent_preview=self._preview(plan.parent_text),
                    children_preview=self._preview(plan.children_text),
                )
            )

        stats = self._compute_stats(metrics)
        histogram, underflow, overflow = self._build_histogram(metrics)
        worst_nodes = self._select_worst(metrics)

        return SummarizationFidelityResult(
            document_id=self._doc_id,
            embedding_model=self.embedding_model,
            embedding_dim=self.vector_dim,
            metrics=metrics,
            stats=stats,
            histogram=histogram,
            histogram_underflow=underflow,
            histogram_overflow=overflow,
            worst_nodes=worst_nodes,
        )

    def _build_plan(self, node_map: dict[str, TreeNode]) -> list[_NodePlan]:
        plans: list[_NodePlan] = []
        for node in node_map.values():
            child_ids = self._child_ids(node, node_map)
            if not child_ids:
                continue

            parent_text = self._prepare_text(node.text)
            children_texts = [
                self._prepare_text(node_map[child_id].text) for child_id in child_ids
            ]
            children_text = "\n".join(text for text in children_texts if text).strip()
            parent_tokens = tokenizer.count_tokens(parent_text) if parent_text else 0
            children_tokens = (
                tokenizer.count_tokens(children_text) if children_text else 0
            )
            parent_span = self._span_chars(node)
            child_span = sum(
                self._span_chars(node_map[child_id]) for child_id in child_ids
            )

            plans.append(
                _NodePlan(
                    node_id=node.id,
                    child_ids=child_ids,
                    parent_text=parent_text,
                    children_text=children_text,
                    parent_tokens=parent_tokens,
                    children_tokens=children_tokens,
                    parent_span_chars=parent_span,
                    children_span_chars=child_span,
                )
            )
        plans.sort(key=lambda plan: plan.parent_span_chars, reverse=True)
        return plans

    @staticmethod
    def _child_ids(node: TreeNode, node_map: dict[str, TreeNode]) -> list[str]:
        child_ids: list[str] = []
        for attr in ("left_child_id", "right_child_id"):
            child_id = getattr(node, attr, None)
            if child_id and child_id in node_map:
                child_ids.append(child_id)
        return child_ids

    def _ensure_embeddings(self, texts: Sequence[str]) -> None:
        missing = [text for text in texts if text and text not in self._embedding_cache]
        if not missing:
            return

        for batch in self._batch_texts(missing):
            embedded = self.embedding_service.embed_texts(
                batch, document_id=self._doc_id
            )
            if len(embedded) != len(batch):
                raise RuntimeError(
                    f"Embedding batch returned {len(embedded)} results for {len(batch)} inputs"
                )
            for text, values in zip(batch, embedded):
                vector = np.asarray(values, dtype=np.float64)
                if vector.shape[0] != self.vector_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch ({vector.shape[0]} vs expected {self.vector_dim})"
                    )
                self._embedding_cache[text] = vector

    def _batch_texts(self, texts: Sequence[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        max_items = 16
        max_tokens = 200_000

        for text in texts:
            token_count = tokenizer.count_tokens(text)
            would_exceed_items = len(current) >= max_items
            would_exceed_tokens = current and current_tokens + token_count > max_tokens

            if current and (would_exceed_items or would_exceed_tokens):
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(text)
            current_tokens += token_count

        if current:
            batches.append(current)

        return batches

    def _compute_stats(
        self, metrics: Sequence[NodeFidelityMetric]
    ) -> SummarizationFidelityStats:
        values = [metric.fidelity for metric in metrics if metric.fidelity is not None]
        if not values:
            return SummarizationFidelityStats(
                count=len(metrics),
                mean=None,
                median=None,
                minimum=None,
                maximum=None,
                stddev=None,
            )

        mean = statistics.fmean(values)
        median = statistics.median(values)
        minimum = min(values)
        maximum = max(values)
        stddev = statistics.pstdev(values) if len(values) >= 2 else 0.0

        return SummarizationFidelityStats(
            count=len(metrics),
            mean=mean,
            median=median,
            minimum=minimum,
            maximum=maximum,
            stddev=stddev,
        )

    def _build_histogram(
        self, metrics: Sequence[NodeFidelityMetric]
    ) -> tuple[list[FidelityHistogramBucket], int, int]:
        values = [metric.fidelity for metric in metrics if metric.fidelity is not None]
        if not values:
            return [], 0, 0

        start = self.settings.histogram_start
        size = max(1e-6, self.settings.histogram_bucket_size)
        buckets = max(1, self.settings.histogram_buckets)
        histogram: list[FidelityHistogramBucket] = []
        current = start
        for index in range(buckets):
            upper = current + size
            bucket_end = min(1.0, upper)
            inclusive = index == buckets - 1
            count = sum(
                1
                for value in values
                if value is not None
                and (
                    current <= value <= bucket_end
                    if inclusive
                    else current <= value < bucket_end
                )
            )
            histogram.append(
                FidelityHistogramBucket(start=current, end=bucket_end, count=count)
            )
            current = upper

        last_end = histogram[-1].end if histogram else start
        underflow = sum(1 for value in values if value < start)
        overflow = sum(1 for value in values if value > last_end)
        return histogram, underflow, overflow

    def _select_worst(
        self, metrics: Sequence[NodeFidelityMetric]
    ) -> list[NodeFidelityMetric]:
        scored = [metric for metric in metrics if metric.fidelity is not None]
        scored.sort(
            key=lambda metric: cast(float, metric.fidelity)
        )  # ascending fidelity
        return scored[: self.settings.top_k_worst]

    @staticmethod
    def _clean_text(text: str | None) -> str:
        return (text or "").strip()

    def _prepare_text(self, text: str | None) -> str:
        cleaned = self._clean_text(text)
        if not cleaned:
            return ""
        # Fast path when already within limits
        tokens = tokenizer.encode(cleaned)
        if len(tokens) <= self._max_embed_tokens:
            return cleaned
        truncated = tokenizer.decode(tokens[: self._max_embed_tokens])
        return truncated

    @staticmethod
    def _span_chars(node: TreeNode) -> int:
        start = int(getattr(node, "span_start", 0) or 0)
        end = int(getattr(node, "span_end", start) or start)
        return max(0, end - start)

    @staticmethod
    def _preview(text: str, limit: int = 80) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 1] + "…"

    @staticmethod
    def _cosine(vec_a: Vector, vec_b: Vector) -> tuple[float | None, float | None]:
        norm_a = float(np.linalg.norm(vec_a))
        norm_b = float(np.linalg.norm(vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return None, None
        cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        cosine = max(-1.0, min(1.0, cosine))
        angle = math.degrees(math.acos(cosine))
        return cosine, angle
