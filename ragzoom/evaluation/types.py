"""Data types for summary evaluation."""

from dataclasses import dataclass, field
from statistics import mean, quantiles, stdev


@dataclass(frozen=True)
class DimensionScore:
    """Score for a single evaluation dimension."""

    score: int  # 1-5
    explanation: str

    def __post_init__(self) -> None:
        if not 1 <= self.score <= 5:
            raise ValueError(f"Score must be 1-5, got {self.score}")

    def to_dict(self) -> dict[str, int | str]:
        """Convert to JSON-serializable dict."""
        return {"score": self.score, "explanation": self.explanation}

    @classmethod
    def from_dict(cls, data: dict[str, int | str]) -> "DimensionScore":
        """Create from dict."""
        return cls(score=int(data["score"]), explanation=str(data["explanation"]))


@dataclass(frozen=True)
class NodeEvaluation:
    """Evaluation results for a single inner node."""

    node_id: str
    height: int
    level_index: int
    span_start: int
    compression_ratio: float  # children_tokens / summary_tokens

    retention: DimensionScore
    isolation: DimensionScore
    faithfulness: DimensionScore
    continuity: DimensionScore

    @property
    def min_score(self) -> int:
        """Return the lowest score across all dimensions."""
        return min(
            self.retention.score,
            self.isolation.score,
            self.faithfulness.score,
            self.continuity.score,
        )

    @property
    def mean_score(self) -> float:
        """Return the mean score across all dimensions."""
        return mean(
            [
                self.retention.score,
                self.isolation.score,
                self.faithfulness.score,
                self.continuity.score,
            ]
        )

    def to_dict(self) -> dict[str, str | int | float | dict[str, int | str]]:
        """Convert to JSON-serializable dict."""
        return {
            "node_id": self.node_id,
            "height": self.height,
            "level_index": self.level_index,
            "span_start": self.span_start,
            "compression_ratio": self.compression_ratio,
            "retention": self.retention.to_dict(),
            "isolation": self.isolation.to_dict(),
            "faithfulness": self.faithfulness.to_dict(),
            "continuity": self.continuity.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, str | int | float | dict[str, int | str]]
    ) -> "NodeEvaluation":
        """Create from dict."""
        retention_data = data["retention"]
        isolation_data = data["isolation"]
        faithfulness_data = data["faithfulness"]
        continuity_data = data["continuity"]
        assert isinstance(retention_data, dict)
        assert isinstance(isolation_data, dict)
        assert isinstance(faithfulness_data, dict)
        assert isinstance(continuity_data, dict)
        return cls(
            node_id=str(data["node_id"]),
            height=int(data["height"]),  # type: ignore[arg-type]
            level_index=int(data["level_index"]),  # type: ignore[arg-type]
            span_start=int(data["span_start"]),  # type: ignore[arg-type]
            compression_ratio=float(data["compression_ratio"]),  # type: ignore[arg-type]
            retention=DimensionScore.from_dict(retention_data),
            isolation=DimensionScore.from_dict(isolation_data),
            faithfulness=DimensionScore.from_dict(faithfulness_data),
            continuity=DimensionScore.from_dict(continuity_data),
        )


DIMENSIONS = ("retention", "isolation", "faithfulness", "continuity")


@dataclass
class EvaluationReport:
    """Aggregated evaluation report for a document."""

    document_id: str
    total_inner_nodes: int
    nodes_evaluated: int
    evaluations: list[NodeEvaluation] = field(default_factory=list)

    def mean_scores(self) -> dict[str, float]:
        """Return mean score for each dimension."""
        if not self.evaluations:
            return {dim: 0.0 for dim in DIMENSIONS}

        return {
            dim: mean(getattr(e, dim).score for e in self.evaluations)
            for dim in DIMENSIONS
        }

    def std_scores(self) -> dict[str, float]:
        """Return standard deviation for each dimension."""
        if len(self.evaluations) < 2:
            return {dim: 0.0 for dim in DIMENSIONS}

        return {
            dim: stdev(getattr(e, dim).score for e in self.evaluations)
            for dim in DIMENSIONS
        }

    def outliers(self, threshold: int = 2) -> list[NodeEvaluation]:
        """Return evaluations with any score at or below threshold."""
        return [e for e in self.evaluations if e.min_score <= threshold]

    def overall_mean(self) -> float:
        """Return mean across all dimensions and nodes."""
        if not self.evaluations:
            return 0.0
        return mean(e.mean_score for e in self.evaluations)

    def passed(self, min_mean: float) -> bool:
        """Return True if overall mean meets threshold."""
        return self.overall_mean() >= min_mean

    def percentile_scores(self, percentile: int) -> dict[str, float]:
        """Return the given percentile (0-100) for each dimension.

        Uses exclusive method: percentile 5 gives the value at the 5th percentile.
        """
        if len(self.evaluations) < 2:
            return {dim: 0.0 for dim in DIMENSIONS}

        # quantiles(data, n=20) gives 19 cut points for 5% intervals
        # Index 0 = p5, index 1 = p10, ..., index 18 = p95
        result: dict[str, float] = {}
        for dim in DIMENSIONS:
            scores = sorted(getattr(e, dim).score for e in self.evaluations)
            # quantiles with n=20 gives cut points at 5%, 10%, ..., 95%
            cuts = quantiles(scores, n=20)
            # Map percentile to index: p5->0, p10->1, ..., p95->18
            idx = (percentile // 5) - 1
            if idx < 0:
                result[dim] = float(min(scores))
            elif idx >= len(cuts):
                result[dim] = float(max(scores))
            else:
                result[dim] = cuts[idx]
        return result

    def failure_count(self, threshold: float = 2.5) -> int:
        """Count nodes with any dimension score below threshold."""
        return sum(1 for e in self.evaluations if e.min_score < threshold)
