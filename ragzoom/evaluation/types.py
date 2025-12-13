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
