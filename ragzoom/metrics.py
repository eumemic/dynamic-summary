"""Performance metrics collection for RagZoom indexing."""

import logging
import statistics
import time
from dataclasses import dataclass, field

import psutil

from ragzoom.config import RagZoomConfig

logger = logging.getLogger(__name__)


@dataclass
class SummaryStats:
    """Statistics for summaries at a specific target size."""

    count: int = 0
    total_tokens: int = 0
    total_deviation: float = 0.0
    over_target_count: int = 0
    under_target_count: int = 0
    max_overage_percent: float = 0.0
    max_underage_percent: float = 0.0

    # Distribution tracking
    deviations: list[float] = field(default_factory=list)
    histogram_buckets: dict[str, int] = field(
        default_factory=lambda: {
            "0-10%": 0,
            "10-25%": 0,
            "25-50%": 0,
            "50-100%": 0,
            "100%+": 0,
        }
    )

    def add_summary(self, target: int, actual: int) -> None:
        """Record a summary result."""
        self.count += 1
        self.total_tokens += actual

        deviation_percent = abs(actual - target) / target * 100
        self.total_deviation += deviation_percent

        # Track deviation for distribution analysis
        self.deviations.append(deviation_percent)

        # Update histogram bucket
        if deviation_percent <= 10:
            self.histogram_buckets["0-10%"] += 1
        elif deviation_percent <= 25:
            self.histogram_buckets["10-25%"] += 1
        elif deviation_percent <= 50:
            self.histogram_buckets["25-50%"] += 1
        elif deviation_percent <= 100:
            self.histogram_buckets["50-100%"] += 1
        else:
            self.histogram_buckets["100%+"] += 1

        if actual > target:
            self.over_target_count += 1
            overage = (actual - target) / target * 100
            self.max_overage_percent = max(self.max_overage_percent, overage)
        else:
            self.under_target_count += 1
            underage = (target - actual) / target * 100
            self.max_underage_percent = max(self.max_underage_percent, underage)

    @property
    def avg_tokens(self) -> float:
        """Average summary size in tokens."""
        return self.total_tokens / self.count if self.count > 0 else 0

    @property
    def avg_deviation_percent(self) -> float:
        """Average absolute deviation from target."""
        return self.total_deviation / self.count if self.count > 0 else 0

    @property
    def percent_over_target(self) -> float:
        """Percentage of summaries over target."""
        return self.over_target_count / self.count * 100 if self.count > 0 else 0

    @property
    def percent_under_target(self) -> float:
        """Percentage of summaries under target."""
        return self.under_target_count / self.count * 100 if self.count > 0 else 0

    @property
    def median_deviation_percent(self) -> float:
        """Median deviation from target (more robust than mean)."""
        if not self.deviations:
            return 0.0
        return statistics.median(self.deviations)

    @property
    def percentile_50(self) -> float:
        """50th percentile (median) of deviations."""
        return self.median_deviation_percent

    @property
    def percentile_90(self) -> float:
        """90th percentile of deviations."""
        if not self.deviations:
            return 0.0
        return statistics.quantiles(self.deviations, n=10)[8]  # 9th of 9 cut points

    @property
    def percentile_95(self) -> float:
        """95th percentile of deviations."""
        if not self.deviations:
            return 0.0
        return statistics.quantiles(self.deviations, n=20)[18]  # 19th of 19 cut points

    @property
    def std_deviation_percent(self) -> float:
        """Standard deviation of deviation percentages."""
        if len(self.deviations) < 2:
            return 0.0
        return statistics.stdev(self.deviations)

    @property
    def histogram(self) -> dict[str, dict[str, float]]:
        """Histogram with counts and percentages for each bucket."""
        if self.count == 0:
            return {}

        result = {}
        for bucket, count in self.histogram_buckets.items():
            result[bucket] = {"count": count, "percentage": (count / self.count) * 100}
        return result


@dataclass
class IndexingMetrics:
    """Complete metrics from an indexing operation."""

    # Timing
    start_time: float
    end_time: float

    # Document info
    source_document_tokens: int
    chunks_created: int

    # Cost configuration (per 1K tokens) - must be provided, no defaults
    embedding_cost_per_1k: float
    summary_input_cost_per_1k: float
    summary_output_cost_per_1k: float

    # API usage
    embedding_api_calls: int = 0
    summary_api_calls: int = 0
    total_embedding_tokens: int = 0
    total_summary_prompt_tokens: int = 0
    total_summary_completion_tokens: int = 0

    # Batch tracking
    embedding_batch_sizes: list[int] = field(default_factory=list)

    # Summary accuracy by target size
    summary_stats: dict[int, SummaryStats] = field(default_factory=dict)

    # Amplification tracking (per-operation)
    input_amplifications: list[float] = field(default_factory=list)
    output_amplifications: list[float] = field(default_factory=list)
    cost_amplifications: list[float] = field(default_factory=list)

    # Amplifications by tree level
    amplifications_by_level: dict[int, dict[str, list[float]]] = field(
        default_factory=dict
    )

    # Tree structure
    tree_height: int = 0
    nodes_per_level: list[int] = field(default_factory=list)

    # Memory usage (in MB)
    peak_memory_mb: float = 0.0
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0

    @property
    def total_duration_seconds(self) -> float:
        """Total indexing time in seconds."""
        return self.end_time - self.start_time

    @property
    def tokens_per_second(self) -> float:
        """Source document tokens processed per second."""
        duration = self.total_duration_seconds
        return self.source_document_tokens / duration if duration > 0 else 0

    @property
    def time_per_1k_tokens(self) -> float:
        """Time to process 1000 source tokens."""
        if self.source_document_tokens > 0:
            return self.total_duration_seconds / (self.source_document_tokens / 1000)
        return 0

    @property
    def avg_embedding_batch_size(self) -> float:
        """Average embedding batch size."""
        if self.embedding_batch_sizes:
            return sum(self.embedding_batch_sizes) / len(self.embedding_batch_sizes)
        return 0

    @property
    def total_api_calls(self) -> int:
        """Total API calls made."""
        return self.embedding_api_calls + self.summary_api_calls

    @property
    def embedding_tokens_per_1k(self) -> float:
        """Embedding tokens used per 1K source tokens."""
        if self.source_document_tokens > 0:
            return self.total_embedding_tokens / (self.source_document_tokens / 1000)
        return 0

    @property
    def summary_tokens_per_1k(self) -> float:
        """Summary tokens (prompt + completion) per 1K source tokens."""
        total_summary_tokens = (
            self.total_summary_prompt_tokens + self.total_summary_completion_tokens
        )
        if self.source_document_tokens > 0:
            return total_summary_tokens / (self.source_document_tokens / 1000)
        return 0

    @property
    def api_calls_per_1k(self) -> float:
        """API calls per 1K source tokens."""
        if self.source_document_tokens > 0:
            return self.total_api_calls / (self.source_document_tokens / 1000)
        return 0

    @property
    def cost_per_1k_tokens(self) -> float:
        """Estimated cost per 1K source tokens using configured pricing."""
        embedding_cost = (
            self.total_embedding_tokens / 1000
        ) * self.embedding_cost_per_1k
        prompt_cost = (
            self.total_summary_prompt_tokens / 1000
        ) * self.summary_input_cost_per_1k
        completion_cost = (
            self.total_summary_completion_tokens / 1000
        ) * self.summary_output_cost_per_1k

        total_cost = embedding_cost + prompt_cost + completion_cost

        if self.source_document_tokens > 0:
            return total_cost / (self.source_document_tokens / 1000)
        return 0

    @property
    def memory_usage_mb(self) -> float:
        """Peak memory usage during indexing in MB."""
        return self.peak_memory_mb - self.memory_start_mb

    @property
    def median_cost_amplification(self) -> float:
        """Median cost amplification factor across all operations."""
        if not self.cost_amplifications:
            return 0.0
        return statistics.median(self.cost_amplifications)

    @property
    def cost_amplification_p90(self) -> float:
        """90th percentile of cost amplification."""
        if not self.cost_amplifications:
            return 0.0
        sorted_values = sorted(self.cost_amplifications)
        index = int(len(sorted_values) * 0.9)
        return sorted_values[min(index, len(sorted_values) - 1)]

    @property
    def cost_amplification_p95(self) -> float:
        """95th percentile of cost amplification."""
        if not self.cost_amplifications:
            return 0.0
        sorted_values = sorted(self.cost_amplifications)
        index = int(len(sorted_values) * 0.95)
        return sorted_values[min(index, len(sorted_values) - 1)]

    @property
    def median_input_amplification(self) -> float:
        """Median input amplification factor."""
        if not self.input_amplifications:
            return 0.0
        return statistics.median(self.input_amplifications)

    @property
    def median_output_amplification(self) -> float:
        """Median output amplification factor."""
        if not self.output_amplifications:
            return 0.0
        return statistics.median(self.output_amplifications)

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for JSON serialization."""
        summary_stats_dict = {}
        for target, stats in self.summary_stats.items():
            summary_stats_dict[str(target)] = {
                "count": stats.count,
                "avg_tokens": stats.avg_tokens,
                "avg_deviation_percent": stats.avg_deviation_percent,
                "median_deviation_percent": stats.median_deviation_percent,
                "std_deviation_percent": stats.std_deviation_percent,
                "percentile_50": stats.percentile_50,
                "percentile_90": stats.percentile_90,
                "percentile_95": stats.percentile_95,
                "percent_over_target": stats.percent_over_target,
                "percent_under_target": stats.percent_under_target,
                "max_overage_percent": stats.max_overage_percent,
                "max_underage_percent": stats.max_underage_percent,
                "histogram": stats.histogram,
            }

        return {
            "timing": {
                "total_duration_seconds": self.total_duration_seconds,
                "tokens_per_second": self.tokens_per_second,
                "time_per_1k_tokens": self.time_per_1k_tokens,
            },
            "document": {
                "source_tokens": self.source_document_tokens,
                "chunks_created": self.chunks_created,
            },
            "api_usage": {
                "total_calls": self.total_api_calls,
                "embedding_calls": self.embedding_api_calls,
                "summary_calls": self.summary_api_calls,
                "embedding_tokens": self.total_embedding_tokens,
                "summary_prompt_tokens": self.total_summary_prompt_tokens,
                "summary_completion_tokens": self.total_summary_completion_tokens,
            },
            "efficiency": {
                "avg_embedding_batch_size": self.avg_embedding_batch_size,
                "embedding_tokens_per_1k": self.embedding_tokens_per_1k,
                "summary_tokens_per_1k": self.summary_tokens_per_1k,
                "api_calls_per_1k": self.api_calls_per_1k,
                "cost_per_1k_tokens": self.cost_per_1k_tokens,
            },
            "summary_accuracy": summary_stats_dict,
            "tree_structure": {
                "height": self.tree_height,
                "nodes_per_level": self.nodes_per_level,
            },
            "memory": {
                "peak_mb": self.peak_memory_mb,
                "start_mb": self.memory_start_mb,
                "end_mb": self.memory_end_mb,
                "usage_mb": self.memory_usage_mb,
            },
            "amplification": {
                "median_cost": self.median_cost_amplification,
                "cost_p90": self.cost_amplification_p90,
                "cost_p95": self.cost_amplification_p95,
                "median_input": self.median_input_amplification,
                "median_output": self.median_output_amplification,
                "by_level": self.amplifications_by_level,
            },
        }


class IndexingMetricsReporter:
    """Collects performance metrics during indexing with minimal intrusion.

    Design decisions:
    - Metrics collection is completely optional via the reporter parameter
    - Errors in metrics collection are silently ignored to never interfere with indexing
    - The reporter pattern avoids polluting core logic with metrics concerns
    - All metrics are collected in a single pass during normal indexing flow
    """

    def __init__(
        self,
        document_id: str,
        source_tokens: int,
        config: RagZoomConfig,
    ):
        """Initialize reporter for a document.

        Args:
            document_id: Document being indexed
            source_tokens: Total tokens in source document
            config: Config for cost estimation (required for pricing)
        """
        self.document_id = document_id

        # Initialize metrics with config-based pricing
        if not config:
            raise ValueError(
                "RagZoomConfig is required for metrics collection to provide pricing information"
            )

        # Get current process for memory tracking
        self.process = psutil.Process()

        # Get initial memory usage
        memory_info = self.process.memory_info()
        initial_memory_mb = memory_info.rss / 1024 / 1024

        self.metrics = IndexingMetrics(
            start_time=time.time(),
            end_time=0,
            source_document_tokens=source_tokens,
            chunks_created=0,
            embedding_cost_per_1k=config.embedding_cost_per_1k,
            summary_input_cost_per_1k=config.summary_input_cost_per_1k,
            summary_output_cost_per_1k=config.summary_output_cost_per_1k,
            memory_start_mb=initial_memory_mb,
            peak_memory_mb=initial_memory_mb,
        )
        self._current_level = 0
        self._nodes_at_current_level = 0

    def _update_memory_usage(self) -> None:
        """Update peak memory usage if current usage is higher."""
        try:
            memory_info = self.process.memory_info()
            current_memory_mb = memory_info.rss / 1024 / 1024
            if current_memory_mb > self.metrics.peak_memory_mb:
                self.metrics.peak_memory_mb = current_memory_mb
        except Exception as e:
            # Log but don't fail indexing due to memory tracking issues
            logger.warning(f"Failed to update memory usage: {e}")

    def record_chunk_created(self, chunk_id: str, tokens: int) -> None:
        """Called when a chunk is created during splitting."""
        self.metrics.chunks_created += 1
        # Track for leaf-level nodes
        if self._current_level == 0:
            self._nodes_at_current_level += 1
        self._update_memory_usage()

    def record_embedding_call(self, batch_size: int, token_counts: list[int]) -> None:
        """Called before embedding API call.

        Args:
            batch_size: Number of texts in batch
            token_counts: Token count for each text
        """
        self.metrics.embedding_api_calls += 1
        self.metrics.embedding_batch_sizes.append(batch_size)
        self.metrics.total_embedding_tokens += sum(token_counts)
        self._update_memory_usage()

    def record_summary_result(
        self,
        target_tokens: int,
        actual_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        input_text_tokens: int,
    ) -> None:
        """Record summary generation result with size tracking and amplification metrics.

        Args:
            target_tokens: Target size for summary
            actual_tokens: Actual size of generated summary
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            input_text_tokens: Tokens in the text being summarized (left + right)
        """
        self.metrics.summary_api_calls += 1
        self.metrics.total_summary_prompt_tokens += prompt_tokens
        self.metrics.total_summary_completion_tokens += completion_tokens

        # Track accuracy by target size
        if target_tokens not in self.metrics.summary_stats:
            self.metrics.summary_stats[target_tokens] = SummaryStats()

        self.metrics.summary_stats[target_tokens].add_summary(
            target_tokens, actual_tokens
        )

        # Calculate amplification factors
        if input_text_tokens > 0:
            input_amplification = prompt_tokens / input_text_tokens
            output_amplification = (
                completion_tokens / actual_tokens if actual_tokens > 0 else 1.0
            )

            # Calculate cost-weighted amplification
            # Cost amplification = (actual cost / theoretical minimum cost)
            actual_cost = (
                prompt_tokens * self.metrics.summary_input_cost_per_1k
                + completion_tokens * self.metrics.summary_output_cost_per_1k
            ) / 1000

            min_cost = (
                input_text_tokens * self.metrics.summary_input_cost_per_1k
                + actual_tokens * self.metrics.summary_output_cost_per_1k
            ) / 1000

            cost_amplification = actual_cost / min_cost if min_cost > 0 else 1.0

            # Record per-operation amplifications
            self.metrics.input_amplifications.append(input_amplification)
            self.metrics.output_amplifications.append(output_amplification)
            self.metrics.cost_amplifications.append(cost_amplification)

            # Track by level
            level = self._current_level
            if level not in self.metrics.amplifications_by_level:
                self.metrics.amplifications_by_level[level] = {
                    "input": [],
                    "output": [],
                    "cost": [],
                }

            self.metrics.amplifications_by_level[level]["input"].append(
                input_amplification
            )
            self.metrics.amplifications_by_level[level]["output"].append(
                output_amplification
            )
            self.metrics.amplifications_by_level[level]["cost"].append(
                cost_amplification
            )

        self._update_memory_usage()

    def record_tree_level_complete(self, level: int, nodes_created: int) -> None:
        """Called when a tree level is built.

        Args:
            level: Tree level (0 = leaves)
            nodes_created: Number of nodes at this level
        """
        # Record nodes from previous level when moving to new level
        if level > self._current_level:
            self.metrics.nodes_per_level.append(self._nodes_at_current_level)
            self._current_level = level
            self._nodes_at_current_level = nodes_created
        else:
            self._nodes_at_current_level = nodes_created

        self.metrics.tree_height = max(self.metrics.tree_height, level)
        self._update_memory_usage()

    def finalize(self) -> IndexingMetrics:
        """Compute final metrics after indexing completes."""
        self.metrics.end_time = time.time()

        # Record final level
        if self._nodes_at_current_level > 0:
            self.metrics.nodes_per_level.append(self._nodes_at_current_level)

        # Record final memory usage
        try:
            memory_info = self.process.memory_info()
            self.metrics.memory_end_mb = memory_info.rss / 1024 / 1024
        except Exception:
            self.metrics.memory_end_mb = self.metrics.peak_memory_mb

        return self.metrics
