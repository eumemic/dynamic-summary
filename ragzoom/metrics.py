"""Performance metrics collection for RagZoom indexing."""

import time
from dataclasses import dataclass, field

from ragzoom.config import RagZoomConfig


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

    def add_summary(self, target: int, actual: int) -> None:
        """Record a summary result."""
        self.count += 1
        self.total_tokens += actual

        deviation_percent = abs(actual - target) / target * 100
        self.total_deviation += deviation_percent

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

    # Tree structure
    tree_height: int = 0
    nodes_per_level: list[int] = field(default_factory=list)

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

    def to_dict(self) -> dict:
        """Convert metrics to dictionary for JSON serialization."""
        summary_stats_dict = {}
        for target, stats in self.summary_stats.items():
            summary_stats_dict[str(target)] = {
                "count": stats.count,
                "avg_tokens": stats.avg_tokens,
                "avg_deviation_percent": stats.avg_deviation_percent,
                "percent_over_target": stats.percent_over_target,
                "percent_under_target": stats.percent_under_target,
                "max_overage_percent": stats.max_overage_percent,
                "max_underage_percent": stats.max_underage_percent,
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

        self.metrics = IndexingMetrics(
            start_time=time.time(),
            end_time=0,
            source_document_tokens=source_tokens,
            chunks_created=0,
            embedding_cost_per_1k=config.embedding_cost_per_1k,
            summary_input_cost_per_1k=config.summary_input_cost_per_1k,
            summary_output_cost_per_1k=config.summary_output_cost_per_1k,
        )
        self._current_level = 0
        self._nodes_at_current_level = 0

    def record_chunk_created(self, chunk_id: str, tokens: int) -> None:
        """Called when a chunk is created during splitting."""
        self.metrics.chunks_created += 1
        # Track for leaf-level nodes
        if self._current_level == 0:
            self._nodes_at_current_level += 1

    def record_embedding_call(self, batch_size: int, token_counts: list[int]) -> None:
        """Called before embedding API call.

        Args:
            batch_size: Number of texts in batch
            token_counts: Token count for each text
        """
        self.metrics.embedding_api_calls += 1
        self.metrics.embedding_batch_sizes.append(batch_size)
        self.metrics.total_embedding_tokens += sum(token_counts)

    def record_summary_result(
        self,
        target_tokens: int,
        actual_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Record summary generation result with size tracking.

        Args:
            target_tokens: Target size for summary
            actual_tokens: Actual size of generated summary
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
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

    def finalize(self) -> IndexingMetrics:
        """Compute final metrics after indexing completes."""
        self.metrics.end_time = time.time()

        # Record final level
        if self._nodes_at_current_level > 0:
            self.metrics.nodes_per_level.append(self._nodes_at_current_level)

        return self.metrics
