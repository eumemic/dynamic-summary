"""Telemetry analysis tools for computing metrics from raw telemetry data.

This module enables retroactive metric computation from telemetry data collected
during indexing. It supports computing amplification metrics, retry analysis,
and other insights from historical benchmark data.
"""

import logging
import os
import statistics
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from ragzoom.config import RagZoomConfig

logger = logging.getLogger(__name__)

# Current supported telemetry format versions
SUPPORTED_TELEMETRY_VERSIONS = ["1.0", "2.0"]

# Default token estimate for leaf nodes when source tokens are not available
DEFAULT_LEAF_TOKEN_ESTIMATE = 150


@dataclass
class SummaryStats:
    """Statistics for summaries at a specific target size, computed from telemetry."""

    count: int = 0
    total_tokens: int = 0
    total_deviation: float = 0.0
    over_target_count: int = 0
    under_target_count: int = 0
    max_overage_percent: float = 0.0
    max_underage_percent: float = 0.0
    deviations: list[float] = field(default_factory=list)
    histogram: dict[str, dict[str, float]] = field(default_factory=dict)

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
        return median(self.deviations)

    @property
    def percentile_50(self) -> float:
        """50th percentile (median) of deviations."""
        return self.median_deviation_percent

    @property
    def percentile_90(self) -> float:
        """90th percentile of deviations."""
        if not self.deviations:
            return 0.0
        if len(self.deviations) < 2:
            return max(self.deviations)
        return _compute_percentile(self.deviations, 0.9)

    @property
    def percentile_95(self) -> float:
        """95th percentile of deviations."""
        if not self.deviations:
            return 0.0
        if len(self.deviations) < 2:
            return max(self.deviations)
        return _compute_percentile(self.deviations, 0.95)

    @property
    def std_deviation_percent(self) -> float:
        """Standard deviation of deviation percentages."""
        if len(self.deviations) < 2:
            return 0.0
        return statistics.stdev(self.deviations)


class TelemetryThresholds:
    """Configurable thresholds for telemetry analysis and visualization.

    Thresholds can be overridden via environment variables:
    - RAGZOOM_HIGH_INPUT_AMPLIFICATION_THRESHOLD (default: 3.0)
    - RAGZOOM_HIGH_COST_AMPLIFICATION_THRESHOLD (default: 2.0)
    - RAGZOOM_GOOD_COST_AMPLIFICATION_THRESHOLD (default: 1.5)
    - RAGZOOM_HIGH_RETRY_RATE_THRESHOLD (default: 20)
    - RAGZOOM_GOOD_BATCH_UTILIZATION_THRESHOLD (default: 70)
    - RAGZOOM_LOW_BATCH_UTILIZATION_THRESHOLD (default: 50)
    - RAGZOOM_MULTIPLE_RETRY_THRESHOLD (default: 1)
    """

    def __init__(self) -> None:
        self.high_input_amplification = float(
            os.getenv("RAGZOOM_HIGH_INPUT_AMPLIFICATION_THRESHOLD", "3.0")
        )
        self.high_cost_amplification = float(
            os.getenv("RAGZOOM_HIGH_COST_AMPLIFICATION_THRESHOLD", "2.0")
        )
        self.good_cost_amplification = float(
            os.getenv("RAGZOOM_GOOD_COST_AMPLIFICATION_THRESHOLD", "1.5")
        )
        self.high_retry_rate = float(
            os.getenv("RAGZOOM_HIGH_RETRY_RATE_THRESHOLD", "20")
        )
        self.good_batch_utilization = float(
            os.getenv("RAGZOOM_GOOD_BATCH_UTILIZATION_THRESHOLD", "70")
        )
        self.low_batch_utilization = float(
            os.getenv("RAGZOOM_LOW_BATCH_UTILIZATION_THRESHOLD", "50")
        )
        self.multiple_retry = int(os.getenv("RAGZOOM_MULTIPLE_RETRY_THRESHOLD", "1"))


# Global instance for convenience
_telemetry_thresholds = TelemetryThresholds()


def get_telemetry_thresholds() -> TelemetryThresholds:
    """Get the global telemetry thresholds instance."""
    return _telemetry_thresholds


class TelemetryAnalysisError(Exception):
    """Raised when telemetry analysis encounters an error."""

    pass


def parse_telemetry_format(telemetry_data: dict) -> dict:
    """Parse telemetry data, handling version differences gracefully.

    Args:
        telemetry_data: Raw telemetry data from benchmark file

    Returns:
        Parsed telemetry data in standardized format

    Raises:
        TelemetryAnalysisError: If format version is unsupported
    """
    if not isinstance(telemetry_data, dict):
        raise TelemetryAnalysisError("Telemetry data must be a dictionary")

    format_version = telemetry_data.get("format_version")
    if not format_version:
        raise TelemetryAnalysisError("Missing format_version in telemetry data")

    if format_version not in SUPPORTED_TELEMETRY_VERSIONS:
        raise TelemetryAnalysisError(
            f"Unsupported telemetry format version: {format_version}. "
            f"Supported versions: {SUPPORTED_TELEMETRY_VERSIONS}"
        )

    documents = telemetry_data.get("documents", {})
    if not isinstance(documents, dict):
        raise TelemetryAnalysisError("Invalid documents structure in telemetry data")

    return {
        "format_version": format_version,
        "documents": documents,
    }


def compute_amplification_metrics(telemetry_data: dict, config: RagZoomConfig) -> dict:
    """Compute amplification metrics from telemetry data.

    Args:
        telemetry_data: Parsed telemetry data
        config: Configuration for cost calculations

    Returns:
        Dictionary containing amplification metrics:
        - median_cost: Median cost amplification
        - cost_p90: 90th percentile cost amplification
        - cost_p95: 95th percentile cost amplification
        - median_input: Median input amplification
        - median_output: Median output amplification
        - by_height: Amplification metrics broken down by tree height
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    all_cost_amplifications = []
    all_input_amplifications = []
    all_output_amplifications = []
    amplifications_by_height: dict[int, dict[str, list[float]]] = {}

    # Process all documents
    for doc_type, doc_data in parsed_data["documents"].items():
        nodes = doc_data.get("nodes", [])

        for node in nodes:
            # Only process summary nodes
            # v1.0: check node_type, v2.0: check height > 0
            # Get height (compatible with both v1.0 and v2.0)
            height = node.get("height", node.get("level", 0))
            is_leaf = node.get("node_type") == "leaf" or height == 0
            if is_leaf:
                continue

            summary_attempts = node.get("summary_attempts", [])

            # Track cumulative tokens for this node across all attempts
            node_total_prompt_tokens = 0
            node_total_completion_tokens = 0
            node_input_text_tokens = 0
            node_final_summary_tokens = 0
            has_accepted = False

            # Accumulate tokens from ALL attempts
            for attempt in summary_attempts:
                prompt_tokens = attempt.get("prompt_tokens", 0)
                completion_tokens = attempt.get("completion_tokens", 0)

                node_total_prompt_tokens += prompt_tokens
                node_total_completion_tokens += completion_tokens

                # Track the input text tokens (should be same across attempts)
                if node_input_text_tokens == 0:
                    node_input_text_tokens = attempt.get("input_text_tokens", 0)

                # Track the final accepted attempt's output
                if attempt.get("status") == "accepted":
                    has_accepted = True
                    node_final_summary_tokens = attempt.get("actual_tokens", 0)

            # Calculate amplification using ALL attempts' tokens
            if has_accepted and node_input_text_tokens > 0:
                # Input amplification = total prompt tokens / original text tokens
                input_amplification = node_total_prompt_tokens / node_input_text_tokens

                # Output amplification = total completion tokens / final summary tokens
                output_amplification = (
                    node_total_completion_tokens / node_final_summary_tokens
                    if node_final_summary_tokens > 0
                    else 1.0
                )

                # Calculate cost-weighted amplification using cumulative tokens
                cost_amplification = _calculate_cost_amplification(
                    node_total_prompt_tokens,
                    node_total_completion_tokens,
                    node_input_text_tokens,
                    node_final_summary_tokens,
                    config,
                )

                # Collect amplifications
                all_cost_amplifications.append(cost_amplification)
                all_input_amplifications.append(input_amplification)
                all_output_amplifications.append(output_amplification)

                # Track by height
                if height not in amplifications_by_height:
                    amplifications_by_height[height] = {
                        "input": [],
                        "output": [],
                        "cost": [],
                    }

                amplifications_by_height[height]["input"].append(input_amplification)
                amplifications_by_height[height]["output"].append(output_amplification)
                amplifications_by_height[height]["cost"].append(cost_amplification)

    # Compute summary statistics
    result = {
        "median_cost": 0.0,
        "cost_p90": 0.0,
        "cost_p95": 0.0,
        "median_input": 0.0,
        "median_output": 0.0,
        "by_height": amplifications_by_height,
    }

    if all_cost_amplifications:
        result["median_cost"] = median(all_cost_amplifications)
        result["cost_p90"] = _compute_percentile(all_cost_amplifications, 0.9)
        result["cost_p95"] = _compute_percentile(all_cost_amplifications, 0.95)

    if all_input_amplifications:
        result["median_input"] = median(all_input_amplifications)

    if all_output_amplifications:
        result["median_output"] = median(all_output_amplifications)

    return result


def _compute_percentile(values: list[float], percentile: float) -> float:
    """Compute percentile using linear interpolation (consistent with metrics.py).

    Args:
        values: List of numeric values
        percentile: Percentile to compute (0.0 to 1.0, e.g., 0.9 for 90th percentile)

    Returns:
        The computed percentile value
    """
    if not values:
        return 0.0

    n = len(values)
    if n == 1:
        return values[0]

    sorted_values = sorted(values)
    pos = (n - 1) * percentile
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    fraction = pos - lower

    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def _calculate_cost_amplification(
    prompt_tokens: int,
    completion_tokens: int,
    input_text_tokens: int,
    actual_tokens: int,
    config: RagZoomConfig,
) -> float:
    """Calculate cost-weighted amplification factor.

    Cost amplification = (actual cost / theoretical minimum cost)

    Uses completion_tokens for both actual and theoretical costs to eliminate
    variability from tokenizer differences. This focuses the metric on what we
    control (prompt overhead) rather than external factors.

    Args:
        prompt_tokens: Tokens in the API prompt
        completion_tokens: Tokens in the API completion
        input_text_tokens: Tokens in the original text being summarized
        actual_tokens: Actual tokens in the generated summary (unused, kept for compatibility)
        config: Configuration with pricing information

    Returns:
        Cost amplification factor (1.0 = no amplification)
    """
    # Calculate actual cost
    actual_cost = (
        prompt_tokens * config.summary_input_cost_per_1k
        + completion_tokens * config.summary_output_cost_per_1k
    ) / 1000

    # Calculate theoretical minimum cost using completion_tokens instead of actual_tokens
    # This eliminates noise from tokenizer differences between local and API counting
    min_cost = (
        input_text_tokens * config.summary_input_cost_per_1k
        + completion_tokens * config.summary_output_cost_per_1k
    ) / 1000

    # Return amplification factor with zero check
    return actual_cost / min_cost if min_cost > 0 else 1.0


def compute_batch_efficiency(telemetry_data: dict) -> dict:
    """Analyze embedding batch utilization from telemetry data.

    Args:
        telemetry_data: Parsed telemetry data

    Returns:
        Dictionary containing batch efficiency metrics:
        - avg_batch_size: Average embedding batch size
        - batch_sizes: List of all batch sizes
        - total_batches: Total number of embedding batches
        - total_embeddings: Total number of embeddings generated
        - batch_utilization: Average utilization as percentage
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    batch_sizes = []
    total_embeddings = 0

    # Process all documents
    for doc_type, doc_data in parsed_data["documents"].items():
        nodes = doc_data.get("nodes", [])

        # Track batches by collecting unique (batch_size, timestamp) pairs
        seen_batches = set()

        for node in nodes:
            embedding = node.get("embedding")
            if not embedding:
                continue

            batch_size = embedding.get("batch_size", 1)
            # v1.0: use timestamp, v2.0: use start_time
            timestamp = embedding.get("timestamp", embedding.get("start_time", 0))
            batch_key = (batch_size, timestamp)

            # Only count each batch once
            if batch_key not in seen_batches:
                batch_sizes.append(batch_size)
                seen_batches.add(batch_key)

            total_embeddings += 1

    # Calculate metrics
    result = {
        "avg_batch_size": 0.0,
        "batch_sizes": batch_sizes,
        "total_batches": len(batch_sizes),
        "total_embeddings": total_embeddings,
        "batch_utilization": 0.0,
    }

    if batch_sizes:
        avg_batch_size = sum(batch_sizes) / len(batch_sizes)
        result["avg_batch_size"] = avg_batch_size
        # Batch efficiency: percentage of embeddings that benefited from batching.
        # For each batch of size N, (N-1) embeddings were batched together rather
        # than sent individually.
        batched_embeddings = sum(max(0, batch_size - 1) for batch_size in batch_sizes)
        result["batch_utilization"] = (
            (batched_embeddings / total_embeddings) * 100
            if total_embeddings > 0
            else 0.0
        )

    return result


def analyze_retry_patterns(telemetry_data: dict) -> dict:
    """Analyze summary retry patterns from telemetry data.

    Args:
        telemetry_data: Parsed telemetry data

    Returns:
        Dictionary containing retry analysis:
        - retry_rate: Percentage of summaries that needed retry
        - total_attempts: Total number of summary attempts
        - successful_attempts: Number of accepted attempts
        - retry_attempts: Number of retry attempts
        - retry_success_rate: Percentage of retries that succeeded
        - rejection_reasons: Distribution of rejection reasons
        - nodes_with_retries: Number of nodes that required retries
        - total_nodes_with_summaries: Total number of nodes with summary attempts
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    total_attempts = 0
    successful_attempts = 0
    retry_attempts = 0
    successful_retries = 0
    rejection_reasons: dict[str, int] = {}
    nodes_with_retries = 0
    total_nodes_with_summaries = 0

    # Process all documents
    for doc_type, doc_data in parsed_data["documents"].items():
        nodes = doc_data.get("nodes", [])

        for node in nodes:
            # Only process summary nodes
            # v1.0: check node_type, v2.0: check height > 0
            # Get height (compatible with both v1.0 and v2.0)
            height = node.get("height", node.get("level", 0))
            is_leaf = node.get("node_type") == "leaf" or height == 0
            if is_leaf:
                continue

            summary_attempts = node.get("summary_attempts", [])
            if not summary_attempts:
                continue

            total_nodes_with_summaries += 1
            node_has_retry = False

            for attempt in summary_attempts:
                total_attempts += 1
                is_retry = attempt.get("is_retry", False)
                status = attempt.get("status", "unknown")

                if is_retry:
                    retry_attempts += 1
                    node_has_retry = True
                    if status == "accepted":
                        successful_retries += 1

                if status == "accepted":
                    successful_attempts += 1
                elif status in ["rejected_over", "rejected_under", "error"]:
                    reason = attempt.get("rejection_reason", status)
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

            if node_has_retry:
                nodes_with_retries += 1

    # Calculate metrics
    result = {
        "retry_rate": 0.0,
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "retry_attempts": retry_attempts,
        "retry_success_rate": 0.0,
        "rejection_reasons": rejection_reasons,
        "nodes_with_retries": nodes_with_retries,
        "total_nodes_with_summaries": total_nodes_with_summaries,
    }

    if total_nodes_with_summaries > 0:
        result["retry_rate"] = (nodes_with_retries / total_nodes_with_summaries) * 100

    if retry_attempts > 0:
        result["retry_success_rate"] = (successful_retries / retry_attempts) * 100

    return result


def compute_summary_stats_from_telemetry(
    telemetry_data: dict,
) -> dict[int, SummaryStats]:
    """Compute summary statistics from raw telemetry data.

    Args:
        telemetry_data: Parsed telemetry data

    Returns:
        Dictionary mapping target sizes to SummaryStats objects
    """
    parsed_data = parse_telemetry_format(telemetry_data)
    summary_stats_by_target: dict[int, SummaryStats] = {}

    # Process all documents
    for doc_type, doc_data in parsed_data["documents"].items():
        nodes = doc_data.get("nodes", [])

        for node in nodes:
            # Process summary attempts for non-leaf nodes
            summary_attempts = node.get("summary_attempts", [])

            for attempt in summary_attempts:
                # Only process accepted attempts for summary stats
                if attempt.get("status") == "accepted":
                    target_tokens = attempt.get("target_tokens", 0)
                    actual_tokens = attempt.get("actual_tokens", 0)

                    if target_tokens > 0 and actual_tokens > 0:
                        # Create stats object if needed
                        if target_tokens not in summary_stats_by_target:
                            summary_stats_by_target[target_tokens] = SummaryStats()

                        stats = summary_stats_by_target[target_tokens]

                        # Update basic counts
                        stats.count += 1
                        stats.total_tokens += actual_tokens

                        # Calculate deviation
                        deviation_percent = (
                            abs(actual_tokens - target_tokens) / target_tokens * 100
                        )
                        stats.total_deviation += deviation_percent
                        stats.deviations.append(deviation_percent)

                        # Track over/under target
                        if actual_tokens > target_tokens:
                            stats.over_target_count += 1
                            overage = (
                                (actual_tokens - target_tokens) / target_tokens * 100
                            )
                            stats.max_overage_percent = max(
                                stats.max_overage_percent, overage
                            )
                        else:
                            stats.under_target_count += 1
                            underage = (
                                (target_tokens - actual_tokens) / target_tokens * 100
                            )
                            stats.max_underage_percent = max(
                                stats.max_underage_percent, underage
                            )

    # Compute histograms for each target size
    for target_size, stats in summary_stats_by_target.items():
        if stats.count > 0:
            histogram_buckets = {
                "0-10%": 0,
                "10-25%": 0,
                "25-50%": 0,
                "50-100%": 0,
                "100%+": 0,
            }

            for deviation in stats.deviations:
                if deviation <= 10:
                    histogram_buckets["0-10%"] += 1
                elif deviation <= 25:
                    histogram_buckets["10-25%"] += 1
                elif deviation <= 50:
                    histogram_buckets["25-50%"] += 1
                elif deviation <= 100:
                    histogram_buckets["50-100%"] += 1
                else:
                    histogram_buckets["100%+"] += 1

            # Convert to percentage format
            stats.histogram = {}
            for bucket, count in histogram_buckets.items():
                stats.histogram[bucket] = {
                    "count": count,
                    "percentage": (count / stats.count) * 100,
                }

    return summary_stats_by_target


@dataclass
class ComputedMetrics:
    """Computed metrics from telemetry data.

    This provides the same attributes that were previously computed by IndexingMetrics,
    but calculated from raw telemetry data during analysis.
    """

    # Timing
    start_time: float
    end_time: float
    total_duration_seconds: float

    # Document info
    source_document_tokens: int
    chunks_created: int
    tokens_per_second: float
    time_per_1k_tokens: float

    # Cost configuration
    embedding_cost_per_1k: float
    summary_input_cost_per_1k: float
    summary_output_cost_per_1k: float

    # API usage
    embedding_api_calls: int
    summary_api_calls: int
    total_api_calls: int
    total_embedding_tokens: int
    total_summary_prompt_tokens: int
    total_summary_completion_tokens: int

    # Derived metrics
    embedding_tokens_per_1k: float
    summary_tokens_per_1k: float
    api_calls_per_1k: float
    cost_per_1k_tokens: float
    avg_embedding_batch_size: float

    # Memory metrics
    peak_memory_mb: float
    memory_start_mb: float
    memory_end_mb: float
    memory_usage_mb: float

    # Collections
    embedding_batch_sizes: list[int]
    input_amplifications: list[float]
    output_amplifications: list[float]
    cost_amplifications: list[float]
    summary_stats: dict[int, SummaryStats]
    amplifications_by_height: dict[int, dict[str, list[float]]]
    tree_height: int
    nodes_per_height: list[int]


def compute_metrics_from_telemetry(
    telemetry_data: dict, config: RagZoomConfig
) -> ComputedMetrics:
    """Compute metrics from raw telemetry data.

    This function computes all metrics from raw telemetry data that were
    previously available as computed properties in IndexingMetrics.

    Args:
        telemetry_data: Raw telemetry data from benchmark file
        config: Configuration for cost calculations and pricing

    Returns:
        ComputedMetrics object with calculated values

    Raises:
        TelemetryAnalysisError: If telemetry data is invalid or incomplete
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    # Initialize metrics collection
    metrics_data: dict[str, Any] = {
        "start_time": 0.0,
        "end_time": 0.0,
        "source_document_tokens": 0,
        "chunks_created": 0,
        "embedding_cost_per_1k": config.embedding_cost_per_1k,
        "summary_input_cost_per_1k": config.summary_input_cost_per_1k,
        "summary_output_cost_per_1k": config.summary_output_cost_per_1k,
        "total_embedding_tokens": 0,
        "embedding_api_calls": 0,
        "embedding_batch_sizes": [],
        "total_summary_prompt_tokens": 0,
        "total_summary_completion_tokens": 0,
        "summary_api_calls": 0,
        "input_amplifications": [],
        "output_amplifications": [],
        "cost_amplifications": [],
        "summary_stats": {},
        "amplifications_by_height": {},  # Initialize empty dict
    }

    # Track various metrics as we process telemetry
    min_timestamp = float("inf")
    max_timestamp = 0.0
    embedding_batches = set()

    # Process all documents
    for doc_type, doc_data in parsed_data["documents"].items():
        nodes = doc_data.get("nodes", [])

        for node in nodes:
            created_at = node.get("created_at", 0)
            min_timestamp = min(min_timestamp, created_at)
            max_timestamp = max(max_timestamp, created_at)

            # Process embedding telemetry
            embedding = node.get("embedding")
            if embedding:
                text_tokens = embedding.get("text_tokens", 0)
                batch_size = embedding.get("batch_size", 1)
                # v1.0: use timestamp, v2.0: use start_time
                timestamp = embedding.get("timestamp", embedding.get("start_time", 0))

                metrics_data["total_embedding_tokens"] += text_tokens

                # Track unique batches by (batch_size, timestamp)
                batch_key = (batch_size, timestamp)
                if batch_key not in embedding_batches:
                    embedding_batches.add(batch_key)
                    metrics_data["embedding_api_calls"] += 1
                    metrics_data["embedding_batch_sizes"].append(batch_size)

            # Process summary attempts
            summary_attempts = node.get("summary_attempts", [])

            # Track cumulative tokens for this node across all attempts
            node_total_prompt_tokens = 0
            node_total_completion_tokens = 0
            node_input_text_tokens = 0
            node_final_summary_tokens = 0
            final_attempt = None

            for attempt in summary_attempts:
                metrics_data["summary_api_calls"] += 1
                prompt_tokens = attempt.get("prompt_tokens", 0)
                completion_tokens = attempt.get("completion_tokens", 0)

                metrics_data["total_summary_prompt_tokens"] += prompt_tokens
                metrics_data["total_summary_completion_tokens"] += completion_tokens

                # Accumulate tokens for amplification calculation
                node_total_prompt_tokens += prompt_tokens
                node_total_completion_tokens += completion_tokens

                # Track the input text tokens (should be same across attempts)
                if node_input_text_tokens == 0:
                    node_input_text_tokens = attempt.get("input_text_tokens", 0)

                # Track the final accepted attempt
                if attempt.get("status") == "accepted":
                    final_attempt = attempt
                    node_final_summary_tokens = attempt.get("actual_tokens", 0)

            # Calculate amplification using ALL attempts' tokens
            if summary_attempts and node_input_text_tokens > 0 and final_attempt:
                # Input amplification = total prompt tokens / original text tokens
                input_amplification = node_total_prompt_tokens / node_input_text_tokens

                # Output amplification = total completion tokens / final summary tokens
                output_amplification = (
                    node_total_completion_tokens / node_final_summary_tokens
                    if node_final_summary_tokens > 0
                    else 1.0
                )

                # Calculate cost-weighted amplification using cumulative tokens
                cost_amplification = _calculate_cost_amplification(
                    node_total_prompt_tokens,
                    node_total_completion_tokens,
                    node_input_text_tokens,
                    node_final_summary_tokens,
                    config,
                )

                # Record amplifications
                metrics_data["input_amplifications"].append(input_amplification)
                metrics_data["output_amplifications"].append(output_amplification)
                metrics_data["cost_amplifications"].append(cost_amplification)

            # Count chunks (leaf nodes)
            # v1.0: check node_type, v2.0: check height == 0
            height = node.get("height", node.get("level", 0))
            is_leaf = node.get("node_type") == "leaf" or height == 0
            if is_leaf:
                metrics_data["chunks_created"] += 1

    # Set timing
    if min_timestamp != float("inf"):
        metrics_data["start_time"] = min_timestamp
        metrics_data["end_time"] = max_timestamp

    # Compute summary stats from raw telemetry
    metrics_data["summary_stats"] = compute_summary_stats_from_telemetry(telemetry_data)

    # Calculate total source tokens from all documents
    total_source_tokens = 0
    for doc_data in parsed_data["documents"].values():
        # Try to get source tokens from metadata first (new format)
        metadata = doc_data.get("metadata", {})
        if "source_document_tokens" in metadata:
            total_source_tokens += metadata["source_document_tokens"]
        else:
            # Fallback: estimate from leaf nodes for backward compatibility
            leaf_count = sum(
                1
                for node in doc_data.get("nodes", [])
                if node.get("node_type") == "leaf"
            )
            # Rough estimate: assume average leaf size
            # This is only used for old telemetry data without metadata
            estimated_tokens = leaf_count * DEFAULT_LEAF_TOKEN_ESTIMATE
            total_source_tokens += estimated_tokens
            logger.warning(
                f"Source tokens not found in telemetry metadata, using estimate: {estimated_tokens}"
            )

    metrics_data["source_document_tokens"] = total_source_tokens

    # Tree structure analysis (estimate from height info)
    tree_height = 0
    nodes_per_height_dict: dict[int, int] = {}
    for doc_data in parsed_data["documents"].values():
        for node in doc_data.get("nodes", []):
            # v1.0: level, v2.0: height
            height = node.get("height", node.get("level", 0))
            tree_height = max(tree_height, height)
            nodes_per_height_dict[height] = nodes_per_height_dict.get(height, 0) + 1

    metrics_data["tree_height"] = tree_height
    metrics_data["nodes_per_height"] = [
        nodes_per_height_dict.get(h, 0) for h in range(tree_height + 1)
    ]

    # Compute derived metrics that were previously properties
    duration = metrics_data["end_time"] - metrics_data["start_time"]
    if duration > 0:
        metrics_data["total_duration_seconds"] = duration
        metrics_data["tokens_per_second"] = (
            metrics_data["source_document_tokens"] / duration
        )
        metrics_data["time_per_1k_tokens"] = (
            duration / (metrics_data["source_document_tokens"] / 1000)
            if metrics_data["source_document_tokens"] > 0
            else 0
        )
    else:
        metrics_data["total_duration_seconds"] = 0
        metrics_data["tokens_per_second"] = 0
        metrics_data["time_per_1k_tokens"] = 0

    # Compute cost metrics
    embedding_cost = (metrics_data["total_embedding_tokens"] / 1000) * metrics_data[
        "embedding_cost_per_1k"
    ]
    prompt_cost = (metrics_data["total_summary_prompt_tokens"] / 1000) * metrics_data[
        "summary_input_cost_per_1k"
    ]
    completion_cost = (
        metrics_data["total_summary_completion_tokens"] / 1000
    ) * metrics_data["summary_output_cost_per_1k"]
    total_cost = embedding_cost + prompt_cost + completion_cost

    if metrics_data["source_document_tokens"] > 0:
        metrics_data["cost_per_1k_tokens"] = total_cost / (
            metrics_data["source_document_tokens"] / 1000
        )
        metrics_data["embedding_tokens_per_1k"] = metrics_data[
            "total_embedding_tokens"
        ] / (metrics_data["source_document_tokens"] / 1000)
        total_summary_tokens = (
            metrics_data["total_summary_prompt_tokens"]
            + metrics_data["total_summary_completion_tokens"]
        )
        metrics_data["summary_tokens_per_1k"] = total_summary_tokens / (
            metrics_data["source_document_tokens"] / 1000
        )
        metrics_data["api_calls_per_1k"] = (
            metrics_data["embedding_api_calls"] + metrics_data["summary_api_calls"]
        ) / (metrics_data["source_document_tokens"] / 1000)
    else:
        metrics_data["cost_per_1k_tokens"] = 0
        metrics_data["embedding_tokens_per_1k"] = 0
        metrics_data["summary_tokens_per_1k"] = 0
        metrics_data["api_calls_per_1k"] = 0

    # Compute batch size metrics
    if metrics_data["embedding_batch_sizes"]:
        metrics_data["avg_embedding_batch_size"] = sum(
            metrics_data["embedding_batch_sizes"]
        ) / len(metrics_data["embedding_batch_sizes"])
    else:
        metrics_data["avg_embedding_batch_size"] = 0

    metrics_data["total_api_calls"] = (
        metrics_data["embedding_api_calls"] + metrics_data["summary_api_calls"]
    )

    # Default memory values (not tracked in telemetry)
    metrics_data["peak_memory_mb"] = 0.0
    metrics_data["memory_start_mb"] = 0.0
    metrics_data["memory_end_mb"] = 0.0
    metrics_data["memory_usage_mb"] = 0.0

    return ComputedMetrics(**metrics_data)
