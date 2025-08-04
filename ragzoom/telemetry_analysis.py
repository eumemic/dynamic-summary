"""Telemetry analysis with simplified metrics.

This module provides simplified, actionable metrics focused on:
- Target-fit accuracy
- Retry efficiency
- Latency
- Cost
- Consistency (dispersion)

All metrics are aggregated at the chunk-size level only (no tree-level breakdowns).

Legacy functions are preserved for backward compatibility with telemetry_viz.py.
"""

import logging
import os
import statistics
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry_types import (
    AmplificationSummaryDict,
    BatchEfficiencyDict,
    RetryAnalysisDict,
    TelemetryDataDict,
)

logger = logging.getLogger(__name__)

# Current supported telemetry format versions
SUPPORTED_TELEMETRY_VERSIONS = ["1.0", "2.0"]

# Default token estimate for leaf nodes when source tokens are not available
# This is set to 150 tokens (75% of the default 200 token chunk size) as a conservative
# estimate for backward compatibility with old telemetry data that didn't track source tokens.
# The actual chunk size may vary, but this provides a reasonable approximation for cost metrics.
DEFAULT_LEAF_TOKEN_ESTIMATE = 150


# ============================================================================
# NEW SIMPLIFIED METRICS
# ============================================================================


@dataclass
class SimplifiedMetrics:
    """Simplified metrics structure organized by chunk size."""

    metrics_by_chunk_size: dict[int, dict[str, dict[str, float]]]


def compute_simplified_metrics(
    telemetry_data: dict, config: RagZoomConfig
) -> SimplifiedMetrics:
    """Compute simplified metrics from telemetry data.

    Args:
        telemetry_data: Raw telemetry data
        config: Configuration with pricing information

    Returns:
        SimplifiedMetrics object with metrics organized by chunk size
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    # Group nodes by target chunk size
    nodes_by_target: dict[int, list[dict[Any, Any]]] = {}

    for doc_type, doc_data in parsed_data["documents"].items():
        nodes = doc_data.get("nodes", [])

        node: Any
        for node in nodes:
            # Skip leaf nodes (no summaries)
            height = node.get("height", node.get("level", 0))
            if height == 0:
                continue

            # Get the accepted summary attempt to find target size
            summary_attempts = node.get("summary_attempts", [])
            for attempt in summary_attempts:
                if attempt.get("status") == "accepted":
                    target_tokens = attempt.get("target_tokens", 0)
                    if target_tokens > 0:
                        if target_tokens not in nodes_by_target:
                            nodes_by_target[target_tokens] = []
                        nodes_by_target[target_tokens].append(node)
                    break

    # Compute metrics for each chunk size
    metrics_by_chunk_size = {}
    for target_size in sorted(nodes_by_target.keys()):
        chunk_nodes: list[dict[Any, Any]] = nodes_by_target[target_size]
        if chunk_nodes:
            metrics_by_chunk_size[target_size] = {
                "target_fit": compute_target_fit_metrics(chunk_nodes, target_size),
                "retries": compute_retry_metrics(chunk_nodes),
                "latency": compute_latency_metrics(chunk_nodes),
                "cost": compute_cost_metrics(chunk_nodes, config),
                "dispersion": compute_dispersion_metrics(chunk_nodes),
            }

    return SimplifiedMetrics(metrics_by_chunk_size=metrics_by_chunk_size)


def compute_target_fit_metrics(
    nodes: list[dict[Any, Any]], target_size: int
) -> dict[str, float]:
    """Compute target-fit accuracy metrics.

    Returns:
        - median_error: Median of (actual - target)
        - p95_error: 95th percentile of (actual - target)
        - percent_within_10: Percentage within ±10 tokens
        - max_overshoot: Maximum positive error
        - max_undershoot: Maximum negative error (most negative)
    """
    errors = []
    within_10_count = 0
    max_overshoot = 0
    max_undershoot = 0

    for node in nodes:
        # Find accepted attempt
        for attempt in node.get("summary_attempts", []):
            if attempt.get("status") == "accepted":
                actual_tokens = attempt.get("actual_tokens", 0)
                if actual_tokens > 0:
                    error = actual_tokens - target_size
                    errors.append(error)

                    if abs(error) <= 10:
                        within_10_count += 1

                    if error > max_overshoot:
                        max_overshoot = error
                    if error < max_undershoot:
                        max_undershoot = error
                break

    if not errors:
        return {
            "median_error": 0.0,
            "p95_error": 0.0,
            "percent_within_10": 0.0,
            "max_overshoot": 0.0,
            "max_undershoot": 0.0,
        }

    sorted_errors = sorted(errors)
    median_error = median(sorted_errors)
    p95_error = _compute_percentile(sorted_errors, 0.95)
    percent_within_10 = (within_10_count / len(errors)) * 100

    return {
        "median_error": median_error,
        "p95_error": p95_error,
        "percent_within_10": percent_within_10,
        "max_overshoot": max_overshoot,
        "max_undershoot": max_undershoot,
    }


def compute_retry_metrics(nodes: list[dict[Any, Any]]) -> dict[str, float]:
    """Compute retry efficiency metrics.

    Returns:
        - retry_rate: Average extra attempts per node ((total_attempts - num_nodes) / num_nodes)
        - max_retries: Maximum retries on any single node
    """
    total_attempts = 0
    max_retries = 0

    for node in nodes:
        attempts = node.get("summary_attempts", [])
        num_attempts = len(attempts)
        total_attempts += num_attempts

        # Retries = attempts - 1 (first attempt is not a retry)
        node_retries = max(0, num_attempts - 1)
        if node_retries > max_retries:
            max_retries = node_retries

    num_nodes = len(nodes)
    retry_rate = ((total_attempts - num_nodes) / num_nodes) if num_nodes > 0 else 0.0

    return {
        "retry_rate": retry_rate,
        "max_retries": max_retries,
    }


def compute_latency_metrics(nodes: list[dict[Any, Any]]) -> dict[str, float]:
    """Compute latency metrics.

    Returns:
        - median_seconds: Median wall-clock time per accepted summary (including retries)
        - p95_seconds: 95th percentile latency
        - total_indexing_seconds: Total time for all nodes
    """
    node_times = []
    total_time = 0.0

    for node in nodes:
        # Calculate total time for this node (all attempts)
        node_start = float("inf")
        node_end = 0.0

        for attempt in node.get("summary_attempts", []):
            start_time = attempt.get("start_time", 0)
            end_time = attempt.get("end_time", 0)

            if start_time > 0:
                node_start = min(node_start, start_time)
            if end_time > 0:
                node_end = max(node_end, end_time)

        if node_start != float("inf") and node_end > node_start:
            node_time = node_end - node_start
            node_times.append(node_time)
            total_time += node_time

    if not node_times:
        return {
            "median_seconds": 0.0,
            "p95_seconds": 0.0,
            "total_indexing_seconds": 0.0,
        }

    sorted_times = sorted(node_times)
    median_seconds = median(sorted_times)
    p95_seconds = _compute_percentile(sorted_times, 0.95)

    return {
        "median_seconds": median_seconds,
        "p95_seconds": p95_seconds,
        "total_indexing_seconds": total_time,
    }


def compute_cost_metrics(
    nodes: list[dict[Any, Any]], config: RagZoomConfig
) -> dict[str, float]:
    """Compute cost and token metrics.

    Returns:
        - total_prompt_tokens: Total prompt tokens across all nodes
        - total_completion_tokens: Total completion tokens
        - total_tokens: Sum of prompt and completion
        - usd_per_node: Average cost per node (including embeddings and summaries)
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_embedding_tokens = 0

    for node in nodes:
        # Count embedding tokens
        embedding = node.get("embedding")
        if embedding:
            total_embedding_tokens += embedding.get("text_tokens", 0)

        # Count summary tokens (all attempts)
        for attempt in node.get("summary_attempts", []):
            total_prompt_tokens += attempt.get("prompt_tokens", 0)
            total_completion_tokens += attempt.get("completion_tokens", 0)

    total_tokens = (
        total_prompt_tokens + total_completion_tokens + total_embedding_tokens
    )

    # Calculate costs
    embedding_cost = (total_embedding_tokens / 1000) * config.embedding_cost_per_1k
    prompt_cost = (total_prompt_tokens / 1000) * config.summary_input_cost_per_1k
    completion_cost = (
        total_completion_tokens / 1000
    ) * config.summary_output_cost_per_1k
    total_cost = embedding_cost + prompt_cost + completion_cost

    num_nodes = len(nodes)
    usd_per_node = (total_cost / num_nodes) if num_nodes > 0 else 0.0

    return {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "usd_per_node": usd_per_node,
    }


def compute_dispersion_metrics(nodes: list[dict[Any, Any]]) -> dict[str, float]:
    """Compute dispersion metrics.

    Returns:
        - mad: Median Absolute Deviation of actual token counts
    """
    actual_tokens = []

    for node in nodes:
        # Find accepted attempt
        for attempt in node.get("summary_attempts", []):
            if attempt.get("status") == "accepted":
                tokens = attempt.get("actual_tokens", 0)
                if tokens > 0:
                    actual_tokens.append(tokens)
                break

    if not actual_tokens:
        return {"mad": 0.0}

    # Calculate MAD: median(|x_i - median(x)|)
    median_tokens = median(actual_tokens)
    absolute_deviations = [abs(x - median_tokens) for x in actual_tokens]
    mad = median(absolute_deviations)

    return {"mad": mad}


def _compute_percentile(values: list[float], percentile: float) -> float:
    """Compute percentile using linear interpolation.

    Args:
        values: Sorted list of values
        percentile: Percentile to compute (0.0 to 1.0)

    Returns:
        Computed percentile value
    """
    if not values:
        return 0.0

    n = len(values)
    if n == 1:
        return values[0]

    pos = (n - 1) * percentile
    lower = int(pos)
    upper = min(lower + 1, n - 1)
    fraction = pos - lower

    return values[lower] + fraction * (values[upper] - values[lower])


# ============================================================================
# LEGACY FUNCTIONS FOR BACKWARD COMPATIBILITY
# ============================================================================


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


def parse_telemetry_format(telemetry_data: dict) -> TelemetryDataDict:
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

    # Return as TypedDict - cast is safe after validation
    result: TelemetryDataDict = {
        "format_version": format_version,
        "documents": documents,
    }
    return result


def compute_amplification_metrics(
    telemetry_data: dict, config: RagZoomConfig
) -> AmplificationSummaryDict:
    """Compute amplification metrics from telemetry data.

    Args:
        telemetry_data: Parsed telemetry data
        config: Configuration for cost calculations

    Returns:
        Dictionary containing amplification metrics
    """
    # Use compute_metrics_from_telemetry as the single source of truth
    metrics = compute_metrics_from_telemetry(telemetry_data, config)

    # Extract summary statistics from the computed metrics
    return get_amplification_summary(metrics)


def _calculate_cost_amplification(
    total_prompt_tokens: int,
    total_completion_tokens: int,
    input_text_tokens: int,
    final_summary_tokens: int,
    config: RagZoomConfig,
) -> float:
    """Calculate cost-weighted amplification factor.

    Cost amplification = (actual cost / theoretical minimum cost)

    Measures the cost inefficiency from prompt overhead and retry attempts.
    The theoretical minimum represents sending just the raw text and receiving
    the final summary in a single attempt.

    Args:
        total_prompt_tokens: Total prompt tokens across all attempts
        total_completion_tokens: Total completion tokens across all attempts
        input_text_tokens: Tokens in the original text being summarized
        final_summary_tokens: Tokens in the final accepted summary
        config: Configuration with pricing information

    Returns:
        Cost amplification factor (1.0 = no amplification)
    """
    # Calculate actual cost (what we paid across all attempts)
    actual_cost = (
        total_prompt_tokens * config.summary_input_cost_per_1k
        + total_completion_tokens * config.summary_output_cost_per_1k
    ) / 1000

    # Calculate theoretical minimum cost (input text → final summary in one shot)
    min_cost = (
        input_text_tokens * config.summary_input_cost_per_1k
        + final_summary_tokens * config.summary_output_cost_per_1k
    ) / 1000

    # Return amplification factor with zero check
    return actual_cost / min_cost if min_cost > 0 else 1.0


def compute_batch_efficiency(telemetry_data: dict) -> BatchEfficiencyDict:
    """Analyze embedding batch utilization from telemetry data.

    Args:
        telemetry_data: Parsed telemetry data

    Returns:
        Dictionary containing batch efficiency metrics
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
    result: BatchEfficiencyDict = {
        "avg_batch_size": 0.0,
        "batch_sizes": batch_sizes,  # Legacy field
        "total_batches": len(batch_sizes),  # Legacy field
        "total_embeddings": total_embeddings,
        "batch_utilization": 0.0,
        "batched_embeddings": 0,  # Will be calculated below
        "single_embeddings": 0,  # Will be calculated below
        "max_batch_size": max(batch_sizes) if batch_sizes else 0,
        "batch_size_distribution": {},  # Will be calculated below
    }

    if batch_sizes:
        avg_batch_size = sum(batch_sizes) / len(batch_sizes)
        result["avg_batch_size"] = avg_batch_size

        # Calculate additional metrics
        batched_embeddings = sum(max(0, batch_size - 1) for batch_size in batch_sizes)
        single_embeddings = sum(1 for batch_size in batch_sizes if batch_size == 1)

        result["batched_embeddings"] = batched_embeddings
        result["single_embeddings"] = single_embeddings
        result["batch_utilization"] = (
            (batched_embeddings / total_embeddings) * 100
            if total_embeddings > 0
            else 0.0
        )

        # Calculate batch size distribution
        from collections import Counter

        result["batch_size_distribution"] = dict(Counter(batch_sizes))

    return result


def analyze_retry_patterns(telemetry_data: dict) -> RetryAnalysisDict:
    """Analyze summary retry patterns from telemetry data.

    Args:
        telemetry_data: Parsed telemetry data

    Returns:
        Dictionary containing retry analysis
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    total_attempts = 0
    successful_attempts = 0
    retry_attempts = 0
    successful_retries = 0
    rejection_reasons: dict[str, int] = {}
    nodes_with_retries = 0
    total_nodes_with_summaries = 0

    # New metrics for retry distribution and timing
    retry_distribution: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}  # 3 means 3+
    max_retries = 0
    total_retry_time = 0.0
    total_rejected_time = 0.0
    total_retries_per_node = 0

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
            node_retry_count = 0
            node_retry_time = 0.0
            node_rejected_time = 0.0

            for attempt_idx, attempt in enumerate(summary_attempts):
                total_attempts += 1
                status = attempt.get("status", "unknown")

                # Calculate time for this attempt if available
                start_time = attempt.get("start_time", 0)
                end_time = attempt.get("end_time", 0)
                attempt_time = end_time - start_time if end_time > start_time else 0

                # In v2 format: first attempt (index 0) is initial, rest are retries
                # In v1 format: check the is_retry field for backward compatibility
                is_retry = attempt.get("is_retry", attempt_idx > 0)

                if is_retry:
                    retry_attempts += 1
                    node_has_retry = True
                    node_retry_count += 1
                    node_retry_time += attempt_time
                    if status == "accepted":
                        successful_retries += 1
                    else:
                        node_rejected_time += attempt_time

                if status == "accepted":
                    successful_attempts += 1
                elif status in ["rejected_over", "rejected_under", "error"]:
                    reason = attempt.get("rejection_reason", status)
                    if reason:  # Type guard to ensure reason is not None
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                    if not is_retry:  # Also count rejected time for initial attempts
                        node_rejected_time += attempt_time

            if node_has_retry:
                nodes_with_retries += 1

            # Track retry distribution and totals
            retry_bucket = min(node_retry_count, 3)  # Cap at 3+ for distribution
            retry_distribution[retry_bucket] += 1
            max_retries = max(max_retries, node_retry_count)
            total_retries_per_node += node_retry_count
            total_retry_time += node_retry_time
            total_rejected_time += node_rejected_time

    # Calculate metrics
    result: RetryAnalysisDict = {
        "retry_rate": 0.0,
        "total_attempts": total_attempts,
        "successful_attempts": successful_attempts,
        "retry_attempts": retry_attempts,
        "retry_success_rate": 0.0,
        "rejection_reasons": rejection_reasons,
        "nodes_with_retries": nodes_with_retries,
        "total_nodes_with_summaries": total_nodes_with_summaries,
        # New distribution metrics
        "retry_distribution": {
            "0": retry_distribution[0],
            "1": retry_distribution[1],
            "2": retry_distribution[2],
            "3+": retry_distribution[3],
        },
        "avg_retries_per_node": 0.0,
        "max_retries": max_retries,
        # New timing metrics
        "retry_time_seconds": total_retry_time,
        "avg_time_per_retry": 0.0,
        "time_wasted_on_rejections": total_rejected_time,
    }

    if total_nodes_with_summaries > 0:
        result["retry_rate"] = (nodes_with_retries / total_nodes_with_summaries) * 100
        result["avg_retries_per_node"] = (
            total_retries_per_node / total_nodes_with_summaries
        )

    if retry_attempts > 0:
        result["retry_success_rate"] = (successful_retries / retry_attempts) * 100
        result["avg_time_per_retry"] = total_retry_time / retry_attempts

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


def get_amplification_summary(metrics: ComputedMetrics) -> AmplificationSummaryDict:
    """Compute amplification summary statistics from raw metrics.

    Args:
        metrics: ComputedMetrics object containing raw amplification data

    Returns:
        Dictionary containing amplification summary statistics
    """
    result: AmplificationSummaryDict = {
        "median_cost": 0.0,
        "cost_p90": 0.0,
        "cost_p95": 0.0,
        "median_input": 0.0,
        "median_output": 0.0,
        "by_height": {
            height: {
                "median_cost": median(data["cost"]) if data["cost"] else 0.0,
                "median_input": median(data["input"]) if data["input"] else 0.0,
                "median_output": median(data["output"]) if data["output"] else 0.0,
                "count": len(data["cost"]),
                # Legacy field names for backward compatibility
                "cost": median(data["cost"]) if data["cost"] else 0.0,
                "input": median(data["input"]) if data["input"] else 0.0,
                "output": median(data["output"]) if data["output"] else 0.0,
            }
            for height, data in metrics.amplifications_by_height.items()
        },
    }

    if metrics.cost_amplifications:
        result["median_cost"] = median(metrics.cost_amplifications)
        result["cost_p90"] = _compute_percentile(metrics.cost_amplifications, 0.9)
        result["cost_p95"] = _compute_percentile(metrics.cost_amplifications, 0.95)

    if metrics.input_amplifications:
        result["median_input"] = median(metrics.input_amplifications)

    if metrics.output_amplifications:
        result["median_output"] = median(metrics.output_amplifications)

    return result


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
            if (
                summary_attempts
                and node_input_text_tokens > 0
                and final_attempt
                and node_final_summary_tokens > 0  # Ensure we have valid final tokens
            ):
                # Input amplification = total prompt tokens / original text tokens
                input_amplification = node_total_prompt_tokens / node_input_text_tokens

                # Output amplification = total completion tokens / final summary tokens
                output_amplification = (
                    node_total_completion_tokens / node_final_summary_tokens
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

                # Track amplifications by height
                # Need height - already computed earlier for leaf check
                node_height = node.get("height", node.get("level", 0))
                if node_height not in metrics_data["amplifications_by_height"]:
                    metrics_data["amplifications_by_height"][node_height] = {
                        "input": [],
                        "output": [],
                        "cost": [],
                    }
                metrics_data["amplifications_by_height"][node_height]["input"].append(
                    input_amplification
                )
                metrics_data["amplifications_by_height"][node_height]["output"].append(
                    output_amplification
                )
                metrics_data["amplifications_by_height"][node_height]["cost"].append(
                    cost_amplification
                )

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
