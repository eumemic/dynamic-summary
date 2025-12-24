"""Telemetry analysis with simplified metrics.

This module provides simplified, actionable metrics focused on:
- Target-fit accuracy
- Retry efficiency
- Latency
- Cost
- Consistency (dispersion)

All metrics are aggregated at the chunk-size level only (no tree-level breakdowns).

Simplified telemetry analysis for format 4.2 only.
"""

import logging
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import numpy as np
from typing_extensions import TypedDict

from ragzoom.config import get_embedding_cost, get_llm_costs
from ragzoom.cost import (
    calculate_completion_cost,
    calculate_embedding_cost,
    calculate_prompt_cost_with_cache,
)
from ragzoom.telemetry_query import QueryPhaseMetrics
from ragzoom.telemetry_types import (
    BatchEfficiencyDict,
    NodeTelemetryDict,
    RetryAnalysisDict,
    SummaryAttemptDict,
    TelemetryDataDict,
)

logger = logging.getLogger(__name__)

# Current supported telemetry format version
SUPPORTED_TELEMETRY_VERSION = "4.3"


def _calculate_mad_and_std(values: Sequence[float | int]) -> tuple[float, float]:
    """Calculate Median Absolute Deviation (MAD) and standard deviation.

    Args:
        values: List of numeric values

    Returns:
        Tuple of (mad, std_dev)
    """
    if not values:
        return 0.0, 0.0

    median_val = median(values)
    absolute_deviations = [abs(v - median_val) for v in values]
    mad = median(absolute_deviations)

    if len(values) > 1:
        std_dev = statistics.stdev(values)
    else:
        std_dev = 0.0

    return mad, std_dev


def _calculate_percentage_deviation(error: float, target_size: int) -> float:
    """Calculate percentage deviation with edge case handling.

    Args:
        error: Signed error (actual - target)
        target_size: Target size value

    Returns:
        Percentage deviation as float
    """
    if target_size == 0:
        # Edge case: if target is 0, return a large deviation to indicate the issue
        return 100.0 if error != 0 else 0.0

    return abs(error) / target_size * 100


# Default token estimate for leaf nodes when source tokens are not available
# This is set to 150 tokens (75% of the default 200 token chunk size) as a conservative
# estimate when source tokens are missing.
# The actual chunk size may vary, but this provides a reasonable approximation for cost metrics.
DEFAULT_LEAF_TOKEN_ESTIMATE = 150


# ============================================================================
# PRICING UTILITIES
# ============================================================================


def get_model_pricing(summary_model: str, embedding_model: str) -> dict[str, float]:
    """Get pricing for specific models from pricing constants.

    Args:
        summary_model: Name of the LLM model
        embedding_model: Name of the embedding model

    Returns:
        Dictionary with pricing information:
        - summary_input_cost_per_1k: Cost per 1K input tokens
        - summary_output_cost_per_1k: Cost per 1K output tokens
        - embedding_cost_per_1k: Cost per 1K embedding tokens
    """
    embedding_cost = get_embedding_cost(embedding_model)
    summary_input_cost, summary_output_cost = get_llm_costs(summary_model)

    return {
        "summary_input_cost_per_1k": summary_input_cost,
        "summary_output_cost_per_1k": summary_output_cost,
        "embedding_cost_per_1k": embedding_cost,
    }


# ============================================================================
# NEW SIMPLIFIED METRICS
# ============================================================================
# The functions below are the new simplified telemetry metrics system.
# They focus on actionable insights at the chunk-size level only.
# ============================================================================


class VerbatimOffender(TypedDict):
    """A node that was detected as verbatim concatenation."""

    node_id: str
    height: int
    input_tokens: int
    output_tokens: int
    ratio: float


class VerbatimDetectionResult(TypedDict):
    """Result of verbatim concatenation detection."""

    total_summaries: int
    verbatim_count: int
    verbatim_percentage: float
    worst_offenders: list[VerbatimOffender]
    height_distribution: dict[int, int]


@dataclass
class TargetFitMetrics:
    """Metrics for target-fit accuracy."""

    # Signed error metrics (existing)
    median_error: float
    p95_error: float
    percent_within_10: float
    max_overshoot: float
    max_undershoot: float
    error_mad: float
    error_iqr: float
    error_std: float
    percent_within_10_mad: float
    percent_within_10_std: float

    # Absolute deviation metrics (new for clearer regression detection)
    mean_absolute_error: float  # More sensitive to outliers than median
    median_absolute_error: float  # Robust to outliers
    absolute_error_mad: float  # Variance for absolute errors
    absolute_error_std: float  # Standard deviation for absolute errors

    # Percentage-based metrics (chunk-size invariant)
    mean_percent_deviation: float  # Mean of |actual - target| / target * 100
    median_percent_deviation: float  # Median of |actual - target| / target * 100
    percent_deviation_mad: float  # MAD of percentage deviations
    percent_deviation_std: float  # Standard deviation of percentage deviations

    # Acceptance distribution metrics (multiple thresholds for better insight)
    percent_within_5: float  # Percentage within ±5 tokens
    percent_within_20: float  # Percentage within ±20 tokens
    percent_within_50: float  # Percentage within ±50 tokens
    # Note: percent_within_10 already exists above


@dataclass
class RetryMetrics:
    """Metrics for retry patterns."""

    retry_rate: float
    oversized_summary_rate: (
        float  # Percentage of nodes that produced oversized summaries
    )
    max_retries: float
    retry_mad: float
    retry_std: float


@dataclass
class LatencyMetrics:
    """Metrics for processing latency."""

    median_seconds: float
    p95_seconds: float
    total_indexing_seconds: float
    latency_mad: float
    latency_iqr: float
    latency_std: float


@dataclass
class CostMetrics:
    """Metrics for cost analysis."""

    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    usd_per_node: float
    usd_per_million_source_tokens: float
    cost_mad: float
    cost_iqr: float
    cost_std: float


@dataclass
class DispersionMetrics:
    """Metrics for data dispersion."""

    mad: float
    iqr: float
    p25: float
    p75: float
    cv: float
    std: float


@dataclass
class ChunkMetrics:
    """All metrics for a specific chunk size."""

    target_fit: TargetFitMetrics
    retries: RetryMetrics
    latency: LatencyMetrics
    cost: CostMetrics
    dispersion: DispersionMetrics
    pipeline_efficiency: float
    fidelity: "FidelityMetrics | None"


@dataclass
class SimplifiedMetrics:
    """Simplified metrics structure organized by chunk size."""

    metrics_by_chunk_size: dict[int, ChunkMetrics]


class FidelityWorstNode(TypedDict):
    """Details about a low-fidelity merge."""

    node_id: str | None
    height: int | None
    span: tuple[int, int] | None
    fidelity: float


@dataclass
class FidelityMetrics:
    """Summarization fidelity metrics."""

    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    stddev: float
    worst_nodes: list[FidelityWorstNode]


def compute_simplified_metrics(telemetry_data: TelemetryDataDict) -> SimplifiedMetrics:
    """Compute simplified metrics from telemetry data.

    Args:
        telemetry_data: Raw telemetry data

    Returns:
        SimplifiedMetrics object with metrics organized by chunk size
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    # Extract models from config for accurate pricing
    config = telemetry_data.get("config", {})
    embedding_model = config.get("embedding_model")
    summary_model = config.get("summary_model")

    if not embedding_model or not summary_model:
        raise ValueError(
            "Telemetry data missing model information in config. "
            "Cannot compute cost metrics without knowing which models were used."
        )

    # Extract source document tokens for cost calculations
    source_document_tokens = parsed_data.get("source_document_tokens", 0)

    # Group nodes by target chunk size
    nodes_by_target: dict[int, list[NodeTelemetryDict]] = {}

    # Nodes are at the top level
    nodes = parsed_data.get("nodes", [])

    for node in nodes:
        # Skip leaf nodes (no summaries)
        height = node["height"]
        if height == 0:
            continue

        # Get the accepted summary attempt to find target size
        summary_attempts = node.get("summary_attempts", [])
        if not summary_attempts:
            continue

        # Determine which attempt was accepted
        accepted_idx = node.get("accepted_attempt")
        final_attempt = None

        if accepted_idx is not None:
            # Format 4.2: use the explicitly marked accepted attempt
            if 0 <= accepted_idx < len(summary_attempts):
                final_attempt = summary_attempts[accepted_idx]
        else:
            # Fallback: use last attempt if accepted_attempt index is missing
            final_attempt = summary_attempts[-1]

        if final_attempt:
            target_tokens = final_attempt.get("target_tokens", 0)
            if target_tokens > 0:
                if target_tokens not in nodes_by_target:
                    nodes_by_target[target_tokens] = []
                nodes_by_target[target_tokens].append(node)

    # Compute metrics for each chunk size
    metrics_by_chunk_size = {}
    for target_size in sorted(nodes_by_target.keys()):
        chunk_nodes = nodes_by_target[target_size]
        if chunk_nodes:
            metrics_by_chunk_size[target_size] = ChunkMetrics(
                target_fit=compute_target_fit_metrics(chunk_nodes, target_size),
                retries=compute_retry_metrics(chunk_nodes),
                latency=compute_latency_metrics(chunk_nodes),
                cost=compute_cost_metrics(
                    chunk_nodes,
                    {"summary": summary_model, "embedding": embedding_model},
                    source_document_tokens,
                ),
                dispersion=compute_dispersion_metrics(chunk_nodes),
                pipeline_efficiency=calculate_pipeline_efficiency(telemetry_data),
                fidelity=compute_fidelity_metrics(chunk_nodes),
            )

    return SimplifiedMetrics(metrics_by_chunk_size=metrics_by_chunk_size)


def compute_target_fit_metrics(
    nodes: list[NodeTelemetryDict], target_size: int
) -> TargetFitMetrics:
    """Compute target-fit accuracy metrics.

    Returns:
        - median_error: Median of (actual - target)
        - p95_error: 95th percentile of (actual - target)
        - percent_within_10: Percentage within ±10 tokens
        - max_overshoot: Maximum positive error
        - max_undershoot: Maximum negative error (most negative)
        - error_mad: MAD of errors (for dynamic thresholds)
        - error_iqr: IQR of errors
        - error_std: Standard deviation of errors
    """
    errors = []
    within_5_count = 0
    within_10_count = 0
    within_20_count = 0
    within_50_count = 0
    max_overshoot = 0
    max_undershoot = 0
    # Track per-node within_10 status for variance calculation
    node_within_10_list = []

    for node in nodes:
        # Find the accepted attempt using the helper function
        accepted_attempt, _ = get_accepted_attempt(node)
        if accepted_attempt:
            actual_tokens = accepted_attempt.get("actual_tokens", 0)
            if actual_tokens > 0:
                error = actual_tokens - target_size
                errors.append(error)

                abs_error = abs(error)
                is_within_5 = abs_error <= 5
                is_within_10 = abs_error <= 10
                is_within_20 = abs_error <= 20
                is_within_50 = abs_error <= 50

                if is_within_5:
                    within_5_count += 1
                if is_within_10:
                    within_10_count += 1
                if is_within_20:
                    within_20_count += 1
                if is_within_50:
                    within_50_count += 1

                # Store 1 if within ±10, 0 if not (for variance calculation)
                node_within_10_list.append(1.0 if is_within_10 else 0.0)

                if error > max_overshoot:
                    max_overshoot = error
                if error < max_undershoot:
                    max_undershoot = error

    if not errors:
        return TargetFitMetrics(
            median_error=0.0,
            p95_error=0.0,
            percent_within_10=0.0,
            max_overshoot=0.0,
            max_undershoot=0.0,
            error_mad=0.0,
            error_iqr=0.0,
            error_std=0.0,
            percent_within_10_mad=0.0,
            percent_within_10_std=0.0,
            mean_absolute_error=0.0,
            median_absolute_error=0.0,
            absolute_error_mad=0.0,
            absolute_error_std=0.0,
            mean_percent_deviation=0.0,
            median_percent_deviation=0.0,
            percent_deviation_mad=0.0,
            percent_deviation_std=0.0,
            percent_within_5=0.0,
            percent_within_20=0.0,
            percent_within_50=0.0,
        )

    sorted_errors = sorted(errors)
    median_error = median(sorted_errors)
    p95_error = float(np.percentile(sorted_errors, 95))
    percent_within_5 = (within_5_count / len(errors)) * 100
    percent_within_10 = (within_10_count / len(errors)) * 100
    percent_within_20 = (within_20_count / len(errors)) * 100
    percent_within_50 = (within_50_count / len(errors)) * 100

    # Calculate variance metrics for errors
    absolute_deviations = [abs(e - median_error) for e in errors]
    error_mad = median(absolute_deviations)

    p25_error = float(np.percentile(sorted_errors, 25))
    p75_error = float(np.percentile(sorted_errors, 75))
    error_iqr = p75_error - p25_error

    if len(errors) > 1:
        error_std = statistics.stdev(errors)
    else:
        error_std = 0.0

    # Calculate variance metrics for percent_within_10
    if node_within_10_list:
        # Convert binary values (0s and 1s) to percentages once
        percent_within_10_values = [w * 100 for w in node_within_10_list]
        percent_within_10_mad, percent_within_10_std = _calculate_mad_and_std(
            percent_within_10_values
        )
    else:
        percent_within_10_mad = 0.0
        percent_within_10_std = 0.0

    # Calculate absolute deviation metrics
    absolute_errors = [abs(error) for error in errors]
    mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
    median_absolute_error = median(absolute_errors)
    absolute_error_mad, absolute_error_std = _calculate_mad_and_std(absolute_errors)

    # Calculate percentage-based metrics (chunk-size invariant)
    percent_deviations = [
        _calculate_percentage_deviation(error, target_size) for error in errors
    ]
    mean_percent_deviation = sum(percent_deviations) / len(percent_deviations)
    median_percent_deviation = median(percent_deviations)
    percent_deviation_mad, percent_deviation_std = _calculate_mad_and_std(
        percent_deviations
    )

    return TargetFitMetrics(
        median_error=median_error,
        p95_error=p95_error,
        percent_within_10=percent_within_10,
        max_overshoot=max_overshoot,
        max_undershoot=max_undershoot,
        error_mad=error_mad,
        error_iqr=error_iqr,
        error_std=error_std,
        percent_within_10_mad=percent_within_10_mad,
        percent_within_10_std=percent_within_10_std,
        mean_absolute_error=mean_absolute_error,
        median_absolute_error=median_absolute_error,
        absolute_error_mad=absolute_error_mad,
        absolute_error_std=absolute_error_std,
        mean_percent_deviation=mean_percent_deviation,
        median_percent_deviation=median_percent_deviation,
        percent_deviation_mad=percent_deviation_mad,
        percent_deviation_std=percent_deviation_std,
        percent_within_5=percent_within_5,
        percent_within_20=percent_within_20,
        percent_within_50=percent_within_50,
    )


def compute_retry_metrics(nodes: list[NodeTelemetryDict]) -> RetryMetrics:
    """Compute retry efficiency metrics.

    Returns:
        - retry_rate: Average extra attempts per node ((total_attempts - num_nodes) / num_nodes)
        - max_retries: Maximum retries on any single node
        - retry_mad: MAD of per-node retry counts (for dynamic thresholds)
        - retry_iqr: IQR of per-node retry counts
        - retry_std: Standard deviation of per-node retry counts
    """
    total_attempts = 0
    max_retries = 0
    node_retries_list = []

    for node in nodes:
        attempts = node.get("summary_attempts", [])
        num_attempts = len(attempts)
        total_attempts += num_attempts

        # Retries = attempts - 1 (first attempt is not a retry)
        node_retries = max(0, num_attempts - 1)
        node_retries_list.append(node_retries)
        if node_retries > max_retries:
            max_retries = node_retries

    num_nodes = len(nodes)
    retry_rate = ((total_attempts - num_nodes) / num_nodes) if num_nodes > 0 else 0.0

    # Calculate rejection rate - percentage of nodes that needed retries
    nodes_with_retries = sum(1 for retries in node_retries_list if retries > 0)
    rejection_rate = (nodes_with_retries / num_nodes * 100) if num_nodes > 0 else 0.0

    # Compute variance metrics for per-node retry counts
    if node_retries_list:
        median_retries = median(node_retries_list)
        absolute_deviations = [abs(r - median_retries) for r in node_retries_list]
        retry_mad = median(absolute_deviations)

        if len(node_retries_list) > 1:
            retry_std = statistics.stdev(node_retries_list)
        else:
            retry_std = 0.0
    else:
        retry_mad = 0.0
        retry_std = 0.0

    return RetryMetrics(
        retry_rate=retry_rate,
        oversized_summary_rate=rejection_rate,
        max_retries=float(max_retries),
        retry_mad=retry_mad,
        retry_std=retry_std,
    )


def compute_latency_metrics(nodes: list[NodeTelemetryDict]) -> LatencyMetrics:
    """Compute latency metrics.

    Returns:
        - median_seconds: Median wall-clock time per accepted summary (including retries)
        - p95_seconds: 95th percentile latency
        - total_indexing_seconds: Total time for all nodes
        - latency_mad: MAD of latencies (for dynamic thresholds)
        - latency_iqr: IQR of latencies
        - latency_std: Standard deviation of latencies
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
        return LatencyMetrics(
            median_seconds=0.0,
            p95_seconds=0.0,
            total_indexing_seconds=0.0,
            latency_mad=0.0,
            latency_iqr=0.0,
            latency_std=0.0,
        )

    sorted_times = sorted(node_times)
    median_seconds = median(sorted_times)
    p95_seconds = float(np.percentile(sorted_times, 95))

    # Calculate variance metrics for latency
    absolute_deviations = [abs(t - median_seconds) for t in node_times]
    latency_mad = median(absolute_deviations)

    p25_latency = float(np.percentile(sorted_times, 25))
    p75_latency = float(np.percentile(sorted_times, 75))
    latency_iqr = p75_latency - p25_latency

    if len(node_times) > 1:
        latency_std = statistics.stdev(node_times)
    else:
        latency_std = 0.0

    return LatencyMetrics(
        median_seconds=median_seconds,
        p95_seconds=p95_seconds,
        total_indexing_seconds=total_time,
        latency_mad=latency_mad,
        latency_iqr=latency_iqr,
        latency_std=latency_std,
    )


def compute_cost_metrics(
    nodes: list[NodeTelemetryDict], models: dict[str, str], source_document_tokens: int
) -> CostMetrics:
    """Compute cost and token metrics.

    Args:
        nodes: List of node telemetry data
        models: Dictionary with 'summary' and 'embedding' model names

    Returns:
        - total_prompt_tokens: Total prompt tokens across all nodes
        - total_completion_tokens: Total completion tokens
        - total_tokens: Sum of prompt and completion
        - usd_per_node: Average cost per node (including embeddings and summaries)
        - cost_mad: MAD of per-node costs (for dynamic thresholds)
        - cost_iqr: IQR of per-node costs
        - cost_std: Standard deviation of per-node costs
    """
    # Get model-specific pricing
    summary_model = models["summary"]
    embedding_model = models["embedding"]
    pricing = get_model_pricing(summary_model, embedding_model)

    # Get cache discount for the summary model
    try:
        from ragzoom.model_info import ModelInfo

        model_info = ModelInfo()
        cache_discount = model_info.get_cache_discount(summary_model)
    except (ImportError, ValueError):
        # Default to no discount if model info not available
        cache_discount = 0.0

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_embedding_tokens = 0
    total_cached_tokens = 0
    node_costs = []

    for node in nodes:
        node_prompt_tokens = 0
        node_cached_tokens = 0
        node_completion_tokens = 0
        node_embedding_tokens = 0

        # Count embedding tokens
        embedding = node.get("embedding")
        if embedding:
            node_embedding_tokens = embedding.get("text_tokens", 0)
            total_embedding_tokens += node_embedding_tokens

        # Count summary tokens (all attempts) with cache discount
        for attempt in node.get("summary_attempts", []):
            prompt_tokens = attempt.get("prompt_tokens", 0)
            cached_tokens = attempt.get("cached_tokens", 0)
            completion_tokens = attempt.get("completion_tokens", 0)

            node_prompt_tokens += prompt_tokens
            node_cached_tokens += cached_tokens
            node_completion_tokens += completion_tokens

        total_prompt_tokens += node_prompt_tokens
        total_cached_tokens += node_cached_tokens
        total_completion_tokens += node_completion_tokens

        # Calculate per-node cost in USD with cache discount
        embedding_cost = calculate_embedding_cost(
            node_embedding_tokens, pricing["embedding_cost_per_1k"]
        )

        # Apply cache discount: cached tokens cost less
        prompt_cost = calculate_prompt_cost_with_cache(
            node_prompt_tokens,
            node_cached_tokens,
            pricing["summary_input_cost_per_1k"],
            cache_discount,
        )

        completion_cost = calculate_completion_cost(
            node_completion_tokens, pricing["summary_output_cost_per_1k"]
        )
        node_cost = embedding_cost + prompt_cost + completion_cost
        node_costs.append(node_cost)

    total_tokens = (
        total_prompt_tokens + total_completion_tokens + total_embedding_tokens
    )

    # Calculate total costs with cache discount
    embedding_cost = calculate_embedding_cost(
        total_embedding_tokens, pricing["embedding_cost_per_1k"]
    )

    # Apply cache discount to total prompt costs
    prompt_cost = calculate_prompt_cost_with_cache(
        total_prompt_tokens,
        total_cached_tokens,
        pricing["summary_input_cost_per_1k"],
        cache_discount,
    )

    completion_cost = calculate_completion_cost(
        total_completion_tokens, pricing["summary_output_cost_per_1k"]
    )
    total_cost = embedding_cost + prompt_cost + completion_cost

    num_nodes = len(nodes)
    usd_per_node = (total_cost / num_nodes) if num_nodes > 0 else 0.0

    # Calculate USD per 1M source tokens
    usd_per_million_source_tokens = (
        (total_cost / source_document_tokens) * 1_000_000
        if source_document_tokens > 0
        else 0.0
    )

    # Compute variance metrics for per-node costs
    if node_costs:
        median_cost = median(node_costs)
        absolute_deviations = [abs(c - median_cost) for c in node_costs]
        cost_mad = median(absolute_deviations)

        sorted_costs = sorted(node_costs)
        p25_cost = float(np.percentile(sorted_costs, 25))
        p75_cost = float(np.percentile(sorted_costs, 75))
        cost_iqr = p75_cost - p25_cost

        if len(node_costs) > 1:
            cost_std = statistics.stdev(node_costs)
        else:
            cost_std = 0.0
    else:
        cost_mad = 0.0
        cost_iqr = 0.0
        cost_std = 0.0

    return CostMetrics(
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        usd_per_node=usd_per_node,
        usd_per_million_source_tokens=usd_per_million_source_tokens,
        cost_mad=cost_mad,
        cost_iqr=cost_iqr,
        cost_std=cost_std,
    )


def compute_dispersion_metrics(nodes: list[NodeTelemetryDict]) -> DispersionMetrics:
    """Compute comprehensive dispersion metrics.

    Returns:
        - mad: Median Absolute Deviation of actual token counts
        - iqr: Interquartile range (75th - 25th percentile)
        - p25: 25th percentile
        - p75: 75th percentile
        - cv: Coefficient of variation (std/mean)
        - std: Standard deviation
    """
    actual_tokens = []

    for node in nodes:
        # Find the accepted attempt using the helper function
        accepted_attempt, _ = get_accepted_attempt(node)
        if accepted_attempt:
            tokens = accepted_attempt.get("actual_tokens", 0)
            if tokens > 0:
                actual_tokens.append(tokens)

    if not actual_tokens:
        return DispersionMetrics(
            mad=0.0,
            iqr=0.0,
            p25=0.0,
            p75=0.0,
            cv=0.0,
            std=0.0,
        )

    # Calculate MAD: median(|x_i - median(x)|)
    median_tokens = median(actual_tokens)
    absolute_deviations = [abs(x - median_tokens) for x in actual_tokens]
    mad = median(absolute_deviations)

    # Calculate additional metrics
    sorted_tokens = sorted(actual_tokens)
    p25 = float(np.percentile(sorted_tokens, 25))
    p75 = float(np.percentile(sorted_tokens, 75))
    iqr = p75 - p25

    # Calculate CV (coefficient of variation)
    mean_tokens = sum(actual_tokens) / len(actual_tokens)
    if mean_tokens > 0 and len(actual_tokens) > 1:
        std = statistics.stdev(actual_tokens)
        cv = std / mean_tokens
    else:
        std = 0.0
        cv = 0.0

    return DispersionMetrics(
        mad=mad,
        iqr=iqr,
        p25=p25,
        p75=p75,
        cv=cv,
        std=std,
    )


def compute_fidelity_metrics(
    nodes: Sequence[NodeTelemetryDict],
) -> FidelityMetrics:
    """Compute summarization fidelity metrics using stored scalar values."""

    values: list[tuple[NodeTelemetryDict, float]] = []
    for node in nodes:
        fidelity_value = node.get("fidelity")
        if fidelity_value is None:
            continue
        values.append((node, float(fidelity_value)))

    count = len(values)
    if count == 0:
        return FidelityMetrics(
            count=0,
            mean=0.0,
            median=0.0,
            minimum=0.0,
            maximum=0.0,
            stddev=0.0,
            worst_nodes=[],
        )

    fidelity_values = [value for _, value in values]
    mean_val = statistics.fmean(fidelity_values)
    median_val = median(fidelity_values)
    min_val = min(fidelity_values)
    max_val = max(fidelity_values)
    std_val = statistics.pstdev(fidelity_values) if count > 1 else 0.0

    values.sort(key=lambda item: item[1])
    worst_nodes = []
    for node, value in values[:5]:
        worst_nodes.append(
            FidelityWorstNode(
                node_id=node.get("node_id"),
                height=node.get("height"),
                span=node.get("span"),
                fidelity=value,
            )
        )

    return FidelityMetrics(
        count=count,
        mean=mean_val,
        median=median_val,
        minimum=min_val,
        maximum=max_val,
        stddev=std_val,
        worst_nodes=worst_nodes,
    )


def calculate_pipeline_efficiency(telemetry_data: TelemetryDataDict) -> float:
    """Calculate pipeline efficiency as a percentage of parallelism utilization.

    Pipeline Efficiency = (sequential_duration - actual_duration) /
                         (sequential_duration - max_parallel_duration) × 100%

    Args:
        telemetry_data: Raw telemetry data

    Returns:
        Pipeline efficiency percentage (0.0-100.0). Higher values indicate
        better parallelism utilization.
    """
    parsed_data = parse_telemetry_format(telemetry_data)
    nodes = parsed_data.get("nodes", [])

    if not nodes:
        return 0.0

    # Calculate wall clock time and collect unique operation durations
    min_time = float("inf")
    max_time = 0.0
    all_durations = []
    # Track unique operations to avoid counting batched operations multiple times
    seen_operations: set[str] = set()

    for node in nodes:
        # Track overall time bounds from node creation times
        created_at = node.get("created_at", 0)
        if created_at > 0:
            min_time = min(min_time, created_at)
            max_time = max(max_time, created_at)

        # Process embedding durations - deduplicate batched operations
        embedding = node.get("embedding")
        if embedding:
            start_time = embedding.get("start_time", 0)
            end_time = embedding.get("end_time", 0)
            if start_time > 0 and end_time > start_time:
                # Use (start_time, end_time) as key to identify unique embedding batches
                operation_key = f"embedding_{start_time}_{end_time}"
                if operation_key not in seen_operations:
                    duration = end_time - start_time
                    all_durations.append(duration)
                    seen_operations.add(operation_key)
                # Update wall clock bounds with actual API times
                min_time = min(min_time, start_time)
                max_time = max(max_time, end_time)

        # Process summary attempt durations - each attempt is unique
        for attempt_idx, attempt in enumerate(node.get("summary_attempts", [])):
            start_time = attempt.get("start_time", 0)
            end_time = attempt.get("end_time", 0)
            if start_time > 0 and end_time > start_time:
                # Summary attempts are unique per node, so include node_id and attempt_idx
                operation_key = (
                    f"summary_{node['node_id']}_{attempt_idx}_{start_time}_{end_time}"
                )
                if operation_key not in seen_operations:
                    duration = end_time - start_time
                    all_durations.append(duration)
                    seen_operations.add(operation_key)
                # Update wall clock bounds with actual API times
                min_time = min(min_time, start_time)
                max_time = max(max_time, end_time)

    if not all_durations or min_time == float("inf") or max_time <= min_time:
        return 0.0

    # Calculate the three key durations
    sequential_duration = sum(all_durations)  # Everything runs sequentially
    max_parallel_duration = max(all_durations)  # Everything runs in parallel
    actual_duration = max_time - min_time  # What actually happened

    # Handle edge cases
    if sequential_duration <= max_parallel_duration:
        # Nothing to parallelize (only one operation or very close durations)
        return 100.0

    # Calculate efficiency percentage
    efficiency = (
        (sequential_duration - actual_duration)
        / (sequential_duration - max_parallel_duration)
    ) * 100.0

    # Clamp to valid range
    return max(0.0, min(100.0, efficiency))


def detect_verbatim_concatenations(
    nodes: list[NodeTelemetryDict], tolerance: float | None = None
) -> VerbatimDetectionResult:
    """Detect nodes where the LLM returned input text verbatim.

    Args:
        nodes: List of node telemetry data
        tolerance: Ratio tolerance for considering compression as verbatim.
                  If None, uses RAGZOOM_VERBATIM_TOLERANCE env var (default 0.02 = 2%)

    Returns:
        VerbatimDetectionResult containing:
        - total_summaries: Total number of summary nodes
        - verbatim_count: Number of verbatim concatenations detected
        - verbatim_percentage: Percentage of summaries that are verbatim
        - worst_offenders: List of worst cases with details (VerbatimOffender TypedDict)
        - height_distribution: Count of verbatim issues by tree height
    """
    if tolerance is None:
        # Allow configuration via environment variable
        import os

        tolerance = float(os.environ.get("RAGZOOM_VERBATIM_TOLERANCE", "0.02"))
    verbatim_nodes: list[VerbatimOffender] = []
    height_distribution: dict[int, int] = {}
    total_summaries = 0

    for node in nodes:
        # Skip leaf nodes and nodes without summaries
        summary_attempts = node.get("summary_attempts", [])
        if not summary_attempts:
            continue

        # Get input tokens from the node itself (this is what was actually summarized)
        input_tokens = node.get("input_text_tokens", 0)
        if input_tokens == 0:
            # Fallback to prompt_tokens from attempts if input_text_tokens not available
            if summary_attempts:
                input_tokens = summary_attempts[0].get("prompt_tokens", 0)

        if input_tokens == 0:
            continue

        # Get the accepted attempt
        accepted_idx = node.get("accepted_attempt")
        if accepted_idx is not None and 0 <= accepted_idx < len(summary_attempts):
            summary = summary_attempts[accepted_idx]
        else:
            # Fallback to last attempt
            summary = summary_attempts[-1]

        output_tokens = summary.get("actual_tokens", 0)

        if output_tokens == 0:
            continue

        total_summaries += 1

        # Calculate compression ratio
        ratio = output_tokens / input_tokens

        # Check if it's essentially verbatim (within tolerance)
        if abs(ratio - 1.0) <= tolerance:
            height = node["height"]
            offender: VerbatimOffender = {
                "node_id": node["node_id"],
                "height": height,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "ratio": ratio,
            }
            verbatim_nodes.append(offender)

            # Track height distribution
            height_distribution[height] = height_distribution.get(height, 0) + 1

    # Sort worst offenders by token count (highest first)
    worst_offenders = sorted(
        verbatim_nodes, key=lambda x: x["input_tokens"], reverse=True
    )[
        :5
    ]  # Top 5 worst cases

    verbatim_count = len(verbatim_nodes)
    verbatim_percentage = (
        (verbatim_count / total_summaries * 100) if total_summaries > 0 else 0
    )

    return VerbatimDetectionResult(
        total_summaries=total_summaries,
        verbatim_count=verbatim_count,
        verbatim_percentage=verbatim_percentage,
        worst_offenders=worst_offenders,
        height_distribution=height_distribution,
    )


# Telemetry analysis thresholds
HIGH_RETRY_RATE_THRESHOLD = 20.0
GOOD_BATCH_UTILIZATION_THRESHOLD = 70.0
LOW_BATCH_UTILIZATION_THRESHOLD = 50.0
MULTIPLE_RETRY_THRESHOLD = 1
HIGH_TARGET_FIT_ERROR_THRESHOLD = 20.0
GOOD_TARGET_FIT_THRESHOLD = 10.0
HIGH_COST_PER_NODE_THRESHOLD = 0.001


class TelemetryAnalysisError(Exception):
    """Raised when telemetry analysis encounters an error."""

    pass


def parse_telemetry_format(telemetry_data: TelemetryDataDict) -> TelemetryDataDict:
    """Parse telemetry data.

    Args:
        telemetry_data: Raw telemetry data from benchmark file

    Returns:
        Parsed telemetry data

    Raises:
        TelemetryAnalysisError: If format version is unsupported
    """
    if not isinstance(telemetry_data, dict):
        raise TelemetryAnalysisError("Telemetry data must be a dictionary")

    format_version = telemetry_data.get("format_version")

    if not format_version:
        raise TelemetryAnalysisError("Missing format_version in telemetry data")

    if format_version != SUPPORTED_TELEMETRY_VERSION:
        raise TelemetryAnalysisError(
            f"Unsupported telemetry format version: {format_version}. "
            f"Supported version: {SUPPORTED_TELEMETRY_VERSION}"
        )

    # Return format 4.2 data as-is
    result: TelemetryDataDict = telemetry_data
    return result


def compute_batch_efficiency(telemetry_data: TelemetryDataDict) -> BatchEfficiencyDict:
    """Analyze embedding batch utilization from telemetry data.

    Args:
        telemetry_data: Parsed telemetry data

    Returns:
        Dictionary containing batch efficiency metrics
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    batch_sizes = []
    total_embeddings = 0

    # Process nodes directly
    nodes = parsed_data.get("nodes", [])

    # Track batches by collecting unique (batch_size, timestamp) pairs
    seen_batches = set()

    for node in nodes:
        embedding = node.get("embedding")
        if not embedding:
            continue

        batch_size = embedding.get("batch_size", 1)
        start_time = embedding.get("start_time", 0.0)
        batch_key = (batch_size, start_time)

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


def get_accepted_attempt(
    node: NodeTelemetryDict,
) -> tuple[SummaryAttemptDict | None, int]:
    """Get the accepted attempt from a node's summary attempts.

    Returns:
        tuple: (accepted_attempt, index) or (None, -1) if no attempts
    """
    summary_attempts = node.get("summary_attempts", [])
    if not summary_attempts:
        return None, -1

    # Check for explicit accepted_attempt index
    accepted_idx = node.get("accepted_attempt")
    if accepted_idx is not None:
        if 0 <= accepted_idx < len(summary_attempts):
            return summary_attempts[accepted_idx], accepted_idx
        # Fallback to last if index is invalid
        return summary_attempts[-1], len(summary_attempts) - 1

    # Default: use last attempt when accepted_attempt field is missing
    return summary_attempts[-1], len(summary_attempts) - 1


def analyze_retry_patterns(telemetry_data: TelemetryDataDict) -> RetryAnalysisDict:
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

    # Process nodes directly
    nodes = parsed_data.get("nodes", [])

    for node in nodes:
        # Only process summary nodes
        # Get height for tree traversal
        height = node["height"]
        is_leaf = height == 0
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

        # Get the accepted attempt
        accepted_attempt, accepted_idx = get_accepted_attempt(node)

        for attempt_idx, attempt in enumerate(summary_attempts):
            total_attempts += 1

            # Determine if this is the accepted attempt
            # We should have exactly one accepted attempt per node
            is_accepted = attempt_idx == accepted_idx

            # Get status for rejection tracking (but don't use it to determine acceptance)
            attempt.get("status", "unknown")

            # Calculate time for this attempt if available
            start_time = attempt.get("start_time", 0)
            end_time = attempt.get("end_time", 0)
            attempt_time = end_time - start_time if end_time > start_time else 0

            # Check retry status (first attempt is initial, rest are retries)
            is_retry = attempt_idx > 0

            if is_retry:
                retry_attempts += 1
                node_has_retry = True
                node_retry_count += 1
                node_retry_time += attempt_time
                if is_accepted:
                    successful_retries += 1
                else:
                    node_rejected_time += attempt_time

            if is_accepted:
                successful_attempts += 1

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


@dataclass
class ComputedMetrics:
    """Computed metrics from telemetry data.

    This provides attributes that were previously computed by IndexingMetrics,
    calculated from raw telemetry data during analysis.
    """

    # Timing
    start_time: float
    end_time: float
    total_duration_seconds: float

    # Document info
    source_document_tokens: int

    # Token metrics
    total_tokens: int
    total_embedding_tokens: int
    total_summary_prompt_tokens: int
    total_summary_completion_tokens: int

    # API calls
    embedding_api_calls: int
    summary_api_calls: int
    chunks_created: int

    # Cost metrics
    embedding_cost_per_1k: float
    summary_input_cost_per_1k: float
    summary_output_cost_per_1k: float
    total_cost: float
    embedding_cost: float
    summary_cost: float

    # Memory metrics
    peak_memory_mb: float
    memory_start_mb: float
    memory_end_mb: float
    memory_usage_mb: float

    # Collections
    embedding_batch_sizes: list[int]
    tree_height: int
    nodes_per_height: list[int]


def compute_metrics_from_telemetry(
    telemetry_data: TelemetryDataDict,
) -> ComputedMetrics:
    """Compute metrics from raw telemetry data.

    This function computes all metrics from raw telemetry data that were
    previously available as computed properties in IndexingMetrics.

    Args:
        telemetry_data: Raw telemetry data

    Returns:
        ComputedMetrics object with all computed values
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    # Get models from telemetry for cost calculations
    models = telemetry_data.get("model_metadata", {})
    embedding_model = str(models.get("embedding_model", "text-embedding-3-small"))
    summary_model = str(models.get("summary_model", "gpt-5-nano"))

    # Get costs using helper functions
    embedding_cost_per_1k = get_embedding_cost(embedding_model)
    summary_input_cost_per_1k, summary_output_cost_per_1k = get_llm_costs(summary_model)

    # Initialize metrics data with specific types
    start_time = 0.0
    end_time = 0.0
    source_document_tokens = parsed_data.get("source_document_tokens", 0)
    total_embedding_tokens = 0
    embedding_api_calls = 0
    embedding_batch_sizes: list[int] = []
    total_summary_prompt_tokens = 0
    total_summary_completion_tokens = 0
    summary_api_calls = 0

    # Track various metrics as we process telemetry
    min_timestamp = float("inf")
    max_timestamp = 0.0
    embedding_batches = set()

    # Process nodes directly
    nodes = parsed_data.get("nodes", [])
    chunks_created = 0
    height_counts: dict[int, int] = {}

    for node in nodes:
        # Track node creation time
        created_at = node.get("created_at", 0)
        if created_at > 0:
            min_timestamp = min(min_timestamp, created_at)
            max_timestamp = max(max_timestamp, created_at)

        # Track height distribution
        height = node["height"]
        height_counts[height] = height_counts.get(height, 0) + 1

        # Process embeddings
        embedding = node.get("embedding")
        if embedding:
            text_tokens = embedding.get("text_tokens", 0)
            total_embedding_tokens += text_tokens

            # Track unique batches
            batch_size = embedding.get("batch_size", 1)
            start_time = embedding.get("start_time", 0.0)
            batch_key = (batch_size, start_time)

            if batch_key not in embedding_batches:
                embedding_batches.add(batch_key)
                embedding_api_calls += 1
                embedding_batch_sizes.append(batch_size)

        # Process summary attempts
        summary_attempts = node.get("summary_attempts", [])

        # Track cumulative tokens for this node across all attempts
        node_total_prompt_tokens = 0
        node_total_completion_tokens = 0

        for attempt in summary_attempts:
            summary_api_calls += 1
            prompt_tokens = attempt.get("prompt_tokens", 0)
            completion_tokens = attempt.get("completion_tokens", 0)

            total_summary_prompt_tokens += prompt_tokens
            total_summary_completion_tokens += completion_tokens

            # Accumulate tokens for cost calculation
            node_total_prompt_tokens += prompt_tokens
            node_total_completion_tokens += completion_tokens

        # Count chunks (leaf nodes)
        # Check if this is a leaf node (height 0)
        is_leaf = height == 0
        if is_leaf:
            chunks_created += 1

    # Set timing
    if min_timestamp != float("inf"):
        start_time = min_timestamp
        end_time = max_timestamp

    # Calculate total duration
    total_duration = end_time - start_time

    # Calculate total tokens
    total_tokens = (
        total_embedding_tokens
        + total_summary_prompt_tokens
        + total_summary_completion_tokens
    )

    # Calculate costs
    embedding_cost = (total_embedding_tokens / 1000) * embedding_cost_per_1k
    summary_cost = (total_summary_prompt_tokens / 1000) * summary_input_cost_per_1k + (
        total_summary_completion_tokens / 1000
    ) * summary_output_cost_per_1k
    total_cost = embedding_cost + summary_cost

    # Calculate tree height and nodes per height
    tree_height = max(height_counts.keys()) if height_counts else 0
    nodes_per_height = [0] * (tree_height + 1)
    for h, count in height_counts.items():
        if h <= tree_height:
            nodes_per_height[h] = count

    # Memory metrics (not available from telemetry)
    memory_metrics = {
        "peak_memory_mb": 0.0,
        "memory_start_mb": 0.0,
        "memory_end_mb": 0.0,
        "memory_usage_mb": 0.0,
    }

    return ComputedMetrics(
        # Timing
        start_time=start_time,
        end_time=end_time,
        total_duration_seconds=total_duration,
        # Document info
        source_document_tokens=source_document_tokens,
        # Token metrics
        total_tokens=total_tokens,
        total_embedding_tokens=total_embedding_tokens,
        total_summary_prompt_tokens=total_summary_prompt_tokens,
        total_summary_completion_tokens=total_summary_completion_tokens,
        # API calls
        embedding_api_calls=embedding_api_calls,
        summary_api_calls=summary_api_calls,
        chunks_created=chunks_created,
        # Cost metrics
        embedding_cost_per_1k=embedding_cost_per_1k,
        summary_input_cost_per_1k=summary_input_cost_per_1k,
        summary_output_cost_per_1k=summary_output_cost_per_1k,
        total_cost=total_cost,
        embedding_cost=embedding_cost,
        summary_cost=summary_cost,
        # Memory metrics
        **memory_metrics,
        # Collections
        embedding_batch_sizes=embedding_batch_sizes,
        tree_height=tree_height,
        nodes_per_height=nodes_per_height,
    )


# ============================================================================
# QUERY TELEMETRY ANALYSIS
# ============================================================================


def analyze_query_telemetry(telemetry_data: dict[str, object]) -> QueryPhaseMetrics:
    """Analyze query performance telemetry.

    Args:
        telemetry_data: Query telemetry data dictionary

    Returns:
        QueryPhaseMetrics with aggregated performance metrics
    """
    # Handle different telemetry formats
    format_version = telemetry_data.get("format_version")
    telemetries: list[dict[str, object]]
    statistics_data: dict[str, object]

    if format_version in ("1.1", "1.2") and "telemetries" in telemetry_data:
        # v1.1/v1.2 format with multiple runs and optional pre-calculated statistics
        telemetries_raw = telemetry_data["telemetries"]
        telemetries = telemetries_raw if isinstance(telemetries_raw, list) else []
        statistics_data_raw = telemetry_data.get("statistics", {})
        statistics_data = (
            statistics_data_raw if isinstance(statistics_data_raw, dict) else {}
        )
    elif "telemetry" in telemetry_data:
        # v1.0 format - single telemetry file
        telemetry_item = telemetry_data["telemetry"]
        telemetries = [telemetry_item] if isinstance(telemetry_item, dict) else []
        statistics_data = {}
    else:
        # Assume it's already the telemetry dict
        telemetries = [telemetry_data]
        statistics_data = {}

    if not telemetries:
        raise ValueError("No query telemetry data provided")

    # Collect all timings and metrics
    all_timings: dict[str, list[float]] = {
        "embedding_time": [],
        "search_time": [],
        "mmr_time": [],
        "coverage_map_time": [],
        "scoring_time": [],
        "tiling_time": [],
        "assembly_time": [],
        "total_time": [],
    }

    seeds_utilizations: list[float] = []
    budget_utilizations: list[float] = []
    coverage_efficiencies: list[float] = []

    for telemetry in telemetries:
        timings_raw = telemetry.get("timings", {})
        metrics_raw = telemetry.get("metrics", {})

        timings = timings_raw if isinstance(timings_raw, dict) else {}
        metrics = metrics_raw if isinstance(metrics_raw, dict) else {}

        # Collect phase timings
        for phase in all_timings:
            if phase in timings:
                timing_value = timings[phase]
                if isinstance(timing_value, int | float):
                    all_timings[phase].append(float(timing_value))

        # Calculate efficiency metrics
        seeds_requested = metrics.get("seeds_requested", 0)
        if isinstance(seeds_requested, int | float) and seeds_requested > 0:
            seeds_found = metrics.get("seeds_found", 0)
            if isinstance(seeds_found, int | float):
                seeds_utilizations.append(seeds_found / seeds_requested)

        budget_tokens = telemetry.get("budget_tokens", 0)
        if isinstance(budget_tokens, int | float) and budget_tokens > 0:
            output_tokens = metrics.get("output_tokens", 0)
            if isinstance(output_tokens, int | float):
                budget_utilizations.append(output_tokens / budget_tokens)

        coverage_size = metrics.get("coverage_size", 0)
        if isinstance(coverage_size, int | float) and coverage_size > 0:
            tiling_size = metrics.get("tiling_size", 0)
            if isinstance(tiling_size, int | float):
                coverage_efficiencies.append(tiling_size / coverage_size)

    # Calculate phase breakdown (median values) and variance (MAD)
    phase_breakdown = {}
    phase_variance = {}

    # Use pre-calculated statistics if available (v1.1 format)
    if statistics_data:
        for phase in all_timings.keys():
            if phase in statistics_data:
                phase_stats_raw = statistics_data[phase]
                if isinstance(phase_stats_raw, dict):
                    median_val = phase_stats_raw.get("median", 0.0)
                    mad_val = phase_stats_raw.get("mad", 0.0)
                    phase_breakdown[phase] = (
                        float(median_val)
                        if isinstance(median_val, int | float)
                        else 0.0
                    )
                    phase_variance[phase] = (
                        float(mad_val) if isinstance(mad_val, int | float) else 0.0
                    )
                else:
                    phase_breakdown[phase] = 0.0
                    phase_variance[phase] = 0.0
            else:
                phase_breakdown[phase] = 0.0
                phase_variance[phase] = 0.0
    else:
        # Calculate from raw telemetry data (v1.0 format or legacy)
        for phase, times in all_timings.items():
            if times:
                median_time = statistics.median(times)
                phase_breakdown[phase] = median_time

                # Calculate MAD for robust variance measurement
                absolute_deviations = [abs(t - median_time) for t in times]
                phase_variance[phase] = (
                    statistics.median(absolute_deviations)
                    if absolute_deviations
                    else 0.0
                )
            else:
                phase_breakdown[phase] = 0.0
                phase_variance[phase] = 0.0

    # Calculate efficiency metrics (averages)
    seeds_utilization = (
        sum(seeds_utilizations) / len(seeds_utilizations) if seeds_utilizations else 0.0
    )
    budget_utilization = (
        sum(budget_utilizations) / len(budget_utilizations)
        if budget_utilizations
        else 0.0
    )
    coverage_efficiency = (
        sum(coverage_efficiencies) / len(coverage_efficiencies)
        if coverage_efficiencies
        else 0.0
    )

    # Calculate latency percentiles
    total_times = all_timings["total_time"]
    if total_times:
        sorted_times = sorted(total_times)
        n = len(sorted_times)
        p50_latency = sorted_times[n // 2]
        p95_latency = sorted_times[int(n * 0.95)] if n > 1 else sorted_times[0]
        p99_latency = sorted_times[int(n * 0.99)] if n > 1 else sorted_times[0]
    else:
        p50_latency = p95_latency = p99_latency = 0.0

    return QueryPhaseMetrics(
        phase_breakdown=phase_breakdown,
        phase_variance=phase_variance,
        seeds_utilization=seeds_utilization,
        budget_utilization=budget_utilization,
        coverage_efficiency=coverage_efficiency,
        p50_latency=p50_latency,
        p95_latency=p95_latency,
        p99_latency=p99_latency,
        query_count=len(telemetries),
    )


def compare_query_performance(
    baseline_telemetry: dict[str, object],
    current_telemetry: dict[str, object],
    regression_threshold: float = 0.5,  # Increased from 20% to 50% for query variance
) -> tuple[bool, dict[str, object]]:
    """Compare query performance and detect regressions.

    Args:
        baseline_telemetry: Baseline query telemetry data
        current_telemetry: Current query telemetry data
        regression_threshold: Threshold for regression detection (default 20%)

    Returns:
        Tuple of (has_regression, comparison_report)
    """
    baseline_metrics = analyze_query_telemetry(baseline_telemetry)
    current_metrics = analyze_query_telemetry(current_telemetry)

    has_regression = False
    regressions = []
    improvements = []

    # Check for regressions in each phase
    for phase, baseline_time in baseline_metrics.phase_breakdown.items():
        current_time = current_metrics.phase_breakdown.get(phase, 0)

        if baseline_time > 0:
            change_ratio = (current_time - baseline_time) / baseline_time

            if change_ratio > regression_threshold:
                has_regression = True
                regressions.append(
                    {
                        "phase": phase.replace("_time", ""),
                        "baseline": baseline_time,
                        "current": current_time,
                        "change_percent": change_ratio * 100,
                    }
                )
            elif change_ratio < -regression_threshold:
                improvements.append(
                    {
                        "phase": phase.replace("_time", ""),
                        "baseline": baseline_time,
                        "current": current_time,
                        "change_percent": change_ratio * 100,
                    }
                )

    # Check overall latency
    if baseline_metrics.p50_latency > 0:
        p50_change = (
            current_metrics.p50_latency - baseline_metrics.p50_latency
        ) / baseline_metrics.p50_latency
        if p50_change > regression_threshold:
            has_regression = True
            regressions.append(
                {
                    "phase": "p50_latency",
                    "baseline": baseline_metrics.p50_latency,
                    "current": current_metrics.p50_latency,
                    "change_percent": p50_change * 100,
                }
            )

    comparison_report = {
        "has_regression": has_regression,
        "regressions": regressions,
        "improvements": improvements,
        "baseline_metrics": baseline_metrics.to_dict(),
        "current_metrics": current_metrics.to_dict(),
        "summary": {
            "baseline_p50": baseline_metrics.p50_latency,
            "current_p50": current_metrics.p50_latency,
            "baseline_queries": baseline_metrics.query_count,
            "current_queries": current_metrics.query_count,
        },
    }

    return has_regression, comparison_report
