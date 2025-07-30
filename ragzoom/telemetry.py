"""Telemetry analysis tools for computing metrics from raw telemetry data.

This module enables retroactive metric computation from telemetry data collected
during indexing. It supports computing amplification metrics, retry analysis,
and other insights from historical benchmark data.
"""

import logging
from statistics import median

from ragzoom.config import RagZoomConfig
from ragzoom.metrics import (
    IndexingMetrics,
    SummaryStats,
)

logger = logging.getLogger(__name__)

# Current supported telemetry format version
SUPPORTED_TELEMETRY_VERSIONS = ["1.0"]


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
        - by_level: Amplification metrics broken down by tree level
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    all_cost_amplifications = []
    all_input_amplifications = []
    all_output_amplifications = []
    amplifications_by_level: dict[int, dict[str, list[float]]] = {}

    # Process all documents
    for doc_type, doc_data in parsed_data["documents"].items():
        nodes = doc_data.get("nodes", [])

        for node in nodes:
            # Only process summary nodes (they have summary attempts)
            if node.get("node_type") != "summary":
                continue

            summary_attempts = node.get("summary_attempts", [])
            level = node.get("level", 0)

            # Process accepted attempts only
            for attempt in summary_attempts:
                if attempt.get("status") != "accepted":
                    continue

                # Extract attempt data
                prompt_tokens = attempt.get("prompt_tokens", 0)
                completion_tokens = attempt.get("completion_tokens", 0)
                input_text_tokens = attempt.get("input_text_tokens", 0)
                actual_tokens = attempt.get("actual_tokens", 0)

                if input_text_tokens == 0:
                    continue  # Skip invalid data

                # Calculate amplification factors with consistent zero checks
                input_amplification = (
                    prompt_tokens / input_text_tokens if input_text_tokens > 0 else 1.0
                )
                output_amplification = (
                    completion_tokens / actual_tokens if actual_tokens > 0 else 1.0
                )

                # Calculate cost-weighted amplification using helper
                cost_amplification = _calculate_cost_amplification(
                    prompt_tokens,
                    completion_tokens,
                    input_text_tokens,
                    actual_tokens,
                    config,
                )

                # Collect amplifications
                all_cost_amplifications.append(cost_amplification)
                all_input_amplifications.append(input_amplification)
                all_output_amplifications.append(output_amplification)

                # Track by level
                if level not in amplifications_by_level:
                    amplifications_by_level[level] = {
                        "input": [],
                        "output": [],
                        "cost": [],
                    }

                amplifications_by_level[level]["input"].append(input_amplification)
                amplifications_by_level[level]["output"].append(output_amplification)
                amplifications_by_level[level]["cost"].append(cost_amplification)

    # Compute summary statistics
    result = {
        "median_cost": 0.0,
        "cost_p90": 0.0,
        "cost_p95": 0.0,
        "median_input": 0.0,
        "median_output": 0.0,
        "by_level": amplifications_by_level,
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
    """Compute percentile using linear interpolation (consistent with metrics.py)."""
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

    Args:
        prompt_tokens: Tokens in the API prompt
        completion_tokens: Tokens in the API completion
        input_text_tokens: Tokens in the original text being summarized
        actual_tokens: Actual tokens in the generated summary
        config: Configuration with pricing information

    Returns:
        Cost amplification factor (1.0 = no amplification)
    """
    # Calculate actual cost
    actual_cost = (
        prompt_tokens * config.summary_input_cost_per_1k
        + completion_tokens * config.summary_output_cost_per_1k
    ) / 1000

    # Calculate theoretical minimum cost
    min_cost = (
        input_text_tokens * config.summary_input_cost_per_1k
        + actual_tokens * config.summary_output_cost_per_1k
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
            timestamp = embedding.get("timestamp", 0)
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
        # Batch utilization calculation assumes the optimal batch size is the
        # maximum observed batch size in this telemetry data. This provides a
        # relative measure of how well batching was utilized compared to the
        # best case observed in this specific run.
        max_batch_size = max(batch_sizes) if batch_sizes else 1
        result["batch_utilization"] = (avg_batch_size / max_batch_size) * 100

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
        - cost_overhead: Additional cost from retries (requires config)
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
            if node.get("node_type") != "summary":
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


def compute_metrics_from_telemetry(
    telemetry_data: dict, config: RagZoomConfig
) -> IndexingMetrics:
    """Compute full IndexingMetrics from raw telemetry data.

    This function reconstructs an IndexingMetrics object as if it were collected
    during indexing, enabling retroactive analysis of benchmark data.

    Args:
        telemetry_data: Raw telemetry data from benchmark file
        config: Configuration for cost calculations and pricing

    Returns:
        IndexingMetrics object computed from telemetry

    Raises:
        TelemetryAnalysisError: If telemetry data is invalid or incomplete
    """
    parsed_data = parse_telemetry_format(telemetry_data)

    # Initialize metrics with dummy values - we'll compute real ones from telemetry
    metrics = IndexingMetrics(
        start_time=0.0,
        end_time=0.0,
        source_document_tokens=0,
        chunks_created=0,
        embedding_cost_per_1k=config.embedding_cost_per_1k,
        summary_input_cost_per_1k=config.summary_input_cost_per_1k,
        summary_output_cost_per_1k=config.summary_output_cost_per_1k,
    )

    # Track various metrics as we process telemetry
    min_timestamp = float("inf")
    max_timestamp = 0.0
    embedding_batches = set()
    summary_stats_by_target = {}

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
                timestamp = embedding.get("timestamp", 0)

                metrics.total_embedding_tokens += text_tokens

                # Track unique batches by (batch_size, timestamp)
                batch_key = (batch_size, timestamp)
                if batch_key not in embedding_batches:
                    embedding_batches.add(batch_key)
                    metrics.embedding_api_calls += 1
                    metrics.embedding_batch_sizes.append(batch_size)

            # Process summary attempts
            summary_attempts = node.get("summary_attempts", [])
            for attempt in summary_attempts:
                metrics.summary_api_calls += 1
                metrics.total_summary_prompt_tokens += attempt.get("prompt_tokens", 0)
                metrics.total_summary_completion_tokens += attempt.get(
                    "completion_tokens", 0
                )

                # Only process accepted attempts for accuracy and amplification
                if attempt.get("status") != "accepted":
                    continue

                target_tokens = attempt.get("target_tokens", 0)
                actual_tokens = attempt.get("actual_tokens", 0)

                # Track summary accuracy
                if target_tokens > 0:
                    if target_tokens not in summary_stats_by_target:
                        summary_stats_by_target[target_tokens] = SummaryStats()

                    summary_stats_by_target[target_tokens].add_summary(
                        target_tokens, actual_tokens
                    )

                # Calculate amplification factors
                input_text_tokens = attempt.get("input_text_tokens", 0)
                prompt_tokens = attempt.get("prompt_tokens", 0)
                completion_tokens = attempt.get("completion_tokens", 0)

                if input_text_tokens > 0:
                    input_amplification = (
                        prompt_tokens / input_text_tokens
                        if input_text_tokens > 0
                        else 1.0
                    )
                    output_amplification = (
                        completion_tokens / actual_tokens if actual_tokens > 0 else 1.0
                    )

                    # Calculate cost-weighted amplification using helper
                    cost_amplification = _calculate_cost_amplification(
                        prompt_tokens,
                        completion_tokens,
                        input_text_tokens,
                        actual_tokens,
                        config,
                    )

                    # Record amplifications
                    metrics.input_amplifications.append(input_amplification)
                    metrics.output_amplifications.append(output_amplification)
                    metrics.cost_amplifications.append(cost_amplification)

            # Count chunks (leaf nodes)
            if node.get("node_type") == "leaf":
                metrics.chunks_created += 1

    # Set timing
    if min_timestamp != float("inf"):
        metrics.start_time = min_timestamp
        metrics.end_time = max_timestamp

    # Set summary stats
    metrics.summary_stats = summary_stats_by_target

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
            # Rough estimate: assume average leaf size of 150 tokens
            # This is only used for old telemetry data without metadata
            estimated_tokens = leaf_count * 150
            total_source_tokens += estimated_tokens
            logger.warning(
                f"Source tokens not found in telemetry metadata, using estimate: {estimated_tokens}"
            )

    metrics.source_document_tokens = total_source_tokens

    return metrics
