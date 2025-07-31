"""Telemetry data collection for RagZoom indexing.

This module provides data structures and collection mechanisms for raw telemetry data
during indexing operations. All analysis and metric computation is done separately
in telemetry_analysis.py to maintain a clean separation between data collection
and analysis.
"""

import logging
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import psutil

from ragzoom.config import RagZoomConfig

logger = logging.getLogger(__name__)

# Telemetry format version
# Version history:
# - 1.0: Initial telemetry format with node-level tracking
# - 2.0: Improved telemetry format:
#   - Removed redundant fields: is_retry, node_type, span fields
#   - Renamed 'level' to 'height' throughout
#   - Added start_time/end_time to EmbeddingTelemetry and SummaryAttempt
#   - Removed timestamp field in favor of start_time/end_time
#
# Migration strategy:
# - Telemetry format changes should be backward compatible when possible
# - When breaking changes are needed:
#   1. Increment major version (e.g., 1.0 -> 2.0)
#   2. Add migration code in analysis tools to handle old formats
#   3. Document the changes in this version history
# - Minor additions that don't break parsing can use minor version bumps (e.g., 1.0 -> 1.1)
TELEMETRY_FORMAT_VERSION = "2.0"


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
class EmbeddingTelemetry:
    """Telemetry for embedding API call."""

    text_tokens: int
    batch_size: int
    batch_position: int
    model: str
    start_time: float
    end_time: float


@dataclass
class SummaryAttempt:
    """Telemetry for a single summary attempt.

    Designed to support PR #29's retry mechanism where summaries may be
    regenerated if they don't meet size constraints. Each attempt is recorded
    separately to enable analysis of retry patterns and costs.

    Attributes:
        target_tokens: Target size for the summary
        input_text_tokens: Combined tokens from children being summarized
        prompt_tokens: Tokens used in the API prompt
        completion_tokens: Tokens reported by the OpenAI API
        actual_tokens: Tokens measured by our tokenizer from the generated text
        status: Outcome - 'accepted', 'rejected_over', 'rejected_under', 'error'
        model: Model used for generation
        start_time: When this attempt started
        end_time: When this attempt completed
        rejection_reason: Optional explanation for rejection
        prompt_hash: Optional hash for deduplication analysis
    """

    # Inputs
    target_tokens: int
    input_text_tokens: int  # Combined left+right child tokens

    # API usage
    prompt_tokens: int
    completion_tokens: int  # Tokens reported by API

    # Results
    actual_tokens: int  # Tokens we measure in the summary text
    status: Literal["accepted", "rejected_over", "rejected_under", "error"]

    # Model info
    model: str
    start_time: float
    end_time: float

    # Optional fields with defaults
    rejection_reason: str | None = None
    prompt_hash: str | None = None  # Hash of prompt for deduplication analysis


@dataclass
class NodeTelemetry:
    """Telemetry data for a single node.

    This data structure captures all API interactions and metadata for a node
    during indexing. It enables retroactive metric computation without re-running
    expensive indexing operations.

    Attributes:
        node_id: Unique identifier for the node
        height: Tree height (0 = leaves, increases up the tree)
        embedding: Embedding API call details (optional)
        summary_attempts: List of summary generation attempts
        created_at: Timestamp when node was created
    """

    node_id: str
    height: int

    # Embedding telemetry
    embedding: EmbeddingTelemetry | None = None

    # Summary telemetry (multiple attempts for retries)
    summary_attempts: list[SummaryAttempt] = field(default_factory=list)

    # Timing
    created_at: float = field(default_factory=time.time)

    def to_telemetry_dict(self) -> dict:
        """Convert to telemetry format for analysis.

        Returns a dictionary in the format expected by telemetry analysis tools.
        """
        result = {
            "node_id": self.node_id,
            "height": self.height,
            "created_at": self.created_at,
        }

        # Add embedding info if present
        if self.embedding:
            result["embedding"] = {
                "text_tokens": self.embedding.text_tokens,
                "batch_size": self.embedding.batch_size,
                "batch_position": self.embedding.batch_position,
                "model": self.embedding.model,
                "start_time": self.embedding.start_time,
                "end_time": self.embedding.end_time,
            }

        # Add summary attempts if present
        if self.summary_attempts:
            attempts_list: list[dict] = []
            for attempt in self.summary_attempts:
                attempt_dict = {
                    "target_tokens": attempt.target_tokens,
                    "input_text_tokens": attempt.input_text_tokens,
                    "prompt_tokens": attempt.prompt_tokens,
                    "completion_tokens": attempt.completion_tokens,
                    "actual_tokens": attempt.actual_tokens,
                    "status": attempt.status,
                    "model": attempt.model,
                    "start_time": attempt.start_time,
                    "end_time": attempt.end_time,
                }
                if attempt.rejection_reason:
                    attempt_dict["rejection_reason"] = attempt.rejection_reason
                attempts_list.append(attempt_dict)
            result["summary_attempts"] = attempts_list

        return result


class TelemetryCollector:
    """Collects raw telemetry data during indexing operations.

    This collector gathers raw performance data without computing any derived metrics.
    All metric computation is done separately during analysis to maintain clean
    separation of concerns.

    Design principles:
    - Telemetry collection is optional and never interferes with indexing
    - Only raw data is collected (no computed aggregations)
    - Zero overhead when not enabled
    - Thread-safe for concurrent operations
    """

    def __init__(
        self,
        document_id: str,
        source_tokens: int,
        config: RagZoomConfig,
    ):
        """Initialize telemetry collector for a document.

        Args:
            document_id: Document being indexed
            source_tokens: Total tokens in source document
            config: Config containing pricing information for telemetry metadata
        """
        self.document_id = document_id
        self.source_tokens = source_tokens
        self.config = config

        # Initialize telemetry data storage
        self.start_time = time.time()
        self.end_time = 0.0
        self.chunks_created = 0

        # Pricing configuration (stored for metadata)
        self.embedding_cost_per_1k = config.embedding_cost_per_1k
        self.summary_input_cost_per_1k = config.summary_input_cost_per_1k
        self.summary_output_cost_per_1k = config.summary_output_cost_per_1k

        # API usage tracking
        self.embedding_api_calls = 0
        self.summary_api_calls = 0
        self.total_embedding_tokens = 0
        self.total_summary_prompt_tokens = 0
        self.total_summary_completion_tokens = 0

        # Batch tracking
        self.embedding_batch_sizes: list[int] = []

        # Summary accuracy by target size
        self.summary_stats: dict[int, SummaryStats] = {}

        # Amplification tracking (per-operation)
        self.input_amplifications: list[float] = []
        self.output_amplifications: list[float] = []
        self.cost_amplifications: list[float] = []

        # Amplifications by tree height
        self.amplifications_by_height: dict[int, dict[str, list[float]]] = {}

        # Tree structure
        self.tree_height = 0
        self.nodes_per_height: list[int] = []

        # Memory tracking
        self.process = psutil.Process()
        memory_info = self.process.memory_info()
        self.memory_start_mb = memory_info.rss / 1024 / 1024
        self.peak_memory_mb = self.memory_start_mb
        self.memory_end_mb = 0.0

        # Node telemetry storage
        self.node_telemetry: dict[str, NodeTelemetry] = {}

        # Internal state
        self._current_height = 0
        self._nodes_at_current_height = 0
        self._pending_embeddings: dict[str, NodeTelemetry] = {}
        self._memory_lock = threading.Lock()

    def track_node_created(
        self,
        node_id: str,
        height: int,
    ) -> None:
        """Track when a node is created (before any API calls).

        Args:
            node_id: Unique identifier for the node
            height: Tree height (0 = leaves)
        """
        telemetry = NodeTelemetry(
            node_id=node_id,
            height=height,
        )
        self.node_telemetry[node_id] = telemetry

        # Also track pending for embedding batch correlation
        if height == 0:  # Leaf nodes have height 0
            self._pending_embeddings[node_id] = telemetry

    def _update_memory_usage(self) -> None:
        """Update peak memory usage if current usage is higher.

        Thread-safe: Uses a lock to prevent race conditions when multiple
        async tasks update peak memory concurrently. Memory reading is done
        inside the lock to prevent missing spikes between read and comparison.
        """
        try:
            # Use lock to ensure thread-safe memory reading and peak update
            with self._memory_lock:
                memory_info = self.process.memory_info()
                current_memory_mb = memory_info.rss / 1024 / 1024
                if current_memory_mb > self.peak_memory_mb:
                    self.peak_memory_mb = current_memory_mb
        except Exception as e:
            # Log but don't fail indexing due to memory tracking issues
            logger.warning(f"Failed to update memory usage: {e}")

    def record_chunk_created(self, chunk_id: str, tokens: int) -> None:
        """Called when a chunk is created during splitting."""
        self.chunks_created += 1
        # Track for leaf-height nodes
        if self._current_height == 0:
            self._nodes_at_current_height += 1
        self._update_memory_usage()

    def record_embedding_call(self, batch_size: int, token_counts: list[int]) -> None:
        """Called before embedding API call.

        Args:
            batch_size: Number of texts in batch
            token_counts: Token count for each text
        """
        self.embedding_api_calls += 1
        self.embedding_batch_sizes.append(batch_size)
        self.total_embedding_tokens += sum(token_counts)
        self._update_memory_usage()

    def record_embedding_call_v2(
        self,
        node_embeddings: list[tuple[str, int]],
        batch_size: int,
        model: str,
        start_time: float,
    ) -> None:
        """Enhanced embedding tracking with node-level detail.

        Args:
            node_embeddings: List of (node_id, token_count) tuples
            batch_size: Total batch size
            model: Model used for embeddings
            start_time: When the API call started

        Raises:
            ValueError: If a node_id is not found in telemetry (indicates tracking bug)
        """
        # Update aggregate metrics (backward compatible)
        self.record_embedding_call(batch_size, [tc for _, tc in node_embeddings])

        # Update telemetry
        end_time = time.time()
        for position, (node_id, token_count) in enumerate(node_embeddings):
            if node_id not in self.node_telemetry:
                # This indicates a bug - nodes should be tracked before embeddings
                logger.error(
                    f"Node {node_id} not found in telemetry. "
                    f"This suggests track_node_created() was not called for this node."
                )
                raise ValueError(
                    f"Node {node_id} not found in telemetry. "
                    f"Nodes must be tracked with track_node_created() before embedding."
                )

            self.node_telemetry[node_id].embedding = EmbeddingTelemetry(
                text_tokens=token_count,
                batch_size=batch_size,
                batch_position=position,
                model=model,
                start_time=start_time,
                end_time=end_time,
            )

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
        self.summary_api_calls += 1
        self.total_summary_prompt_tokens += prompt_tokens
        self.total_summary_completion_tokens += completion_tokens

        # Track accuracy by target size
        if target_tokens not in self.summary_stats:
            self.summary_stats[target_tokens] = SummaryStats()

        self.summary_stats[target_tokens].add_summary(target_tokens, actual_tokens)

        # Calculate amplification factors
        if input_text_tokens > 0:
            input_amplification = prompt_tokens / input_text_tokens
            output_amplification = (
                completion_tokens / actual_tokens if actual_tokens > 0 else 1.0
            )

            # Calculate cost-weighted amplification
            # Cost amplification = (actual cost / theoretical minimum cost)
            actual_cost = (
                prompt_tokens * self.summary_input_cost_per_1k
                + completion_tokens * self.summary_output_cost_per_1k
            ) / 1000

            min_cost = (
                input_text_tokens * self.summary_input_cost_per_1k
                + actual_tokens * self.summary_output_cost_per_1k
            ) / 1000

            cost_amplification = actual_cost / min_cost if min_cost > 0 else 1.0

            # Record per-operation amplifications
            self.input_amplifications.append(input_amplification)
            self.output_amplifications.append(output_amplification)
            self.cost_amplifications.append(cost_amplification)

            # Track by height
            height = self._current_height
            if height not in self.amplifications_by_height:
                self.amplifications_by_height[height] = {
                    "input": [],
                    "output": [],
                    "cost": [],
                }

            self.amplifications_by_height[height]["input"].append(input_amplification)
            self.amplifications_by_height[height]["output"].append(output_amplification)
            self.amplifications_by_height[height]["cost"].append(cost_amplification)

        self._update_memory_usage()

    def record_summary_attempt_v2(
        self,
        node_id: str,
        target_tokens: int,
        input_text_tokens: int,
        prompt_tokens: int,
        completion_tokens: int,
        actual_tokens: int,
        status: Literal["accepted", "rejected_over", "rejected_under", "error"],
        model: str,
        start_time: float,
        rejection_reason: str | None = None,
    ) -> None:
        """Record a summary attempt (compatible with PR #29 retry mechanism).

        Args:
            node_id: Node being summarized
            target_tokens: Target token count for summary
            input_text_tokens: Combined tokens from children being summarized
            prompt_tokens: Tokens in the prompt
            completion_tokens: Tokens in the completion (from API)
            actual_tokens: Actual tokens in summary text (measured)
            status: Outcome of this attempt
            model: Model used for summary
            start_time: When the API call started
            rejection_reason: Optional reason for rejection
        """
        # Update aggregate metrics
        self.summary_api_calls += 1
        self.total_summary_prompt_tokens += prompt_tokens
        self.total_summary_completion_tokens += completion_tokens

        # Only update result metrics if accepted
        if status == "accepted":
            # Update result-specific metrics (accuracy, amplification)
            # Track accuracy by target size
            if target_tokens not in self.summary_stats:
                self.summary_stats[target_tokens] = SummaryStats()

            self.summary_stats[target_tokens].add_summary(target_tokens, actual_tokens)

            # Calculate amplification factors
            if input_text_tokens > 0:
                input_amplification = prompt_tokens / input_text_tokens
                output_amplification = (
                    completion_tokens / actual_tokens if actual_tokens > 0 else 1.0
                )

                # Calculate cost-weighted amplification
                # Cost amplification = (actual cost / theoretical minimum cost)
                actual_cost = (
                    prompt_tokens * self.summary_input_cost_per_1k
                    + completion_tokens * self.summary_output_cost_per_1k
                ) / 1000

                min_cost = (
                    input_text_tokens * self.summary_input_cost_per_1k
                    + actual_tokens * self.summary_output_cost_per_1k
                ) / 1000

                cost_amplification = actual_cost / min_cost if min_cost > 0 else 1.0

                # Record per-operation amplifications
                self.input_amplifications.append(input_amplification)
                self.output_amplifications.append(output_amplification)
                self.cost_amplifications.append(cost_amplification)

                # Track by height
                height = self._current_height
                if height not in self.amplifications_by_height:
                    self.amplifications_by_height[height] = {
                        "input": [],
                        "output": [],
                        "cost": [],
                    }

                self.amplifications_by_height[height]["input"].append(
                    input_amplification
                )
                self.amplifications_by_height[height]["output"].append(
                    output_amplification
                )
                self.amplifications_by_height[height]["cost"].append(cost_amplification)

        # Update telemetry
        if node_id not in self.node_telemetry:
            # This indicates a bug - nodes should be tracked before summaries
            logger.error(
                f"Node {node_id} not found in telemetry for summary attempt. "
                f"This suggests track_node_created() was not called for this node."
            )
            raise ValueError(
                f"Node {node_id} not found in telemetry. "
                f"Nodes must be tracked with track_node_created() before recording summary attempts."
            )

        attempt = SummaryAttempt(
            target_tokens=target_tokens,
            input_text_tokens=input_text_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            actual_tokens=actual_tokens,
            status=status,
            model=model,
            start_time=start_time,
            end_time=time.time(),
            rejection_reason=rejection_reason,
        )
        self.node_telemetry[node_id].summary_attempts.append(attempt)

        self._update_memory_usage()

    def record_tree_height_complete(self, height: int, nodes_created: int) -> None:
        """Called when a tree height is complete.

        Args:
            height: Tree height (0 = leaves)
            nodes_created: Number of nodes at this height
        """
        # Record nodes from previous height when moving to new height
        if height > self._current_height:
            self.nodes_per_height.append(self._nodes_at_current_height)
            self._current_height = height
            self._nodes_at_current_height = nodes_created
        else:
            self._nodes_at_current_height = nodes_created

        self.tree_height = max(self.tree_height, height)
        self._update_memory_usage()

    def finalize(self) -> dict:
        """Finalize telemetry collection and return raw telemetry data.

        Returns:
            Dictionary containing all collected telemetry data in standard format
        """
        self.end_time = time.time()

        # Record final height
        if self._nodes_at_current_height > 0:
            self.nodes_per_height.append(self._nodes_at_current_height)

        # Record final memory usage
        try:
            memory_info = self.process.memory_info()
            self.memory_end_mb = memory_info.rss / 1024 / 1024
        except Exception:
            self.memory_end_mb = self.peak_memory_mb

        # Return telemetry data in standard format
        return self.get_telemetry_data(self.document_id, self.config.leaf_tokens)

    def get_telemetry_data(self, document_id: str, chunk_size: int) -> dict:
        """Export raw telemetry data in standard format for analysis.

        Args:
            document_id: Document identifier for the telemetry
            chunk_size: Chunk size used for indexing (for metadata)

        Returns:
            Dictionary in telemetry format
        """
        # Convert all node telemetry to dict format
        nodes_data = []
        for node in self.node_telemetry.values():
            nodes_data.append(node.to_telemetry_dict())

        # Sort nodes by creation time for consistent output
        nodes_data.sort(key=lambda x: x["created_at"])

        return {
            "format_version": TELEMETRY_FORMAT_VERSION,
            "documents": {
                document_id: {
                    "metadata": {
                        "source_document_tokens": self.source_tokens,
                        "chunk_size": chunk_size,
                        "indexed_at": self.start_time,
                    },
                    "nodes": nodes_data,
                }
            },
        }
