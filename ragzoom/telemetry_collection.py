"""Telemetry data collection for RagZoom indexing.

This module provides data structures and collection mechanisms for raw telemetry data
during indexing operations. All analysis and metric computation is done separately
in telemetry_analysis.py to maintain a clean separation between data collection
and analysis.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import psutil

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry_types import NodeTelemetryDict

logger = logging.getLogger(__name__)

# Telemetry format version
# Version history:
# - 1.0: Initial telemetry format with node-level tracking
# - 2.0: Improved telemetry format:
#   - Removed redundant fields: is_retry, node_type, span fields
#   - Renamed 'level' to 'height' throughout
#   - Added start_time/end_time to EmbeddingTelemetry and SummaryAttempt
#   - Removed timestamp field in favor of start_time/end_time
# - 3.0: Flattened structure to eliminate redundancy:
#   - Removed nested "documents" dict (always single document)
#   - Moved metadata fields to top level
#   - Added models field at top level
#   - Eliminated duplicate document_id and chunk_size fields
#
# Current format version (increment for breaking changes)
TELEMETRY_FORMAT_VERSION = "3.0"


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
    prompt_hash: str | None = None
    cached_tokens: int = 0  # Number of cached prompt tokens (for prompt caching)


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

    def to_telemetry_dict(self) -> NodeTelemetryDict:
        """Convert to telemetry format for analysis.

        Returns a dictionary in the format expected by telemetry analysis tools.
        """
        result: NodeTelemetryDict = {
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
            from ragzoom.telemetry_types import SummaryAttemptDict

            attempts_list: list[SummaryAttemptDict] = []
            for attempt in self.summary_attempts:
                attempt_dict: SummaryAttemptDict = {
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
                # Handle cached_tokens, which might be MagicMock in tests
                cached_tokens_value = getattr(attempt, "cached_tokens", 0)
                if hasattr(cached_tokens_value, "__gt__"):  # Check if it's comparable
                    try:
                        if cached_tokens_value > 0:
                            attempt_dict["cached_tokens"] = cached_tokens_value
                    except (TypeError, AttributeError):
                        # If it's a MagicMock or similar, skip it
                        pass
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

        # API usage tracking
        self.embedding_api_calls = 0
        self.summary_api_calls = 0
        self.total_embedding_tokens = 0
        self.total_summary_prompt_tokens = 0
        self.total_summary_completion_tokens = 0

        # Batch tracking
        self.embedding_batch_sizes: list[int] = []

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
        """Record summary generation result with size tracking.

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
        cached_tokens: int = 0,
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
            cached_tokens: Number of cached prompt tokens (for prompt caching)
        """
        # Update aggregate metrics
        self.summary_api_calls += 1
        self.total_summary_prompt_tokens += prompt_tokens
        self.total_summary_completion_tokens += completion_tokens

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
            cached_tokens=cached_tokens,
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
            Dictionary in telemetry format v3.0 (flat structure)
        """
        # Convert all node telemetry to dict format
        nodes_data = []
        for node in self.node_telemetry.values():
            nodes_data.append(node.to_telemetry_dict())

        # Sort nodes by creation time for consistent output
        nodes_data.sort(key=lambda x: x["created_at"])

        return {
            "format_version": TELEMETRY_FORMAT_VERSION,
            "document_id": document_id,
            "source_document_tokens": self.source_tokens,
            "chunk_size": chunk_size,
            "indexed_at": self.start_time,
            "models": {
                "summary": self.config.summary_model,
                "embedding": self.config.embedding_model,
            },
            "nodes": nodes_data,
        }
