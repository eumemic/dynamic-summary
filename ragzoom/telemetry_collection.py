"""Telemetry data collection for RagZoom indexing.

This module provides data structures and collection mechanisms for raw telemetry data
during indexing operations. All analysis and metric computation is done separately
in telemetry_analysis.py to maintain a clean separation between data collection
and analysis.
"""

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    pass  # Imports moved to telemetry_types

import psutil

from ragzoom.config import IndexConfig
from ragzoom.constants import DEFAULT_SUMMARY_SYSTEM_PROMPT
from ragzoom.telemetry_types import (
    ModelMetadataDict,
    NodeTelemetryDict,
    RuntimeInfoDict,
    TelemetryDataDict,
)

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
# - 3.1: Optimized data structure:
#   - Moved input_text_tokens from SummaryAttempt to NodeTelemetry (node level)
#   - Added document_path field for telemetry file tracking
#   - Removed dead amplification code from analysis
# - 4.0: Configuration improvements:
#   - Renamed indexing_config to config
#   - Config now includes all indexing parameters from new config system
# - 4.1: Enhanced reproducibility:
#   - Added model_metadata for complete model information
#   - Added system_prompts used during indexing
#   - Added runtime_info for environment details
# - 4.2: Re-added span fields:
#   - Added span_start and span_end from actual TreeNode data
#   - These are the real character positions from the document
# - 4.3: Chunk splitting telemetry:
#   - Added raw chunk splitting timing metadata to telemetry payload
#   - Exposed chunk split details for visualization of preprocessing overhead
#
# Current format version (increment for breaking changes)
TELEMETRY_FORMAT_VERSION = "4.3"


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
class RetrievalTelemetry:
    """Telemetry for preceding context retrieval call."""

    start_time: float
    end_time: float
    tiling_node_count: int  # Number of nodes in result tiling
    tiling_tokens: int  # Total tokens in tiling nodes


@dataclass
class SummaryAttempt:
    """Telemetry for a single summary attempt.

    Records raw metrics for each summary generation attempt without
    categorizing as accepted/rejected. Analysis tools can compute
    deviation and apply thresholds as needed.

    Attributes:
        target_tokens: Target size for the summary
        prompt_tokens: Tokens used in the API prompt
        completion_tokens: Tokens reported by the OpenAI API
        actual_tokens: Tokens measured by our tokenizer from the generated text
        model: Model used for generation
        start_time: When this attempt started
        end_time: When this attempt completed
        cached_tokens: Number of cached prompt tokens (for prompt caching)
    """

    # Inputs
    target_tokens: int

    # API usage
    prompt_tokens: int
    completion_tokens: int  # Tokens reported by API

    # Results
    actual_tokens: int  # Tokens we measure in the summary text

    # Model info
    model: str
    start_time: float
    end_time: float

    # Optional fields with defaults
    cached_tokens: int = 0  # Number of cached prompt tokens (for prompt caching)

    @property
    def deviation_percent(self) -> float:
        """Calculate deviation from target as a percentage."""
        if self.target_tokens == 0:
            return 0.0
        return (self.actual_tokens - self.target_tokens) / self.target_tokens * 100


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
        accepted_attempt: Index of the attempt that was actually used
        input_text_tokens: Combined tokens from children being summarized (for non-leaf nodes)
        created_at: Timestamp when node was created
    """

    node_id: str
    height: int

    # Document span (character positions)
    span: tuple[int, int] | None = None

    # Embedding telemetry
    embedding: EmbeddingTelemetry | None = None

    # Retrieval telemetry (preceding context lookup)
    retrieval: RetrievalTelemetry | None = None

    # Summary telemetry (multiple attempts for retries)
    summary_attempts: list[SummaryAttempt] = field(default_factory=list)

    # Which attempt was actually used (None means use last attempt for backward compat)
    accepted_attempt: int | None = None

    # Combined tokens from children being summarized (for non-leaf nodes)
    input_text_tokens: int | None = None

    fidelity: float | None = None

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

        # Add span if present
        if self.span is not None:
            result["span"] = self.span

        # Add input_text_tokens at node level if present
        if self.input_text_tokens is not None:
            result["input_text_tokens"] = self.input_text_tokens

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

        # Add retrieval info if present
        if self.retrieval:
            result["retrieval"] = {
                "start_time": self.retrieval.start_time,
                "end_time": self.retrieval.end_time,
                "tiling_node_count": self.retrieval.tiling_node_count,
                "tiling_tokens": self.retrieval.tiling_tokens,
            }

        if self.fidelity is not None:
            result["fidelity"] = self.fidelity

        # Add summary attempts if present
        if self.summary_attempts:
            from ragzoom.telemetry_types import SummaryAttemptDict

            attempts_list: list[SummaryAttemptDict] = []
            for attempt in self.summary_attempts:
                attempt_dict: SummaryAttemptDict = {
                    "target_tokens": attempt.target_tokens,
                    "prompt_tokens": attempt.prompt_tokens,
                    "completion_tokens": attempt.completion_tokens,
                    "actual_tokens": attempt.actual_tokens,
                    "model": attempt.model,
                    "start_time": attempt.start_time,
                    "end_time": attempt.end_time,
                }
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

            # Add accepted_attempt index if set
            if self.accepted_attempt is not None:
                result["accepted_attempt"] = self.accepted_attempt

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
        config: IndexConfig,
        document_path: str | None = None,
    ):
        """Initialize telemetry collector for a document.

        Args:
            document_id: Document being indexed
            source_tokens: Total tokens in source document
            config: Index config for telemetry metadata
            document_path: Optional absolute path to the source document
        """
        self.document_id = document_id
        self.source_tokens = source_tokens
        self.config = config
        self.document_path = document_path

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
        # Don't store process object - create fresh each time to avoid thread safety issues
        memory_info = psutil.Process().memory_info()
        self.memory_start_mb = memory_info.rss / 1024 / 1024
        self.peak_memory_mb = self.memory_start_mb
        self.memory_end_mb = 0.0

        # Node telemetry storage
        self.node_telemetry: dict[str, NodeTelemetry] = {}
        self._chunk_split_info: dict[str, object] | None = None
        self._chunk_split_start_time: float | None = None

        # Internal state
        self._current_height = 0
        self._nodes_at_current_height = 0
        self._pending_embeddings: dict[str, NodeTelemetry] = {}
        self._memory_lock = threading.Lock()
        self.append_metadata: dict[str, object] | None = None

    def record_append_metadata(
        self,
        *,
        span_start: int,
        span_end: int,
        mutated_nodes: int,
        summary_nodes: int,
        leaf_delta: int,
    ) -> None:
        """Record metadata describing an incremental append patch."""

        self.append_metadata = {
            "scope": "append",
            "span_start": int(span_start),
            "span_end": int(span_end),
            "mutated_nodes": int(mutated_nodes),
            "summary_nodes": int(summary_nodes),
            "leaf_delta": int(leaf_delta),
        }

    def track_node_created(
        self,
        node_id: str,
        height: int,
        span: tuple[int, int] | None = None,
    ) -> None:
        """Track when a node is created (before any API calls).

        Args:
            node_id: Unique identifier for the node
            height: Tree height (0 = leaves)
            span: Character positions in document (start, end)
        """
        telemetry = NodeTelemetry(
            node_id=node_id,
            height=height,
            span=span,
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
                # Create fresh process object each time for thread safety
                memory_info = psutil.Process().memory_info()
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

    def record_chunk_split_start(
        self,
        *,
        start_time: float,
        new_text_chars: int,
        existing_tail_chars: int,
        combined_chars: int,
    ) -> None:
        """Record the start of a chunk splitting operation."""

        self._chunk_split_start_time = start_time
        self._chunk_split_info = {
            "new_text_chars": int(new_text_chars),
            "existing_tail_chars": int(existing_tail_chars),
            "combined_chars": int(combined_chars),
        }
        self._update_memory_usage()

    def record_chunk_split_end(
        self,
        *,
        end_time: float,
        chunk_count: int,
        total_tokens: int,
    ) -> None:
        """Record completion details for chunk splitting."""

        info = self._chunk_split_info or {}
        info.update(
            {
                "chunk_count": int(chunk_count),
                "total_tokens": int(total_tokens),
            }
        )
        start_time = self._chunk_split_start_time
        if start_time is not None:
            info["start_time"] = start_time
            info["end_time"] = end_time
            info["duration"] = max(0.0, end_time - start_time)
        self._chunk_split_info = info
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
        node_embeddings: Sequence[tuple[str, int]],
        batch_size: int,
        model: str,
        start_time: float,
    ) -> None:
        """Enhanced embedding tracking with node-level detail.

        Args:
            node_embeddings: List of (node_id, token_count, vector) tuples
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

    def record_retrieval_call(
        self,
        node_id: str,
        tiling_node_count: int,
        tiling_tokens: int,
        start_time: float,
    ) -> None:
        """Record retrieval timing for preceding context lookup.

        Args:
            node_id: The node for which context was retrieved
            tiling_node_count: Number of nodes in the result tiling
            tiling_tokens: Total tokens in the tiling nodes
            start_time: When the retrieval call started
        """
        end_time = time.time()
        if node_id not in self.node_telemetry:
            # Node may not be tracked yet for leaves during initial creation
            return
        self.node_telemetry[node_id].retrieval = RetrievalTelemetry(
            start_time=start_time,
            end_time=end_time,
            tiling_node_count=tiling_node_count,
            tiling_tokens=tiling_tokens,
        )

    def record_node_fidelity(self, node_id: str, fidelity: float) -> None:
        telemetry = self.node_telemetry.get(node_id)
        if telemetry is None:
            return
        telemetry.fidelity = fidelity

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
        model: str,
        start_time: float,
        cached_tokens: int = 0,
    ) -> None:
        """Record a summary attempt.

        Args:
            node_id: Node being summarized
            target_tokens: Target token count for summary
            input_text_tokens: Combined tokens from children being summarized
            prompt_tokens: Tokens in the prompt
            completion_tokens: Tokens in the completion (from API)
            actual_tokens: Actual tokens in summary text (measured)
            model: Model used for summary
            start_time: When the API call started
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

        # Store input_text_tokens at node level on first attempt
        node = self.node_telemetry[node_id]
        if node.input_text_tokens is None:
            node.input_text_tokens = input_text_tokens

        attempt = SummaryAttempt(
            target_tokens=target_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            actual_tokens=actual_tokens,
            model=model,
            start_time=start_time,
            end_time=time.time(),
            cached_tokens=cached_tokens,
        )
        self.node_telemetry[node_id].summary_attempts.append(attempt)

        self._update_memory_usage()

    def mark_accepted_attempt(self, node_id: str, attempt_index: int) -> None:
        """Mark which summary attempt was actually used for the node.

        Args:
            node_id: Node being summarized
            attempt_index: Index of the accepted attempt (0-based)
        """
        if node_id not in self.node_telemetry:
            logger.warning(
                f"Node {node_id} not found in telemetry for marking accepted attempt"
            )
            return

        self.node_telemetry[node_id].accepted_attempt = attempt_index

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

    def metadata_snapshot(self) -> dict[str, object]:
        """Build the immutable metadata payload for persistent telemetry."""

        metadata: dict[str, object] = {
            "format_version": TELEMETRY_FORMAT_VERSION,
            "document_id": self.document_id,
            "config": asdict(self.config),
            "model_metadata": self._get_model_metadata(),
            "system_prompts": self._get_system_prompts(),
            "runtime_info": self._get_runtime_info(),
        }

        if self.document_path:
            metadata["document_path"] = self.document_path

        return metadata

    def finalize(self) -> TelemetryDataDict:
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
            # Create fresh process object each time for thread safety
            memory_info = psutil.Process().memory_info()
            self.memory_end_mb = memory_info.rss / 1024 / 1024
        except Exception:
            self.memory_end_mb = self.peak_memory_mb

        # Return telemetry data in standard format
        # For telemetry, use target_embedding_tokens as fallback
        chunk_tokens = (
            self.config.target_chunk_tokens
            if self.config.target_chunk_tokens is not None
            else self.config.target_embedding_tokens
        )
        return self.get_telemetry_data(self.document_id, chunk_tokens)

    def get_telemetry_data(
        self, document_id: str, chunk_size: int
    ) -> TelemetryDataDict:
        """Export raw telemetry data in standard format for analysis.

        Args:
            document_id: Document identifier for the telemetry
            chunk_size: Chunk size used for indexing (for metadata)

        Returns:
            Dictionary in telemetry format v4.2 (flat structure)
        """
        # Convert all node telemetry to dict format
        nodes_data = []
        for node in self.node_telemetry.values():
            nodes_data.append(node.to_telemetry_dict())

        # Sort nodes by creation time for consistent output
        nodes_data.sort(key=lambda x: x["created_at"])

        # Save only the index config (the parameters that affect indexing)
        config_dict = asdict(self.config)

        # Capture model metadata for reproducibility
        model_metadata = self._get_model_metadata()

        # Capture system prompts used during indexing
        system_prompts = self._get_system_prompts()

        # Capture runtime environment info
        runtime_info = self._get_runtime_info()

        telemetry_data = {
            "format_version": TELEMETRY_FORMAT_VERSION,
            "document_id": document_id,
            "source_document_tokens": self.source_tokens,
            "indexed_at": self.start_time,
            "config": config_dict,
            "model_metadata": model_metadata,
            "system_prompts": system_prompts,
            "runtime_info": runtime_info,
            "nodes": nodes_data,
        }

        # Add document path if available
        if self.document_path:
            telemetry_data["document_path"] = self.document_path

        if self.append_metadata is not None:
            telemetry_data["append_metadata"] = self.append_metadata

        if self._chunk_split_info is not None:
            telemetry_data["chunk_split"] = dict(self._chunk_split_info)

        return cast(TelemetryDataDict, telemetry_data)

    def _get_model_metadata(self) -> ModelMetadataDict:
        """Get complete model metadata for reproducibility.

        Returns:
            Dictionary containing model capabilities, costs, and configuration
        """
        try:
            from ragzoom.model_info import ModelInfo

            model_info = ModelInfo()

            metadata: dict[str, object] = {}

            # Get embedding model metadata
            try:
                metadata["embedding"] = {
                    "model": self.config.embedding_model,
                    "dimensions": model_info.get_embedding_dimensions(
                        self.config.embedding_model
                    ),
                    "cost_per_1k": model_info.get_embedding_cost(
                        self.config.embedding_model
                    ),
                }
            except ValueError as e:
                logger.warning(f"Could not get embedding model metadata: {e}")
                metadata["embedding"] = {
                    "model": self.config.embedding_model,
                    "error": str(e),
                }

            # Get LLM model metadata
            try:
                input_cost, output_cost = model_info.get_llm_costs(
                    self.config.summary_model
                )
                supports_temperature = model_info.supports_temperature(
                    self.config.summary_model
                )
                is_gpt5 = model_info.is_gpt5_model(self.config.summary_model)

                metadata["summary"] = {
                    "model": self.config.summary_model,
                    "input_cost_per_1k": input_cost,
                    "output_cost_per_1k": output_cost,
                    "supports_temperature": supports_temperature,
                    "is_gpt5": is_gpt5,
                }

                # Get cache discount if available
                try:
                    cache_discount = model_info.get_cache_discount(
                        self.config.summary_model
                    )
                    cast(dict[str, object], metadata["summary"])[
                        "cache_discount"
                    ] = cache_discount
                except ValueError:
                    pass  # Cache discount not available for this model

            except ValueError as e:
                logger.warning(f"Could not get summary model metadata: {e}")
                metadata["summary"] = {
                    "model": self.config.summary_model,
                    "error": str(e),
                }

            # Add models.json metadata timestamp if available
            try:
                models_data = model_info._data
                if "last_updated" in models_data:
                    metadata["models_last_updated"] = str(models_data["last_updated"])
            except Exception as e:
                logger.debug(f"Failed to parse model metadata response: {e}")
                # Optional metadata, continue without it

            return cast(ModelMetadataDict, metadata)

        except Exception as e:
            logger.warning(f"Failed to collect model metadata: {e}")
            return {"error": str(e)}

    def _get_system_prompts(self) -> dict[str, str]:
        """Get system prompts used during indexing for reproducibility.

        Returns:
            Dictionary containing the actual system prompt used (custom or default)
        """
        return {
            "summary_system_prompt": (
                self.config.summary_system_prompt
                if self.config.summary_system_prompt is not None
                else DEFAULT_SUMMARY_SYSTEM_PROMPT
            )
        }

    def _get_runtime_info(self) -> RuntimeInfoDict:
        """Get runtime environment information for reproducibility.

        Returns:
            Dictionary containing runtime environment details
        """
        import platform
        import sys

        runtime_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "ragzoom_version": self._get_ragzoom_version(),
        }

        # Add key library versions that could affect results
        try:
            import tiktoken

            runtime_info["tiktoken_version"] = tiktoken.__version__
        except (ImportError, AttributeError):
            pass

        try:
            import openai

            runtime_info["openai_version"] = openai.__version__
        except (ImportError, AttributeError):
            pass

        return cast(RuntimeInfoDict, runtime_info)

    def _get_ragzoom_version(self) -> str:
        """Get RagZoom version for telemetry.

        Returns:
            Version string or 'unknown' if not available
        """
        try:
            from importlib.metadata import version

            return version("ragzoom")
        except Exception:
            return "unknown"
