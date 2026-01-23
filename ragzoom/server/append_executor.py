"""Server-side append pipeline that creates leaf nodes.

Leaf embedding is handled asynchronously via WorkerCoordinator.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING

from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.document_store import DocumentStore
from ragzoom.splitter import TextSplitter
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer
from ragzoom.wrapper import AppendUnit

logger = logging.getLogger(__name__)


def parse_timestamp(iso_string: str) -> float:
    """Parse an ISO 8601 timestamp string to Unix timestamp (float seconds).

    Args:
        iso_string: ISO 8601 formatted string with timezone info.
            Examples: "2024-01-21T14:30:00Z", "2024-01-21T14:30:00+00:00"

    Returns:
        Unix timestamp as float seconds since epoch.

    Raises:
        ValueError: If the string is not valid ISO 8601 or lacks timezone info.
    """
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"Invalid ISO 8601 timestamp format: {iso_string}") from e

    if dt.tzinfo is None:
        raise ValueError(
            f"Timestamp must include timezone info (e.g., 'Z' or '+00:00'): {iso_string}"
        )

    return dt.timestamp()


def validate_timestamp_range(*, time_start: float, time_end: float) -> None:
    """Validate that time_end >= time_start.

    Args:
        time_start: Start timestamp as Unix float seconds.
        time_end: End timestamp as Unix float seconds.

    Raises:
        ValueError: If time_end < time_start.
    """
    if time_end < time_start:
        raise ValueError(f"time_end ({time_end}) must be >= time_start ({time_start})")


def parse_timestamp_param(
    timestamp: str | tuple[str, str] | None,
) -> tuple[float, float] | None:
    """Parse a timestamp parameter into (time_start, time_end) tuple.

    Args:
        timestamp: One of:
            - None: Returns None
            - ISO 8601 string: Used for both start and end
            - Tuple of (start, end) ISO 8601 strings

    Returns:
        Tuple of (time_start, time_end) as Unix float seconds, or None.

    Raises:
        ValueError: If timestamp format is invalid or time_end < time_start.
    """
    if timestamp is None:
        return None

    if isinstance(timestamp, str):
        t = parse_timestamp(timestamp)
        return (t, t)

    if len(timestamp) != 2:
        raise ValueError(
            f"Timestamp tuple must have exactly 2 elements, got {len(timestamp)}"
        )
    time_start = parse_timestamp(timestamp[0])
    time_end = parse_timestamp(timestamp[1])
    validate_timestamp_range(time_start=time_start, time_end=time_end)
    return (time_start, time_end)


# Maximum characters per unit in client-managed chunking mode.
# Units exceeding this limit are truncated with a warning.
# Exposed as module constant for test overriding.
MAX_UNIT_CHARS = 50000


@dataclass
class LeafSpec:
    node_id: str
    text: str
    span_start: int
    span_end: int
    token_count: int
    preceding_neighbor_id: str | None
    following_neighbor_id: str | None
    level_index: int
    time_start: float | None = None
    time_end: float | None = None


@dataclass
class AppendOutcome:
    document_id: str
    appended_span_start: int
    appended_span_end: int
    new_leaf_ids: list[str]
    deleted_node_ids: list[str]
    total_leaves: int


class AppendExecutor:
    """Create new leaves for appended content.

    Embedding is handled asynchronously via WorkerCoordinator after append completes.
    """

    def __init__(
        self,
        config: IndexConfig,
        embedder: EmbeddingProvider,
        *,
        splitter: TextSplitter | None = None,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._splitter = splitter or TextSplitter(config)

    async def append(
        self,
        *,
        store: DocumentStore,
        document_id: str,
        new_text: str,
        timestamp: str | tuple[str, str] | None = None,
        reporter: TelemetryCollector | None = None,
        run_context: IndexRunContext | None = None,
        telemetry_manager: TelemetryRunManager | None = None,
    ) -> AppendOutcome:
        """Append new text as new leaf nodes without modifying existing leaves.

        This is an append-only operation: existing leaves are never deleted or
        modified. New leaves are created starting from the span_end of the
        rightmost existing leaf.
        """
        # Only reject empty text when not in client-managed chunking mode
        if not new_text and self._config.target_chunk_tokens is not None:
            raise ValueError("append requires non-empty text")

        # Parse timestamp parameter
        parsed = parse_timestamp_param(timestamp)
        time_start = parsed[0] if parsed else None
        time_end = parsed[1] if parsed else None

        # Client-managed chunking: truncate units > MAX_UNIT_CHARS
        if self._config.target_chunk_tokens is None and len(new_text) > MAX_UNIT_CHARS:
            logger.warning(
                "append[%s]: truncating unit from %d to %d characters "
                "(client-managed chunking mode)",
                document_id,
                len(new_text),
                MAX_UNIT_CHARS,
            )
            new_text = new_text[:MAX_UNIT_CHARS]

        right_leaf = store.nodes.get_rightmost_leaf_for_document(document_id)
        logger.debug(
            "append[%s]: starting append (new_text_chars=%d, has_existing=%s)",
            document_id,
            len(new_text),
            bool(right_leaf),
        )

        # Handle temporal document validation and inference
        is_first_append = right_leaf is None
        has_timestamps = timestamp is not None

        if is_first_append:
            # First append: infer is_temporal from presence of timestamps
            if has_timestamps:
                # Temporal documents require client-controlled chunking
                if self._config.target_chunk_tokens is not None:
                    raise ValueError(
                        "Temporal documents require target_chunk_tokens=null in config. "
                        "This setting preserves one-to-one mapping between input units and "
                        "leaf nodes, which is required for accurate timestamp-based queries. "
                        "Set 'target_chunk_tokens: null' in your config file or use "
                        "'--target-chunk-tokens null' on the CLI."
                    )
                store._doc_repo.set_document_is_temporal(document_id, is_temporal=True)
        else:
            # Subsequent append: validate timestamp presence matches document temporality
            is_temporal = store._doc_repo.get_document_is_temporal(document_id)
            if is_temporal is None:
                raise ValueError(
                    f"Document '{document_id}' not found. Cannot validate temporality."
                )
            if is_temporal and not has_timestamps:
                raise ValueError(
                    f"Document '{document_id}' is temporal and requires timestamps on all appends"
                )
            if not is_temporal and has_timestamps:
                raise ValueError(
                    f"Document '{document_id}' is non-temporal and does not accept timestamps"
                )

        # New leaves start where the existing content ends
        span_start = int(right_leaf.span_end) if right_leaf else 0
        start_level_index = (
            int(getattr(right_leaf, "level_index", 0)) + 1 if right_leaf else 0
        )

        split_start_time = time.time()

        if (
            telemetry_manager is not None
            and run_context is not None
            and run_context.collect_telemetry
        ):
            await telemetry_manager.log_chunk_event(
                run_context,
                event="chunk_split_started",
                new_text_chars=len(new_text),
                existing_tail_chars=0,
                combined_chars=len(new_text),
            )

        if reporter is not None:
            reporter.record_chunk_split_start(
                start_time=split_start_time,
                new_text_chars=len(new_text),
                existing_tail_chars=0,
                combined_chars=len(new_text),
            )

        # Split only the new text - don't touch existing content
        chunks = self._splitter.split_text(new_text)
        if not chunks:
            raise ValueError("splitter returned no chunks for append")

        # Build timestamps sequence: all chunks share the same timestamp
        chunk_timestamps: list[tuple[float, float] | None] | None = None
        if time_start is not None and time_end is not None:
            chunk_timestamps = [(time_start, time_end)] * len(chunks)

        leaf_specs = self._build_leaf_specs(
            chunks,
            span_start=span_start,
            preceding_neighbor_id=right_leaf.id if right_leaf else None,
            start_level_index=start_level_index,
            timestamps=chunk_timestamps,
        )
        logger.debug(
            "append[%s]: prepared %d leaf specs (span_start=%d)",
            document_id,
            len(leaf_specs),
            span_start,
        )

        split_end_time = time.time()
        total_leaf_tokens = sum(leaf.token_count for leaf in leaf_specs)

        if reporter is not None:
            reporter.record_chunk_split_end(
                end_time=split_end_time,
                chunk_count=len(leaf_specs),
                total_tokens=total_leaf_tokens,
            )

        if (
            telemetry_manager is not None
            and run_context is not None
            and run_context.collect_telemetry
        ):
            await telemetry_manager.log_chunk_event(
                run_context,
                event="chunk_split_completed",
                chunk_count=len(leaf_specs),
                duration=split_end_time - split_start_time,
                total_tokens=total_leaf_tokens,
            )

        if reporter is not None:
            for leaf in leaf_specs:
                reporter.track_node_created(
                    node_id=leaf.node_id,
                    height=0,
                    span=(leaf.span_start, leaf.span_end),
                )

        payload: list[NodeDataDict] = []
        for leaf in leaf_specs:
            payload.append(
                {
                    "node_id": leaf.node_id,
                    "text": leaf.text,
                    "span_start": leaf.span_start,
                    "span_end": leaf.span_end,
                    "parent_id": None,
                    "left_child_id": None,
                    "right_child_id": None,
                    "document_id": document_id,
                    "token_count": leaf.token_count,
                    "height": 0,
                    "preceding_neighbor_id": leaf.preceding_neighbor_id,
                    "following_neighbor_id": leaf.following_neighbor_id,
                    "level_index": leaf.level_index,
                    "time_start": leaf.time_start,
                    "time_end": leaf.time_end,
                }
            )

        # Update neighbor links: existing rightmost leaf -> first new leaf
        neighbor_updates: list[tuple[str, str | None, str | None]] = []
        if right_leaf is not None:
            neighbor_updates.append(
                (
                    right_leaf.id,
                    getattr(right_leaf, "preceding_neighbor_id", None),
                    leaf_specs[0].node_id,
                )
            )
        for leaf in leaf_specs:
            neighbor_updates.append(
                (
                    leaf.node_id,
                    leaf.preceding_neighbor_id,
                    leaf.following_neighbor_id,
                )
            )

        with store.transaction() as session:
            store.nodes.add_batch(payload, session=session)
            if neighbor_updates:
                store.nodes.update_neighbors_batch(neighbor_updates, session=session)

        logger.debug(
            "append[%s]: wrote %d leaves span=(%d,%d)",
            document_id,
            len(leaf_specs),
            leaf_specs[0].span_start,
            leaf_specs[-1].span_end,
        )

        affected_nodes = {leaf.node_id for leaf in leaf_specs}
        store.tree.clear_depth_cache(list(affected_nodes))

        total_leaves = store.nodes.leaf_count()
        appended_span_end = leaf_specs[-1].span_end

        if (
            telemetry_manager is not None
            and run_context is not None
            and run_context.collect_telemetry
        ):
            for leaf in leaf_specs:
                await telemetry_manager.record_node_committed(
                    run_context,
                    node_id=leaf.node_id,
                    height=0,
                    span_start=leaf.span_start,
                    span_end=leaf.span_end,
                )

        logger.debug(
            "append[%s]: completed append (new_total_leaves=%d, span_end=%d)",
            document_id,
            total_leaves,
            appended_span_end,
        )

        return AppendOutcome(
            document_id=document_id,
            appended_span_start=leaf_specs[0].span_start,
            appended_span_end=appended_span_end,
            new_leaf_ids=[leaf.node_id for leaf in leaf_specs],
            deleted_node_ids=[],
            total_leaves=total_leaves,
        )

    # jscpd:ignore-start - Parallel structure to append() intentional (batch vs single)
    async def append_batch(
        self,
        *,
        store: DocumentStore,
        document_id: str,
        units: Sequence[str] | Sequence[AppendUnit],
        timestamps: Sequence[str | tuple[str, str]] | None = None,
        reporter: TelemetryCollector | None = None,
        run_context: IndexRunContext | None = None,
        telemetry_manager: TelemetryRunManager | None = None,
    ) -> AppendOutcome:
        """Append multiple text units with forced split boundaries between them.

        Each unit in the batch creates a forced split boundary, meaning text is
        never merged across unit boundaries. This is semantically equivalent to
        calling append() for each unit sequentially, but executed in a single
        transaction for efficiency.

        Args:
            store: Document store to append to
            document_id: ID of the document
            units: Sequence of text units or AppendUnit objects. When using
                AppendUnit objects, timestamps are extracted from them and
                the timestamps parameter must be None.
            timestamps: Optional sequence parallel to units. Each entry can be:
                - ISO 8601 string (used for both time_start and time_end)
                - Tuple of (time_start, time_end) ISO 8601 strings
                Must match length of units when provided. Must be None when
                units contains AppendUnit objects.
            reporter: Optional telemetry collector
            run_context: Optional run context for telemetry
            telemetry_manager: Optional telemetry manager

        Returns:
            AppendOutcome with all new leaf IDs and span info

        Raises:
            ValueError: If timestamps length doesn't match units length, or if
                timestamps is provided when units contains AppendUnit objects.
        """
        # Normalize units: convert AppendUnit objects to (text, timestamps) format
        text_units, parsed_timestamps = self._normalize_units_input(
            units=units,
            timestamps=timestamps,
        )

        # Process units based on chunking mode
        non_empty_units, unit_timestamps = self._process_units_for_batch(
            units=text_units,
            parsed_timestamps=parsed_timestamps,
            document_id=document_id,
        )

        if not non_empty_units:
            # No content to append - return empty outcome
            total_leaves = store.nodes.leaf_count()
            right_leaf = store.nodes.get_rightmost_leaf_for_document(document_id)
            span_pos = int(right_leaf.span_end) if right_leaf else 0
            return AppendOutcome(
                document_id=document_id,
                appended_span_start=span_pos,
                appended_span_end=span_pos,
                new_leaf_ids=[],
                deleted_node_ids=[],
                total_leaves=total_leaves,
            )

        right_leaf = store.nodes.get_rightmost_leaf_for_document(document_id)
        logger.debug(
            "append_batch[%s]: starting batch append (units=%d, has_existing=%s)",
            document_id,
            len(non_empty_units),
            bool(right_leaf),
        )

        # Handle temporal document validation and inference
        is_first_append = right_leaf is None
        has_timestamps = parsed_timestamps is not None

        if is_first_append:
            # First append: infer is_temporal from presence of timestamps
            if has_timestamps:
                # Temporal documents require client-controlled chunking
                if self._config.target_chunk_tokens is not None:
                    raise ValueError(
                        "Temporal documents require target_chunk_tokens=null in config. "
                        "This setting preserves one-to-one mapping between input units and "
                        "leaf nodes, which is required for accurate timestamp-based queries. "
                        "Set 'target_chunk_tokens: null' in your config file or use "
                        "'--target-chunk-tokens null' on the CLI."
                    )
                store._doc_repo.set_document_is_temporal(document_id, is_temporal=True)
        else:
            # Subsequent append: validate timestamp presence matches document temporality
            is_temporal = store._doc_repo.get_document_is_temporal(document_id)
            if is_temporal is None:
                raise ValueError(
                    f"Document '{document_id}' not found. Cannot validate temporality."
                )
            if is_temporal and not has_timestamps:
                raise ValueError(
                    f"Document '{document_id}' is temporal and requires timestamps on all appends"
                )
            if not is_temporal and has_timestamps:
                raise ValueError(
                    f"Document '{document_id}' is non-temporal and does not accept timestamps"
                )

        # Track position across all units
        span_start = int(right_leaf.span_end) if right_leaf else 0
        initial_span_start = span_start
        level_index = (
            int(getattr(right_leaf, "level_index", 0)) + 1 if right_leaf else 0
        )
        preceding_id = right_leaf.id if right_leaf else None

        split_start_time = time.time()
        total_new_chars = sum(len(u) for u in non_empty_units)

        if (
            telemetry_manager is not None
            and run_context is not None
            and run_context.collect_telemetry
        ):
            await telemetry_manager.log_chunk_event(
                run_context,
                event="chunk_split_started",
                new_text_chars=total_new_chars,
                existing_tail_chars=0,
                combined_chars=total_new_chars,
            )

        if reporter is not None:
            reporter.record_chunk_split_start(
                start_time=split_start_time,
                new_text_chars=total_new_chars,
                existing_tail_chars=0,
                combined_chars=total_new_chars,
            )

        # Process each unit independently, writing to DB incrementally
        # This avoids accumulating O(total chunks) LeafSpec objects in memory
        all_new_ids: list[str] = []
        total_leaf_tokens = 0
        appended_span_end = initial_span_start
        prev_unit_last_id: str | None = None  # Last node ID from previous unit
        prev_unit_last_preceding_id: str | None = None  # Its preceding neighbor

        for unit_idx, unit_text in enumerate(non_empty_units):
            # Get timestamp for this unit (if provided)
            unit_timestamp = (
                unit_timestamps[unit_idx] if unit_timestamps is not None else None
            )
            time_start = unit_timestamp[0] if unit_timestamp else None
            time_end = unit_timestamp[1] if unit_timestamp else None

            # Split this unit (may produce 1+ chunks)
            chunks = self._splitter.split_text(unit_text)
            if not chunks:
                raise ValueError("splitter returned no chunks for unit in append_batch")

            # Build timestamps sequence: all chunks from this unit share the same timestamp
            chunk_timestamps: list[tuple[float, float] | None] | None = None
            if time_start is not None and time_end is not None:
                chunk_timestamps = [(time_start, time_end)] * len(chunks)

            # Build leaf specs for this unit's chunks
            unit_specs = self._build_leaf_specs(
                chunks,
                span_start=span_start,
                preceding_neighbor_id=preceding_id,
                start_level_index=level_index,
                timestamps=chunk_timestamps,
            )

            if not unit_specs:
                raise ValueError(
                    "build_leaf_specs returned no specs for unit in append_batch"
                )

            # Build payload for this unit
            payload: list[NodeDataDict] = []
            for leaf in unit_specs:
                payload.append(
                    {
                        "node_id": leaf.node_id,
                        "text": leaf.text,
                        "span_start": leaf.span_start,
                        "span_end": leaf.span_end,
                        "parent_id": None,
                        "left_child_id": None,
                        "right_child_id": None,
                        "document_id": document_id,
                        "token_count": leaf.token_count,
                        "height": 0,
                        "preceding_neighbor_id": leaf.preceding_neighbor_id,
                        "following_neighbor_id": leaf.following_neighbor_id,
                        "level_index": leaf.level_index,
                        "time_start": leaf.time_start,
                        "time_end": leaf.time_end,
                    }
                )

            # Build neighbor updates for this unit
            neighbor_updates: list[tuple[str, str | None, str | None]] = []

            # Link existing rightmost leaf to first new leaf (only for first unit)
            if right_leaf is not None and not all_new_ids:
                neighbor_updates.append(
                    (
                        right_leaf.id,
                        getattr(right_leaf, "preceding_neighbor_id", None),
                        unit_specs[0].node_id,
                    )
                )

            # Link previous unit's last node to this unit's first node
            if prev_unit_last_id is not None:
                neighbor_updates.append(
                    (
                        prev_unit_last_id,
                        prev_unit_last_preceding_id,
                        unit_specs[0].node_id,
                    )
                )

            # Add neighbor updates for all nodes in this unit
            for leaf in unit_specs:
                neighbor_updates.append(
                    (
                        leaf.node_id,
                        leaf.preceding_neighbor_id,
                        leaf.following_neighbor_id,
                    )
                )

            # Write this unit to DB
            with store.transaction() as session:
                store.nodes.add_batch(payload, session=session)
                if neighbor_updates:
                    store.nodes.update_neighbors_batch(
                        neighbor_updates, session=session
                    )

            # Track telemetry for this unit
            if reporter is not None:
                for leaf in unit_specs:
                    reporter.track_node_created(
                        node_id=leaf.node_id,
                        height=0,
                        span=(leaf.span_start, leaf.span_end),
                    )

            if (
                telemetry_manager is not None
                and run_context is not None
                and run_context.collect_telemetry
            ):
                for leaf in unit_specs:
                    await telemetry_manager.record_node_committed(
                        run_context,
                        node_id=leaf.node_id,
                        height=0,
                        span_start=leaf.span_start,
                        span_end=leaf.span_end,
                    )

            # Update tracking for next iteration
            all_new_ids.extend(leaf.node_id for leaf in unit_specs)
            total_leaf_tokens += sum(leaf.token_count for leaf in unit_specs)
            last_spec = unit_specs[-1]
            appended_span_end = last_spec.span_end
            span_start = last_spec.span_end
            preceding_id = last_spec.node_id
            prev_unit_last_id = last_spec.node_id
            prev_unit_last_preceding_id = last_spec.preceding_neighbor_id
            level_index = last_spec.level_index + 1
            # unit_specs goes out of scope here, memory freed

        if not all_new_ids:
            # Splitter returned no chunks for any unit
            total_leaves = store.nodes.leaf_count()
            return AppendOutcome(
                document_id=document_id,
                appended_span_start=initial_span_start,
                appended_span_end=initial_span_start,
                new_leaf_ids=[],
                deleted_node_ids=[],
                total_leaves=total_leaves,
            )

        logger.debug(
            "append_batch[%s]: wrote %d leaves from %d units",
            document_id,
            len(all_new_ids),
            len(non_empty_units),
        )

        split_end_time = time.time()

        if reporter is not None:
            reporter.record_chunk_split_end(
                end_time=split_end_time,
                chunk_count=len(all_new_ids),
                total_tokens=total_leaf_tokens,
            )

        if (
            telemetry_manager is not None
            and run_context is not None
            and run_context.collect_telemetry
        ):
            await telemetry_manager.log_chunk_event(
                run_context,
                event="chunk_split_completed",
                chunk_count=len(all_new_ids),
                duration=split_end_time - split_start_time,
                total_tokens=total_leaf_tokens,
            )

        store.tree.clear_depth_cache(all_new_ids)

        total_leaves = store.nodes.leaf_count()

        logger.debug(
            "append_batch[%s]: completed batch append (new_total_leaves=%d)",
            document_id,
            total_leaves,
        )

        return AppendOutcome(
            document_id=document_id,
            appended_span_start=initial_span_start,
            appended_span_end=appended_span_end,
            new_leaf_ids=all_new_ids,
            deleted_node_ids=[],
            total_leaves=total_leaves,
        )

    # jscpd:ignore-end

    def _normalize_units_input(
        self,
        *,
        units: Sequence[str] | Sequence[AppendUnit],
        timestamps: Sequence[str | tuple[str, str]] | None,
    ) -> tuple[list[str], list[tuple[float, float] | None] | None]:
        """Normalize units input to (text_list, parsed_timestamps) format.

        Handles both the legacy API (list[str] + timestamps) and the new API
        (list[AppendUnit] with embedded timestamps).

        Args:
            units: Either a sequence of strings or AppendUnit objects.
            timestamps: Optional timestamps for string units. Must be None
                when units contains AppendUnit objects.

        Returns:
            Tuple of (text_units, parsed_timestamps) where text_units is a list
            of strings and parsed_timestamps is either None or a list of
            (time_start, time_end) tuples.

        Raises:
            ValueError: If timestamps is provided with AppendUnit objects, or
                if timestamps length doesn't match units length.
        """
        if not units:
            return ([], None)

        # Check if units are AppendUnit objects
        first_unit = units[0]
        if isinstance(first_unit, AppendUnit):
            # New API: extract text and timestamps from AppendUnit objects
            if timestamps is not None:
                raise ValueError(
                    "timestamps parameter must be None when using AppendUnit objects "
                    "(timestamps are embedded in AppendUnit)"
                )

            text_units: list[str] = []
            parsed_timestamps: list[tuple[float, float] | None] = []
            has_any_timestamps = False

            for unit in units:
                if not isinstance(unit, AppendUnit):
                    raise ValueError(
                        "All units must be AppendUnit objects when first unit is AppendUnit"
                    )
                text_units.append(unit.text)

                if unit.is_temporal:
                    # AppendUnit validates that both are set or neither
                    assert unit.time_start is not None and unit.time_end is not None
                    unit_ts = parse_timestamp_param((unit.time_start, unit.time_end))
                    parsed_timestamps.append(unit_ts)
                    has_any_timestamps = True
                else:
                    parsed_timestamps.append(None)

            # Return timestamps list only if any unit had timestamps
            final_timestamps = parsed_timestamps if has_any_timestamps else None
            return (text_units, final_timestamps)

        # Legacy API: units are strings
        # Validate timestamps length matches units
        if timestamps is not None and len(timestamps) != len(units):
            raise ValueError(
                f"timestamps length ({len(timestamps)}) must match "
                f"units length ({len(units)})"
            )

        # Cast to list[str] since we verified first element is str
        string_units = [str(u) for u in units]

        # Parse all timestamps upfront (before filtering which changes indices)
        parsed = (
            [parse_timestamp_param(ts) for ts in timestamps]
            if timestamps is not None
            else None
        )
        return (string_units, parsed)

    def _build_leaf_specs(
        self,
        chunks: Sequence[str],
        *,
        span_start: int,
        preceding_neighbor_id: str | None,
        start_level_index: int,
        timestamps: Sequence[tuple[float, float] | None] | None = None,
    ) -> list[LeafSpec]:
        """Build leaf specs for new chunks starting at span_start.

        Args:
            chunks: Text chunks to create leaf specs for.
            span_start: Starting character position in document.
            preceding_neighbor_id: ID of leaf node that precedes these new leaves.
            start_level_index: Starting level index for the new leaves.
            timestamps: Per-chunk timestamps as (time_start, time_end) tuples.
                If None, all chunks get None timestamps.
                If provided, length must match chunks length.
        """
        if timestamps is not None and len(timestamps) != len(chunks):
            raise ValueError(
                f"timestamps length ({len(timestamps)}) must match chunks length "
                f"({len(chunks)})"
            )

        specs: list[LeafSpec] = []
        span_cursor = span_start

        for index, chunk in enumerate(chunks):
            node_id = str(uuid.uuid4())
            span_end = span_cursor + len(chunk)
            token_count = tokenizer.count_tokens(chunk)

            # Neighbor links: chain leaves together, first links to preceding_neighbor_id
            prev_id = specs[index - 1].node_id if index > 0 else preceding_neighbor_id

            # Get timestamp for this chunk (or None if no timestamps provided)
            chunk_timestamp = timestamps[index] if timestamps is not None else None
            time_start = chunk_timestamp[0] if chunk_timestamp else None
            time_end = chunk_timestamp[1] if chunk_timestamp else None

            specs.append(
                LeafSpec(
                    node_id=node_id,
                    text=chunk,
                    span_start=span_cursor,
                    span_end=span_end,
                    token_count=token_count,
                    preceding_neighbor_id=prev_id,
                    following_neighbor_id=None,  # Set in second pass
                    level_index=start_level_index + index,
                    time_start=time_start,
                    time_end=time_end,
                )
            )
            span_cursor = span_end

        # Set following_neighbor_id links
        for idx in range(len(specs) - 1):
            specs[idx] = replace(
                specs[idx], following_neighbor_id=specs[idx + 1].node_id
            )

        return specs

    def _process_units_for_batch(
        self,
        *,
        units: Sequence[str],
        parsed_timestamps: list[tuple[float, float] | None] | None,
        document_id: str,
    ) -> tuple[list[str], list[tuple[float, float] | None] | None]:
        """Process units for batch append, handling truncation or filtering.

        In client-managed chunking mode (target_chunk_tokens is None), truncates
        oversized units but preserves all units including empty ones.

        In normal mode, filters out empty/whitespace-only units.

        Returns:
            Tuple of (processed_units, corresponding_timestamps).
        """
        result_units: list[str] = []
        result_timestamps: list[tuple[float, float] | None] = []

        if self._config.target_chunk_tokens is None:
            # Client-managed chunking: truncate oversized units, preserve all
            for i, unit in enumerate(units):
                if len(unit) > MAX_UNIT_CHARS:
                    logger.warning(
                        "append_batch[%s]: truncating unit from %d to %d characters "
                        "(client-managed chunking mode)",
                        document_id,
                        len(unit),
                        MAX_UNIT_CHARS,
                    )
                    result_units.append(unit[:MAX_UNIT_CHARS])
                else:
                    result_units.append(unit)
                if parsed_timestamps is not None:
                    result_timestamps.append(parsed_timestamps[i])
        else:
            # Normal mode: filter out empty units
            for i, unit in enumerate(units):
                if unit and unit.strip():
                    result_units.append(unit)
                    if parsed_timestamps is not None:
                        result_timestamps.append(parsed_timestamps[i])

        final_timestamps = result_timestamps if parsed_timestamps is not None else None
        return (result_units, final_timestamps)


if TYPE_CHECKING:
    from ragzoom.server.run_manager import IndexRunContext, TelemetryRunManager
