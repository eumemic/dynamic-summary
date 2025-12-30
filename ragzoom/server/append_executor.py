"""Server-side append pipeline that creates leaf nodes.

Leaf embedding is handled asynchronously via WorkerCoordinator.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragzoom.config import IndexConfig
from ragzoom.contracts.embedding_model import EmbeddingProvider
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.document_store import DocumentStore
from ragzoom.splitter import TextSplitter
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


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
        reporter: TelemetryCollector | None = None,
        run_context: IndexRunContext | None = None,
        telemetry_manager: TelemetryRunManager | None = None,
    ) -> AppendOutcome:
        """Append new text as new leaf nodes without modifying existing leaves.

        This is an append-only operation: existing leaves are never deleted or
        modified. New leaves are created starting from the span_end of the
        rightmost existing leaf.
        """
        if not new_text:
            raise ValueError("append requires non-empty text")

        right_leaf = store.nodes.get_rightmost_leaf_for_document(document_id)
        logger.debug(
            "append[%s]: starting append (new_text_chars=%d, has_existing=%s)",
            document_id,
            len(new_text),
            bool(right_leaf),
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

        leaf_specs = self._build_leaf_specs(
            chunks,
            span_start=span_start,
            preceding_neighbor_id=right_leaf.id if right_leaf else None,
            start_level_index=start_level_index,
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
        units: Sequence[str],
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
            units: Sequence of text units, each creating a forced boundary
            reporter: Optional telemetry collector
            run_context: Optional run context for telemetry
            telemetry_manager: Optional telemetry manager

        Returns:
            AppendOutcome with all new leaf IDs and span info
        """
        # Filter out empty units
        non_empty_units = [u for u in units if u and u.strip()]

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

        # Process each unit independently - this creates forced boundaries
        all_leaf_specs: list[LeafSpec] = []
        for unit_text in non_empty_units:
            # Split this unit (may produce 1+ chunks)
            chunks = self._splitter.split_text(unit_text)
            if not chunks:
                continue

            # Build leaf specs for this unit's chunks
            unit_specs = self._build_leaf_specs(
                chunks,
                span_start=span_start,
                preceding_neighbor_id=preceding_id,
                start_level_index=level_index,
            )

            if unit_specs:
                all_leaf_specs.extend(unit_specs)
                last_spec = unit_specs[-1]
                span_start = last_spec.span_end
                preceding_id = last_spec.node_id
                level_index = last_spec.level_index + 1

        if not all_leaf_specs:
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

        # Fix following_neighbor_id links across unit boundaries
        # _build_leaf_specs sets following_neighbor_id within each unit's specs,
        # but the last spec of each unit has following_neighbor_id=None.
        # We need to link them to the next spec.
        for idx in range(len(all_leaf_specs) - 1):
            if all_leaf_specs[idx].following_neighbor_id is None:
                all_leaf_specs[idx] = LeafSpec(
                    node_id=all_leaf_specs[idx].node_id,
                    text=all_leaf_specs[idx].text,
                    span_start=all_leaf_specs[idx].span_start,
                    span_end=all_leaf_specs[idx].span_end,
                    token_count=all_leaf_specs[idx].token_count,
                    preceding_neighbor_id=all_leaf_specs[idx].preceding_neighbor_id,
                    following_neighbor_id=all_leaf_specs[idx + 1].node_id,
                    level_index=all_leaf_specs[idx].level_index,
                )

        logger.debug(
            "append_batch[%s]: prepared %d leaf specs from %d units",
            document_id,
            len(all_leaf_specs),
            len(non_empty_units),
        )

        split_end_time = time.time()
        total_leaf_tokens = sum(leaf.token_count for leaf in all_leaf_specs)

        if reporter is not None:
            reporter.record_chunk_split_end(
                end_time=split_end_time,
                chunk_count=len(all_leaf_specs),
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
                chunk_count=len(all_leaf_specs),
                duration=split_end_time - split_start_time,
                total_tokens=total_leaf_tokens,
            )

        if reporter is not None:
            for leaf in all_leaf_specs:
                reporter.track_node_created(
                    node_id=leaf.node_id,
                    height=0,
                    span=(leaf.span_start, leaf.span_end),
                )

        # Build payload for batch insert
        payload: list[NodeDataDict] = []
        for leaf in all_leaf_specs:
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
                }
            )

        # Update neighbor links: existing rightmost leaf -> first new leaf
        neighbor_updates: list[tuple[str, str | None, str | None]] = []
        if right_leaf is not None:
            neighbor_updates.append(
                (
                    right_leaf.id,
                    getattr(right_leaf, "preceding_neighbor_id", None),
                    all_leaf_specs[0].node_id,
                )
            )
        for leaf in all_leaf_specs:
            neighbor_updates.append(
                (
                    leaf.node_id,
                    leaf.preceding_neighbor_id,
                    leaf.following_neighbor_id,
                )
            )

        # Store all leaves in a single transaction
        with store.transaction() as session:
            store.nodes.add_batch(payload, session=session)
            if neighbor_updates:
                store.nodes.update_neighbors_batch(neighbor_updates, session=session)

        logger.debug(
            "append_batch[%s]: wrote %d leaves span=(%d,%d)",
            document_id,
            len(all_leaf_specs),
            all_leaf_specs[0].span_start,
            all_leaf_specs[-1].span_end,
        )

        affected_nodes = {leaf.node_id for leaf in all_leaf_specs}
        store.tree.clear_depth_cache(list(affected_nodes))

        total_leaves = store.nodes.leaf_count()
        appended_span_end = all_leaf_specs[-1].span_end

        if (
            telemetry_manager is not None
            and run_context is not None
            and run_context.collect_telemetry
        ):
            for leaf in all_leaf_specs:
                await telemetry_manager.record_node_committed(
                    run_context,
                    node_id=leaf.node_id,
                    height=0,
                    span_start=leaf.span_start,
                    span_end=leaf.span_end,
                )

        logger.debug(
            "append_batch[%s]: completed batch append (new_total_leaves=%d)",
            document_id,
            total_leaves,
        )

        return AppendOutcome(
            document_id=document_id,
            appended_span_start=initial_span_start,
            appended_span_end=appended_span_end,
            new_leaf_ids=[leaf.node_id for leaf in all_leaf_specs],
            deleted_node_ids=[],
            total_leaves=total_leaves,
        )

    # jscpd:ignore-end

    def _build_leaf_specs(
        self,
        chunks: Sequence[str],
        *,
        span_start: int,
        preceding_neighbor_id: str | None,
        start_level_index: int,
    ) -> list[LeafSpec]:
        """Build leaf specs for new chunks starting at span_start."""
        specs: list[LeafSpec] = []
        span_cursor = span_start

        for index, chunk in enumerate(chunks):
            node_id = str(uuid.uuid4())
            span_end = span_cursor + len(chunk)
            token_count = tokenizer.count_tokens(chunk)

            # Neighbor links: chain leaves together, first links to preceding_neighbor_id
            prev_id = specs[index - 1].node_id if index > 0 else preceding_neighbor_id

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
                )
            )
            span_cursor = span_end

        # Set following_neighbor_id links
        for idx in range(len(specs) - 1):
            specs[idx] = LeafSpec(
                node_id=specs[idx].node_id,
                text=specs[idx].text,
                span_start=specs[idx].span_start,
                span_end=specs[idx].span_end,
                token_count=specs[idx].token_count,
                preceding_neighbor_id=specs[idx].preceding_neighbor_id,
                following_neighbor_id=specs[idx + 1].node_id,
                level_index=specs[idx].level_index,
            )

        return specs


if TYPE_CHECKING:
    from ragzoom.server.run_manager import IndexRunContext, TelemetryRunManager
