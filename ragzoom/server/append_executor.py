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
    # Data for async embedding (leaves no longer embedded during append)
    leaf_texts: list[str]
    leaf_metadata: list[dict[str, object]]


class AppendExecutor:
    """Create new leaves for appended content.

    Embedding is handled asynchronously via WorkerCoordinator after append completes.
    The AppendOutcome includes leaf_texts and leaf_metadata for queuing embedding work.
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

        # Build metadata for async embedding
        leaf_metadata: list[dict[str, object]] = []
        for leaf in leaf_specs:
            leaf_metadata.append(
                {
                    "document_id": document_id,
                    "span_start": leaf.span_start,
                    "span_end": leaf.span_end,
                    "is_leaf": 1,
                    "height": 0,
                    "level_index": leaf.level_index,
                    "coord_version": 1,
                }
            )

        return AppendOutcome(
            document_id=document_id,
            appended_span_start=leaf_specs[0].span_start,
            appended_span_end=appended_span_end,
            new_leaf_ids=[leaf.node_id for leaf in leaf_specs],
            deleted_node_ids=[],
            total_leaves=total_leaves,
            leaf_texts=[leaf.text for leaf in leaf_specs],
            leaf_metadata=leaf_metadata,
        )

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
