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
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex
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
        vector_index: VectorIndex,
        document_id: str,
        new_text: str,
        reporter: TelemetryCollector | None = None,
        run_context: IndexRunContext | None = None,
        telemetry_manager: TelemetryRunManager | None = None,
    ) -> AppendOutcome:
        if not new_text:
            raise ValueError("append requires non-empty text")

        right_leaf = store.nodes.get_rightmost_leaf_for_document(document_id)
        logger.debug(
            "append[%s]: starting append (new_text_chars=%d, replace_leaf=%s)",
            document_id,
            len(new_text),
            bool(right_leaf),
        )
        tail_start = int(right_leaf.span_start) if right_leaf else 0
        preceding_neighbor = (
            getattr(right_leaf, "preceding_neighbor_id", None) if right_leaf else None
        )
        following_neighbor = (
            getattr(right_leaf, "following_neighbor_id", None) if right_leaf else None
        )

        existing_tail_text = right_leaf.text or "" if right_leaf else ""
        combined_text = existing_tail_text + new_text
        if not combined_text:
            raise ValueError("append produced no text to index")

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
                existing_tail_chars=len(existing_tail_text),
                combined_chars=len(combined_text),
            )

        if reporter is not None:
            reporter.record_chunk_split_start(
                start_time=split_start_time,
                new_text_chars=len(new_text),
                existing_tail_chars=len(existing_tail_text),
                combined_chars=len(combined_text),
            )

        chunks = self._splitter.split_text(combined_text)
        if not chunks:
            raise ValueError("splitter returned no chunks for append")

        leaf_specs = self._build_leaf_specs(
            chunks,
            tail_start=tail_start,
            first_leaf_id=right_leaf.id if right_leaf else None,
            preceding_neighbor_id=preceding_neighbor,
            following_neighbor_id=following_neighbor,
            start_level_index=(
                int(getattr(right_leaf, "level_index", 0)) if right_leaf else 0
            ),
        )
        logger.debug(
            "append[%s]: prepared %d leaf specs (tail_start=%d, first_leaf=%s)",
            document_id,
            len(leaf_specs),
            tail_start,
            right_leaf.id if right_leaf else None,
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

        deleted_node_ids = self._collect_deletion_ids(store, right_leaf)

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

        neighbor_updates = self._build_neighbor_updates(
            store,
            leaf_specs,
            preceding_neighbor,
            following_neighbor,
        )

        with store.transaction() as session:
            if deleted_node_ids:
                store.nodes.delete_nodes(deleted_node_ids, session=session)
            store.nodes.add_batch(payload, session=session)
            if neighbor_updates:
                store.nodes.update_neighbors_batch(neighbor_updates, session=session)

            # Clean up vectors for deleted nodes
            if deleted_node_ids:
                self._delete_vectors(vector_index, deleted_node_ids)

        logger.debug(
            "append[%s]: wrote %d leaves (deleted=%d) span=(%d,%d)",
            document_id,
            len(leaf_specs),
            len(deleted_node_ids),
            leaf_specs[0].span_start,
            leaf_specs[-1].span_end,
        )

        affected_nodes = set(deleted_node_ids)
        affected_nodes.update(leaf.node_id for leaf in leaf_specs)
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
            deleted_node_ids=deleted_node_ids,
            total_leaves=total_leaves,
            leaf_texts=[leaf.text for leaf in leaf_specs],
            leaf_metadata=leaf_metadata,
        )

    def _build_leaf_specs(
        self,
        chunks: Sequence[str],
        *,
        tail_start: int,
        first_leaf_id: str | None,
        preceding_neighbor_id: str | None,
        following_neighbor_id: str | None,
        start_level_index: int,
    ) -> list[LeafSpec]:
        specs: list[LeafSpec] = []
        span_cursor = tail_start

        for index, chunk in enumerate(chunks):
            node_id = (
                first_leaf_id if index == 0 and first_leaf_id else str(uuid.uuid4())
            )
            span_end = span_cursor + len(chunk)
            token_count = tokenizer.count_tokens(chunk)

            specs.append(
                LeafSpec(
                    node_id=node_id,
                    text=chunk,
                    span_start=span_cursor,
                    span_end=span_end,
                    token_count=token_count,
                    preceding_neighbor_id=None,
                    following_neighbor_id=None,
                    level_index=start_level_index + index,
                )
            )
            span_cursor = span_end

        for idx, leaf in enumerate(specs):
            prev_id = specs[idx - 1].node_id if idx > 0 else preceding_neighbor_id
            next_id = (
                specs[idx + 1].node_id
                if idx + 1 < len(specs)
                else following_neighbor_id
            )
            specs[idx] = LeafSpec(
                node_id=leaf.node_id,
                text=leaf.text,
                span_start=leaf.span_start,
                span_end=leaf.span_end,
                token_count=leaf.token_count,
                preceding_neighbor_id=prev_id,
                following_neighbor_id=next_id,
                level_index=leaf.level_index,
            )

        return specs

    def _collect_deletion_ids(
        self,
        store: DocumentStore,
        right_leaf: TreeNode | None,
    ) -> list[str]:
        if right_leaf is None:
            return []
        to_delete: list[str] = []
        current: TreeNode | None = right_leaf
        visited: set[str] = set()
        while current is not None and current.id not in visited:
            visited.add(current.id)
            to_delete.append(current.id)
            parent_id = getattr(current, "parent_id", None)
            if not parent_id:
                inferred_parent = self._infer_structural_parent(store, current)
                if inferred_parent is None:
                    break
                current = inferred_parent
                continue
            current = store.nodes.get(parent_id)
        return to_delete

    def _infer_structural_parent(
        self, store: DocumentStore, node: TreeNode
    ) -> TreeNode | None:
        """Infer the parent by structural metadata when parent_id is missing."""

        level_index = int(getattr(node, "level_index", 0))
        height = int(getattr(node, "height", 0))
        parent_height = height + 1
        if parent_height <= height:
            return None

        parent_level_index = level_index // 2
        inferred = store.nodes.get_by_height_and_level(
            height=parent_height, level_index=parent_level_index
        )
        if inferred is None:
            return None
        if getattr(inferred, "document_id", None) != getattr(node, "document_id", None):
            return None
        if inferred.id == node.id:
            return None
        return inferred

    def _build_neighbor_updates(
        self,
        store: DocumentStore,
        leaves: Sequence[LeafSpec],
        preceding_neighbor: str | None,
        following_neighbor: str | None,
    ) -> list[tuple[str, str | None, str | None]]:
        updates: list[tuple[str, str | None, str | None]] = []
        if preceding_neighbor:
            prev_node = store.nodes.get(preceding_neighbor)
            if prev_node is not None:
                updates.append(
                    (
                        preceding_neighbor,
                        getattr(prev_node, "preceding_neighbor_id", None),
                        leaves[0].node_id,
                    )
                )
        if following_neighbor:
            next_node = store.nodes.get(following_neighbor)
            if next_node is not None:
                updates.append(
                    (
                        following_neighbor,
                        leaves[-1].node_id,
                        getattr(next_node, "following_neighbor_id", None),
                    )
                )
        for leaf in leaves:
            updates.append(
                (
                    leaf.node_id,
                    leaf.preceding_neighbor_id,
                    leaf.following_neighbor_id,
                )
            )
        return updates

    @staticmethod
    def _delete_vectors(vector_index: VectorIndex, node_ids: Sequence[str]) -> None:
        if not node_ids:
            return
        try:
            vector_index.delete(ids=list(node_ids))
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to delete vectors during append")


if TYPE_CHECKING:
    from ragzoom.server.run_manager import IndexRunContext, TelemetryRunManager
