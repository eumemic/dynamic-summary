"""Service for building coverage maps during retrieval."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragzoom.backends.vector_common import coerce_int
from ragzoom.error_handling import handle_graceful_error
from ragzoom.tree_coordinate import TreeCoordinate

if TYPE_CHECKING:
    from ragzoom.contracts.tree_node import TreeNode
    from ragzoom.document_store import DocumentStore
    from ragzoom.vector_api import MetaDict

logger = logging.getLogger(__name__)


@dataclass
class CoverageResult:
    """Container for coverage map construction output."""

    coverage_map: dict[str, bool]
    nodes: dict[str, TreeNode]


@dataclass
class WindowBounds:
    """Computed window boundaries aligned to leaf node spans.

    Edge-max coordinates are the highest ancestors of the boundary leaves
    that stay entirely within the window. They ensure full coverage of the
    window edges even when no seeds are present there.
    """

    actual_start: int
    actual_end: int
    left_leaf_index: int
    right_leaf_index: int
    left_edge_max: TreeCoordinate
    right_edge_max: TreeCoordinate


class CoverageBuilder:
    """Builds coverage maps including selected nodes, ancestors, and siblings."""

    def __init__(self, store: DocumentStore):
        self.store = store

    def _get_document_pinned_nodes(self) -> list[TreeNode]:
        """Fetch pinned nodes scoped to this document.

        Returns:
            List of pinned TreeNode objects for the current document.
        """
        depth_max = getattr(self.store, "PIN_DEPTH_MAX", 2)
        if hasattr(self.store, "get_pinned_nodes"):
            return list(self.store.get_pinned_nodes(depth_max))
        # Fallback for stores without get_pinned_nodes method
        nodes_wrapper = getattr(self.store, "nodes", None)
        repo = getattr(nodes_wrapper, "_repo", None)
        document_id = getattr(self.store, "document_id", None)
        if repo is not None and document_id is not None:
            all_pinned = repo.get_pinned_nodes(depth_max)
            return [n for n in all_pinned if n.document_id == document_id]
        return []

    def build_complete_coverage_map(
        self,
        selected_ids: list[str],
        *,
        seed_metadata: Mapping[str, MetaDict] | None = None,
    ) -> dict[str, bool]:
        """Backward-compatible wrapper returning only the coverage map."""

        return self.build_complete_coverage(
            selected_ids, seed_metadata=seed_metadata
        ).coverage_map

    def build_complete_coverage(
        self,
        selected_ids: list[str],
        *,
        seed_metadata: Mapping[str, MetaDict] | None = None,
        pinned_ids: set[str] | None = None,
    ) -> CoverageResult:
        """Build coverage for the full document.

        This is equivalent to build_windowed_coverage with default span bounds
        (0 to document end).

        Args:
            selected_ids: Node IDs to include in coverage (seeds)
            seed_metadata: Optional metadata for seeds (for coordinate extraction)
            pinned_ids: Node IDs that are pinned and should not have ancestors
                in coverage. This makes each pinned node a root of its own tree,
                avoiding unnecessary ancestor fetches and scoring.
        """
        document_id = getattr(self.store, "document_id", None)
        if document_id is None:
            raise ValueError("Cannot build complete coverage: no document_id on store")

        return self.build_windowed_coverage(
            selected_ids,
            document_id,
            seed_metadata=seed_metadata,
            pinned_ids=pinned_ids,
        )

    def build_coverage_map(
        self,
        selected_ids: list[str],
        *,
        seed_metadata: Mapping[str, MetaDict] | None = None,
    ) -> dict[str, bool]:
        """Build coverage map including selected nodes and structural relatives."""

        if not selected_ids:
            return {}
        coverage_map, _ = self._build_coordinate_closure(selected_ids, seed_metadata)
        return coverage_map

    def _filter_pinned_ancestors(
        self,
        coords: list[TreeCoordinate],
        pinned_coords: list[TreeCoordinate],
        max_height: int,
    ) -> list[TreeCoordinate]:
        """Remove ancestors of pinned nodes from coordinate list.

        Pinned nodes should be roots of their own trees - they never roll up.
        By removing their ancestors from coverage, we:
        1. Skip unnecessary fetches for nodes that won't participate in tiling
        2. Eliminate scoring overhead for those nodes
        3. Make pinned nodes natural roots (parent not in coverage)

        Document roots are added to the coordinate pool before this filter,
        so roots that are ancestors of pinned nodes are filtered naturally.
        """
        if not pinned_coords:
            return coords

        # Build set of coordinates to remove (ancestors of pinned nodes)
        to_remove: set[tuple[int, int]] = set()

        for pinned in pinned_coords:
            current = pinned.parent()
            while current.height <= max_height:
                key = current.as_tuple()
                if key in to_remove:
                    # Another pinned node already covered this ancestor chain
                    break
                to_remove.add(key)
                current = current.parent()

        # Filter out ancestors (siblings are preserved)
        return [c for c in coords if c.as_tuple() not in to_remove]

    def _build_coordinate_closure(
        self,
        selected_ids: list[str],
        seed_metadata: Mapping[str, MetaDict] | None = None,
        *,
        pinned_ids: set[str] | None = None,
        window_bounds: WindowBounds | None = None,
    ) -> tuple[dict[str, bool], dict[str, TreeNode]]:
        """Build coverage by computing coordinate closure for seeds.

        This is the core coverage algorithm. It:
        1. Extracts coordinates from seed metadata or fetches missing seeds
        2. Adds synthetic seeds (roots for full-doc, edge-max for windowed)
        3. Computes ancestor/sibling coordinates for all seeds
        4. Optionally filters by window bounds
        5. Fetches nodes for all coordinates

        Args:
            selected_ids: Node IDs to include in coverage (seeds)
            seed_metadata: Optional metadata for seeds (for coordinate extraction)
            pinned_ids: Node IDs that are pinned (their ancestors are excluded)
            window_bounds: If provided, restricts coverage to the window and uses
                edge-max instead of roots as synthetic seeds.

        Returns:
            Tuple of (coverage_map, nodes dict).
        """
        coverage_map: dict[str, bool] = {}
        nodes: dict[str, TreeNode] = {}

        document_id = getattr(self.store, "document_id", None)
        meta_map = seed_metadata or {}

        # Extract coordinates from metadata where available
        meta_seed_coords: dict[str, TreeCoordinate] = {}
        missing_seed_ids: list[str] = []
        for node_id in selected_ids:
            meta = meta_map.get(node_id)
            if (
                not meta
                or meta.get("coord_version") != 1
                or "height" not in meta
                or "level_index" not in meta
            ):
                missing_seed_ids.append(node_id)
                continue
            coord_doc = str(meta.get("document_id", "")) or document_id
            if coord_doc is None:
                missing_seed_ids.append(node_id)
                continue
            coord = TreeCoordinate(
                document_id=coord_doc,
                height=coerce_int(meta.get("height", 0)),
                level_index=coerce_int(meta.get("level_index", 0)),
            )
            meta_seed_coords[node_id] = coord

        # Fetch nodes for seeds without metadata coordinates
        try:
            existing_nodes = (
                self.store.nodes.get_nodes(missing_seed_ids) if missing_seed_ids else []
            )
        except Exception as exc:
            existing_nodes = handle_graceful_error(
                exc, "Failed to fetch nodes for coverage seeds", default=[]
            )

        def register(node: TreeNode) -> None:
            coverage_map[node.id] = True
            nodes.setdefault(node.id, node)

        def coord_for_node(node: TreeNode) -> TreeCoordinate | None:
            raw_index = getattr(node, "level_index", None)
            if raw_index is None:
                return None
            return TreeCoordinate(
                document_id=getattr(node, "document_id", None),
                height=coerce_int(getattr(node, "height", 0)),
                level_index=coerce_int(raw_index),
            )

        for node in existing_nodes:
            register(node)

        max_height = self.store.nodes.max_height()
        coordinates: list[TreeCoordinate] = []

        def enqueue(coord: TreeCoordinate) -> None:
            coords = [coord]
            coords.extend(coord.ancestors(include_self=False, stop_height=max_height))
            for c in coords:
                coordinates.append(c)
                coordinates.append(c.sibling())

        # Enqueue real seeds
        for node in existing_nodes:
            node_coord = coord_for_node(node)
            if node_coord is not None:
                enqueue(node_coord)

        for coord in meta_seed_coords.values():
            enqueue(coord)

        # Add roots - document may be a forest, so we need all roots even if
        # some trees have no seeds. Roots are fetched (not computed) because
        # different trees in the forest may have different heights.
        try:
            root_nodes = self.store.nodes.get_root_nodes()
            for root in root_nodes:
                root_coord = TreeCoordinate(
                    document_id=getattr(root, "document_id", None),
                    height=getattr(root, "height", 0),
                    level_index=getattr(root, "level_index", 0),
                )
                coordinates.append(root_coord)
        except Exception as exc:
            handle_graceful_error(
                exc, "Failed to get root nodes for coverage", default=None
            )

        # Add edge-max as synthetic seeds. Edge-max nodes are enqueued so their
        # ancestors' siblings fill any gap between the window edge and actual seeds.
        if window_bounds is not None:
            enqueue(window_bounds.left_edge_max)
            if window_bounds.right_edge_max != window_bounds.left_edge_max:
                enqueue(window_bounds.right_edge_max)

        unique_coords = TreeCoordinate.unique(coordinates)

        # Filter out ancestors of pinned nodes - they become roots of their own trees
        if pinned_ids:
            pinned_coords: list[TreeCoordinate] = []
            for pinned_id in pinned_ids:
                if pinned_id in meta_seed_coords:
                    pinned_coords.append(meta_seed_coords[pinned_id])
            for node in existing_nodes:
                if node.id in pinned_ids:
                    node_coord = coord_for_node(node)
                    if node_coord is not None:
                        pinned_coords.append(node_coord)
            unique_coords = self._filter_pinned_ancestors(
                unique_coords, pinned_coords, max_height
            )

        # Filter coordinates outside window bounds
        if window_bounds is not None:
            unique_coords = [
                c
                for c in unique_coords
                if c.is_within_leaf_range(
                    window_bounds.left_leaf_index, window_bounds.right_leaf_index
                )
            ]

        # Fetch nodes for all coordinates
        coordinate_tuples = [coord.as_tuple() for coord in unique_coords]

        if coordinate_tuples:
            fetched = self.store.nodes.get_by_height_levels(coordinate_tuples)
            for node in fetched:
                register(node)

        # Fallback fetch for seeds that weren't found via coordinates
        missing_after_coord = [
            node_id for node_id in meta_seed_coords if node_id not in coverage_map
        ]
        if missing_after_coord:
            try:
                fallback_nodes = self.store.nodes.get_nodes(missing_after_coord)
            except Exception as exc:
                fallback_nodes = handle_graceful_error(
                    exc, "Failed to fetch fallback nodes for coverage", default=[]
                )
            for node in fallback_nodes:
                # For windowed queries, only include if within window
                if window_bounds is not None:
                    if (
                        node.span_start >= window_bounds.actual_start
                        and node.span_end <= window_bounds.actual_end
                    ):
                        register(node)
                else:
                    register(node)

        return coverage_map, nodes

    def compute_window_bounds(
        self,
        span_start: int,
        span_end: int,
        document_id: str,
    ) -> WindowBounds:
        """Compute the actual window bounds aligned to leaf node spans.

        Args:
            span_start: Requested start character position
            span_end: Requested end character position
            document_id: Document to query

        Returns:
            WindowBounds with actual_start/actual_end and edge-max coordinates.

        Raises:
            ValueError: If span bounds are invalid or no leaves found at boundaries.
        """
        nodes_wrapper = getattr(self.store, "nodes", None)
        repo = getattr(nodes_wrapper, "_repo", None) if nodes_wrapper else None

        if repo is None:
            raise ValueError("Node repository not available")

        # Find the leaf containing span_start
        left_leaf = repo.get_leaf_at_span_position(document_id, span_start)
        if left_leaf is None:
            raise ValueError(f"No leaf found containing position {span_start}")

        # For span_end, we want the leaf containing (span_end - 1) since span_end is exclusive
        # But if span_end is exactly at a boundary, we want the leaf ending there
        right_position = span_end - 1 if span_end > span_start else span_start
        right_leaf = repo.get_leaf_at_span_position(document_id, right_position)
        if right_leaf is None:
            # span_end might be exactly at document end - try the rightmost leaf
            doc_span_end = repo.get_document_span_end(document_id)
            if doc_span_end is not None and span_end >= doc_span_end:
                # Get the leaf ending at doc_span_end
                right_leaf = repo.get_leaf_at_span_position(
                    document_id, doc_span_end - 1
                )
        if right_leaf is None:
            raise ValueError(f"No leaf found containing position {right_position}")

        actual_start = left_leaf.span_start
        actual_end = right_leaf.span_end

        # Build coordinates for the boundary leaves
        left_coord = TreeCoordinate(
            document_id=document_id,
            height=0,
            level_index=left_leaf.level_index,
        )
        right_coord = TreeCoordinate(
            document_id=document_id,
            height=0,
            level_index=right_leaf.level_index,
        )

        # Find edge-max: highest ancestor of each boundary leaf that stays within
        # the window. For a full document window, this climbs all the way to the root.
        max_height = self.store.nodes.max_height()
        doc_span_end = repo.get_document_span_end(document_id)

        left_edge_max = left_coord.highest_ancestor_within_window(
            left_edge=True,
            left_leaf_idx=left_leaf.level_index,
            right_leaf_idx=right_leaf.level_index,
            max_height=max_height,
        )

        right_edge_max = right_coord.highest_ancestor_within_window(
            left_edge=False,
            left_leaf_idx=left_leaf.level_index,
            right_leaf_idx=right_leaf.level_index,
            max_height=max_height,
        )

        return WindowBounds(
            actual_start=actual_start,
            actual_end=actual_end,
            left_leaf_index=left_leaf.level_index,
            right_leaf_index=right_leaf.level_index,
            left_edge_max=left_edge_max,
            right_edge_max=right_edge_max,
        )

    def build_windowed_coverage(
        self,
        selected_ids: list[str],
        document_id: str,
        span_start: int | None = None,
        span_end: int | None = None,
        *,
        seed_metadata: Mapping[str, MetaDict] | None = None,
        pinned_ids: set[str] | None = None,
    ) -> CoverageResult:
        """Build coverage for a document window.

        This method:
        1. Resolves span defaults (0 for start, doc_span_end for end)
        2. Computes window bounds from span
        3. Adds edge-max nodes as synthetic seeds to ensure full window coverage
        4. Computes ancestor/sibling coordinates for all seeds
        5. Filters coordinates outside the window
        6. Fetches remaining nodes
        7. Adds pinned nodes from store (filtered by window bounds)

        Args:
            selected_ids: Node IDs from vector search (seeds within window)
            document_id: Document to build coverage for
            span_start: Start of window in character positions (default: 0)
            span_end: End of window in character positions (default: document end)
            seed_metadata: Optional metadata for seeds (for coordinate extraction)
            pinned_ids: Node IDs that are pinned (verbatim leaves)

        Returns:
            CoverageResult with coverage_map and nodes within the window.
        """
        # Resolve span defaults
        resolved_start = span_start if span_start is not None else 0

        if span_end is None:
            nodes_wrapper = getattr(self.store, "nodes", None)
            repo = getattr(nodes_wrapper, "_repo", None) if nodes_wrapper else None
            if repo is None:
                raise ValueError("Cannot determine document span end: no repository")
            doc_span_end = repo.get_document_span_end(document_id)
            if doc_span_end is None:
                raise ValueError(
                    f"Cannot determine document span end for {document_id}"
                )
            resolved_end = doc_span_end
        else:
            resolved_end = span_end

        # Compute window bounds
        window_bounds = self.compute_window_bounds(
            resolved_start, resolved_end, document_id
        )

        coverage_map, nodes = self._build_coordinate_closure(
            selected_ids,
            seed_metadata,
            pinned_ids=pinned_ids,
            window_bounds=window_bounds,
        )

        # Include pinned nodes scoped to this document, filtered by window bounds
        try:
            for node in self._get_document_pinned_nodes():
                # Filter by window bounds
                if (
                    node.span_start >= window_bounds.actual_start
                    and node.span_end <= window_bounds.actual_end
                ):
                    coverage_map[node.id] = True
                    nodes.setdefault(node.id, node)
        except Exception as exc:  # pragma: no cover - defensive logging
            handle_graceful_error(
                exc, "Failed to include pinned nodes in coverage map", default=None
            )

        return CoverageResult(coverage_map=coverage_map, nodes=nodes)
