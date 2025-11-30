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


class CoverageBuilder:
    """Builds coverage maps including selected nodes, ancestors, and siblings."""

    def __init__(self, store: DocumentStore):
        self.store = store

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
        """Build coverage along with the materialised nodes needed downstream.

        Args:
            selected_ids: Node IDs to include in coverage (seeds)
            seed_metadata: Optional metadata for seeds (for coordinate extraction)
            pinned_ids: Node IDs that are pinned and should not have ancestors
                in coverage. This makes each pinned node a root of its own tree,
                avoiding unnecessary ancestor fetches and scoring.
        """

        coverage_map, nodes, excluded_root_coords = self._build_coordinate_closure(
            selected_ids, seed_metadata, pinned_ids=pinned_ids
        )

        # Ensure every root participates (forest support)
        # But exclude roots that are ancestors of pinned nodes
        try:
            root_nodes = self.store.nodes.get_root_nodes()
        except Exception as exc:  # pragma: no cover - defensive logging
            handle_graceful_error(
                exc, "Failed to load root nodes for coverage map", default=None
            )
        else:
            for root in root_nodes:
                # Skip roots that are ancestors of pinned nodes
                root_coord = (
                    getattr(root, "height", 0),
                    getattr(root, "level_index", 0),
                )
                if root_coord in excluded_root_coords:
                    continue
                coverage_map[root.id] = True
                nodes.setdefault(root.id, root)

        # Include pinned nodes scoped to this document
        try:
            depth_max = getattr(self.store, "PIN_DEPTH_MAX", 2)
            if hasattr(self.store, "get_pinned_nodes"):
                pinned_nodes = self.store.get_pinned_nodes(depth_max)
            else:
                pinned_nodes = []
                nodes_wrapper = getattr(self.store, "nodes", None)
                repo = getattr(nodes_wrapper, "_repo", None)
                document_id = getattr(self.store, "document_id", None)
                if repo is not None and document_id is not None:
                    all_pinned = repo.get_pinned_nodes(depth_max)
                    pinned_nodes = [
                        n for n in all_pinned if n.document_id == document_id
                    ]
            for node in pinned_nodes:
                coverage_map[node.id] = True
                nodes.setdefault(node.id, node)
        except Exception as exc:  # pragma: no cover - defensive logging
            handle_graceful_error(
                exc, "Failed to include pinned nodes in coverage map", default=None
            )

        return CoverageResult(coverage_map=coverage_map, nodes=nodes)

    def build_coverage_map(
        self,
        selected_ids: list[str],
        *,
        seed_metadata: Mapping[str, MetaDict] | None = None,
    ) -> dict[str, bool]:
        """Build coverage map including selected nodes and structural relatives."""

        if not selected_ids:
            return {}
        coverage_map, _, _ = self._build_coordinate_closure(selected_ids, seed_metadata)
        return coverage_map

    def _filter_pinned_ancestors(
        self,
        coords: list[TreeCoordinate],
        pinned_coords: list[TreeCoordinate],
        max_height: int,
    ) -> tuple[list[TreeCoordinate], set[tuple[int, int]]]:
        """Remove ancestors of pinned nodes from coordinate list.

        Pinned nodes should be roots of their own trees - they never roll up.
        By removing their ancestors from coverage, we:
        1. Skip unnecessary fetches for nodes that won't participate in tiling
        2. Eliminate scoring overhead for those nodes
        3. Make pinned nodes natural roots (parent not in coverage)

        Returns:
            Tuple of (filtered coordinates, root coordinates that were removed).
            The root coordinates are those at max_height that are ancestors of
            pinned nodes - these should be excluded from forest support.
        """
        excluded_roots: set[tuple[int, int]] = set()
        if not pinned_coords:
            return coords, excluded_roots

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
                # Track root-level coordinates for forest support exclusion
                if current.height == max_height:
                    excluded_roots.add(key)
                current = current.parent()

        # Filter out ancestors (siblings are preserved)
        filtered = [c for c in coords if c.as_tuple() not in to_remove]
        return filtered, excluded_roots

    def _build_coordinate_closure(
        self,
        selected_ids: list[str],
        seed_metadata: Mapping[str, MetaDict] | None = None,
        *,
        pinned_ids: set[str] | None = None,
    ) -> tuple[dict[str, bool], dict[str, TreeNode], set[tuple[int, int]]]:
        coverage_map: dict[str, bool] = {}
        nodes: dict[str, TreeNode] = {}
        excluded_root_coords: set[tuple[int, int]] = set()
        if not selected_ids:
            return coverage_map, nodes, excluded_root_coords

        document_default = getattr(self.store, "document_id", None)
        meta_map = seed_metadata or {}

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
            coord_doc = str(meta.get("document_id", "")) or document_default
            if coord_doc is None:
                missing_seed_ids.append(node_id)
                continue
            coord = TreeCoordinate(
                document_id=coord_doc,
                height=coerce_int(meta.get("height", 0)),
                level_index=coerce_int(meta.get("level_index", 0)),
            )
            meta_seed_coords[node_id] = coord

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

        for node in existing_nodes:
            node_coord = coord_for_node(node)
            if node_coord is not None:
                enqueue(node_coord)

        for coord in meta_seed_coords.values():
            enqueue(coord)

        unique_coords = TreeCoordinate.unique(coordinates)

        # Filter out ancestors of pinned nodes - they become roots of their own trees
        if pinned_ids:
            pinned_coords: list[TreeCoordinate] = []
            # Collect pinned node coordinates from metadata
            for pinned_id in pinned_ids:
                if pinned_id in meta_seed_coords:
                    pinned_coords.append(meta_seed_coords[pinned_id])
            # Collect pinned node coordinates from fetched nodes
            for node in existing_nodes:
                if node.id in pinned_ids:
                    node_coord = coord_for_node(node)
                    if node_coord is not None:
                        pinned_coords.append(node_coord)
            unique_coords, excluded_root_coords = self._filter_pinned_ancestors(
                unique_coords, pinned_coords, max_height
            )

        coordinate_tuples = [coord.as_tuple() for coord in unique_coords]

        if coordinate_tuples:
            fetched = self.store.nodes.get_by_height_levels(coordinate_tuples)
            for node in fetched:
                register(node)

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
                register(node)

        return coverage_map, nodes, excluded_root_coords
