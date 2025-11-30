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
    ) -> CoverageResult:
        """Build coverage along with the materialised nodes needed downstream."""

        coverage_map, nodes = self._build_coordinate_closure(
            selected_ids, seed_metadata
        )

        # Ensure every root participates (forest support)
        try:
            root_nodes = self.store.nodes.get_root_nodes()
        except Exception as exc:  # pragma: no cover - defensive logging
            handle_graceful_error(
                exc, "Failed to load root nodes for coverage map", default=None
            )
        else:
            for root in root_nodes:
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
        coverage_map, _ = self._build_coordinate_closure(selected_ids, seed_metadata)
        return coverage_map

    def _build_coordinate_closure(
        self,
        selected_ids: list[str],
        seed_metadata: Mapping[str, MetaDict] | None = None,
    ) -> tuple[dict[str, bool], dict[str, TreeNode]]:
        coverage_map: dict[str, bool] = {}
        nodes: dict[str, TreeNode] = {}
        if not selected_ids:
            return coverage_map, nodes

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

        return coverage_map, nodes
