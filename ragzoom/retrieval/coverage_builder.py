"""Service for building coverage maps during retrieval."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ragzoom.tree_coordinate import TreeCoordinate

if TYPE_CHECKING:
    from ragzoom.contracts.tree_node import TreeNode
    from ragzoom.document_store import DocumentStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CoverageResult:
    """Container for coverage map construction output."""

    coverage_map: dict[str, bool]
    nodes: dict[str, "TreeNode"]


class CoverageBuilder:
    """Builds coverage maps including selected nodes, ancestors, and siblings."""

    def __init__(self, store: "DocumentStore"):
        """Initialize coverage builder.

        Args:
            store: DocumentStore instance for node operations
        """
        self.store = store

    def build_complete_coverage_map(self, selected_ids: list[str]) -> dict[str, bool]:
        """Backward-compatible wrapper returning only the coverage map."""

        return self.build_complete_coverage(selected_ids).coverage_map

    def build_complete_coverage(self, selected_ids: list[str]) -> CoverageResult:
        """Build coverage along with the materialised nodes needed downstream."""

        coverage_map, nodes = self._build_coordinate_closure(selected_ids)

        # Ensure every root node participates in coverage so forests remain contiguous.
        try:
            root_nodes = self.store.nodes.get_root_nodes()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load root nodes for coverage map: %s", exc)
        else:
            for root in root_nodes:
                coverage_map[root.id] = True
                nodes.setdefault(root.id, root)

        # Include pinned nodes, scoped appropriately if a DocumentStore is provided.
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
        except Exception as e:  # pragma: no cover - defensive logging
            logger.warning(f"Failed to include pinned nodes in coverage map: {e}")

        return CoverageResult(coverage_map=coverage_map, nodes=nodes)

    # jscpd:ignore-start - Similar docstring structure for related methods (false positive)
    def build_coverage_map(self, selected_ids: list[str]) -> dict[str, bool]:
        """Build a coverage map including selected nodes, their ancestors, and all required siblings.

        This maintains the coverage property: if a child is in the coverage set,
        its sibling must also be included (if it exists) so the parent span equals
        the union of children spans.

        Args:
            selected_ids: List of selected node IDs

        Returns:
            Coverage map with node IDs as keys
        """
        # jscpd:ignore-end
        if not selected_ids:
            return {}

        coverage_map, _ = self._build_coordinate_closure(selected_ids)
        return coverage_map

    def _build_coordinate_closure(
        self, selected_ids: list[str]
    ) -> tuple[dict[str, bool], dict[str, "TreeNode"]]:
        """Compute coverage and load nodes using coordinate batching."""

        coverage_map: dict[str, bool] = {}
        nodes: dict[str, TreeNode] = {}
        if not selected_ids:
            return coverage_map, nodes

        try:
            existing_nodes = self.store.nodes.get_nodes(selected_ids)
        except Exception:
            existing_nodes = []

        for node in existing_nodes:
            coverage_map[node.id] = True
            nodes[node.id] = node

        touched_ids: set[str] = {node.id for node in existing_nodes}

        if existing_nodes:
            document_id = getattr(self.store, "document_id", None)
            max_height = self.store.nodes.max_height()
            coordinates: list[TreeCoordinate] = []

            for node in existing_nodes:
                raw_index = getattr(node, "level_index", None)
                if raw_index is None:
                    continue
                coord = TreeCoordinate(
                    document_id=document_id,
                    height=int(getattr(node, "height", 0)),
                    level_index=int(raw_index),
                )
                coords_for_node = [coord]
                coords_for_node.extend(
                    coord.ancestors(include_self=False, stop_height=max_height)
                )

                for c in coords_for_node:
                    coordinates.append(c)
                    coordinates.append(c.sibling())

            unique_coords = TreeCoordinate.unique(coordinates)
            coordinate_tuples = [coord.as_tuple() for coord in unique_coords]

            fetched = self.store.nodes.get_by_height_levels(coordinate_tuples)
            for node in fetched:
                coverage_map[node.id] = True
                nodes.setdefault(node.id, node)
                touched_ids.add(node.id)

        for node_id in touched_ids:
            self.store.nodes.update_access(node_id)

        return coverage_map, nodes
