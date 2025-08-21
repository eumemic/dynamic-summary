"""Service for building coverage maps during retrieval."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragzoom.store import StoreManager

logger = logging.getLogger(__name__)


class CoverageBuilder:
    """Builds coverage maps including selected nodes, ancestors, and siblings."""

    def __init__(self, store: "StoreManager"):
        """Initialize coverage builder.

        Args:
            store: StoreManager instance for node operations
        """
        self.store = store

    def build_complete_coverage_map(self, selected_ids: list[str]) -> dict[str, bool]:
        """Build complete coverage map including selected nodes, ancestors, and pinned nodes.

        Args:
            selected_ids: List of selected node IDs

        Returns:
            Coverage map with node IDs as keys
        """
        coverage_map = self.build_coverage_map(selected_ids)

        pinned_nodes = self.store.get_pinned_nodes(self.store.PIN_DEPTH_MAX)
        for node in pinned_nodes:
            coverage_map[node.id] = True

        return coverage_map

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

        coverage_map = {node_id: True for node_id in selected_ids}
        for node_id in selected_ids:
            self.store.nodes.update_node_access(node_id)

        ancestors = self.store.tree.get_ancestors(selected_ids)
        for ancestor in ancestors:
            coverage_map[ancestor.id] = True

        self._ensure_sibling_coverage(coverage_map)

        return coverage_map

    def _ensure_sibling_coverage(self, coverage_map: dict[str, bool]) -> None:
        """Iteratively ensure coverage property by including siblings.

        If a node is in coverage and is an internal node in the main tree,
        ensure both children are present to maintain the coverage property.

        Args:
            coverage_map: Coverage map to update in place
        """
        while True:
            nodes_in_coverage = self.store.nodes.get_nodes(list(coverage_map.keys()))
            new_nodes_added = False

            for node in nodes_in_coverage:
                left = node.left_child_id
                right = node.right_child_id

                if left or right:
                    if left and left in coverage_map:
                        if right and right not in coverage_map:
                            coverage_map[right] = True
                            new_nodes_added = True
                    elif right and right in coverage_map:
                        if left and left not in coverage_map:
                            coverage_map[left] = True
                            new_nodes_added = True

            if not new_nodes_added:
                break
