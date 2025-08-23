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
        """Ensure coverage property by including siblings using path-based logic.

        If a node is in coverage, its sibling must also be included (if it exists)
        to maintain the coverage property that parent span equals union of children spans.

        Args:
            coverage_map: Coverage map to update in place
        """
        # Get all nodes currently in coverage to access their paths
        nodes_in_coverage = self.store.nodes.get_nodes(list(coverage_map.keys()))

        # All nodes should have valid paths - use optimized path-based logic
        # Use fast path-based sibling detection
        from ragzoom.utils.path_utils import get_sibling_path

        # Compute sibling paths using direct string manipulation
        sibling_paths = set()
        for node in nodes_in_coverage:
            sibling_path = get_sibling_path(node.path)
            if sibling_path is not None:  # Root has no sibling
                sibling_paths.add(sibling_path)

        # Single batch fetch of all potential siblings
        if sibling_paths:
            siblings = self.store.nodes.get_nodes_by_paths(list(sibling_paths))
            # Add existing siblings to coverage map
            for sibling in siblings:
                coverage_map[sibling.id] = True
