"""Service for building coverage maps during retrieval."""

import logging
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ragzoom.document_store import DocumentStore
    from ragzoom.store import StoreManager

logger = logging.getLogger(__name__)


class CoverageBuilder:
    """Builds coverage maps including selected nodes, ancestors, and siblings."""

    def __init__(self, store: Union["StoreManager", "DocumentStore"]):
        """Initialize coverage builder.

        Args:
            store: Store (system-wide) or DocumentStore (scoped) for node operations
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

        # Include pinned nodes, scoped appropriately if a DocumentStore is provided.
        try:
            depth_max = getattr(self.store, "PIN_DEPTH_MAX", 2)

            # Preferred: Store exposes get_pinned_nodes (system-wide)
            if hasattr(self.store, "get_pinned_nodes"):
                pinned_nodes = self.store.get_pinned_nodes(depth_max)
            else:
                # DocumentStore path: pull from underlying repo and filter by document
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
        except Exception as e:
            logger.warning(f"Failed to include pinned nodes in coverage map: {e}")

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
            # Support both StoreManager and DocumentStore
            if hasattr(self.store.nodes, "update_node_access"):
                self.store.nodes.update_node_access(node_id)
            elif hasattr(self.store.nodes, "update_node_access_time"):
                self.store.nodes.update_node_access_time(node_id)

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
        # Get all nodes currently in coverage (robust against mocks)
        node_ids = list(coverage_map.keys())
        nodes_in_coverage = []
        try:
            get_nodes_fn = getattr(self.store.nodes, "get_nodes", None)
            if callable(get_nodes_fn):
                result = get_nodes_fn(node_ids)
                if isinstance(result, list):
                    nodes_in_coverage = result
        except Exception:
            nodes_in_coverage = []
        if not nodes_in_coverage:
            try:
                get_many_fn = getattr(self.store.nodes, "get_many", None)
                if callable(get_many_fn):
                    result = get_many_fn(node_ids)
                    if isinstance(result, list):
                        nodes_in_coverage = result
            except Exception:
                nodes_in_coverage = []
        if not nodes_in_coverage:
            getter = getattr(self.store.nodes, "get", None)
            if callable(getter):
                for node_id in node_ids:
                    try:
                        node = getter(node_id)
                        if node is not None:
                            nodes_in_coverage.append(node)
                    except Exception:
                        continue

        # Use parent-based sibling inclusion first (works without path fields)
        for node in nodes_in_coverage:
            parent_id = getattr(node, "parent_id", None)
            if not parent_id:
                continue
            try:
                left, right = self.store.tree.get_children(parent_id)
            except Exception:
                left, right = None, None
            if left and getattr(left, "id", None) != node.id:
                coverage_map[left.id] = True
            if right and getattr(right, "id", None) != node.id:
                coverage_map[right.id] = True

        # Additionally, if path-based retrieval is available, include any missing siblings by path
        try:
            get_by_paths_fn = getattr(self.store.nodes, "get_nodes_by_paths", None)
            if callable(get_by_paths_fn):
                from ragzoom.utils.path_utils import get_sibling_path

                sibling_paths = set()
                for node in nodes_in_coverage:
                    path = getattr(node, "path", None)
                    if path is None:
                        continue
                    sibling_path = get_sibling_path(path)
                    if sibling_path is not None:
                        sibling_paths.add(sibling_path)

                if sibling_paths:
                    siblings = get_by_paths_fn(list(sibling_paths))
                    for sibling in siblings:
                        coverage_map[sibling.id] = True
        except Exception:
            # If anything goes wrong with path-based method, we already did parent-based inclusion
            pass
