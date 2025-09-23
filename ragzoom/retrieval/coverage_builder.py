"""Service for building coverage maps during retrieval."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragzoom.document_store import DocumentStore

logger = logging.getLogger(__name__)


class CoverageBuilder:
    """Builds coverage maps including selected nodes, ancestors, and siblings."""

    def __init__(self, store: "DocumentStore"):
        """Initialize coverage builder.

        Args:
            store: DocumentStore instance for node operations
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

        # Ensure every root node participates in coverage so forests remain contiguous.
        try:
            root_nodes = self.store.nodes.get_root_nodes()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to load root nodes for coverage map: %s", exc)
        else:
            for root in root_nodes:
                coverage_map[root.id] = True

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

        # Start with only node IDs that actually exist in the store
        existing_nodes = []
        try:
            existing_nodes = self.store.nodes.get_nodes(selected_ids)
        except Exception:
            existing_nodes = []
        coverage_map = {node.id: True for node in existing_nodes}
        for node in existing_nodes:
            self.store.nodes.update_access(node.id)

        # Only fetch ancestors for existing nodes to avoid phantom IDs
        ancestors = self.store.tree.get_ancestors(list(coverage_map.keys()))
        for ancestor in ancestors:
            coverage_map[ancestor.id] = True

        self._ensure_sibling_coverage(coverage_map)

        return coverage_map

    def _ensure_sibling_coverage(self, coverage_map: dict[str, bool]) -> None:
        """Ensure coverage property by including siblings via structural traversal."""
        while True:
            nodes_in_coverage = self.store.nodes.get_nodes(list(coverage_map.keys()))
            if not nodes_in_coverage:
                return

            missing_siblings: set[str] = set()

            for node in nodes_in_coverage:
                left_id = getattr(node, "left_child_id", None)
                right_id = getattr(node, "right_child_id", None)

                if not left_id or not right_id:
                    continue

                left_present = left_id in coverage_map
                right_present = right_id in coverage_map

                if left_present and not right_present:
                    missing_siblings.add(right_id)
                elif right_present and not left_present:
                    missing_siblings.add(left_id)

            if not missing_siblings:
                return

            fetched = self.store.nodes.get_nodes(list(missing_siblings))
            if not fetched:
                return

            new_nodes_added = False
            for sibling in fetched:
                if sibling.id not in coverage_map:
                    coverage_map[sibling.id] = True
                    new_nodes_added = True

            if not new_nodes_added:
                return
