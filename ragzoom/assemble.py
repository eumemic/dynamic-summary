"""Assembly logic for creating coherent summaries from tiling nodes."""

import logging

from ragzoom.retrieve import RetrievalResult
from ragzoom.store import TreeNode
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


class Assembler:
    """Assembles tiling nodes into coherent summary with optional smoothing."""

    def __init__(self, store):
        """Initialize assembler.

        Args:
            store: DocumentStore instance for node operations
        """
        self.store = store
        self.tokenizer = tokenizer

    def assemble(self, retrieval_result: RetrievalResult) -> str:
        """
        Assemble tiling nodes into final summary using the DP-based assembly path only.
        """
        if retrieval_result.tiling is None:
            raise ValueError("Assembly requires tiling from the retrieval result.")
        return self.assemble_dp(retrieval_result.tiling, retrieval_result.nodes)

    def assemble_dp(
        self,
        tiling: list[str],  # List of node IDs
        nodes: dict[str, "TreeNode"] | None = None,
    ) -> str:
        """Assemble a tiling from a list of node IDs."""
        if not tiling:
            return ""

        texts = [self._get_text_for_node(node_id, nodes) for node_id in tiling]
        # Filter out empty texts to avoid extra newlines
        texts = [t for t in texts if t]
        return "\n\n".join(texts)

    def _get_text_for_node(
        self, node_id: str, nodes: dict[str, "TreeNode"] | None = None
    ) -> str:
        """Extract the text for a single node."""
        # Use pre-loaded nodes if available, otherwise fall back to store
        node: TreeNode | None
        if nodes and node_id in nodes:
            node = nodes[node_id]
        else:
            node = self.store.nodes.get(node_id)

        if not node or not node.text:
            return ""

        # Return the full text of the node
        return node.text

    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        return self.tokenizer.count_tokens(text)
