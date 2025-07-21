"""Assembly logic for creating coherent summaries from tiling nodes."""

import logging
from typing import Optional

import tiktoken
from openai import OpenAI

from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import RetrievalResult, Segment
from ragzoom.store import Store, TreeNode

logger = logging.getLogger(__name__)


class Assembler:
    """Assembles tiling nodes into coherent summary with optional smoothing."""

    def __init__(self, config: RagZoomConfig, store: Store):
        """Initialize assembler."""
        self.config = config
        self.store = store
        self.client = OpenAI(api_key=config.openai_api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def assemble(self, retrieval_result: RetrievalResult) -> str:
        """
        Assemble tiling nodes into final summary using the DP-based assembly path only.
        """
        if retrieval_result.tiling is None:
            raise ValueError("Assembly requires tiling from the retrieval result.")
        return self.assemble_dp(retrieval_result.tiling, retrieval_result.nodes)

    def assemble_dp(
        self,
        tiling: list["Segment"],
        nodes: Optional[dict[str, "TreeNode"]] = None,
    ) -> str:
        """Assemble a tiling from a list of Segments."""
        if not tiling:
            return ""

        texts = [self._get_text_for_segment(seg, nodes) for seg in tiling]
        # Filter out empty texts to avoid extra newlines
        texts = [t for t in texts if t]
        return "\n\n".join(texts)

    def _get_text_for_segment(
        self, segment: "Segment", nodes: Optional[dict[str, "TreeNode"]] = None
    ) -> str:
        """Extract the text for a single Segment."""
        # Use pre-loaded nodes if available, otherwise fall back to store
        node: Optional[TreeNode]
        if nodes and segment.node_id in nodes:
            node = nodes[segment.node_id]
        else:
            node = self.store.get_node(segment.node_id)

        if not node or not node.text:
            return ""

        # If it's a leaf or has no mid_offset, return full text.
        # These nodes should have side=None according to our invariant.
        if self.store.is_leaf_node(node.id) or node.mid_offset is None:
            return node.text

        if segment.side == "LEFT":
            return node.text[: node.mid_offset].strip()
        else:  # RIGHT
            return node.text[node.mid_offset :].strip()

    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        return len(self.tokenizer.encode(text))
