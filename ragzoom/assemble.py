"""Assembly logic for creating coherent summaries from frontier nodes."""

import logging
from typing import Optional

import tiktoken
from openai import OpenAI

from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import RetrievalResult, Segment
from ragzoom.store import Store, TreeNode

logger = logging.getLogger(__name__)


class Assembler:
    """Assembles frontier nodes into coherent summary with optional smoothing."""

    def __init__(self, config: RagZoomConfig, store: Store):
        """Initialize assembler."""
        self.config = config
        self.store = store
        self.client = OpenAI(api_key=config.openai_api_key)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def assemble(self, retrieval_result: RetrievalResult) -> str:
        """
        Assemble frontier nodes into final summary using the DP-based assembly path only.
        """
        if retrieval_result.frontier_segments is None:
            raise ValueError(
                "DP assembly requires frontier_segments. Legacy assembly is no longer supported."
            )
        return self.assemble_dp(
            retrieval_result.frontier_segments, retrieval_result.nodes
        )

    def assemble_dp(
        self,
        frontier_segments: list["Segment"],
        nodes: Optional[dict[str, "TreeNode"]] = None,
    ) -> str:
        """Assemble a frontier from a list of Segments."""
        if not frontier_segments:
            return ""

        texts = [self._get_text_for_segment(seg, nodes) for seg in frontier_segments]
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
            right_text = node.text[node.mid_offset :].strip()
            # Clean the MID delimiter from RIGHT side
            return self._clean_mid_delimiter(right_text)

    def _clean_mid_delimiter(self, text: str) -> str:
        """Remove <<<MID>>> delimiter from text."""
        return text.replace("<<<MID>>>", "").strip()

    def _find_intermediate_path(
        self, start_id: str, end_id: str
    ) -> list[tuple[str, int]]:
        """Find intermediate nodes to satisfy slope cap between two nodes."""
        start_node = self.store.get_node(start_id)
        end_node = self.store.get_node(end_id)

        if not start_node or not end_node:
            return []

        path = []

        # Get heights for comparison
        start_height = self.store.get_node_height(start_id)
        end_height = self.store.get_node_height(end_id)

        # Handle both upward and downward transitions
        if start_height > end_height:
            # Going up (toward root)
            current_id = start_id
            current_height = start_height

            while current_height > end_height + 1:
                node = self.store.get_node(current_id)
                if not node or not node.parent_id:
                    break

                parent = self.store.get_node(node.parent_id)
                if parent:
                    parent_height = self.store.get_node_height(parent.id)
                    path.append((parent.id, parent_height))
                    current_id = parent.id
                    current_height = parent_height
                else:
                    break

        elif start_height < end_height:
            # Going down (toward leaves) - need to find a path
            # Try to find nodes at intermediate depths in the same span range
            target_span_start = end_node.span_start
            target_span_end = end_node.span_end

            # Search for nodes at intermediate heights that cover the target span
            for target_height in range(start_height + 1, end_height):
                # Find a node at this height that overlaps with target span
                intermediate = self._find_node_at_height_in_span(
                    target_height, target_span_start, target_span_end
                )
                if intermediate:
                    intermediate_height = self.store.get_node_height(intermediate.id)
                    path.append((intermediate.id, intermediate_height))

        return path

    def _find_node_at_height_in_span(
        self, target_height: int, span_start: int, span_end: int
    ):
        """Find a node at given height that overlaps with the span."""
        with self.store.SessionLocal() as session:
            # Query for nodes that overlap the span
            from ragzoom.store import TreeNode

            # Get all nodes that overlap the span
            candidates = (
                session.query(TreeNode)
                .filter(
                    TreeNode.span_start < span_end,
                    TreeNode.span_end > span_start,
                )
                .all()
            )

            # Find the first node with the target height
            for node in candidates:
                if self.store.get_node_height(node.id) == target_height:
                    return node

            return None

    def _apply_smoothing_pass(self, frontier_nodes: list[str], texts: list[str]) -> str:
        """Apply smoothing pass to improve coherence at boundaries."""
        if len(texts) <= 1:
            return "\n\n".join(texts)

        smoothed_parts = []

        for i in range(len(texts)):
            if i == 0:
                # First chunk
                smoothed = self._smooth_boundary(
                    None,
                    texts[i],
                    texts[i + 1] if i + 1 < len(texts) else None,
                    "start",
                )
                smoothed_parts.append(smoothed)
            elif i == len(texts) - 1:
                # Last chunk
                smoothed = self._smooth_boundary(texts[i - 1], texts[i], None, "end")
                smoothed_parts.append(smoothed)
            else:
                # Middle chunk
                smoothed = self._smooth_boundary(
                    texts[i - 1], texts[i], texts[i + 1], "middle"
                )
                smoothed_parts.append(smoothed)

        return "\n\n".join(smoothed_parts)

    def _smooth_boundary(
        self,
        prev_text: Optional[str],
        current_text: str,
        next_text: Optional[str],
        position: str,
    ) -> str:
        """Smooth a single boundary using LLM."""
        # Build prompt
        prompt_parts = []

        if prev_text:
            # Take last 50 tokens of previous
            prev_tokens = self.tokenizer.encode(prev_text)
            if len(prev_tokens) > 50:
                prev_context = self.tokenizer.decode(prev_tokens[-50:])
            else:
                prev_context = prev_text
            prompt_parts.append(f"<<PREVIOUS>>\n...{prev_context}")

        prompt_parts.append(f"<<CURRENT>>\n{current_text}")

        if next_text:
            # Take first 50 tokens of next
            next_tokens = self.tokenizer.encode(next_text)
            if len(next_tokens) > 50:
                next_context = self.tokenizer.decode(next_tokens[:50])
            else:
                next_context = next_text
            prompt_parts.append(f"<<NEXT>>\n{next_context}...")

        prompt_parts.append(
            "\nLightly edit the CURRENT section to flow smoothly with the context. "
            "Preserve all facts and key information. "
            "Add minimal transition phrases only where needed. "
            "Return only the edited CURRENT section."
        )

        full_prompt = "\n\n".join(prompt_parts)

        try:
            response = self.client.chat.completions.create(
                model=self.config.smoothing_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a text editor focusing on smooth transitions.",
                    },
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.3,
                max_tokens=self.config.smoothing_max_tokens,
            )
            content = response.choices[0].message.content
            return content.strip() if content else current_text
        except Exception as e:
            logger.error(f"Error in smoothing pass: {e}")
            # Fall back to original text
            return current_text

    def get_token_count(self, text: str) -> int:
        """Get token count for text."""
        return len(self.tokenizer.encode(text))
