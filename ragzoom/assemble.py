"""Assembly logic for creating coherent summaries from frontier nodes."""

import logging
from typing import Optional

import tiktoken
from openai import OpenAI

from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import RetrievalResult, SummarySegment
from ragzoom.store import Store

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
        return self.assemble_dp(retrieval_result.frontier_segments)

    def assemble_dp(self, frontier_segments: list["SummarySegment"]) -> str:
        """Assemble a frontier from a list of SummarySegments."""
        if not frontier_segments:
            return ""

        texts = [self._get_text_for_segment(seg) for seg in frontier_segments]
        # Filter out empty texts to avoid extra newlines
        texts = [t for t in texts if t]
        return "\n\n".join(texts)

    def _get_text_for_segment(self, segment: "SummarySegment") -> str:
        """Extract the text for a single SummarySegment."""
        node = self.store.get_node(segment.node_id)
        if not node or not node.text:
            return ""

        # If it's a leaf or has no mid_offset, we can't split it.
        # This shouldn't happen with the DP model, but as a fallback, return full text.
        if node.depth == 0 or node.mid_offset is None:
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

    def _apply_slope_cap(self, frontier_nodes: list[str]) -> list[str]:
        """Apply slope cap constraint (max ±1 depth change between adjacent nodes)."""
        if len(frontier_nodes) <= 1:
            return frontier_nodes

        # Get depths
        node_depths = []
        for node_id in frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                node_depths.append((node_id, node.depth))

        if not node_depths:
            return []

        # Apply slope cap with replacement (not insertion)
        capped_nodes = [node_depths[0]]
        seen = {node_depths[0][0]}  # Track seen node IDs

        for i in range(1, len(node_depths)):
            current_id, current_depth = node_depths[i]
            prev_id, prev_depth = capped_nodes[-1]

            # Check depth difference
            depth_diff = abs(current_depth - prev_depth)

            if depth_diff <= 1:
                # Within slope cap
                if current_id not in seen:
                    capped_nodes.append((current_id, current_depth))
                    seen.add(current_id)
            else:
                # Need to replace deep node with ancestor to satisfy slope cap
                logger.info(f"Slope cap violation: {prev_depth} -> {current_depth}")

                # Find appropriate ancestor depth (max 1 level change from previous)
                target_depth = (
                    prev_depth + 1 if current_depth > prev_depth else prev_depth - 1
                )

                # Find ancestor at target depth
                replacement_node = self._find_ancestor_at_depth(
                    current_id, target_depth
                )
                if replacement_node and replacement_node.id not in seen:
                    capped_nodes.append((replacement_node.id, replacement_node.depth))
                    seen.add(replacement_node.id)
                elif current_id not in seen:
                    # Fallback to original node if no suitable ancestor
                    capped_nodes.append((current_id, current_depth))
                    seen.add(current_id)

        # Don't re-sort - maintain the slope cap order
        # The caller is responsible for any final sorting if needed
        return [node_id for node_id, _ in capped_nodes]

    def _find_ancestor_at_depth(self, node_id: str, target_depth: int):
        """Find ancestor of node at target depth."""
        current_node = self.store.get_node(node_id)

        while current_node and current_node.depth > target_depth:
            if current_node.parent_id:
                current_node = self.store.get_node(current_node.parent_id)
            else:
                break

        return (
            current_node
            if current_node and current_node.depth == target_depth
            else None
        )

    def _find_intermediate_path(
        self, start_id: str, end_id: str
    ) -> list[tuple[str, int]]:
        """Find intermediate nodes to satisfy slope cap between two nodes."""
        start_node = self.store.get_node(start_id)
        end_node = self.store.get_node(end_id)

        if not start_node or not end_node:
            return []

        path = []

        # Handle both upward and downward transitions
        if start_node.depth > end_node.depth:
            # Going up (toward root)
            current_id = start_id
            current_depth = start_node.depth

            while current_depth > end_node.depth + 1:
                node = self.store.get_node(current_id)
                if not node or not node.parent_id:
                    break

                parent = self.store.get_node(node.parent_id)
                if parent:
                    path.append((parent.id, parent.depth))
                    current_id = parent.id
                    current_depth = parent.depth
                else:
                    break

        elif start_node.depth < end_node.depth:
            # Going down (toward leaves) - need to find a path
            # Try to find nodes at intermediate depths in the same span range
            target_span_start = end_node.span_start
            target_span_end = end_node.span_end
            current_depth = start_node.depth

            # Search for nodes at intermediate depths that cover the target span
            for depth in range(current_depth + 1, end_node.depth):
                # Find a node at this depth that overlaps with target span
                intermediate = self._find_node_at_depth_in_span(
                    depth, target_span_start, target_span_end
                )
                if intermediate:
                    path.append((intermediate.id, intermediate.depth))

        return path

    def _find_node_at_depth_in_span(self, depth: int, span_start: int, span_end: int):
        """Find a node at given depth that overlaps with the span."""
        with self.store.SessionLocal() as session:
            # Query for nodes at target depth that overlap the span
            from ragzoom.store import TreeNode

            node = (
                session.query(TreeNode)
                .filter(
                    TreeNode.depth == depth,
                    TreeNode.span_start < span_end,
                    TreeNode.span_end > span_start,
                )
                .first()
            )
            return node

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

    def _count_frontier_tokens(self, frontier_nodes: list[str]) -> int:
        """Count actual tokens that would be extracted from frontier nodes."""
        total_tokens = 0
        for node_id in frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                # Get actual text that would be extracted for this node
                actual_text = self._extract_node_text(node, {}, set(frontier_nodes))
                total_tokens += len(self.tokenizer.encode(actual_text))
        return total_tokens

    def trim_frontier_to_budget(
        self, frontier_nodes: list[str], budget_tokens: int, scores: dict[str, float]
    ) -> list[str]:
        """Trim frontier by dropping lowest-utility nodes until under budget."""
        if not frontier_nodes:
            return frontier_nodes

        # Calculate exact token cost per node
        node_costs = []
        for node_id in frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                # Get actual text that would be extracted for this node
                actual_text = self._extract_node_text(node, {}, set(frontier_nodes))
                token_cost = len(self.tokenizer.encode(actual_text))

                # Calculate utility ratio (score per token)
                utility_ratio = scores.get(node_id, 0.0) / max(token_cost, 1)
                node_costs.append((node_id, token_cost, utility_ratio))

        # Sort by utility ratio (highest first)
        node_costs.sort(key=lambda x: x[2], reverse=True)

        # Select nodes that fit within budget
        selected = []
        total_tokens = 0

        for node_id, cost, utility_ratio in node_costs:
            if total_tokens + cost <= budget_tokens:
                selected.append(node_id)
                total_tokens += cost
                logger.info(
                    f"Selected node {node_id} (utility={utility_ratio:.4f}, cost={cost})"
                )
            else:
                logger.info(
                    f"Dropping node {node_id} (utility={utility_ratio:.4f}, cost={cost}) to stay within budget"
                )

        # Maintain chronological order
        selected_set = set(selected)
        return [nid for nid in frontier_nodes if nid in selected_set]

    def assemble_with_budget(
        self, retrieval_result: RetrievalResult, token_budget: Optional[int] = None
    ) -> tuple[str, int]:
        """Assemble with strict token budget enforcement."""
        if token_budget is None:
            token_budget = self.config.budget_tokens

        # Check budget strategy configuration
        if self.config.budget_strategy == "drop":
            # Apply drop strategy before assembly
            trimmed_frontier = self.trim_frontier_to_budget(
                retrieval_result.frontier_nodes, token_budget, retrieval_result.scores
            )

            # Re-apply slope cap after trimming to ensure constraints are maintained
            # Removing nodes can create "bridge node" violations where a ±1,±1 sequence
            # becomes a ±2 sequence after dropping the middle node
            if self.config.slope_cap and len(trimmed_frontier) > 1:
                trimmed_frontier = self._apply_slope_cap(trimmed_frontier)

                # Check if slope cap added nodes that exceed budget
                post_slope_cap_tokens = self._count_frontier_tokens(trimmed_frontier)
                if post_slope_cap_tokens > token_budget:
                    logger.warning(
                        f"Slope cap caused budget overflow: {post_slope_cap_tokens} > {token_budget}, re-trimming"
                    )
                    # Second trim pass to restore budget compliance
                    trimmed_frontier = self.trim_frontier_to_budget(
                        trimmed_frontier, token_budget, retrieval_result.scores
                    )

                # Guard against empty frontier after aggressive trimming
                if not trimmed_frontier:
                    root_node = self.store.get_root_node()
                    if root_node:
                        trimmed_frontier = [root_node.id]
                        logger.warning(
                            "Budget trimming left empty frontier, falling back to root node"
                        )

            # Create new retrieval result with trimmed frontier
            trimmed_result = RetrievalResult(
                node_ids=retrieval_result.node_ids,
                scores=retrieval_result.scores,
                coverage_map=retrieval_result.coverage_map,
                frontier_nodes=trimmed_frontier,
            )

            # Assemble with trimmed frontier
            assembled = self.assemble(trimmed_result)
            token_count = self.get_token_count(assembled)

            return assembled, token_count

        else:
            # Use truncate strategy (existing behavior)
            assembled = self.assemble(retrieval_result)
            token_count = self.get_token_count(assembled)

            if token_count <= token_budget:
                return assembled, token_count

            # Over budget - need to truncate
            logger.warning(f"Assembly over budget: {token_count} > {token_budget}")

            # Safely truncate without breaking UTF-8
            tokens = self.tokenizer.encode(assembled)

            # tiktoken handles token boundaries properly
            if len(tokens) > token_budget:
                # Decode only the tokens that fit in budget
                truncated_tokens = tokens[:token_budget]
                truncated_text = self.tokenizer.decode(truncated_tokens)
            else:
                truncated_text = assembled

            return truncated_text, min(token_count, token_budget)
