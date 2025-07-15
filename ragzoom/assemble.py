"""Assembly logic for creating coherent summaries from frontier nodes."""

import logging
from typing import List, Optional, Tuple

import tiktoken
from openai import OpenAI

from ragzoom.config import RagZoomConfig
from ragzoom.retrieve import RetrievalResult
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
        """Assemble frontier nodes into final summary."""
        # Get frontier nodes in order
        frontier_nodes = retrieval_result.frontier_nodes

        if not frontier_nodes:
            # Return root synopsis if available
            root = self.store.get_root_node()
            if root:
                return root.text
            else:
                logger.warning("No frontier nodes and no root node found")
                return ""

        # Step 1: Remove children if their parents are in frontier
        # This handles invalid frontiers where both parent and child are included
        frontier_nodes = self._remove_children_with_parents_in_frontier(frontier_nodes)

        # Step 2: Apply slope cap if enabled
        if self.config.slope_cap:
            frontier_nodes = self._apply_slope_cap(frontier_nodes)

        # Step 3: Deduplicate by node ID (slope-cap may create duplicates)
        seen_node_ids = set()
        unique_frontier = []
        for node_id in frontier_nodes:
            if node_id not in seen_node_ids:
                unique_frontier.append(node_id)
                seen_node_ids.add(node_id)
        frontier_nodes = unique_frontier

        # Step 4: Sort frontier nodes by span_start for chronological order
        frontier_nodes = self._sort_nodes_chronologically(frontier_nodes)

        # Step 5: Build coverage map AFTER all frontier mutations
        final_coverage_map = self._build_coverage_map(frontier_nodes)

        # Step 6: Extract texts with <<<MID>>> delimiter handling and span deduplication
        texts = []
        seen_spans = set()  # Store (span_start, span_end) for span dedup
        frontier_set = set(frontier_nodes)

        for node_id in frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                # Check span BEFORE extracting text
                span = (node.span_start, node.span_end)
                if span not in seen_spans:
                    text = self._extract_node_text(node, final_coverage_map, frontier_set)
                    if text:  # Only add non-empty text
                        logger.info(f"Extracted from node {node_id} (depth {node.depth}, span {span}): {len(text)} chars")
                        texts.append(text)
                    else:
                        logger.info(f"Node {node_id} produced empty text, skipping")
                    seen_spans.add(span)
                else:
                    logger.info(f"Skipping duplicate span node {node_id} with span {span}")

        # Basic concatenation
        assembled = "\n\n".join(texts)

        # Apply smoothing pass if enabled
        if self.config.smoothing_pass_enabled:
            assembled = self._apply_smoothing_pass(frontier_nodes, texts)

        return assembled

    def _clean_mid_delimiter(self, text: str) -> str:
        """Remove <<<MID>>> delimiter from text."""
        return text.replace('<<<MID>>>', '')

    def _extract_node_text(self, node, coverage_map, frontier_set=None):
        """Extract appropriate text from node based on <<<MID>>> delimiter logic."""
        # If node has no mid_offset, return full text (with <<<MID>>> removed if present)
        if not hasattr(node, 'mid_offset') or node.mid_offset is None:
            logger.info(f"Node {node.id} has no mid_offset, using full text")
            return self._clean_mid_delimiter(node.text)

        # Get children
        left_child, right_child = self.store.get_children(node.id)

        # Check if children are in the frontier (not coverage map for parent processing)
        if frontier_set:
            left_in_frontier = left_child and left_child.id in frontier_set
            right_in_frontier = right_child and right_child.id in frontier_set
        else:
            left_in_frontier = False
            right_in_frontier = False

        # For parent nodes: if a child is in the frontier, it will handle its own text
        # So we should only include the OTHER child's summary from the parent
        if left_in_frontier and not right_in_frontier:
            # Left child will output its own text, parent should only output right summary
            mid_delimiter_len = len('<<<MID>>>')
            parent_right = node.text[node.mid_offset + mid_delimiter_len:].strip()
            logger.info(f"Node {node.id}: left child in frontier, outputting only right summary")
            return parent_right

        elif right_in_frontier and not left_in_frontier:
            # Right child will output its own text, parent should only output left summary
            parent_left = node.text[:node.mid_offset]
            logger.info(f"Node {node.id}: right child in frontier, outputting only left summary")
            return parent_left.strip()

        elif left_in_frontier and right_in_frontier:
            # Both children in frontier - parent should output nothing
            logger.info(f"Node {node.id}: both children in frontier, skipping parent")
            return ""

        # Neither child is in frontier - use normal coverage logic
        left_covered = left_child and left_child.id in coverage_map
        right_covered = right_child and right_child.id in coverage_map

        logger.info(f"Node {node.id}: left_child={left_child.id if left_child else None} (covered={left_covered}), right_child={right_child.id if right_child else None} (covered={right_covered})")

        # Apply the three cases for normal coverage
        if left_covered and not right_covered:
            # Use left child + parent's right half
            left_text = left_child.text
            mid_delimiter_len = len('<<<MID>>>')
            parent_right = node.text[node.mid_offset + mid_delimiter_len:]
            logger.info("Case 1: Using left child + parent right half")
            return f"{left_text}\n\n{parent_right}"

        elif right_covered and not left_covered:
            # Use parent's left half + right child
            parent_left = node.text[:node.mid_offset]
            right_text = right_child.text
            logger.info("Case 2: Using parent left half + right child")
            return f"{parent_left}\n\n{right_text}"

        else:
            # Both or neither covered - use full parent (remove <<<MID>>>)
            logger.info("Case 3: Both or neither covered, using full parent")
            return self._clean_mid_delimiter(node.text)

    def _has_span_overlap_detailed(self, span, seen_items):
        """Check if span overlaps with any item in seen_items (includes depth/node info)."""
        span_start, span_end = span
        for seen_start, seen_end, seen_depth, seen_id in seen_items:
            # Two spans overlap if one doesn't end before the other starts
            if not (span_end <= seen_start or seen_end <= span_start):
                return True
        return False

    def _sort_nodes_chronologically(self, frontier_nodes):
        """Sort frontier nodes by span_start for chronological order."""
        # Get nodes with their span_start values
        node_spans = []
        for node_id in frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                node_spans.append((node_id, node.span_start))

        # Sort by span_start, then by depth (leaves first - depth 0 before depth 1)
        node_spans.sort(key=lambda x: (x[1], self.store.get_node(x[0]).depth))

        # Return just the node IDs in sorted order
        return [node_id for node_id, _ in node_spans]

    def _remove_children_with_parents_in_frontier(self, frontier_nodes):
        """Smart deduplication: only remove nodes when their content would be fully redundant."""
        # Build set of all nodes in frontier for quick lookup
        frontier_set = set(frontier_nodes)

        # Track which children each parent has in the frontier
        parent_children_in_frontier = {}  # parent_id -> set of child_ids

        for node_id in frontier_nodes:
            node = self.store.get_node(node_id)
            if node and node.parent_id and node.parent_id in frontier_set:
                if node.parent_id not in parent_children_in_frontier:
                    parent_children_in_frontier[node.parent_id] = set()
                parent_children_in_frontier[node.parent_id].add(node_id)

        # Decide what to remove
        nodes_to_remove = set()

        for parent_id, children_in_frontier in parent_children_in_frontier.items():
            # Get all children of this parent
            left_child, right_child = self.store.get_children(parent_id)

            # Count how many children are in frontier
            left_in_frontier = left_child and left_child.id in children_in_frontier
            right_in_frontier = right_child and right_child.id in children_in_frontier

            if left_in_frontier and right_in_frontier:
                # Both children are in frontier - parent is redundant
                nodes_to_remove.add(parent_id)
                logger.info(f"Removing parent {parent_id} because both children are in frontier")
            # If only one child is in frontier, keep both parent and child
            # The <<<MID>>> extraction will combine them properly

        # Return filtered list maintaining order
        return [n for n in frontier_nodes if n not in nodes_to_remove]

    def _build_coverage_map(self, frontier_nodes):
        """Build coverage map from final frontier nodes (includes ancestors)."""
        coverage_map = set()

        for node_id in frontier_nodes:
            # Mark this node and all its ancestors as covered
            current_id = node_id
            while current_id:
                coverage_map.add(current_id)
                node = self.store.get_node(current_id)
                current_id = node.parent_id if node else None

        return coverage_map

    def _apply_slope_cap(self, frontier_nodes: List[str]) -> List[str]:
        """Apply slope cap constraint (max ±1 depth change between adjacent nodes)."""
        if len(frontier_nodes) <= 1:
            return frontier_nodes

        # Get depths
        node_depths = []
        for node_id in frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                node_depths.append((node_id, node.depth))

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
                target_depth = prev_depth + 1 if current_depth > prev_depth else prev_depth - 1

                # Find ancestor at target depth
                replacement_node = self._find_ancestor_at_depth(current_id, target_depth)
                if replacement_node and replacement_node.id not in seen:
                    capped_nodes.append((replacement_node.id, replacement_node.depth))
                    seen.add(replacement_node.id)
                elif current_id not in seen:
                    # Fallback to original node if no suitable ancestor
                    capped_nodes.append((current_id, current_depth))
                    seen.add(current_id)

        # Re-sort by span_start after replacements to maintain chronological order
        capped_with_spans = []
        for node_id, depth in capped_nodes:
            node = self.store.get_node(node_id)
            if node:
                capped_with_spans.append((node_id, depth, node.span_start))

        capped_with_spans.sort(key=lambda x: x[2])  # Sort by span_start
        return [node_id for node_id, _, _ in capped_with_spans]

    def _find_ancestor_at_depth(self, node_id: str, target_depth: int):
        """Find ancestor of node at target depth."""
        current_node = self.store.get_node(node_id)

        while current_node and current_node.depth > target_depth:
            if current_node.parent_id:
                current_node = self.store.get_node(current_node.parent_id)
            else:
                break

        return current_node if current_node and current_node.depth == target_depth else None

    def _find_intermediate_path(
        self, start_id: str, end_id: str
    ) -> List[Tuple[str, int]]:
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
            node = session.query(TreeNode).filter(
                TreeNode.depth == depth,
                TreeNode.span_start < span_end,
                TreeNode.span_end > span_start
            ).first()
            return node

    def _apply_smoothing_pass(
        self, frontier_nodes: List[str], texts: List[str]
    ) -> str:
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
                    "start"
                )
                smoothed_parts.append(smoothed)
            elif i == len(texts) - 1:
                # Last chunk
                smoothed = self._smooth_boundary(
                    texts[i - 1],
                    texts[i],
                    None,
                    "end"
                )
                smoothed_parts.append(smoothed)
            else:
                # Middle chunk
                smoothed = self._smooth_boundary(
                    texts[i - 1],
                    texts[i],
                    texts[i + 1],
                    "middle"
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
            return response.choices[0].message.content.strip()
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
        self, frontier_nodes: List[str], budget_tokens: int, scores: dict[str, float]
    ) -> List[str]:
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
                logger.info(f"Selected node {node_id} (utility={utility_ratio:.4f}, cost={cost})")
            else:
                logger.info(f"Dropping node {node_id} (utility={utility_ratio:.4f}, cost={cost}) to stay within budget")

        # Maintain chronological order
        selected_set = set(selected)
        return [nid for nid in frontier_nodes if nid in selected_set]

    def assemble_with_budget(
        self, retrieval_result: RetrievalResult, token_budget: Optional[int] = None
    ) -> Tuple[str, int]:
        """Assemble with strict token budget enforcement."""
        if token_budget is None:
            token_budget = self.config.budget_tokens

        # Check budget strategy configuration
        if self.config.budget_strategy == "drop":
            # Apply drop strategy before assembly
            trimmed_frontier = self.trim_frontier_to_budget(
                retrieval_result.frontier_nodes,
                token_budget,
                retrieval_result.scores
            )

            # Re-apply slope cap after trimming to ensure constraints are maintained
            # Removing nodes can create "bridge node" violations where a ±1,±1 sequence
            # becomes a ±2 sequence after dropping the middle node
            if self.config.slope_cap and len(trimmed_frontier) > 1:
                trimmed_frontier = self._apply_slope_cap(trimmed_frontier)
                
                # Check if slope cap added nodes that exceed budget
                post_slope_cap_tokens = self._count_frontier_tokens(trimmed_frontier)
                if post_slope_cap_tokens > token_budget:
                    logger.warning(f"Slope cap caused budget overflow: {post_slope_cap_tokens} > {token_budget}, re-trimming")
                    # Second trim pass to restore budget compliance
                    trimmed_frontier = self.trim_frontier_to_budget(
                        trimmed_frontier,
                        token_budget,
                        retrieval_result.scores
                    )
                    
                # Guard against empty frontier after aggressive trimming
                if not trimmed_frontier:
                    root_node = self.store.get_root_node()
                    if root_node:
                        trimmed_frontier = [root_node.id]
                        logger.warning("Budget trimming left empty frontier, falling back to root node")

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
