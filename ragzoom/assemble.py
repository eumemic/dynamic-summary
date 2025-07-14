"""Assembly logic for creating coherent summaries from frontier nodes."""

import logging
from typing import List, Optional, Tuple

from openai import OpenAI

import tiktoken

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
        
        # Apply slope cap if enabled
        if self.config.slope_cap:
            frontier_nodes = self._apply_slope_cap(frontier_nodes)
        
        # Concatenate frontier texts
        texts = []
        for node_id in frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                texts.append(node.text)
        
        # Basic concatenation
        assembled = "\n\n".join(texts)
        
        # Apply smoothing pass if enabled
        if self.config.smoothing_pass_enabled:
            assembled = self._apply_smoothing_pass(frontier_nodes, texts)
        
        return assembled

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
        
        # Apply slope cap with deduplication
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
                # Need to find intermediate nodes
                logger.info(f"Slope cap violation: {prev_depth} -> {current_depth}")
                
                # Try to find a path with gradual depth changes
                intermediate = self._find_intermediate_path(prev_id, current_id)
                if intermediate:
                    # Add only unseen intermediates
                    for node_id, depth in intermediate:
                        if node_id not in seen:
                            capped_nodes.append((node_id, depth))
                            seen.add(node_id)
                
                # Always add the target node if not seen
                if current_id not in seen:
                    capped_nodes.append((current_id, current_depth))
                    seen.add(current_id)
        
        return [node_id for node_id, _ in capped_nodes]

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
            node = session.query(self.store.TreeNode).filter(
                self.store.TreeNode.depth == depth,
                self.store.TreeNode.span_start < span_end,
                self.store.TreeNode.span_end > span_start
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

    def assemble_with_budget(
        self, retrieval_result: RetrievalResult, token_budget: Optional[int] = None
    ) -> Tuple[str, int]:
        """Assemble with strict token budget enforcement."""
        if token_budget is None:
            token_budget = self.config.budget_tokens
        
        # Assemble normally
        assembled = self.assemble(retrieval_result)
        
        # Check token count
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