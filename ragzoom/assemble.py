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
        
        # Apply slope cap
        capped_nodes = [node_depths[0]]
        
        for i in range(1, len(node_depths)):
            current_id, current_depth = node_depths[i]
            prev_id, prev_depth = capped_nodes[-1]
            
            # Check depth difference
            depth_diff = abs(current_depth - prev_depth)
            
            if depth_diff <= 1:
                # Within slope cap
                capped_nodes.append((current_id, current_depth))
            else:
                # Need to find intermediate nodes
                logger.info(f"Slope cap violation: {prev_depth} -> {current_depth}")
                
                # Try to find a path with gradual depth changes
                intermediate = self._find_intermediate_path(prev_id, current_id)
                if intermediate:
                    capped_nodes.extend(intermediate)
                else:
                    # If no path found, include anyway but log warning
                    logger.warning(f"No intermediate path found, including anyway")
                    capped_nodes.append((current_id, current_depth))
        
        return [node_id for node_id, _ in capped_nodes]

    def _find_intermediate_path(
        self, start_id: str, end_id: str
    ) -> List[Tuple[str, int]]:
        """Find intermediate nodes to satisfy slope cap between two nodes."""
        start_node = self.store.get_node(start_id)
        end_node = self.store.get_node(end_id)
        
        if not start_node or not end_node:
            return []
        
        # This is a simplified version - in practice, you'd want a more
        # sophisticated path-finding algorithm
        path = []
        
        # If going up in depth (toward root)
        if start_node.depth > end_node.depth:
            current_id = start_id
            while current_id:
                node = self.store.get_node(current_id)
                if not node:
                    break
                
                if node.depth <= end_node.depth:
                    break
                
                if node.parent_id:
                    parent = self.store.get_node(node.parent_id)
                    if parent and abs(parent.depth - node.depth) == 1:
                        path.append((parent.id, parent.depth))
                        current_id = parent.id
                else:
                    break
        
        return path

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
        
        # Truncate from the end (could be smarter about this)
        tokens = self.tokenizer.encode(assembled)
        truncated_tokens = tokens[:token_budget]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        return truncated_text, token_budget