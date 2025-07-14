"""Tree building and indexing functionality for RagZoom."""

import logging
import uuid
from typing import List, Optional, Tuple

import openai
from openai import OpenAI

from ragzoom.config import RagZoomConfig
from ragzoom.splitter import TextSplitter
from ragzoom.store import Store

logger = logging.getLogger(__name__)


class TreeBuilder:
    """Builds and maintains the hierarchical tree structure."""

    def __init__(self, config: RagZoomConfig, store: Store):
        """Initialize tree builder."""
        self.config = config
        self.store = store
        self.splitter = TextSplitter(config)
        self.client = OpenAI(api_key=config.openai_api_key)

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        return str(uuid.uuid4())

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI."""
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=text,
                dimensions=self.config.embedding_dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def _summarize_text(
        self,
        text: str,
        target_tokens: int,
        prev_context: Optional[str] = None,
        next_context: Optional[str] = None,
    ) -> str:
        """Summarize text using LLM."""
        # Build prompt with adjacent context
        prompt_parts = []
        
        if prev_context:
            prompt_parts.append(f"Previous context: ...{prev_context}")
        
        prompt_parts.append(f"Main passage: {text}")
        
        if next_context:
            prompt_parts.append(f"Next context: {next_context}...")
        
        prompt_parts.append(
            f"\nSummarize the main passage in ≤{target_tokens} tokens. "
            "Use third-person past tense, no pronouns, keep all proper names. "
            "Focus on key events, facts, and themes."
        )
        
        full_prompt = "\n\n".join(prompt_parts)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.summary_model,
                messages=[
                    {"role": "system", "content": "You are a precise summarizer."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=self.config.summary_temperature,
                max_tokens=target_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            raise

    def add_document(self, text: str, document_id: Optional[str] = None) -> str:
        """Add a document to the tree, creating leaf nodes."""
        if not document_id:
            document_id = self._generate_node_id()
        
        # Split into chunks
        chunks = self.splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Create leaf nodes
        leaf_ids = []
        for i, chunk in enumerate(chunks):
            node_id = self._generate_node_id()
            embedding = self._get_embedding(chunk)
            
            # Calculate span positions
            span_start = i * self.config.leaf_tokens
            span_end = span_start + len(self.splitter.tokenizer.encode(chunk))
            
            self.store.add_node(
                node_id=node_id,
                text=chunk,
                embedding=embedding,
                depth=0,
                span_start=span_start,
                span_end=span_end,
            )
            leaf_ids.append(node_id)
        
        # Build tree from leaves
        self._build_tree_from_leaves(leaf_ids, chunks)
        
        return document_id

    def _build_tree_from_leaves(self, leaf_ids: List[str], leaf_texts: List[str]) -> str:
        """Build tree bottom-up from leaf nodes."""
        current_level_ids = leaf_ids
        current_level_texts = leaf_texts
        current_depth = 0
        
        while len(current_level_ids) > 1:
            next_level_ids = []
            next_level_texts = []
            current_depth += 1
            
            # Process pairs of nodes
            for i in range(0, len(current_level_ids), 2):
                left_id = current_level_ids[i]
                left_text = current_level_texts[i]
                
                # Check if we have a right child
                if i + 1 < len(current_level_ids):
                    right_id = current_level_ids[i + 1]
                    right_text = current_level_texts[i + 1]
                    
                    # Create parent node
                    parent_id = self._generate_node_id()
                    
                    # Get adjacent context for summarization
                    prev_context = None
                    next_context = None
                    
                    if i > 0:
                        prev_context, _ = self.splitter.get_adjacent_context(
                            current_level_texts, i - 1
                        )
                    
                    if i + 2 < len(current_level_texts):
                        _, next_context = self.splitter.get_adjacent_context(
                            current_level_texts, i + 1
                        )
                    
                    # Combine texts for parent
                    combined_text = f"{left_text}\n\n{right_text}"
                    
                    # Calculate target tokens for summary
                    target_tokens = self.config.leaf_tokens * 2  # Roughly same size as children
                    
                    # Generate summary
                    summary = self._summarize_text(
                        combined_text, target_tokens, prev_context, next_context
                    )
                    
                    # Get embedding for summary
                    embedding = self._get_embedding(summary)
                    
                    # Calculate span
                    left_node = self.store.get_node(left_id)
                    right_node = self.store.get_node(right_id)
                    
                    self.store.add_node(
                        node_id=parent_id,
                        text=summary,
                        embedding=embedding,
                        depth=current_depth,
                        span_start=left_node.span_start,
                        span_end=right_node.span_end,
                        left_child_id=left_id,
                        right_child_id=right_id,
                        summary=summary,
                    )
                    
                    # Update children's parent references
                    self._update_parent_reference(left_id, parent_id)
                    self._update_parent_reference(right_id, parent_id)
                    
                    next_level_ids.append(parent_id)
                    next_level_texts.append(summary)
                else:
                    # Odd node at end - promote to next level
                    next_level_ids.append(left_id)
                    next_level_texts.append(left_text)
            
            current_level_ids = next_level_ids
            current_level_texts = next_level_texts
        
        # Return root node ID
        return current_level_ids[0] if current_level_ids else None

    def _update_parent_reference(self, node_id: str, parent_id: str) -> None:
        """Update a node's parent reference."""
        # This would typically be done via SQLAlchemy update
        # For now, we'll handle this in the store layer
        pass

    def append_chunks(self, chunks: List[str], continue_from_root: bool = True) -> None:
        """Append new chunks to existing tree."""
        # Get current tree state
        root = self.store.get_root_node()
        if not root and not continue_from_root:
            # Start new tree
            leaf_ids = []
            for chunk in chunks:
                node_id = self._generate_node_id()
                embedding = self._get_embedding(chunk)
                
                self.store.add_node(
                    node_id=node_id,
                    text=chunk,
                    embedding=embedding,
                    depth=0,
                    span_start=0,  # Would need proper calculation
                    span_end=len(chunk),
                )
                leaf_ids.append(node_id)
            
            self._build_tree_from_leaves(leaf_ids, chunks)
        else:
            # Append to existing tree - more complex logic needed
            logger.warning("Incremental append not yet fully implemented")

    def recompute_dirty_summaries(self) -> int:
        """Recompute summaries for nodes marked as dirty."""
        # This would traverse dirty nodes and regenerate summaries
        # Implementation depends on specific update patterns
        logger.info("Recomputing dirty summaries...")
        count = 0
        # TODO: Implement dirty node recomputation
        return count