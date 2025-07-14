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
            
            # Calculate span positions in tokens (not bytes)
            tokens_before = sum(len(self.splitter.tokenizer.encode(chunks[j])) 
                              for j in range(i))
            chunk_tokens = len(self.splitter.tokenizer.encode(chunk))
            span_start = tokens_before
            span_end = span_start + chunk_tokens
            
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
                    
                    # Calculate target tokens for summary (≤½ of combined children)
                    left_tokens = len(self.splitter.tokenizer.encode(left_text))
                    right_tokens = len(self.splitter.tokenizer.encode(right_text))
                    target_tokens = max((left_tokens + right_tokens) // 2, 50)
                    
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
        with self.store.SessionLocal() as session:
            node = session.query(self.store.TreeNode).filter_by(id=node_id).first()
            if node:
                node.parent_id = parent_id
                session.commit()

    def append_chunks(self, chunks: List[str], document_id: Optional[str] = None) -> str:
        """Append new chunks to existing tree."""
        if not document_id:
            document_id = self._generate_node_id()
            
        # Get the rightmost leaf to calculate span offsets
        rightmost_leaf = self._get_rightmost_leaf()
        span_offset = rightmost_leaf.span_end if rightmost_leaf else 0
        
        # Create new leaf nodes
        new_leaf_ids = []
        for i, chunk in enumerate(chunks):
            node_id = self._generate_node_id()
            embedding = self._get_embedding(chunk)
            
            # Calculate span in tokens with proper offset
            tokens_before = sum(len(self.splitter.tokenizer.encode(chunks[j])) 
                              for j in range(i))
            chunk_tokens = len(self.splitter.tokenizer.encode(chunk))
            span_start = span_offset + tokens_before
            span_end = span_start + chunk_tokens
            
            self.store.add_node(
                node_id=node_id,
                text=chunk,
                embedding=embedding,
                depth=0,
                span_start=span_start,
                span_end=span_end,
            )
            new_leaf_ids.append(node_id)
        
        # Now merge new leaves into existing tree
        self._merge_into_tree(new_leaf_ids, chunks)
        return document_id
    
    def _get_rightmost_leaf(self) -> Optional["TreeNode"]:
        """Get the rightmost (highest span_end) leaf node."""
        with self.store.SessionLocal() as session:
            return session.query(self.store.TreeNode).filter_by(
                summary=None  # Leaf nodes have no summary
            ).order_by(self.store.TreeNode.span_end.desc()).first()
    
    def _merge_into_tree(self, new_leaf_ids: List[str], new_leaf_texts: List[str]) -> None:
        """Merge new leaves into the existing tree structure."""
        # Find nodes at depth 0 that need new parents
        existing_leaves = self.store.get_leaf_nodes()
        
        # Group leaves that need to be paired
        unpaired_leaves = []
        for leaf in existing_leaves:
            if not leaf.parent_id:
                unpaired_leaves.append(leaf)
        
        # Combine unpaired existing leaves with new leaves
        all_unpaired_ids = [leaf.id for leaf in unpaired_leaves] + new_leaf_ids
        all_unpaired_texts = [leaf.text for leaf in unpaired_leaves] + new_leaf_texts
        
        # Build tree from all unpaired leaves
        if len(all_unpaired_ids) > 1:
            root_id = self._build_tree_from_leaves(all_unpaired_ids, all_unpaired_texts)
            
            # Connect new subtree to existing root if needed
            existing_root = self.store.get_root_node()
            if existing_root and root_id != existing_root.id:
                # Create new root combining old and new
                new_root_id = self._generate_node_id()
                combined_text = f"{existing_root.text}\n\n{self.store.get_node(root_id).text}"
                
                # Summary for new root
                existing_tokens = len(self.splitter.tokenizer.encode(existing_root.text))
                new_subtree_tokens = len(self.splitter.tokenizer.encode(self.store.get_node(root_id).text))
                target_tokens = max((existing_tokens + new_subtree_tokens) // 2, 50)
                summary = self._summarize_text(combined_text, target_tokens)
                embedding = self._get_embedding(summary)
                
                # Get combined span
                new_subtree_root = self.store.get_node(root_id)
                
                self.store.add_node(
                    node_id=new_root_id,
                    text=summary,
                    embedding=embedding,
                    depth=max(existing_root.depth, new_subtree_root.depth) + 1,
                    span_start=min(existing_root.span_start, new_subtree_root.span_start),
                    span_end=max(existing_root.span_end, new_subtree_root.span_end),
                    left_child_id=existing_root.id,
                    right_child_id=root_id,
                    summary=summary,
                )
                
                # Update parent references
                self._update_parent_reference(existing_root.id, new_root_id)
                self._update_parent_reference(root_id, new_root_id)

    def recompute_dirty_summaries(self) -> int:
        """Recompute summaries for nodes marked as dirty."""
        logger.info("Recomputing dirty summaries...")
        count = 0
        
        with self.store.SessionLocal() as session:
            # Get all dirty nodes, ordered by depth (bottom-up)
            dirty_nodes = session.query(self.store.TreeNode).filter_by(
                is_dirty=1
            ).order_by(self.store.TreeNode.depth).all()
            
            for node in dirty_nodes:
                if node.left_child_id and node.right_child_id:
                    # Get children
                    left_child = session.query(self.store.TreeNode).filter_by(
                        id=node.left_child_id
                    ).first()
                    right_child = session.query(self.store.TreeNode).filter_by(
                        id=node.right_child_id
                    ).first()
                    
                    if left_child and right_child:
                        # Combine texts
                        combined_text = f"{left_child.text}\n\n{right_child.text}"
                        
                        # Get adjacent context
                        prev_context, next_context = self._get_node_context(session, node)
                        
                        # Regenerate summary
                        target_tokens = self.config.leaf_tokens
                        new_summary = self._summarize_text(
                            combined_text, target_tokens, prev_context, next_context
                        )
                        
                        # Update node
                        node.text = new_summary
                        node.summary = new_summary
                        node.is_dirty = 0
                        
                        # Update embedding
                        new_embedding = self._get_embedding(new_summary)
                        self.store.collection.update(
                            ids=[node.id],
                            embeddings=[new_embedding],
                            documents=[new_summary]
                        )
                        
                        count += 1
                        logger.info(f"Recomputed summary for node {node.id}")
            
            session.commit()
        
        return count
    
    def _get_node_context(
        self, session, node: "TreeNode"
    ) -> Tuple[Optional[str], Optional[str]]:
        """Get adjacent context for a node during re-summarization."""
        # Find siblings at same depth with adjacent spans
        prev_node = session.query(self.store.TreeNode).filter(
            self.store.TreeNode.depth == node.depth,
            self.store.TreeNode.span_end <= node.span_start
        ).order_by(self.store.TreeNode.span_end.desc()).first()
        
        next_node = session.query(self.store.TreeNode).filter(
            self.store.TreeNode.depth == node.depth,
            self.store.TreeNode.span_start >= node.span_end
        ).order_by(self.store.TreeNode.span_start).first()
        
        prev_context = None
        next_context = None
        
        if prev_node:
            prev_tokens = self.splitter.tokenizer.encode(prev_node.text)
            if len(prev_tokens) > self.config.adjacent_context_tokens:
                prev_context = self.splitter.tokenizer.decode(
                    prev_tokens[-self.config.adjacent_context_tokens:]
                )
            else:
                prev_context = prev_node.text
        
        if next_node:
            next_tokens = self.splitter.tokenizer.encode(next_node.text)
            if len(next_tokens) > self.config.adjacent_context_tokens:
                next_context = self.splitter.tokenizer.decode(
                    next_tokens[:self.config.adjacent_context_tokens]
                )
            else:
                next_context = next_node.text
        
        return prev_context, next_context