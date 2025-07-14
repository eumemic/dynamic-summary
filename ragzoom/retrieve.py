"""Retrieval logic with MMR diversity for RagZoom."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from openai import OpenAI

from ragzoom.config import RagZoomConfig
from ragzoom.store import Store, TreeNode

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    node_ids: List[str]
    scores: Dict[str, float]
    coverage_map: Dict[str, bool]
    frontier_nodes: List[str]


class Retriever:
    """Handles retrieval and MMR diversity for query processing."""

    def __init__(self, config: RagZoomConfig, store: Store):
        """Initialize retriever."""
        self.config = config
        self.store = store
        self.client = OpenAI(api_key=config.openai_api_key)
        
        # Track access history for freshness scoring
        self.access_history: Dict[str, Tuple[float, int]] = {}  # node_id -> (similarity, turns_ago)
        self.current_turn = 0

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query text."""
        try:
            response = self.client.embeddings.create(
                model=self.config.embedding_model,
                input=query,
                dimensions=self.config.embedding_dimensions,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting query embedding: {e}")
            raise

    def retrieve(self, query: str, n_max: Optional[int] = None) -> RetrievalResult:
        """Main retrieval method with MMR diversity."""
        # Calculate n_max if not provided
        if n_max is None:
            n_max = self.config.n_max
        
        # Get query embedding
        query_embedding = self._get_query_embedding(query)
        
        # Step 1: Initial retrieval (2 * n_max candidates)
        k_candidates = int(n_max * self.config.mmr_k_multiplier)
        candidates = self.store.search_similar(query_embedding, k_candidates)
        
        # Step 2: Apply MMR to get diverse n_max results
        selected_ids = self.store.compute_mmr_diverse_results(
            query_embedding,
            candidates,
            self.config.mmr_lambda,
            n_max
        )
        
        # Step 3: Build coverage map (selected + ancestors)
        coverage_map = self._build_coverage_map(selected_ids)
        
        # Step 4: Apply pinned nodes
        pinned_nodes = self.store.get_pinned_nodes(self.config.pin_depth_max)
        for node in pinned_nodes:
            coverage_map[node.id] = True
        
        # Step 5: Extract frontier
        frontier_nodes = self._extract_frontier(coverage_map)
        
        # Update access history
        self._update_access_history(selected_ids, candidates)
        
        # Build scores map
        scores = {cand[0]: 1.0 - cand[1] for cand in candidates}  # Convert distance to similarity
        
        return RetrievalResult(
            node_ids=selected_ids,
            scores=scores,
            coverage_map=coverage_map,
            frontier_nodes=frontier_nodes,
        )

    def _build_coverage_map(self, selected_ids: List[str]) -> Dict[str, bool]:
        """Build coverage map including selected nodes and their ancestors."""
        coverage_map = {}
        
        # Mark selected nodes as covered
        for node_id in selected_ids:
            coverage_map[node_id] = True
            self.store.update_node_access(node_id)
        
        # Get and mark ancestors
        ancestors = self.store.get_ancestors(selected_ids)
        for ancestor in ancestors:
            coverage_map[ancestor.id] = True
        
        return coverage_map

    def _extract_frontier(self, coverage_map: Dict[str, bool]) -> List[str]:
        """Extract frontier nodes (covered nodes with uncovered children)."""
        frontier = []
        
        # Check each covered node
        for node_id in coverage_map:
            if not coverage_map.get(node_id):
                continue
            
            node = self.store.get_node(node_id)
            if not node:
                continue
            
            # Check if this is a frontier node
            is_frontier = True
            
            # Get children
            left_child, right_child = self.store.get_children(node_id)
            
            # If both children are covered, this is not a frontier node
            if left_child and coverage_map.get(left_child.id):
                if right_child and coverage_map.get(right_child.id):
                    is_frontier = False
            
            # If no children, it's a leaf and thus frontier
            if not left_child and not right_child:
                is_frontier = True
            
            if is_frontier:
                frontier.append(node_id)
        
        # Sort frontier by span_start for chronological order
        frontier_nodes = []
        for node_id in frontier:
            node = self.store.get_node(node_id)
            if node:
                frontier_nodes.append((node.span_start, node_id))
        
        frontier_nodes.sort()
        return [node_id for _, node_id in frontier_nodes]

    def _update_access_history(
        self, selected_ids: List[str], candidates: List[Tuple[str, float, dict]]
    ) -> None:
        """Update access history for freshness scoring."""
        self.current_turn += 1
        
        # Update selected nodes
        for node_id in selected_ids:
            # Find similarity score
            sim_score = 1.0
            for cand in candidates:
                if cand[0] == node_id:
                    sim_score = 1.0 - cand[1]  # Convert distance to similarity
                    break
            
            self.access_history[node_id] = (sim_score, 0)
        
        # Age existing entries
        for node_id in list(self.access_history.keys()):
            if node_id not in selected_ids:
                sim, turns_ago = self.access_history[node_id]
                self.access_history[node_id] = (sim, turns_ago + 1)
                
                # Remove old entries based on TTL
                if self.config.ttl_turns > 0 and turns_ago + 1 > self.config.ttl_turns:
                    del self.access_history[node_id]

    def get_priority_scores(self) -> Dict[str, float]:
        """Calculate priority scores for sliding queue eviction."""
        priority_scores = {}
        
        # Guard against div-by-zero if decay is 0
        decay = self.config.freshness_decay
        if decay <= 0 or decay > 1:
            decay = 0.9  # Safe default
            logger.warning(f"Invalid freshness_decay {self.config.freshness_decay}, using 0.9")
        
        for node_id, (similarity, turns_ago) in self.access_history.items():
            # Priority = similarity * decay^turns_ago
            priority = similarity * (decay ** turns_ago)
            priority_scores[node_id] = priority
        
        return priority_scores

    def retrieve_with_eviction(self, query: str, token_budget: Optional[int] = None) -> RetrievalResult:
        """Retrieve with sliding queue eviction to fit token budget."""
        if token_budget is None:
            token_budget = self.config.budget_tokens
        
        # Initial retrieval
        result = self.retrieve(query)
        
        # Calculate token usage
        total_tokens = 0
        node_tokens = {}
        
        import tiktoken
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        for node_id in result.frontier_nodes:
            node = self.store.get_node(node_id)
            if node:
                tokens = len(tokenizer.encode(node.text))
                node_tokens[node_id] = tokens
                total_tokens += tokens
        
        # If within budget, return as-is
        if total_tokens <= token_budget:
            return result
        
        # Need eviction - get priority scores
        priority_scores = self.get_priority_scores()
        
        # Sort frontier nodes by priority (lowest first for eviction)
        frontier_with_priority = []
        for node_id in result.frontier_nodes:
            priority = priority_scores.get(node_id, 0.0)
            frontier_with_priority.append((priority, node_id))
        
        frontier_with_priority.sort()
        
        # Evict nodes until within budget
        evicted = set()
        for priority, node_id in frontier_with_priority:
            if total_tokens <= token_budget:
                break
            
            if node_id in node_tokens:
                total_tokens -= node_tokens[node_id]
                evicted.add(node_id)
        
        # Update frontier
        result.frontier_nodes = [
            nid for nid in result.frontier_nodes if nid not in evicted
        ]
        
        return result