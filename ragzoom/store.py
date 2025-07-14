"""Storage layer for RagZoom - SQLite for tree structure, Chroma for vectors."""

import logging
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from ragzoom.config import RagZoomConfig

logger = logging.getLogger(__name__)

Base = declarative_base()


class TreeNode(Base):
    """SQLite model for tree nodes."""

    __tablename__ = "tree_nodes"

    id = Column(String, primary_key=True)
    parent_id = Column(String, ForeignKey("tree_nodes.id"), nullable=True)
    left_child_id = Column(String, nullable=True)
    right_child_id = Column(String, nullable=True)
    depth = Column(Integer, nullable=False)
    span_start = Column(Integer, nullable=False)
    span_end = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)  # NULL for leaf nodes
    is_dirty = Column(Integer, default=0)  # Boolean flag for re-summarization
    is_pinned = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class Store:
    """Combined storage for tree structure (SQLite) and embeddings (Chroma)."""

    def __init__(self, config: RagZoomConfig):
        """Initialize storage backends."""
        self.config = config
        
        # Initialize SQLite
        self.engine = create_engine(
            config.sqlite_database_url, connect_args={"check_same_thread": False}
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize Chroma
        self.chroma_client = chromadb.PersistentClient(
            path=config.chroma_persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="ragzoom_nodes",
            metadata={"hnsw:space": "cosine"},
        )
        
        # LRU cache for hot nodes
        self.node_cache: Dict[str, TreeNode] = {}
        self.cache_order = deque(maxlen=1000)

    def _get_from_cache(self, node_id: str) -> Optional[TreeNode]:
        """Get node from cache if available."""
        if node_id in self.node_cache:
            # Move to end (most recently used)
            self.cache_order.remove(node_id)
            self.cache_order.append(node_id)
            return self.node_cache[node_id]
        return None

    def _add_to_cache(self, node: TreeNode) -> None:
        """Add node to cache, evicting LRU if necessary."""
        if node.id in self.node_cache:
            self.cache_order.remove(node.id)
        elif len(self.cache_order) >= self.cache_order.maxlen:
            # Evict LRU
            lru_id = self.cache_order.popleft()
            del self.node_cache[lru_id]
        
        self.node_cache[node.id] = node
        self.cache_order.append(node.id)

    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: List[float],
        depth: int,
        span_start: int,
        span_end: int,
        parent_id: Optional[str] = None,
        left_child_id: Optional[str] = None,
        right_child_id: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> TreeNode:
        """Add a node to both SQLite and Chroma."""
        with self.SessionLocal() as session:
            node = TreeNode(
                id=node_id,
                parent_id=parent_id,
                left_child_id=left_child_id,
                right_child_id=right_child_id,
                depth=depth,
                span_start=span_start,
                span_end=span_end,
                text=text,
                summary=summary,
            )
            session.add(node)
            session.commit()
            
            # Add to cache
            self._add_to_cache(node)
        
        # Add to Chroma
        self.collection.add(
            ids=[node_id],
            embeddings=[embedding],
            metadatas=[{
                "depth": depth,
                "span_start": span_start,
                "span_end": span_end,
                "parent_id": parent_id or "",
                "is_leaf": 1 if summary is None else 0,
            }],
            documents=[text],
        )
        
        return node

    def get_node(self, node_id: str) -> Optional[TreeNode]:
        """Get a node by ID."""
        # Check cache first
        cached = self._get_from_cache(node_id)
        if cached:
            return cached
        
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                self._add_to_cache(node)
            return node

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node."""
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                node.last_accessed = datetime.utcnow()
                node.access_count += 1
                session.commit()

    def mark_dirty_upward(self, node_id: str) -> None:
        """Mark a node and all ancestors as dirty (needs re-summarization)."""
        with self.SessionLocal() as session:
            current_id = node_id
            while current_id:
                node = session.query(TreeNode).filter_by(id=current_id).first()
                if not node:
                    break
                node.is_dirty = 1
                current_id = node.parent_id
            session.commit()

    def get_children(self, node_id: str) -> Tuple[Optional[TreeNode], Optional[TreeNode]]:
        """Get left and right children of a node."""
        node = self.get_node(node_id)
        if not node:
            return None, None
        
        left = self.get_node(node.left_child_id) if node.left_child_id else None
        right = self.get_node(node.right_child_id) if node.right_child_id else None
        return left, right

    def get_ancestors(self, node_ids: List[str]) -> List[TreeNode]:
        """Get all ancestors of given nodes."""
        ancestors = set()
        to_process = list(node_ids)
        
        while to_process:
            node_id = to_process.pop()
            node = self.get_node(node_id)
            if node and node.parent_id and node.parent_id not in ancestors:
                ancestors.add(node.parent_id)
                to_process.append(node.parent_id)
        
        return [self.get_node(aid) for aid in ancestors if self.get_node(aid)]

    def search_similar(
        self, query_embedding: List[float], n_results: int, where: Optional[dict] = None
    ) -> List[Tuple[str, float, dict]]:
        """Search for similar nodes using Chroma."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
        )
        
        # Return list of (id, distance, metadata) tuples
        output = []
        for i in range(len(results["ids"][0])):
            output.append((
                results["ids"][0][i],
                results["distances"][0][i],
                results["metadatas"][0][i],
            ))
        
        return output

    def get_pinned_nodes(self, max_depth: Optional[int] = None) -> List[TreeNode]:
        """Get all pinned nodes up to a certain depth."""
        with self.SessionLocal() as session:
            query = session.query(TreeNode).filter_by(is_pinned=1)
            if max_depth is not None:
                query = query.filter(TreeNode.depth <= max_depth)
            return query.all()

    def pin_node(self, node_id: str) -> bool:
        """Pin a node if it's within allowed depth."""
        node = self.get_node(node_id)
        if not node or node.depth > self.config.pin_depth_max:
            return False
        
        with self.SessionLocal() as session:
            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                node.is_pinned = 1
                session.commit()
                return True
        return False

    def get_leaf_nodes(self) -> List[TreeNode]:
        """Get all leaf nodes (nodes without children)."""
        with self.SessionLocal() as session:
            return session.query(TreeNode).filter_by(summary=None).all()

    def get_root_node(self) -> Optional[TreeNode]:
        """Get the root node (node without parent)."""
        with self.SessionLocal() as session:
            return session.query(TreeNode).filter_by(parent_id=None).first()

    def compute_mmr_diverse_results(
        self,
        query_embedding: List[float],
        candidates: List[Tuple[str, float, dict]],
        lambda_param: float,
        k: int,
    ) -> List[str]:
        """Apply MMR (Maximal Marginal Relevance) to get diverse results."""
        if not candidates or k <= 0:
            return []
        
        # Get embeddings for all candidates
        candidate_ids = [c[0] for c in candidates]
        candidate_embeddings = []
        
        results = self.collection.get(
            ids=candidate_ids,
            include=["embeddings"]
        )
        
        for i, cid in enumerate(candidate_ids):
            idx = results["ids"].index(cid)
            candidate_embeddings.append(results["embeddings"][idx])
        
        # Convert to numpy for efficient computation
        query_emb = np.array(query_embedding)
        cand_embs = np.array(candidate_embeddings)
        
        # Compute similarities to query
        query_sims = np.dot(cand_embs, query_emb)
        
        # MMR iterative selection
        selected_indices = []
        unselected_indices = list(range(len(candidates)))
        
        # Select first item (highest relevance)
        first_idx = np.argmax(query_sims)
        selected_indices.append(first_idx)
        unselected_indices.remove(first_idx)
        
        # Select remaining items
        while len(selected_indices) < k and unselected_indices:
            selected_embs = cand_embs[selected_indices]
            
            mmr_scores = []
            for idx in unselected_indices:
                # Relevance to query
                relevance = query_sims[idx]
                
                # Max similarity to already selected
                sims_to_selected = np.dot(selected_embs, cand_embs[idx])
                max_sim = np.max(sims_to_selected) if len(sims_to_selected) > 0 else 0
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            
            # Select best MMR score
            best_idx = unselected_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            unselected_indices.remove(best_idx)
        
        # Return selected node IDs
        return [candidates[i][0] for i in selected_indices]