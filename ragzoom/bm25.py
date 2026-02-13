"""BM25 lexical search for hybrid retrieval.

Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index

This module provides BM25 (Okapi BM25) lexical search to complement vector
similarity search. BM25 excels at finding exact term matches like names,
IDs, error codes, and technical jargon that embeddings may miss.
"""

import re
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragzoom.contracts.tree_node import TreeNode


def _create_bm25_index(corpus: Sequence[list[str]]) -> object:
    """Create a BM25Okapi index from tokenized corpus.

    Factored out to isolate the untyped import from the class definition.
    """
    from rank_bm25 import BM25Okapi

    return BM25Okapi(corpus)


def _get_bm25_scores(bm25: object, tokens: list[str]) -> list[float]:
    """Get BM25 scores for query tokens.

    Factored out to isolate the untyped method call.

    Args:
        bm25: A BM25Okapi instance (untyped due to lack of type stubs).
        tokens: Query tokens to score against the corpus.

    Returns:
        BM25 scores for each document in the corpus.
    """
    scores = bm25.get_scores(tokens)  # type: ignore[attr-defined]
    return [float(s) for s in scores]


class BM25Index:
    """In-memory BM25 index for a document's nodes.

    Builds a BM25Okapi index from node text using word-boundary tokenization
    (punctuation stripped). Supports search queries returning ranked
    (node_id, score) pairs.

    Example:
        >>> nodes = {"n1": node1, "n2": node2}
        >>> index = BM25Index(nodes)
        >>> results = index.search("error code E1234", top_k=10)
        >>> for node_id, score in results:
        ...     print(f"{node_id}: {score:.4f}")
    """

    def __init__(self, nodes: Mapping[str, "TreeNode"]) -> None:
        """Build BM25 index from nodes.

        Args:
            nodes: Mapping of node_id to TreeNode. Each node must have a `text`
                attribute containing the content to index.
        """
        self.node_ids: list[str] = list(nodes.keys())

        if not self.node_ids:
            # Empty corpus - BM25Okapi requires at least one document
            self._bm25: object | None = None
            return

        corpus = [self._tokenize(nodes[node_id].text) for node_id in self.node_ids]
        self._bm25 = _create_bm25_index(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search for nodes matching query terms.

        Args:
            query: Query text to search for.
            top_k: Maximum number of results to return.

        Returns:
            List of (node_id, score) tuples sorted by BM25 score descending.
            Scores are non-negative floats where higher is better.
            Returns fewer than top_k results if the corpus is smaller.
        """
        if self._bm25 is None:
            return []

        tokens = self._tokenize(query)
        scores = _get_bm25_scores(self._bm25, tokens)

        ranked = sorted(
            zip(self.node_ids, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return ranked[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into lowercase word tokens, stripping punctuation.

        Uses \\w+ regex to extract word characters, so punctuation-attached
        words like ``"Prius."`` and ``"Prius"`` produce the same token.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase word tokens.
        """
        return re.findall(r"\w+", text.lower())


class BM25IndexCache:
    """LRU cache for document BM25 indexes.

    Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index Caching

    Caches BM25Index instances per document_id with LRU eviction policy.
    When the cache exceeds max_size, the least recently used entry is evicted.

    Example:
        >>> cache = BM25IndexCache(max_size=10)
        >>> index = cache.get_or_build("doc123", nodes)
        >>> # Later, get same index from cache:
        >>> index = cache.get_or_build("doc123", nodes)  # Returns cached
    """

    def __init__(self, max_size: int = 10) -> None:
        """Initialize the cache with a maximum size.

        Args:
            max_size: Maximum number of BM25 indexes to cache. When exceeded,
                the least recently used index is evicted. Must be at least 1.

        Raises:
            ValueError: If max_size is less than 1.
        """
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")

        self._cache: OrderedDict[str, BM25Index] = OrderedDict()
        self._max_size = max_size

    def get_or_build(
        self,
        document_id: str,
        nodes: Mapping[str, "TreeNode"],
    ) -> BM25Index:
        """Get cached index or build and cache a new one.

        Args:
            document_id: Unique identifier for the document.
            nodes: Mapping of node_id to TreeNode for building the index.
                Only used if the index is not already cached.

        Returns:
            BM25Index for the document, either from cache or newly built.
        """
        if document_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(document_id)
            return self._cache[document_id]

        # Build new index
        index = BM25Index(nodes)
        self._cache[document_id] = index

        # Evict LRU entries if over capacity
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return index

    def __len__(self) -> int:
        """Return the number of cached indexes."""
        return len(self._cache)

    def __contains__(self, document_id: str) -> bool:
        """Check if a document's index is cached."""
        return document_id in self._cache

    def clear(self) -> None:
        """Remove all cached indexes."""
        self._cache.clear()


def reciprocal_rank_fusion(
    vector_ranking: Sequence[str],
    bm25_ranking: Sequence[str],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Combine two rankings using Reciprocal Rank Fusion.

    Spec: specs/bm25-hybrid-search.md § Architecture > Reciprocal Rank Fusion

    RRF assigns each item a score of 1/(k + rank + 1) from each ranking it
    appears in. Items appearing in both rankings accumulate scores from both.
    The k constant (typically 60) dampens the impact of high ranks.

    Args:
        vector_ranking: Node IDs ordered by vector similarity (best first).
        bm25_ranking: Node IDs ordered by BM25 score (best first).
        k: RRF constant controlling rank sensitivity. Default 60 is standard.

    Returns:
        List of (node_id, rrf_score) tuples sorted by score descending.
        Ties are broken by insertion order (vector ranking first).

    Example:
        >>> vector = ["a", "b", "c"]
        >>> bm25 = ["b", "a", "d"]
        >>> reciprocal_rank_fusion(vector, bm25)
        [('a', 0.0327...), ('b', 0.0327...), ('c', 0.0159...), ('d', 0.0159...)]
    """
    scores: dict[str, float] = {}

    for rank, node_id in enumerate(vector_ranking):
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank + 1)

    for rank, node_id in enumerate(bm25_ranking):
        scores[node_id] = scores.get(node_id, 0.0) + 1.0 / (k + rank + 1)

    # Sort by score descending, then by first appearance for stability
    return sorted(scores.items(), key=lambda item: -item[1])
