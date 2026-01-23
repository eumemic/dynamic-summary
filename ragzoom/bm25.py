"""BM25 lexical search for hybrid retrieval.

Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index

This module provides BM25 (Okapi BM25) lexical search to complement vector
similarity search. BM25 excels at finding exact term matches like names,
IDs, error codes, and technical jargon that embeddings may miss.
"""

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

    Builds a BM25Okapi index from node text using simple whitespace tokenization.
    Supports search queries returning ranked (node_id, score) pairs.

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
        """Tokenize text using simple whitespace + lowercase.

        Args:
            text: Text to tokenize.

        Returns:
            List of lowercase tokens split on whitespace.
        """
        return text.lower().split()
