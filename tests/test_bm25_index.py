"""Tests for BM25Index class.

Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
"""

from dataclasses import dataclass

from ragzoom.bm25 import BM25Index


@dataclass
class MockNode:
    """Minimal TreeNode-compatible mock for testing BM25Index.

    BM25Index only uses the text field, but we must implement the full
    TreeNode protocol for type compatibility.
    """

    id: str
    text: str
    document_id: str | None = None
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None
    span_start: int = 0
    span_end: int = 0
    token_count: int = 0
    height: int = 0
    is_pinned: bool | int = False
    preceding_neighbor_id: str | None = None
    following_neighbor_id: str | None = None
    level_index: int = 0
    preceding_context: str | None = None
    preceding_context_summary: str | None = None
    embedding: bytes | None = None
    time_start: float | None = None
    time_end: float | None = None

    def is_leaf(self) -> bool:
        return self.height == 0

    def is_root(self) -> bool:
        return self.parent_id is None

    def get_depth(self) -> int:
        return 0


class TestBM25Index:
    """Tests for BM25Index build and search functionality."""

    def test_build_from_nodes(self) -> None:
        """Test that BM25Index builds correctly from nodes dict.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
        Success: BM25Index can be instantiated from a dict of nodes
        """

        nodes = {
            "node1": MockNode(id="node1", text="hello world"),
            "node2": MockNode(id="node2", text="foo bar baz"),
            "node3": MockNode(id="node3", text="hello foo"),
        }

        index = BM25Index(nodes)

        # Verify index was built with correct node count
        assert len(index.node_ids) == 3
        assert set(index.node_ids) == {"node1", "node2", "node3"}

    def test_search_returns_ranked_results(self) -> None:
        """Test that search returns (node_id, score) pairs sorted by score.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
        Success: search(query, top_k) returns ranked (node_id, score) pairs
        """

        nodes = {
            "node1": MockNode(id="node1", text="hello world"),
            "node2": MockNode(id="node2", text="foo bar baz"),
            "node3": MockNode(id="node3", text="hello foo"),
        }

        index = BM25Index(nodes)
        results = index.search("hello", top_k=3)

        # Should return list of (node_id, score) tuples
        assert len(results) == 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], str) and isinstance(r[1], float) for r in results)

        # Results should be sorted by score descending
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

        # "hello" appears in node1 and node3, so they should score higher than node2
        node_ids = [r[0] for r in results]
        assert node_ids[0] in {"node1", "node3"}
        assert node_ids[1] in {"node1", "node3"}
        assert node_ids[2] == "node2"

    def test_search_top_k_limit(self) -> None:
        """Test that search respects top_k limit.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
        Success: search returns at most top_k results
        """

        nodes = {
            f"node{i}": MockNode(id=f"node{i}", text=f"word{i} common")
            for i in range(10)
        }

        index = BM25Index(nodes)
        results = index.search("common", top_k=3)

        assert len(results) == 3

    def test_search_returns_fewer_if_less_nodes(self) -> None:
        """Test that search returns all nodes if fewer than top_k exist.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
        Success: search returns fewer than top_k if corpus is smaller
        """

        nodes = {
            "node1": MockNode(id="node1", text="hello world"),
            "node2": MockNode(id="node2", text="foo bar"),
        }

        index = BM25Index(nodes)
        results = index.search("hello", top_k=10)

        assert len(results) == 2

    def test_tokenization_lowercase(self) -> None:
        """Test that tokenization is case-insensitive.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
        Success: Queries match regardless of case
        """

        nodes = {
            "node1": MockNode(id="node1", text="HELLO World"),
            "node2": MockNode(id="node2", text="foo bar"),
        }

        index = BM25Index(nodes)

        # Both "hello", "HELLO", and "Hello" should match
        results_lower = index.search("hello", top_k=2)
        results_upper = index.search("HELLO", top_k=2)
        results_mixed = index.search("Hello", top_k=2)

        # All queries should rank node1 first
        assert results_lower[0][0] == "node1"
        assert results_upper[0][0] == "node1"
        assert results_mixed[0][0] == "node1"

    def test_tokenization_whitespace_split(self) -> None:
        """Test that tokenization splits on whitespace.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
        Success: Simple whitespace + lowercase tokenization
        """

        # Use larger corpus to avoid BM25 IDF edge cases with tiny corpora
        # (BM25 IDF can go negative when a term appears in most documents)
        nodes = {
            "node1": MockNode(id="node1", text="one two three"),
            "node2": MockNode(id="node2", text="onetwo three"),
            "node3": MockNode(id="node3", text="foo bar baz"),
            "node4": MockNode(id="node4", text="qux quux corge"),
        }

        index = BM25Index(nodes)

        # "two" should only match node1 (whitespace-separated)
        results = index.search("two", top_k=4)
        assert results[0][0] == "node1"
        assert results[0][1] > 0  # node1 should have positive score
        # All other nodes should have zero score (no match)
        assert all(r[1] == 0.0 for r in results[1:])

    def test_empty_query_returns_zero_scores(self) -> None:
        """Test that empty query returns zero scores.

        Spec: N/A (edge case handling)
        Success: Empty query returns results with zero scores
        """

        nodes = {
            "node1": MockNode(id="node1", text="hello world"),
            "node2": MockNode(id="node2", text="foo bar"),
        }

        index = BM25Index(nodes)
        results = index.search("", top_k=2)

        # Should still return results, but all with zero score
        assert len(results) == 2
        assert all(r[1] == 0.0 for r in results)

    def test_empty_nodes_dict(self) -> None:
        """Test that BM25Index handles empty nodes dict.

        Spec: N/A (edge case handling)
        Success: Empty nodes dict creates valid index with no results
        """

        nodes: dict[str, MockNode] = {}

        index = BM25Index(nodes)
        results = index.search("hello", top_k=3)

        assert len(results) == 0

    def test_query_no_matches(self) -> None:
        """Test search with query that matches nothing.

        Spec: N/A (edge case handling)
        Success: Query with no matches returns results with zero scores
        """

        nodes = {
            "node1": MockNode(id="node1", text="hello world"),
            "node2": MockNode(id="node2", text="foo bar"),
        }

        index = BM25Index(nodes)
        results = index.search("nonexistent xyz", top_k=3)

        # Should return all nodes but with zero scores
        assert len(results) == 2
        assert all(r[1] == 0.0 for r in results)

    def test_multiword_query(self) -> None:
        """Test search with multi-word query.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
        Success: Multi-word queries score based on combined term matches
        """

        nodes = {
            "node1": MockNode(id="node1", text="hello world goodbye"),
            "node2": MockNode(id="node2", text="hello there"),
            "node3": MockNode(id="node3", text="goodbye friend"),
        }

        index = BM25Index(nodes)
        results = index.search("hello goodbye", top_k=3)

        # node1 matches both "hello" and "goodbye", should rank highest
        assert results[0][0] == "node1"

    def test_term_frequency_affects_score(self) -> None:
        """Test that term frequency affects BM25 score.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index
        Success: Documents with more occurrences of query term score higher
        """

        # Use larger corpus to avoid BM25 IDF edge cases with tiny corpora
        # (BM25 IDF can go negative when a term appears in most documents)
        nodes = {
            "node1": MockNode(id="node1", text="hello hello hello"),
            "node2": MockNode(id="node2", text="hello world"),
            "node3": MockNode(id="node3", text="world world world"),
            "node4": MockNode(id="node4", text="foo bar baz"),
            "node5": MockNode(id="node5", text="qux quux corge"),
        }

        index = BM25Index(nodes)
        results = index.search("hello", top_k=5)

        # node1 has "hello" 3 times, should score highest
        # node2 has "hello" 1 time, should score second
        # Other nodes have no "hello", should score zero
        assert results[0][0] == "node1"
        assert results[1][0] == "node2"
        assert results[0][1] > results[1][1] > 0  # Both positive, node1 higher
        # Remaining nodes have zero scores
        assert all(r[1] == 0.0 for r in results[2:])


class TestBM25IndexCache:
    """Tests for BM25IndexCache LRU eviction functionality.

    Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index Caching
    """

    def test_get_or_build_creates_new_index(self) -> None:
        """Test that get_or_build creates index when not cached.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index Caching
        Success: get_or_build returns a BM25Index for new document_id
        """
        from ragzoom.bm25 import BM25IndexCache

        cache = BM25IndexCache(max_size=10)
        nodes = {
            "node1": MockNode(id="node1", text="hello world"),
            "node2": MockNode(id="node2", text="foo bar"),
        }

        index = cache.get_or_build("doc1", nodes)

        assert isinstance(index, BM25Index)
        # Verify the index works
        results = index.search("hello", top_k=2)
        assert results[0][0] == "node1"

    def test_get_or_build_returns_cached_index(self) -> None:
        """Test that get_or_build returns same index on subsequent calls.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index Caching
        Success: Same BM25Index instance returned for same document_id
        """
        from ragzoom.bm25 import BM25IndexCache

        cache = BM25IndexCache(max_size=10)
        nodes = {
            "node1": MockNode(id="node1", text="hello world"),
        }

        index1 = cache.get_or_build("doc1", nodes)
        index2 = cache.get_or_build("doc1", nodes)

        assert index1 is index2  # Same instance, not rebuilt

    def test_lru_eviction_on_overflow(self) -> None:
        """Test that LRU entries are evicted when cache exceeds max_size.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index Caching
        Success: Cache stores up to N indexes, evicts LRU on overflow
        """
        from ragzoom.bm25 import BM25IndexCache

        cache = BM25IndexCache(max_size=3)

        # Add 3 documents to fill cache
        for i in range(3):
            nodes = {f"node{i}": MockNode(id=f"node{i}", text=f"text{i}")}
            cache.get_or_build(f"doc{i}", nodes)

        # Verify all 3 are cached
        assert len(cache) == 3

        # Add a 4th document - should evict doc0 (LRU)
        nodes4 = {"node4": MockNode(id="node4", text="text4")}
        cache.get_or_build("doc3", nodes4)

        # Cache should still be at max_size
        assert len(cache) == 3

        # doc0 should be evicted, doc1, doc2, doc3 should remain
        assert "doc0" not in cache
        assert "doc1" in cache
        assert "doc2" in cache
        assert "doc3" in cache

    def test_access_updates_lru_order(self) -> None:
        """Test that accessing a cached entry moves it to most recently used.

        Spec: specs/bm25-hybrid-search.md § Architecture > BM25 Index Caching
        Success: Accessed entries moved to end, not evicted next
        """
        from ragzoom.bm25 import BM25IndexCache

        cache = BM25IndexCache(max_size=3)

        # Add 3 documents: doc0, doc1, doc2
        for i in range(3):
            nodes = {f"node{i}": MockNode(id=f"node{i}", text=f"text{i}")}
            cache.get_or_build(f"doc{i}", nodes)

        # Access doc0 to make it most recently used
        nodes0 = {"node0": MockNode(id="node0", text="text0")}
        cache.get_or_build("doc0", nodes0)

        # Add doc3 - should evict doc1 (now LRU), not doc0
        nodes3 = {"node3": MockNode(id="node3", text="text3")}
        cache.get_or_build("doc3", nodes3)

        # doc0 should still be cached (accessed recently)
        # doc1 should be evicted (oldest untouched)
        assert "doc0" in cache
        assert "doc1" not in cache
        assert "doc2" in cache
        assert "doc3" in cache

    def test_cache_with_max_size_one(self) -> None:
        """Test cache behavior with max_size=1 (extreme case).

        Spec: N/A (edge case handling)
        Success: Only one index cached at a time
        """
        from ragzoom.bm25 import BM25IndexCache

        cache = BM25IndexCache(max_size=1)

        nodes1 = {"node1": MockNode(id="node1", text="text1")}
        cache.get_or_build("doc1", nodes1)
        assert len(cache) == 1

        nodes2 = {"node2": MockNode(id="node2", text="text2")}
        cache.get_or_build("doc2", nodes2)

        # doc1 should be evicted
        assert len(cache) == 1
        assert "doc1" not in cache
        assert "doc2" in cache

    def test_clear_empties_cache(self) -> None:
        """Test that clear() removes all cached indexes.

        Spec: N/A (utility method)
        Success: clear() empties the cache
        """
        from ragzoom.bm25 import BM25IndexCache

        cache = BM25IndexCache(max_size=10)

        for i in range(5):
            nodes = {f"node{i}": MockNode(id=f"node{i}", text=f"text{i}")}
            cache.get_or_build(f"doc{i}", nodes)

        assert len(cache) == 5

        cache.clear()

        assert len(cache) == 0

    def test_max_size_validation(self) -> None:
        """Test that max_size must be at least 1.

        Spec: N/A (input validation)
        Success: ValueError raised for invalid max_size values
        """
        import pytest

        from ragzoom.bm25 import BM25IndexCache

        # Zero is invalid
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            BM25IndexCache(max_size=0)

        # Negative is invalid
        with pytest.raises(ValueError, match="max_size must be at least 1"):
            BM25IndexCache(max_size=-1)
