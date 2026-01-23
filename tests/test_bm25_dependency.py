"""Tests for BM25 dependency availability."""


def test_rank_bm25_import() -> None:
    """Verify rank_bm25 package can be imported.

    Spec: specs/bm25-hybrid-search.md § Dependencies
    Success: rank_bm25 listed in pyproject.toml, import succeeds
    """
    from rank_bm25 import BM25Okapi

    # Verify it can be instantiated with tokenized documents
    corpus = [["hello", "world"], ["foo", "bar"], ["baz", "qux"]]
    bm25 = BM25Okapi(corpus)

    # Verify get_scores API works
    scores = bm25.get_scores(["hello"])
    assert len(scores) == 3

    # Verify get_top_n API works
    top_results = bm25.get_top_n(["hello"], corpus, n=2)
    assert len(top_results) == 2
