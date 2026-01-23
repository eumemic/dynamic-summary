---
status: READY
---

# BM25 Hybrid Search (Contextual Retrieval)

## Overview

Add BM25 lexical search alongside existing vector search, combining results via Reciprocal Rank Fusion (RRF). This implements the hybrid retrieval component of Anthropic's [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) approach.

## Motivation

Vector embeddings excel at semantic similarity but can miss exact term matches - names, IDs, error codes, acronyms. BM25 lexical search catches these exact matches. Combined via rank fusion, the hybrid approach reduces retrieval failures by ~49% compared to embeddings alone (per Anthropic's research).

## Goals

1. **Hybrid retrieval** - Combine BM25 and vector search for better recall
2. **Zero persistence overhead** - BM25 index built on-demand in memory
3. **Transparent integration** - Works with existing retrieval pipeline

## Non-Goals

- Contextual chunk preprocessing (RagZoom's hierarchical summaries provide context)
- Persisted BM25 indexes
- External search engine integration
- Re-ranking with cross-encoder models (future work)

## Architecture

### BM25 Index

Built on-demand per document from node text:

```python
class BM25Index:
    """In-memory BM25 index for a document's nodes."""

    def __init__(self, nodes: dict[str, Node]):
        self.node_ids = list(nodes.keys())
        corpus = [self._tokenize(n.text) for n in nodes.values()]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return (node_id, score) pairs sorted by BM25 score."""
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Return top_k by score
        ranked = sorted(zip(self.node_ids, scores), key=lambda x: -x[1])
        return ranked[:top_k]

    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()
```

### Reciprocal Rank Fusion

Combine vector and BM25 rankings:

```python
def reciprocal_rank_fusion(
    vector_ranking: list[str],  # node_ids ordered by vector similarity
    bm25_ranking: list[str],    # node_ids ordered by BM25 score
    k: int = 60,                # RRF constant (standard value)
) -> list[tuple[str, float]]:
    """Combine rankings using RRF. Returns (node_id, rrf_score) pairs."""
    scores: dict[str, float] = {}

    for rank, node_id in enumerate(vector_ranking):
        scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank + 1)

    for rank, node_id in enumerate(bm25_ranking):
        scores[node_id] = scores.get(node_id, 0) + 1 / (k + rank + 1)

    return sorted(scores.items(), key=lambda x: -x[1])
```

### Integration with Retriever

Modify `Retriever.retrieve_async()`:

```python
async def retrieve_async(
    self,
    query: str,
    *,
    num_seeds: int | None = None,
    budget_tokens: int | None = None,
    document_id: str,
    use_bm25: bool = True,  # NEW: enabled by default
    **kwargs,
) -> RetrievalResult:
    # 1. Vector search (existing)
    vector_results = await self._vector_search(query, document_id, top_k=50)

    # 2. BM25 search (NEW)
    if use_bm25:
        bm25_index = self._get_or_build_bm25_index(document_id)
        bm25_results = bm25_index.search(query, top_k=50)

        # 3. Fuse rankings
        fused = reciprocal_rank_fusion(
            [r.node_id for r in vector_results],
            [node_id for node_id, _ in bm25_results],
        )
        seed_candidates = [node_id for node_id, _ in fused[:num_seeds * 2]]
    else:
        seed_candidates = [r.node_id for r in vector_results]

    # 4. Continue with existing seed selection, scoring, tiling...
```

### BM25 Index Caching

Cache BM25 indexes per document in memory with LRU eviction:

```python
class BM25IndexCache:
    """LRU cache for document BM25 indexes."""

    def __init__(self, max_size: int = 10):
        self._cache: OrderedDict[str, BM25Index] = OrderedDict()
        self._max_size = max_size

    def get_or_build(
        self,
        document_id: str,
        nodes: dict[str, Node],
    ) -> BM25Index:
        if document_id in self._cache:
            self._cache.move_to_end(document_id)
            return self._cache[document_id]

        index = BM25Index(nodes)
        self._cache[document_id] = index

        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return index
```

## Configuration

### QueryConfig Field

```python
@dataclass
class QueryConfig:
    # ... existing fields ...

    use_bm25: bool = True
    """Enable BM25 hybrid search. Default True."""

    bm25_weight: float = 1.0
    """Weight for BM25 in RRF. 1.0 = equal weight with vector."""
```

### CLI Flag

```bash
# Disable BM25 for pure vector search
ragzoom query "find the error" -d logs.txt --no-bm25
```

## Dependencies

Add `rank_bm25` package:

```toml
# pyproject.toml
dependencies = [
    # ... existing ...
    "rank_bm25>=0.2.2",
]
```

## Performance Considerations

### Memory

BM25 index size is roughly proportional to unique terms × documents. For a 100K token document with ~20K unique terms, expect ~5-10MB per document index.

### Latency

- Index build: ~10-50ms for typical documents
- BM25 search: <5ms
- Overall retrieval impact: minimal (parallelizable with vector search)

### When BM25 Helps Most

- Exact term queries: "error code E1234"
- Names and identifiers: "John Smith", "PR-4567"
- Technical jargon: acronyms, function names
- Rare words: domain-specific terminology

### When BM25 Helps Least

- Conceptual queries: "what caused the problem"
- Paraphrased content: query uses different words than document
- Short documents: limited term diversity

## Testing

### Unit Tests

- BM25 index builds correctly from nodes
- RRF produces expected fusion ordering
- Cache eviction works correctly
- `--no-bm25` flag disables BM25

### Integration Tests

- Exact term query finds node containing term
- Hybrid beats pure vector on keyword queries
- Hybrid doesn't hurt semantic queries

### Benchmarks

Compare retrieval quality:
1. Vector only (baseline)
2. BM25 only
3. Hybrid (vector + BM25 + RRF)

Metrics: recall@k, MRR, success rate on test queries.

## Rollout

1. Add `rank_bm25` dependency
2. Implement BM25Index and cache
3. Implement RRF fusion
4. Integrate into Retriever (behind feature flag)
5. Enable by default after validation
