# RagZoom System Architecture

**Last Verified**: January 2026

High-level overview of the RagZoom system, its core components, and data flow.

## 1. Core Concepts

### 1.1. The Node Tree

The central data structure is a binary tree of **Nodes**.

- **Leaf Nodes (height=0)**: Raw text chunks from the source document (~200 tokens each)
- **Parent Nodes (height>0)**: LLM-generated summaries of their two children
- **Spans**: Every node has `(span_start, span_end)` character offsets in the original document
- **Coordinates**: `(height, level_index)` computed on demand via `TreeNavigator` service

### 1.2. The Tiling

A **Tiling** is the retrieval output - a "correct-by-construction" list of node IDs that:
1. Covers the entire document span without gaps or overlaps
2. Maintains chronological order by span
3. Adheres to the specified token budget

The greedy algorithm chooses between parent nodes (summaries) and children (detail) to maximize relevance within budget.

### 1.3. Tree Invariants

RagZoom maintains a **forest of perfect binary trees**:

- **Perfect binary structure**: Every internal node has exactly two children
- **Forest model**: Non-power-of-2 leaves become multiple trees (e.g., 5 leaves = trees of 4+1)
- **Sibling adjacency**: `left.span_end == right.span_start` (no gaps)
- **Span coverage**: Parent span = union of children spans

## 2. System Components

### 2.1. Core Modules

| Module | Location | Purpose |
|--------|----------|---------|
| `GreedyTilingGenerator` | `ragzoom/greedy_tiling.py` | Core algorithm - rolls up leaves to fit budget |
| `Retriever` | `ragzoom/retrieve.py` | Query orchestration - seeds → coverage → tiling |
| `Assembler` | `ragzoom/assemble.py` | Concatenates tiling nodes into final output |
| `IndexerRuntime` | `ragzoom/indexing/runtime.py` | Document indexing orchestration |
| `AppendExecutor` | `ragzoom/server/append_executor.py` | Tree construction and LLM summarization |

### 2.2. Storage Layer

**Contracts** (`ragzoom/contracts/`):
- `StorageBackend` - main storage protocol
- `VectorIndex` - embedding search protocol
- `TreeNode` - node data protocol

**Backends** (`ragzoom/backends/`):
- `SQLiteStorageBackend` - development (SQLite + Chroma/Python vectors)
- `PostgresStorageBackend` - production (PostgreSQL + pgvector)

**Document isolation** (`ragzoom/document_store.py`):
```python
store = create_store(config)
doc_store = store.for_document(document_id)
```

### 2.3. Services Layer

| Service | Location | Purpose |
|---------|----------|---------|
| `TreeNavigator` | `services/tree_navigator.py` | Node relationships with caching |
| `QueryService` | `services/query_service.py` | High-level query orchestration |
| `LLMService` | `services/llm_service.py` | LLM API interactions |
| `CacheManager` | `services/cache_manager.py` | LRU caching for nodes |

### 2.4. Retrieval Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BudgetPlanner` | `retrieval/budget_planner.py` | Calculates conservative seed counts |
| `CoverageBuilder` | `retrieval/coverage_builder.py` | Expands seeds to coverage tree |
| `ScoringService` | `retrieval/scoring_service.py` | Assigns relevance scores |
| `VerbatimSelector` | `retrieval/verbatim_selector.py` | Selects recent leaves for verbatim |
| `MMR` | `retrieval/mmr.py` | Maximal Marginal Relevance diversity |

## 3. Data Flow

### 3.1. Indexing Flow

```
Source Text
    ↓
IndexerRuntime.append_text()
    ↓
Text Splitter → Leaf Nodes (~200 tokens each)
    ↓
AppendExecutor.build_tree()
    ↓
LLMService → Parent Summaries (bottom-up)
    ↓
StorageBackend.persist()
    ↓
Nodes + Embeddings stored
```

### 3.2. Query Flow

```
Query + Config
    ↓
EmbeddingService → Query Embedding
    ↓
VectorIndex.search() → Seed Nodes
    ↓
ScoringService → Relevance Scores
    ↓
CoverageBuilder → Coverage Tree (seeds + ancestors + siblings)
    ↓
GreedyTilingGenerator → Optimal Tiling
    ↓
Assembler → Final Summary Text
```

## 4. Design Principles

### 4.1. Correct-by-Construction

The greedy algorithm produces valid tilings in a single pass - no multi-stage correction needed.

### 4.2. Character-Based Spans

All spans use character coordinates (not tokens) for stability and exact source mapping.

### 4.3. Async Indexing, Sync Retrieval

- **Indexing**: AsyncOpenAI with semaphore-controlled concurrency (default: 10)
- **Retrieval**: Synchronous for simplicity (queries are single-threaded)

### 4.4. Document Isolation

Each document is completely isolated. Queries require explicit document IDs - no cross-document contamination.

## 5. Configuration

Key parameters in `IndexConfig`, `QueryConfig`, `OperationalConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_chunk_tokens` | 200 | Leaf node size |
| `budget_tokens` | None | Max tokens in output (None = no limit) |
| `mmr_lambda` | 0.7 | Relevance vs diversity (0-1) |
| `recent_verbatim_token_budget` | 0 | Verbatim recent content |

## 6. Performance

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Greedy tiling | O(n log n) | <10ms |
| Tree building | O(n) API calls | Depends on doc size |
| Vector search | O(log n) | <100ms |
| Assembly | O(k) | <1ms |

## 7. See Also

- **`references/codebase-guide.md`** - Detailed learning path for the codebase
- **`references/tiling-algorithm.md`** - Deep dive into the greedy algorithm
