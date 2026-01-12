# RagZoom Codebase Guide

**Last Verified**: January 2026

A systematic guide for developing expertise in the RagZoom codebase. Start with the core algorithm, then expand to supporting systems.

## Module Structure

```
ragzoom/
├── greedy_tiling.py      # Core algorithm - start here
├── retrieve.py           # Query orchestration
├── assemble.py           # Output assembly
├── store.py              # Backend factory
├── config.py             # Configuration classes
├── contracts/            # Protocol definitions (interfaces)
│   ├── storage_backend.py
│   ├── tree_node.py
│   ├── vector_index.py
│   └── ...
├── backends/             # Storage implementations
│   ├── sqlite_backend.py
│   ├── postgres_backend.py
│   └── vector_index_*.py
├── services/             # Business logic services
│   ├── tree_navigator.py
│   ├── query_service.py
│   ├── llm_service.py
│   └── ...
├── retrieval/            # Query-time operations
│   ├── coverage_builder.py
│   ├── budget_planner.py
│   ├── scoring_service.py
│   ├── mmr.py
│   └── verbatim_selector.py
├── indexing/             # Document indexing
│   └── runtime.py        # IndexerRuntime
├── server/               # gRPC server components
│   ├── append_executor.py
│   ├── app.py
│   └── ...
└── document_store.py     # Per-document isolation layer
```

## Learning Path

### Phase 1: Core Algorithm

**Start with `ragzoom/greedy_tiling.py`** - the heart of RagZoom.

Key elements:
- `GreedyTilingGenerator` class
- `find_optimal_tiling_over_roots()` - main entry point
- Roll-up priority calculation: `quality_lost / tokens_saved`
- Quality metric: `relevance × token_count`

The algorithm starts with all leaves (maximum detail) and iteratively rolls up the least-valuable sibling pairs until within budget.

### Phase 2: Data Structures

**Read `ragzoom/contracts/tree_node.py`** for the TreeNode protocol.

Key concepts:
- Binary tree with character spans `(span_start, span_end)`
- Leaf nodes (height=0): raw document text
- Parent nodes (height>0): LLM-generated summaries
- Coordinates: `(height, level_index)` computed on demand

**Read `ragzoom/tiling.py`** for the Tiling dataclass - the output format.

### Phase 3: Query Flow

**Trace through `ragzoom/retrieve.py`**:

1. `Retriever` receives query and config
2. `EmbeddingService` generates query embedding
3. Vector search finds seed nodes
4. `ScoringService` computes relevance scores
5. `CoverageBuilder` expands seeds to coverage tree (ancestors + siblings)
6. `GreedyTilingGenerator` produces optimal tiling
7. `Assembler` concatenates node texts

Supporting modules in `ragzoom/retrieval/`:
- `budget_planner.py` - calculates conservative seed counts
- `coverage_builder.py` - builds coverage tree from seeds
- `scoring_service.py` - assigns relevance scores to all nodes
- `mmr.py` - Maximal Marginal Relevance for diversity
- `verbatim_selector.py` - selects recent leaves for verbatim inclusion

### Phase 4: Indexing Flow

**Trace through `ragzoom/indexing/runtime.py`**:

1. `IndexerRuntime.append_text()` receives document text
2. Text splitter creates leaf nodes (~200 tokens each)
3. `AppendExecutor` builds tree bottom-up
4. `LLMService` generates parent summaries
5. `StorageBackend` persists nodes and embeddings

Key class: `ragzoom/server/append_executor.py` - `AppendExecutor`
- Builds `TreePatch` for document changes
- Drives LLM summarization with semaphore-controlled concurrency
- Maintains tree invariants during construction

### Phase 5: Storage Layer

**The contracts pattern** (`ragzoom/contracts/`):

Protocols define interfaces; backends implement them:
- `StorageBackend` - main storage protocol
- `VectorIndex` - embedding search protocol
- `TreeNode` - node data protocol
- `NodeRepository`, `DocumentRepository` - data access

**Backend implementations** (`ragzoom/backends/`):

SQLite (development):
- `sqlite_backend.py` - SQLiteStorageBackend
- `sqlite_db.py` - database schema and queries
- `vector_index_python.py` or `vector_index_chroma.py` - embeddings

PostgreSQL (production):
- `postgres_backend.py` - PostgresStorageBackend
- `vector_index_pgvector.py` - pgvector embeddings

**Document isolation** (`ragzoom/document_store.py`):

All application code accesses storage through `DocumentStore`:
```python
store = create_store(config)
doc_store = store.for_document(document_id)
```

This enforces strict document isolation - queries only see their own document's nodes.

### Phase 6: Services Layer

**`ragzoom/services/`** contains business logic:

- `tree_navigator.py` - computes node relationships (parent, children, siblings) with caching
- `query_service.py` - high-level query orchestration
- `indexing_service.py` - high-level indexing orchestration
- `llm_service.py` - LLM API interactions
- `summarizer.py` - summary generation logic
- `cache_manager.py` - LRU caching for nodes

## Test Organization

Tests mirror the module structure with `_sqlite` suffix for integration tests:

| Module | Unit Tests | Integration Tests |
|--------|------------|-------------------|
| `greedy_tiling.py` | `test_greedy_tiling.py` | - |
| `retrieve.py` | `test_retrieval_invariants_sqlite.py` | `test_context_retrieval_sqlite.py` |
| `assemble.py` | `test_assemble.py` | - |
| Backends | - | `test_budget_guarantee_sqlite.py` |
| Indexing | `test_builders.py` | `test_append_executor_sqlite.py` |
| API | - | `test_api_*.py` |
| CLI | - | `test_cli.py` |

**Key test utilities** in `tests/`:
- `conftest.py` - fixtures for both unit and integration tests
- `test_builders.py` - `DocumentBuilder`, `TreeNodeBuilder` for test data

Run validation test:
```bash
./scripts/run-checks.sh --impacted-only ragzoom/greedy_tiling.py
```

## Key Invariants

Understanding these invariants prevents bugs:

1. **Token budget is strict** - tiling never exceeds specified budget
2. **Complete coverage** - tiling covers entire document span, no gaps
3. **No overlap** - tiling nodes have adjacent, non-overlapping spans
4. **Forest of perfect binary trees** - each tree has power-of-2 leaves
5. **Sibling adjacency** - `left.span_end == right.span_start`
6. **Document isolation** - queries only see their document's nodes

## Performance Characteristics

- **Greedy tiling**: O(n log n) via heap operations
- **Tree building**: O(n) LLM API calls
- **Vector search**: typically <100ms for thousands of nodes
- **Assembly**: O(k) where k = nodes in tiling

## Debugging Tips

**Type errors**: Run `dmypy run -- ragzoom/` for fast incremental checking

**Test a specific file's impact**:
```bash
./scripts/run-checks.sh --impacted-only ragzoom/retrieve.py --fail-fast
```

**Trace query flow**: Add logging in `Retriever.retrieve_async()` to see:
- Seed selection
- Coverage tree construction
- Tiling decisions

**Verify tree structure**: Use `ragzoom validate <document_id>` CLI command
