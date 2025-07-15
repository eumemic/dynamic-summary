# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RagZoom is an incremental, hierarchical RAG (Retrieval-Augmented Generation) memory system that creates dynamic summaries with intelligent resolution control. It builds a binary tree structure from documents where leaf nodes contain original text chunks and internal nodes contain AI-generated summaries. During retrieval, it "zooms in" on relevant content while maintaining global context.

## Key Commands

### Development
```bash
# Run all tests (takes ~4.5 seconds)
pytest tests/ -v

# Run specific test files based on what you're working on
pytest tests/test_splitter.py      # After modifying splitter.py
pytest tests/test_store.py         # After modifying store.py  
pytest tests/test_integration.py   # After modifying index.py, retrieve.py, or assemble.py
pytest tests/test_concurrency.py   # After modifying api.py

# Quick test runner with pattern matching
./test_quick.sh              # Run all tests
./test_quick.sh splitter     # Run tests matching 'splitter'

# Linting and formatting
ruff check ragzoom/ tests/   # Check code style
black ragzoom/ tests/        # Format code
mypy ragzoom/               # Type checking

# Index documents
ragzoom index <file> --max-concurrent 10  # Default parallelism
ragzoom index <file> --max-concurrent 50  # Higher parallelism for large docs

# Start API server
ragzoom serve
```

### Git Hooks
- **pre-commit**: Runs relevant tests for modified files (~1-2 seconds)
- **pre-push**: Runs full test suite before pushing (~4.5 seconds)

## Architecture

### Core Flow
1. **Indexing**: Documents → TextSplitter → Leaf nodes → TreeBuilder → Binary tree with summaries
2. **Retrieval**: Query → Embedding → Vector search → MMR diversity → Coverage map → Frontier extraction
3. **Assembly**: Frontier nodes → Slope capping → Token budget → Final summary

### Key Components

**Storage Layer (`store.py`)**
- SQLite for tree structure (nodes, relationships, metadata)
- ChromaDB for vector embeddings
- LRU cache for frequently accessed nodes
- Node states: normal, dirty (needs re-summarization), pinned

**Tree Building (`index.py`)**  
- Async implementation using AsyncOpenAI
- Concurrent API calls controlled by semaphore (default: 10)
- Global progress tracking across leaf creation and tree building
- Automatic parent summarization when both children exist
- Dirty node recomputation for incremental updates

**Retrieval (`retrieve.py`)**
- MMR (Maximal Marginal Relevance) for diversity
- Coverage map propagation (selected nodes + ancestors)
- Frontier extraction (covered nodes with uncovered children)
- Optional sliding queue eviction with freshness decay
- Budget guarantee modes:
  - Budget-only: Conservative n_max calculation to prevent overflow
  - Budget + n_max: Intelligent node dropping to respect both constraints
  - n_max-only: Traditional retrieval without budget enforcement

**Assembly (`assemble.py`)**
- Slope capping: ±1 depth transitions for coherence
- Token budget enforcement with drop vs truncate strategies
- Optional smoothing pass with transition markers
- Lazy refresh of dirty nodes to maintain summary consistency

### Configuration

Key settings in `RagZoomConfig`:
- `budget_tokens`: Maximum tokens for final summary (default: 8000)
- `budget_strategy`: Budget enforcement strategy: "drop" or "truncate" (default: "drop")
- `leaf_tokens`: Target size for leaf chunks (default: 200)
- `slope_cap_size`: Maximum depth difference between adjacent frontier nodes (default: 1)
- `mmr_lambda`: Relevance vs diversity trade-off (default: 0.7)
- `embedding_model`: Default "text-embedding-3-small" 
- `summary_model`: Default "gpt-4o" for high-quality summaries
- `max_concurrent`: Parallelism for API calls (CLI flag, default: 10)

### Testing Strategy

**Test Coverage Map**:
- `test_splitter.py` → `splitter.py` (unit tests)
- `test_store.py` → `store.py` (unit tests)
- `test_integration.py` → `index.py`, `retrieve.py`, `assemble.py` (integration)
- `test_concurrency.py` → `api.py` (thread safety, FastAPI)
- `test_cli.py` → `cli.py` (CLI commands and options)
- `test_progress.py` → `progress.py` (progress tracking)
- `test_utils.py` → `utils.py` (utility functions)
- `test_dirty_refresh.py` → dirty node refresh functionality (async refresh, retrieval integration)
- `test_budget_guarantee.py` → budget constraint enforcement (worst-case bounds, strategies)

**Coverage Status**: All core modules now have test coverage

### Performance Considerations

1. **Batch Embeddings**: Process up to 100 texts per API call (OpenAI limit: 2048)
2. **Concurrent Summarization**: Async processing with configurable parallelism
3. **Progress Tracking**: Unified progress bar showing both leaf and tree operations
4. **Cache Strategy**: LRU cache (1000 nodes) for hot path optimization

### Common Patterns

**Adding New Features**:
1. Update `RagZoomConfig` if configuration needed
2. Implement core logic in appropriate module
3. Add CLI command in `cli.py`
4. Add API endpoint in `api.py` if REST access needed
5. Write tests - unit tests for isolated logic, integration tests for cross-module features

**Debugging Tips**:
- Check `~/.ragzoom/ragzoom.log` for detailed logs
- Use `--no-progress` flag to see raw output without progress bars
- SQLite DB at `ragzoom.db`, ChromaDB at `chroma_db/`

### Critical Implementation Details

1. **Async vs Sync**: The codebase uses AsyncOpenAI for indexing (tree building) but regular OpenAI for retrieval/assembly. This is intentional - indexing benefits from high concurrency while retrieval is typically single-threaded.

2. **Node IDs**: Format is `{depth}_{span_start}_{span_end}_{hash[:8]}` for uniqueness

3. **Tree Structure**: Left-balanced binary tree. Parents created only when both children exist. Tree may be ragged during incremental updates.

4. **Token Counting**: Uses tiktoken with cl100k_base encoding throughout

5. **Error Handling**: Store operations use database transactions. API calls have retry logic with exponential backoff.

## Recent Changes

- Unified async implementation for tree building (removed sync version)
- Added global progress tracking across all indexing operations  
- Fixed priority score clamping for eviction (must be in [0,1] range)
- Improved test mocking for AsyncOpenAI client
- Added comprehensive git hooks for testing
- Added complete test coverage for CLI and progress modules
- Fixed <<<MID>>> delimiter missing in summaries with retry logic (up to 3 attempts)
- Fixed parent-child frontier deduplication to work with <<<MID>>> extraction logic
- Fixed chunk size configuration to use tokens directly (was creating 775-token chunks instead of 200)
- Added comprehensive assembly integration tests
- Simplified pre-commit hook to use test_quick.sh script
- **Fixed budget guarantee calculations**: Updated to use dynamic (slope_cap_size + 2) multiplier for mathematically correct worst-case bounds
- **Added drop vs truncate budget strategies**: Intelligent node dropping preserves coherence vs tail truncation
- **Fixed cache invalidation bug**: Added existence check before removing nodes from cache_order deque
- **Implemented dirty node refresh**: Full async implementation with TreeBuilder.refresh_nodes_async() for re-summarizing stale nodes
- **Enhanced slope cap handling**: Re-apply slope cap after budget trimming to prevent "bridge node" violations
- **Fixed post-slope-cap budget overflow**: Added second budget check and trim after slope cap re-application
- **Added embedding dimension validation**: Validates embeddings match expected dimensions, preventing runtime errors
- **Fixed cache eviction after refresh**: Ensures refreshed nodes are properly re-added to LRU cache
- **Added empty frontier guard**: Falls back to root node when budget trimming leaves empty frontier
- **Implemented async retrieval**: Added retrieve_async() with proper sync wrappers for FastAPI compatibility

## Development Practices

- Always write regression tests when regressions are discovered
- Group related changes into single commits that leave the app in a working state
- Run tests for modified components before committing