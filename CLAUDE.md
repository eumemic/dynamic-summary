# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@docs/project-brief.md
@docs/implementation-notes.md
@docs/testing-strategy.md
@docs/bug-investigations.md

## Project Overview

RagZoom is an incremental, hierarchical RAG (Retrieval-Augmented Generation) memory system that creates dynamic summaries with intelligent resolution control. It builds a binary tree structure from documents where leaf nodes contain original text chunks and internal nodes contain AI-generated summaries. During retrieval, it "zooms in" on relevant content while maintaining global context.

## Key Commands

### Development
```bash
# Run fast tests (excludes @slow and @integration)
pytest tests/ -m "not slow and not integration" -n 8

# Run all tests including slow/integration
pytest tests/ -n 8

# Run with real store for all tests
pytest tests/ --use-real-store -n 8

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
ragzoom index <file>                      # Uses filename as document ID
ragzoom index <file> --document-id my-doc # Custom document ID
ragzoom index <file> --clear              # Clear existing document first
ragzoom index <file> --max-concurrent 10  # Default parallelism
ragzoom index <file> --max-concurrent 50  # Higher parallelism for large docs
ragzoom index <file> --validate           # Enable validation checks

# Query documents (document ID is REQUIRED)
ragzoom query "search text" -d <doc-id>              # Query specific document
ragzoom query "search text" -d <doc-id> --validate   # With validation checks

# Document management
ragzoom documents                         # List all indexed documents
ragzoom clear -d <doc-id> --confirm      # Clear specific document
ragzoom clear --confirm                   # Clear all documents

# Start API server
ragzoom serve
```

### Git Hooks
- **pre-commit**: Runs fast tests + linting + type checking (~8 seconds with 8 workers)
- Excludes @slow and @integration tests for speed
- Uses `pytest tests/ -m "not slow and not integration" -n 8`

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

**Test Performance**: Full test suite optimized for speed with mock storage layer
- **Fast tests**: 137 tests in ~8.5 seconds with 8 parallel workers
- **Integration tests**: 3 tests using real SQLite + ChromaDB (marked @pytest.mark.integration)
- **Slow tests**: 3 tests taking >5 seconds (marked @pytest.mark.slow)

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
- `test_validate.py` → `validate.py` (validation functions)
- `test_indexing_fast.py` → fast versions of indexing tests using mock store
- `test_incomplete_indexing.py` → slow integration tests for indexing edge cases

**Mock Store**: `tests/mock_store.py` provides SimpleMockStore for 4.5x faster unit tests
- In-memory tree structure and state management
- Compatible with all Store methods used in tests
- Automatic selection via pytest fixtures (mock by default, real store for @integration)

### Validation Features

**Validation System**: Use `--validate` flag on index/query commands to enable comprehensive validation checks:
- Early validation during indexing (chunk sizes, document coverage, tree structure)
- Frontier validation during querying (completeness, no overlaps, ordering)
- Whitespace-only gaps are allowed (text splitter limitation)
- Fast-fail behavior with exit code 1 on validation errors

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
- **Fixed ChromaDB test configuration**: Tests now use tempfile.TemporaryDirectory() instead of ":memory:" which ChromaDB doesn't support
- **Fixed token budget allocation**: Removed depth-based compression that artificially limited higher-level nodes to as few as 50 tokens; all nodes now get consistent RAGZOOM_LEAF_TOKENS budget, with LLM instructed via prompt rather than hard API limits
- **Added --validate flag**: Comprehensive validation for indexing (document coverage, chunk sizes, tree structure) and retrieval (frontier completeness, no overlaps) to ensure correctness
- **Removed chunk overlap**: Set chunk_overlap=0 in text splitter since RagZoom requires sequential non-overlapping chunks for correct span calculation
- **Fixed whitespace gaps**: Implemented comprehensive gap reconstruction that appends ALL gaps to previous chunks, ensuring complete coverage with no character loss (Issue 10)
- **Removed leaf_overlap_tokens parameter**: Simplified codebase by removing unused overlap handling since chunks are now guaranteed to be contiguous with no gaps
- **Implemented document isolation**: Complete namespace separation between indexed documents
  - Queries now require document_id parameter to prevent cross-document contamination
  - Filename used as default document_id when indexing files
  - Added `documents` command to list all indexed documents
  - Added `--document-id` parameter to `clear` command for targeted deletion
  - Added `--clear` flag to `index` command for atomic re-indexing
  - Updated API endpoints to require document_id in query requests

## Development Practices

**Type Safety**:
- **ALWAYS write type annotations** for all new functions and methods
- Include parameter types and return types: `def func(x: str, y: int) -> bool:`
- Use `from typing import` imports for complex types: `List`, `Dict`, `Optional`, `Union`, etc.
- Type checking runs in pre-commit hook and will warn about missing annotations
- For SQLAlchemy ORM code, focus on business logic types rather than Column types
- When in doubt, `Any` is better than no annotation, but prefer specific types

**Testing & Commits**:
- Always write regression tests when regressions are discovered
- Group related changes into single commits that leave the app in a working state
- Run tests for modified components before committing