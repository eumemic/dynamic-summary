# Dynamic Depth Refactoring - Issues to Fix

This document lists all issues discovered during the refactoring to make tree depth a dynamically calculated property.

## Critical Issues (Blocking)

### 1. Duplicate Method Definitions in `tests/mock_store.py`
- **Problem**: `get_node_depth()` and `get_node_height()` are defined twice (lines 191/279 and 209/295)
- **Impact**: Python syntax error - code won't run
- **Fix**: Remove duplicate definitions

### 2. Tree Visualization Loads Entire Document
- **Problem**: `build_ascii_tree()` calls `store.get_all_nodes(document_id)` instead of using the coverage map
- **Impact**: Massive performance waste - loads ALL nodes when only ~10-20 are needed
- **Fix**: Use coverage map to load only relevant nodes:
  ```python
  if coverage_map:
      all_nodes = [store.get_node(node_id) for node_id in coverage_map]
  else:
      all_nodes = store.get_all_nodes(document_id)
  ```

## High Priority Performance Issues

### 3. No Caching for Calculated Depth/Height
- **Problem**: `get_node_depth()` and `get_node_height()` recalculate on every call
- **Impact**: O(log n) operation repeated many times during visualization/assembly
- **Fix**: Add caching layer:
  ```python
  self._depth_cache: Dict[str, int] = {}
  self._height_cache: Dict[str, int] = {}
  ```
  Clear cache when nodes are modified

### 4. Redundant Node Loading Across Pipeline
- **Problem**: Nodes are loaded from DB/cache multiple times:
  1. During retrieval for MMR computation
  2. Again during retrieval for ancestor lookup
  3. Again during assembly to get node text
  4. Again during visualization (and loads ALL nodes!)
- **Impact**: Massive inefficiency - same nodes loaded 3-4 times
- **Fix**: 
  - Load nodes ONCE at the beginning of retrieval
  - Pass loaded nodes through RetrievalResult
  - Remove `get_all_nodes()` method entirely - it's an anti-pattern
  - Update assembly and visualization to use pre-loaded nodes

## Important Implementation Issues

### 5. Manual Migration Required
- **Problem**: Users must manually run `migrations/drop_depth_column.py`
- **Impact**: Runtime errors for existing users with old schema
- **Fix**: 
  - Add automatic migration on Store initialization
  - Add schema version tracking
  - Document migration in README

### 6. ChromaDB Metadata Inconsistency
- **Problem**: Existing ChromaDB entries have old `depth` field in metadata
- **Impact**: Inconsistent metadata between old and new entries
- **Fix**: Add one-time cleanup of ChromaDB metadata during migration

### 7. Missing Test Coverage
- **Problem**: No tests for `get_node_depth()` and `get_node_height()` methods
- **Impact**: Edge cases (cycles, missing parents) not tested
- **Fix**: Add comprehensive tests including:
  - Normal cases
  - Edge cases (orphaned nodes, cycles)
  - Performance benchmarks

## Code Quality Issues

### 8. Terminology Confusion
- **Problem**: Mixed use of "depth" vs "height" in comments and variable names
- **Impact**: Confusing for future developers
- **Fix**: Standardize on one convention throughout codebase

### 9. Dead Code
- **Problem**: `fix_depth_in_tests.py` migration helper still in repo
- **Impact**: Code clutter
- **Fix**: Delete the file

### 10. Test Fragility
- **Problem**: `test_span_coverage` relies on exact tree structure with tiny chunks
- **Impact**: Test can fail for unrelated changes
- **Fix**: Make test more robust or use larger, predictable chunks

## Priority Order for Fixes

1. **Immediate** (blocking bugs):
   - Fix duplicate methods in mock_store.py (#1)
   - Fix tree visualization to use coverage map (#2)

2. **High Priority** (performance):
   - Fix redundant node loading (#4) - this is architectural and affects entire pipeline
   - Add depth/height caching (#3)

3. **Important** (correctness):
   - Integrate migration system (#5)
   - Fix ChromaDB metadata (#6)
   - Add test coverage (#7)

4. **Nice to Have** (cleanup):
   - Fix terminology (#8)
   - Remove dead code (#9)
   - Improve test robustness (#10)

## Estimated Impact

- The visualization fix (#2) alone could reduce node loads by 95%+ for large documents
- Fixing redundant loading (#4) would reduce database queries by 75% (load once instead of 4 times)
- Caching (#3) would eliminate thousands of redundant calculations
- Together, these fixes could improve performance by 10-100x for large documents with deep trees

## Architectural Notes

The current architecture treats each component (retrieval, assembly, visualization) as independent, leading to repeated data loading. The correct approach is to:

1. Load all necessary nodes ONCE at the start of retrieval
2. Pass the loaded nodes through the pipeline via RetrievalResult
3. Never use `get_all_nodes()` - it's always wrong to load an entire document
4. Each component should work with the pre-loaded node set