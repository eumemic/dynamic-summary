# Legacy Assembly Test Analysis

## Overview

This document analyzes the tests that depend on the legacy assembly path and proposes a migration strategy for completing the DP transition.

## Current State

- The DP algorithm is the only path for retrieval (frontier_mode removed)
- The legacy assembly path remains in `assemble.py` (lines 36-159)
- No tests currently use the DP assembly path (no tests create RetrievalResult with frontier_segments)
- 8 tests are skipped (3 incomplete indexing, 4 legacy assembly, 1 legacy retrieval)

## Tests Using Legacy Assembly

### 1. test_assembly_integration.py (8 tests)
Tests critical assembly behaviors:
- `test_no_duplicate_content_in_assembly`: Ensures no repeated content
- `test_span_consistency_in_tree`: Validates span boundaries
- `test_mid_delimiter_extraction_no_overlaps`: Tests MID delimiter handling
- `test_slope_cap_deduplication`: Tests slope cap doesn't create duplicates
- `test_coverage_map_includes_ancestors`: Validates coverage map correctness
- `test_invalid_frontier_with_parent_and_child`: Tests parent-child deduplication
- `test_zero_width_span_handling`: Tests edge case with zero-width spans
- `test_full_pipeline_no_repetition`: End-to-end test

**Verdict**: These test important invariants that DP assembly should also maintain.

### 2. test_assembly_ordering_bug.py (3 tests)
Tests specific bugs in legacy assembly:
- `test_assembly_ordering_bug_exact_scenario`: Tests a specific ordering bug
- `test_parent_and_child_span_overlap`: Tests span overlap detection
- `test_sorting_by_depth_when_spans_are_identical`: Tests sorting logic

**Verdict**: May be testing bugs specific to legacy implementation.

### 3. test_budget_guarantee.py (multiple tests)
Tests budget enforcement in legacy assembly.

**Verdict**: DP handles budget by construction, so these tests may not apply.

### 4. test_chunk_size_regression.py
Tests related to chunk size handling.

**Verdict**: Needs analysis to determine if relevant to DP.

### 5. test_parent_child_frontier.py (3 tests - SKIPPED)
Already marked as legacy tests to be removed.

### 6. test_mid_delimiter.py (1 test - SKIPPED)
Already marked as legacy test to be removed.

## Key Behaviors to Preserve

Based on the test analysis, the following behaviors must be maintained in any assembly implementation:

1. **No duplicate content**: The same text should not appear multiple times
2. **Span consistency**: Text should be extracted according to node spans
3. **MID delimiter handling**: Proper extraction of left/right halves of summaries
4. **Parent-child deduplication**: If both parent and child are in frontier, only keep parent
5. **Coverage map correctness**: Ancestors of selected nodes should be marked as covered
6. **Zero-width span handling**: Edge case where span_start == span_end

## Migration Strategy

### Option 1: Full Migration (Recommended)
1. Create equivalent tests that use the full DP pipeline (index → retrieve → assemble)
2. Verify DP assembly handles all the important invariants
3. Remove legacy assembly code and tests
4. Un-skip the already marked legacy tests

### Option 2: Adapter Pattern
1. Create a `LegacyToDP` adapter that converts legacy RetrievalResult to one with frontier_segments
2. Modify existing tests to use the adapter
3. Gradually migrate tests to use full DP pipeline
4. Remove adapter and legacy code once all tests migrated

### Option 3: Parallel Testing
1. Keep legacy tests but mark as integration tests
2. Create new unit tests for DP assembly
3. Run both sets of tests during transition
4. Remove legacy once confident in DP coverage

## Recommended Next Steps

1. **Implement DP assembly tests**: Create `test_dp_assembly.py` with tests for:
   - Basic segment assembly
   - LEFT/RIGHT side extraction
   - MID delimiter handling
   - Edge cases (empty segments, missing nodes)

2. **Create migration tests**: For each legacy test, create a corresponding test using the full DP pipeline:
   ```python
   # Instead of manually creating RetrievalResult:
   retrieval_result = retriever.retrieve(query, document_id)
   assembly = assembler.assemble(retrieval_result)
   ```

3. **Verify invariants**: Ensure DP maintains the critical behaviors:
   - No duplicate content (already handled by DP's span-based approach)
   - Parent-child deduplication (already handled by DP's frontier generation)
   - Proper text extraction (already handled by SummarySegment)

4. **Remove legacy code**: Once all tests pass with DP:
   - Remove legacy assembly path from `assemble.py`
   - Remove all helper methods only used by legacy path
   - Remove skipped tests
   - Update documentation

## Risk Assessment

- **Low Risk**: DP algorithm is mathematically sound and simpler than legacy
- **Medium Risk**: Some edge cases may not be covered by current DP implementation
- **Mitigation**: Comprehensive testing before removal, keep legacy code until confident

## Timeline Estimate

1. Create DP assembly tests: 1-2 hours
2. Migrate existing tests: 2-3 hours
3. Verify all invariants: 1-2 hours
4. Remove legacy code: 1 hour
5. Update documentation: 1 hour

Total: 6-9 hours of focused work

## Conclusion

The legacy assembly code can be safely removed once we:
1. Create comprehensive tests for DP assembly
2. Verify DP handles all important invariants
3. Migrate or replace tests that depend on legacy behavior

The DP approach is simpler and more correct by construction, making this transition worthwhile.