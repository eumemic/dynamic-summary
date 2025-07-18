# Dynamic Programming Transition Analysis

## Current State of the Codebase

The RagZoom codebase is currently in a transitional state between two **frontier generation** approaches:

1. **Legacy Approach** (currently in production): 
   - Uses MMR to select diverse nodes
   - Builds coverage map (selected nodes + ancestors)
   - Extracts frontier using `_extract_frontier()` 
   - Often produces frontier exceeding budget
   - Applies `_enforce_budget_constraint()` to trim
   - In assembly, applies `_apply_slope_cap()` which may push over budget again
   - May need another trim pass

2. **Dynamic Programming Approach** (mostly implemented):
   - Uses the SAME MMR to select diverse nodes
   - Builds the SAME coverage map
   - Generates frontier using `dp_generator.find_optimal_frontier()`
   - Produces a frontier that respects budget from the start
   - No post-hoc trimming needed
   - Slope capping integrated into the algorithm (if implemented)

The DP algorithm is currently behind a `frontier_mode` flag. The key insight: both approaches share the entire retrieval pipeline except for the frontier generation step.

## Documentation Discrepancies Explained

The apparent inconsistencies between documents stem from this transitional state:

### Document Purposes:
- **AGENT_INSTRUCTIONS.md**: Describes the current production system (legacy algorithm)
- **docs/architecture.md**: Describes the target architecture with DP algorithm as if already complete
- **docs/v2/dynamic-frontier-design.md**: Design proposal for the DP algorithm (mostly implemented)
- **docs/agent-handoff.md**: Chronicles the beginning of the DP implementation in Session 1

## What Needs to Happen

### 1. Complete DP Implementation
- [ ] Finish any remaining gaps in the dynamic frontier implementation
- [ ] Ensure all edge cases are handled correctly
- [ ] Verify performance meets or exceeds the legacy system

### 2. Switch to Dynamic Mode as Default
- [ ] Change default `frontier_mode` to "dynamic"
- [ ] Run comprehensive tests to ensure no regressions
- [ ] Remove the frontier_mode configuration option entirely

### 3. Remove Dead Code
Once dynamic mode is proven stable, remove:

#### From `ragzoom/retrieve.py`:
- `_extract_frontier()` method (the legacy frontier building logic)
- `_enforce_budget_constraint()` method (post-hoc budget trimming)
- Any conditional logic checking `frontier_mode` (lines 260-272 in retrieve flow)
- Keep MMR diversity selection and coverage map propagation (used by both approaches)

#### From `ragzoom/assemble.py`:
- `_apply_slope_cap()` (if handled in DP algorithm)
- `_find_ancestor_at_depth()`
- `_find_intermediate_path()`
- `trim_frontier_to_budget()`
- `_count_frontier_tokens()`
- `_has_span_overlap_detailed()` and deduplication logic
- `assemble_with_budget()`

#### Other Files:
- Remove `frontier_mode` from configuration
- Remove any tests specific to the legacy algorithm
- Update CLI and API to remove frontier_mode parameters

### 4. Update Documentation

#### AGENT_INSTRUCTIONS.md:
- Update "Retrieval" section to describe DP frontier generation
- Update architecture overview to show:
  - Query → Embedding → Vector search → MMR diversity → Coverage map → DynamicFrontierGenerator → Frontier
- Keep mentions of MMR diversity and coverage maps (they're still used!)
- Update frontier extraction description to reference DP algorithm
- Update "Recent Changes" to note the completion of DP transition

#### docs/architecture.md:
- Ensure mermaid diagrams accurately reflect the implemented DP flow
- Update component descriptions to match final implementation
- Remove any remaining references to the legacy algorithm

#### docs/developer-guide.md:
- Update to reflect current development practices
- Add information about the simplified codebase post-DP
- Document any new testing patterns

#### docs/v2/dynamic-frontier-design.md:
- Add an "Implementation Status" section marking it as complete
- Note any deviations from the original design
- Move to a "completed designs" folder or mark as historical

### 5. Update Tests

- Ensure comprehensive test coverage for the DP algorithm
- Remove tests specific to the legacy algorithm
- Update mock store if needed for DP-specific behavior
- Verify all integration tests pass with the new algorithm

### 6. Performance Validation

Before fully removing the legacy code:
- Benchmark DP algorithm against legacy for various document sizes
- Verify memory usage is acceptable
- Ensure API response times meet requirements
- Test with edge cases (very large documents, deeply nested trees)

## Benefits After Transition

1. **Simpler Frontier Generation**: Removal of multi-stage corrective logic for frontier building
2. **Budget Guarantees**: Frontier respects budget constraint by construction, not through trimming
3. **No More Budget Overruns**: Eliminates the slope-cap → over-budget → re-trim cycle
4. **Better Performance**: Memoization and optimal substructure for frontier generation
5. **Clearer Logic**: Single-pass frontier generation instead of extract → trim → slope-cap → re-trim

## Migration Checklist

- [ ] Complete DP implementation gaps
- [ ] Comprehensive testing of dynamic mode
- [ ] Performance benchmarking
- [ ] Switch default to dynamic mode
- [ ] Monitor for issues in production
- [ ] Remove frontier_mode flag
- [ ] Delete legacy algorithm code
- [ ] Update all documentation
- [ ] Update tests
- [ ] Final code review
- [ ] Tag release marking the transition complete

## Notes for Future Agents

When you encounter references to two different frontier generation approaches in the codebase or documentation, remember this transition period. The goal is to have a single, clean implementation where:

1. MMR and coverage maps continue to select which nodes are relevant
2. The DP algorithm builds an optimal, budget-respecting frontier from those nodes
3. No post-processing corrections are needed

The DP transition is NOT about replacing the entire retrieval system - it's about replacing the error-prone frontier extraction and budget management logic with a mathematically sound, single-pass algorithm.

## Transition Progress

### Phase 1: Remove frontier_mode flag (COMPLETED)
- Removed `frontier_mode` from `RagZoomConfig` in `config.py`
- Updated both occurrences in `retrieve.py` to always use DP path
- Updated `test_dp_frontier.py` to not reference frontier_mode
- All tests pass (137 passed, 4 skipped)

### Phase 2: Identified Dead Code
After removing the frontier_mode conditionals, the following methods are now dead code:

**In retrieve.py:**
- `_extract_frontier()` (line 342) - Legacy frontier extraction
- `_enforce_budget_constraint()` (line 600) - Post-hoc budget trimming

**In utils.py:**
- `get_actual_node_text()` - Only used by `_enforce_budget_constraint`

### Phase 3: Dead Code Removal (COMPLETED)
Successfully removed:
- `_extract_frontier()` from retrieve.py
- `_enforce_budget_constraint()` from retrieve.py  
- `get_actual_node_text()` from utils.py
- Skipped `test_extract_frontier_logic` with note for future analysis

All tests still pass (136 passed, 5 skipped).

**Note:** `clean_mid_delimiter` in utils.py is still used (by assemble.py).

### Phase 4: Analysis of assemble.py
The `assemble()` method still has a conditional that routes between DP and legacy assembly based on whether `frontier_segments` is set. Since we always use DP now, this should always be set.

**Potential dead code in assemble.py (lines 36-159):**
- The entire legacy assembly path
- `_remove_children_with_parents_in_frontier()`
- `_sort_nodes_chronologically()`
- `_apply_slope_cap()`
- `_build_coverage_map()`
- `_extract_node_text_with_span()`
- `_has_span_overlap_detailed()`
- `_apply_smoothing_pass()`
- And potentially more helper methods

**Action needed:** Verify that frontier_segments is always set, then remove the legacy assembly path.