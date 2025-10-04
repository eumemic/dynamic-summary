# Validate Flag Implementation

> **Historical Note:** The `--validate` CLI flag has been replaced by the standalone
> `ragzoom validate` command. This document captures the original design rationale.

## Overview

The `--validate` flag enables comprehensive validation checks throughout the RagZoom pipeline to ensure correctness of indexing and retrieval operations.

## Validation Checks

### Indexing Validation

When `--validate` is enabled during indexing, the following checks are performed:

1. **Complete Document Coverage**
   - Verify that the entire document text is indexed
   - Check that concatenating all leaf nodes reproduces the original document (accounting for overlaps)
   - No missing characters or sections

2. **Chunk Size Validation**
   - All chunks should be within ±20% of RAGZOOM_LEAF_TOKENS
   - Warn if any chunks are significantly over/under sized
   - Exception: Last chunk may be smaller

3. **Tree Structure Validation**
   - Parent spans must equal union of child spans
   - No gaps between sibling nodes
   - Spans must be valid (start < end)
   - No wraparound spans (span_end < span_start)
   - Tree is properly balanced (left-balanced binary tree)

4. **Summary Validation**
   - All non-leaf nodes have summaries
   - All summaries contain <<<MID>>> delimiter
   - mid_offset is valid (0 < mid_offset < len(summary))

### Retrieval/Assembly Validation

When `--validate` is enabled during retrieval, the following checks are performed:

1. **Frontier Completeness** (Invariant 1)
   - The frontier must cover the entire document span [0, max_span_end]
   - No gaps between adjacent frontier segments
   - Verify by checking that frontier spans form contiguous coverage

2. **No Overlapping Content** (Invariant 2)
   - No two frontier segments should cover the same span positions
   - When a parent and child are both in frontier nodes, parent should only output the half NOT covered by the child

3. **Coverage Map Validation**
   - All ancestors of selected nodes are marked as covered
   - Coverage map is consistent with frontier extraction

4. **Extraction Rule Validation**
   - Verify correct application of the extraction rule:
     - Both children covered → Node contributes nothing
     - Left child covered → Node contributes right half only
     - Right child covered → Node contributes left half only
     - Neither child covered → Node contributes full text

## Implementation Plan

### Phase 1: Core Validation Infrastructure
- [x] Add `--validate` flag to CLI commands (index, query)
- [x] Create `ragzoom/validate.py` module with validation functions
- [x] Add validate parameter to TreeBuilder and Retriever classes

### Phase 2: Indexing Validation
- [x] Implement document coverage check
- [x] Implement chunk size validation
- [x] Implement tree structure validation
- [x] Add validation hooks to CLI index command

### Phase 3: Retrieval Validation
- [x] Implement frontier completeness check
- [x] Implement no-overlap validation
- [x] Add validation to assemble.py frontier extraction
- [ ] Create detailed validation report

### Phase 4: Testing
- [x] Unit tests for each validation function
- [ ] Integration tests with known-bad scenarios
- [ ] Performance impact measurement

## Implementation Status

The `--validate` flag has been successfully implemented for both indexing and retrieval operations.

### What's Working

1. **Index validation** checks:
   - Document coverage (detects missing text at start/end or gaps)
   - Chunk sizes (warns about chunks outside ±20% of target)
   - Tree structure (validates spans, parent-child relationships, summaries)
   - **Early validation**: Checks happen as soon as possible:
     - Chunk sizes validated immediately after splitting
     - Document coverage checked before computing embeddings
     - Tree structure validated as each parent node is created

2. **Query validation** checks:
   - Frontier completeness (detects gaps in coverage)
   - No overlapping segments
   - Logs warnings when invariants are violated

3. **CLI Integration**:
   - `ragzoom index document.txt --validate`
   - `ragzoom query "search term" --validate`
   - Fast-fail behavior: Process exits with code 1 on validation failure

### Known Limitations

1. Frontier validation uses simplified node IDs in segments
2. Tree structure validation is comprehensive but may be slow for very large documents

## Usage Examples

```bash
# Validate during indexing
ragzoom index document.txt --validate

# Validate during query
ragzoom query "What happened?" --validate

# Validation is controlled via CLI flags only
# No environment variable support - use --validate explicitly
```

## Error Handling

- Validation failures should be clearly reported with:
  - What invariant was violated
  - Where in the document/tree the violation occurred
  - Suggested fixes or debugging steps
- In production, validation errors should be logged but not necessarily fatal
- In development/testing, validation errors should fail fast

## Performance Considerations

- Validation adds overhead, especially for large documents
- Some checks can be sampled (e.g., check every Nth node)
- Consider different validation levels: quick, standard, thorough
- **Early validation benefits**:
  - Fails fast before expensive operations (e.g., embeddings)
  - Saves computational resources by catching issues early
  - Provides immediate feedback during development
  - Helps identify the exact point where issues occur

## Future Enhancements

- Visual validation output showing tree structure and coverage
- Automatic fixing of minor validation issues
- Integration with monitoring/alerting systems
- Validation metrics and statistics
