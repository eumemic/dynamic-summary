# JSCPD Line Number Bug Analysis

## Summary

We discovered a bug in jscpd where it reports incorrect line numbers for cross-file duplications. Specifically, it sometimes reports end line numbers that are:
1. Before the start line (e.g., start: 463, end: 226)
2. Incorrect even when not backwards (e.g., reporting 463-466 when only line 463 matches)

## Root Cause

The bug is in the Rabin-Karp algorithm implementation in `packages/core/src/rabin-karp.ts`. The `enlargeClone` method incorrectly updates the end position of duplicates.

### How the Algorithm Works

1. The algorithm iterates through tokens in File A
2. For each token, it checks if that token's hash exists in the store (from other files)
3. If found, it creates a clone between the current position in File A and the stored position from File B
4. As it continues iterating, if consecutive tokens also match, it calls `enlargeClone` to extend the duplicate

### The Bug

The problem occurs in `enlargeClone`:

```typescript
clone = RabinKarp.enlargeClone(clone, iteration.value, mapFrameInStore);
```

Here, `mapFrameInStore` is the frame that was stored when the clone was first created. It doesn't get updated as the algorithm continues. This causes several issues:

1. **Stale Frame Data**: The `mapFrameInStore` contains the original position from when the match started, not the current corresponding position in File B
2. **No Verification**: The algorithm doesn't verify that the tokens actually continue to match in both files
3. **Incorrect Line Calculation**: The end line for File B gets set to unrelated values from the stale frame

## Example

In our test case:
- File A (telemetry_analysis.py:761-768) contains an 8-line block starting with `summary_attempts = node.get("summary_attempts", [])`
- File B (telemetry_viz.py:463) contains only the first line of that block
- jscpd incorrectly reports it as lines 463-226 (or 463-466 with our partial fix)

## Attempted Fix

We attempted to fix this by calculating the offset and creating a synthetic frame:

```typescript
const offsetInA = iteration.value.start.loc.start.line - clone.duplicationA.start.line;
const syntheticFrameB = {
  ...cloneStartFrameB,
  start: { ...cloneStartFrameB.start, loc: { ...cloneStartFrameB.start.loc, 
    start: { line: clone.duplicationB.start.line + offsetInA, column: cloneStartFrameB.start.loc.start.column }
  }},
  // similar for end
};
```

This partially fixes the backwards line numbers but still produces incorrect results because it assumes both files have identical duplicate content.

## Proper Fix

A proper fix would require:

1. **Token Verification**: When enlarging a clone, verify that the tokens actually match in both files
2. **Dynamic Frame Lookup**: Instead of using a stale frame, dynamically look up the corresponding frame in File B
3. **Early Termination**: Stop enlarging the clone when tokens no longer match

This would be a significant change to the algorithm and might impact performance.

## Workaround

For now, users can:
1. Ignore cross-file duplications with suspicious line numbers
2. Use higher thresholds to reduce false positives
3. Post-process results to filter out duplicates with end < start

## Impact

This bug affects the accuracy of duplicate detection, especially for:
- Cross-file duplications where the duplicate content differs in length
- Files with similar starting patterns but different continuations
- Any analysis that relies on accurate line number reporting