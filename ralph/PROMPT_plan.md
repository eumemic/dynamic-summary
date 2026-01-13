0a. Study `specs/*` with up to 250 parallel subagents to learn the application specifications.
0b. Study `ralph/IMPLEMENTATION_PLAN.md` (if present) to understand the plan so far.
0c. For reference, the application source code is in the project root.
0d. When spawning subagents via the Task tool, use `model: "${RALPH_SUBAGENT_MODEL}"` for efficiency.

1. Study `ralph/IMPLEMENTATION_PLAN.md` (if present; it may be incorrect) and use up to 500 subagents to study existing source code and compare it against `specs/*`. Analyze findings, prioritize tasks, and create/update `ralph/IMPLEMENTATION_PLAN.md` as a bullet point list sorted by priority of items yet to be implemented. Consider searching for TODO, minimal implementations, placeholders, skipped/flaky tests, and inconsistent patterns. Keep the plan up to date with items considered complete/incomplete.

2. Each work item in the plan MUST include:
   - **Spec reference**: Which spec file defines the requirement (e.g., `[spec: client-managed-chunking.md]`)
   - **Success criterion**: A testable condition that proves the item is complete (e.g., "Success: `append_batch(['A', 'B', 'C'])` with `target_chunk_tokens=None` creates exactly 3 leaves")
   - **Test**: The test function that will verify the success criterion (e.g., "Test: `test_append_batch_preserves_atomic_units`")

Example work item format:
```
- [ ] Update `IndexConfig.target_chunk_tokens` from `int` to `int | None`
  - Spec: specs/client-managed-chunking.md § Activation
  - Success: `IndexConfig(target_chunk_tokens=None, ...)` is valid; existing configs with int values still work
  - Test: `test_config_accepts_none_target_chunk_tokens`, `test_config_backward_compatible`
  - Location: ragzoom/config.py:277
```

IMPORTANT: Plan only. Do NOT implement anything. Do NOT assume functionality is missing; confirm with code search first. Use the "don't assume not implemented" principle - always search before concluding something doesn't exist.

IMPORTANT: Don't make trivial tweaks to the plan. If the existing plan is substantively correct and complete, leave it alone. Only modify the plan when there are real gaps, incorrect items, or missing work. Reformatting, rewording, or reorganizing without adding substance is wasted effort.

ULTIMATE GOAL: Sync the codebase to match the specifications. Consider missing elements and plan accordingly. If an element is missing from specs, search first to confirm it doesn't exist in code, then if needed author the specification at `specs/FILENAME.md`.
