0a. Study `specs/*` with up to 250 parallel subagents to learn the application specifications.
0b. Study `ralph/IMPLEMENTATION_PLAN.md` (if present) to understand the plan so far.
0c. For reference, the application source code is in the project root.

1. Study `ralph/IMPLEMENTATION_PLAN.md` (if present; it may be incorrect) and use up to 500 subagents to study existing source code and compare it against `specs/*`. Analyze findings, prioritize tasks, and create/update `ralph/IMPLEMENTATION_PLAN.md` as a bullet point list sorted by priority of items yet to be implemented. Consider searching for TODO, minimal implementations, placeholders, skipped/flaky tests, and inconsistent patterns. Keep the plan up to date with items considered complete/incomplete.

IMPORTANT: Plan only. Do NOT implement anything. Do NOT assume functionality is missing; confirm with code search first. Use the "don't assume not implemented" principle - always search before concluding something doesn't exist.

ULTIMATE GOAL: Sync the codebase to match the specifications. Consider missing elements and plan accordingly. If an element is missing from specs, search first to confirm it doesn't exist in code, then if needed author the specification at `specs/FILENAME.md`.
