0a. Study `specs/*` with up to 500 parallel subagents to learn the application specifications.
0b. Study `ralph/IMPLEMENTATION_PLAN.md`.
0c. For reference, the application source code is in the project root.

1. Your task is to implement functionality per the specifications using parallel subagents. Follow `ralph/IMPLEMENTATION_PLAN.md` and choose the most important item to address. Before making changes, search the codebase (don't assume not implemented) using subagents. You may use up to 500 parallel subagents for searches/reads and only 1 subagent for build/tests.

2. After implementing functionality or resolving problems, run the tests for that unit of code. If functionality is missing then it's your job to add it as per the specifications.

3. When you discover issues, immediately update `ralph/IMPLEMENTATION_PLAN.md` with your findings. When resolved, delete the item from the plan.

4. Before committing, use the code-simplifier skill to clean up the implementation, then use the code-review skill to check for issues. Address any issues raised.

5. When ready to commit, `git add -A` then `git commit` with a message describing the changes. The pre-commit hook will validate your work. If it fails, fix the issues and try again.

99999. Important: When authoring documentation, capture the why — tests and implementation importance.
999999. Important: Single sources of truth, no migrations/adapters. If tests unrelated to your work fail, resolve them as part of the increment.
9999999. Keep `ralph/IMPLEMENTATION_PLAN.md` current with learnings — future work depends on this to avoid duplicating efforts. Update especially after finishing your turn.
99999999. For any bugs you notice, resolve them or document them in `ralph/IMPLEMENTATION_PLAN.md` even if unrelated to current work.
999999999. Implement functionality completely. Placeholders and stubs waste efforts and time redoing the same work.
9999999999. When `ralph/IMPLEMENTATION_PLAN.md` becomes large, clean out completed items.
99999999999. If you find inconsistencies in `specs/*`, update the specs to resolve them.
999999999999. When you discover operational knowledge (how to run tests, build gotchas), update the relevant skill or create a new one using skill-development.
