# /issue

Arguments: "$ARGUMENTS"

Capture a GitHub issue without interrupting current work. Use conversation context to create a concise, actionable issue.

## Core Intent

When developers discover bugs or think of features while coding, they need to quickly document them without losing focus. Create a GitHub issue that captures the essence of the discovery with just enough context to be actionable later.

## Process

1. **Understand**: What issue did the user discover? Use arguments (if provided) combined with conversation context
2. **Extract context**: Current files, functions, error messages from conversation
3. **Check labels**: Run `gh label list` to see available labels
4. **Create missing labels**: If needed, create appropriate label with `gh label create` (e.g., "tech-debt", "refactor", "performance")
5. **Create issue**:
   - Title: Action-oriented, specific (e.g., "Fix tree traversal at boundary conditions")
   - Body: Essential context only - what/where/why, reproduction steps for bugs
   - Labels: Type (bug/enhancement/feature/tech-debt) + any obvious tags
6. **Confirm**: Show title and one-line summary, get user approval
7. **Submit**: `gh issue create`, return issue URL
8. **Continue**: Let user resume their work

## Key Principles

- Brevity over completeness - capture essence, not every detail
- Trust future readers' intelligence
- Include specific code locations when relevant
- No speculation or implementation details unless obvious
- One clear next action for whoever picks it up

## Examples

Input: "tree traversal skips nodes when budget equals size"
→ Creates: "Fix tree traversal skipping nodes at exact budget boundaries"
   Body: "In `dynamic_tiling.py:_find_optimal_tiling_for_span()`, nodes are skipped when remaining budget exactly equals node token count. Should include the node instead."
   Label: bug

Input: (no args, context shows flaky async tests)
→ Creates: "Fix intermittent async summarization test failures"
   Body: "test_concurrent_summarization fails ~20% in CI, passes locally. Likely race condition in semaphore handling. Start with `index.py:_summarize_node_pair()`"
   Label: bug

Input: "lots of dead code in dirty node marking"
→ Checks labels, creates "tech-debt" if missing
→ Creates: "Remove dead dirty node marking code"
   Body: "Dirty node infrastructure exists but is never used in production. Either remove or implement document updates."
   Label: tech-debt

Remember: Quick capture beats perfect documentation. A good issue points someone in the right direction, not holds their hand.