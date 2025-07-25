---
allowed-tools: Bash
description: Create GitHub issue from current context
argument-hint: [issue description]
---

# /issue
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Available labels: !`gh label list --json name -q '.[].name' | head -20 | tr '\n' ', ' | sed 's/,$//'`
- Recent issues: !`gh issue list --limit 3 --json number,title -q '.[] | "#" + (.number|tostring) + ": " + .title'`

## Strategic Guidance
Capture issues in the moment of discovery. The best bug report is written while the context is fresh. Include code locations and error messages, but trust future readers to investigate details.

## Task
Arguments: "$ARGUMENTS"

Create a concise, actionable issue from current context without disrupting flow.

## Process

1. **Parse**: Extract issue from args + conversation context
2. **Shape**: 
   - Title: Action-oriented ("Fix X" not "X is broken")
   - Body: What/where/why + code locations
   - Labels: Pick from available or create if needed
3. **Confirm**: "Create issue: [title]?"
4. **Submit**: `gh issue create --title "..." --body "..." --label "..."`

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

## Retrospective
After creating the issue, reflect on three levels:
1. **Command**: Did this enable quick, effective issue capture?
2. **Conformance**: Is the brevity principle well-balanced?
3. **Meta**: Should commands include more GitHub-specific patterns?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.