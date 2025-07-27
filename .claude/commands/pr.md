---
allowed-tools: Bash, Read, Edit, MultiEdit, Grep
description: Create PR if needed, monitor CI, fix issues
argument-hint: [PR title]
---

# /pr
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Current branch: !`git branch --show-current`
- Existing PR: !`gh pr view --json number,state 2>/dev/null | jq -r '"#" + (.number|tostring) + " (" + .state + ")"' || echo "No PR"`

Arguments: "$ARGUMENTS"

Create PR if needed, monitor CI, and fix issues proactively. Assumes code is already pushed by /commit.

## Core Intent

Get code through CI successfully with minimal back-and-forth. Monitor CI, fix issues immediately, batch fixes to avoid CI churn.

## Workflow

1. **Create PR**: If no PR exists, create one (reference related issues with "Fixes #123")
2. **Monitor**: Watch CI status with `gh pr checks --watch --fail-fast`
3. **Fix Immediately**: When issues found:
   - Stop monitoring
   - Fix ALL issues (CI failures, review comments)
   - Commit and push fixes using /commit
4. **Iterate**: Continue monitoring and fixing until all checks pass

## Key Principles

- **Fail fast**: `--fail-fast` flag exits on first CI failure
- **Batch fixes**: Multiple commits locally, one push to minimize CI runs
- **Be proactive**: Auto-fix build/test/lint issues
- **Ask first**: For style preferences and non-blocking suggestions

## Issue Priority

**Auto-fix**: Build failures, test failures, linting, missing imports
**Ask first**: Reviewer nits, refactoring suggestions, style preferences

## Final Output

When complete, always include:
```
✅ PR ready for review
PR #N: https://github.com/owner/repo/pull/N
```

## Examples

```
Creating PR...
✅ PR #42: https://github.com/owner/repo/pull/42

Monitoring CI...
❌ Build failed: missing import

Fixing import issue...
✅ Fixed and ready to commit

(User runs /commit to push the fix)

Resuming CI monitoring...
✅ All checks passed!

PR #42: https://github.com/owner/repo/pull/42
```

Remember: The goal is a green build with minimal CI runs. Use /commit to push fixes.

## Retrospective
After PR is ready, reflect on three levels:
1. **Command**: Did this minimize CI churn effectively?
2. **Conformance**: Is the separation of concerns (commit vs PR) clear?
3. **Meta**: Should commands include more CI/CD best practices?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.