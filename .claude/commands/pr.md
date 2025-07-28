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

2. **Monitor Everything Concurrently**: Watch for THREE possible outcomes:
   - **CI Failure** → Fix immediately and resume monitoring
   - **Review Completed** (e.g., claude-review finishes) → Address feedback and resume monitoring  
   - **All Checks Pass + No Outstanding Reviews** → Declare success

3. **Monitoring Strategy**:
   - Use `gh pr checks --watch` to see all CI statuses
   - Periodically check for completed reviews: `gh pr view --comments`
   - Watch for automated reviews (claude-review) transitioning from pending → complete
   - Act on the FIRST actionable event (failure or review)

4. **Fix Issues Immediately**:
   - Stop monitoring when issues found
   - Fix ALL known issues (CI failures + review feedback)
   - Commit and push fixes using /commit
   - Resume monitoring for more issues

5. **Success Criteria**:
   - All CI checks pass (including automated reviews)
   - All review feedback addressed
   - No CHANGES_REQUESTED status on any review

## Key Principles

- **Monitor concurrently**: Watch CI and reviews simultaneously, not sequentially
- **Act on first issue**: Don't wait for all CI to pass if reviews are ready
- **Batch fixes**: Address all known issues before pushing
- **Be proactive**: Auto-fix build/test/lint/security issues
- **Ask first**: For style preferences and non-blocking suggestions

## Issue Priority

**Auto-fix**: Build failures, test failures, linting, missing imports, security issues
**Ask first**: Reviewer nits, refactoring suggestions, style preferences
**Must address**: Any issues marked as "Critical" or "Important" in code reviews

## Final Output

Before declaring PR ready:
1. Verify all CI checks passed
2. Read and address all code review comments
3. Ensure no CHANGES_REQUESTED reviews

Only then output:
```
✅ PR ready for review
PR #N: https://github.com/owner/repo/pull/N
```

If reviews found issues that were fixed, mention:
```
✅ All review comments addressed
✅ PR ready for final review
PR #N: https://github.com/owner/repo/pull/N
```

## Examples

```
Creating PR...
✅ PR #42: https://github.com/owner/repo/pull/42

Monitoring PR status...
⏳ CI: 5 pending, 2 passed | Reviews: claude-review pending

❌ Build failed: missing import
Stopping monitor to fix...
✅ Fixed import issue - ready to commit
(User runs /commit)

Resuming PR monitoring...
⏳ CI: 3 pending, 4 passed | Reviews: claude-review pending

📝 claude-review completed with feedback (while CI still running):
- Temp file conflicts in pre-commit hook  
- Missing safety check for branch reset

Stopping monitor to address review...
✅ Fixed all review issues - ready to commit
(User runs /commit)

Resuming PR monitoring...
✅ CI: All passed | Reviews: All addressed

✅ PR ready for final review
PR #42: https://github.com/owner/repo/pull/42
```

Remember: The goal is a green build with minimal CI runs. Use /commit to push fixes.

## Retrospective
After PR is ready, reflect on three levels:
1. **Command**: Did this minimize CI churn effectively?
2. **Conformance**: Is the separation of concerns (commit vs PR) clear?
3. **Meta**: Should commands include more CI/CD best practices?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.