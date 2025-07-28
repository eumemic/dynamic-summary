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

2. **Monitor CI with fail-fast**: Use `gh pr checks --watch --fail-fast`
   - Exits immediately on first CI failure
   - Fix the failure, commit, and resume monitoring
   - Continues until all checks pass

3. **Read Reviews**: Once all CI checks pass (build complete):
   - Check for reviews: `gh pr view --comments`
   - Look for automated review feedback (e.g., claude-review)
   - If issues found, fix them and return to step 2

4. **Success Criteria**:
   - All CI checks pass
   - All review feedback read and addressed
   - No outstanding issues to fix

## Key Principles

- **Fail fast**: `--fail-fast` flag exits on first CI failure for quick fixes
- **Reviews after CI**: Read reviews only after all checks pass
- **Batch fixes**: Fix all issues before pushing
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

Monitoring CI with fail-fast...
❌ Build failed: missing import

Fixing import issue...
✅ Fixed and ready to commit
(User runs /commit)

Resuming CI monitoring...
✅ All CI checks passed!

Checking for code reviews...
📝 Found claude-review with feedback:
- Temp file conflicts in pre-commit hook
- Missing safety check for branch reset

Fixing review issues...
✅ Fixed all issues - ready to commit
(User runs /commit)

Resuming CI monitoring...
✅ All CI checks passed!

Checking for code reviews...
✅ No new issues found

✅ PR ready for review
PR #42: https://github.com/owner/repo/pull/42
```

Remember: The goal is a green build with minimal CI runs. Use /commit to push fixes.

## Retrospective
After PR is ready, reflect on three levels:
1. **Command**: Did this minimize CI churn effectively?
2. **Conformance**: Is the separation of concerns (commit vs PR) clear?
3. **Meta**: Should commands include more CI/CD best practices?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.