# /push

Arguments: "$ARGUMENTS"

Push code, create PR if needed, monitor CI, and fix issues proactively.

## Core Intent

Get code through CI successfully with minimal back-and-forth. Push, monitor, fix issues immediately, batch fixes to avoid CI churn.

## Workflow

1. **Push**: Current branch to remote
2. **Create PR**: If needed, reference related issues (e.g., "Fixes #123")
3. **Monitor**: Watch CI status with `gh pr checks --watch --fail-fast`
4. **Fix Immediately**: When issues found:
   - Stop monitoring
   - Fix ALL issues (CI failures, review comments)
   - Commit fixes locally (don't push yet)
5. **Push Once**: After build completes successfully, push all fixes together

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
$ git push
✅ Pushed to origin/feature-branch

Creating PR...
✅ PR #42: https://github.com/owner/repo/pull/42

Monitoring CI...
❌ Build failed: missing import

Fixing import issue...
✅ Fixed and committed locally

Waiting for build completion...
✅ Build passed!

Pushing fixes...
✅ All issues resolved

PR #42: https://github.com/owner/repo/pull/42
```

Remember: Fix fast, push once. The goal is a green build with minimal CI runs.