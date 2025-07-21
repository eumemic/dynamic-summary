# /push

Push code to remote, create PR if needed, then monitor for build status and comments.

## Workflow Overview

**The workflow follows this sequence**:
1. **PUSH** current commits to remote
2. **CREATE PR** if on a feature branch without an existing PR
3. **MONITOR** the PR for build failures and review comments
4. **FIX** any issues immediately when detected
5. **COMMIT** fixes incrementally (but don't push yet)
6. **PUSH** all fixes after build completes successfully

This minimizes CI churn by batching fixes before pushing.

## Instructions:

1. **Push Code**:
   - Push current branch to remote with `git push`
   - If branch not tracked, use `git push -u origin <branch-name>`

2. **Create PR (if needed)**:
   - Check if PR already exists for current branch
   - If not, create PR with `gh pr create`
   - **IMPORTANT**: Reference any related GitHub issues in PR body (e.g., "Fixes #123")
   - Use descriptive title and comprehensive body
   - Include test plan and summary of changes

3. **Initial PR Check**:
   - Get PR number and status
   - Check CI build status and existing comments
   - If ANY failures or issues exist, immediately proceed to fixing (skip monitoring)

4. **Monitoring Loop** (only if no issues):
   - Use `gh pr checks <PR#> --watch --fail-fast` to monitor CI status
   - This command will:
     - Update check status every 10 seconds automatically
     - Exit immediately when ANY check fails (--fail-fast)
     - Show live status of all checks in parallel
   - If a check fails:
     - The command exits automatically
     - STOP monitoring, fix the failure immediately
   - Also check for new review comments separately
   - **IMPORTANT**: The fail-fast flag ensures we detect and fix issues as soon as they occur

5. **Issue Resolution Mode**:
   - **When you detect ANY issue, STOP monitoring immediately**
   - Use built-in todo tracking to record ALL known issues:
     - CI failures (check logs thoroughly)
     - Review comments requiring fixes
     - Linting/formatting issues
   - Fix ALL issues before returning to monitoring
   - Commit fixes incrementally but DO NOT push yet
   
6. **Fix Priority**:
   - **Automatically fix**:
     - Build failures
     - Test failures
     - Linting/formatting issues
     - Missing dependencies
     - Broken imports
   
   - **Ask before fixing**:
     - Minor "nits" from reviewers
     - Refactoring suggestions
     - Style preferences
     - Non-blocking suggestions

7. **Completion and Push**:
   - Once build completes AND all issues are fixed:
     - Review all commits made during fixes
     - Push once to minimize CI runs
   - If new issues appear after push, repeat the cycle

8. **Final Dashboard**:
   - When presenting the final status (whether success or current state):
     - Include a direct link to the PR for easy access
     - Format: `PR #N: https://github.com/owner/repo/pull/N`
   - This allows the user to quickly navigate to the PR

9. **Stop Conditions**:
   - User explicitly asks to stop
   - PR is merged or closed
   - All builds complete with all issues resolved

## Correct Flow Example:

```
User: /push
Assistant: Pushing current branch to remote...

$ git push
✅ Successfully pushed to origin/feature-branch

Checking for existing PR...
No PR found. Creating new PR...

$ gh pr create --title "Add new feature X" --body "..."
✅ Created PR #9: https://github.com/owner/repo/pull/9

Checking PR #9 status...

❌ Found issues that need fixing:
- CI Build: Failed (test failure in test_feature.py)
- Claude Review: Completed (found import issue)

Stopping monitoring to fix all issues...

Using todo tracking to manage issues...

Fixing broken import in orchestrator/CLAUDE.md...
[Makes fix and commits]
✅ Todo #1 completed

Analyzing CI failure... 
[Investigates and fixes]
[Commits fix]
✅ Todo #2 completed

All issues fixed and committed locally.
Resuming monitoring to wait for build completion...

[30s later]
⏳ Monitoring... (build still running)

[30s later]
✅ Build completed successfully!

Pushing all commits to remote...

✅ All issues resolved. PR is ready for review.

PR #9: https://github.com/eumemic/dynamic-summary/pull/9
```

## BAD Example (what NOT to do):

```
❌ WRONG: See failure → Continue monitoring → Fix later
❌ WRONG: Fix one issue → Push → Fix another → Push again
❌ WRONG: See review comment → Ignore and keep monitoring
```

## Implementation Notes:

- Use built-in todo tracking (TodoWrite) to manage all issues
- Track all issues before starting fixes
- Commit incrementally as fixes are completed
- Only push after all issues resolved and build completes
- Maintain issue state between monitoring cycles
- Be proactive with all actionable feedback