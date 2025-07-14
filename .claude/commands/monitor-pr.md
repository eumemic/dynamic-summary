# /monitor-pr

Monitor the current PR for build status changes and new comments, taking action on issues that need to be addressed.

## Workflow Overview

**CRITICAL**: The workflow must follow this sequence:
1. **STOP monitoring** as soon as any failure or actionable issue is detected
2. **FIX all known issues** before resuming monitoring
3. **COMMIT incrementally** as you fix issues (but DO NOT push)
4. **Use built-in todo tracking** to manage all identified issues
5. **RESUME monitoring** only after all issues are fixed
6. **PUSH everything** only after build completes successfully

This minimizes CI churn by avoiding multiple push-triggered builds.

## Instructions:

1. **Initial Check**:
   - Identify the current branch and associated PR number
   - Check CI build status and existing comments
   - If ANY failures or issues exist, immediately proceed to fixing (skip monitoring)

2. **Monitoring Loop** (only if no issues):
   - Check every 30 seconds for:
     - Failed builds → STOP monitoring, fix immediately
     - New review comments → STOP monitoring, address immediately
     - Build completion → Proceed to final push
   - Show "⏳ Monitoring... (no changes)" if nothing new
   - **IMPORTANT**: Always use 30-second intervals. NEVER increase the delay between checks, even for long-running builds

3. **Issue Resolution Mode**:
   - **When you detect ANY issue, STOP monitoring immediately**
   - Use built-in todo tracking to record ALL known issues:
     - CI failures (check logs thoroughly)
     - Review comments requiring fixes
     - Linting/formatting issues
   - Fix ALL issues before returning to monitoring
   - Commit fixes incrementally but DO NOT push yet
   
4. **Fix Priority**:
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

5. **Completion and Push**:
   - Once build completes AND all issues are fixed:
     - Review all commits made during fixes
     - Push once to minimize CI runs
   - If new issues appear after push, repeat the cycle

6. **Stop Conditions**:
   - User explicitly asks to stop
   - PR is merged or closed
   - All builds complete with all issues resolved

## Correct Flow Example:

```
User: /monitor-pr
Assistant: Checking PR #9 status...

❌ Found issues that need fixing:
- CI Build: Failed (Postgres pull error)
- Claude Review: Completed (found docs-lint.py bug)
- Broken import in orchestrator/CLAUDE.md

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