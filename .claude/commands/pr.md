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
- Existing PR: !`gh pr list --head $(git branch --show-current) --state open --json number,state -q 'if length > 0 then "#" + (.[0].number|tostring) + " (OPEN)" else "No PR" end'`

Arguments: "$ARGUMENTS"

Create PR if needed, monitor CI, and fix issues proactively. Assumes code is already pushed by /commit.

## Core Intent

Get code through CI successfully and request reviews intelligently. Fix CI issues immediately, ensure clean state before reviews, batch changes to minimize CI runs.

## Workflow

### Phase 1: Initial Setup & CI
1. **Create PR**: If no PR exists, create one (reference related issues with "Fixes #123")

2. **Fix CI Issues**: Poll for CI status and fix any failures
   - Check status every 30 seconds with `gh pr checks`
   - Exit immediately on first failure to fix it
   - Commit fixes, push, resume monitoring
   - Continue until all checks pass

### Phase 2: Complete Implementation
3. **Ensure Work is Complete**:
   - Review PR objectives - is implementation complete?
   - Check if any known work remains (TODOs, planned features)
   - If work remains, complete it before proceeding
   - Once truly done, move to clean state verification

4. **Verify Clean State**:
   - Check for uncommitted changes: `git status`
   - If changes exist, commit them
   - Check if commits need pushing: `git status`
   - If unpushed commits exist, push them
   - Monitor CI again until all checks pass

### Phase 3: Request Reviews (Only When Ready)
5. **Request Code Review** (only after clean state + CI passing):
   - Review ALL accumulated changes since PR creation
   - Identify areas of complexity or concern
   - Post review request: "@claude please review this PR. [specific concerns]"
   - Example: "@claude please review this PR. Key changes include [summary]. I'm particularly concerned about [specific areas]."

6. **Request Benchmarks if Needed**:
   - Check if ANY changes (accumulated) affect:
     - `ragzoom/dynamic_tiling.py` (core algorithm)
     - `ragzoom/index.py` (indexing pipeline)
     - `ragzoom/retrieve.py` (query performance)
     - Config defaults, parallel/async code
   - If yes, include "/benchmark" in comment
   - Track that benchmarks were requested

### Phase 4: Handle Feedback
7. **Review Dialogue**:
   - Read review feedback when it arrives
   - Discuss with user: "The reviewer identified [issues]. Should I fix [X]?"
   - Fix agreed issues, commit, push
   - Post follow-up: "@claude I've addressed [what]. Regarding [other issue], keeping as-is because [reason]"
   - Continue until consensus reached

8. **Success Criteria**:
   - All CI checks pass
   - No more work to do
   - Clean state (everything committed and pushed)
   - Reviews addressed and consensus reached
   - Benchmarks acceptable (if run)

## Key Principles

- **Complete work first**: Finish all known tasks before requesting reviews
- **Clean state required**: Everything committed and pushed before reviews
- **CI must pass**: Don't request reviews with failing CI
- **Review accumulated changes**: Consider ALL changes when requesting review, not just latest
- **Batch operations**: Complete all work, then push once, minimizing CI runs
- **Fail fast on CI**: Exit immediately on failures to fix them
- **Guide the reviewer**: Provide context about areas of concern
- **Request benchmarks selectively**: Only for performance-critical changes
- **Engage in dialogue**: Work with reviewer to reach consensus

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

Monitoring CI...
❌ Build failed: missing import

Fixing import issue...
✅ Fixed and ready to commit
(User runs /commit)

Resuming CI monitoring...
✅ All CI checks passed!

Implementation complete. Requesting code review...
Posting: "@claude please review this PR. I'm particularly concerned about
the new retry logic in index.py and whether the semaphore usage is correct."

Also noticing changes to ragzoom/dynamic_tiling.py - requesting benchmarks...
Including "/benchmark" in comment.

Waiting for review completion...
📝 Review complete! Found feedback:
- Critical: Potential race condition in retry logic
- Minor: Consider extracting magic number to constant

Discussing with user: "The reviewer found a race condition in the retry logic.
Should I add a lock to fix this?"

User: "Yes, fix the race condition. The magic number is fine as-is."

Fixing race condition...
✅ Fixed and ready to commit
(User runs /commit)

Posting follow-up: "@claude I've addressed the race condition by adding
a lock around the critical section. The magic number is intentional for
consistency with existing patterns."

Reviewer responds: "✅ Changes look good, race condition resolved."

Checking benchmark results...
📊 Performance Report Analysis:
- Median processing time: 2.8s (under 3s threshold ✅)
- Cost per 1M tokens: $0.95 (under $1 threshold ✅)
✅ Performance within acceptable bounds

✅ All review comments addressed
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