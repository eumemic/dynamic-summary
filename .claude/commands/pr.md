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

Get code through CI successfully with minimal back-and-forth. Monitor CI, fix issues immediately, batch fixes to avoid CI churn.

## Workflow

1. **Create PR**: If no PR exists, create one (reference related issues with "Fixes #123")

2. **Monitor CI with fail-fast**: Use `gh pr checks --watch --fail-fast`
   - Exits immediately on first CI failure
   - Fix the failure, commit, and resume monitoring
   - Continues until all checks pass

3. **Request Code Review**: Once implementation is complete and all CI checks pass:
   - Assess complexity of changes and identify areas needing review
   - Post review request: "@claude please review this PR. [specific concerns or focus areas]"
   - Example: "@claude please review this PR. I'm particularly concerned about the error handling in the retry logic and whether the caching approach is thread-safe."
   - Wait for review completion (check comments periodically)
   - Read review feedback and identify issues to address

4. **Request Benchmarks if Needed**: Decide if performance testing is warranted:
   - Check if changes affect performance-critical files:
     - `ragzoom/dynamic_tiling.py` (core algorithm)
     - `ragzoom/index.py` (indexing pipeline)
     - `ragzoom/retrieve.py` (query performance)
     - Config changes affecting defaults
     - Parallel/async processing code
   - If yes, include "/benchmark" in review comment or separate comment
   - Track that benchmarks were requested to avoid duplicate requests
   - When results arrive, assess if regressions are acceptable given PR goals

5. **Handle Review Dialogue**:
   - Discuss review findings with user: "The reviewer identified [issues]. Should I fix [specific issue]?"
   - Fix agreed-upon issues, commit changes
   - Post follow-up to reviewer: "@claude I've addressed [what was fixed]. Regarding [other issue], we're keeping it as-is because [justification]"
   - Continue dialogue until consensus reached
   - Track which issues were addressed vs. intentionally not fixed

6. **Success Criteria**:
   - All CI checks pass
   - Code review requested and feedback addressed
   - Consensus reached with reviewer on all issues
   - Performance benchmarks run if needed, results acceptable
   - No outstanding issues to fix

## Key Principles

- **Fail fast**: `--fail-fast` flag exits on first CI failure for quick fixes
- **Request reviews intelligently**: Only when implementation complete, CI passing
- **Guide the reviewer**: Provide context about areas of concern
- **Request benchmarks selectively**: Only for performance-critical changes
- **Engage in dialogue**: Work with reviewer to reach consensus
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