---
name: pr-monitor
description: This skill should be used when the user asks to "monitor CI", "fix CI", "watch the build", "CI is failing", "fix the build", "check CI status", or mentions CI/build failures that need fixing.
---

# Monitor CI and Fix Failures

Monitor CI status, fix any failures, and loop until all checks pass. When CI goes green, assess whether the changes warrant an independent code review.

## Process

### 1. Initial Status Check

```bash
# Get current PR
gh pr list --head $(git branch --show-current) --state open --json number -q '.[0].number'

# Check CI status
gh pr checks --json name,state,conclusion
```

### 2. Monitor Loop

Poll CI status every 30 seconds until all checks pass:

```bash
gh pr checks --json name,state,conclusion -q '.[] | select(.state != "SUCCESS")'
```

**States to watch for:**
- `IN_PROGRESS` - Still running, keep waiting
- `FAILURE` - Failed, needs fixing
- `SUCCESS` - Passed

**Exit the loop immediately on first failure** to start fixing.

### 3. Fix Failures

When a check fails:

1. **Identify the failure:**
   ```bash
   gh pr checks --json name,state,conclusion,link -q '.[] | select(.conclusion == "FAILURE")'
   ```

2. **Get failure details** from the check's log link or run locally

3. **Common fixes:**
   - Test failures → Fix the test or the code
   - Lint errors → Run linter and fix
   - Type errors → Fix type annotations
   - Build failures → Fix compilation issues

4. **Commit and push the fix:**
   - Use `dev-workflow:commit` to create a fix commit
   - Use `dev-workflow:push` to push the fix

5. **Resume monitoring** - go back to step 2

### 4. All Checks Green

When all CI checks pass:

```
✅ All CI checks passed!
```

**Assess complexity for review:**

Evaluate whether the changes warrant an independent code review:

**Request review (invoke `dev-workflow:pr-review`) if:**
- Multiple files changed with significant logic
- New features or architectural changes
- Security-sensitive code (auth, crypto, permissions)
- Complex algorithms or business logic
- Changes to critical paths

**Skip review if:**
- Documentation-only changes
- Trivial fixes (typos, formatting)
- Test-only changes
- Single-line bug fixes
- Dependency updates (unless major)

When invoking pr-review, explain: "CI is green. These changes are [complex enough to warrant / simple enough to skip] independent review because [reasoning]."

## Key Principles

- **Fail fast**: Exit monitoring loop immediately on failure to fix it
- **Batch fixes**: If multiple issues, fix them all before pushing
- **Minimal commits**: One fix commit is better than many tiny ones
- **Resume monitoring**: After pushing fixes, continue the loop

## What This Skill Does NOT Do

- **Create PR**: Assumes PR exists. Use `dev-workflow:pr-create` first.
- **Merge**: Use `dev-workflow:merge` after review is complete.

## Related Skills

- To commit fixes: Use `dev-workflow:commit`
- To push fixes: Use `dev-workflow:push`
- To request review: Use `dev-workflow:pr-review` (invoked automatically for complex changes)

## Examples

**CI passes immediately:**
```
User: "monitor CI"
→ Check status, all green, assess complexity, either request review or report done
```

**CI fails, then passes:**
```
User: "fix CI"
→ Check status, find test failure, fix test, commit, push, resume monitoring, all green
```

**Multiple failures:**
```
User: "the build is broken"
→ Find lint + test failures, fix both, commit, push, resume, all green
```
