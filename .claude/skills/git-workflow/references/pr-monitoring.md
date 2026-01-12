# PR Monitor Reference

## Initial Status Check

```bash
# Get current PR
gh pr list --head $(git branch --show-current) --state open --json number -q '.[0].number'

# Check CI status
gh pr checks --json name,state,link
```

## Poll Until Complete

Poll CI status until all checks finish:

```bash
# Loop until no pending checks remain
while gh pr checks --json state -q '.[] | select(.state == "PENDING")' | grep -q PENDING; do
  sleep 10
done

# Check final status
gh pr checks --json name,state,link
```

**States:**
- `PENDING` - Still running, keep waiting
- `FAILURE` - Failed, needs fixing
- `SUCCESS` - Passed

**Note:** Avoid `--watch` flag as it produces excessive output.

## Diagnosing Failures

Identify what failed:
```bash
gh pr checks --json name,state,link -q '.[] | select(.state == "FAILURE")'
```

Get details from check's log link or run locally.

## Common CI Fixes

**Test failures:**
- Run tests locally to reproduce
- Fix the test or the code causing failure
- Verify fix locally before pushing

**Lint errors:**
- Run linter locally: `ruff check .` or `eslint .`
- Apply auto-fixes if available
- Manual fixes for remaining issues

**Type errors:**
- Run type checker: `mypy .` or `tsc`
- Fix type annotations
- Update type stubs if needed

**Build failures:**
- Check compilation output
- Fix import/dependency issues
- Verify build works locally

## Fix Workflow

1. Identify the failure from CI output
2. Reproduce locally if possible
3. Implement the fix
4. Use commit operation to create fix commit
5. Use push operation to push
6. Resume monitoring - go back to polling

## Batch Fixes

If multiple issues:
- Fix them all before pushing
- One fix commit is better than many tiny ones
- Commit message should summarize all fixes

## All Checks Green

When CI passes:

```
All CI checks passed!
```

**Assess complexity for review:**

**Request review if:**
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

## Examples

**CI passes immediately:**
```
User: "monitor CI"
→ Check status, all green, assess complexity, request review or report done
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
