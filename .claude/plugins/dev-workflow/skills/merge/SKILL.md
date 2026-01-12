---
name: merge
description: This skill should be used when the user asks to "merge", "merge the PR", "squash merge", "merge this PR", or mentions merging a pull request into the main branch.
---

# Merge Pull Request

Squash merge the current PR and sync the branch with master.

## Prerequisites

The merge script handles all validation, but these conditions must be met:
- Not on master branch
- No uncommitted changes
- PR exists and is mergeable
- CI checks are passing

## Process

### 1. Run the Squash Merge Script

```bash
./scripts/squash-merge.sh
```

This script handles:
1. Verifying branch is not master
2. Checking for uncommitted changes
3. Verifying PR exists and is mergeable
4. Checking CI status
5. Squash merging with PR body as commit message
6. Syncing branch with master
7. Pushing to recreate remote worktree branch

### 2. Handle Script Output

**On success:**
```
✅ PR merged successfully
Branch synced with master and ready for next feature
```

**On failure**, the script provides clear error messages:
- "Not on a feature branch" → Switch to the correct branch
- "Uncommitted changes" → Commit or stash changes first
- "No open PR" → Create a PR first with `dev-workflow:pr-create`
- "CI not passing" → Use `dev-workflow:pr-monitor` to fix CI
- "PR not mergeable" → Check for conflicts or required reviews

## What This Skill Does NOT Do

- **Create commits**: Use `dev-workflow:commit` for that
- **Push changes**: Use `dev-workflow:push` for that
- **Create PR**: Use `dev-workflow:pr-create` for that
- **Fix CI**: Use `dev-workflow:pr-monitor` for that

## Related Skills

If merge fails:
- CI not green → "Use `dev-workflow:pr-monitor` to fix CI issues"
- No PR exists → "Use `dev-workflow:pr-create` to create a PR first"
- Uncommitted changes → "Use `dev-workflow:commit` to commit changes first"

## Examples

**Successful merge:**
```
User: "merge this PR"
→ Run squash-merge.sh, report success, branch now synced with master
```

**CI failing:**
```
User: "merge"
→ Script fails with CI error, suggest using pr-monitor to fix
```

**On wrong branch:**
```
User: "merge the PR"
→ Script fails, not on feature branch, ask which branch to switch to
```
