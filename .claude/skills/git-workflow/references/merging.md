# Merge Reference

## Prerequisites

The merge script validates these conditions:
- Not on master branch
- No uncommitted changes
- PR exists and is mergeable
- CI checks are passing

## Run the Script

```bash
./scripts/squash-merge.sh
```

## What the Script Does

1. Verify branch is not master
2. Check for uncommitted changes
3. Verify PR exists and is mergeable
4. Check CI status
5. Squash merge with PR body as commit message
6. Sync branch with master
7. Push to recreate remote worktree branch

## Success Output

```
PR merged successfully
Branch synced with master and ready for next feature
```

## Error Handling

**"Not on a feature branch"**
- Switch to the correct branch
- `git checkout feature-branch`

**"Uncommitted changes"**
- Commit changes first using commit operation
- Or stash: `git stash`

**"No open PR"**
- Create PR first using pr-create operation

**"CI not passing"**
- Use pr-monitor operation to fix CI
- Wait for all checks to pass

**"PR not mergeable"**
- Check for merge conflicts: `git fetch origin && git merge origin/master`
- Check for required reviews: `gh pr view --json reviewDecision`
- Resolve conflicts or get required approvals

## Manual Merge (If Script Unavailable)

```bash
# Verify prerequisites
git branch --show-current  # Not master
git status  # No uncommitted changes
gh pr checks  # All passing
gh pr view --json mergeable  # Is mergeable

# Squash merge
gh pr merge --squash

# Sync branch
git fetch origin master
git reset --hard origin/master
git push -f origin $(git branch --show-current)
```

**Note:** Manual merge should be avoided when script is available.

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

**Uncommitted changes:**
```
User: "merge"
→ Script fails, suggest committing or stashing changes first
```
