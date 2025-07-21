# /merge

Merge the current PR, delete the branch, and return to master with the latest changes.

## Instructions:

1. **Identify Current PR**:
   - Get the current branch name
   - Find the associated PR number
   - Verify the PR exists and is open

2. **Check PR Status**:
   - Ensure all CI checks have passed
   - Verify there are no requested changes or unresolved review comments
   - If any checks are failing or issues exist, stop and report to user

3. **Merge the PR**:
   - Use `gh pr merge` with the `--merge` strategy (not squash or rebase)
   - Include the `--delete-branch` flag to clean up the branch
   - This will merge the PR and delete both local and remote branches
   - Run `git fetch --prune` immediately to clean up any stale remote branch references

4. **Return to Master**:
   - Switch to master branch
   - Pull the latest changes

5. **Confirm Success**:
   - Show the user the merge commit
   - Confirm we're on master and up to date

## Example Flow:

```bash
# Get current branch and PR
git branch --show-current
gh pr list --head <branch> --json number

# Check PR status
gh pr checks <PR#>

# Merge and cleanup
gh pr merge <PR#> --merge --delete-branch
git fetch --prune

# Return to master
git checkout master
git pull

# Confirm
git log --oneline -5
```

## Error Handling:

- If no PR is found for the current branch, inform the user
- If CI checks are failing, stop and show the failing checks
- If merge conflicts exist, inform the user they need to be resolved first
- If the user is already on master, inform them they need to be on a feature branch

## Notes:

- This command assumes the user wants a regular merge (not squash or rebase)
- The `--delete-branch` flag in `gh pr merge` handles both local and remote branch deletion
- The `git fetch --prune` ensures any stale remote references are cleaned up