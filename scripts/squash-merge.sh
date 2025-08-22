#!/bin/bash
set -euo pipefail

# Squash merge PR and sync worktree with master
# Usage: ./scripts/squash-merge.sh

CURRENT_BRANCH=$(git branch --show-current)

# Safety check: Don't run on master or detached HEAD
if [ "$CURRENT_BRANCH" = "master" ] || [ "$CURRENT_BRANCH" = "main" ] || [ -z "$CURRENT_BRANCH" ]; then
    if [ -z "$CURRENT_BRANCH" ]; then
        echo "❌ Cannot merge from detached HEAD state"
    else
        echo "❌ Cannot merge from master/main branch"
    fi
    echo "Switch to a worktree or feature branch first"
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD 2>/dev/null; then
    echo "❌ Uncommitted changes detected - commit or stash them first"
    exit 1
fi

# Check if PR exists and get its data
echo "🔍 Checking PR status for branch $CURRENT_BRANCH..."
PR_DATA=$(gh pr list --head "$CURRENT_BRANCH" --state open --json number,mergeable,body)

if [ "$(echo "$PR_DATA" | jq length)" -eq 0 ]; then
    echo "❌ No open PR found for branch $CURRENT_BRANCH"
    echo "Create a PR first with: gh pr create"
    exit 1
fi

# Check if PR is mergeable
MERGEABLE=$(echo "$PR_DATA" | jq -r '.[0].mergeable')
if [ "$MERGEABLE" != "MERGEABLE" ]; then
    echo "❌ PR is not ready to merge (status: $MERGEABLE)"
    echo "Check CI status and resolve any conflicts"
    exit 1
fi

# Check CI status
echo "🔍 Checking CI status..."
CI_CHECKS=$(gh pr checks --json state)
if [ "$(echo "$CI_CHECKS" | jq length)" -gt 0 ]; then
    FAILED_CHECKS=$(echo "$CI_CHECKS" | jq -r '.[] | select(.state == "failure") | .state' | wc -l)
    PENDING_CHECKS=$(echo "$CI_CHECKS" | jq -r '.[] | select(.state != "success" and .state != "failure") | .state' | wc -l)
    
    if [ "$FAILED_CHECKS" -gt 0 ]; then
        echo "❌ CI checks failed - fix issues before merging"
        gh pr checks
        exit 1
    fi
    
    if [ "$PENDING_CHECKS" -gt 0 ]; then
        echo "❌ CI checks still running - wait for completion before merging"
        gh pr checks
        exit 1
    fi
fi

# Get PR body for commit message
PR_BODY=$(echo "$PR_DATA" | jq -r '.[0].body')

echo "✅ All checks passed. Merging PR for branch $CURRENT_BRANCH..."

# Merge with custom commit message (GitHub auto-deletes remote branch)
if ! gh pr merge --squash --body "$PR_BODY"; then
    echo "❌ Failed to merge PR"
    exit 1
fi

echo "🔄 Syncing $CURRENT_BRANCH with master..."

# Sync with master
git fetch origin
git reset --hard origin/master

# Recreate remote worktree branch
git push -u origin "$CURRENT_BRANCH"

echo "✅ Successfully merged and synced!"
echo "Branch $CURRENT_BRANCH is ready for the next PR"