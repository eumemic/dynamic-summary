---
allowed-tools: Bash
description: Merge PR and sync with master
argument-hint: [PR number]
---

# /merge

## Context
- Current branch: !`git branch --show-current`
- PR status: !`gh pr list --head $(git branch --show-current) --state open --json state -q 'if length > 0 then .[0].state else "No PR" end'`
- CI checks: !`gh pr checks --json state --jq 'if length == 0 then "No checks" else map(.state) | unique | join(", ") end' 2>/dev/null || echo "No checks"`

## Strategic Guidance
Merging completes the feature cycle. Use squash merge to maintain a clean commit history on master. For worktree branches, we don't delete the branch - just sync with master for the next cycle.

## Task
Arguments: "$ARGUMENTS"

Merge the current PR and sync with master using the worktree workflow.

## Process
Run the squash merge script:
```bash
./scripts/squash-merge.sh
```

This script will:
1. Verify branch is not master
2. Check for uncommitted changes  
3. Verify PR exists and is mergeable
4. Check CI status
5. Merge with PR body as commit message
6. Sync branch with master
7. Push to recreate remote worktree branch

The script handles all error conditions and provides clear feedback.