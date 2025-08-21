---
allowed-tools: Bash
description: Merge PR and sync with master
argument-hint: [PR number]
---

# /merge
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Current branch: !`git branch --show-current`
- PR status: !`gh pr list --head $(git branch --show-current) --state open --json mergeable -q 'if length > 0 then if .[0].mergeable == "MERGEABLE" then "OPEN / Ready" else "OPEN / Not ready" end else "No PR" end'`
- CI status: !`gh pr checks --json conclusion --jq 'if length == 0 then "No checks" else "See details below" end'`
- Uncommitted changes: !`git diff-index --quiet HEAD && echo "None" || echo "Present - will be lost!"`

## Strategic Guidance
Merging completes the feature cycle. Use squash merge to maintain a clean commit history on master. For worktree branches, we don't delete the branch - just sync with master for the next cycle.

## Task
Arguments: "$ARGUMENTS"

Merge the current PR and sync with master.

## Process

1. **Verify Ready**: If CI failing or uncommitted changes present (see **Context** section above), stop and inform user
2. **Merge**: `gh pr merge --squash` (GitHub auto-deletes the remote branch)
3. **Sync with master**: `git fetch origin && git reset --hard origin/master`
4. **Push to recreate remote**: `git push -u origin $(git branch --show-current)` (recreate remote worktree branch)
5. **Ready for next PR**: The worktree branch is now synced and ready for the next feature

## Error Handling
- No PR found → "Create PR first with /pr"
- CI failing → Show failures, stop
- Already on master → "Switch to worktree or feature branch first"

## Retrospective
After merging, reflect on three levels:
1. **Command**: Did this handle the full merge workflow smoothly?
2. **Conformance**: Is the merge process clear enough?
3. **Meta**: Should commands assume more git/GitHub knowledge?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.