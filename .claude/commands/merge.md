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
- PR status: !`gh pr view --json state,mergeable -q 'if .mergeable == "MERGEABLE" then .state + " / Ready" else .state + " / Not ready" end' 2>/dev/null || echo "No PR"`

## Strategic Guidance
Merging completes the feature cycle. Use regular merge (not squash) to preserve commit history. For worktree branches, we don't delete the branch - just sync with master for the next cycle.

## Task
Arguments: "$ARGUMENTS"

Merge the current PR and sync with master.

## Process

1. **Verify Ready**: Check CI passed, no review blockers
2. **Merge**: `gh pr merge --merge`
3. **Sync with master**: `git pull origin master`
4. **Ready for next PR**: The worktree branch is now synced and ready for the next feature

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