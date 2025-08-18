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
Merging completes the feature cycle. Use squash merge to maintain a clean commit history on master. For worktree branches, we don't delete the branch - just sync with master for the next cycle.

## Task
Arguments: "$ARGUMENTS"

Merge the current PR and sync with master.

## Process

1. **Verify Ready**: Check CI passed, no review blockers
2. **Check for uncommitted changes**: Ensure no work will be lost
   ```bash
   git diff-index --quiet HEAD || {
       echo "⚠️ Warning: You have uncommitted changes that will be lost!"
       echo "Commit or stash them before merging."
       exit 1
   }
   ```
3. **Merge**: `gh pr merge --squash` (GitHub auto-deletes the remote branch)
4. **Sync with master**: `git fetch origin && git reset --hard origin/master`
5. **Push to recreate remote**: `git push -u origin <branch>` (recreate remote worktree branch)
6. **Ready for next PR**: The worktree branch is now synced and ready for the next feature

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