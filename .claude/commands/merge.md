---
allowed-tools: Bash
description: Merge PR and clean up branch/worktree
argument-hint: [PR number]
---

# /merge

## Context
- Current branch: !`git branch --show-current`
- PR status: !`gh pr view --json state,statusCheckRollup -q '.state + " / " + .statusCheckRollup.state' 2>/dev/null || echo "No PR"`

## Strategic Guidance
Merging completes the feature cycle. Use regular merge (not squash) to preserve commit history. The branch cleanup is automatic, including any worktrees. This keeps your workspace tidy.

## Task
Arguments: "$ARGUMENTS"

Merge the current PR, clean up branches/worktrees, return to master.

## Process

1. **Verify Ready**: Check CI passed, no review blockers
2. **Merge**: `gh pr merge --merge --delete-branch`
3. **Clean Remote**: `git fetch --prune`
4. **Return Home**:
   ```bash
   WORKTREE=$(git worktree list --porcelain | grep -B2 "branch refs/heads/$(git branch --show-current)" | head -1 | cut -d' ' -f2)
   [ -n "$WORKTREE" ] && cd $(git rev-parse --show-toplevel)
   git checkout master && git pull
   [ -n "$WORKTREE" ] && git worktree remove "$WORKTREE"
   ```

## Error Handling
- No PR found → "Create PR first with /push"
- CI failing → Show failures, stop
- Already on master → "Switch to feature branch first"

## Retrospective
After merging, reflect on three levels:
1. **Command**: Did this handle the full merge workflow smoothly?
2. **Conformance**: Is the worktree cleanup logic clear enough?
3. **Meta**: Should commands assume more git/GitHub knowledge?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.