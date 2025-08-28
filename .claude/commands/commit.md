---
allowed-tools: Bash, Read, Edit, Grep, MultiEdit
description: Clean up, commit, push, and update PR if needed
argument-hint: [commit message]
---

# /commit
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Current branch: !`git branch --show-current`
- Changed files: !`git status --porcelain`
- Changes preview: !`git diff --stat`
- PR title: "!`gh pr list --head $(git branch --show-current) --state open --json title -q '.[0].title // "No PR"'`"
- PR description:
!`gh pr list --head $(git branch --show-current) --state open --json body -q '.[0].body // "No PR"'`

## Strategic Guidance
Atomic commits tell a story. Each commit should be a complete, working change that could be reverted independently. Think features, not files. In worktree slots, we work directly on the worktree branch and push after each commit.

## Task
Arguments: "$ARGUMENTS"

Clean up debug code, update docs, and create well-organized commits.

## Process

1. **Safety Check**: Never commit to master.
   - If on master: STOP and ask user to switch to a worktree or feature branch
   - If on worktree-N: This is expected - proceed with commits
   - **NEVER use `--no-verify`** unless given EXPLICIT permission by user

2. **Cleanup**:
   - Remove debug prints/console.logs
   - Delete temp files
   - Check for secrets: !`git diff | grep -E '(password|key|token|secret)' >/dev/null 2>&1 && echo "⚠️ Potential secrets detected!" || echo "Clean"`

3. **Organize by Feature**:
   ```bash
   git status  # Review all changes
   git diff    # Examine specifics
   ```
   - Group by complete features ("Add auth" = routes + UI + tests)
   - Each commit leaves app working
   - Message format: "verb: description" (50 chars)

4. **Consider Before Pushing**:
   - Do you have more changes to make? If yes, hold off on pushing
   - Every push triggers CI runs - batch your pushes to minimize runs
   - Only push when: all foreseeable work is complete, or you need feedback
   
5. **Push**: `git push -u origin <branch>` (never force without asking)
   - For worktree branches, this maintains the sequential PR workflow
   - Remember: batch pushes to avoid unnecessary CI runs

6. **Update PR (if needed)**: Check if PR title/description need updating
   - Only for substantial changes that expand scope
   - Use context from what was just committed
   - Skip for fixes, CI issues, or review feedback
   - Update when: new features, major refactoring, scope expansion
   - Skip when: fixing tests, addressing reviews, minor tweaks
   - Consider updating title if original scope has grown significantly

## Examples

**Commit organization:**
❌ Three commits: "Add component", "Add styles", "Wire up component"
✅ One commit: "Add user profile modal with avatar upload"

**PR updates:**
✅ Update PR after: "feat: add export to CSV functionality" (new feature)
✅ Update PR after: "refactor: extract common logic into shared service" (major change)
❌ Skip update after: "fix: address review feedback on temp files" (review response)
❌ Skip update after: "fix: resolve CI test failures" (CI fix)

**PR title updates:**
✅ Update title: "Fix login bug" → "Fix login bug and add session management"
✅ Update title: "Update docs" → "Update docs and add API examples"
❌ Keep title: "Refactor auth system" (when just fixing tests)

## Retrospective
After committing, reflect on three levels:
1. **Command**: Did this promote good commit hygiene?
2. **Conformance**: Is the guidance strategic without micromanaging?
3. **Meta**: Should commands include more project-specific patterns?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.