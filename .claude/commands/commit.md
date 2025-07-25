---
allowed-tools: Bash, Read, Edit, Grep
description: Clean up and create atomic git commits
argument-hint: [commit message]
---

# /commit

## Context
- Current branch: !`git branch --show-current`
- Changed files: !`git status --porcelain | wc -l` files
- CLAUDE.md todos: @CLAUDE.md:10-20

## Strategic Guidance
Atomic commits tell a story. Each commit should be a complete, working change that could be reverted independently. Think features, not files. If reverting your commit would break the app, you've split too finely.

## Task
Arguments: "$ARGUMENTS"

Clean up debug code, update docs, and create well-organized commits.

## Process

1. **Safety Check**: Never commit to master. If on master, ask what feature we're working on.

2. **Cleanup**:
   - Remove debug prints/console.logs
   - Delete temp files
   - Update CLAUDE.md (completed TODOs, new utilities)
   - Check for secrets: !`git diff | grep -E '(password|key|token|secret)' || echo "Clean"`

3. **Organize by Feature**:
   ```bash
   git status  # Review all changes
   git diff    # Examine specifics
   ```
   - Group by complete features ("Add auth" = routes + UI + tests)
   - Each commit leaves app working
   - Message format: "verb: description" (50 chars)

4. **Push**: `git push -u origin <branch>` (never force without asking)

## Examples
❌ Three commits: "Add component", "Add styles", "Wire up component"
✅ One commit: "Add user profile modal with avatar upload"

## Retrospective
After committing, reflect on three levels:
1. **Command**: Did this promote good commit hygiene?
2. **Conformance**: Is the guidance strategic without micromanaging?
3. **Meta**: Should commands include more project-specific patterns?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.