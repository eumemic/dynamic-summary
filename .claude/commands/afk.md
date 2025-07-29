---
allowed-tools: Bash, Read, Edit, MultiEdit, Write, Grep, Glob, TodoWrite
description: Infinite AFK mode for mobile dev via PR comments
argument-hint: [pr-number] [initial-task]
---

# /afk
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Current PR: !`gh pr view --json number -q .number 2>/dev/null || echo ""`
- Branch: !`git branch --show-current`

## Strategic Guidance
You're entering an infinite loop where terminal communication is replaced by PR comments. NEVER wait for terminal input - that breaks the mobile workflow. Replace your "final answer" terminal responses with PR comments. Check for new instructions periodically, even mid-task. The user will interrupt you when back at keyboard, at which point normal mode will resume.

Think of this as running a modified `.claude/commands/pr.md` that also responds to user comments and never exits. You're monitoring TWO things simultaneously: CI/reviews AND user instructions.

## Task
Arguments: "$ARGUMENTS"

Setup phase: 
1. **CRITICAL**: If you made ANY code changes, commit and push them immediately!
2. If mid-task, finish it first
3. Commit any outstanding changes (follow `.claude/commands/commit.md`)
4. Ensure PR exists (follow `.claude/commands/pr.md` if needed)
5. Post "🤖 Entering AFK mode" to PR

Then enter dual monitoring loop:
1. Monitor CI/reviews like pr.md - post updates about failing checks, review feedback
2. Monitor user comments - execute instructions, post results
3. **ALWAYS commit and push after making any code changes!**
4. Polling intervals:
   - If monitoring active CI/reviews: Fixed 30s intervals
   - If idle: Exponential backoff (30s * 1.5^n, capped at 300s/5min)
     - Example: 30s → 45s → 68s → 102s → 153s → 230s → 300s (max)
   - **Reset to 30s whenever user comments**
5. Always check for user comments during any poll

Never stop until interrupted. Post "final answers" only as PR comments, not intermediate thoughts.

## Comment Detection
Since both your comments and user comments appear from the same GitHub account, use content patterns to distinguish:

**Your comments contain**: Status emojis (🤖, 📝, ✅, ⚠️, ⏳, 🔄), formal CI/test updates, content you just posted

**User comments**: Everything else - natural language, questions, observations, any non-status content

**Get recent comments**: `gh pr view PR_NUMBER --comments | tail -30`

Track last processed comment to identify new ones.

## Communication Protocol
- 🤖 Started → 📝 Working → ✅/⚠️ Done
- ⏳ Polling → 🔄 Progress updates
- Post PR comments for ALL terminal output

## Retrospective
After completing this task, reflect on three levels:
1. **Command Improvement**: How could this specific command guide future agents better?
2. **Rubric Conformance**: Does this command follow the /command design principles well?
3. **Meta Evolution**: Should the /command rubric itself evolve based on your experience?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.