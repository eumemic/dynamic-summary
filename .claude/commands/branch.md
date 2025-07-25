---
allowed-tools: Bash, Read
description: Create git worktree and branch for feature development
argument-hint: [feature name]
---

# /branch

## Context
- Current branch: !`git branch --show-current`
- Existing worktrees: !`git worktree list | wc -l` worktrees

## Strategic Guidance
Worktrees isolate features without stashing or context switching. Each feature gets its own directory, preserving your working state. Think of them as parallel universes for your code.

## Task
Arguments: "$ARGUMENTS"

Create a worktree for the specified feature (or infer from context). Conventions: feature/, fix/, refactor/, docs/.

## Process
1. **Name**: Generate from args or context (lowercase, hyphens)
2. **Confirm**: "Create worktree at worktrees/X with branch Y?"
3. **Execute**: 
   ```bash
   git checkout master && git pull
   git worktree add worktrees/[name] -b [branch]
   cd worktrees/[name]
   [ -f ../../.env ] && cp ../../.env .
   ```

## Examples
- "auth system" → feature/auth-system
- "memory leak" → fix/memory-leak  
- (no args) → Infer from conversation

## Retrospective
After creating the worktree, reflect on three levels:
1. **Command**: Did this guide worktree creation smoothly?
2. **Conformance**: Is the process clear without being prescriptive?
3. **Meta**: Should commands include more git workflow patterns?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.