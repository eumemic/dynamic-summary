---
allowed-tools: Read, Grep, Bash, Task
description: Review code for architecture, correctness, and maintainability
argument-hint: [file/PR/commit]
---

# /review
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Current branch: !`git branch --show-current`
- Changed files: !`git diff --name-status HEAD~ 2>/dev/null | head -10 || git status --porcelain | head -10`

Arguments: "$ARGUMENTS"

Review code changes with focus on architecture, correctness, and maintainability.

## Strategic Guidance

Look beyond syntax to question design decisions. Think carefully about the architecture - trace data flows, understand component relationships. Is this the simplest solution? Could we achieve the same with less code? What would a new developer think?

## Review Focus

1. **Architecture First**
   - Why does each new component exist?
   - Could services/functions be combined or eliminated?
   - Is the data flow unnecessarily complex?
   - Are we solving the actual problem or one we created?

2. **Code Quality**
   - Consistency with existing patterns
   - Dead code (unused functions, imports, variables)
   - Error handling and edge cases
   - Security vulnerabilities
   - Performance issues

3. **Maintainability**
   - Clear naming and purpose
   - Appropriate documentation
   - Test coverage
   - Configuration management

## Key Questions

- What simpler alternative did we reject and why?
- Could this functionality use existing code?
- What's the ratio of boilerplate to business logic?
- Are there multiple ways to do the same thing?

## Output Format

**🏗️ Architecture Issues**
[Big picture problems, unnecessary complexity]

**✅ Good Practices**
[Positive patterns to acknowledge]

**⚠️ Code Issues**
- Critical: [Must fix - bugs, security]
- Important: [Should fix - consistency, docs]
- Minor: [Nice to fix - style]

**🎯 Summary**
[Overall assessment and key actions]

Remember: Great code is simple code. Question every abstraction.

## Retrospective
After reviewing, reflect on three levels:
1. **Command**: Did this promote architecture-first thinking?
2. **Conformance**: Is the output format helpful without being rigid?
3. **Meta**: Should commands include more emphasis on simplicity metrics?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.