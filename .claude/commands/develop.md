---
allowed-tools: Read, Write, Edit, MultiEdit, Grep, Bash, Task, WebFetch
description: Autonomously develop feature from GitHub issue to PR
argument-hint: <issue-number>
---

# /develop
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Current branch: !`git branch --show-current`
- Open issues: !`gh issue list --limit 5 --json number,title --jq '.[] | "\(.number): \(.title)"' | head -5`

## Strategic Guidance
Autonomous development requires judgment at every step. Start by deeply understanding the problem before writing any code. Use the issue as your communication channel - be transparent about your understanding, approach, and any blockers. Remember: working code that partially solves the problem beats perfect code that never ships.

## Task
Arguments: "$ARGUMENTS"

Autonomously develop a complete solution for GitHub issue #$ARGUMENTS, from requirements clarification through PR creation.

## Process Overview

### 1. Issue Analysis & Clarification
- Fetch issue details and all comments
- Assess if implementation-ready (clear requirements, unambiguous scope, no conflicts)
- If unclear: post clarifying questions and poll for responses with exponential backoff
- Continue until you have a complete understanding
- Do NOT post implementation plans or status updates - only clarifying questions

### 2. Codebase Study
- Identify affected areas from issue description
- Study relevant code systematically (equivalent of `/study <area>`)
- Build mental model of how changes will integrate
- Post architectural questions to issue if needed

### 3. Implementation Planning
- Use extended thinking to design 2-3 approaches
- Choose approach that best fits existing architecture
- Keep planning internal - document in PR body later

### 4. Development
- Implement incrementally (equivalent of `/implement`)
- Run tests frequently to catch issues early
- If blocked: post specific questions to issue and wait
- Ensure all tests pass before proceeding

### 5. Integration
- Commit with descriptive message referencing issue
- Create PR that fixes the issue (document plan and changes in PR body)
- Monitor CI and reviews using equivalent of `/pr`
- Continue until build is green and approved
- Only then merge and close issue

## Polling Strategy
When waiting for responses:
- Initial wait: 30 seconds
- Exponential backoff: 30s → 1m → 2m → 4m → 8m → 15m (max)
- Check for new comments with: `gh issue view <number> --json comments`

## Key Principles
- **Clarity before code**: Never guess requirements
- **Minimal issue comments**: Only post clarifying questions, not status updates
- **Document in PR**: Implementation details go in PR body, not issue comments
- **See it through**: Monitor CI/reviews until merged
- **Fail gracefully**: If truly blocked, explain why and stop
- **Test continuously**: Catch issues early
- **Match the codebase**: Follow existing patterns

## Example Flow
```
/develop 123
→ "Reviewing issue #123..."
→ Posts to issue: "To implement this, I need clarification on..."
→ Polls for response...
→ "Studying authentication system..."
→ Implements solution
→ "Running tests..."
→ Creates PR with detailed body
→ Monitors CI: "Build failed, fixing..."
→ Pushes fixes
→ "CI green, awaiting review..."
→ Addresses review feedback
→ Merges when approved
```

## Retrospective
After completing development, reflect on three levels:
1. **Command**: How could this command better support autonomous development?
2. **Conformance**: Does it balance autonomy with appropriate human oversight?
3. **Meta**: What patterns emerged that could improve other commands?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.