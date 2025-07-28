---
allowed-tools: Bash
description: Create GitHub issue from current context
argument-hint: [issue description]
---

# /issue
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Available labels: !`gh label list --json name -q '.[].name' | head -20 | tr '\n' ', ' | sed 's/,$//'`
- Recent issues: !`gh issue list --limit 3 --json number,title -q '.[] | "#" + (.number|tostring) + ": " + .title'`

## Strategic Guidance
Capture issues in the moment of discovery. The best bug report is written while the context is fresh. Include code locations and error messages, but trust future readers to investigate details.

## Task
Arguments: "$ARGUMENTS"

Create a GitHub issue from current context and arguments, using

```
gh issue create --title "..." --body "..." --label "..."
```

## Key Principles

- Include all useful information you know about the issue from the context
- If this was just a quick TODO off the top of the user's head which you otherwise don't know much about, keep it barebones
- If this was something you've been designing extensively together, dump all that hard-won knowledge into the issue body
- Don't speculate, only include details to the extent that they've been worked out

## Retrospective
After creating the issue, reflect on three levels:
1. **Command**: Did this enable quick, effective issue capture?
2. **Conformance**: Is the brevity principle well-balanced?
3. **Meta**: Should commands include more GitHub-specific patterns?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.