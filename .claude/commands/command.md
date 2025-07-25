---
allowed-tools: Write
description: Create a new custom Claude command following best practices
argument-hint: <command-name> <purpose>
---

# /command

Arguments: "$ARGUMENTS"

Create a new custom Claude command. Use extended thinking to anticipate what the executing agent will need to succeed.

## Design Principles

1. **YAML Frontmatter**: Security and UX through tool restrictions and metadata
2. **Dynamic Context**: Use `!` for bash output, `@` for file contents  
3. **Strategic Preparation**: Do the hard thinking upfront about approach and necessary context
4. **Guided Autonomy**: Provide mental models and key insights, not step-by-step instructions
5. **Information Density**: Be concise. Every line costs tokens. Pack maximum insight into minimum words
6. **Argument Hints**: Use `[brackets]` for optional arguments, `<angles>` for required
7. **Continuous Improvement**: Include retrospective to evolve commands based on usage

## Key Questions to Consider

When designing a command, think deeply about:
- What mental model or framework helps approach this problem?
- What context is crucial to gather before starting?
- What constraints or invariants must be maintained?
- Where should the agent pause and think carefully?
- What does success look like?
- How can I convey this in the fewest possible words?

## Command Template

```markdown
---
allowed-tools: [Specify only needed tools]
description: [Brief description for autocomplete]
argument-hint: [optional args] or <required args>
---

# /command-name
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.


## Context
[Dynamic context using ! and @ prefixes]
- Current state: !`relevant command`
- Key files: @important/file.md

## Strategic Guidance
[Key insights that prepare the agent for success. Think: what would you tell someone about to tackle this problem? What approach works best? What should they understand first?]

## Task
Arguments: "$ARGUMENTS"

[Clear outcome and essential constraints. Trust the agent to figure out the how.]

## Retrospective
After completing this task, reflect on three levels:
1. **Command Improvement**: How could this specific command guide future agents better?
2. **Rubric Conformance**: Does this command follow the /command design principles well?
3. **Meta Evolution**: Should the /command rubric itself evolve based on your experience?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.
```

## Example

Input: "refactor-module clean up and modernize old code"
→ Creates:
```markdown
---
allowed-tools: Read, Write, Edit, Grep, Bash(npm test:*)
description: Refactor module to modern patterns
argument-hint: <module name>
---

# /refactor-module
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Module structure: !`find $1 -name "*.js" -o -name "*.ts" | head -20`
- Test coverage: !`npm test -- --coverage $1 2>&1 | grep -A5 "File.*%"`
- Dependencies: @package.json

## Strategic Guidance
Before refactoring, understand the module's public API and how it's used throughout the codebase. Maintain backward compatibility unless explicitly breaking. Modern patterns are good, but consistency with the codebase is better. Run tests frequently - they're your safety net.

## Task
Arguments: "$ARGUMENTS"

Modernize the specified module while maintaining its behavior. Focus on clarity, testability, and alignment with project patterns. Think carefully about the module's role before making structural changes.

## Retrospective
After refactoring, reflect on three levels:
1. **Command**: Could this command better prepare agents for common refactoring challenges?
2. **Conformance**: Does it provide enough strategic guidance while maintaining autonomy?
3. **Meta**: Did you discover patterns that suggest new design principles for commands?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.
```

Remember: This three-level reflection creates evolutionary pressure at both tactical (command) and strategic (design philosophy) levels.