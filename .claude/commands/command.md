# /command

Arguments: "$ARGUMENTS"

Create a new custom Claude command following best practices for clarity, conciseness, and effectiveness.

## Command Design Principles

1. **Tight & Information-Dense**: Every word should add value. No fluff or repetitive instructions.
2. **Trust Agent Intelligence**: Don't micromanage. State intent clearly and let the agent think creatively.
3. **Crystal Clear Intent**: The user's goal and desired outcome must be unmistakable.
4. **Action-Oriented**: Commands should drive specific, valuable actions.
5. **Context-Aware**: Use conversation history and codebase state intelligently.

## Process

1. **Parse Request**: Extract command name and purpose from arguments (if provided) or infer from conversation context
2. **Design Command**: 
   - Single clear purpose
   - Minimal required context ($ARGUMENTS pattern)
   - Specific, measurable outcome
3. **Write Command File**:
   - Location: `.claude/commands/{name}.md`
   - Structure: Purpose → Process → Principles → Examples
   - Length: Aim for <100 lines unless complexity demands more
4. **Test & Refine**: Run the command to ensure it works as intended

## Command Template

```markdown
# /{name}

Arguments: "$ARGUMENTS"

{One-line purpose statement. What problem does this solve?}

## Core Intent

{2-3 sentences on why this command exists and when to use it}

## Process

{Numbered steps, each one clear and actionable}

## Key Principles

{Bullet points of crucial guidelines, not obvious things}

## Examples

{1-2 concise examples showing input → output}

{Optional: One-line reminder of the key value prop}
```

## Examples

Input: "command for finding dead code"
→ Creates `/deadcode` command that searches for unused functions, imports, and files

Input: "command benchmark to run performance tests"  
→ Creates `/benchmark` command that profiles critical paths and compares against baselines

Remember: Great commands are tools that developers reach for repeatedly. Make them sharp, focused, and reliable.