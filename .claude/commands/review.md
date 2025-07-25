# /review

Arguments: "$ARGUMENTS"

Review code changes with focus on architecture, correctness, and maintainability.

## Scope

Use arguments if provided, otherwise:
- If working directory dirty → uncommitted changes
- If on feature branch → diff against master
- Otherwise → complete repository state

## Core Intent

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