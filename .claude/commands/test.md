---
allowed-tools: Bash, Read, Write, Edit
description: Thoroughly test features and clean up artifacts
argument-hint: [feature to test]
---

# /test

## Context
- Test commands: !`grep -E "^(test|check):" Makefile package.json pyproject.toml 2>/dev/null | head -5 || echo "No standard test commands found"`
- Recent test files: !`find . -name "*test*" -type f -mtime -7 | grep -E "\.(py|js|ts)$" | head -5`

Arguments: "$ARGUMENTS"

Thoroughly test the specified feature/fix through both automated and manual testing. Clean up all test artifacts before reporting.

## Core Intent

Verify functionality works correctly across happy paths, edge cases, and error scenarios. Think carefully about what could go wrong. Think like a user trying to break the system.

## Process

1. **Plan**: Identify what to test based on arguments or recent changes
2. **Execute**: Run tests systematically, trying to break things
3. **Clean Up**: Remove ALL test artifacts (files, data, logs)
4. **Report**: Concise summary with go/no-go recommendation

## Testing Toolkit

- **Automated**: Run existing test suites
- **Manual**: UI interactions, API calls, data verification
- **Exploratory**: Try unexpected inputs and workflows
- **Performance**: Basic load and response time checks
- **Integration**: Verify component interactions

## Key Scenarios

- Happy paths (normal usage)
- Edge cases (boundaries, limits)
- Error cases (invalid input, missing data)
- Concurrent operations
- State transitions

## Report Format

**Test Target**: [Feature/component name]
**Results**: X/Y scenarios passed
**Critical Issues**: [Blocking problems, if any]
**Recommendation**: [Go/No-go with reasoning]

Notable findings:
- [Key observation or issue]
- [Another finding]

## Examples

Arguments: "file upload with large files"
→ Test various file sizes, concurrent uploads, network interruptions, file type validation

Arguments: (none, after implementing auth)
→ Test login/logout, invalid credentials, session expiry, concurrent sessions, permission checks

Remember: Test destructively but clean up completely. The goal is finding issues, not fixing them.

## Retrospective
After testing, reflect on three levels:
1. **Command**: Did this help find issues systematically?
2. **Conformance**: Is the cleanup emphasis strong enough?
3. **Meta**: Should commands include more specific test scenario templates?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.