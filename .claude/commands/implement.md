# /implement

Task: $ARGUMENTS

_Note: if no task is specified above, assume it's whatever task is apparent from your recent context._

Systematically implement the requested feature or bugfix following the disciplined approach of a master developer. Think deeply at each step, prioritizing correctness, maintainability, and long-term code health. The following advice is assuming a maximally complex feature implementation; you can judiciously simplify your process as appropriate for the size and complexity of the task.

## Phase 0: Initial Questions

Before beginning any work, consider if you need clarification on any aspects of the task. If anything is unclear, **STOP** and ask the user:
- Ambiguous requirements or specifications
- Unclear scope boundaries
- Missing context about user needs or use cases
- Technical constraints or preferences
- Integration points or dependencies
- Performance or scale requirements
- Timeline or priority considerations

If everything is clear and you have all the information needed, proceed to Phase 1. It's better to ask questions upfront than to make incorrect assumptions that lead to rework.

## Phase 1: Understanding & Planning

1. **Clarify the requirements**:
   - What exactly needs to be built/fixed?
   - What are the acceptance criteria?
   - Who are the users/consumers of this code?
   - What are the non-functional requirements (performance, security, etc.)?

2. **Thoroughly understand the existing code**:
   - Read and comprehend ALL files in the area you'll be modifying, as well as components that depend on them
   - Trace through the current execution flow
   - Understand the data structures and their relationships
   - Identify implicit assumptions and invariants
   - Map out dependencies and consumers of the code
   - Note existing patterns, conventions, and architectural decisions

3. **Research the codebase**:
   - Search for similar implementations elsewhere
   - Check for utilities or helpers you should reuse
   - Look for related issues or previous attempts
   - Understand why the current code is structured as it is

4. **Design the solution**:
   - Consider at least 2-3 different approaches
   - Evaluate trade-offs (complexity, performance, maintainability)
   - Choose the approach that best fits the existing architecture
   - Identify potential edge cases and error scenarios
   - Consider how your changes will affect other parts of the system

5. **Create implementation plan**:
   - Use TodoWrite to create a detailed task list
   - Break down into small, testable increments
   - Order tasks by dependencies
   - Include testing and documentation tasks
   - Plan for incremental delivery of value

## Phase 2: Test-Driven Development

1. **Write tests FIRST** (when possible):
   - Start with the simplest test case
   - Write tests that currently fail
   - Include edge cases and error scenarios
   - Consider unit, integration, and e2e tests as appropriate
   - At a minimum, perform manual testing:
      - Use `Bash("claude -p '/test [thorough feature/fix description]'")` for agentic testing
      - If required, ask the user to test things for you

2. **Test checklist**:
   - [ ] Happy path scenarios
   - [ ] Error cases and exceptions
   - [ ] Boundary conditions
   - [ ] Performance considerations (if relevant)
   - [ ] Security scenarios (if relevant)

3. **Make tests run in CI**:
   - Ensure tests are discoverable by test runners
   - Add to appropriate test suites
   - Verify CI configuration includes new tests

## Phase 3: Implementation

1. **Start simple**:
   - Implement the minimal working version first
   - Make the tests pass
   - Get feedback early on your approach

2. **Follow codebase conventions**:
   - Match existing code style exactly
   - Use established patterns and libraries
   - Maintain consistency with surrounding code
   - Respect existing abstractions

3. **Iterate and refine**:
   - Refactor for clarity and maintainability
   - Extract common functionality
   - Remove duplication
   - Optimize only when necessary and measured

4. **Add appropriate instrumentation**:
   - Logging for debugging (but not excessive)
   - Metrics/telemetry if applicable
   - Error tracking integration

## Phase 4: Quality Checkpoints

### Before considering any piece of work complete:
1. **Run all affected tests**: Ensure nothing is broken
2. **Manual testing**: Verify the behavior works as expected
3. **Code self-review**: Read your changes as if reviewing someone else's code
4. **Clean up**: Remove debug statements, commented code, TODOs

### Consultation triggers (STOP and ask the user):
- **Spec ambiguity**: Requirements are unclear or contradictory
- **Major obstacles**: Technical blockers that require architectural changes
- **Scope creep**: Implementation revealing much larger changes needed
- **Breaking changes**: Existing behavior would be altered
- **Performance concerns**: Solution may degrade performance significantly
- **Security implications**: Changes affect authentication, authorization, or data protection
- **Architecture questions**: Unsure about the right pattern or approach
- **External dependencies**: Need to add new libraries or services

### If stuck on a specific technical problem:
```
Bash(claude -p "/diagnose [detailed problem description including all context needed]")
```

## Phase 5: Documentation & Polish

1. **Update documentation**:
   - Add/update relevant CLAUDE.md files
   - Document new APIs or configuration options
   - Include usage examples if helpful
   - Update architecture docs if structure changed

2. **Code documentation**:
   - Add docstrings for public APIs
   - Comment complex algorithms (but prefer self-documenting code)
   - Document any non-obvious decisions

3. **Final cleanup**:
   - Ensure consistent formatting
   - Remove any temporary code
   - Verify no sensitive data in code
   - Check for any remaining TODOs

## Phase 6: Completion Checklist

Before considering the implementation complete:

- [ ] All tests passing (unit, integration, e2e)
- [ ] Manual testing completed successfully
- [ ] Edge cases handled appropriately
- [ ] Error messages are helpful and actionable
- [ ] Documentation updated
- [ ] Code follows established patterns
- [ ] No performance regressions
- [ ] Security considerations addressed
- [ ] TodoWrite list is fully completed
- [ ] Code is ready for review

## Implementation Best Practices

- **Incremental progress**: Small, working changes over large, broken ones
- **Fail fast**: Validate assumptions early
- **YAGNI**: Don't add functionality "just in case"
- **DRY**: Don't repeat yourself, but don't over-abstract
- **Explicit over implicit**: Clear code over clever code
- **Test the contract**: Test behavior, not implementation details
- **Leave it better**: Improve surrounding code when touching it

## Common Pitfalls to Avoid

1. **Skipping tests**: "I'll add tests later" rarely happens
2. **Over-engineering**: Building for imaginary future requirements
3. **Under-engineering**: Ignoring obvious extension points
4. **Inconsistent style**: Not matching existing patterns
5. **Poor error handling**: Swallowing exceptions or unclear messages
6. **Missing edge cases**: Only testing the happy path
7. **Premature optimization**: Optimizing before measuring
8. **Not understanding existing code**: Making changes without full context

## Important Notes

- **DO NOT commit code** during implementation. Focus on getting everything working correctly first.
- The user will instruct you when to commit using the `/commit` command
- DO NOT push code - the user will handle all git push operations

Remember: Great code is not just code that works, but code that can be understood, maintained, and extended by others (including future you).