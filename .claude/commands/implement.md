# /implement

Arguments: "$ARGUMENTS"

Implement the requested feature or fix with correctness, maintainability, and long-term code health in mind.

## Core Intent

Transform requirements into working code that fits naturally into the existing codebase. Think deeply, design carefully, implement incrementally, and verify thoroughly.

## Process

1. **Clarify**: If requirements are ambiguous, ask. Better to clarify than build the wrong thing.

2. **Understand**: Read the code carefully and think deeply. Trace through execution flows, understand data structures, identify invariants. Know why things are the way they are before changing them.

3. **Design**: Consider 2-3 approaches. Choose what best fits the architecture. Plan incremental delivery.

4. **Build**: Start simple, make it work, then refine. Follow existing patterns. Write tests first when possible.

5. **Verify**: Run tests, manual testing, check edge cases. Use `/test` for thorough testing if needed.

## Key Principles

- **Incremental progress**: Small working changes over large broken ones
- **Match the codebase**: Follow existing patterns, conventions, libraries
- **Test the behavior**: Not the implementation details
- **Leave it better**: Improve what you touch
- **No commits**: User will tell you when to commit via `/commit`

## Consultation Triggers

Stop and ask if you encounter:
- Spec ambiguity or contradictions
- Major architectural decisions needed
- Breaking changes to existing behavior
- Performance or security concerns
- Need for new dependencies

## Examples

Arguments: "add retry logic to API calls"
→ Study existing error handling, implement exponential backoff matching current patterns, add tests

Arguments: "fix memory leak"
→ Use `/diagnose` first if needed, implement fix, verify with memory profiling

Remember: Great code fits naturally into the codebase and can be understood by future developers.