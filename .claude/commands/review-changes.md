# /review-changes

Scope: $ARGUMENTS

If no scope provided above, use ONLY THE FIRST applicable condition below:
- If the git working directory is dirty, scope = the uncommitted files
- Else if on a feature branch, scope = changes since `master`
- Otherwise, scope = the complete current state of the repository

_Note: default to changes on this branch since `master` if no scope provided_

Perform a comprehensive review of the changes in the above scope with both deep architectural analysis and implementation quality checks.

## Phase 1: Deep Architectural Review (DO THIS FIRST!)

Start by asking the hard questions before getting lost in details:

1. **Challenge every new component**:
   - For each new file/class/module: Why does this exist? What simpler alternative did we reject?
   - Are there multiple entry points or ways to start the same service? Why?
   - Count similar files: Are there multiple indexing scripts? Multiple servers? Multiple configs?
   - Draw the data flow - are there unnecessary middlemen or wrappers?
   - What would break if we deleted this component?

2. **Quantify the complexity**:
   - How many lines of code are being added for this feature?
   - How many files have similar/overlapping functionality? 
   - What's the ratio of boilerplate to actual business logic?
   - Could the same functionality be achieved with significantly less code?

3. **Analyze key architectural elements**:
   - **Entry points**: Are there multiple ways to start the same service? Why?
   - **APIs/Interfaces**: Is each API endpoint solving a real user need or just connecting internal components?
   - **Service boundaries**: Could any services be combined or eliminated?
   - **Data flow**: Trace requests through the system - look for unnecessary hops

4. **Question the solution approach**:
   - Are we solving the user's actual problem or a problem we created?
   - Is this the simplest thing that could possibly work?
   - What would a new developer think this code does?
   - Could we achieve the same result with existing code?

5. **Apply the "5 Whys" to major design decisions**:
   - Example: Why do we need this HTTP API? → To trigger reindexing
   - Why can't we use docker exec? → We can, actually...
   - Why spawn subprocesses? → We could import functions directly
   - Keep asking until you reach the core requirement

6. **Look for architectural code smells**:
   - Two files/components doing almost the same thing
   - Wrappers that just call other wrappers
   - Abstractions with only one implementation
   - Configuration that's only used in one place
   - "Just in case" features with no current use

## Phase 2: Implementation Review (AFTER architecture review)

Now examine the tactical details:

1. **Get the full diff**:
   - Run `git diff master...HEAD` to see all changes in this branch
   - Look at the complete picture, not just individual files

2. **Check for inconsistencies**:
   - Naming conventions across files (variables, functions, classes)
   - API endpoint patterns and response formats
   - Error handling approaches
   - Import styles (absolute vs relative)
   - Code formatting and structure

3. **Look for code issues**:
   - Duplicate or redundant code that could be refactored
   - **Dead code detection** (actively search for these):
     * Functions/methods that are defined but never called
     * Use grep to verify: if a function is only found in its definition, it's dead
     * Imports that are never used
     * Classes that are never instantiated
     * Variables assigned but never read
     * Files that aren't imported anywhere
   - Missing error handling or edge cases
   - Potential race conditions or async issues
   - Security vulnerabilities (exposed secrets, unsafe operations)
   - Performance problems (N+1 queries, inefficient algorithms)

4. **Verify documentation**:
   - All new features/utilities are documented in CLAUDE.md files
   - Essential commands are updated if new scripts were added
   - API endpoints have proper docstrings
   - Complex logic has explanatory comments
   - TODOs are tracked in docs/todos.md

5. **Check CLAUDE.md invariants compliance**:
   - Validate that all rules specified in CLAUDE.md files and linked docs are followed
   - Particularly docs/global-invariants.md

6. **Review architecture decisions**:
   - Changes align with the service's stated purpose
   - No violation of service boundaries
   - Consistent with existing patterns
   - Proper separation of concerns

7. **Test coverage**:
   - New functionality has tests or clear test plans
   - Existing tests weren't broken
   - Edge cases are considered

8. **Dependencies and configuration**:
   - New dependencies are justified and documented
   - Environment variables follow UPPER_SNAKE_CASE
   - Docker configurations are consistent
   - No hardcoded values that should be configurable
   - **Check configuration consistency**:
     * Are the same values (models, ports, paths) defined in multiple places?
     * Do Docker env vars match code defaults?
     * Is there a single source of truth for settings?

9. **User experience**:
   - Error messages are helpful and actionable
   - APIs return appropriate status codes
   - Performance impact is considered
   - Backward compatibility is maintained

10. **Final checklist**:
    - No temporary debugging code left behind
    - No commented-out code without explanation
    - Secrets/keys are not exposed
    - File permissions are appropriate
    - Line endings are consistent

## Output Format:

### 🏗️ Architecture & Design Issues
Start with the big picture findings from Phase 1:
- Unnecessary complexity or overengineering
- Simpler alternatives that were overlooked
- Components that could be removed or combined

### ✅ Good Practices Found
- List positive patterns and well-implemented features

### ⚠️ Implementation Issues
- **Critical**: Must fix before merge (bugs, security, breaking changes)
- **Important**: Should fix (inconsistencies, missing docs, tech debt)  
- **Minor**: Nice to fix (style issues, small improvements)

### 📝 Recommendations
- Specific suggestions for simplification
- Refactoring opportunities
- Future considerations

### 🎯 Summary
- Overall assessment of PR readiness
- Key action items before merge
- Most impactful improvements to make