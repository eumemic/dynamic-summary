# /test

Test Target: $ARGUMENTS

_Note: if no test target is specified above, assume it's the most recently implemented feature or fix from your current context._

Perform thorough manual testing of the specified feature, fix, or component. Think strategically about using your available tools to simulate real-world usage and edge cases. This is a read-only operation - clean up any test artifacts before presenting your final report.

## Phase 1: Test Planning

1. **Identify what to test**:
   - What is the specific feature/fix/component?
   - What are the expected behaviors?
   - What are the main user flows?
   - What integrations or dependencies exist?

2. **Define test scope**:
   - In scope: What specifically will you test?
   - Out of scope: What won't you test (and why)?
   - Test environment: Local, staging, production?
   - Test data requirements

3. **Create test scenarios**:
   Develop a comprehensive list covering:
   - **Happy path**: Normal expected usage
   - **Edge cases**: Boundary conditions, limits
   - **Error cases**: Invalid inputs, missing data
   - **Integration points**: API calls, database operations
   - **User experience**: UI/UX flows, accessibility
   - **Performance**: Load times, resource usage
   - **Security**: Authorization, data validation

## Phase 2: Test Execution

1. **Tool usage strategy**:
   - **UI Testing**: Use MCP Playwright browser tools for web interfaces
   - **API Testing**: Use Bash with curl for endpoint testing
   - **Data Verification**: Query databases, check files
   - **Log Analysis**: Read service logs for errors/warnings
   - **State Inspection**: Verify data integrity and consistency
   - **Debug Helpers**: Add temporary logging/debugging code as needed

2. **Execute systematically**:
   - Test each scenario methodically
   - Observe actual behavior vs. expected
   - Note any deviations or anomalies
   - Capture evidence (but clean up after)
   - Test both positive and negative cases

3. **Testing techniques**:
   - **Exploratory testing**: Try unexpected user behaviors
   - **Negative testing**: Invalid inputs, edge cases
   - **Regression testing**: Ensure existing features still work
   - **Concurrent usage**: Test simultaneous operations
   - **State transitions**: Test workflow sequences

## Phase 3: Issue Investigation

When you find issues:

1. **Reproduce consistently**:
   - Verify the issue occurs reliably
   - Identify minimal reproduction steps
   - Note any intermittent behavior

2. **Gather evidence**:
   - Error messages and stack traces
   - Log entries at time of issue
   - System state when issue occurs
   - Screenshots if UI-related

3. **Determine impact**:
   - Who/what is affected?
   - Is there a workaround?
   - How severe is the issue?

### If stuck on a specific technical problem:
```
Bash(claude -p "/diagnose [detailed problem description including all context needed]")
```

## Phase 4: Cleanup

**IMPORTANT**: Before presenting your report:
- Remove any temporary debug logging/print statements added
- Delete any test files created
- Remove test data from databases
- Clean up temporary directories
- Restore any modified configurations
- Close any opened browser sessions
- Revert any code modifications made for testing

## Phase 5: Test Report Format

Present your findings as follows:

### 📋 Test Report

**Test Target**: [Feature/Fix name]  
**Test Date**: [Today's date]  
**Test Environment**: [Local/Staging/Production]

#### Executive Summary
[2-3 sentences summarizing overall findings and go/no-go recommendation]

#### Test Results
- **Total Scenarios**: [Number]
- **Passed**: [Number] ✅
- **Failed**: [Number] ❌
- **Warnings**: [Number] ⚠️

#### Critical Issues
[List any blocking issues that must be fixed]

1. **[Issue Name]**
   - Description: [What's wrong]
   - Steps to reproduce: [How to trigger]
   - Impact: [Who/what is affected]
   - Severity: [Critical/High/Medium/Low]

#### Non-Critical Issues
[List issues that should be fixed but aren't blocking]

1. **[Issue Name]**
   - Description: [What's wrong]
   - Impact: [Who/what is affected]
   - Workaround: [If any]

#### Observations
[Notes about performance, UX, or other concerns]
- [Observation 1]
- [Observation 2]

#### Recommendation
[Clear go/no-go recommendation with reasoning]

---

## Testing Guidelines

- **Test destructively**: Try to break the system
- **Think like a user**: Test realistic workflows
- **Be thorough**: Check edge cases and errors
- **Stay objective**: Report what you find, not what you hope
- **Clean up completely**: Leave no trace of testing

## Important Reminders

- This is a READ-ONLY operation in net effect
- You may modify code temporarily for testing (e.g., add debug logging)
- ALL modifications must be reverted before presenting the report
- Do NOT commit any changes
- Do NOT update documentation
- Clean up ALL test artifacts before reporting
- The final output should be just the test report

Remember: Your job is to find issues, not fix them. Test thoroughly, clean up completely, and report objectively.