# /diagnose

Problem: $ARGUMENTS

_Note: if no problem is specified above, assume it's whatever issue is apparent from your recent context._

Perform a systematic, scientific investigation of the problem described above, thinking carefully at every step. Follow the scientific method rigorously to diagnose the issue and propose solutions without implementing them yet.

## Phase 1: Problem Definition & Initial Observations

1. **Restate the problem clearly**:
   - What is the expected behavior?
   - What is the actual behavior?
   - What are the symptoms?
   - When does it occur?

2. **Gather initial context**:
   - What components/services are involved?
   - What recent changes might be relevant?
   - What is the data flow?
   - Are there error messages or logs?

3. **Define success criteria**:
   - How will we know when the problem is solved?
   - What metrics or outputs should change?

## Phase 2: Deep Code Analysis & Hypothesis Formation

1. **Map the system**:
   - Trace the complete flow from input to output
   - Identify all touchpoints and transformations
   - Note any assumptions or invariants

2. **Examine critical code paths**:
   - Read relevant source files thoroughly
   - Pay attention to:
     - Data structures and their transformations
     - Conditional logic and edge cases
     - Error handling (or lack thereof)
     - External dependencies and integrations

3. **Form hypotheses** (at least 3):
   Common hypothesis patterns to consider:
   - **State**: Previous state affecting current behavior
   - **Input**: Unexpected or malformed input data
   - **Logic**: Conditional branch taking wrong path
   - **Dependencies**: External resource unavailable/changed
   - **Concurrency**: Race condition or deadlock
   - **Configuration**: Setting or environment difference
   
   For each hypothesis:
   - **H1**: [Clear statement of what might be wrong]
     - Evidence for: [What supports this]
     - Evidence against: [What contradicts this]
     - Testable prediction: [What we'd see if true]
     - Test approach: [How to verify cheaply]

## Phase 3: Experimental Testing

### Investigation Toolkit:
- **Binary Search**: Disable half the code/features to isolate
- **Time Travel**: When did it last work? What changed?
- **Minimal Reproduction**: Smallest input that triggers issue
- **Environment Swap**: Does it work elsewhere?
- **Logging Injection**: Add strategic print/log statements

For each hypothesis, design and run minimal tests:

1. **Quick verification tests**:
   - Use grep/search to verify assumptions
   - Check logs or debug output
   - Examine data at key points
   - Create minimal test scripts if needed

2. **Document each experiment**:
   ```
   Experiment 1: [What you're testing]
   Method: [How you're testing it]
   Expected: [What you expect to see]
   Actual: [What you actually saw]
   Conclusion: [What this tells us]
   Next Step: [What this result suggests to try next]
   ```

3. **Iterative refinement**:
   - If a hypothesis is disproven, form new ones based on findings
   - If partially confirmed, dig deeper
   - Follow unexpected leads

## Phase 4: Root Cause Analysis

Once you've identified the likely cause:

1. **Confirm the root cause**:
   - Can you reproduce the issue reliably?
   - Does your explanation account for ALL symptoms?
   - Are there any edge cases not explained?

2. **Check common blind spots**:
   - Assumptions about data types/formats
   - Implicit dependencies not in code
   - Side effects from seemingly unrelated changes
   - Platform/environment differences
   - Caching at any layer
   - Default values and edge cases (null, empty, zero)

3. **Understand the impact**:
   - What else might be affected?
   - Are there similar issues elsewhere?
   - What are the risks of the current state?

## Phase 5: Solution Design (NO IMPLEMENTATION)

1. **Propose solutions**:
   For each potential fix:
   - **Solution A**: [Brief description]
     - Pros: [Benefits]
     - Cons: [Drawbacks]
     - Effort: [Low/Medium/High]
     - Risk: [Low/Medium/High]

2. **Recommend the best approach**:
   - Which solution addresses the root cause most directly?
   - Which has the best effort/risk/reward ratio?
   - What's the implementation plan?

## Phase 6: Cleanup & Report

1. **Clean up any test artifacts**:
   - Delete test files created during investigation
   - Revert any temporary changes
   - Close any opened resources

2. **Final Report**:

### 🔍 Diagnosis Complete

**Problem**: [One sentence]

**Root Cause**: [One sentence]

**Fix**: [Recommended solution]

**Evidence**:
- [Key proof point 1]
- [Key proof point 2]
- [Key proof point 3]

**Risk**: [What could go wrong with the fix]

**Implementation Steps**:
1. [Step 1]
2. [Step 2]
3. [Step 3]

---

## Investigation Best Practices

- **Be systematic**: Don't jump to conclusions
- **Test cheaply first**: Use grep, small scripts, logs before complex tests
- **Document everything**: Your future self will thank you
- **Question assumptions**: What "everyone knows" might be wrong
- **Follow the data**: Let evidence guide you, not intuition
- **Consider multiple causes**: Problems often have multiple contributing factors
- **Think about side effects**: What else uses the code you're investigating?

## Common Investigation Patterns

1. **Binary search debugging**: Narrow down where things go wrong
2. **Differential diagnosis**: What changed between working and broken states?
3. **Trace analysis**: Follow data through the entire pipeline
4. **Invariant checking**: What assumptions are being violated?
5. **Boundary testing**: Do edge cases reveal the issue?

Remember: The goal is to understand deeply before acting. A well-diagnosed problem is already half-solved.