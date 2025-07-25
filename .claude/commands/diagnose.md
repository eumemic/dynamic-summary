# /diagnose

Arguments: "$ARGUMENTS"

Apply the scientific method to systematically diagnose problems without implementing solutions.

## Core Intent

When something's broken, resist the urge to guess. Use extended thinking to trace through the code carefully. Form hypotheses, test them cheaply, and follow the evidence to the root cause.

## Process

1. **Define Problem**: Expected vs actual behavior, symptoms, when it occurs
2. **Form Hypotheses**: At least 3 plausible causes (state, input, logic, dependencies, timing, config)
3. **Test Systematically**:
   - Binary search to isolate
   - Minimal reproduction case
   - Strategic logging/debugging
   - Check assumptions with grep/search
4. **Identify Root Cause**: Confirm it explains ALL symptoms
5. **Propose Fix**: Clear solution addressing the root cause

## Investigation Toolkit

- **Binary Search**: Disable half the code to isolate
- **Time Travel**: When did it last work? What changed?
- **Minimal Repro**: Smallest input that triggers issue
- **Differential Analysis**: Working vs broken states
- **Trace Analysis**: Follow data through the pipeline

## Hypothesis Template

**H1: [What might be wrong]**
- Evidence for: [Supporting observations]
- Evidence against: [Contradicting observations]
- Test: [How to verify cheaply]

## Report Format

**Problem**: [One sentence]
**Root Cause**: [One sentence]
**Evidence**: [Key proof points]
**Fix**: [Recommended solution]

## Examples

Arguments: "API returns 500 on file upload"
→ Test file size limits, check logs at failure point, verify multipart handling, isolate to specific file types

Arguments: "Memory usage grows over time"
→ Profile allocation patterns, check for retained references, test with minimal workload, identify leak source

Remember: Let evidence guide you, not intuition. A well-diagnosed problem is half-solved.