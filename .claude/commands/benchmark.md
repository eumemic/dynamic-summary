---
allowed-tools: [Bash, Read, LS, Glob]
description: Run benchmarks against baseline and analyze results
argument-hint: [--baseline path] [--corpus file] [--output dir]
---

# /benchmark
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Benchmark script: @scripts/run-indexing-benchmarks
- Default baseline check: !ls -la telemetry-baseline.json 2>/dev/null || echo "not found"
- Available telemetry files: !ls -1 telemetry*.json 2>/dev/null | head -5
- Working directory: !pwd | sed 's|.*/||'

## Strategic Guidance

The benchmarking script now has smart defaults - it can run with zero arguments if `telemetry-baseline.json` exists. Your job is to:

1. **Validate baseline appropriateness**: Check if telemetry-baseline.json exists and is suitable for the current experiment based on conversation context and arguments. Consider:
   - Is the chunk_size appropriate for what we're testing?
   - Is the document_id relevant to our experiment?
   - When was it created and with what configuration?

2. **Guide baseline setup if needed**: If no suitable baseline exists:
   - Suggest creating telemetry-baseline.json from an existing telemetry file
   - Help run an initial indexing to create a baseline
   - Explain what makes a good baseline for the specific experiment

3. **Always scan for regressions**: Check ALL metrics (token accuracy, API costs, tree height, retry patterns) regardless of experiment focus. Flag any degradation immediately.

4. **Context-aware analysis**: Use conversation history to understand what's being tested (prompt changes, algorithm changes, model updates) and emphasize relevant metrics while maintaining comprehensive coverage.

## Task
Arguments: "$ARGUMENTS"

Run benchmarks comparing current implementation against baseline. Parse any arguments for custom baseline/corpus/output paths. Execute the benchmark script, analyze the outputs (comparison.md, visualization.png, logs), and present findings that are:
1. Regression-aware (any metric worse?)
2. Context-relevant (what are we trying to improve?)
3. Actionable (should we adopt this change?)

## Retrospective
After completing this task, reflect on three levels:
1. **Command Improvement**: How could this specific command guide future agents better?
2. **Rubric Conformance**: Does this command follow the /command design principles well?
3. **Meta Evolution**: Should the /command rubric itself evolve based on your experience?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.