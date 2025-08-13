---
allowed-tools: Read, Write, Edit, Bash, Grep
description: Run interactive prompt strategy experiments
argument-hint: [goals or focus area]
---

# /prompt-experiment
# This command was created with the `/command` command. If you are making changes to this
# file, make sure to observe the rubric laid out in `.claude/commands/command.md`.

## Context
- Runner options: !`python prompt-experiments/run_experiments.py --help`
- Active strategies: @prompt-experiments/active_strategies.py
- Active strategy count: !`python -c "import sys; sys.path.insert(0, 'prompt-experiments'); from active_strategies import ACTIVE_STRATEGIES; print(f'{len(ACTIVE_STRATEGIES)} strategies active')" 2>/dev/null || echo "Check active_strategies.py"`
- Available strategies: !`ls prompt-experiments/strategies/library/*.py | xargs -n1 basename | sed 's/.py//' | grep -v base_strategy | grep -v __init__`
- Recent experiment runs: !`ls -dt prompt-experiments/results/*/ 2>/dev/null | head -3 | xargs -n1 basename`
- Current corpus chunks: !`grep -c '"id"' prompt-experiments/results/corpus.json 2>/dev/null || echo "No corpus found"`
- Example strategy: @prompt-experiments/strategies/library/word_count.py
- Base strategy template: @prompt-experiments/strategies/base_strategy.py
- Typical performance: ~10 experiments/second with 30 concurrent requests

## Strategic Guidance
You're running an interactive evolutionary process to find optimal summarization length targeting strategies. Each generation: run experiments → analyze results → discuss findings → curate strategies → repeat. The evolution happens through conversation, not algorithms.

Key insights from prior work:
- Word-based strategies consistently outperform token-based (~85% vs ~30% accuracy)
- "Shorten by X%" worked perfectly in isolation but failed in batch tests
- Percentage strategies fundamentally misunderstood by models
- Systematic biases can be compensated (word count overshoots by ~6.5%)

## Task
Arguments: "$ARGUMENTS"

Use any provided arguments to understand the user's goals (e.g., "focus on word-based variants", "test extreme compressions", "find fastest strategies"). If no arguments, infer from context or start broadly.

Before first generation:
- Parse arguments for any new strategy ideas (e.g., "try 'condense to X words'" → create CondenseWordsStrategy)
- Implement any new strategies in `prompt-experiments/strategies/library/`
- Update `active_strategies.py` to include them

For each generation:
1. Propose experiment parameters:
   - Sample size (default: 10 chunks)
   - Compression ratios (default: 14 ratios from 10-90%)
   - Max concurrent requests (default: 30)
   - Adjust based on: number of strategies, time constraints, desired confidence
2. After confirmation, run: `python prompt-experiments/run_experiments.py [params]`
3. Analyze results in `results/latest/` - focus on rankings and error patterns
4. List visualization paths for easy access (command-clickable)
5. Summarize findings concisely with:
   - Top 3 performers with accuracy metrics
   - Bottom 3 failures and why they failed
   - Interesting patterns or surprises
6. Propose evolution:
   - Which strategies to drop (consistent failures)
   - New variants to try (based on what's working)
   - Hypotheses to test
   - Parameter adjustments for next run (e.g., larger sample if results are noisy)
7. Edit `active_strategies.py` based on discussion
8. Run next generation with refined parameters

Keep summary focused: what worked, what failed, what to try next. User drives decisions.

## Retrospective
After completing this task, reflect on three levels:
1. **Command Improvement**: How could this specific command guide future agents better?
2. **Rubric Conformance**: Does this command follow the /command design principles well?
3. **Meta Evolution**: Should the /command rubric itself evolve based on your experience?

ONLY if you spot a significant issue or opportunity for improvement, bring it to the user's attention. Don't waste the user's time and your tokens with pedantic corrections or things that are not broadly applicable to all uses of the command.