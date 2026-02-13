# RagZoom Benchmarking Strategy

## The Google Maps Analogy

RagZoom is Google Maps for memory. The mapping is precise:

| Maps | RagZoom |
|------|---------|
| Pixels | Tokens |
| Browser window | Context window |
| Image resolution | Textual detail |
| Downscaling | Summarization |
| Hierarchical image tiles | Summary tree |

The intended usage pattern is iterative zoom: start broad, find interesting areas, tighten the time window, repeat until you reach raw transcript. A single-shot query is one zoom level. The full experience is the multi-step investigation.

## Two Levels of Evaluation

### System-Level: Fixed-Budget Curve

Measure RagZoom's retrieval quality in isolation. Given exactly N tokens, how good is the tiling?

- **Controlled variable**: Token budget (500, 1K, 2K, 4K, 8K, ...)
- **Output**: Accuracy at each budget
- **Analogy**: Benchmarking the map renderer — given a fixed viewport, how well do you render?
- **Unique to RagZoom**: No competitor can produce this curve

### Product-Level: Agentic Evaluation

Measure the full experience — an intelligent agent iteratively querying RagZoom to answer questions.

- **Controlled variable**: Total tokens consumed (cost)
- **Output**: Accuracy vs total cost
- **Analogy**: Benchmarking the user experience — does a user find what they need, and how many pan/zoom operations?
- **Comparable to**: Letta's agent (apples-to-apples)

The key insight: setting `max_iterations=1` in the agentic evaluator collapses to the single-shot case. One harness, one framework, parameterized by agent strategy.

## The Agentic Evaluator

The agent replicates the intended RagZoom workflow:

1. **Survey** — broad query, generous budget, get the lay of the land
2. **Identify** — find time ranges or topics that look relevant
3. **Zoom** — tighter time window, lower budget (scope is narrower)
4. **Repeat** — keep drilling until verbatim content answers the question

Metrics tracked per question:
- Total input tokens across all calls
- Total output tokens across all calls
- Number of retrieval calls (zoom iterations)
- Number of LLM reasoning calls
- Estimated cost at current API rates
- Judge verdict (A/B/C)
- Token F1

## Multi-Objective Optimization Framework

### Inputs (Parameters to Optimize)

Agent parameters:
- `max_iterations` — how many zoom steps allowed
- `initial_budget` — token budget for the survey call
- `zoom_factor` — how much to narrow per step
- Agent model (reasoning capability vs cost)

RagZoom algorithm parameters:
- Summarization depth / compaction ratio
- Seed selection strategy
- Tiling algorithm parameters

### Outputs (Objectives)

Per question:
- Accuracy (judge verdict)
- Total tokens consumed
- Cost in dollars

### Pareto Frontier

A parameter configuration is Pareto-optimal when you can't improve accuracy without increasing cost, or reduce cost without losing accuracy. The frontier shows the fundamental accuracy-cost tradeoff of the system.

This is what "AI Agents That Matter" (Princeton, TMLR 2025) argues all agent benchmarks should report, and no memory benchmark currently does.

## Multi-Benchmark Strategy

Each benchmark stresses different capabilities:

| Benchmark | Scale | Tests | RagZoom's Challenge |
|-----------|-------|-------|---------------------|
| LoCoMo | 9K tokens | Factoid recall | Too small for compression to shine |
| LongMemEval | 115K-1.5M | Scalable histories | Where hierarchical compaction should dominate |
| BEAM | Up to 10M | Extreme scale | RagZoom's home turf |
| MemoryAgentBench | 128K-200K | Four competencies | Long-Range Understanding is the target |

### Per-Benchmark Pareto Frontiers

Compute a (accuracy, cost) Pareto frontier for each benchmark independently. Look for parameter configurations that appear on multiple frontiers — these are robustly good.

### Cross-Benchmark Analysis

The full objective space is (accuracy, cost) per benchmark — 2B dimensions for B benchmarks. Practical approaches to collapse:

1. **Per-benchmark Pareto, then meta-analysis** — find configurations robust across benchmarks
2. **Normalized aggregation** — normalize each benchmark's accuracy to [0,1], costs to dollars, compute a single 2D Pareto frontier over (mean_accuracy, mean_cost)
3. **Lexicographic** — "don't embarrass on any benchmark (min threshold), then minimize cost"
4. **Scalarization** — weighted sum, but requires choosing weights

Different benchmarks will favor different parameter settings (higher budgets help factoid recall, lower budgets suffice for broad understanding). That tension in the Pareto frontier is itself diagnostic.

## Cost-Awareness Gap in the Field

Research finding: almost no memory benchmark tracks cost. Out of 10 benchmarks surveyed:

| Benchmark | Tracks Cost? |
|-----------|-------------|
| GoodAI LTM | Aggregate USD per run (not per-question) |
| Letta Leaderboard | Shows $ alongside accuracy (not combined) |
| WritePolicyBench | Byte-budget utility-per-KB (storage, not inference) |
| Everything else | No |

Nobody computes a formal cost-normalized accuracy metric for memory systems. Publishing per-question token consumption alongside LoCoMo accuracy would be a first.

## Implementation Roadmap

| Priority | What | Why |
|----------|------|-----|
| Done | Fixed-budget LoCoMo evaluation | Baseline numbers on the board |
| Next | Agentic evaluator for LoCoMo | Product-level benchmark, comparable to Letta |
| Next | Per-question token/cost tracking | First cost-aware memory benchmark |
| Future | LongMemEval at multiple scales | Where compression shines |
| Future | BEAM at 10M tokens | RagZoom's home turf |
| Future | Multi-benchmark Pareto analysis | Publishable contribution |

## Current Results

### LoCoMo (Fixed Budget, gpt-4.1 Judge, Letta Methodology)

| Budget | Overall | Single-hop | Multi-hop | Temporal | Open-domain |
|--------|---------|-----------|-----------|----------|-------------|
| 500 | 11.2% | 13.1% | 1.2% | 17.7% | 13.7% |
| 1,000 | 12.3% | 14.5% | 2.2% | 19.8% | 14.6% |
| 2,000 | 21.3% | 21.6% | 4.7% | 20.8% | 27.6% |
| 4,000 | 31.7% | 32.3% | 7.5% | 22.9% | 41.7% |
| 8,000 | 45.3% | 44.7% | 8.7% | 24.0% | 62.0% |

Comparison: Letta filesystem agent = 74% (full context, no budget constraint, agentic).

Verdict breakdown at budget=2000: A=21.6%, B=17.6%, C=60.8%.
