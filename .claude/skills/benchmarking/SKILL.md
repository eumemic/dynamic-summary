---
name: benchmarking
description: This skill should be used when the user asks to "run the benchmark", "run locomo", "benchmark ragzoom", "evaluate recall quality", "compare with letta", "check accuracy", "run the evaluation harness", "agentic evaluation", "max iterations", or mentions benchmarking, evaluation, or competitive comparison of the memory system.
---

# RagZoom Benchmarking

Run and interpret memory system benchmarks to measure RagZoom's recall quality against the competitive landscape.

## Quick Start: Running LoCoMo

LoCoMo is the de facto standard benchmark for conversational memory systems. Every competitor reports numbers here.

**Dataset location**: `test_data/locomo10.json` (10 conversations, ~1500 QA pairs)

### Prerequisites

1. Source the `.env` file: `set -a && source .env && set +a`
2. Dataset at `test_data/locomo10.json` (already in the repo)

### Running a Benchmark

**Always use `--isolated-server`**. This spawns a temporary server on port 50053 with a fresh state directory, completely isolated from dev and production servers. No stale leases, no port conflicts, no state bleed.

```bash
set -a && source .env && set +a
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --isolated-server
```

Ingestion takes ~5 minutes (summarization tree building). Evaluation takes ~5 minutes at concurrency=10.

**Never use a shared dev server for benchmarks.** The `--skip-ingest` shortcut without `--isolated-server` depends on a running dev server, which creates stale lease, port conflict, and state bleed problems.

### Pre-Run Cleanup

Before each benchmark run, kill any stale server processes from crashed previous runs:

```bash
# Kill stale benchmark server on port 50053 ONLY
lsof -ti :50053 | xargs kill -9 2>/dev/null
```

**CRITICAL: NEVER use `pkill -f ragzoom` or other broad patterns.** This will kill the production daemon on port 50051, breaking the MCP server, stop hooks, and CLI. Always target the specific benchmark port (50053).

The `--isolated-server` flag uses a temp directory that gets cleaned up automatically when the server exits cleanly. If a run crashes mid-flight, the temp dir lives in `/tmp/ragzoom-bench-*` and gets cleaned by the OS eventually. To force-clean:

```bash
rm -rf /tmp/ragzoom-bench-*
```

### Key CLI Parameters

| Flag | Default | Purpose |
|------|---------|---------|
| `--search-model MODEL` | `gpt-4.1-mini` | LLM for agentic search (OpenAI or Anthropic) |
| `--judge-model MODEL` | `gpt-4.1` | LLM-as-Judge (matches Letta leaderboard) |
| `--sample N` | all | Random subset of N questions (seed=42) |
| `--max-iterations N` | 5 | Max recall iterations per question |
| `--max-budget N` | 4000 | Max token budget per recall call |
| `--f1-only` | off | Skip LLM judge, token F1 only |
| `--rejudge PATH` | — | Re-judge from previous results.json |
| `--isolated-server` | off | Spawn isolated server (clean slate) |
| `--skip-ingest` | off | Skip ingestion (docs already indexed) |
| `--profiling` | off | Search profiling (retrospective per question) |
| `-v` | off | Verbose logging |

### Cheap Iteration Modes

Three modes to reduce benchmark cost from ~$22 to <$1 or $0:

**Sample mode** (`--sample N`): Evaluate a random subset of N questions (seed=42 for reproducibility).
```bash
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --sample 20 --isolated-server
```

**F1-only mode** (`--f1-only`): Skip the LLM judge entirely, compute token F1 only. No judge API costs.
```bash
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --f1-only --isolated-server
```

**Rejudge mode** (`--rejudge PATH`): Re-run the LLM judge on previously cached answers. No RagZoom server needed.
```bash
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --rejudge locomo_results/results.json
```

Modes combine: `--sample 5 --f1-only --isolated-server` is the cheapest way to smoke-test evaluation.

## Apples-to-Apples Comparison with Letta Leaderboard

Critical: Use Letta's exact methodology for comparable numbers.

| Setting | Letta Leaderboard | Our Defaults |
|---------|-------------------|--------------|
| Judge model | gpt-4.1 | gpt-4.1 |
| Judge prompt | 3-way GRADER_TEMPLATE (A/B/C) | Same (adopted from Letta) |
| Answer model | GPT-4o-mini (via Letta agent) | gpt-4o-mini |
| Retrieval | Full conversation via file search | Token-budgeted hierarchical |
| Scoring | A=1.0, B/C=0.0 | Same |

The judge rubric (in `scoring.py`) is Letta's exact `GRADER_TEMPLATE` with 3-way grading:
- **A (CORRECT)**: Contains key facts, no contradictions. Hedging OK. Typos OK. Inferable info can be omitted.
- **B (INCORRECT)**: Contains factual contradictions, even with hedging.
- **C (NOT_ATTEMPTED)**: Missing key info but no contradictions ("I don't know").

For detailed comparison methodology, see **`references/letta-comparison.md`**.

## Interpreting Results

### Accuracy and F1

The report outputs aggregate scores (overall + per-category accuracy and F1):
- **Accuracy**: Judge verdict A=1.0, B/C=0.0 (only present with a judge, not in `--f1-only` mode)
- **Token F1**: Token-level overlap between generated and gold answer

### Agent Cost Summary

The markdown report includes an Agent Cost Summary with average retrieval calls, reasoning turns, and input/output/retrieved tokens per question. Compare across models/configs to assess the cost-accuracy tradeoff.

### Category Breakdown

| Category | Tests | RagZoom's Challenge |
|----------|-------|---------------------|
| Single-hop | One specific fact | Summaries may lose the exact detail |
| Multi-hop | Connect facts across turns | Hardest — needs multiple specific nodes |
| Temporal | Time-based reasoning | Timestamps lost in summarization |
| Open-domain | Broad understanding | Best fit for hierarchical summaries |

### Diagnostic Signals

- **High NOT_ATTEMPTED rate**: Retrieval not surfacing relevant content. Need higher budget or better seed selection.
- **High INCORRECT rate**: Model hallucinating from summaries. Summaries may be misleading.
- **Low F1 but high judge accuracy**: Judge is lenient (semantic matching), F1 is strict (token overlap). Expected divergence.

## Architecture

```
ragzoom/agent/                    # Model-agnostic agent layer (shared by search + evaluation)
├── protocol.py                   # BenchmarkingAgent protocol, CostMetrics, ToolDefinition
├── factory.py                    # create_backend() — routes to OpenAI or Anthropic
└── backends/
    ├── openai.py                 # OpenAI function-calling agent loop
    └── anthropic.py              # Claude Agent SDK backend

ragzoom/search/                   # Production search agent
├── agent.py                      # SearchAgent — uses BenchmarkingAgent backend
├── retrospective.py              # Self-critique via backend (profiling only)
├── config.py                     # SearchConfig (model, iterations, budget)
└── prompt.py                     # System prompt + retrospective prompt

ragzoom/evaluation/locomo/        # Benchmark harness
├── types.py                      # Data types: AnswerResult, AggregateScores, BenchmarkReport
├── ingest.py                     # Conversation → AppendUnit ingestion
├── scoring.py                    # Token F1 + Letta GRADER_TEMPLATE judge
├── runner.py                     # Orchestration: ingest → evaluate via agentic search → aggregate
├── report.py                     # JSON + Markdown output with cost metrics
└── agent/
    └── prompt.py                 # Benchmark-specific system prompt
```

CLI entry point: `scripts/run-locomo`

## Common Issues

### "No module named ragzoom.evaluation.locomo"
Production ragzoom is installed non-editable. Use `PYTHONPATH=.` to pick up local code.

### Stale server process blocking port 50053
If `--isolated-server` fails with a port binding error, a previous benchmark server didn't exit cleanly:
```bash
lsof -ti :50053 | xargs kill -9 2>/dev/null
```

### API key not propagating
`source .env` doesn't export. Use: `set -a && source .env && set +a`

### Agent zoom errors
If the agent zooms into a time range with no content, `rz.query` may error. The backend handles this gracefully by returning the error as a tool result, letting the agent try a different approach.

## Additional Resources

### Reference Files

- **`references/letta-comparison.md`** — Detailed Letta Leaderboard methodology, their exact grader prompt, and how to ensure apples-to-apples comparison
- **`references/dataset-format.md`** — LoCoMo JSON format gotchas discovered during implementation
- **`references/competitive-landscape.md`** — Summary of competing systems and benchmarks. Full research at `docs/benchmarking/competitive-landscape.md`

### Strategy Documents

- **`docs/benchmarking/strategy.md`** — Multi-objective optimization framework, Pareto frontiers, parameter space design
- **`docs/benchmarking/TODO.md`** — Roadmap for model coverage, parameter sweep framework, duration tracking
