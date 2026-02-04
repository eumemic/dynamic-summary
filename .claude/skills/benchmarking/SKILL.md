---
name: benchmarking
description: This skill should be used when the user asks to "run the benchmark", "run locomo", "benchmark ragzoom", "evaluate recall quality", "compare with letta", "check accuracy", "run the evaluation harness", "budget-accuracy curve", or mentions benchmarking, evaluation, or competitive comparison of the memory system.
---

# RagZoom Benchmarking

Run and interpret memory system benchmarks to measure RagZoom's recall quality against the competitive landscape.

## Quick Start: Running LoCoMo

LoCoMo is the de facto standard benchmark for conversational memory systems. Every competitor reports numbers here.

### Prerequisites

1. Dev server running on port 50052 (`python -m ragzoom.cli server start`)
2. `OPENAI_API_KEY` exported (lives in `~/.config/ragzoom/.env`)
3. Dataset at `test_data/locomo10.json` (10 conversations, ~1500 QA pairs)

### First Run (with ingestion)

```bash
export $(grep -v '^#' ~/.config/ragzoom/.env | xargs)
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --budgets 2000
```

Ingestion takes ~5 minutes (summarization tree building). Evaluation takes ~5 minutes per budget level at concurrency=10.

### Subsequent Runs (skip ingestion)

```bash
export $(grep -v '^#' ~/.config/ragzoom/.env | xargs)
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --skip-ingest --budgets 2000
```

### Full Budget Sweep

```bash
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --skip-ingest
```

Default budgets: 500, 1000, 2000, 4000, 8000. Results saved to `locomo_results/`.

### Cheap Iteration Modes

Three modes to reduce benchmark cost from ~$22 to <$1 or $0:

**Sample mode** (`--sample N`): Evaluate a random subset of N questions (seed=42 for reproducibility).
```bash
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --skip-ingest --budgets 2000 --sample 200
```

**F1-only mode** (`--f1-only`): Skip the LLM judge entirely, compute token F1 only. No judge API costs.
```bash
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --skip-ingest --f1-only --budgets 2000
```

**Rejudge mode** (`--rejudge PATH`): Re-run the LLM judge on previously cached answers. No RagZoom server needed.
```bash
PYTHONPATH=. python scripts/run-locomo --data test_data/locomo10.json --rejudge locomo_results/results.json
```

Modes combine: `--sample 200 --f1-only` evaluates 200 questions with F1 only (~$0.15).

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

### Budget-Accuracy Curve

The unique RagZoom metric. Shows how accuracy scales with token budget:

- **Low budgets (500-1000)**: Mostly summary nodes. Good for open-domain, poor for specific facts.
- **Mid budgets (2000-4000)**: Mix of summaries and verbatim. Accuracy inflection point.
- **High budgets (8000+)**: More leaf nodes. Approaching full-context performance.

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
ragzoom/evaluation/locomo/
├── types.py      # Data types, JSON parsing, QACategory enum
├── ingest.py     # Conversation → AppendUnit ingestion, timestamp parsing
├── answer.py     # RagZoom query → LLM answer generation
├── scoring.py    # Token F1 + Letta GRADER_TEMPLATE judge
├── runner.py     # Orchestration: ingest → sweep budgets → aggregate
└── report.py     # JSON + Markdown output
```

CLI entry point: `scripts/run-locomo`

## Common Issues

### "No module named ragzoom.evaluation.locomo"
Production ragzoom is installed non-editable. Use `PYTHONPATH=.` to pick up local code.

### Stale database after code changes
If server hits missing column errors, kill server, delete `data/sqlite.db`, restart.

### API key not propagating
`source .env` doesn't export. Use: `export $(grep -v '^#' ~/.config/ragzoom/.env | xargs)`

### Re-ingestion after server restart
If the dev database was cleared, drop `--skip-ingest`. Ingestion takes ~5 min.

## Additional Resources

### Reference Files

- **`references/letta-comparison.md`** — Detailed Letta Leaderboard methodology, their exact grader prompt, and how to ensure apples-to-apples comparison
- **`references/dataset-format.md`** — LoCoMo JSON format gotchas discovered during implementation
- **`references/competitive-landscape.md`** — Summary of competing systems and benchmarks. Full research at `docs/benchmarking/competitive-landscape.md`
