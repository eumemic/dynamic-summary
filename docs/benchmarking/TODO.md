# Benchmarking TODOs

## Model Coverage

- [ ] Add other OpenAI models (gpt-5.2, gpt-5-mini, gpt-5-nano) to benchmark sweeps
- [x] Implement Anthropic agent backend using Claude Agent SDK, add Claude models (haiku, sonnet, opus) to benchmark sweeps

## Parameter Space

Three distinct model roles are part of the optimization parameter space:

- **Summarization model** (indexing time) — used to build the summary tree
- **Embedding model** (indexing time) — used for semantic search/seed selection
- **Agent model** (query time) — the model driving the recall tool loop

Each of these can vary independently in model choice and reasoning effort.

The **judge model** is fixed and outside the parameter space — it's the measuring instrument, not what's being measured.

- [ ] Expose summarization and embedding model as configurable parameters in the benchmark harness
- [ ] Add reasoning effort as a parameter (e.g. OpenAI reasoning_effort, extended thinking budget)
- [ ] Build parameter sweep framework that varies all parameters jointly and computes Pareto frontiers over (accuracy, cost, duration)

## Objectives

Current objectives tracked per question: accuracy (judge verdict), token F1, cost (input/output/retrieved tokens), USD cost, wall-clock duration.

- [x] Track wall-clock duration per question (query-time latency)
- [x] Track indexing duration per conversation
- [x] Track USD cost per question using model pricing from models.json

## Infrastructure

- [x] Benchmark should manage its own isolated RagZoom server (dedicated port + state directory)
- [x] Recall tool output matches production format (Span tags with temporal metadata)
- [ ] Script the full benchmark lifecycle: start server → ingest → evaluate → stop server — no manual server management

## Multi-Objective Optimization

- [ ] Develop full multi-objective optimization with Pareto frontier discovery
  - Sweep all parameters jointly (model choices × reasoning effort × token budgets × max iterations)
  - Compute Pareto frontiers over multiple objectives (accuracy, cost, duration)
  - Identify non-dominated configurations that represent optimal tradeoffs
  - Produce visualizations (2D projections, parallel coordinates) for parameter exploration
  - Store results in a format that supports incremental runs (don't re-evaluate known points)
