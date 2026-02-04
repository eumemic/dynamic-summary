# Competitive Landscape Summary

Full research document: `docs/research/agent-memory-benchmarks.md`

## Key Competitors on LoCoMo

| System | Architecture | LoCoMo Performance | Notes |
|--------|-------------|-------------------|-------|
| Letta filesystem agent | GPT-4o-mini + file grep | 74% | Simple tools beat complex systems |
| Mem0 | Fact extraction + vector DB | Self-reported, not reproduced | No evaluation code released |
| SimpleMem | Entropy-aware compression + adaptive-k retrieval | +26% F1 over Mem0 | Closest architectural parallel |
| MemoryOS | Hierarchical short/mid/long-term tiers | Competitive | Discrete tiers, no budget control |
| Memory-R1 | RL-trained memory manager | +69% F1 over Mem0 | LLaMA-3.1-8B backbone |

## Architectural Camps

1. **Fact extraction** (Mem0, Cognee): Extract structured facts, retrieve on demand. Lossy by design.
2. **Agent-managed storage** (Letta/MemGPT): LLM decides what to store/retrieve via tool calls.
3. **Unified memory** (MemOS/MemTensor): Unify parametric, activation, and plaintext memory.
4. **Progressive summarization** (RagZoom): Hierarchical compression with token-budgeted retrieval. Unique position.

## RagZoom's Unique Differentiators

1. **Token-budgeted retrieval**: No competitor lets the caller control the detail/token trade-off.
2. **Continuous granularity**: Same content at multiple heights (verbatim through high-level summary).
3. **Background compaction**: Zero query-time extraction cost.
4. **Budget-accuracy curve**: A metric only RagZoom can produce — shows accuracy as a function of token spend.

## Key Industry Findings

- Sophistication loses to simplicity (Letta filesystem agent beats Mem0, knowledge graphs)
- No advanced system consistently beats simple RAG (MemoryBench meta-analysis)
- 30-73% performance gap versus humans across all systems
- Multi-hop conflict resolution: <=7% accuracy for all systems
- Over-personalization causes 26-61% drops (OP-Bench)

## Benchmark Roadmap

| Priority | Benchmark | Why |
|----------|-----------|-----|
| Done | LoCoMo | Table stakes, every competitor reports here |
| Next | LongMemEval | Scalable to 1.5M tokens — where compression shines |
| Future | MemoryAgentBench | LRU competency directly tests hierarchical summarization |
| Future | BEAM | 10M token scale — RagZoom's home turf |
