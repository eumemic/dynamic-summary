# Agent Memory Systems: Competitive Landscape & RagZoom Strategy

*Research compiled: 2026-02-03*

## Competitive Landscape Summary

The agentic memory space has exploded into 10+ specialized benchmarks and a dozen competing systems, but the field's own results contain a signal RagZoom should exploit: **sophistication is losing to simplicity**. Letta's trivial filesystem agent (GPT-4o-mini + grep) scored 74% on LoCoMo, beating Mem0, knowledge graphs, and OpenAI's built-in memory. The MemoryBench meta-analysis found no advanced system consistently beats simple RAG. Meanwhile, every system struggles on the same two frontiers -- multi-hop conflict resolution (<=7% accuracy) and temporal reasoning (73% below human performance).

The competitive field clusters into three architectural camps:

| Camp | Systems | Core Idea | Weakness |
|------|---------|-----------|----------|
| **Fact extraction** | Mem0, Mem0^g, Cognee | Extract structured facts at write time, retrieve at query time | Lossy extraction, no granularity control |
| **Agent-managed storage** | Letta/MemGPT, MemoryOS | The LLM decides what to store/retrieve via tool calls | Relies on model's tool-use quality; no compression hierarchy |
| **Unified memory abstraction** | MemOS/MemTensor | Unify parametric, activation, and plaintext memory in one API | Heavyweight, complex, early-stage |

RagZoom occupies a fourth position: **progressive summarization with token-budgeted retrieval**. No competitor offers caller-controlled granularity. No existing benchmark tests it. This is both a differentiation advantage and an evaluation gap that needs closing.

Key numbers to internalize:
- Current systems show a **30-73% performance gap versus humans** on memory tasks
- Even GPT-4o achieves only **60%** on single-hop conflict resolution
- LTM agents with 8K context match GPT-4 with 128K context at **16% of cost** (GoodAI)
- Mem0, HippoRAG-v2, Cognee show **20,000x higher construction time** than simple approaches
- Over-personalization causes **26-61% performance drops** on general questions (OP-Bench)

---

## Benchmarks by Relevance to RagZoom

### High Relevance

#### MemoryAgentBench (ICLR 2026)

- **Paper**: [arXiv:2507.05257](https://arxiv.org/abs/2507.05257)
- **Dataset**: 17 datasets, multi-turn interactions, 128K-200K token contexts
- **Tests**: Four competencies -- Accurate Retrieval (AR), Test-Time Learning (TTL), Long-Range Understanding (LRU), Conflict Resolution (CR)
- **Why it matters for RagZoom**: The Long-Range Understanding competency is the closest existing proxy for RagZoom's hierarchical summarization. RAG methods achieve 100% on NIAH-MQ retrieval but fail at global understanding; long-context models show the opposite pattern. RagZoom's zoom mechanism could bridge this gap. The four-competency framework also exposes where RagZoom is strong (LRU) and where it needs augmentation (CR).
- **Critical finding**: No current method masters all four competencies. This fragmentation is an opportunity -- RagZoom doesn't need to win everything, just demonstrate unique strength on LRU and competitive performance elsewhere.

#### LongMemEval (ICLR 2025)

- **Paper**: [arXiv:2410.10813](https://arxiv.org/abs/2410.10813)
- **Dataset**: 500 questions, scalable histories (115K to 1.5M tokens)
- **Tests**: Information Extraction, Multi-Session Reasoning, Temporal Reasoning, Knowledge Updates, Abstention
- **Why it matters for RagZoom**: The scalable token range (115K to 1.5M) directly tests how systems degrade as history grows -- exactly where hierarchical compaction should shine. The three-stage evaluation framework (Indexing, Retrieval, Reading) lets RagZoom isolate which stage benefits from the tiling hierarchy. Commercial assistants show 30-60% accuracy drops moving from oracle to realistic retrieval; RagZoom's progressive disclosure could narrow this gap.
- **Used by**: MemOS, Zep, Memory-R1

#### BEAM (Beyond a Million Tokens)

- **Paper**: [arXiv:2510.27246](https://arxiv.org/abs/2510.27246) (Oct 2025)
- **Dataset**: 100 conversations, 2K questions, up to 10M tokens
- **Tests**: Contradiction resolution, event ordering, instruction following at extreme scale
- **Why it matters for RagZoom**: The 10M token scale is where flat retrieval collapses and hierarchical summarization becomes essential. BEAM's finding that even 1M-context LLMs with RAG struggle as dialogues lengthen is the exact problem RagZoom's compaction addresses.

#### Episodic Memory Benchmark (ICLR 2025)

- **Dataset**: 11 synthetic narrative datasets, 10K to 1M tokens, 36 question templates
- **Tests**: Spatiotemporal awareness, entity tracking, chronological relationships
- **Why it matters for RagZoom**: Performance degrades as related events increase -- directly testing whether a system can maintain coherent summaries across growing event clusters. RagZoom's bridging mechanism is designed for exactly this pattern.

### Medium Relevance

#### LoCoMo (ACL 2024) -- The De Facto Standard

- **Paper**: [arXiv:2402.17753](https://arxiv.org/abs/2402.17753) (Snap Research)
- **Dataset**: 10 conversations, ~300 turns each, ~9K tokens, spanning up to 35 sessions
- **Tasks**: Single-hop, multi-hop, temporal, open-domain, adversarial QA
- **Metrics**: Token-level F1, BLEU-1, LLM-as-a-Judge (binary correct/wrong via GPT-4o)
- **Status**: Used by Mem0, Letta, SimpleMem, MemOS, MemoryOS, Memory-R1. The lingua franca.
- **Why medium**: RagZoom must run LoCoMo for credibility (everyone does), but its small scale (9K tokens) doesn't stress hierarchical compaction. The benchmark tests factoid recall, not progressive granularity or working context. Well-known enough that systems may overfit.

#### MemBench (ACL 2025)

- **Focus**: Factual + reflective memory across multiple scenarios
- **Relevance**: The reflective memory dimension (going beyond stored facts to derive insights) maps to RagZoom's higher-height summary nodes. Worth monitoring but not the primary evaluation target.

#### EverMemBench (Feb 2026)

- **Focus**: Multi-party conversations, >1M tokens, temporally evolving
- **Relevance**: Multi-party and temporal evolution are relevant but the benchmark is too new to have established baselines.

#### PerLTQA (SIGHAN 2024)

- **Dataset**: 8,593 questions for 30 characters
- **Tests**: Cognitive science-inspired classification between semantic memory (facts, profiles, relationships) and episodic memory (events, dialogues, experiences)
- **Subtasks**: Memory Classification, Memory Retrieval, Memory Synthesis
- **Relevance**: The semantic/episodic distinction could inform how RagZoom handles different content types in its hierarchy.

### Low Relevance

#### Evo-Memory (DeepMind)

- **Paper**: [arXiv:2511.20857](https://arxiv.org/abs/2511.20857) (Nov 2025)
- **Tests**: Self-evolving experiential memory (strategy accumulation)
- **Environments**: AIME, GPQA, MMLU-Pro, ToolBench, AlfWorld, BabyAI, ScienceWorld
- **Why low**: Evaluates dynamic experiential memory (learning *how* to do things), not conversational recall or summarization. The ReMem framework's Think-Act-Memory Refine loop is architecturally interesting but orthogonal to RagZoom's recall use case.

#### WebChoreArena (2025)

- **Dataset**: 532 tasks, 117 "Massive Memory" tasks
- **Finding**: Gemini 2.5 Pro drops from 54.8% to 37.8%; GPT-4o shows 36-point drops on memory tasks
- **Why low**: Tests working memory during autonomous web browsing, not persistent conversational memory.

#### MEMTRACK (Patronus AI)

- **Tests**: Software development memory across Linear, Slack, Git
- **Finding**: GPT-5 achieves only 60% correctness with performance decay on follow-ups
- **Why low**: Multi-platform integration testing, not a memory architecture benchmark.

#### Other Benchmarks

| Benchmark | Focus | Notes |
|-----------|-------|-------|
| **MSC** (Meta, 2022) | Multi-session chat | Historical; superseded by LoCoMo |
| **MemoryRewardBench** | Reward models for memory management | Niche; reward model training |
| **GoodAI LTM Benchmark** | Configurable 4K-500K+ token tests | Practical deployment metrics; showed LTM agents match 128K context at 16% cost |
| **OP-Bench** (2025) | Over-personalization detection | 1,700 questions; all agents show 26-61% drops when memory is used inappropriately |

---

## System Profiles

### Letta (formerly MemGPT) -- The Uncomfortable Baseline

- **GitHub**: [letta-ai/letta](https://github.com/letta-ai/letta)
- **Paper**: [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) (Oct 2023)
- **Architecture**: OS-inspired -- the agent itself manages what stays in context vs external storage via function calls. Agent is active. Also maintains the **Letta Leaderboard** (May 2025), the most comprehensive industry rankings for agentic memory. Top performers: Claude Sonnet 4 (with Extended Thinking), GPT-4.1, GPT-4o.
- **Key result**: A trivial filesystem agent (GPT-4o-mini + grep/search) scored 74% on LoCoMo, beating every specialized system. "Simpler tools in the LLM's training data are used more effectively than specialized retrieval mechanisms."
- **How it differs from RagZoom**: Letta gives the LLM raw file tools and lets it figure out storage. No compression hierarchy, no token budget control. Works at LoCoMo scale (9K tokens) but has no answer for 1M+ token histories where you can't grep your way to relevance.

### MemOS (MemTensor) -- The Broadest Evaluator

- **GitHub**: [MemTensor/MemOS](https://github.com/MemTensor/MemOS) (4.9K stars, Apache 2.0)
- **Papers**: [arXiv:2505.22101](https://arxiv.org/abs/2505.22101), [arXiv:2507.03724](https://arxiv.org/abs/2507.03724)
- **Architecture**: MemCube abstraction unifying parametric (weights), activation (KV-cache), and plaintext (external) memory. Next-scene prediction for proactive preloading.
- **Benchmarks**: LoCoMo, LongMemEval, PersonaMem, PrefEval (broadest evaluation of any system)
- **Baselines compared**: Mem0, Zep, LangMem, OpenAI Memory, MIRIX, Memobase, MemU, Supermemory
- **How it differs from RagZoom**: Tries to unify all memory types (weights, KV-cache, external). Heavyweight and ambitious but doesn't offer progressive granularity. RagZoom's single-concern focus (external conversation memory with hierarchical compression) is a simpler, more defensible position.

### SimpleMem -- The Closest Architectural Parallel

- **GitHub**: [aiming-lab/SimpleMem](https://github.com/aiming-lab/SimpleMem) (MIT)
- **Paper**: [arXiv:2601.02553](https://arxiv.org/abs/2601.02553) (Jan 2026)
- **Architecture**: Three stages -- (1) entropy-aware semantic compression with coreference resolution and temporal anchoring, (2) recursive memory consolidation across semantic/lexical/symbolic indices, (3) intent-aware adaptive retrieval (k=3 to k=20 based on query complexity)
- **Benchmarks**: LoCoMo. Claims +26% F1 over Mem0, 30x fewer tokens, 14x faster construction.
- **How it differs from RagZoom**: Compression stage parallels RagZoom's summarization. Recursive consolidation maps to bridging/compaction. But SimpleMem's retrieval is still k-based (adaptive k, not token-budgeted). No continuous granularity control. No caller-specified budget.

### Mem0 -- The Market Leader by Stars

- **GitHub**: [mem0ai/mem0](https://github.com/mem0ai/mem0) (46K stars, Apache 2.0)
- **Paper**: [arXiv:2504.19413](https://arxiv.org/abs/2504.19413) (Apr 2025)
- **Architecture**: External pipeline extracts facts from conversations, stores in vector/graph DB, retrieves on demand. Agent is passive. Graph variant shows strength in temporal reasoning (58.1% vs 55.5% standard, vs OpenAI Memory's 21.7%).
- **Benchmarks**: LoCoMo only. Letta couldn't reproduce Mem0's MemGPT comparison numbers. No reproducible evaluation code released.
- **How it differs from RagZoom**: Pure fact extraction -- lossy by design. No summarization hierarchy, no granularity control. 90% token reduction claimed but at the cost of losing everything that isn't a discrete fact.

### Memory-R1 -- The RL Approach

- **Paper**: [arXiv:2508.19828](https://arxiv.org/abs/2508.19828)
- **Architecture**: RL-trained (PPO/GRPO) memory manager and answer agent. Only 152 training QA pairs.
- **Benchmarks**: LoCoMo (primary), MSC, LongMemEval
- **Key result**: +69% F1 over Mem0 on LoCoMo with LLaMA-3.1-8B backbone
- **How it differs from RagZoom**: Uses RL to learn *what* to remember. Complementary rather than competitive -- could theoretically be used to train RagZoom's query routing.

### MemoryOS (BAI-LAB) -- The Hierarchy Competitor

- **GitHub**: [BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS)
- **Paper**: [arXiv:2506.06326](https://arxiv.org/abs/2506.06326) (EMNLP 2025 Oral)
- **Architecture**: OS-inspired hierarchical storage (short/mid/long-term) with four modules (storage, updating, retrieval, generation)
- **Benchmarks**: LoCoMo, GVD
- **How it differs from RagZoom**: Uses discrete tiers (short/mid/long) rather than a continuous height hierarchy. No token-budgeted zoom -- the system decides what tier to pull from, not the caller.

### Claude-Mem -- The Vibes Competitor

- **GitHub**: [thedotmack/claude-mem](https://github.com/thedotmack/claude-mem) (20K stars, AGPL-3.0)
- **Architecture**: Claude Code plugin with hook-based capture, multi-provider AI compression, hybrid search (FTS5 + ChromaDB), 3-layer progressive disclosure
- **Benchmarks**: **None.** All performance claims are self-reported. Pure vibes-based adoption.
- **How it differs from RagZoom**: Same deployment target (Claude Code memory) but no hierarchical compaction, no token-budgeted recall. Competing on ease-of-install, not capability.

### Cognee

- **GitHub**: [topoteretes/cognee](https://github.com/topoteretes/cognee)
- **Architecture**: Graph + vector hybrid memory
- **Benchmarks**: HotPotQA (not LoCoMo). Claims 0.93 human-like correctness, outperforming Mem0 and LightRAG.
- **How it differs from RagZoom**: Graph-first approach. 20,000x construction overhead vs simple approaches.

---

## Standard Metrics Across the Field

| Category | Metrics | Notes |
|----------|---------|-------|
| **Correctness** | Token-level F1, Exact Match, BLEU-1 | F1 most common; LoCoMo standard |
| **Quality** | LLM-as-a-Judge (GPT-4o binary accuracy) | Increasingly the headline number |
| **Efficiency** | Tokens/query, construction time, retrieval latency, p95 | Mem0 graph: 2.6s p95; OpenAI Memory: 0.9s |
| **Competency** | AR, TTL, LRU, CR scores | MemoryAgentBench framework |
| **Robustness** | Adversarial success rate, sequence robustness, over-personalization rate | OP-Bench: 26-61% drops from inappropriate memory use |

LongMemEval's three-stage framework (Indexing, Retrieval, Reading) is particularly useful for isolating bottlenecks.

---

## RagZoom Positioning & Evaluation Strategy

### Competency Map

| Competency | RagZoom's Position | Strength | Gap |
|------------|-------------------|----------|-----|
| **Accurate Retrieval** | Semantic search + height-0 verbatim nodes | Competitive | No k-NN tuning; relies on tiling seed selection |
| **Long-Range Understanding** | Hierarchical summaries at variable heights | **Unique strength** | Untested on standard benchmarks |
| **Temporal Reasoning** | Chronological node ordering with step-based bridging | Structural advantage (time is first-class) | No temporal index or event graph |
| **Conflict Resolution** | Latest summary overwrites earlier during compaction | Implicit (newest wins) | No explicit contradiction detection |
| **Token Efficiency** | Caller-controlled token budget | **Unique strength** | No published efficiency comparisons |
| **Abstention** | Not currently modeled | Absent | Would need explicit "I don't know" signal |
| **Progressive Granularity** | Continuous height spectrum from verbatim to high-level summary | **Unique -- no competitor offers this** | No benchmark exists to test it |

### Strongest Competitive Claims

1. **"No other system lets the caller control the detail/token trade-off."** Every competitor returns either fixed-k results or all-or-nothing context. RagZoom's token-budgeted zoom is architecturally unique. This is the primary differentiator.

2. **"Hierarchical compaction scales where flat retrieval collapses."** BEAM shows RAG + 1M-context LLMs degrade at scale. GoodAI shows LTM agents with 8K context match 128K context at 16% cost. RagZoom's compaction is designed for exactly this regime.

3. **"Simple tools beat complex retrieval" -- and RagZoom is simple tooling.** Letta's filesystem result (74% LoCoMo) proves that tools the LLM understands beat specialized APIs. RagZoom's `recall` tool is a single function with two parameters (query + budget). It's closer to "simple grep" than to Mem0's extraction pipeline.

4. **"Background compaction means zero query-time extraction cost."** Mem0, HippoRAG-v2, and Cognee show 20,000x higher construction time than simple approaches, but this cost hits at write time. RagZoom's compaction runs asynchronously in the background -- neither write latency nor read latency suffers.

### Proposed Benchmarks to Run

#### Tier 1: Credibility (run first)

| Benchmark | Why | Expected Outcome |
|-----------|-----|-----------------|
| **LoCoMo** | Table stakes -- every competitor reports numbers here | Competitive with Mem0/Letta; scale is too small for RagZoom's compaction to shine, but must not embarrass |
| **LongMemEval** | Scalable to 1.5M tokens; three-stage evaluation isolates RagZoom's strengths | Should outperform flat retrieval at higher token scales; the Indexing-Retrieval-Reading decomposition reveals where the hierarchy helps |

#### Tier 2: Differentiation (run second)

| Benchmark | Why | Expected Outcome |
|-----------|-----|-----------------|
| **MemoryAgentBench** | The LRU competency directly tests what RagZoom is built for | Strong on LRU; competitive on AR; likely weak on CR (acceptable -- everyone is) |
| **BEAM** | 10M token scale is RagZoom's home turf | Should demonstrate graceful degradation where RAG-only systems collapse |

#### Tier 3: RagZoom-Specific (design and publish)

| Test | What It Measures |
|------|-----------------|
| **Budget-Accuracy Curve** | Given fixed history, vary token budget from 500 to 50K tokens. Measure answer quality at each budget. No competitor can run this test. |
| **Granularity Appropriateness** | For the same history, ask high-level ("summarize the project") and specific ("what was the exact error message?") questions. Measure whether zoom returns the right height level for each. |
| **Compaction Efficiency** | Measure recall quality as a function of compaction ratio (original tokens / compacted tokens). At what ratio does quality degrade? |
| **Latency Under Growth** | Plot p95 recall latency as conversation history grows from 10K to 10M tokens. Demonstrate sublinear growth from the hierarchy. |

### Gaps RagZoom Uniquely Fills

**The granularity gap.** Every existing benchmark treats memory as binary -- you either retrieve the right fact or you don't. No benchmark asks "can you answer at the right level of detail for a given token budget?" This is the core capability RagZoom provides and the market has no way to evaluate it yet. Publishing this benchmark would define the evaluation category.

**The budget-controlled retrieval gap.** Current systems return either k results (Mem0, SimpleMem) or the full context (raw LLM). None let the caller say "give me the best answer you can in 2,000 tokens." This is a product feature that maps directly to real deployment constraints (context window limits, cost control, latency targets).

**The progressive compaction gap.** MemoryOS has discrete tiers (short/mid/long). SimpleMem has recursive consolidation. Neither produces a continuous hierarchy where the same content exists at multiple granularity levels simultaneously. RagZoom's tiling means a single `recall` call can mix height-0 verbatim content (for the seed/relevant area) with height-N summaries (for surrounding context) in a single response.

**The "when not to remember" gap.** OP-Bench shows 26-61% drops from inappropriate memory use. RagZoom's query-driven seed selection naturally limits what gets surfaced -- if the query doesn't match anything, the budget gets filled with high-level summaries rather than irrelevant specifics. This architectural property could be a natural defense against over-personalization, but it needs testing.

### Immediate Next Steps

1. **Run LoCoMo.** Get a number on the board. Even a mediocre result is fine -- the narrative is "competitive at small scale, dominant at large scale."
2. **Run LongMemEval at multiple scales.** This is where the compaction hierarchy should demonstrably outperform. Plot the degradation curve versus scale and compare against Mem0/Letta baselines.
3. **Design and publish the Budget-Accuracy Curve benchmark.** This defines the category. No competitor can run it. It makes RagZoom the evaluation standard for token-budgeted retrieval.
4. **Measure against SimpleMem directly.** Closest architectural parallel. Head-to-head on LoCoMo + a scale benchmark would clarify whether RagZoom's continuous hierarchy beats SimpleMem's adaptive-k approach.

---

## Key References

- [Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) -- comprehensive paper tracker
- [Memory in the Age of AI Agents](https://arxiv.org/abs/2512.13564) -- 102-page survey (Dec 2025) proposing Forms-Functions-Dynamics taxonomy
- [Letta Benchmark Critique](https://www.letta.com/blog/benchmarking-ai-agent-memory) -- the filesystem agent result + Letta Leaderboard
- [MemoryBench meta-analysis](https://arxiv.org/abs/2510.17281) -- found no advanced system consistently beats simple RAG
