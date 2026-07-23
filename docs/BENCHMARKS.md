# RagZoom — Honest Benchmark Results

*Experiments run June 2026. This document reports the results as measured, including the ones where RagZoom loses. An earlier apparent win was retracted when a larger sample reversed it (see [Aggregation](#aggregation-regime-oolong-r--b)).*

**Summary:** RagZoom is a coverage-preserving memory operator with a characterizable envelope. On needle-style QA over histories that fit in context (the regime where a resolution-reducing operator is *expected* to be dominated), its default coverage retrieval **loses badly to flat top-k RAG (43% vs 80%)**. Controlled experiments pinned the mechanism: a `concentrate` retrieval mode — top-k over the tree's own verbatim leaves — recovers **+16.7pp in a verified A/B**, proving the gap is a retrieval-*objective* choice, not summarization loss. On the aggregation regime (histories that physically cannot fit in context), RagZoom **works where full-context physically cannot run**, but does not beat flat top-k on accuracy. The durable contribution is the envelope characterization and the per-query coverage/concentrate unification, not a leaderboard win.

## Needle regime (LongMemEval-S, R ≤ B)

N=30 (seed-42, identical question set across every arm), fixed reader budget B=8192, fixed answerer `gpt-5-mini`, fixed judge `gpt-4.1`. Only the memory strategy varies.

| Strategy | Budget | Overall | Ingest cost | Query $ | tok/query |
|---|---|---|---|---|---|
| full-context ("stuff it all in") | ∞ | **80.0%** | 0 | $0.793 | 111K |
| flat top-k RAG | 8192 | **80.0%** | 0 | $0.062 | 8.2K |
| RagZoom (Sonnet summarizer) | 8192 | 43.3% | ~13 h | $0.111 | 18.6K |
| RagZoom (nano summarizer) | 8192 | 33.3% | ~3.5 h | $0.111 | 18.2K |
| truncation (recency) | 8192 | 10.0% | 0 | $0.062 | 8.0K |

Flat top-k Pareto-dominates RagZoom here. This is thesis-consistent: LongMemEval-S (~115K tokens) fits in context, so the relevant evidence fits the budget — the regime where a coverage-preserving operator is supposed to be dominated.

### Why it loses (the investigation)

The naive reading — "summarization blurs the needle" — is **wrong**, and the data says so:

- The verbatim needle was served on **30/30 questions** (every served tiling contains a height-0 leaf span). RagZoom is not starving the reader of evidence.
- On the 11 questions where flat top-k won and RagZoom-Sonnet lost, **8 had the gold-evidence turn served verbatim** to the answerer. The answer was in context and the reader still got it wrong. Only 3/11 were genuine retrieval misses.

Representative failure (multi-session aggregation): *"total spent on gifts for my coworker and brother?"* — gold $200. Flat top-k served a clean, relevant-only set → `gpt-5-mini` answered $200 ✓. RagZoom served the same two $100 gifts verbatim **plus** coverage-spread distractors → the same reader mis-aggregated to $295 ✗.

Three measured causes, none fundamental to the architecture:

1. **Summarizer quality** (nano → Sonnet): +10pp.
2. **Reader quality** (`gpt-5-mini` → Opus, trees held fixed): +20pp.
3. **Coverage-biased recall policy** — the structural residual that capability upgrades don't remove: coverage spreads the budget across the whole timeline; a needle wants the budget *concentrated* on the most query-relevant verbatim chunks.

Plus a robustness bug (an empty-tiling recall path hard-raises instead of degrading) that fired ~10× in the Sonnet run — every RagZoom number here is a slight lower bound.

### Summarizer × reader capability matrix

All-Claude, both axes ∈ {haiku, sonnet, opus}; trees built once per summarizer and reused. LongMemEval-S, N=30, B=8192.

| trees ↓ / reader → | haiku | sonnet | opus |
|---|---|---|---|
| **haiku** | 30.0% | 36.7% | 50.0% |
| **sonnet** | 53.3% | 46.7% | 63.3% |

Both axes contribute ~+10–20pp and roughly stack; the reader is the single biggest knob. But the best combo (~63%) still sits ~17pp under flat top-k's 80% — a residual **invariant to capability**, the fingerprint of the structural cause. (At N=30, per-cell noise is ~±9pp; the matrix shows the shape, not reliable adjacent-cell rankings.)

### Concentrate mode: the controlled confirmation

A `concentrate` retrieval mode — rank the verbatim leaves by the query-relevance score RagZoom already computes, admit the top ones until budget — A/B'd against the default on LongMemEval-S, N=30, identical otherwise:

| retrieval mode (nano summarizer, gpt-5-mini reader, B=8192) | overall | served tiling |
|---|---|---|
| coverage (default) | 40.0% | 352 spans, heights 0–5, 13% verbatim |
| **concentrate** | **56.7%** | 43 spans, 100% verbatim leaves |

**+16.7 points from one objective switch**, verified to have engaged (the tiling is provably leaves-only), with gains landing exactly on the needle slices and the aggregation slice untouched. This is the controlled proof that the coverage-vs-concentration policy — not summarizer or reader quality — was the dominant cause of the needle gap, and that RagZoom can *contain* flat-RAG as a per-query mode.

## Aggregation regime (Oolong, R > B)

Windows spanning 50K–1.19M tokens — genuinely bigger than any context budget (the full-context baseline physically overflows `gpt-5-mini`'s 272K window on the large ones, recorded as failures). Deterministic scoring, B=8192, `gpt-5-mini` reader.

A methodology note that matters: an ingest truncation bug in early runs (whole 150–200K-char episodes silently truncated to 50K) meant RagZoom initially saw ~⅓ of the history while baselines saw all of it. It was fixed and verified (0 truncations) before any number below was recorded.

**The honest verdict, after firming up N=20 → N=60:**

| arm | N=20 | **N=60 (firm)** |
|---|---|---|
| flat top-k RAG | 16.0% | **21.0%** |
| RagZoom — Sonnet summarizer | 19.3% | 17.3% |
| RagZoom — nano summarizer | 18.0% | 16.5% |
| full-context | 10.6% (overflow) | 13.7% (overflow) |
| truncation | 8.4% | 10.2% |

At N=20, RagZoom appeared to beat flat-RAG and the niche looked real. **At N=60 that reversed** — the N=20 edge was sampling noise (the per-type story flipped too). So on this configuration, RagZoom does **not** beat flat top-k retrieval in its own predicted-win regime, and the earlier claim is retracted.

What survives, stated narrowly: RagZoom reliably beats the approaches that **can't scale** — full-context (physically overflows on large windows; it cannot run, not just runs worse) and truncation. When the history genuinely doesn't fit, RagZoom works and stuffing can't. That is a real, regime-defined advantage — not an accuracy win over flat retrieval.

Untested levers (noted, not claimed): the Oolong runs used the weakest reader (`gpt-5-mini`; the matrix showed +20pp from a stronger one), larger budgets B, and the concentrate mode (a needle-regime fix, untested here).

## Caveats

- N=30 needle / N=60 aggregation; per-cell variance on the matrix is ~±9pp. Solid on the big gaps, directional on adjacent cells.
- The empty-tiling recall bug depresses all RagZoom numbers (lower bounds).
- LongMemEval-**S** is needle-heavy and fits in context; Oolong is the only R>B benchmark run.

## Reproducibility

Raw per-question results for every arm are available on request. The LiteLLM-based multi-model harness used for these runs (baseline arms, the capability matrix, and the concentrate A/B) is being upstreamed to this repository.
