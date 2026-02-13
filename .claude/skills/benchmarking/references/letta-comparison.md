# Letta Leaderboard Comparison Methodology

## Source Code

Letta's benchmark code: https://github.com/letta-ai/letta-leaderboard

Key files:
- `leaderboard/utils.py` — `GRADER_TEMPLATE`, `grade_sample()`, `request_openai()`
- `leaderboard/locomo/locomo_benchmark.py` — `LoCoMoQAFileBenchmark`

## Judge Configuration

Letta uses `request_openai()` with these defaults:
- **Model**: `gpt-4.1` (not gpt-4o or gpt-4o-mini)
- **Temperature**: 0
- **Max tokens**: 2048
- **System message**: "You are a helpful assistant."

The grader prompt is sent as a **user message** (not system), with the system message being the generic "You are a helpful assistant."

## Three-Way Grading

Letta's `GRADER_TEMPLATE` produces A/B/C verdicts:
- **A (CORRECT)**: Gold target fully contained, no contradictions
- **B (INCORRECT)**: Factual contradiction present (even with hedging)
- **C (NOT_ATTEMPTED)**: Key info missing, no contradictions

Scoring: `1.0 if result == "A" else 0.0` — both B and C count as failures.

## Key Rubric Rules

These rules significantly affect scoring compared to a naive judge:

1. **Semantic meaning matters, surface form doesn't**: Capitalization, punctuation, grammar, order are irrelevant.
2. **Hedging is OK**: "I think it might be X" counts as correct if X matches the gold target.
3. **Partial gold targets**: If gold has more info than the question asks, answering just the question part is correct.
4. **Inferable omissions**: "San Francisco" is correct for "San Francisco, California" when the question asks about a city.
5. **Typo tolerance**: "Hyoong Won Choong" matches "Hyung Won Chung".
6. **Numeric precision**: Must be correct to last significant figure of gold. "120k" gold → "124k" correct, "100k" incorrect, "around 100k" not attempted.

## Letta's Retrieval Setup

Letta's LoCoMo benchmark creates file-based memory:
1. Conversations are chunked (by session, turn, time window, or SeCom segmentation)
2. Files are uploaded to a Letta data source with embeddings
3. Agent uses `search_files` tool to grep/search conversation files
4. Agent uses `answer_question` tool to return answer

The agent has access to the **full conversation text** through file search — there is no token budget constraint. This is fundamentally different from RagZoom's budgeted retrieval.

## What This Means for Comparison

Letta's 74% accuracy uses:
- Full conversation access (~9K tokens per conversation — small enough to fit)
- GPT-4o-mini as the agent (file search + answer generation)
- GPT-4.1 as judge with detailed rubric

RagZoom's comparison is fair when:
- Using the same judge (gpt-4.1) with the same rubric (GRADER_TEMPLATE)
- Using the same answer model (gpt-4o-mini)
- Acknowledging the retrieval difference: RagZoom retrieves a token-budgeted subset

The budget-accuracy curve is the key differentiator — plot where RagZoom matches Letta's accuracy and at what token cost. At LoCoMo scale (~9K tokens), the budget constraint is the main handicap. At larger scales (100K+ tokens), RagZoom's compression should be the advantage.

## Reproducing Letta's Numbers

To verify Letta's reported numbers, their leaderboard code can be run directly:
```bash
git clone https://github.com/letta-ai/letta-leaderboard
cd letta-leaderboard
# Follow their setup instructions
```

Their leaderboard shows per-model results. The "filesystem agent" result (74%) is the key comparison point — it uses GPT-4o-mini with basic file tools, proving that simple retrieval + good tools beats complex memory systems at LoCoMo scale.
