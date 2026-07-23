"""Oolong (oolong-real) aggregation benchmark harness for RagZoom.

Oolong (arXiv:2511.02817) is purpose-built for *information aggregation*: every
question's answer is a statistic over the whole transcript — a count, a ranking,
an ordered list — so no single chunk contains it. The oolong-real split poses
these questions over live-play Dungeons & Dragons transcripts (Critical Role /
CRD3), with human-annotated gold answers from CritRoleStats.

This is RagZoom's *home regime*. Where LongMemEval still leans needle (one fact,
one session), Oolong rejects retrieval-style tasks outright and demands a
faithful overview of the entire history — exactly what a multi-resolution,
token-budgeted tile is for. The harness builds RagZoom trees from a context
window's episodes and answers each question via the recall agent under a fixed
token budget B (the H/B knob), scoring with Oolong's own deterministic metric
(partial-credit for numbers, exact-match for labels, set recall for lists) — not
the binary LLM-judge LongMemEval uses.
"""
