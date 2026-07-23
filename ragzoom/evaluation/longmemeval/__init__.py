"""LongMemEval benchmark evaluation harness for RagZoom.

LongMemEval evaluates long-term interactive memory over multi-session
haystacks that exceed the reader's context window (the ``_S`` ~115K and ``_M``
~1.5M tiers). This harness builds RagZoom trees from a haystack's sessions and
answers each question via the recall agent under a fixed token budget B — the
core knob of the decisive summarizer experiment.
"""
