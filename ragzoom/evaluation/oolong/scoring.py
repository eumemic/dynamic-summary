"""Deterministic scoring for the Oolong (oolong-real) benchmark.

This is where Oolong differs fundamentally from LongMemEval. LongMemEval grades
each answer with a type-specific LLM-as-judge that returns a binary yes/no.
Oolong instead scores answers with a *deterministic, distributional* metric that
gives partial credit — there is no judge model, no API call, no judge noise.

The metric is taken verbatim from the upstream reference implementation
(``abertsch72/oolong`` ``src/eval/eval_helpers.py::dnd_process_response``). The
gold answer's *form* selects the metric:

* **numeric**  ``score = 0.75 ** |gold - pred|``  — exponential partial credit,
  so an off-by-one count still scores 0.75 (this is the heart of "aggregation
  under compression": being approximately right counts).
* **label**    ``score = 1.0 if gold == pred (case-insensitive) else 0.0``.
* **list**     ``score = |gold ∩ pred| / |gold|`` — set recall over the gold
  items (the upstream code divides by the gold size, not the F1 denominator).

A type mismatch between the parsed gold and the parsed prediction scores 0.0.

Model answers are expected inside ``\\boxed{...}`` (the format the oolong-real
prompt asks for); if no box is present we fall back to the raw output, matching
upstream's best-effort parse.
"""

from __future__ import annotations

import re

# Upstream regex (eval_helpers.dnd_parse_response): prefer the \text{...} form,
# fall back to a bare \boxed{...}. Both are matched here exactly.
_BOXED_TEXT_RE = re.compile(r"\\boxed\{\\text\{([^}]*)\}\}")
_BOXED_RE = re.compile(r"\\boxed[{]+([^}]*)[}]+")


def extract_boxed_answer(output: str) -> str:
    """Extract the answer from ``\\boxed{...}`` / ``\\boxed{\\text{...}}``.

    Falls back to the raw output when no box is present — mirroring the upstream
    parser, which treats a box-less response as a low-confidence whole-string
    candidate rather than a hard failure.
    """
    m = _BOXED_TEXT_RE.search(output) or _BOXED_RE.search(output)
    if m is not None:
        return m.group(1).strip()
    return output.strip()


def parse_answer(answer: str) -> int | str | list[str]:
    """Parse an answer string into int, list-of-str, or str.

    Order matters and matches upstream ``dnd_parse_answer``:
    1. an integer if it parses as one,
    2. otherwise a comma-separated list if it contains a comma,
    3. otherwise a plain string.
    """
    try:
        return int(answer)
    except ValueError:
        pass

    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]

    return answer


# The gold answer is parsed with the same rules as the prediction.
parse_gold_answer = parse_answer


def score_parsed(
    gold: int | str | list[str],
    pred: int | str | list[str],
) -> float:
    """Score a parsed prediction against a parsed gold answer.

    Type-matched scoring; a mismatch (e.g. numeric gold vs string prediction)
    scores 0.0. Kept separate from parsing so the metric can be unit-tested on
    already-typed values.
    """
    if isinstance(gold, int) and isinstance(pred, int):
        return 0.75 ** abs(gold - pred)
    if isinstance(gold, str) and isinstance(pred, str):
        return float(gold.strip().lower() == pred.strip().lower())
    if isinstance(gold, list) and isinstance(pred, list):
        if not gold:
            return 0.0
        overlap = set(gold) & set(pred)
        return len(overlap) / len(gold)
    return 0.0


def score_answer(*, gold: str, model_output: str) -> float:
    """Score a raw model output against a gold answer string.

    Extracts the ``\\boxed{}`` answer from ``model_output``, parses both sides
    into int/str/list, and applies the Oolong metric. Returns a float in [0, 1].
    """
    parsed_pred = parse_answer(extract_boxed_answer(model_output))
    parsed_gold = parse_gold_answer(gold)
    return score_parsed(parsed_gold, parsed_pred)
