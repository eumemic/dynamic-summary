"""Dataset-agnostic machinery shared by the LongMemEval and Oolong baselines.

Both baselines answer their benchmark WITHOUT RagZoom by turning a question's
history into a flat context string under one of three strategies, then asking a
held-constant answer model. The *strategies* are identical across benchmarks —
only the way a history is rendered into chronological lines, the query text, the
answer prompt, and the scoring differ. Everything that does not depend on the
dataset lives here so the two harnesses share it rather than copy it:

  * :class:`BaselineStrategy` — the three arms (``full`` / ``truncate`` /
    ``topk``);
  * the matched-budget *truncate* walk and the flat-RAG *topk* admission, both
    operating on a pre-rendered ``list[str]`` of chronological lines so each
    harness supplies only its renderer;
  * context-overflow detection (a provider context-length rejection is a
    reportable outcome, never a crash);
  * the seed-42 sample that pins the baseline's question set to the RagZoom
    runner's;
  * the served-context descriptor recorded where RagZoom records its tilings.

The embedder is injected (:class:`EmbedderProtocol`) so the whole pipeline is
unit-tested with mocks; the CLI is the only place that builds the real client.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol, TypeVar

from ragzoom.agent.protocol import BenchmarkingAgent, CostMetrics
from ragzoom.utils.tokenization import count_tokens, decode_tokens, encode_text

logger = logging.getLogger(__name__)

# The embedding model the RagZoom harness indexes with — reused for the flat-RAG
# baseline so retrieval quality is attributable to the strategy, not the encoder.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# What the answerer is handed in place of an answer when the context overflows
# the model's window: an explicit statement, so a downstream judge/scorer grades
# reality rather than a fabricated answer.
OVERFLOW_ANSWER = (
    "Context overflow: the conversation history is too long to fit in the "
    "answer model's context window, so this question could not be answered."
)


class BaselineStrategy(str, Enum):
    """How a question's history is turned into the answerer's context."""

    FULL = "full"
    TRUNCATE = "truncate"
    TOPK = "topk"


class EmbedderProtocol(Protocol):
    """The slice of an embedding model the top-k baseline depends on."""

    async def embed(
        self, texts: Sequence[str]
    ) -> list[list[float]]:  # pragma: no cover - protocol surface
        ...


# ---------------------------------------------------------------------------
# Sampling — byte-for-byte the runner's seed-42 logic
# ---------------------------------------------------------------------------

_T = TypeVar("_T")


def sample_seed42(items: list[_T], sample_size: int | None) -> list[_T]:
    """Sample exactly as the RagZoom runners do (``random.seed(42)``).

    Reusing the identical seed + ``random.sample`` call is what guarantees a
    baseline scores the *same* question set as ``--sample N`` of the matching
    RagZoom runner — the whole point of the head-to-head comparison. ``None``
    means "use them all".
    """
    if sample_size is None:
        return items
    random.seed(42)
    return random.sample(items, min(sample_size, len(items)))


# ---------------------------------------------------------------------------
# Line-based context construction (full / truncate)
# ---------------------------------------------------------------------------


def full_context(lines: list[str]) -> str:
    """The whole history, every line, in the order given. No budget."""
    return "\n".join(lines)


def truncate_lines_to_budget(lines: list[str], budget: int) -> str:
    """Keep the most-recent lines until ``budget`` tokens are reached.

    Walks lines newest-first, prepending each that still fits, so the answerer
    gets a contiguous tail of the history that honours the same budget B RagZoom
    is held to. A single line larger than the whole budget is itself truncated
    (keeping its end) rather than dropped to an empty context.
    """
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = count_tokens(line) + 1  # +1 for the joining newline
        if used + cost <= budget:
            kept.append(line)
            used += cost
            continue
        if not kept:
            # The most-recent line alone exceeds the budget: keep its tail so
            # the answerer is not handed an empty context.
            return truncate_text_to_budget(line, budget)
        break
    kept.reverse()
    return "\n".join(kept)


def truncate_text_to_budget(text: str, budget: int) -> str:
    """Keep the last ``budget`` tokens of ``text`` (truncate from the start)."""
    tokens = encode_text(text)
    if len(tokens) <= budget:
        return text
    return decode_tokens(tokens[-budget:])


# ---------------------------------------------------------------------------
# Top-k flat retrieval
# ---------------------------------------------------------------------------


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x * y for x, y in zip(a, b, strict=True)), 0.0)


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; zero vectors score 0 (never divide by zero)."""
    na: float = math.sqrt(_dot(a, a))
    nb: float = math.sqrt(_dot(b, b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)


@dataclass(frozen=True)
class TopKResult:
    """A flat-RAG context plus the number of chunks it admitted.

    ``num_chunks`` is the flat-RAG analogue of RagZoom's retrieval-call count; it
    is carried explicitly rather than re-derived from the text because a chunk
    may contain internal newlines, so counting lines would over-count chunks.
    """

    context: str
    num_chunks: int


async def topk_lines(
    *,
    query: str,
    lines: list[str],
    embedder: EmbedderProtocol,
    budget: int,
) -> TopKResult:
    """Flat top-k retrieval: the most query-similar lines until ``budget``.

    Each line is one chunk. We embed the query and every chunk once, rank chunks
    by cosine similarity, then greedily admit the highest-ranked chunks that
    still fit the budget. The admitted chunks are emitted in their original
    (chronological) order so the answerer reads a coherent — if sparse — history
    rather than a relevance-sorted jumble, and the admitted count is returned
    alongside (the flat-RAG analogue of RagZoom's recall-call count).

    This is the structurally decisive arm: flat top-k can only surface the *k*
    most-similar chunks, so on an aggregation question it cannot represent the
    whole-history distribution the answer is a statistic over.
    """
    if not lines:
        return TopKResult(context="", num_chunks=0)

    vectors = await embedder.embed([query, *lines])
    query_vec = vectors[0]
    chunk_vecs = vectors[1:]

    scored = sorted(
        range(len(lines)),
        key=lambda i: cosine(query_vec, chunk_vecs[i]),
        reverse=True,
    )

    admitted: set[int] = set()
    used = 0
    for idx in scored:
        cost = count_tokens(lines[idx]) + 1
        if used + cost > budget:
            continue
        admitted.add(idx)
        used += cost

    context = "\n".join(lines[i] for i in range(len(lines)) if i in admitted)
    return TopKResult(context=context, num_chunks=len(admitted))


# ---------------------------------------------------------------------------
# Overflow detection
# ---------------------------------------------------------------------------

_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
    "reduce the length",
)


def is_context_overflow_error(exc: BaseException) -> bool:
    """Is this exception a provider rejection for an over-long context?

    Matches OpenAI's ``BadRequestError`` carrying the ``context_length_exceeded``
    signature (in its ``code`` or its message). We deliberately key on the
    library's own error class plus the documented marker strings rather than a
    bare substring search, so an unrelated 400 is not misread as an overflow.
    """
    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency
        return False

    if not isinstance(exc, openai.BadRequestError):
        return False

    haystack = str(exc).lower()
    body = exc.body
    if isinstance(body, dict):
        code = body.get("code")
        if isinstance(code, str):
            haystack += " " + code.lower()
    return any(marker in haystack for marker in _OVERFLOW_MARKERS)


# ---------------------------------------------------------------------------
# Served-context descriptor
# ---------------------------------------------------------------------------


def served_descriptor(
    label: str,
    context_tokens: int,
    *,
    retrieved_chunks: int | None = None,
    overflow: bool = False,
) -> str:
    """A short, human-readable record of what the answerer was served.

    Stored in ``served_tilings`` (where RagZoom records the tilings it served) so
    failure attribution reads the same field for both kinds of run. For top-k it
    also notes how many chunks were retrieved (the flat-RAG analogue of RagZoom's
    retrieval-call count).
    """
    if overflow:
        return (
            f"[{label}] context_overflow: {context_tokens} tokens exceed the "
            "answer model's context window; no answer was produced"
        )
    if retrieved_chunks is not None:
        return (
            f"[{label}] {retrieved_chunks} chunks, {context_tokens} tokens "
            "context, no RagZoom"
        )
    return f"[{label}] {context_tokens} tokens context, no RagZoom"


# ---------------------------------------------------------------------------
# Serve the context and answer (shared try/overflow/descriptor flow)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServedAnswer:
    """The outcome of building a context and calling the answerer.

    Carries exactly the fields both harnesses' result-builders need — the raw
    generated answer (for the judge or the deterministic scorer), the cost, and
    the single served-context descriptor — so each harness keeps only its own
    result-type assembly and scoring, not the serve/overflow plumbing.
    """

    generated_answer: str
    cost: CostMetrics
    served_tiling: str


async def serve_and_answer(
    *,
    answer_backend: BenchmarkingAgent,
    lines: list[str],
    query: str,
    strategy: BaselineStrategy,
    budget: int | None,
    embedder: EmbedderProtocol | None,
    answer_model: str,
    answer_system_prompt: str,
    build_user_prompt: Callable[[str], str],
    question_label: str,
) -> ServedAnswer:
    """Build the strategy's context, call the answerer, and handle overflow.

    The dataset-specific inputs are the rendered ``lines``, the retrieval
    ``query`` (used only by ``topk``), the answer system prompt, and a
    ``build_user_prompt`` that wraps the assembled context in the harness's
    question framing. Everything else — the three-way strategy dispatch, the
    no-temperature ``generate`` call, the context-overflow recording, and the
    served-context descriptor — is identical across harnesses and lives here.

    A context-length rejection is caught and recorded as an explicit
    ``context_overflow`` outcome (a stated-overflow answer + zero cost), never
    crashed and never silently truncated. The temperature is left at the model
    default to match the RagZoom search path (forcing it breaks gpt-5 models and
    makes the comparison non-apples-to-apples).
    """
    label = f"baseline:{strategy.value}"

    retrieved_chunks: int | None = None
    if strategy is BaselineStrategy.TOPK:
        if budget is None:
            raise ValueError("topk strategy requires a token budget")
        if embedder is None:
            raise ValueError("topk strategy requires an embedder")
        topk = await topk_lines(
            query=query, lines=lines, embedder=embedder, budget=budget
        )
        context = topk.context
        # The admitted-chunk count is the flat-RAG analogue of RagZoom's
        # retrieval-call count (k retrieved chunks vs N recall calls).
        retrieved_chunks = topk.num_chunks
    elif strategy is BaselineStrategy.TRUNCATE:
        if budget is None:
            raise ValueError("truncate strategy requires a token budget")
        context = truncate_lines_to_budget(lines, budget)
    else:
        context = full_context(lines)

    context_tokens = count_tokens(context)
    user_prompt = build_user_prompt(context)

    overflow = False
    try:
        result = await answer_backend.generate(answer_system_prompt, user_prompt)
        generated_answer = result.answer
        cost = result.cost
        if retrieved_chunks is not None:
            # Record k (retrieved chunks) where RagZoom records its recall-call
            # count, so the cost block is comparable across both kinds of run.
            cost = replace(cost, retrieval_call_count=retrieved_chunks)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is an overflow
        if not is_context_overflow_error(exc):
            raise
        logger.warning(
            "Context overflow on %s (%d tokens, model=%s): recording as "
            "context_overflow",
            question_label,
            context_tokens,
            answer_model,
        )
        overflow = True
        generated_answer = OVERFLOW_ANSWER
        cost = CostMetrics.zero()

    return ServedAnswer(
        generated_answer=generated_answer,
        cost=cost,
        served_tiling=served_descriptor(
            label,
            context_tokens,
            retrieved_chunks=retrieved_chunks,
            overflow=overflow,
        ),
    )
