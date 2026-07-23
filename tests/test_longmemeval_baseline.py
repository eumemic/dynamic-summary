"""Unit tests for the LongMemEval naive-baseline harness.

The baseline answers questions WITHOUT RagZoom by stuffing the haystack
straight into the answer model. Every LLM/embedding/judge side effect is
mocked — no real API calls, no server, no dataset download. These tests lock
down the four behaviours the experiment depends on:

  * ``full`` concatenates every session/turn in chronological order with
    timestamp + role prefixes (the "stuff it all in" baseline);
  * ``truncate`` honours a token budget by keeping the most-recent turns;
  * a provider context-length rejection is recorded as ``context_overflow``
    rather than crashing the run or silently truncating;
  * the seed-42 sample hits the exact same question ids as the RagZoom runner,
    so the two harnesses score the same questions;
  * ``results.json`` matches the shared schema an analysis script already
    consumes, and the judge is wired with the right arguments.
"""

from __future__ import annotations

import json
import random
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from ragzoom.agent.protocol import (
    AgentResult,
    BenchmarkingAgent,
    CostMetrics,
    ToolDefinition,
)
from ragzoom.evaluation.baseline_common import (
    BaselineStrategy,
    EmbedderProtocol,
    is_context_overflow_error,
)
from ragzoom.evaluation.longmemeval.baseline import (
    BaselineConfig,
    answer_one,
    build_context,
    chronological_turns,
    run_baseline,
)
from ragzoom.evaluation.longmemeval.types import (
    LongMemEvalQuestion,
    QuestionType,
    Session,
    Turn,
    parse_longmemeval_file,
)

# ---------------------------------------------------------------------------
# Fixtures — two questions whose sessions are deliberately out of file order
# so a chronological-sort assertion has teeth.
# ---------------------------------------------------------------------------

_FIXTURE: list[dict[str, object]] = [
    {
        "question_id": "q-multi-1",
        "question_type": "multi-session",
        "question": "Where is the user planning to move?",
        "answer": "Portland",
        "question_date": "2023/05/20 (Sat) 09:00",
        "haystack_session_ids": ["s_late", "s_early"],
        # Sessions are stored newest-first; chronological order must reorder them.
        "haystack_dates": ["2023/05/01 (Mon) 08:30", "2023/04/10 (Mon) 17:50"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "Decided on Portland."},
                {"role": "assistant", "content": "Great choice!"},
            ],
            [
                {"role": "user", "content": "I'm thinking about relocating."},
                {"role": "assistant", "content": "Where to?"},
            ],
        ],
        "answer_session_ids": ["s_late"],
    },
    {
        "question_id": "q-abs-1_abs",
        "question_type": "single-session-user",
        "question": "What car does the user drive?",
        "answer": "The user never mentioned a car.",
        "question_date": "2023/05/21 (Sun) 10:00",
        "haystack_session_ids": ["s1"],
        "haystack_dates": ["2023/04/11 (Tue) 12:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I like hiking."},
                {"role": "assistant", "content": "Nice!"},
            ],
        ],
        "answer_session_ids": [],
    },
]


def _write_fixture(data: object) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return Path(f.name)


def _question(index: int) -> LongMemEvalQuestion:
    return parse_longmemeval_file(_write_fixture(_FIXTURE))[index]


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class _FakeAnswerBackend:
    """Records the prompt it was asked and returns a scripted answer."""

    def __init__(self, answer: str = "Portland") -> None:
        self._answer = answer
        self.last_system: str | None = None
        self.last_user: str | None = None
        self.last_temperature: float | None = None
        self.temperature_was_passed: bool = False

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        self.last_system = system_prompt
        self.last_user = user_prompt
        self.last_temperature = temperature
        self.temperature_was_passed = temperature is not None
        return AgentResult(
            answer=self._answer,
            cost=CostMetrics(
                total_input_tokens=120,
                total_output_tokens=5,
                retrieval_call_count=0,
                reasoning_turn_count=1,
                retrieved_tokens_per_call=(),
                query_duration_seconds=0.5,
                total_cost_usd=0.01,
            ),
            history=(),
        )


class _OverflowBackend:
    """Raises a synthetic OpenAI context-length error on every call."""

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        raise _make_context_length_error()


def _make_context_length_error() -> Exception:
    import httpx
    import openai

    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=400, request=request)
    return openai.BadRequestError(
        "This model's maximum context length is 272000 tokens.",
        response=response,
        body={"code": "context_length_exceeded", "type": "invalid_request_error"},
    )


class _FakeJudgeBackend:
    """Returns a scripted yes/no and records the prompt it graded."""

    def __init__(self, answer: str = "yes") -> None:
        self._answer = answer
        self.prompts: list[str] = []

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        self.prompts.append(user_prompt)
        return AgentResult(
            answer=self._answer,
            cost=CostMetrics.zero(),
            history=(),
        )


class _FakeEmbedder:
    """Deterministic embedder: a chunk is "relevant" if it mentions Portland.

    Returns a 2-D vector so cosine similarity is trivial to reason about:
    Portland-bearing text points along x, everything else along y, and the
    query points along x — so Portland chunks always win.
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if "Portland" in t:
                out.append([1.0, 0.0])
            elif t == "Where is the user planning to move?":
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out


# ---------------------------------------------------------------------------
# chronological_turns / build_context: full
# ---------------------------------------------------------------------------


class TestChronologicalTurns:
    def test_orders_sessions_by_timestamp(self) -> None:
        q = _question(0)
        rendered = chronological_turns(q)
        # The April session must come before the May session even though the
        # file stores May first.
        joined = "\n".join(rendered)
        assert joined.index("relocating") < joined.index("Portland")

    def test_each_line_has_timestamp_and_role(self) -> None:
        q = _question(0)
        rendered = chronological_turns(q)
        first = rendered[0]
        assert first.startswith("[2023-04-10T17:50:00+00:00] user:")


class TestBuildContextFull:
    def test_full_includes_every_turn(self) -> None:
        q = _question(0)
        ctx = build_context(q, BaselineStrategy.FULL, budget=None)
        for needle in ["relocating", "Where to?", "Portland", "Great choice!"]:
            assert needle in ctx

    def test_full_is_chronological(self) -> None:
        q = _question(0)
        ctx = build_context(q, BaselineStrategy.FULL, budget=None)
        assert ctx.index("relocating") < ctx.index("Portland")

    def test_full_ignores_budget(self) -> None:
        q = _question(0)
        # Even a tiny budget must not drop turns in full mode.
        ctx = build_context(q, BaselineStrategy.FULL, budget=1)
        assert "relocating" in ctx
        assert "Portland" in ctx


# ---------------------------------------------------------------------------
# build_context: truncate
# ---------------------------------------------------------------------------


def _wide_question() -> LongMemEvalQuestion:
    """A question with many turns of known size for budget assertions."""
    turns_a = tuple(
        Turn(role="user", content=f"early message number {i} " * 5) for i in range(10)
    )
    turns_b = tuple(
        Turn(role="user", content=f"late message number {i} " * 5) for i in range(10)
    )
    return LongMemEvalQuestion(
        question_id="q-wide",
        question_type=QuestionType.MULTI_SESSION,
        question="q?",
        answer="a",
        question_date="2023/06/01 (Thu) 00:00",
        haystack_sessions=(
            Session(session_id="early", date="2023/01/01 (Sun) 00:00", turns=turns_a),
            Session(session_id="late", date="2023/05/01 (Mon) 00:00", turns=turns_b),
        ),
        answer_session_ids=(),
        is_abstention=False,
    )


class TestBuildContextTruncate:
    def test_truncate_respects_budget(self) -> None:
        from ragzoom.utils.tokenization import count_tokens

        q = _wide_question()
        budget = 60
        ctx = build_context(q, BaselineStrategy.TRUNCATE, budget=budget)
        assert count_tokens(ctx) <= budget

    def test_truncate_keeps_most_recent_turns(self) -> None:
        q = _wide_question()
        ctx = build_context(q, BaselineStrategy.TRUNCATE, budget=60)
        # The recent ("late") session must survive; the oldest must be dropped.
        assert "late message number 9" in ctx
        assert "early message number 0" not in ctx

    def test_truncate_requires_budget(self) -> None:
        q = _wide_question()
        with pytest.raises(ValueError, match="budget"):
            build_context(q, BaselineStrategy.TRUNCATE, budget=None)


# ---------------------------------------------------------------------------
# Overflow detection + recording
# ---------------------------------------------------------------------------


class TestOverflowDetection:
    def test_detects_openai_context_length_error(self) -> None:
        assert is_context_overflow_error(_make_context_length_error()) is True

    def test_ignores_unrelated_errors(self) -> None:
        assert is_context_overflow_error(ValueError("nope")) is False


class TestAnswerOneOverflow:
    @pytest.mark.asyncio
    async def test_overflow_is_recorded_not_raised(self) -> None:
        q = _question(0)
        judge = _FakeJudgeBackend("no")
        result = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _OverflowBackend()),
            judge=cast(BenchmarkingAgent, judge),
            judge_model="gpt-4.1",
            question=q,
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert result.judge_verdict == "no"
        assert "context_overflow" in result.served_tilings[0]
        assert "overflow" in result.generated_answer.lower()
        # An overflow burns no answer-model tokens we can attribute.
        assert result.cost.total_input_tokens == 0

    @pytest.mark.asyncio
    async def test_overflow_skips_judge_for_abstention_safety(self) -> None:
        # An overflow on a non-abstention question is simply wrong; the judge
        # still runs so the verdict reflects reality ("no"). We assert the judge
        # WAS consulted, not bypassed — silent scoring is forbidden.
        q = _question(0)
        judge = _FakeJudgeBackend("no")
        await answer_one(
            answer_backend=cast(BenchmarkingAgent, _OverflowBackend()),
            judge=cast(BenchmarkingAgent, judge),
            judge_model="gpt-4.1",
            question=q,
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert len(judge.prompts) == 1


# ---------------------------------------------------------------------------
# answer_one happy path + judge wiring
# ---------------------------------------------------------------------------


class TestAnswerOne:
    @pytest.mark.asyncio
    async def test_passes_context_and_question_to_answer_model(self) -> None:
        q = _question(0)
        backend = _FakeAnswerBackend("Portland")
        await answer_one(
            answer_backend=cast(BenchmarkingAgent, backend),
            judge=cast(BenchmarkingAgent, _FakeJudgeBackend("yes")),
            judge_model="gpt-4.1",
            question=q,
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert backend.last_user is not None
        assert "relocating" in backend.last_user  # context present
        assert "Where is the user planning to move?" in backend.last_user

    @pytest.mark.asyncio
    async def test_answerer_uses_model_default_temperature(self) -> None:
        """The answerer must be invoked identically to the RagZoom search path,
        which passes no temperature (model default). Forcing temperature=0.0
        both breaks gpt-5 models (which reject any non-default value) and would
        make the baseline a non-apples-to-apples comparison against RagZoom."""
        q = _question(0)
        backend = _FakeAnswerBackend("Portland")
        await answer_one(
            answer_backend=cast(BenchmarkingAgent, backend),
            judge=cast(BenchmarkingAgent, _FakeJudgeBackend("yes")),
            judge_model="gpt-4.1",
            question=q,
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert backend.temperature_was_passed is False
        assert backend.last_temperature is None

    @pytest.mark.asyncio
    async def test_judge_receives_gold_and_generated(self) -> None:
        q = _question(0)
        judge = _FakeJudgeBackend("yes")
        result = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend("Portland")),
            judge=cast(BenchmarkingAgent, judge),
            judge_model="gpt-4.1",
            question=q,
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert result.judge_verdict == "yes"
        assert len(judge.prompts) == 1
        graded = judge.prompts[0]
        assert "Portland" in graded  # gold + generated both appear

    @pytest.mark.asyncio
    async def test_descriptor_records_strategy_and_tokens(self) -> None:
        q = _question(0)
        result = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend("Portland")),
            judge=cast(BenchmarkingAgent, _FakeJudgeBackend("yes")),
            judge_model="gpt-4.1",
            question=q,
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        descriptor = result.served_tilings[0]
        assert "baseline:full" in descriptor
        assert "tokens" in descriptor


# ---------------------------------------------------------------------------
# topk retrieval
# ---------------------------------------------------------------------------


class TestBuildContextTopK:
    @pytest.mark.asyncio
    async def test_topk_retrieves_relevant_chunk_within_budget(self) -> None:
        from ragzoom.evaluation.longmemeval.baseline import build_context_topk
        from ragzoom.utils.tokenization import count_tokens

        q = _question(0)
        result = await build_context_topk(
            q, embedder=cast(EmbedderProtocol, _FakeEmbedder()), budget=40
        )
        assert "Portland" in result.context  # the relevant chunk was retrieved
        assert count_tokens(result.context) <= 40
        assert result.num_chunks >= 1

    @pytest.mark.asyncio
    async def test_topk_chunk_count_ignores_internal_newlines(self) -> None:
        # A turn whose content spans multiple lines is still ONE chunk; the
        # count must not be derived from newlines in the rendered context.
        from ragzoom.evaluation.longmemeval.baseline import build_context_topk

        q = LongMemEvalQuestion(
            question_id="q-nl",
            question_type=QuestionType.MULTI_SESSION,
            question="Where is the user planning to move?",
            answer="Portland",
            question_date="2023/06/01 (Thu) 00:00",
            haystack_sessions=(
                Session(
                    session_id="s1",
                    date="2023/01/01 (Sun) 00:00",
                    turns=(
                        Turn(role="user", content="Decided on Portland.\nFor sure."),
                    ),
                ),
            ),
            answer_session_ids=(),
            is_abstention=False,
        )
        result = await build_context_topk(
            q, embedder=cast(EmbedderProtocol, _FakeEmbedder()), budget=200
        )
        assert result.num_chunks == 1

    @pytest.mark.asyncio
    async def test_topk_records_retrieved_chunk_count_as_k(self) -> None:
        q = _question(0)
        result = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend("Portland")),
            judge=cast(BenchmarkingAgent, _FakeJudgeBackend("yes")),
            judge_model="gpt-4.1",
            question=q,
            strategy=BaselineStrategy.TOPK,
            budget=40,
            embedder=cast(EmbedderProtocol, _FakeEmbedder()),
            answer_model="gpt-5-mini",
        )
        # k retrieved chunks land in retrieval_call_count and the descriptor.
        k = result.cost.retrieval_call_count
        assert k >= 1
        assert f"{k} chunks" in result.served_tilings[0]


# ---------------------------------------------------------------------------
# Sampling parity with the RagZoom runner
# ---------------------------------------------------------------------------


class TestSamplingParity:
    def test_same_seed_same_qids_as_runner(self) -> None:
        from ragzoom.evaluation.longmemeval.baseline import sample_questions

        # Build a pool of distinguishable questions.
        pool = [
            LongMemEvalQuestion(
                question_id=f"q{i}",
                question_type=QuestionType.MULTI_SESSION,
                question="q?",
                answer="a",
                question_date="2023/01/01 (Sun) 00:00",
                haystack_sessions=(),
                answer_session_ids=(),
                is_abstention=False,
            )
            for i in range(50)
        ]

        baseline_sample = sample_questions(list(pool), 30)

        # Reproduce the runner's exact sampling logic.
        random.seed(42)
        runner_sample = random.sample(list(pool), 30)

        assert [q.question_id for q in baseline_sample] == [
            q.question_id for q in runner_sample
        ]

    def test_sample_none_returns_all(self) -> None:
        from ragzoom.evaluation.longmemeval.baseline import sample_questions

        pool = [_question(0), _question(1)]
        assert sample_questions(list(pool), None) == pool


# ---------------------------------------------------------------------------
# Config + end-to-end run (everything mocked)
# ---------------------------------------------------------------------------


class TestConfig:
    def test_summary_model_label_reflects_strategy(self) -> None:
        config = BaselineConfig(
            data_path=Path("d.json"),
            strategy=BaselineStrategy.TRUNCATE,
            budget=8192,
        )
        d = config.to_dict()
        assert d["summary_model"] == "baseline:truncate"
        assert d["strategy"] == "truncate"
        assert d["budget"] == 8192
        assert d["answer_model"] == config.answer_model

    def test_to_dict_omits_transient_fields(self) -> None:
        config = BaselineConfig(
            data_path=Path("d.json"), strategy=BaselineStrategy.FULL, budget=None
        )
        d = config.to_dict()
        assert "output_dir" not in d


class TestRunBaselineEndToEnd:
    @pytest.mark.asyncio
    async def test_run_produces_shared_schema(self) -> None:
        from ragzoom.evaluation.longmemeval.report import save_json

        path = _write_fixture(_FIXTURE)
        config = BaselineConfig(
            data_path=path,
            strategy=BaselineStrategy.FULL,
            budget=None,
            sample_size=None,
        )
        report = await run_baseline(
            config,
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend("Portland")),
            judge=cast(BenchmarkingAgent, _FakeJudgeBackend("yes")),
            embedder=None,
        )

        # Schema parity: dump and re-load through the SHARED report writer.
        out = _write_fixture({})
        save_json(report, out)
        data = json.loads(out.read_text())

        assert set(data.keys()) == {
            "metadata",
            "scores",
            "per_question",
            "haystack_metrics",
        }
        meta = data["metadata"]
        assert meta["config"]["summary_model"] == "baseline:full"
        assert meta["config"]["strategy"] == "full"
        assert meta["dataset_variant"] == "unknown"
        row = data["per_question"][0]
        assert set(row.keys()) >= {
            "question_id",
            "question",
            "gold_answer",
            "question_type",
            "is_abstention",
            "generated_answer",
            "verdict",
            "served_tilings",
            "cost",
        }
        assert row["cost"]["retrieval_call_count"] == 0
        # Haystack metrics carry session/turn counts for the same qids.
        hm_ids = {m["question_id"] for m in data["haystack_metrics"]}
        assert hm_ids == {"q-multi-1", "q-abs-1_abs"}

    @pytest.mark.asyncio
    async def test_run_scores_every_sampled_question(self) -> None:
        path = _write_fixture(_FIXTURE)
        config = BaselineConfig(
            data_path=path,
            strategy=BaselineStrategy.FULL,
            budget=None,
        )
        report = await run_baseline(
            config,
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend("Portland")),
            judge=cast(BenchmarkingAgent, _FakeJudgeBackend("yes")),
            embedder=None,
        )
        assert report.num_questions == 2
        assert len(report.per_question) == 2
        assert report.scores.overall_accuracy == pytest.approx(1.0)
