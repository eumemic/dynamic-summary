"""Unit tests for the Oolong naive-baseline harness.

The baseline answers Oolong's aggregation questions WITHOUT RagZoom by stuffing
the transcript straight into the answer model. Every LLM/embedding side effect
is mocked — no real API calls, no server, no dataset download. These tests lock
down the behaviours the head-to-head experiment depends on:

  * ``full`` concatenates every episode/turn in chronological (episode) order
    (the "stuff it all in" upper bound);
  * ``truncate`` honours a token budget B by keeping the most-recent lines;
  * ``topk`` does flat retrieval to budget B and records k where RagZoom records
    its recall-call count — the structurally decisive aggregation comparison;
  * a provider context-length rejection is recorded as ``context_overflow``
    rather than crashing the run or silently truncating;
  * the answerer is invoked with NO temperature (model default), exactly as the
    RagZoom search path invokes it (forcing temperature breaks gpt-5 models and
    would make the comparison non-apples-to-apples);
  * the seed-42 sample hits the exact same questions as the Oolong runner, so the
    two harnesses score the same set;
  * scoring REUSES the deterministic Oolong metric (``score_answer``) rather than
    reimplementing it;
  * ``results.json`` matches the SHARED Oolong report schema, so baseline and
    RagZoom Oolong runs are directly comparable.
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
)
from ragzoom.evaluation.oolong.baseline import (
    OolongBaselineConfig,
    answer_one,
    build_context,
    build_context_topk,
    chronological_lines,
    run_baseline,
    sample_questions,
)
from ragzoom.evaluation.oolong.types import OolongQuestion, QuestionType

# ---------------------------------------------------------------------------
# Synthetic fixture — a two-episode D&D window whose answer is an aggregate.
# Episode 1 has two spell casts; episode 2 has one. The transcript is laid out
# so a chronological-order assertion (episode 1 before episode 2) has teeth.
# ---------------------------------------------------------------------------

_PREAMBLE = (
    "The following lines contains a single episode transcript of a Dungeons and "
    "Dragons game. The episode transcript is delimited by [START OF EPISODE] and "
    "[END OF EPISODE]. Return the final answer in \\boxed{{}}.\n\n"
)

_EPISODE_1 = (
    "[START OF EPISODE]\n"
    "Matt: You enter the tavern.\n"
    "Travis: I cast Eldritch Blast.\n"
    "Travis: I cast Hex.\n"
    "[END OF EPISODE]"
)

_EPISODE_2 = (
    "[START OF EPISODE]\n"
    "Matt: A goblin appears.\n"
    "Travis: I cast Fireball on it.\n"
    "[END OF EPISODE]"
)

_WINDOW = _PREAMBLE + _EPISODE_1 + "\n" + _EPISODE_2


def _count_question() -> OolongQuestion:
    """A numeric-gold question: 3 total spell casts across the window."""
    return OolongQuestion(
        id="rec-count-1",
        context_window_id="cw-1",
        context_window_text=_WINDOW,
        question="How many spells were cast in total?",
        answer="3",
        question_type=QuestionType.MULTIDOC_SPELLS,
        episodes=(1, 2),
        campaign="campaign2",
    )


def _write_fixture(data: object) -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        return Path(f.name)


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class _FakeAnswerBackend:
    """Records the prompt it was asked and returns a scripted boxed answer."""

    def __init__(self, answer: str = r"\boxed{3}") -> None:
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


class _FakeEmbedder:
    """Deterministic embedder: a chunk is "relevant" if it mentions a spell cast.

    Returns a 2-D vector so cosine is trivial to reason about: spell-cast text
    and the query point along x, everything else along y — so spell lines win.
    """

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            if "cast" in t.lower() or "spell" in t.lower():
                out.append([1.0, 0.0])
            else:
                out.append([0.0, 1.0])
        return out


# ---------------------------------------------------------------------------
# chronological_lines / build_context: full
# ---------------------------------------------------------------------------


class TestChronologicalLines:
    def test_orders_episodes_chronologically(self) -> None:
        lines = chronological_lines(_count_question())
        joined = "\n".join(lines)
        # Episode 1 content must precede episode 2 content.
        assert joined.index("Eldritch Blast") < joined.index("Fireball")

    def test_strips_instruction_preamble(self) -> None:
        lines = chronological_lines(_count_question())
        joined = "\n".join(lines)
        # The task-framing preamble (and its \boxed directive) is not transcript.
        assert "Dungeons and Dragons game" not in joined
        assert "\\boxed" not in joined

    def test_includes_episode_order_marker(self) -> None:
        lines = chronological_lines(_count_question())
        joined = "\n".join(lines)
        # Episode boundaries must be legible so the answerer can reason per-episode.
        assert "Episode 1" in joined
        assert "Episode 2" in joined


class TestBuildContextFull:
    def test_full_includes_every_turn(self) -> None:
        ctx = build_context(_count_question(), BaselineStrategy.FULL, budget=None)
        for needle in ["Eldritch Blast", "Hex", "Fireball", "goblin"]:
            assert needle in ctx

    def test_full_ignores_budget(self) -> None:
        # Even a tiny budget must not drop turns in full mode.
        ctx = build_context(_count_question(), BaselineStrategy.FULL, budget=1)
        assert "Eldritch Blast" in ctx
        assert "Fireball" in ctx


# ---------------------------------------------------------------------------
# build_context: truncate
# ---------------------------------------------------------------------------


def _wide_question() -> OolongQuestion:
    """A window with many lines of known size for budget assertions."""
    early = "\n".join(f"Player: early line number {i} word word" for i in range(12))
    late = "\n".join(f"Player: late line number {i} word word" for i in range(12))
    window = (
        _PREAMBLE
        + "[START OF EPISODE]\n"
        + early
        + "\n[END OF EPISODE]\n"
        + "[START OF EPISODE]\n"
        + late
        + "\n[END OF EPISODE]"
    )
    return OolongQuestion(
        id="q-wide",
        context_window_id="cw-wide",
        context_window_text=window,
        question="count?",
        answer="1",
        question_type=QuestionType.MULTIDOC_ROLLS,
        episodes=(1, 2),
        campaign="campaign2",
    )


class TestBuildContextTruncate:
    def test_truncate_respects_budget(self) -> None:
        from ragzoom.utils.tokenization import count_tokens

        budget = 60
        ctx = build_context(_wide_question(), BaselineStrategy.TRUNCATE, budget=budget)
        assert count_tokens(ctx) <= budget

    def test_truncate_keeps_most_recent_lines(self) -> None:
        ctx = build_context(_wide_question(), BaselineStrategy.TRUNCATE, budget=60)
        # The recent ("late") episode must survive; the oldest must be dropped.
        assert "late line number 11" in ctx
        assert "early line number 0" not in ctx

    def test_truncate_requires_budget(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            build_context(_wide_question(), BaselineStrategy.TRUNCATE, budget=None)


# ---------------------------------------------------------------------------
# topk retrieval
# ---------------------------------------------------------------------------


class TestBuildContextTopK:
    @pytest.mark.asyncio
    async def test_topk_retrieves_relevant_chunk_within_budget(self) -> None:
        from ragzoom.utils.tokenization import count_tokens

        result = await build_context_topk(
            _count_question(),
            embedder=cast(EmbedderProtocol, _FakeEmbedder()),
            budget=40,
        )
        assert "cast" in result.context.lower()  # a spell line was retrieved
        assert count_tokens(result.context) <= 40
        assert result.num_chunks >= 1

    @pytest.mark.asyncio
    async def test_topk_records_retrieved_chunk_count_as_k(self) -> None:
        result = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend()),
            question=_count_question(),
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
# Overflow detection + recording
# ---------------------------------------------------------------------------


class TestAnswerOneOverflow:
    @pytest.mark.asyncio
    async def test_overflow_is_recorded_not_raised(self) -> None:
        result = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _OverflowBackend()),
            question=_count_question(),
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert "context_overflow" in result.served_tilings[0]
        assert "overflow" in result.generated_answer.lower()
        # An overflow is scored deterministically: the overflow text parses to no
        # boxed number, so a numeric-gold question scores 0.0 — not a crash.
        assert result.score == 0.0
        # An overflow burns no answer-model tokens we can attribute.
        assert result.cost.total_input_tokens == 0


# ---------------------------------------------------------------------------
# answer_one happy path: prompt wiring, no-temperature contract, scoring reuse
# ---------------------------------------------------------------------------


class TestAnswerOne:
    @pytest.mark.asyncio
    async def test_passes_context_and_boxed_question_to_answer_model(self) -> None:
        backend = _FakeAnswerBackend()
        await answer_one(
            answer_backend=cast(BenchmarkingAgent, backend),
            question=_count_question(),
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert backend.last_user is not None
        assert "Eldritch Blast" in backend.last_user  # context present
        assert "How many spells were cast in total?" in backend.last_user
        # The mandatory \boxed{} directive must reach the answerer (reused from
        # the Oolong agent framing) so the deterministic parser can read it.
        assert "\\boxed" in backend.last_user

    @pytest.mark.asyncio
    async def test_answerer_uses_model_default_temperature(self) -> None:
        """The answerer must be invoked identically to the RagZoom search path,
        which passes no temperature (model default). Forcing temperature=0.0
        both breaks gpt-5 models (which reject any non-default value) and would
        make the baseline a non-apples-to-apples comparison against RagZoom."""
        backend = _FakeAnswerBackend()
        await answer_one(
            answer_backend=cast(BenchmarkingAgent, backend),
            question=_count_question(),
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert backend.temperature_was_passed is False
        assert backend.last_temperature is None

    @pytest.mark.asyncio
    async def test_scores_with_deterministic_oolong_metric(self) -> None:
        # Exact numeric match -> 1.0; this also pins that scoring is the reused
        # Oolong metric, not a reimplementation.
        exact = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend(r"\boxed{3}")),
            question=_count_question(),
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert exact.score == pytest.approx(1.0)
        assert exact.parsed_answer == "3"

        # Off-by-one -> 0.75 partial credit (the Oolong exponential metric).
        near = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend(r"\boxed{4}")),
            question=_count_question(),
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert near.score == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_scoring_matches_runner_scoring_function(self) -> None:
        # The baseline must score with the SAME function the runner uses; assert
        # byte-for-byte agreement on the same raw output rather than trusting a
        # hand-computed constant.
        from ragzoom.evaluation.oolong.scoring import score_answer

        raw = r"\boxed{4}"
        result = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend(raw)),
            question=_count_question(),
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        assert result.score == score_answer(gold="3", model_output=raw)

    @pytest.mark.asyncio
    async def test_descriptor_records_strategy_and_tokens(self) -> None:
        result = await answer_one(
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend()),
            question=_count_question(),
            strategy=BaselineStrategy.FULL,
            budget=None,
            embedder=None,
            answer_model="gpt-5-mini",
        )
        descriptor = result.served_tilings[0]
        assert "baseline:full" in descriptor
        assert "tokens" in descriptor


# ---------------------------------------------------------------------------
# Sampling parity with the Oolong runner
# ---------------------------------------------------------------------------


class TestSamplingParity:
    def test_same_seed_same_qids_as_runner(self) -> None:
        pool = [
            OolongQuestion(
                id=f"q{i}",
                context_window_id=f"cw{i}",
                context_window_text=_WINDOW,
                question="q?",
                answer="1",
                question_type=QuestionType.MULTIDOC_ROLLS,
                episodes=(1,),
                campaign="c",
            )
            for i in range(50)
        ]
        baseline_sample = sample_questions(list(pool), 30)

        # Reproduce the runner's exact sampling logic (runner._run_benchmark_core).
        random.seed(42)
        runner_sample = random.sample(list(pool), 30)

        assert [q.id for q in baseline_sample] == [q.id for q in runner_sample]

    def test_sample_none_returns_all(self) -> None:
        pool = [_count_question()]
        assert sample_questions(list(pool), None) == pool


# ---------------------------------------------------------------------------
# Config + end-to-end run (everything mocked)
# ---------------------------------------------------------------------------


class TestConfig:
    def test_summary_model_label_reflects_strategy(self) -> None:
        config = OolongBaselineConfig(
            data_path=Path("d.parquet"),
            strategy=BaselineStrategy.TOPK,
            budget=8192,
        )
        d = config.to_dict()
        assert d["summary_model"] == "baseline:topk"
        assert d["strategy"] == "topk"
        assert d["budget"] == 8192
        assert d["answer_model"] == config.answer_model
        # Dataset descriptors an analysis script keys on must be recorded.
        assert d["config_name"] == "dnd"
        assert d["split"] == "test"

    def test_to_dict_omits_transient_fields(self) -> None:
        config = OolongBaselineConfig(
            data_path=Path("d.parquet"),
            strategy=BaselineStrategy.FULL,
            budget=None,
        )
        d = config.to_dict()
        assert "output_dir" not in d


def _fixture_parquet_questions() -> list[OolongQuestion]:
    return [_count_question()]


class TestRunBaselineEndToEnd:
    @pytest.mark.asyncio
    async def test_run_produces_shared_schema(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ragzoom.evaluation.oolong import baseline as baseline_mod
        from ragzoom.evaluation.oolong.report import save_json

        # Stand in for the parquet loader so no file/pyarrow is touched.
        monkeypatch.setattr(
            baseline_mod, "load_oolong_jsonl", lambda _p: _fixture_parquet_questions()
        )

        config = OolongBaselineConfig(
            data_path=Path("ignored.parquet"),
            strategy=BaselineStrategy.FULL,
            budget=None,
            sample_size=None,
        )
        report = await run_baseline(
            config,
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend(r"\boxed{3}")),
            embedder=None,
        )

        # Schema parity: dump and re-load through the SHARED Oolong report writer.
        out = _write_fixture({})
        save_json(report, out)
        data = json.loads(out.read_text())

        assert set(data.keys()) >= {"metadata", "scores", "per_question"}
        meta = data["metadata"]
        assert meta["config"]["summary_model"] == "baseline:full"
        assert meta["config"]["strategy"] == "full"
        assert meta["config_name"] == "dnd"
        assert meta["split"] == "test"
        row = data["per_question"][0]
        assert set(row.keys()) >= {
            "question_id",
            "question",
            "gold_answer",
            "question_type",
            "generated_answer",
            "parsed_answer",
            "score",
            "served_tilings",
            "cost",
        }
        assert row["score"] == 1.0
        assert row["cost"]["retrieval_call_count"] == 0

    @pytest.mark.asyncio
    async def test_run_scores_every_sampled_question(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from ragzoom.evaluation.oolong import baseline as baseline_mod

        monkeypatch.setattr(
            baseline_mod, "load_oolong_jsonl", lambda _p: _fixture_parquet_questions()
        )
        config = OolongBaselineConfig(
            data_path=Path("ignored.parquet"),
            strategy=BaselineStrategy.FULL,
            budget=None,
        )
        report = await run_baseline(
            config,
            answer_backend=cast(BenchmarkingAgent, _FakeAnswerBackend(r"\boxed{3}")),
            embedder=None,
        )
        assert report.num_questions == 1
        assert len(report.per_question) == 1
        assert report.scores.overall_score == pytest.approx(1.0)
