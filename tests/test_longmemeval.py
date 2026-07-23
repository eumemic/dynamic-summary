"""Unit tests for the LongMemEval harness.

Every LLM/judge/ingest side effect is mocked — no real API calls, no server,
no dataset download. The synthetic haystack fixture below stands in for a real
LongMemEval question and exercises the loader, ingest, scoring, aggregation,
runner, and report end to end.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from ragzoom.agent.protocol import AgentResult, CostMetrics, ToolDefinition
from ragzoom.evaluation.longmemeval.ingest import (
    doc_id_for,
    ingest_haystack,
    parse_longmemeval_timestamp,
    render_session_text,
)
from ragzoom.evaluation.longmemeval.loader import (
    HF_REPO_ID,
    download_variant,
    filename_for_variant,
)
from ragzoom.evaluation.longmemeval.report import save_json, save_markdown
from ragzoom.evaluation.longmemeval.runner import (
    LongMemEvalConfig,
    aggregate,
    question_with_date,
)
from ragzoom.evaluation.longmemeval.scoring import build_judge_prompt, judge_answer
from ragzoom.evaluation.longmemeval.types import (
    AggregateScores,
    AnswerResult,
    BenchmarkReport,
    CategoryScore,
    JudgeVerdict,
    LongMemEvalQuestion,
    QuestionType,
    detect_variant,
    parse_longmemeval_file,
)
from ragzoom.wrapper import AppendUnit, RagZoom

# ---------------------------------------------------------------------------
# Synthetic fixture — a minimal two-question LongMemEval file
# ---------------------------------------------------------------------------

_FIXTURE: list[dict[str, object]] = [
    {
        "question_id": "q-multi-1",
        "question_type": "multi-session",
        "question": "Where is the user planning to move?",
        "answer": "Portland",
        "question_date": "2023/05/20 (Sat) 09:00",
        "haystack_session_ids": ["s1", "s2"],
        "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/05/01 (Mon) 08:30"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I'm thinking about relocating."},
                {
                    "role": "assistant",
                    "content": "Where to?",
                    "has_answer": False,
                },
            ],
            [
                {
                    "role": "user",
                    "content": "Decided on Portland.",
                    "has_answer": True,
                },
                {"role": "assistant", "content": "Great choice!"},
            ],
        ],
        "answer_session_ids": ["s2"],
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


# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------


class TestParseTimestamp:
    def test_full_datetime(self) -> None:
        assert (
            parse_longmemeval_timestamp("2023/04/10 (Mon) 17:50")
            == "2023-04-10T17:50:00+00:00"
        )

    def test_no_weekday(self) -> None:
        assert (
            parse_longmemeval_timestamp("2023/04/10 17:50")
            == "2023-04-10T17:50:00+00:00"
        )

    def test_date_only_defaults_to_midnight(self) -> None:
        assert parse_longmemeval_timestamp("2023/04/10") == "2023-04-10T00:00:00+00:00"

    def test_single_digit_month_and_day(self) -> None:
        assert (
            parse_longmemeval_timestamp("2023/4/5 (Wed) 8:05")
            == "2023-04-05T08:05:00+00:00"
        )

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_longmemeval_timestamp("not a date")


# ---------------------------------------------------------------------------
# Parsing the dataset file
# ---------------------------------------------------------------------------


class TestParseFile:
    def test_parse_list(self) -> None:
        path = _write_fixture(_FIXTURE)
        qs = parse_longmemeval_file(path)
        assert len(qs) == 2
        assert qs[0].question_id == "q-multi-1"
        assert qs[0].question_type == QuestionType.MULTI_SESSION

    def test_parse_dict_layout(self) -> None:
        path = _write_fixture({q["question_id"]: q for q in _FIXTURE})
        qs = parse_longmemeval_file(path)
        assert len(qs) == 2

    def test_sessions_and_turns(self) -> None:
        path = _write_fixture(_FIXTURE)
        q = parse_longmemeval_file(path)[0]
        assert len(q.haystack_sessions) == 2
        s2 = q.haystack_sessions[1]
        assert s2.session_id == "s2"
        assert s2.date == "2023/05/01 (Mon) 08:30"
        assert s2.turns[0].content == "Decided on Portland."
        assert s2.turns[0].has_answer is True

    def test_abstention_flag_from_id_suffix(self) -> None:
        path = _write_fixture(_FIXTURE)
        qs = parse_longmemeval_file(path)
        assert qs[0].is_abstention is False
        assert qs[1].is_abstention is True

    def test_answer_session_ids(self) -> None:
        path = _write_fixture(_FIXTURE)
        q = parse_longmemeval_file(path)[0]
        assert q.answer_session_ids == ("s2",)

    def test_misaligned_haystack_raises(self) -> None:
        bad = dict(_FIXTURE[0])
        bad["haystack_dates"] = [
            "2023/04/10 (Mon) 17:50"
        ]  # only one date for two sessions
        path = _write_fixture([bad])
        with pytest.raises(ValueError, match="misaligned"):
            parse_longmemeval_file(path)


class TestDetectVariant:
    def test_oracle(self) -> None:
        assert detect_variant(Path("longmemeval_oracle.json")) == "oracle"

    def test_s(self) -> None:
        assert detect_variant(Path("longmemeval_s_cleaned.json")) == "s"

    def test_m(self) -> None:
        assert detect_variant(Path("longmemeval_m_cleaned.json")) == "m"

    def test_unknown(self) -> None:
        assert detect_variant(Path("my_custom_data.json")) == "unknown"


# ---------------------------------------------------------------------------
# Loader (download resolution — download itself is mocked)
# ---------------------------------------------------------------------------


class TestLoader:
    def test_filename_for_known_variants(self) -> None:
        assert filename_for_variant("oracle") == "longmemeval_oracle.json"
        assert filename_for_variant("S") == "longmemeval_s_cleaned.json"
        assert filename_for_variant("m") == "longmemeval_m_cleaned.json"

    def test_filename_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown LongMemEval variant"):
            filename_for_variant("xl")

    def test_download_uses_injected_downloader(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_downloader(**kwargs: object) -> str:
            calls.append(kwargs)
            return "/tmp/cache/longmemeval_s_cleaned.json"

        path = download_variant("s", downloader=fake_downloader)
        assert path == Path("/tmp/cache/longmemeval_s_cleaned.json")
        assert calls[0]["repo_id"] == HF_REPO_ID
        assert calls[0]["filename"] == "longmemeval_s_cleaned.json"
        assert calls[0]["repo_type"] == "dataset"


# ---------------------------------------------------------------------------
# Ingest (RagZoom wrapper is faked — no server)
# ---------------------------------------------------------------------------


class _FakeStatus:
    completion_pct = 100.0


class _FakeRagZoom:
    """Records clear/batch_append calls; no network."""

    def __init__(self) -> None:
        self.cleared: list[str] = []
        self.appended: dict[str, list[AppendUnit]] = {}

    def clear(self, document_id: str) -> None:
        self.cleared.append(document_id)

    def batch_append(
        self,
        document_id: str,
        units: list[AppendUnit],
        *,
        timestamps: object = None,
    ) -> None:
        self.appended[document_id] = list(units)

    def get_document_status(self, document_id: str) -> _FakeStatus:
        return _FakeStatus()


class TestIngest:
    def test_render_session_text(self) -> None:
        path = _write_fixture(_FIXTURE)
        q = parse_longmemeval_file(path)[0]
        text = render_session_text(q, 1)
        assert text == "user: Decided on Portland.\nassistant: Great choice!"

    def test_doc_id(self) -> None:
        path = _write_fixture(_FIXTURE)
        q = parse_longmemeval_file(path)[0]
        assert doc_id_for(q) == "lme-q-multi-1"

    def test_ingest_one_unit_per_session_with_timestamps(self) -> None:
        path = _write_fixture(_FIXTURE)
        q = parse_longmemeval_file(path)[0]
        rz = _FakeRagZoom()
        metrics = ingest_haystack(cast(RagZoom, rz), q)

        assert rz.cleared == ["lme-q-multi-1"]
        units = rz.appended["lme-q-multi-1"]
        assert len(units) == 2  # one AppendUnit per session
        # Session timestamps attached as ISO 8601 metadata
        assert units[0].time_start == "2023-04-10T17:50:00+00:00"
        assert units[1].time_start == "2023-05-01T08:30:00+00:00"
        assert metrics.num_sessions == 2
        assert metrics.num_turns == 4

    def test_ingest_skips_empty_sessions(self) -> None:
        q = LongMemEvalQuestion(
            question_id="q-empty",
            question_type=QuestionType.MULTI_SESSION,
            question="q?",
            answer="a",
            question_date="2023/05/20 (Sat) 09:00",
            haystack_sessions=(),
            answer_session_ids=(),
            is_abstention=False,
        )
        rz = _FakeRagZoom()
        with pytest.raises(ValueError, match="no non-empty sessions"):
            ingest_haystack(cast(RagZoom, rz), q)


# ---------------------------------------------------------------------------
# Scoring — type-specific judge prompts and the judge backend
# ---------------------------------------------------------------------------


class TestJudgePrompt:
    def test_temporal_prompt_mentions_off_by_one(self) -> None:
        prompt = build_judge_prompt(
            QuestionType.TEMPORAL_REASONING, "q", "18 days", "19 days"
        )
        assert "off-by-one" in prompt

    def test_knowledge_update_prompt_accepts_old_plus_new(self) -> None:
        prompt = build_judge_prompt(
            QuestionType.KNOWLEDGE_UPDATE, "q", "new fact", "old then new"
        )
        assert "updated answer" in prompt

    def test_preference_prompt_uses_rubric(self) -> None:
        prompt = build_judge_prompt(
            QuestionType.SINGLE_SESSION_PREFERENCE, "q", "rubric text", "resp"
        )
        assert "Rubric:" in prompt

    def test_abstention_prompt_overrides_type(self) -> None:
        prompt = build_judge_prompt(
            QuestionType.MULTI_SESSION,
            "q",
            "unanswerable explanation",
            "I cannot answer",
            is_abstention=True,
        )
        assert "unanswerable" in prompt

    def test_standard_prompt_for_single_session(self) -> None:
        prompt = build_judge_prompt(
            QuestionType.SINGLE_SESSION_USER, "q", "ans", "resp"
        )
        assert "contains the correct answer" in prompt


class _FakeJudgeBackend:
    """Returns a scripted yes/no answer; tracks the prompt it was given."""

    def __init__(self, answer: str) -> None:
        self._answer = answer
        self.last_prompt: str | None = None

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
        self.last_prompt = user_prompt
        return AgentResult(
            answer=self._answer,
            cost=CostMetrics(
                total_input_tokens=10,
                total_output_tokens=1,
                retrieval_call_count=0,
                reasoning_turn_count=1,
                retrieved_tokens_per_call=(),
            ),
            history=(),
        )


class TestJudgeAnswer:
    @pytest.mark.asyncio
    async def test_yes_verdict(self) -> None:
        verdict = await judge_answer(
            _FakeJudgeBackend("yes"),
            QuestionType.MULTI_SESSION,
            "Where moving?",
            "Portland",
            "Portland",
        )
        assert verdict == "yes"

    @pytest.mark.asyncio
    async def test_no_verdict(self) -> None:
        verdict = await judge_answer(
            _FakeJudgeBackend("no"),
            QuestionType.MULTI_SESSION,
            "Where moving?",
            "Portland",
            "Seattle",
        )
        assert verdict == "no"

    @pytest.mark.asyncio
    async def test_abstention_uses_abstention_prompt(self) -> None:
        backend = _FakeJudgeBackend("yes")
        await judge_answer(
            backend,
            QuestionType.SINGLE_SESSION_USER,
            "What car?",
            "never mentioned",
            "I cannot answer that from the history.",
            is_abstention=True,
        )
        assert backend.last_prompt is not None
        assert "unanswerable" in backend.last_prompt

    @pytest.mark.asyncio
    async def test_unparseable_judge_raises(self) -> None:
        from ragzoom.exceptions import LLMError

        with pytest.raises(LLMError):
            await judge_answer(
                _FakeJudgeBackend("maybe?"),
                QuestionType.MULTI_SESSION,
                "q",
                "a",
                "b",
                max_retries=1,
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _result(
    verdict: JudgeVerdict | None,
    qtype: QuestionType,
    *,
    is_abstention: bool = False,
) -> AnswerResult:
    return AnswerResult(
        question_id="x",
        question="q?",
        gold_answer="a",
        question_type=qtype,
        is_abstention=is_abstention,
        generated_answer="a",
        judge_verdict=verdict,
        cost=CostMetrics.zero(),
    )


class TestAggregate:
    def test_overall_and_per_type(self) -> None:
        results = [
            _result("yes", QuestionType.MULTI_SESSION),
            _result("no", QuestionType.MULTI_SESSION),
            _result("yes", QuestionType.TEMPORAL_REASONING),
        ]
        scores = aggregate(results)
        assert scores.overall_accuracy == pytest.approx(2 / 3)
        assert scores.by_type[QuestionType.MULTI_SESSION].accuracy == pytest.approx(0.5)
        assert scores.by_type[QuestionType.TEMPORAL_REASONING].accuracy == 1.0

    def test_task_averaged_differs_from_overall(self) -> None:
        # multi-session: 1/3 correct; temporal: 1/1 correct.
        # overall = 2/4 = 0.5; task-averaged = (1/3 + 1)/2 = 0.667
        results = [
            _result("yes", QuestionType.MULTI_SESSION),
            _result("no", QuestionType.MULTI_SESSION),
            _result("no", QuestionType.MULTI_SESSION),
            _result("yes", QuestionType.TEMPORAL_REASONING),
        ]
        scores = aggregate(results)
        assert scores.overall_accuracy == pytest.approx(0.5)
        assert scores.task_averaged_accuracy == pytest.approx((1 / 3 + 1) / 2)

    def test_abstention_accuracy_isolated(self) -> None:
        results = [
            _result("yes", QuestionType.MULTI_SESSION),
            _result("no", QuestionType.SINGLE_SESSION_USER, is_abstention=True),
            _result("yes", QuestionType.SINGLE_SESSION_USER, is_abstention=True),
        ]
        scores = aggregate(results)
        assert scores.abstention_accuracy == pytest.approx(0.5)

    def test_no_abstention_questions_gives_none(self) -> None:
        results = [_result("yes", QuestionType.MULTI_SESSION)]
        scores = aggregate(results)
        assert scores.abstention_accuracy is None

    def test_no_verdicts_all_none(self) -> None:
        results = [_result(None, QuestionType.MULTI_SESSION)]
        scores = aggregate(results)
        assert scores.overall_accuracy is None
        assert scores.task_averaged_accuracy is None
        assert scores.abstention_accuracy is None
        assert scores.by_type[QuestionType.MULTI_SESSION].accuracy is None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_budget_is_a_knob(self) -> None:
        config = LongMemEvalConfig(data_path=Path("d.json"), budget=4096)
        assert config.budget == 4096
        assert config.to_dict()["budget"] == 4096

    def test_summary_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGZOOM_SUMMARY_MODEL", "anthropic/claude-opus-4-8")
        config = LongMemEvalConfig(data_path=Path("d.json"))
        assert config.summary_model == "anthropic/claude-opus-4-8"
        assert config.to_dict()["summary_model"] == "anthropic/claude-opus-4-8"

    def test_summary_model_none_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RAGZOOM_SUMMARY_MODEL", raising=False)
        config = LongMemEvalConfig(data_path=Path("d.json"))
        assert config.summary_model is None

    def test_to_dict_omits_transient_fields(self) -> None:
        config = LongMemEvalConfig(data_path=Path("d.json"))
        d = config.to_dict()
        assert "server_address" not in d
        assert "output_dir" not in d

    def test_question_with_date_prefix(self) -> None:
        path = _write_fixture(_FIXTURE)
        q = parse_longmemeval_file(path)[0]
        text = question_with_date(q)
        assert text.startswith("[Question asked on 2023/05/20 (Sat) 09:00]")
        assert "Where is the user planning to move?" in text


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_SCORES = AggregateScores(
    overall_accuracy=0.75,
    task_averaged_accuracy=0.7,
    abstention_accuracy=0.5,
    by_type={
        QuestionType.MULTI_SESSION: CategoryScore(accuracy=0.8, count=10),
        QuestionType.TEMPORAL_REASONING: CategoryScore(accuracy=0.6, count=5),
    },
)


def _report(scores: AggregateScores) -> BenchmarkReport:
    result = AnswerResult(
        question_id="q1",
        question="q?",
        gold_answer="a",
        question_type=QuestionType.MULTI_SESSION,
        is_abstention=False,
        generated_answer="a",
        judge_verdict="yes",
        cost=CostMetrics(
            total_input_tokens=100,
            total_output_tokens=50,
            retrieval_call_count=2,
            reasoning_turn_count=3,
            retrieved_tokens_per_call=(500, 800),
            total_cost_usd=0.01,
        ),
        served_tilings=("tile one", "tile two"),
    )
    return BenchmarkReport(
        answer_model="gpt-5-mini",
        judge_model="gpt-4.1",
        dataset_variant="s",
        num_questions=1,
        scores=scores,
        per_question=[result],
        config={"budget": 8192, "summary_model": "gpt-5-nano", "max_iterations": 5},
    )


class TestReport:
    def test_json_has_per_type_and_overall(self) -> None:
        report = _report(_SCORES)
        path = _write_fixture({})  # reuse temp-path helper for a .json target
        save_json(report, path)
        data = json.loads(path.read_text())
        assert data["scores"]["overall_accuracy"] == 0.75
        assert data["scores"]["task_averaged_accuracy"] == 0.7
        assert data["scores"]["abstention_accuracy"] == 0.5
        assert data["scores"]["by_type"]["multi-session"]["accuracy"] == 0.8
        assert data["metadata"]["dataset_variant"] == "s"
        assert data["metadata"]["config"]["budget"] == 8192

    def test_json_persists_served_tilings(self) -> None:
        report = _report(_SCORES)
        path = _write_fixture({})
        save_json(report, path)
        data = json.loads(path.read_text())
        assert data["per_question"][0]["served_tilings"] == ["tile one", "tile two"]
        assert data["per_question"][0]["is_abstention"] is False

    def test_markdown_has_accuracy_and_abstention(self) -> None:
        report = _report(_SCORES)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)
        save_markdown(report, path)
        content = path.read_text()
        assert "Overall" in content
        assert "Task-averaged" in content
        assert "Abstention" in content
        assert "Multi-session" in content

    def test_markdown_no_judge_mode(self) -> None:
        no_verdict_scores = AggregateScores(
            overall_accuracy=None,
            task_averaged_accuracy=None,
            abstention_accuracy=None,
            by_type={QuestionType.MULTI_SESSION: CategoryScore(accuracy=None, count=1)},
        )
        report = _report(no_verdict_scores)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)
        save_markdown(report, path)
        content = path.read_text()
        assert "no-judge mode" in content


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------


class TestAgentPrompt:
    def test_teaches_zoom(self) -> None:
        from ragzoom.evaluation.longmemeval.agent.prompt import AGENT_SYSTEM_PROMPT

        assert "SURVEY" in AGENT_SYSTEM_PROMPT
        assert "ZOOM" in AGENT_SYSTEM_PROMPT

    def test_covers_abstention(self) -> None:
        from ragzoom.evaluation.longmemeval.agent.prompt import AGENT_SYSTEM_PROMPT

        assert "cannot be answered" in AGENT_SYSTEM_PROMPT

    def test_covers_temporal(self) -> None:
        from ragzoom.evaluation.longmemeval.agent.prompt import AGENT_SYSTEM_PROMPT

        assert "date" in AGENT_SYSTEM_PROMPT.lower()
