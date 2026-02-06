"""Unit tests for LoCoMo scoring (token F1), JSON parsing, and runner logic."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pytest

from ragzoom.evaluation.locomo.report import save_json, save_markdown
from ragzoom.evaluation.locomo.runner import LoCoMoConfig, _aggregate_budget
from ragzoom.evaluation.locomo.scoring import compute_token_f1
from ragzoom.evaluation.locomo.types import (
    AnswerResult,
    BenchmarkReport,
    BudgetPoint,
    CategoryScore,
    CostMetrics,
    LoCoMoConversation,
    QACategory,
    parse_locomo_file,
)

# ---------------------------------------------------------------------------
# compute_token_f1
# ---------------------------------------------------------------------------


class TestComputeTokenF1:
    def test_exact_match(self) -> None:
        assert compute_token_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert compute_token_f1("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        # "hello" is common. precision=1/2, recall=1/2, f1=0.5
        assert compute_token_f1("hello world", "hello there") == pytest.approx(0.5)

    def test_case_insensitive(self) -> None:
        assert compute_token_f1("Hello World", "hello world") == 1.0

    def test_punctuation_stripped(self) -> None:
        assert compute_token_f1("hello, world!", "hello world") == 1.0

    def test_both_empty(self) -> None:
        assert compute_token_f1("", "") == 1.0

    def test_gold_empty_generated_nonempty(self) -> None:
        assert compute_token_f1("hello", "") == 0.0

    def test_generated_empty_gold_nonempty(self) -> None:
        assert compute_token_f1("", "hello") == 0.0

    def test_repeated_tokens(self) -> None:
        # gen: {the: 3}, gold: {the: 2}
        # common: {the: 2}, precision=2/3, recall=2/2=1, f1=2*(2/3)/(5/3)=4/5=0.8
        assert compute_token_f1("the the the", "the the") == pytest.approx(0.8)

    def test_date_format_partial_match(self) -> None:
        # "7 may 2023" vs "may 7" -> common: {may, 7}, p=2/2=1, r=2/3, f1=0.8
        assert compute_token_f1("May 7", "7 May 2023") == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# parse_locomo_file
# ---------------------------------------------------------------------------

_MINIMAL_CONVERSATION: dict[str, object] = {
    "sample_id": "conv-test",
    "conversation": {
        "session_1": [
            {"speaker": "Alice", "text": "Hi Bob!", "dia_id": "D1:1"},
            {"speaker": "Bob", "text": "Hey Alice!", "dia_id": "D1:2"},
        ],
        "session_1_date_time": "2023-05-08T13:56:00",
        "session_2": [
            {"speaker": "Alice", "text": "How was your trip?", "dia_id": "D2:1"},
        ],
        "session_2_date_time": "2023-05-15T10:00:00",
    },
    "qa": [
        {
            "question": "What did Alice say first?",
            "answer": "Hi Bob!",
            "category": 1,
            "evidence": ["D1:1"],
        },
        {
            "question": "What was discussed in the second session?",
            "answer": "Alice asked about Bob's trip",
            "category": 2,
            "evidence": ["D2:1"],
        },
        {
            "question": "This has no category",
            "answer": "skip me",
            "category": None,
            "evidence": [],
        },
    ],
}


class TestParseLocomoFile:
    def _write_and_parse(
        self,
        data: list[dict[str, object]] | Mapping[str, object],
    ) -> list[LoCoMoConversation]:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            return parse_locomo_file(Path(f.name))

    def test_parse_list_format(self) -> None:
        convs = self._write_and_parse([_MINIMAL_CONVERSATION])
        assert len(convs) == 1

        conv = convs[0]
        assert conv.sample_id == "conv-test"
        # Speaker names inferred from [SPEAKER]: format in content
        assert conv.speaker_a == "ALICE"  # upper() from content formatting
        assert conv.speaker_b == "BOB"

    def test_session_count(self) -> None:
        convs = self._write_and_parse([_MINIMAL_CONVERSATION])
        assert len(convs[0].sessions) == 2

    def test_session_turns(self) -> None:
        convs = self._write_and_parse([_MINIMAL_CONVERSATION])
        session_1 = convs[0].sessions[0]  # first session (index=1 in LoCoMo)
        assert len(session_1.turns) == 2
        assert session_1.turns[0].content == "[ALICE]: Hi Bob!"
        assert session_1.turns[0].dia_id == "D1:1"
        assert session_1.timestamp == "2023-05-08T13:56:00"

    def test_qa_pairs_skip_null_category(self) -> None:
        convs = self._write_and_parse([_MINIMAL_CONVERSATION])
        # 3 QA pairs in fixture, but one has category=None -> should be skipped
        assert len(convs[0].qa_pairs) == 2

    def test_qa_categories(self) -> None:
        convs = self._write_and_parse([_MINIMAL_CONVERSATION])
        categories = {qa.category for qa in convs[0].qa_pairs}
        assert categories == {QACategory.SINGLE_HOP, QACategory.MULTI_HOP}  # 1, 2

    def test_qa_evidence_ids(self) -> None:
        convs = self._write_and_parse([_MINIMAL_CONVERSATION])
        single_hop_qa = next(
            qa for qa in convs[0].qa_pairs if qa.category == QACategory.SINGLE_HOP
        )
        assert single_hop_qa.evidence_ids == ("D1:1",)

    def test_parse_dict_format(self) -> None:
        """Some LoCoMo variants use {sample_id: conversation} layout."""
        data = {"conv-test": _MINIMAL_CONVERSATION}
        convs = self._write_and_parse(data)
        assert len(convs) == 1
        assert convs[0].sample_id == "conv-test"


# ---------------------------------------------------------------------------
# LoCoMoConfig new fields
# ---------------------------------------------------------------------------


class TestLoCoMoConfig:
    def test_default_sample_size_is_none(self) -> None:
        config = LoCoMoConfig(data_path=Path("dummy.json"))
        assert config.sample_size is None

    def test_default_f1_only_is_false(self) -> None:
        config = LoCoMoConfig(data_path=Path("dummy.json"))
        assert config.f1_only is False

    def test_default_rejudge_path_is_none(self) -> None:
        config = LoCoMoConfig(data_path=Path("dummy.json"))
        assert config.rejudge_path is None

    def test_sample_size_set(self) -> None:
        config = LoCoMoConfig(data_path=Path("dummy.json"), sample_size=200)
        assert config.sample_size == 200


# ---------------------------------------------------------------------------
# _aggregate_budget with and without verdicts
# ---------------------------------------------------------------------------


def _make_result(
    verdict: str | None,
    f1: float,
    budget: int = 2000,
    category: QACategory = QACategory.SINGLE_HOP,
) -> AnswerResult:
    return AnswerResult(
        sample_id="test",
        question="q?",
        gold_answer="a",
        category=category,
        budget_tokens=budget,
        retrieved_token_count=100,
        generated_answer="a",
        judge_verdict=verdict,  # type: ignore[arg-type]
        token_f1=f1,
    )


class TestAggregateBudget:
    def test_with_verdicts(self) -> None:
        results = [
            _make_result("A", 1.0),
            _make_result("B", 0.5),
        ]
        bp = _aggregate_budget(results, 2000)
        assert bp.overall_accuracy == pytest.approx(0.5)
        assert bp.overall_f1 == pytest.approx(0.75)

    def test_without_verdicts_f1_only(self) -> None:
        results = [
            _make_result(None, 0.8),
            _make_result(None, 0.6),
        ]
        bp = _aggregate_budget(results, 2000)
        assert bp.overall_accuracy is None
        assert bp.overall_f1 == pytest.approx(0.7)
        # Category accuracy should also be None
        for cs in bp.by_category.values():
            assert cs.accuracy is None


# ---------------------------------------------------------------------------
# Report generation with f1-only mode
# ---------------------------------------------------------------------------


class TestReportF1Only:
    def test_markdown_skips_accuracy_table_when_no_verdicts(self) -> None:
        bp = BudgetPoint(
            budget_tokens=2000,
            overall_accuracy=None,
            overall_f1=0.75,
            by_category={
                QACategory.SINGLE_HOP: CategoryScore(accuracy=None, f1=0.75, count=10)
            },
        )
        report = BenchmarkReport(
            answer_model="gpt-4o-mini",
            judge_model="gpt-4.1",
            num_conversations=1,
            num_questions=10,
            budget_curve=[bp],
            per_question=[],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)
        save_markdown(report, path)
        content = path.read_text()
        assert "Judge Accuracy" not in content
        assert "Token F1" in content

    def test_markdown_includes_accuracy_table_when_verdicts_present(self) -> None:
        bp = BudgetPoint(
            budget_tokens=2000,
            overall_accuracy=0.8,
            overall_f1=0.75,
            by_category={
                QACategory.SINGLE_HOP: CategoryScore(accuracy=0.8, f1=0.75, count=10)
            },
        )
        report = BenchmarkReport(
            answer_model="gpt-4o-mini",
            judge_model="gpt-4.1",
            num_conversations=1,
            num_questions=10,
            budget_curve=[bp],
            per_question=[],
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)
        save_markdown(report, path)
        content = path.read_text()
        assert "Judge Accuracy" in content
        assert "Token F1" in content


# ---------------------------------------------------------------------------
# Answer prompt
# ---------------------------------------------------------------------------


class TestAgentPrompt:
    def test_prompt_is_information_dense(self) -> None:
        from ragzoom.evaluation.locomo.agent.prompt import AGENT_SYSTEM_PROMPT

        assert "information-dense" in AGENT_SYSTEM_PROMPT
        assert "1-3 sentences" not in AGENT_SYSTEM_PROMPT

    def test_prompt_teaches_zoom_workflow(self) -> None:
        from ragzoom.evaluation.locomo.agent.prompt import AGENT_SYSTEM_PROMPT

        assert "SURVEY" in AGENT_SYSTEM_PROMPT
        assert "ZOOM" in AGENT_SYSTEM_PROMPT

    def test_no_tool_schemas_in_prompt_module(self) -> None:
        """Tool schemas moved to ToolDefinition; prompt.py should be clean."""
        import ragzoom.evaluation.locomo.agent.prompt as prompt_mod

        assert not hasattr(prompt_mod, "RECALL_TOOL_SCHEMA")
        assert not hasattr(prompt_mod, "RECALL_TOOL_SCHEMA_ANTHROPIC")


# ---------------------------------------------------------------------------
# judge_answer with BenchmarkingAgent
# ---------------------------------------------------------------------------


class TestJudgeAnswerWithBenchmarkingAgent:
    """Verify judge_answer works with the unified BenchmarkingAgent protocol."""

    @pytest.mark.asyncio
    async def test_judge_returns_correct_verdict(self) -> None:
        from collections.abc import Sequence

        from ragzoom.evaluation.locomo.agent.protocol import (
            AgentResult,
            ToolDefinition,
        )
        from ragzoom.evaluation.locomo.scoring import judge_answer
        from ragzoom.evaluation.locomo.types import CostMetrics

        class FakeBackend:
            async def generate(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                tools: Sequence[ToolDefinition] = (),
                max_turns: int = 1,
                temperature: float | None = None,
            ) -> AgentResult:
                return AgentResult(
                    answer="A",
                    cost=CostMetrics(
                        total_input_tokens=10,
                        total_output_tokens=1,
                        retrieval_call_count=0,
                        reasoning_turn_count=1,
                        retrieved_tokens_per_call=(),
                    ),
                )

        verdict = await judge_answer(
            FakeBackend(),
            question="What color is the sky?",
            gold_answer="blue",
            generated_answer="blue",
            model_id="fake-model",
        )
        assert verdict == "A"

    @pytest.mark.asyncio
    async def test_judge_retries_on_bad_response(self) -> None:
        from collections.abc import Sequence

        from ragzoom.evaluation.locomo.agent.protocol import (
            AgentResult,
            ToolDefinition,
        )
        from ragzoom.evaluation.locomo.scoring import judge_answer
        from ragzoom.evaluation.locomo.types import CostMetrics

        call_count = 0

        class FlakeyBackend:
            async def generate(
                self,
                system_prompt: str,
                user_prompt: str,
                *,
                tools: Sequence[ToolDefinition] = (),
                max_turns: int = 1,
                temperature: float | None = None,
            ) -> AgentResult:
                nonlocal call_count
                call_count += 1
                # First call returns garbage, second returns valid verdict
                answer = "hmm not sure" if call_count == 1 else "B"
                return AgentResult(
                    answer=answer,
                    cost=CostMetrics(
                        total_input_tokens=10,
                        total_output_tokens=1,
                        retrieval_call_count=0,
                        reasoning_turn_count=1,
                        retrieved_tokens_per_call=(),
                    ),
                )

        verdict = await judge_answer(
            FlakeyBackend(),
            question="What color is the sky?",
            gold_answer="blue",
            generated_answer="red",
            model_id="flakey-model",
        )
        assert verdict == "B"
        assert call_count == 2


class TestCostMetrics:
    def test_frozen(self) -> None:
        cost = CostMetrics(
            total_input_tokens=100,
            total_output_tokens=50,
            retrieval_call_count=2,
            reasoning_turn_count=3,
            retrieved_tokens_per_call=(500, 800),
        )
        with pytest.raises(AttributeError):
            cost.total_input_tokens = 999  # type: ignore[misc]

    def test_fields(self) -> None:
        cost = CostMetrics(
            total_input_tokens=100,
            total_output_tokens=50,
            retrieval_call_count=2,
            reasoning_turn_count=3,
            retrieved_tokens_per_call=(500, 800),
        )
        assert cost.total_input_tokens == 100
        assert cost.retrieved_tokens_per_call == (500, 800)


class TestAnswerResultBackwardCompat:
    def test_cost_defaults_to_none(self) -> None:
        result = _make_result("A", 1.0)
        assert result.cost is None

    def test_cost_can_be_set(self) -> None:
        cost = CostMetrics(
            total_input_tokens=100,
            total_output_tokens=50,
            retrieval_call_count=1,
            reasoning_turn_count=2,
            retrieved_tokens_per_call=(500,),
        )
        result = AnswerResult(
            sample_id="test",
            question="q?",
            gold_answer="a",
            category=QACategory.SINGLE_HOP,
            budget_tokens=2000,
            retrieved_token_count=500,
            generated_answer="a",
            judge_verdict="A",
            token_f1=1.0,
            cost=cost,
        )
        assert result.cost is not None
        assert result.cost.retrieval_call_count == 1


class TestReportCostSerialization:
    def test_json_includes_cost_when_present(self) -> None:
        cost = CostMetrics(
            total_input_tokens=100,
            total_output_tokens=50,
            retrieval_call_count=1,
            reasoning_turn_count=2,
            retrieved_tokens_per_call=(500,),
        )
        result = AnswerResult(
            sample_id="test",
            question="q?",
            gold_answer="a",
            category=QACategory.SINGLE_HOP,
            budget_tokens=2000,
            retrieved_token_count=500,
            generated_answer="a",
            judge_verdict="A",
            token_f1=1.0,
            cost=cost,
        )
        report = BenchmarkReport(
            answer_model="test-model",
            judge_model="test-judge",
            num_conversations=1,
            num_questions=1,
            budget_curve=[],
            per_question=[result],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_json(report, path)
        data = json.loads(path.read_text())
        assert "cost" in data["per_question"][0]
        assert data["per_question"][0]["cost"]["retrieval_call_count"] == 1
        assert data["metadata"]["max_iterations"] == 1

    def test_json_omits_cost_when_none(self) -> None:
        result = _make_result("A", 1.0)
        report = BenchmarkReport(
            answer_model="test-model",
            judge_model="test-judge",
            num_conversations=1,
            num_questions=1,
            budget_curve=[],
            per_question=[result],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        save_json(report, path)
        data = json.loads(path.read_text())
        assert "cost" not in data["per_question"][0]

    def test_markdown_includes_cost_summary(self) -> None:
        cost = CostMetrics(
            total_input_tokens=1000,
            total_output_tokens=200,
            retrieval_call_count=3,
            reasoning_turn_count=4,
            retrieved_tokens_per_call=(500, 800, 1200),
        )
        result = AnswerResult(
            sample_id="test",
            question="q?",
            gold_answer="a",
            category=QACategory.SINGLE_HOP,
            budget_tokens=2000,
            retrieved_token_count=2500,
            generated_answer="a",
            judge_verdict=None,
            token_f1=0.8,
            cost=cost,
        )
        report = BenchmarkReport(
            answer_model="test-model",
            judge_model="test-judge",
            num_conversations=1,
            num_questions=1,
            budget_curve=[],
            per_question=[result],
        )
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = Path(f.name)
        save_markdown(report, path)
        content = path.read_text()
        assert "Agent Cost Summary" in content
        assert "Avg retrieval calls" in content
