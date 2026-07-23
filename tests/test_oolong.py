"""Unit tests for the Oolong (oolong-real) aggregation benchmark harness.

Every LLM/judge/ingest/dataset side effect is mocked — no real API calls, no
server, no HuggingFace download, no port use. The synthetic D&D transcript
fixture below stands in for a real ``oolongbench/oolong-real`` example and
exercises the loader, ingest, scoring, aggregation, runner, and report
end to end.

Oolong's home regime is *aggregation under compression*: a question whose
answer is a statistic over the whole transcript (a count, a ranking, an ordered
list), scored not by an LLM judge but by a deterministic metric — exponential
partial credit for numbers, case-insensitive exact match for labels, set recall
for lists. These tests pin that metric to the upstream reference implementation
(``abertsch72/oolong`` ``src/eval/eval_helpers.py::dnd_process_response``).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import cast

import pytest

from ragzoom.agent.protocol import CostMetrics
from ragzoom.evaluation.oolong.ingest import (
    _MAX_LEAF_CHARS,
    chunk_episode_text,
    doc_id_for,
    episode_timestamp,
    ingest_window,
    leaf_timestamp,
    render_episode_text,
    split_episodes,
    strip_instruction_preamble,
)
from ragzoom.evaluation.oolong.loader import (
    CONFIG_DND,
    CONFIG_TOY_DND,
    HF_REPO_ID,
    download_oolong_real,
    filename_for_config,
)
from ragzoom.evaluation.oolong.report import save_json, save_markdown
from ragzoom.evaluation.oolong.runner import (
    OolongConfig,
    aggregate,
    boxed_question,
)
from ragzoom.evaluation.oolong.scoring import (
    extract_boxed_answer,
    parse_gold_answer,
    score_answer,
)
from ragzoom.evaluation.oolong.types import (
    AggregateScores,
    AnswerResult,
    BenchmarkReport,
    CategoryScore,
    OolongQuestion,
    QuestionType,
    parse_oolong_records,
)
from ragzoom.wrapper import AppendUnit, RagZoom

# ---------------------------------------------------------------------------
# Synthetic fixture — two oolong-real records sharing one context window.
#
# The context_window_text mirrors the real dataset: an instruction preamble
# (including the \boxed{} directive), a player→character mapping, then the
# transcript delimited by [START OF EPISODE] / [END OF EPISODE].
# ---------------------------------------------------------------------------

_PREAMBLE = (
    "The following lines contains a single episode transcript of a Dungeons and "
    "Dragons game played by a group of players. The episode transcript is "
    "delimited by [START OF EPISODE] and [END OF EPISODE]. The transcript is "
    "followed by a question about the game statistics. Answer the question based "
    "on the transcript. Return the final answer in \\boxed{{}}.\n\n"
    "The following lines contain the mapping between player names and character "
    "names.\nMatt plays the character DM.\nTravis plays the character Fjord.\n\n"
)

_EPISODE_1 = (
    "[START OF EPISODE]\n"
    "Matt: You enter the tavern.\n"
    "Travis: I roll an Investigation check.\n"
    "Matt: Roll for it.\n"
    "Travis: I cast Eldritch Blast.\n"
    "[END OF EPISODE]"
)

_EPISODE_2 = (
    "[START OF EPISODE]\n"
    "Matt: A goblin appears.\n"
    "Travis: I cast Hex on it.\n"
    "[END OF EPISODE]"
)

_SINGLE_WINDOW = _PREAMBLE + _EPISODE_1
_MULTI_WINDOW = _PREAMBLE + _EPISODE_1 + "\n" + _EPISODE_2

_FIXTURE: list[dict[str, object]] = [
    {
        "id": "rec-count-1",
        "context_window_id": "cw-single",
        "context_window_text": _SINGLE_WINDOW,
        "question": "Total number of rolls in this episode?",
        "answer": "2",
        "question_type": "singledoc_rolls",
        "episodes": [1],
        "campaign": "campaign2",
    },
    {
        "id": "rec-spell-1",
        "context_window_id": "cw-multi",
        "context_window_text": _MULTI_WINDOW,
        "question": "List the last spell cast in each episode? Return a comma "
        "separated list.",
        "answer": "Eldritch Blast, Hex",
        "question_type": "multidoc_spells",
        "episodes": [1, 2],
        "campaign": "campaign2",
    },
]


def _write_json(data: object, suffix: str = ".json") -> Path:
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        json.dump(data, f)
        return Path(f.name)


# ---------------------------------------------------------------------------
# Record parsing
# ---------------------------------------------------------------------------


class TestParseRecords:
    def test_parses_all_fields(self) -> None:
        qs = parse_oolong_records(_FIXTURE)
        assert len(qs) == 2
        q = qs[0]
        assert q.id == "rec-count-1"
        assert q.context_window_id == "cw-single"
        assert q.question == "Total number of rolls in this episode?"
        assert q.answer == "2"
        assert q.question_type == QuestionType.SINGLEDOC_ROLLS
        assert q.episodes == (1,)
        assert q.campaign == "campaign2"
        assert q.context_window_text == _SINGLE_WINDOW

    def test_multidoc_episodes_preserved(self) -> None:
        q = parse_oolong_records(_FIXTURE)[1]
        assert q.question_type == QuestionType.MULTIDOC_SPELLS
        assert q.episodes == (1, 2)

    def test_unknown_question_type_raises(self) -> None:
        bad = dict(_FIXTURE[0])
        bad["question_type"] = "not_a_real_type"
        with pytest.raises(ValueError):
            parse_oolong_records([bad])


# ---------------------------------------------------------------------------
# Loader (HF download resolution — download itself is mocked / injected)
# ---------------------------------------------------------------------------


class TestLoader:
    def test_parquet_filename_for_known_configs(self) -> None:
        assert "dnd" in filename_for_config(CONFIG_DND, "test")
        assert "test" in filename_for_config(CONFIG_DND, "test")
        assert "validation" in filename_for_config(CONFIG_DND, "validation")

    def test_parquet_filename_unknown_config_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Oolong config"):
            filename_for_config("synthetic", "test")

    def test_parquet_filename_unknown_split_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown Oolong split"):
            filename_for_config(CONFIG_DND, "train")

    def test_download_uses_injected_downloader(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_downloader(**kwargs: object) -> str:
            calls.append(kwargs)
            return "/tmp/cache/dnd/test.jsonl"

        path = download_oolong_real(
            config=CONFIG_DND, split="test", downloader=fake_downloader
        )
        assert path == Path("/tmp/cache/dnd/test.jsonl")
        assert calls[0]["repo_id"] == HF_REPO_ID
        assert calls[0]["repo_type"] == "dataset"

    def test_toy_config_is_distinct(self) -> None:
        assert CONFIG_TOY_DND != CONFIG_DND
        assert "toy" in filename_for_config(CONFIG_TOY_DND, "test")


# ---------------------------------------------------------------------------
# Ingest — strip the instruction wrapper, split episodes, build temporal trees
# ---------------------------------------------------------------------------


class TestStripPreamble:
    def test_keeps_only_transcript_body(self) -> None:
        body = strip_instruction_preamble(_SINGLE_WINDOW)
        assert "Dungeons and Dragons game played" not in body
        assert "\\boxed" not in body
        assert "Matt: You enter the tavern." in body
        assert "[START OF EPISODE]" in body

    def test_raises_when_no_episode_marker(self) -> None:
        with pytest.raises(ValueError, match="no \\[START OF EPISODE\\]"):
            strip_instruction_preamble("just some text with no markers")


class TestSplitEpisodes:
    def test_single_episode(self) -> None:
        eps = split_episodes(_SINGLE_WINDOW)
        assert len(eps) == 1
        assert "You enter the tavern." in eps[0]
        assert "[START OF EPISODE]" not in eps[0]
        assert "[END OF EPISODE]" not in eps[0]

    def test_multi_episode_split(self) -> None:
        eps = split_episodes(_MULTI_WINDOW)
        assert len(eps) == 2
        assert "Eldritch Blast" in eps[0]
        assert "A goblin appears." in eps[1]

    def test_raises_on_no_episode(self) -> None:
        with pytest.raises(ValueError):
            split_episodes("no markers here")


class TestEpisodeTimestamp:
    def test_monotonic_increasing_with_index(self) -> None:
        t0 = episode_timestamp(0)
        t1 = episode_timestamp(1)
        t2 = episode_timestamp(2)
        assert t0 < t1 < t2

    def test_iso_8601_with_timezone(self) -> None:
        ts = episode_timestamp(0)
        assert ts.endswith("+00:00")
        assert "T" in ts


class TestEpisodeChunking:
    """An Oolong episode (150K-200K chars) must be chunked into leaves that fit
    under the server's per-unit cap; otherwise the server silently truncates each
    episode to 50K chars and RagZoom ingests only a fraction of the history."""

    def test_large_episode_split_into_multiple_compliant_chunks(self) -> None:
        line = "Matt: The dragon attacks for 12 damage.\n"
        big = line * 5000  # ~200K chars, far over the cap
        chunks = chunk_episode_text(big)
        assert len(chunks) > 1
        assert all(len(c) <= _MAX_LEAF_CHARS for c in chunks)
        # No transcript content is dropped: concatenation round-trips.
        assert "".join(chunks) == big

    def test_small_episode_stays_single_chunk(self) -> None:
        text = "Matt: A short scene.\nPlayer: Ok.\n"
        assert chunk_episode_text(text) == [text]

    def test_pathological_long_single_line_is_hard_split(self) -> None:
        one_line = "x" * (_MAX_LEAF_CHARS * 2 + 100)  # no newlines to split on
        chunks = chunk_episode_text(one_line)
        assert all(len(c) <= _MAX_LEAF_CHARS for c in chunks)
        assert "".join(chunks) == one_line

    def test_leaf_timestamps_strictly_increasing_within_and_across_episodes(
        self,
    ) -> None:
        # within an episode
        assert leaf_timestamp(0, 0) < leaf_timestamp(0, 1) < leaf_timestamp(0, 2)
        # last chunk of episode i precedes first chunk of episode i+1
        assert leaf_timestamp(0, 999) < leaf_timestamp(1, 0)
        # j=0 coincides with the per-episode timestamp (backward-compatible)
        assert leaf_timestamp(3, 0) == episode_timestamp(3)


class TestRenderEpisodeText:
    def test_strips_markers(self) -> None:
        text = render_episode_text(_EPISODE_1)
        assert "[START OF EPISODE]" not in text
        assert "Matt: You enter the tavern." in text


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
    def test_doc_id_is_per_context_window(self) -> None:
        # Crucial cost lever: many questions share one context window, so the
        # document id must key on the window, not the question, to dedupe ingest.
        q1, q2 = parse_oolong_records(_FIXTURE)
        assert doc_id_for(q1) == "oolong-cw-single"
        assert doc_id_for(q2) == "oolong-cw-multi"

    def test_ingest_one_unit_per_episode_with_timestamps(self) -> None:
        q = parse_oolong_records(_FIXTURE)[1]  # multi-episode window
        rz = _FakeRagZoom()
        metrics = ingest_window(cast(RagZoom, rz), q)

        assert rz.cleared == ["oolong-cw-multi"]
        units = rz.appended["oolong-cw-multi"]
        assert len(units) == 2  # one AppendUnit per episode
        # Episode order encoded as a monotonic synthetic timestamp.
        assert units[0].time_start is not None
        assert units[1].time_start is not None
        assert units[0].time_start < units[1].time_start
        # Instruction preamble must NOT be ingested as memory.
        assert "Dungeons and Dragons game played" not in units[0].text
        assert metrics.num_episodes == 2

    def test_ingest_strips_preamble_from_leaves(self) -> None:
        q = parse_oolong_records(_FIXTURE)[0]
        rz = _FakeRagZoom()
        ingest_window(cast(RagZoom, rz), q)
        unit = rz.appended["oolong-cw-single"][0]
        assert "\\boxed" not in unit.text
        assert "Matt: You enter the tavern." in unit.text


# ---------------------------------------------------------------------------
# Scoring — the load-bearing, novel part. Pinned to the upstream reference
# implementation (dnd_process_response): partial credit for numbers, exact
# match for labels, set recall for lists.
# ---------------------------------------------------------------------------


class TestExtractBoxed:
    def test_boxed_plain(self) -> None:
        assert extract_boxed_answer(r"The answer is \boxed{42}.") == "42"

    def test_boxed_text_wrapper(self) -> None:
        assert (
            extract_boxed_answer(r"\boxed{\text{Eldritch Blast}}") == "Eldritch Blast"
        )

    def test_no_box_returns_raw_text(self) -> None:
        # Upstream: if no box, fall back to the whole string (low confidence).
        assert extract_boxed_answer("just 42") == "just 42"


class TestParseGold:
    def test_int(self) -> None:
        assert parse_gold_answer("84") == 84

    def test_list_on_comma(self) -> None:
        assert parse_gold_answer("Hex, Eldritch Blast") == ["Hex", "Eldritch Blast"]

    def test_plain_string(self) -> None:
        assert parse_gold_answer("Mage Hand") == "Mage Hand"


class TestScoreNumeric:
    def test_exact_match_is_one(self) -> None:
        assert score_answer(gold="84", model_output=r"\boxed{84}") == pytest.approx(1.0)

    def test_off_by_one_partial_credit(self) -> None:
        # 0.75 ** |84 - 85| = 0.75
        assert score_answer(gold="84", model_output=r"\boxed{85}") == pytest.approx(
            0.75
        )

    def test_off_by_three_partial_credit(self) -> None:
        # 0.75 ** 3
        assert score_answer(gold="10", model_output=r"\boxed{13}") == pytest.approx(
            0.75**3
        )

    def test_unparseable_numeric_scores_zero(self) -> None:
        assert score_answer(gold="84", model_output=r"\boxed{many}") == 0.0


class TestScoreString:
    def test_case_insensitive_exact_match(self) -> None:
        assert score_answer(
            gold="Mage Hand", model_output=r"\boxed{\text{mage hand}}"
        ) == pytest.approx(1.0)

    def test_wrong_string_scores_zero(self) -> None:
        assert (
            score_answer(gold="Mage Hand", model_output=r"\boxed{\text{Fireball}}")
            == 0.0
        )


class TestScoreList:
    def test_full_recall_is_one(self) -> None:
        assert score_answer(
            gold="Eldritch Blast, Hex",
            model_output=r"\boxed{Eldritch Blast, Hex}",
        ) == pytest.approx(1.0)

    def test_partial_recall(self) -> None:
        # gold has 2 items, model gets 1 right -> recall 0.5
        assert score_answer(
            gold="Eldritch Blast, Hex",
            model_output=r"\boxed{Eldritch Blast, Fireball}",
        ) == pytest.approx(0.5)

    def test_recall_over_gold_not_precision(self) -> None:
        # When BOTH sides parse as lists, recall is |gold ∩ pred| / |gold|, so
        # extra correct-but-not-gold predicted items do not lower the score.
        assert score_answer(
            gold="Hex, Sleep",
            model_output=r"\boxed{Hex, Sleep, Eldritch Blast}",
        ) == pytest.approx(1.0)

    def test_single_item_gold_vs_list_output_is_type_mismatch(self) -> None:
        # Upstream dnd_process_response does NOT coerce a scalar to a singleton
        # list: a comma-less gold parses as a string, a comma-bearing prediction
        # parses as a list, the types mismatch, and the score is 0.0. Pinned to
        # eval_helpers.dnd_process_response, which only scores like-typed pairs.
        assert (
            score_answer(
                gold="Hex",
                model_output=r"\boxed{Hex, Eldritch Blast}",
            )
            == 0.0
        )


class TestScoreTypeMismatch:
    def test_numeric_gold_string_output_scores_zero(self) -> None:
        # gold parses as int, output parses as str -> mismatch -> 0.0
        assert score_answer(gold="5", model_output=r"\boxed{\text{five}}") == 0.0


# ---------------------------------------------------------------------------
# Aggregation — mean partial-credit score, per type, plus task-averaged.
# ---------------------------------------------------------------------------


def _result(score: float, qtype: QuestionType) -> AnswerResult:
    return AnswerResult(
        question_id="x",
        question="q?",
        gold_answer="a",
        question_type=qtype,
        generated_answer="a",
        parsed_answer="a",
        score=score,
        cost=CostMetrics.zero(),
    )


class TestAggregate:
    def test_overall_is_mean_score(self) -> None:
        results = [
            _result(1.0, QuestionType.SINGLEDOC_ROLLS),
            _result(0.5, QuestionType.SINGLEDOC_ROLLS),
            _result(0.0, QuestionType.MULTIDOC_SPELLS),
        ]
        scores = aggregate(results)
        assert scores.overall_score == pytest.approx(1.5 / 3)

    def test_per_type_scores(self) -> None:
        results = [
            _result(1.0, QuestionType.SINGLEDOC_ROLLS),
            _result(0.0, QuestionType.SINGLEDOC_ROLLS),
            _result(0.75, QuestionType.MULTIDOC_SPELLS),
        ]
        scores = aggregate(results)
        assert scores.by_type[QuestionType.SINGLEDOC_ROLLS].score == pytest.approx(0.5)
        assert scores.by_type[QuestionType.MULTIDOC_SPELLS].score == pytest.approx(0.75)

    def test_task_averaged_differs_from_overall(self) -> None:
        # rolls: (1+0+0)/3 = 1/3; spells: 1.0. overall = 2/4 = 0.5;
        # task-avg = (1/3 + 1)/2.
        results = [
            _result(1.0, QuestionType.SINGLEDOC_ROLLS),
            _result(0.0, QuestionType.SINGLEDOC_ROLLS),
            _result(0.0, QuestionType.SINGLEDOC_ROLLS),
            _result(1.0, QuestionType.MULTIDOC_SPELLS),
        ]
        scores = aggregate(results)
        assert scores.overall_score == pytest.approx(0.5)
        assert scores.task_averaged_score == pytest.approx((1 / 3 + 1) / 2)

    def test_empty_results_give_none(self) -> None:
        scores = aggregate([])
        assert scores.overall_score is None
        assert scores.task_averaged_score is None


# ---------------------------------------------------------------------------
# Config — the fixed budget B (the H/B knob) and summary_model attribution.
# ---------------------------------------------------------------------------


class TestConfig:
    def test_budget_is_a_knob(self) -> None:
        config = OolongConfig(data_path=Path("d.parquet"), budget=4096)
        assert config.budget == 4096
        assert config.to_dict()["budget"] == 4096

    def test_summary_model_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGZOOM_SUMMARY_MODEL", "anthropic/claude-opus-4-8")
        config = OolongConfig(data_path=Path("d.parquet"))
        assert config.summary_model == "anthropic/claude-opus-4-8"
        assert config.to_dict()["summary_model"] == "anthropic/claude-opus-4-8"

    def test_summary_model_none_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RAGZOOM_SUMMARY_MODEL", raising=False)
        config = OolongConfig(data_path=Path("d.parquet"))
        assert config.summary_model is None

    def test_to_dict_omits_transient_fields(self) -> None:
        config = OolongConfig(data_path=Path("d.parquet"))
        d = config.to_dict()
        assert "server_address" not in d
        assert "output_dir" not in d

    def test_boxed_question_demands_boxed_format(self) -> None:
        q = parse_oolong_records(_FIXTURE)[0]
        text = boxed_question(q)
        assert "Total number of rolls" in text
        assert "\\boxed" in text


# ---------------------------------------------------------------------------
# Runner — single-question evaluation via mocked agentic recall.
# ---------------------------------------------------------------------------


class _FakeQueryExecutor:
    async def __call__(
        self,
        *,
        document_id: str,
        query: str,
        budget_tokens: int,
        time_start: str | None = None,
        time_end: str | None = None,
    ) -> object:
        raise AssertionError("query_executor should not be called in this test")


class _ScriptedAgent:
    """A SearchAgent stand-in that returns a fixed answer + served tilings."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def search(
        self,
        question: str,
        document_id: str,
        query_executor: object,
        *,
        time_start: str | None = None,
        time_end: str | None = None,
        search_guidance: str | None = None,
    ) -> object:
        from ragzoom.search.types import SearchCost, SearchResult

        return SearchResult(
            answer=self._answer,
            cost=SearchCost(
                total_input_tokens=100,
                total_output_tokens=10,
                retrieval_call_count=1,
                reasoning_turn_count=1,
                retrieved_tokens_per_call=(50,),
                duration_seconds=0.1,
                total_cost_usd=0.001,
            ),
            profile=None,
            served_tilings=("tile-1",),
        )


class TestRunnerSingleQuestion:
    @pytest.mark.asyncio
    async def test_evaluate_one_scores_with_oolong_metric(self) -> None:
        import asyncio

        from ragzoom.evaluation.oolong.runner import _evaluate_one

        q = parse_oolong_records(_FIXTURE)[0]  # gold "2", numeric
        agent_queue: asyncio.Queue[object] = asyncio.Queue()
        agent_queue.put_nowait(_ScriptedAgent(r"\boxed{2}"))

        result = await _evaluate_one(
            agent_queue,
            cast("object", _FakeQueryExecutor()),
            q,
        )
        assert result.score == pytest.approx(1.0)
        assert result.parsed_answer == "2"
        assert result.served_tilings == ("tile-1",)

    @pytest.mark.asyncio
    async def test_evaluate_one_partial_credit(self) -> None:
        import asyncio

        from ragzoom.evaluation.oolong.runner import _evaluate_one

        q = parse_oolong_records(_FIXTURE)[0]  # gold "2"
        agent_queue: asyncio.Queue[object] = asyncio.Queue()
        agent_queue.put_nowait(_ScriptedAgent(r"\boxed{3}"))  # off by one

        result = await _evaluate_one(
            agent_queue,
            cast("object", _FakeQueryExecutor()),
            q,
        )
        assert result.score == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_SCORES = AggregateScores(
    overall_score=0.6,
    task_averaged_score=0.55,
    by_type={
        QuestionType.SINGLEDOC_ROLLS: CategoryScore(score=0.7, count=10),
        QuestionType.MULTIDOC_SPELLS: CategoryScore(score=0.4, count=5),
    },
)


def _report(scores: AggregateScores) -> BenchmarkReport:
    result = AnswerResult(
        question_id="q1",
        question="q?",
        gold_answer="2",
        question_type=QuestionType.SINGLEDOC_ROLLS,
        generated_answer=r"\boxed{2}",
        parsed_answer="2",
        score=1.0,
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
        config_name="dnd",
        split="test",
        num_questions=1,
        scores=scores,
        per_question=[result],
        config={"budget": 8192, "summary_model": "gpt-5-nano", "max_iterations": 5},
    )


class TestReport:
    def test_json_has_per_type_and_overall(self) -> None:
        report = _report(_SCORES)
        path = _write_json({})
        save_json(report, path)
        data = json.loads(path.read_text())
        assert data["scores"]["overall_score"] == 0.6
        assert data["scores"]["task_averaged_score"] == 0.55
        assert data["scores"]["by_type"]["singledoc_rolls"]["score"] == 0.7
        assert data["metadata"]["config_name"] == "dnd"
        assert data["metadata"]["config"]["budget"] == 8192

    def test_json_persists_served_tilings_and_parse(self) -> None:
        report = _report(_SCORES)
        path = _write_json({})
        save_json(report, path)
        data = json.loads(path.read_text())
        pq = data["per_question"][0]
        assert pq["served_tilings"] == ["tile one", "tile two"]
        assert pq["parsed_answer"] == "2"
        assert pq["score"] == 1.0

    def test_markdown_has_scores(self) -> None:
        report = _report(_SCORES)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            path = Path(f.name)
        save_markdown(report, path)
        content = path.read_text()
        assert "Overall" in content
        assert "Task-averaged" in content
        assert "Counting (single-doc, rolls)" in content


# ---------------------------------------------------------------------------
# Agent prompt
# ---------------------------------------------------------------------------


class TestAgentPrompt:
    def test_teaches_aggregation_and_zoom(self) -> None:
        from ragzoom.evaluation.oolong.agent.prompt import AGENT_SYSTEM_PROMPT

        assert "SURVEY" in AGENT_SYSTEM_PROMPT
        assert "boxed" in AGENT_SYSTEM_PROMPT.lower()

    def test_warns_against_estimation(self) -> None:
        from ragzoom.evaluation.oolong.agent.prompt import AGENT_SYSTEM_PROMPT

        # Aggregation demands exhaustive counting, not estimation.
        assert "estimat" in AGENT_SYSTEM_PROMPT.lower()


_OOLONG_QUESTION_TYPE_SMOKE = OolongQuestion  # ensure symbol is exported
