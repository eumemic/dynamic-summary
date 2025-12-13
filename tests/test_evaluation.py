"""Tests for summary evaluation module."""

import json
from unittest.mock import AsyncMock

import pytest

from ragzoom.evaluation.judge import _build_user_prompt, _parse_response, evaluate_node
from ragzoom.evaluation.report import print_report
from ragzoom.evaluation.types import (
    DIMENSIONS,
    DimensionScore,
    EvaluationReport,
    NodeEvaluation,
)
from ragzoom.exceptions import LLMError


class TestDimensionScore:
    """Test DimensionScore data type."""

    def test_valid_scores(self) -> None:
        """Valid scores 1-5 should work."""
        for score in range(1, 6):
            dim = DimensionScore(score=score, explanation="test")
            assert dim.score == score

    def test_invalid_score_too_low(self) -> None:
        """Score below 1 should raise ValueError."""
        with pytest.raises(ValueError, match="Score must be 1-5"):
            DimensionScore(score=0, explanation="test")

    def test_invalid_score_too_high(self) -> None:
        """Score above 5 should raise ValueError."""
        with pytest.raises(ValueError, match="Score must be 1-5"):
            DimensionScore(score=6, explanation="test")

    def test_frozen(self) -> None:
        """DimensionScore should be immutable."""
        dim = DimensionScore(score=3, explanation="test")
        with pytest.raises(AttributeError):
            dim.score = 4  # type: ignore[misc]


class TestNodeEvaluation:
    """Test NodeEvaluation data type."""

    @pytest.fixture
    def sample_evaluation(self) -> NodeEvaluation:
        """Create a sample node evaluation."""
        return NodeEvaluation(
            node_id="test-node",
            height=2,
            compression_ratio=2.0,
            level_index=0,
            span_start=100,
            retention=DimensionScore(score=4, explanation="Good retention"),
            isolation=DimensionScore(score=5, explanation="Perfect isolation"),
            faithfulness=DimensionScore(score=3, explanation="Minor issues"),
            continuity=DimensionScore(score=4, explanation="Flows well"),
        )

    def test_min_score(self, sample_evaluation: NodeEvaluation) -> None:
        """min_score should return the lowest dimension score."""
        assert sample_evaluation.min_score == 3

    def test_mean_score(self, sample_evaluation: NodeEvaluation) -> None:
        """mean_score should return average of all dimensions."""
        # (4 + 5 + 3 + 4) / 4 = 4.0
        assert sample_evaluation.mean_score == 4.0

    def test_frozen(self, sample_evaluation: NodeEvaluation) -> None:
        """NodeEvaluation should be immutable."""
        with pytest.raises(AttributeError):
            sample_evaluation.height = 3  # type: ignore[misc]


class TestEvaluationReport:
    """Test EvaluationReport data type."""

    @pytest.fixture
    def sample_evaluations(self) -> list[NodeEvaluation]:
        """Create sample evaluations for testing."""
        return [
            NodeEvaluation(
                node_id="node-1",
                height=1,
                compression_ratio=2.0,
                level_index=0,
                span_start=0,
                retention=DimensionScore(score=4, explanation=""),
                isolation=DimensionScore(score=5, explanation=""),
                faithfulness=DimensionScore(score=5, explanation=""),
                continuity=DimensionScore(score=4, explanation=""),
            ),
            NodeEvaluation(
                node_id="node-2",
                height=2,
                compression_ratio=2.1,
                level_index=0,
                span_start=512,
                retention=DimensionScore(score=3, explanation=""),
                isolation=DimensionScore(score=4, explanation=""),
                faithfulness=DimensionScore(score=4, explanation=""),
                continuity=DimensionScore(score=3, explanation=""),
            ),
        ]

    def test_mean_scores(self, sample_evaluations: list[NodeEvaluation]) -> None:
        """mean_scores should calculate per-dimension means."""
        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=10,
            nodes_evaluated=2,
            evaluations=sample_evaluations,
        )

        means = report.mean_scores()
        assert means["retention"] == 3.5  # (4 + 3) / 2
        assert means["isolation"] == 4.5  # (5 + 4) / 2
        assert means["faithfulness"] == 4.5  # (5 + 4) / 2
        assert means["continuity"] == 3.5  # (4 + 3) / 2

    def test_empty_report_mean_scores(self) -> None:
        """Empty report should return zeros for mean_scores."""
        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=0,
            nodes_evaluated=0,
            evaluations=[],
        )

        means = report.mean_scores()
        for dim in DIMENSIONS:
            assert means[dim] == 0.0

    def test_overall_mean(self, sample_evaluations: list[NodeEvaluation]) -> None:
        """overall_mean should average all dimension means."""
        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=10,
            nodes_evaluated=2,
            evaluations=sample_evaluations,
        )

        # Node 1 mean: (4+5+5+4)/4 = 4.5
        # Node 2 mean: (3+4+4+3)/4 = 3.5
        # Overall: (4.5 + 3.5) / 2 = 4.0
        assert report.overall_mean() == 4.0

    def test_outliers(self) -> None:
        """outliers should return evaluations with any score <= threshold."""
        evals = [
            NodeEvaluation(
                node_id="good",
                height=1,
                compression_ratio=2.0,
                level_index=0,
                span_start=100,
                retention=DimensionScore(score=4, explanation=""),
                isolation=DimensionScore(score=4, explanation=""),
                faithfulness=DimensionScore(score=4, explanation=""),
                continuity=DimensionScore(score=4, explanation=""),
            ),
            NodeEvaluation(
                node_id="bad",
                height=1,
                compression_ratio=2.0,
                level_index=0,
                span_start=100,
                retention=DimensionScore(score=2, explanation="Poor retention"),
                isolation=DimensionScore(score=4, explanation=""),
                faithfulness=DimensionScore(score=4, explanation=""),
                continuity=DimensionScore(score=4, explanation=""),
            ),
        ]

        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=2,
            nodes_evaluated=2,
            evaluations=evals,
        )

        outliers = report.outliers(threshold=2)
        assert len(outliers) == 1
        assert outliers[0].node_id == "bad"

    def test_passed(self, sample_evaluations: list[NodeEvaluation]) -> None:
        """passed should compare overall mean to threshold."""
        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=10,
            nodes_evaluated=2,
            evaluations=sample_evaluations,
        )

        # Overall mean is 4.0
        assert report.passed(3.0) is True
        assert report.passed(4.0) is True
        assert report.passed(4.5) is False


class TestJudgePromptConstruction:
    """Test prompt construction for LLM judge."""

    def test_build_user_prompt_with_context(self) -> None:
        """User prompt should include all sections when context provided."""
        prompt = _build_user_prompt(
            summary="This is the summary",
            source_text="The source text content",
            preceding_context="Previous context",
        )

        assert "## PRECEDING CONTEXT" in prompt
        assert "Previous context" in prompt
        assert "## SOURCE TEXT" in prompt
        assert "The source text content" in prompt
        assert "## SUMMARY TO EVALUATE" in prompt
        assert "This is the summary" in prompt

    def test_build_user_prompt_without_context(self) -> None:
        """User prompt should omit context section when None."""
        prompt = _build_user_prompt(
            summary="This is the summary",
            source_text="The source text content",
            preceding_context=None,
        )

        assert "## PRECEDING CONTEXT" not in prompt
        assert "## SOURCE TEXT" in prompt
        assert "## SUMMARY TO EVALUATE" in prompt


class TestJudgeResponseParsing:
    """Test response parsing for LLM judge."""

    def test_parse_valid_response(self) -> None:
        """Valid JSON response should parse to DimensionScores."""
        response = json.dumps(
            {
                "retention": {"score": 4, "explanation": "Good retention"},
                "isolation": {"score": 5, "explanation": "Perfect"},
                "faithfulness": {"score": 3, "explanation": "Minor issue"},
                "continuity": {"score": 4, "explanation": "Flows well"},
            }
        )

        result = _parse_response(response)

        assert result["retention"].score == 4
        assert result["retention"].explanation == "Good retention"
        assert result["isolation"].score == 5
        assert result["faithfulness"].score == 3
        assert result["continuity"].score == 4

    def test_parse_invalid_json(self) -> None:
        """Invalid JSON should raise JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            _parse_response("not valid json")

    def test_parse_missing_field(self) -> None:
        """Missing dimension should raise KeyError."""
        response = json.dumps(
            {
                "retention": {"score": 4, "explanation": "test"},
                # Missing isolation, faithfulness, continuity
            }
        )

        with pytest.raises(KeyError):
            _parse_response(response)

    def test_parse_invalid_score(self) -> None:
        """Invalid score value should raise ValueError from DimensionScore."""
        response = json.dumps(
            {
                "retention": {"score": 10, "explanation": "test"},
                "isolation": {"score": 5, "explanation": "test"},
                "faithfulness": {"score": 5, "explanation": "test"},
                "continuity": {"score": 5, "explanation": "test"},
            }
        )

        with pytest.raises(ValueError, match="Score must be 1-5"):
            _parse_response(response)


class TestEvaluateNode:
    """Test the evaluate_node async function."""

    @pytest.mark.asyncio
    async def test_evaluate_node_success(self) -> None:
        """Successful evaluation should return dimension scores."""
        mock_chat_model = AsyncMock()
        mock_chat_model.model_id = "gpt-4o"
        mock_chat_model.complete.return_value = {
            "content": json.dumps(
                {
                    "retention": {"score": 4, "explanation": "Good"},
                    "isolation": {"score": 5, "explanation": "Perfect"},
                    "faithfulness": {"score": 4, "explanation": "Faithful"},
                    "continuity": {"score": 3, "explanation": "OK"},
                }
            ),
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

        result = await evaluate_node(
            summary="Test summary",
            source_text="Left text Right text",
            preceding_context="Context",
            chat_model=mock_chat_model,
        )

        assert result["retention"].score == 4
        assert result["isolation"].score == 5
        assert result["faithfulness"].score == 4
        assert result["continuity"].score == 3

        # Verify API was called correctly
        mock_chat_model.complete.assert_called_once()
        call_args = mock_chat_model.complete.call_args
        assert call_args.kwargs.get("temperature") == 0.1

    @pytest.mark.asyncio
    async def test_evaluate_node_api_error(self) -> None:
        """API error should raise LLMError."""
        mock_chat_model = AsyncMock()
        mock_chat_model.model_id = "gpt-4o"
        mock_chat_model.complete.side_effect = Exception("API Error")

        with pytest.raises(LLMError) as exc_info:
            await evaluate_node(
                summary="Test",
                source_text="Left Right",
                preceding_context=None,
                chat_model=mock_chat_model,
            )

        assert exc_info.value.operation == "evaluate_node"
        assert "API Error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_node_empty_content(self) -> None:
        """Empty content in response should raise LLMError via ChatModel."""
        # ChatModel raises LLMError for empty content, so we simulate that
        from ragzoom.exceptions import LLMError as LLMErr

        mock_chat_model = AsyncMock()
        mock_chat_model.model_id = "gpt-4o"
        mock_chat_model.complete.side_effect = LLMErr(
            operation="complete",
            model="gpt-4o",
            message="LLM returned empty response content",
        )

        with pytest.raises(LLMError) as exc_info:
            await evaluate_node(
                summary="Test",
                source_text="Left Right",
                preceding_context=None,
                chat_model=mock_chat_model,
            )

        assert "empty" in str(exc_info.value).lower()


class TestPrintReport:
    """Test report printing."""

    def test_print_report_with_evaluations(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Report with evaluations should display all sections."""
        evals = [
            NodeEvaluation(
                node_id="node-1",
                height=1,
                compression_ratio=2.0,
                level_index=0,
                span_start=0,
                retention=DimensionScore(score=4, explanation="Good"),
                isolation=DimensionScore(score=5, explanation="Perfect"),
                faithfulness=DimensionScore(score=4, explanation="Faithful"),
                continuity=DimensionScore(score=4, explanation="Smooth"),
            ),
        ]

        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=10,
            nodes_evaluated=1,
            evaluations=evals,
        )

        print_report(report, threshold=3.0)

        captured = capsys.readouterr()
        assert "SUMMARY QUALITY REPORT" in captured.out
        assert "test-doc" in captured.out
        assert "1 of 10" in captured.out
        assert "AGGREGATE SCORES" in captured.out
        assert "Retention" in captured.out
        assert "PASSED" in captured.out

    def test_print_report_shows_issue_summary(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Report should display LLM-generated issue summary when provided."""
        evals = [
            NodeEvaluation(
                node_id="bad-node-123",
                height=2,
                compression_ratio=2.0,
                level_index=0,
                span_start=100,
                retention=DimensionScore(score=1, explanation="Very poor retention"),
                isolation=DimensionScore(score=4, explanation="OK"),
                faithfulness=DimensionScore(score=4, explanation="OK"),
                continuity=DimensionScore(score=4, explanation="OK"),
            ),
        ]

        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=1,
            nodes_evaluated=1,
            evaluations=evals,
        )

        issue_summary = (
            "- **Context bleeding**: Summaries include info from outside scope"
        )

        print_report(report, threshold=3.5, issue_summary=issue_summary)

        captured = capsys.readouterr()
        assert "RECURRING ISSUES" in captured.out
        assert "Context bleeding" in captured.out
        assert "FAILED" in captured.out

    def test_print_report_no_issue_summary(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Report should not show RECURRING ISSUES when no summary provided."""
        evals = [
            NodeEvaluation(
                node_id="bad-node-123",
                height=2,
                compression_ratio=2.0,
                level_index=0,
                span_start=100,
                retention=DimensionScore(score=1, explanation="Very poor retention"),
                isolation=DimensionScore(score=4, explanation="OK"),
                faithfulness=DimensionScore(score=4, explanation="OK"),
                continuity=DimensionScore(score=4, explanation="OK"),
            ),
        ]

        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=1,
            nodes_evaluated=1,
            evaluations=evals,
        )

        print_report(report, threshold=3.5)

        captured = capsys.readouterr()
        assert "RECURRING ISSUES" not in captured.out
        assert "FAILED" in captured.out

    def test_print_report_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Empty report should indicate no evaluations."""
        report = EvaluationReport(
            document_id="test-doc",
            total_inner_nodes=0,
            nodes_evaluated=0,
            evaluations=[],
        )

        print_report(report, threshold=3.0)

        captured = capsys.readouterr()
        assert "No evaluations to report" in captured.out
