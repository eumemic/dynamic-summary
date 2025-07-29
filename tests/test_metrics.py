"""Test metrics functionality, particularly amplification calculations."""

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.metrics import IndexingMetrics, IndexingMetricsReporter, SummaryStats


class TestSummaryStats:
    """Test SummaryStats functionality."""

    def test_basic_stats(self) -> None:
        """Test basic summary statistics calculation."""
        stats = SummaryStats()

        # Add some summaries
        stats.add_summary(target=100, actual=90)  # 10% under
        stats.add_summary(target=100, actual=110)  # 10% over
        stats.add_summary(target=100, actual=100)  # exact (counted as under)

        assert stats.count == 3
        assert stats.total_tokens == 300
        assert stats.avg_tokens == 100.0
        assert stats.avg_deviation_percent == pytest.approx(6.67, rel=0.01)
        assert stats.percent_over_target == pytest.approx(33.33, rel=0.01)
        assert stats.percent_under_target == pytest.approx(
            66.67, rel=0.01
        )  # 2 out of 3

    def test_deviation_tracking(self) -> None:
        """Test that deviations are properly tracked."""
        stats = SummaryStats()

        stats.add_summary(target=100, actual=80)  # 20% deviation
        stats.add_summary(target=100, actual=120)  # 20% deviation
        stats.add_summary(target=100, actual=105)  # 5% deviation

        assert len(stats.deviations) == 3
        assert stats.deviations == [20.0, 20.0, 5.0]
        assert stats.median_deviation_percent == 20.0
        assert stats.std_deviation_percent == pytest.approx(8.66, rel=0.01)

    def test_percentiles(self) -> None:
        """Test percentile calculations."""
        stats = SummaryStats()

        # Add 10 summaries with known deviations
        for i in range(10):
            # Creates deviations: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90
            actual = 100 + i * 10
            stats.add_summary(target=100, actual=actual)

        # quantiles with n=10 gives 9 cut points dividing into 10 groups
        # For data [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        # quantiles returns [1.0, 12.0, 23.0, 34.0, 45.0, 56.0, 67.0, 78.0, 89.0]
        assert stats.percentile_50 == 45.0  # median (5th of 9 cut points)
        assert stats.percentile_90 == 89.0  # 90th percentile (9th of 9 cut points)
        assert stats.percentile_95 == 94.5  # 95th percentile using n=20


class TestIndexingMetrics:
    """Test IndexingMetrics calculations."""

    def test_percentile_precision_small_samples(self) -> None:
        """Test that percentile calculations work well for small sample sizes."""
        metrics = IndexingMetrics(
            start_time=0.0,
            end_time=10.0,
            source_document_tokens=1000,
            chunks_created=5,
            embedding_cost_per_1k=0.0001,
            summary_input_cost_per_1k=0.0025,
            summary_output_cost_per_1k=0.01,
        )

        # Test with single value
        metrics.cost_amplifications = [2.5]
        assert metrics.cost_amplification_p90 == 2.5
        assert metrics.cost_amplification_p95 == 2.5

        # Test with two values
        metrics.cost_amplifications = [1.0, 3.0]
        # Linear interpolation: pos = (2-1)*0.9 = 0.9
        # So 90th percentile = 1.0 + 0.9*(3.0-1.0) = 2.8
        assert metrics.cost_amplification_p90 == pytest.approx(2.8, rel=0.01)

        # Test with three values
        metrics.cost_amplifications = [1.0, 2.0, 3.0]
        assert metrics.median_cost_amplification == 2.0
        # For 3 values, 90th percentile should be close to the max
        assert metrics.cost_amplification_p90 > 2.5

    def test_amplification_properties(self) -> None:
        """Test amplification metric properties."""
        # Create metrics with required fields
        metrics = IndexingMetrics(
            start_time=0.0,
            end_time=10.0,
            source_document_tokens=1000,
            chunks_created=5,
            embedding_cost_per_1k=0.0001,
            summary_input_cost_per_1k=0.0025,  # gpt-4o mini input
            summary_output_cost_per_1k=0.01,  # gpt-4o mini output
        )

        # Add some amplification values
        metrics.cost_amplifications = [1.5, 2.0, 2.5, 3.0, 3.5]
        metrics.input_amplifications = [2.0, 3.0, 4.0, 5.0, 6.0]
        metrics.output_amplifications = [0.9, 1.0, 1.0, 1.1, 1.2]

        assert metrics.median_cost_amplification == 2.5
        # With 5 values: pos = (5-1)*0.9 = 3.6, so interpolate between index 3 and 4
        # values[3]=3.0, values[4]=3.5, fraction=0.6
        # p90 = 3.0 + 0.6*(3.5-3.0) = 3.3
        assert metrics.cost_amplification_p90 == pytest.approx(3.3, rel=0.01)
        # With 5 values: pos = (5-1)*0.95 = 3.8, so interpolate between index 3 and 4
        # p95 = 3.0 + 0.8*(3.5-3.0) = 3.4
        assert metrics.cost_amplification_p95 == pytest.approx(3.4, rel=0.01)
        assert metrics.median_input_amplification == 4.0
        assert metrics.median_output_amplification == 1.0


class TestIndexingMetricsReporter:
    """Test IndexingMetricsReporter functionality."""

    def test_summary_amplification_calculation(self) -> None:
        """Test that amplification is correctly calculated when recording summary results."""
        config = RagZoomConfig(
            openai_api_key="test-key",
            summary_model="gpt-4o-mini",
        )
        reporter = IndexingMetricsReporter("test", 10000, config)

        # Record a summary result
        reporter.record_summary_result(
            target_tokens=200,
            actual_tokens=180,
            prompt_tokens=500,  # 2x the input text
            completion_tokens=180,
            input_text_tokens=250,  # What we're summarizing
        )

        # Check amplifications were calculated correctly
        assert len(reporter.metrics.input_amplifications) == 1
        assert len(reporter.metrics.output_amplifications) == 1
        assert len(reporter.metrics.cost_amplifications) == 1

        # Input amplification = prompt_tokens / input_text_tokens = 500/250 = 2.0
        assert reporter.metrics.input_amplifications[0] == 2.0

        # Output amplification = completion_tokens / actual_tokens = 180/180 = 1.0
        assert reporter.metrics.output_amplifications[0] == 1.0

        # Cost amplification calculation:
        # actual_cost = (500 * 0.0025 + 180 * 0.01) / 1000 = 0.003050
        # min_cost = (250 * 0.0025 + 180 * 0.01) / 1000 = 0.002425
        # amplification = 0.003050 / 0.002425 = 1.258
        assert reporter.metrics.cost_amplifications[0] == pytest.approx(1.258, rel=0.01)

    def test_multiple_summaries_amplification(self) -> None:
        """Test amplification tracking across multiple summaries."""
        config = RagZoomConfig(openai_api_key="test-key")
        reporter = IndexingMetricsReporter("test", 10000, config)

        # Record multiple summaries with different amplifications
        test_cases = [
            # (target, actual, prompt, completion, input_text)
            (200, 200, 400, 200, 200),  # 2x input amp, 1x output
            (200, 180, 600, 180, 200),  # 3x input amp, 1x output
            (200, 220, 500, 220, 200),  # 2.5x input amp, 1x output
        ]

        for target, actual, prompt, completion, input_text in test_cases:
            reporter.record_summary_result(
                target_tokens=target,
                actual_tokens=actual,
                prompt_tokens=prompt,
                completion_tokens=completion,
                input_text_tokens=input_text,
            )

        # Check we have all amplifications
        assert len(reporter.metrics.input_amplifications) == 3
        assert reporter.metrics.input_amplifications == [2.0, 3.0, 2.5]

        # Check median calculations
        assert reporter.metrics.median_input_amplification == 2.5
        assert reporter.metrics.median_output_amplification == 1.0

    def test_zero_input_protection(self) -> None:
        """Test that zero input tokens don't cause division errors."""
        config = RagZoomConfig(openai_api_key="test-key")
        reporter = IndexingMetricsReporter("test", 10000, config)

        # Record with zero input tokens
        reporter.record_summary_result(
            target_tokens=200,
            actual_tokens=200,
            prompt_tokens=500,
            completion_tokens=200,
            input_text_tokens=0,  # Edge case
        )

        # Should not have recorded any amplifications
        assert len(reporter.metrics.input_amplifications) == 0
        assert len(reporter.metrics.output_amplifications) == 0
        assert len(reporter.metrics.cost_amplifications) == 0
