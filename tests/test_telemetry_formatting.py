"""Tests for telemetry report formatting functions."""

from ragzoom.telemetry_cli import (
    _format_amplification_section,
    _format_efficiency_section,
    _format_retry_patterns_section,
    _format_summary_accuracy_section,
)


class TestTelemetryFormatting:
    """Test the telemetry report formatting functions."""

    def test_format_summary_accuracy_section_with_data(self):
        """Test summary accuracy formatting with valid data."""
        baseline_metrics = {
            "summary_accuracy": {
                200: {
                    "avg_deviation": 20.5,
                    "median_deviation": 18.2,
                    "p95_deviation": 35.0,
                }
            }
        }
        current_metrics = {
            "summary_accuracy": {
                200: {
                    "avg_deviation": 25.5,  # 24% increase - should trigger warning
                    "median_deviation": 30.2,  # 66% increase - should trigger regression
                    "p95_deviation": 40.0,
                }
            }
        }
        thresholds = {"change_significance": 10.0}

        lines, has_regression = _format_summary_accuracy_section(
            baseline_metrics, current_metrics, thresholds
        )

        # Check structure
        assert "### 📏 Summary Size Accuracy" in lines
        assert any("Average Deviation" in line for line in lines)
        assert any("Median Deviation" in line for line in lines)
        assert any("P95 Deviation" in line for line in lines)

        # Check regression detection
        assert has_regression  # Should detect regression due to median deviation

    def test_format_summary_accuracy_section_no_data(self):
        """Test summary accuracy formatting with no data."""
        baseline_metrics = {}
        current_metrics = {}
        thresholds = {"change_significance": 10.0}

        lines, has_regression = _format_summary_accuracy_section(
            baseline_metrics, current_metrics, thresholds
        )

        assert lines == []
        assert not has_regression

    def test_format_amplification_section(self):
        """Test amplification metrics formatting."""
        baseline_metrics = {
            "amplification": {
                "median_cost": 2.5,
                "cost_p90": 3.0,
                "cost_p95": 3.5,
                "median_input": 1.5,
                "median_output": 2.0,
            }
        }
        current_metrics = {
            "amplification": {
                "median_cost": 3.0,  # 20% increase
                "cost_p90": 3.3,
                "cost_p95": 3.8,
                "median_input": 1.6,
                "median_output": 2.1,
            }
        }
        thresholds = {
            "amplification": {"Median Cost Amplification": 15.0},
            "change_significance": 10.0,
        }

        lines, has_regression = _format_amplification_section(
            baseline_metrics, current_metrics, thresholds
        )

        assert "### 📈 Amplification Metrics" in lines
        assert any("Median Cost Amplification" in line for line in lines)
        assert has_regression  # 20% increase exceeds 15% threshold

    def test_format_efficiency_section(self):
        """Test efficiency metrics formatting."""
        baseline_metrics = {
            "efficiency": {"avg_embedding_batch_size": 50.0, "batch_utilization": 80.0}
        }
        current_metrics = {
            "efficiency": {"avg_embedding_batch_size": 45.0, "batch_utilization": 75.0}
        }

        lines = _format_efficiency_section(baseline_metrics, current_metrics)

        assert "### 📦 Efficiency Metrics" in lines
        assert any("Avg Embedding Batch Size" in line for line in lines)
        assert any("Batch Utilization" in line for line in lines)
        assert any("75.0%" in line for line in lines)  # Current utilization

    def test_format_retry_patterns_section_with_retries(self):
        """Test retry patterns formatting when retries are present."""
        baseline_metrics = {
            "retry_patterns": {"retry_rate": 5.0, "retry_success_rate": 90.0}
        }
        current_metrics = {
            "retry_patterns": {"retry_rate": 10.0, "retry_success_rate": 85.0}
        }

        lines = _format_retry_patterns_section(baseline_metrics, current_metrics)

        assert "### 🔄 Retry Patterns" in lines
        assert any("Retry Rate" in line for line in lines)
        assert any("Retry Success Rate" in line for line in lines)

    def test_format_retry_patterns_section_no_retries(self):
        """Test retry patterns formatting when no retries occurred."""
        baseline_metrics = {"retry_patterns": {"retry_rate": 0.0}}
        current_metrics = {"retry_patterns": {"retry_rate": 0.0}}

        lines = _format_retry_patterns_section(baseline_metrics, current_metrics)

        assert lines == []  # Should not show section when no retries

    def test_format_retry_patterns_new_retries(self):
        """Test retry patterns formatting when new retries appear."""
        baseline_metrics = {"retry_patterns": {"retry_rate": 0.0}}
        current_metrics = {
            "retry_patterns": {"retry_rate": 5.0, "retry_success_rate": 100.0}
        }

        lines = _format_retry_patterns_section(baseline_metrics, current_metrics)

        assert "### 🔄 Retry Patterns" in lines
        assert any("New retries ⚠️" in line for line in lines)
