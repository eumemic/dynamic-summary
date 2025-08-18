"""Tests for telemetry CLI commands."""

import copy
import json

import pytest
from click.testing import CliRunner

from ragzoom.telemetry_cli import cli


class TestTelemetryCompare:
    """Test the compare command with directory support."""

    @pytest.fixture
    def sample_telemetry_data(self):
        """Create sample telemetry data for testing."""
        # Return just the telemetry structure, not wrapped in another object
        return {
            "format_version": "4.1",
            "document_id": "test.txt",
            "source_document_tokens": 1000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "node1",
                    "height": 0,
                    "created_at": 1234567890.0,
                    "embedding": {
                        "text_tokens": 100,
                        "batch_size": 10,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 1234567890.0,
                        "end_time": 1234567890.5,
                    },
                },
                {
                    "node_id": "node2",
                    "height": 1,
                    "created_at": 1234567891.0,
                    "summary_attempts": [
                        {
                            "prompt_tokens": 200,
                            "completion_tokens": 50,
                            "actual_tokens": 50,
                            "target_tokens": 100,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567891.0,
                            "end_time": 1234567892.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
                {
                    "node_id": "node3",
                    "height": 1,
                    "created_at": 1234567892.0,
                    "summary_attempts": [
                        {
                            "prompt_tokens": 210,
                            "completion_tokens": 48,
                            "actual_tokens": 48,
                            "target_tokens": 100,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567892.0,
                            "end_time": 1234567893.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
                {
                    "node_id": "node4",
                    "height": 1,
                    "created_at": 1234567893.0,
                    "summary_attempts": [
                        {
                            "prompt_tokens": 195,
                            "completion_tokens": 52,
                            "actual_tokens": 52,
                            "target_tokens": 100,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567893.0,
                            "end_time": 1234567894.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
            ],
        }

    @pytest.fixture
    def create_test_files(self, tmp_path, sample_telemetry_data):
        """Create test telemetry files in temporary directories."""
        baseline_dir = tmp_path / "baseline"
        current_dir = tmp_path / "current"
        baseline_dir.mkdir()
        current_dir.mkdir()

        # Create baseline files
        baseline_100 = baseline_dir / "telemetry_100_tokens.json"
        baseline_100.write_text(json.dumps(sample_telemetry_data))

        baseline_200 = baseline_dir / "telemetry_200_tokens.json"
        data_200 = copy.deepcopy(sample_telemetry_data)
        # Change target_tokens to 200 for all summary nodes
        data_200["config"]["target_chunk_tokens"] = 200
        for i in [1, 2, 3]:
            data_200["nodes"][i]["summary_attempts"][0]["target_tokens"] = 200
        baseline_200.write_text(json.dumps(data_200))

        # Create current files with slight modifications
        current_data = copy.deepcopy(sample_telemetry_data)
        # Increase token usage slightly
        current_data["nodes"][1]["summary_attempts"][0][
            "prompt_tokens"
        ] = 210  # 5% increase, under threshold

        current_100 = current_dir / "telemetry_100_tokens.json"
        current_100.write_text(json.dumps(current_data))

        current_data_200 = copy.deepcopy(data_200)
        current_data_200["nodes"][1]["summary_attempts"][0][
            "prompt_tokens"
        ] = 210  # Only 5% increase, under threshold
        current_200 = current_dir / "telemetry_200_tokens.json"
        current_200.write_text(json.dumps(current_data_200))

        return baseline_dir, current_dir

    def test_compare_single_files(self, create_test_files):
        """Test comparing two individual files."""
        baseline_dir, current_dir = create_test_files
        baseline_file = baseline_dir / "telemetry_100_tokens.json"
        current_file = current_dir / "telemetry_100_tokens.json"

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(baseline_file), str(current_file)])

        # Debug output
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output:\n{result.output}")
            if result.exception:
                import traceback

                traceback.print_exception(
                    type(result.exception),
                    result.exception,
                    result.exception.__traceback__,
                )

        assert result.exit_code == 0
        # Check for new table format
        assert "100 tokens" in result.output
        assert "Median error" in result.output
        assert "Avg retries/node" in result.output

    def test_compare_directories(self, create_test_files):
        """Test comparing two directories with matching files."""
        baseline_dir, current_dir = create_test_files

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(baseline_dir), str(current_dir)])

        # Debug output
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output: {result.output}")
            print(f"Exception: {result.exception}")

        assert result.exit_code == 0
        # Check for unified table format with chunk sizes
        assert "100 tokens" in result.output
        assert "200 tokens" in result.output
        assert "Median error" in result.output
        # Should be a unified table, not separate sections per file

    def test_compare_directories_with_output(self, create_test_files, tmp_path):
        """Test comparing directories with markdown output."""
        baseline_dir, current_dir = create_test_files

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                str(baseline_dir),
                str(current_dir),
            ],
        )

        assert result.exit_code == 0
        # Check for markdown format
        assert "| Chunk Size |" in result.output or "| Metric |" in result.output

    def test_compare_directories_no_matches(self, tmp_path):
        """Test comparing directories with no matching files."""
        baseline_dir = tmp_path / "baseline"
        current_dir = tmp_path / "current"
        baseline_dir.mkdir()
        current_dir.mkdir()

        # Create files with different names
        (baseline_dir / "telemetry_100_tokens.json").write_text("{}")
        (current_dir / "telemetry_200_tokens.json").write_text("{}")

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(baseline_dir), str(current_dir)])

        assert result.exit_code == 1
        assert "No matching telemetry files found" in result.output

    def test_compare_mixed_types_error(self, tmp_path, sample_telemetry_data):
        """Test error when comparing a file with a directory."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        test_file = tmp_path / "test_file.json"
        test_file.write_text(json.dumps(sample_telemetry_data))

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(test_file), str(test_dir)])

        assert result.exit_code == 1
        assert (
            "Error: Both arguments must be either files or directories" in result.output
        )

    def test_file_matching_logic(self, tmp_path):
        """Test the file matching logic handles various naming patterns."""
        from ragzoom.telemetry_cli import _match_telemetry_files

        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()

        # Create various file patterns
        (dir1 / "telemetry_100_tokens.json").touch()
        (dir1 / "telemetry_200_tokens.json").touch()
        (dir1 / "telemetry.json").touch()
        (dir1 / "other_file.json").touch()  # Should not match

        (dir2 / "telemetry_100_tokens.json").touch()
        (dir2 / "telemetry_200_tokens.json").touch()
        (dir2 / "telemetry.json").touch()
        (dir2 / "telemetry_300_tokens.json").touch()  # Only in dir2

        indexing_matches, query_matches = _match_telemetry_files(dir1, dir2)

        # Should match 3 indexing files
        assert len(indexing_matches) == 3
        assert len(query_matches) == 0  # No query files in this test

        match_names = [m[0].name for m in indexing_matches]
        assert "telemetry_100_tokens.json" in match_names
        assert "telemetry_200_tokens.json" in match_names
        assert "telemetry.json" in match_names
        assert "other_file.json" not in match_names
        assert "telemetry_300_tokens.json" not in match_names

    def test_compare_with_regression(self, tmp_path, sample_telemetry_data):
        """Test that regressions are detected and exit code is 1."""
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(sample_telemetry_data))

        # Create current with significant regression
        current_data = copy.deepcopy(sample_telemetry_data)
        # Double the prompt tokens to trigger cost regression (>10% threshold)
        for i in [1, 2, 3]:  # All summary nodes
            current_data["nodes"][i]["summary_attempts"][0]["prompt_tokens"] = (
                current_data["nodes"][i]["summary_attempts"][0]["prompt_tokens"] * 2
            )  # Double the prompt tokens

        current_file = tmp_path / "current.json"
        current_file.write_text(json.dumps(current_data))

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(baseline_file), str(current_file)])

        # Debug output
        if result.exit_code != 1:
            print(f"Exit code: {result.exit_code}")
            print(f"Output:\n{result.output}")

        assert result.exit_code == 1
        assert (
            "Performance regression detected" in result.output or "❌" in result.output
        )

    def test_dynamic_thresholds_computation(self, tmp_path, sample_telemetry_data):
        """Test that dynamic thresholds are computed from baseline variance."""
        from ragzoom.telemetry_analysis import compute_simplified_metrics
        from ragzoom.telemetry_cli import (
            MetricNames,
            ThresholdConfig,
            compute_dynamic_threshold,
        )

        # Create baseline with known variance
        baseline_data = copy.deepcopy(sample_telemetry_data)
        # Add variance by modifying node errors
        # nodes = baseline_data["documents"]["test.txt"]["nodes"]
        # Node 2: actual=50, target=100, error=-50
        # Node 3: actual=48, target=100, error=-52
        # Node 4: actual=52, target=100, error=-48
        # This gives us MAD = 2.0 tokens

        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(baseline_data))

        # Analyze to get metrics
        baseline_analysis = compute_simplified_metrics(baseline_data)
        metrics = baseline_analysis.metrics_by_chunk_size[100]

        # Test dynamic threshold computation
        config = ThresholdConfig()
        threshold = compute_dynamic_threshold(
            metrics, "median_error", "error_mad", config, is_ci=False
        )

        # Check that dynamic threshold was computed
        assert threshold.is_computed is True
        assert threshold.baseline_variance == metrics.target_fit.error_mad
        assert threshold.k_factors == (3.0, 2.0)

        # The MAD comes from the actual data - errors are [-50, -52, -48]
        # Median error = -50, MAD = median([0, 2, 2]) = 2.0
        expected_mad = metrics.target_fit.error_mad
        if expected_mad == 0.0:  # Zero variance means no threshold
            assert threshold.absolute_value is None
            assert not threshold.is_computed
        else:
            expected_threshold = (3.0 + 2.0) * expected_mad
            assert threshold.absolute_value == expected_threshold
            assert threshold.is_computed

        # Test CI adjustment
        threshold_ci = compute_dynamic_threshold(
            metrics, MetricNames.MEDIAN_ERROR, "error_mad", config, is_ci=True
        )
        # CI adjustment multiplies k-factors by 1.5
        if expected_mad == 0.0:  # Zero variance means no threshold
            assert threshold_ci.absolute_value is None
        else:
            expected_ci_threshold = (3.0 * 1.5 + 2.0 * 1.5) * expected_mad
            assert threshold_ci.absolute_value == expected_ci_threshold

    def test_dynamic_thresholds_emoji_logic(self, tmp_path, sample_telemetry_data):
        """Test emoji assignment based on variance thresholds."""
        from ragzoom.telemetry_cli import (
            DynamicThreshold,
            get_change_emoji,
        )

        # Create threshold with known variance
        threshold = DynamicThreshold(
            absolute_value=50.0,  # 5-sigma threshold
            baseline_variance=10.0,  # 1-sigma
            k_factors=(3.0, 2.0),
            metric_name="median_error",
            is_computed=True,
            emoji_significance_sigma=1.0,
        )

        # Test for error metrics (lower is better)
        # No change: within 1-sigma
        assert (
            get_change_emoji(5.0, higher_is_better=False, threshold=threshold) == "⚪"
        )

        # Degradation: >1-sigma but <5-sigma
        assert (
            get_change_emoji(15.0, higher_is_better=False, threshold=threshold) == "🟡"
        )

        # Improvement: >1-sigma in good direction
        assert (
            get_change_emoji(-15.0, higher_is_better=False, threshold=threshold) == "🟢"
        )

        # Regression: >5-sigma threshold
        assert (
            get_change_emoji(55.0, higher_is_better=False, threshold=threshold) == "🔴"
        )

        # Test for metrics where higher is better
        assert (
            get_change_emoji(-55.0, higher_is_better=True, threshold=threshold) == "🔴"
        )
        assert (
            get_change_emoji(15.0, higher_is_better=True, threshold=threshold) == "🟢"
        )

    def test_variance_metrics_in_output(self, tmp_path, sample_telemetry_data):
        """Test that variance metrics are included in analysis output."""
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(sample_telemetry_data))

        current_file = tmp_path / "current.json"
        current_file.write_text(json.dumps(sample_telemetry_data))

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(baseline_file), str(current_file)])

        assert result.exit_code == 0
        # Check that we have markdown output
        assert (
            "| Metric |" in result.output
            or "Performance Comparison Report" in result.output
        )
        # Check for regression indicator
        assert "regression" in result.output.lower() or "✅" in result.output

    def test_emotional_feedback_functions(self):
        """Test the emotional feedback emoji functions."""
        from ragzoom.telemetry_cli import (
            DynamicThreshold,
            get_change_emoji,
            get_variance_emoji,
        )

        # Create a simple threshold for testing
        threshold = DynamicThreshold(
            absolute_value=100.0,
            baseline_variance=10.0,
            k_factors=(3.0, 2.0),
            metric_name="test_metric",
            is_computed=True,
            emoji_significance_sigma=1.0,
        )

        # Test metric change emoji with threshold
        # No change (below 1-sigma threshold of 10.0)
        assert (
            get_change_emoji(5.0, higher_is_better=False, threshold=threshold) == "⚪"
        )
        assert (
            get_change_emoji(-5.0, higher_is_better=True, threshold=threshold) == "⚪"
        )

        # Significant desirable changes (>1-sigma)
        assert (
            get_change_emoji(-15.0, higher_is_better=False, threshold=threshold) == "🟢"
        )
        assert (
            get_change_emoji(15.0, higher_is_better=True, threshold=threshold) == "🟢"
        )

        # Significant undesirable changes (>1-sigma but <threshold)
        assert (
            get_change_emoji(15.0, higher_is_better=False, threshold=threshold) == "🟡"
        )
        assert (
            get_change_emoji(-15.0, higher_is_better=True, threshold=threshold) == "🟡"
        )

        # Regression (exceeds full threshold of 100.0)
        assert (
            get_change_emoji(101.0, higher_is_better=False, threshold=threshold) == "🔴"
        )
        assert (
            get_change_emoji(-101.0, higher_is_better=True, threshold=threshold) == "🔴"
        )

        # Test variance emoji (new signature with baseline)
        # No change (below 50% threshold)
        assert get_variance_emoji(3.0, 10.0) == "⚪"  # 30% change
        assert get_variance_emoji(-3.0, 10.0) == "⚪"  # -30% change

        # Significant variance changes (>50% of baseline)
        assert get_variance_emoji(6.0, 10.0) == "🟡"  # 60% increase - notable
        assert (
            get_variance_emoji(-6.0, 10.0) == "🟢"
        )  # 60% decrease - improved stability


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
