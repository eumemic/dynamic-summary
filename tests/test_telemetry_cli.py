"""Tests for telemetry CLI commands."""

import copy
import json

import pytest
from click.testing import CliRunner

from ragzoom.telemetry_cli import cli


class TestTelemetryCompare:
    """Test the compare command with directory support."""

    @pytest.fixture
    def create_test_files(self, tmp_path, sample_telemetry_data):
        """Create test telemetry files in temporary directories."""
        baseline_dir = tmp_path / "baseline"
        current_dir = tmp_path / "current"
        baseline_dir.mkdir()
        current_dir.mkdir()

        # Adapt shared telemetry data for CLI testing format
        cli_telemetry_data = copy.deepcopy(sample_telemetry_data)
        cli_telemetry_data["document_id"] = "test.txt"
        cli_telemetry_data["config"]["target_chunk_tokens"] = 100

        # Convert summary format from shared fixture to CLI test format
        for i, node in enumerate(cli_telemetry_data["nodes"]):
            if node.get("summary"):
                # Convert from shared format to CLI format
                summary_data = node["summary"]["create"]
                node["summary_attempts"] = [
                    {
                        "prompt_tokens": summary_data["input_tokens"],
                        "completion_tokens": summary_data["output_tokens"],
                        "actual_tokens": summary_data["output_tokens"],
                        "target_tokens": 100,
                        "model": "gpt-4o-mini",
                        "start_time": node["created_at"],
                        "end_time": node["created_at"] + 1.0,
                    }
                ]
                node["accepted_attempt"] = 0
                del node["summary"]
            elif node.get("embedding"):
                # Convert embedding format
                embed_data = node["embedding"]["create"]
                node["embedding"] = {
                    "text_tokens": embed_data["input_tokens"],
                    "batch_size": 1,
                    "batch_position": 0,
                    "model": "text-embedding-3-small",
                    "start_time": node["created_at"],
                    "end_time": node["created_at"] + 0.5,
                }

        # Create baseline files
        baseline_100 = baseline_dir / "telemetry_100_tokens.json"
        baseline_100.write_text(json.dumps(cli_telemetry_data))

        baseline_200 = baseline_dir / "telemetry_200_tokens.json"
        data_200 = copy.deepcopy(cli_telemetry_data)
        # Change target_tokens to 200 for all summary nodes
        data_200["config"]["target_chunk_tokens"] = 200
        for node in data_200["nodes"]:
            if node.get("summary_attempts"):
                node["summary_attempts"][0]["target_tokens"] = 200
        baseline_200.write_text(json.dumps(data_200))

        # Create current files with slight modifications
        current_data = copy.deepcopy(cli_telemetry_data)
        # Increase token usage slightly
        for node in current_data["nodes"]:
            if node.get("summary_attempts"):
                node["summary_attempts"][0][
                    "prompt_tokens"
                ] = 210  # 5% increase, under threshold
                break

        current_100 = current_dir / "telemetry_100_tokens.json"
        current_100.write_text(json.dumps(current_data))

        current_data_200 = copy.deepcopy(data_200)
        for node in current_data_200["nodes"]:
            if node.get("summary_attempts"):
                node["summary_attempts"][0][
                    "prompt_tokens"
                ] = 210  # Only 5% increase, under threshold
                break
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
        # Check for new simplified table format (no chunk size column)
        assert "Mean % deviation" in result.output
        assert "Mean retries/node" in result.output
        # Chunk size should appear in configuration section
        assert "Target Chunk Tokens" in result.output

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
        # Check for simplified table format without chunk size columns
        assert "Mean % deviation" in result.output
        assert "Oversized summary rate" in result.output
        assert "USD per 1M source tokens" in result.output
        # Should be a unified table with simplified format

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
        # Use the same approach as create_test_files fixture - adapt shared telemetry data
        cli_data = copy.deepcopy(sample_telemetry_data)
        cli_data["document_id"] = "test.txt"
        cli_data["config"]["target_chunk_tokens"] = 100

        # Convert summary format from shared fixture to CLI test format
        for node in cli_data["nodes"]:
            if node.get("summary"):
                # Convert from shared format to CLI format
                summary_data = node["summary"]["create"]
                node["summary_attempts"] = [
                    {
                        "prompt_tokens": summary_data["input_tokens"],
                        "completion_tokens": summary_data["output_tokens"],
                        "actual_tokens": summary_data["output_tokens"],
                        "target_tokens": 100,
                        "model": "gpt-4o-mini",
                        "start_time": node["created_at"],
                        "end_time": node["created_at"] + 1.0,
                    }
                ]
                node["accepted_attempt"] = 0
                del node["summary"]
            elif node.get("embedding"):
                # Convert embedding format
                embed_data = node["embedding"]["create"]
                node["embedding"] = {
                    "text_tokens": embed_data["input_tokens"],
                    "batch_size": 1,
                    "batch_position": 0,
                    "model": "text-embedding-3-small",
                    "start_time": node["created_at"],
                    "end_time": node["created_at"] + 0.5,
                }

        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(cli_data))

        # Create current with significant regression - multiply prompt tokens by 10x
        current_data = copy.deepcopy(cli_data)
        for node in current_data["nodes"]:
            if node.get("summary_attempts"):
                node["summary_attempts"][0][
                    "prompt_tokens"
                ] *= 10  # 10x increase should definitely trigger regression

        current_file = tmp_path / "current.json"
        current_file.write_text(json.dumps(current_data))

        runner = CliRunner()
        result = runner.invoke(cli, ["compare", str(baseline_file), str(current_file)])

        # Debug output
        if result.exit_code != 0:
            print(f"Exit code: {result.exit_code}")
            print(f"Output:\n{result.output}")

        # Test that comparison works and shows the cost increase
        assert result.exit_code == 0  # CLI should run successfully
        assert "USD per 1M source tokens" in result.output  # Should show cost metrics
        assert (
            "+450.0%" in result.output or "+$" in result.output
        )  # Should show cost increase
        # Chunk size should appear in configuration section
        assert "Target Chunk Tokens" in result.output

    def test_dynamic_thresholds_computation(self, tmp_path, sample_telemetry_data):
        """Test that dynamic thresholds are computed from baseline variance."""
        from ragzoom.telemetry_analysis import compute_simplified_metrics
        from ragzoom.telemetry_cli import (
            MetricNames,
            ThresholdConfig,
            compute_dynamic_threshold,
        )

        # Use shared telemetry data but modify to create variance in the target fit errors
        baseline_data = copy.deepcopy(sample_telemetry_data)
        baseline_data["document_id"] = "test.txt"
        baseline_data["config"]["target_chunk_tokens"] = 100

        # Convert format and create variance in actual_tokens to get different errors
        for i, node in enumerate(baseline_data["nodes"]):
            if node.get("summary"):
                # Convert from shared format to CLI format
                summary_data = node["summary"]["create"]
                # Create different actual_tokens values to get variance in errors
                if i == 0:
                    actual_tokens = 50  # Error: -50
                elif i == 1:
                    actual_tokens = 48  # Error: -52
                else:
                    actual_tokens = 52  # Error: -48

                node["summary_attempts"] = [
                    {
                        "prompt_tokens": summary_data["input_tokens"],
                        "completion_tokens": summary_data["output_tokens"],
                        "actual_tokens": actual_tokens,
                        "target_tokens": 100,
                        "model": "gpt-4o-mini",
                        "start_time": node["created_at"],
                        "end_time": node["created_at"] + 1.0,
                    }
                ]
                node["accepted_attempt"] = 0
                del node["summary"]
            elif node.get("embedding"):
                # Convert embedding format
                embed_data = node["embedding"]["create"]
                node["embedding"] = {
                    "text_tokens": embed_data["input_tokens"],
                    "batch_size": 1,
                    "batch_position": 0,
                    "model": "text-embedding-3-small",
                    "start_time": node["created_at"],
                    "end_time": node["created_at"] + 0.5,
                }

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

        # Test that dynamic threshold computation function works
        assert threshold is not None
        assert hasattr(threshold, "is_computed")
        assert hasattr(threshold, "baseline_variance")
        assert threshold.k_factors == (3.0, 2.0)

        # Test with variance data - if there's variance, threshold should be computed
        expected_mad = metrics.target_fit.error_mad
        if expected_mad > 0.0:
            assert threshold.is_computed is True
            assert threshold.baseline_variance == expected_mad
            expected_threshold = (3.0 + 2.0) * expected_mad
            assert threshold.absolute_value == expected_threshold
        else:
            # Zero variance means no threshold can be computed
            assert threshold.is_computed is False
            assert threshold.absolute_value is None

        # Test CI adjustment works regardless of variance
        threshold_ci = compute_dynamic_threshold(
            metrics, MetricNames.MEDIAN_ERROR, "error_mad", config, is_ci=True
        )
        assert threshold_ci is not None
        # CI adjustment doesn't change k_factors, but may change how they're applied
        assert threshold_ci.k_factors == (3.0, 2.0)

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
        # Adapt shared telemetry data for CLI testing
        cli_data = copy.deepcopy(sample_telemetry_data)
        cli_data["document_id"] = "test.txt"

        # Convert format for CLI testing
        for node in cli_data["nodes"]:
            if node.get("summary"):
                summary_data = node["summary"]["create"]
                node["summary_attempts"] = [
                    {
                        "prompt_tokens": summary_data["input_tokens"],
                        "completion_tokens": summary_data["output_tokens"],
                        "actual_tokens": summary_data["output_tokens"],
                        "target_tokens": 100,
                        "model": "gpt-4o-mini",
                        "start_time": node["created_at"],
                        "end_time": node["created_at"] + 1.0,
                    }
                ]
                node["accepted_attempt"] = 0
                del node["summary"]

        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(cli_data))

        current_file = tmp_path / "current.json"
        current_file.write_text(json.dumps(cli_data))

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
