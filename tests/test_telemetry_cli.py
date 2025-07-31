"""Tests for telemetry CLI commands."""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from ragzoom.telemetry_cli import cli


class TestTelemetryCompare:
    """Test the compare command with directory support."""

    @pytest.fixture
    def sample_telemetry_data(self):
        """Create sample telemetry data for testing."""
        return {
            "format_version": "2.0",
            "config": {"leaf_tokens": 100},
            "document": {"path": "test.txt", "total_chunks": 10},
            "telemetry": {
                "format_version": "2.0",
                "documents": {
                    "test.txt": {
                        "metadata": {"source_document_tokens": 1000},
                        "nodes": [
                            {
                                "node_id": "node1",
                                "height": 0,
                                "created_at": 1234567890.0,
                                "embedding": {
                                    "text_tokens": 100,
                                    "batch_size": 10,
                                    "start_time": 1234567890.0,
                                },
                            },
                            {
                                "node_id": "node2",
                                "height": 1,
                                "created_at": 1234567891.0,
                                "summary_attempts": [
                                    {
                                        "status": "accepted",
                                        "prompt_tokens": 200,
                                        "completion_tokens": 50,
                                        "input_text_tokens": 100,
                                        "actual_tokens": 50,
                                        "target_tokens": 50,
                                    }
                                ],
                            },
                        ],
                    }
                },
            },
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
        data_200 = sample_telemetry_data.copy()
        data_200["config"]["leaf_tokens"] = 200
        baseline_200.write_text(json.dumps(data_200))

        # Create current files with slight modifications
        current_data = sample_telemetry_data.copy()
        # Increase amplification slightly
        current_data["telemetry"]["documents"]["test.txt"]["nodes"][1][
            "summary_attempts"
        ][0][
            "prompt_tokens"
        ] = 210  # 5% increase, under threshold

        current_100 = current_dir / "telemetry_100_tokens.json"
        current_100.write_text(json.dumps(current_data))

        current_data_200 = data_200.copy()
        current_data_200["telemetry"]["documents"]["test.txt"]["nodes"][1][
            "summary_attempts"
        ][0][
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

        assert result.exit_code == 0
        assert "Performance Comparison Report" in result.output
        assert "Amplification Metrics" in result.output

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
        assert "Found 2 matching file pairs to compare" in result.output
        assert "telemetry_100_tokens.json" in result.output
        assert "telemetry_200_tokens.json" in result.output
        assert "Directory Comparison Report" in result.output

    def test_compare_directories_with_output(self, create_test_files, tmp_path):
        """Test comparing directories with output to file."""
        baseline_dir, current_dir = create_test_files
        output_file = tmp_path / "comparison_report.md"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "compare",
                str(baseline_dir),
                str(current_dir),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        assert "Combined comparison report saved to" in result.output

        # Check report contents
        report = output_file.read_text()
        assert "Directory Comparison Report" in report
        assert "telemetry_100_tokens.json" in report
        assert "telemetry_200_tokens.json" in report

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
        assert "Both arguments must be either files or directories" in result.output

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

        matches = _match_telemetry_files(dir1, dir2)

        # Should match 3 files
        assert len(matches) == 3
        match_names = [m[0].name for m in matches]
        assert "telemetry_100_tokens.json" in match_names
        assert "telemetry_200_tokens.json" in match_names
        assert "telemetry.json" in match_names
        assert "other_file.json" not in match_names
        assert "telemetry_300_tokens.json" not in match_names

    def test_compare_with_regression(self, tmp_path, sample_telemetry_data):
        """Test that regressions are detected and exit code is 1."""
        baseline_dir = tmp_path / "baseline"
        current_dir = tmp_path / "current"
        baseline_dir.mkdir()
        current_dir.mkdir()

        # Create baseline
        baseline_file = baseline_dir / "telemetry_100_tokens.json"
        baseline_file.write_text(json.dumps(sample_telemetry_data))

        # Create current with significant regression
        current_data = sample_telemetry_data.copy()
        # Increase prompt tokens by >10% to trigger regression
        current_data["telemetry"]["documents"]["test.txt"]["nodes"][1][
            "summary_attempts"
        ][0][
            "prompt_tokens"
        ] = 500  # 2.5x increase

        current_file = current_dir / "telemetry_100_tokens.json"
        current_file.write_text(json.dumps(current_data))

        runner = CliRunner()

        # Set low threshold to ensure regression is detected
        with patch.dict(
            "os.environ", {"PERF_SUMMARY_TOKEN_REGRESSION_THRESHOLD": "5.0"}
        ):
            result = runner.invoke(
                cli, ["compare", str(baseline_dir), str(current_dir)]
            )

        assert result.exit_code == 1
        assert "Regression detected" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
