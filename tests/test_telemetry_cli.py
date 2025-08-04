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
            "format_version": "2.0",
            "documents": {
                "test.txt": {
                    "metadata": {
                        "source_document_tokens": 1000,
                        "indexing_start": 1234567890.0,
                        "indexing_end": 1234567900.0,
                    },
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
                                    "target_tokens": 100,
                                    "start_time": 1234567891.0,
                                    "end_time": 1234567892.0,
                                }
                            ],
                        },
                        {
                            "node_id": "node3",
                            "height": 1,
                            "created_at": 1234567892.0,
                            "summary_attempts": [
                                {
                                    "status": "accepted",
                                    "prompt_tokens": 210,
                                    "completion_tokens": 48,
                                    "input_text_tokens": 100,
                                    "actual_tokens": 48,
                                    "target_tokens": 100,
                                    "start_time": 1234567892.0,
                                    "end_time": 1234567893.0,
                                }
                            ],
                        },
                        {
                            "node_id": "node4",
                            "height": 1,
                            "created_at": 1234567893.0,
                            "summary_attempts": [
                                {
                                    "status": "accepted",
                                    "prompt_tokens": 195,
                                    "completion_tokens": 52,
                                    "input_text_tokens": 100,
                                    "actual_tokens": 52,
                                    "target_tokens": 100,
                                    "start_time": 1234567893.0,
                                    "end_time": 1234567894.0,
                                }
                            ],
                        },
                    ],
                }
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
        data_200 = copy.deepcopy(sample_telemetry_data)
        # Change target_tokens to 200 for all summary nodes
        for i in [1, 2, 3]:
            data_200["documents"]["test.txt"]["nodes"][i]["summary_attempts"][0][
                "target_tokens"
            ] = 200
        baseline_200.write_text(json.dumps(data_200))

        # Create current files with slight modifications
        current_data = copy.deepcopy(sample_telemetry_data)
        # Increase token usage slightly
        current_data["documents"]["test.txt"]["nodes"][1]["summary_attempts"][0][
            "prompt_tokens"
        ] = 210  # 5% increase, under threshold

        current_100 = current_dir / "telemetry_100_tokens.json"
        current_100.write_text(json.dumps(current_data))

        current_data_200 = copy.deepcopy(data_200)
        current_data_200["documents"]["test.txt"]["nodes"][1]["summary_attempts"][0][
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
        assert "Retry rate" in result.output

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
                "--output",
                "markdown",
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
        baseline_file = tmp_path / "baseline.json"
        baseline_file.write_text(json.dumps(sample_telemetry_data))

        # Create current with significant regression
        current_data = copy.deepcopy(sample_telemetry_data)
        # Double the prompt tokens to trigger cost regression (>10% threshold)
        for i in [1, 2, 3]:  # All summary nodes
            current_data["documents"]["test.txt"]["nodes"][i]["summary_attempts"][0][
                "prompt_tokens"
            ] = (
                current_data["documents"]["test.txt"]["nodes"][i]["summary_attempts"][
                    0
                ]["prompt_tokens"]
                * 2
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
