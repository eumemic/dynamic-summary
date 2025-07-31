"""Test telemetry visualization functions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ragzoom.telemetry_viz import TelemetryVisualizer


class TestTelemetryVisualizer:
    """Test TelemetryVisualizer class."""

    @pytest.fixture
    def temp_output_dir(self) -> Path:
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def visualizer(self, temp_output_dir: Path) -> TelemetryVisualizer:
        """Create a TelemetryVisualizer instance."""
        return TelemetryVisualizer(temp_output_dir)

    @pytest.fixture
    def sample_telemetry_data(self) -> dict:
        """Create sample telemetry data for testing."""
        return {
            "format_version": "2.0",
            "documents": {
                "test_doc": {
                    "metadata": {
                        "source_document_tokens": 1000,
                        "chunk_size": 200,
                        "indexed_at": 1234567890.0,
                    },
                    "nodes": [
                        {
                            "node_id": "leaf-1",
                            "height": 0,
                            "created_at": 1234567890.0,
                            "embedding": {
                                "text_tokens": 195,
                                "batch_size": 1,
                                "model": "text-embedding-3-small",
                                "start_time": 1234567890.0,
                                "end_time": 1234567891.0,
                            },
                        },
                        {
                            "node_id": "summary-1",
                            "height": 1,
                            "created_at": 1234567892.0,
                            "summary_attempts": [
                                {
                                    "status": "accepted",
                                    "is_retry": False,
                                    "prompt_tokens": 400,
                                    "completion_tokens": 100,
                                    "input_text_tokens": 195,
                                    "actual_tokens": 98,
                                    "target_tokens": 100,
                                }
                            ],
                        },
                    ],
                }
            },
        }

    @pytest.fixture
    def sample_benchmark_data(self, sample_telemetry_data: dict) -> dict:
        """Create sample benchmark data including telemetry."""
        return {
            "config": {"leaf_tokens": 200},
            "telemetry": sample_telemetry_data,
            "metrics": {
                "summary_accuracy": {
                    "200": {
                        "deviations": [-2.0, 1.5, -0.5, 2.1],
                    }
                }
            },
        }

    def test_visualizer_initialization(self, temp_output_dir: Path) -> None:
        """Test TelemetryVisualizer initialization."""
        visualizer = TelemetryVisualizer(temp_output_dir)
        assert visualizer.output_dir == temp_output_dir
        assert temp_output_dir.exists()
        assert hasattr(visualizer, "thresholds")

    def test_load_benchmark_data(
        self, visualizer: TelemetryVisualizer, temp_output_dir: Path
    ) -> None:
        """Test loading benchmark data from JSON."""
        test_data = {"test": "data"}
        test_file = temp_output_dir / "test.json"

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        loaded_data = visualizer.load_benchmark_data(test_file)
        assert loaded_data == test_data

    def test_calculate_histogram_bins_small_discrete(
        self, visualizer: TelemetryVisualizer
    ) -> None:
        """Test histogram bin calculation for small discrete values."""
        batch_sizes = [1, 1, 1, 2, 2, 3]
        bins, align = visualizer._calculate_histogram_bins(batch_sizes)

        assert align == "left"
        assert isinstance(bins, list)
        assert bins == [0, 1, 2, 3, 4]  # Covers values 0-3

    def test_calculate_histogram_bins_medium_range(
        self, visualizer: TelemetryVisualizer
    ) -> None:
        """Test histogram bin calculation for medium range values."""
        batch_sizes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        bins, align = visualizer._calculate_histogram_bins(batch_sizes)

        assert align == "left"
        assert isinstance(bins, list)
        assert bins[0] == 0
        assert bins[-1] >= 45

    def test_calculate_histogram_bins_large_range(
        self, visualizer: TelemetryVisualizer
    ) -> None:
        """Test histogram bin calculation for large range values."""
        batch_sizes = list(range(1, 150, 10))  # 1, 11, 21, ..., 141
        bins, align = visualizer._calculate_histogram_bins(batch_sizes)

        assert align == "mid"
        assert isinstance(bins, int)
        assert bins == visualizer.LARGE_BIN_COUNT

    def test_visualize_single_benchmark_creates_outputs(
        self, visualizer: TelemetryVisualizer, temp_output_dir: Path
    ) -> None:
        """Test that visualize_single_benchmark creates expected outputs."""
        # Create minimal telemetry data that won't trigger complex plotting
        minimal_data = {
            "config": {"leaf_tokens": 200},
            "telemetry": {"format_version": "2.0", "documents": {}},
            "metrics": {},
        }

        test_file = temp_output_dir / "minimal_test.json"
        with open(test_file, "w") as f:
            json.dump(minimal_data, f)

        # Mock matplotlib to avoid actual plotting
        with (
            patch("matplotlib.pyplot.figure"),
            patch("matplotlib.pyplot.savefig"),
            patch("matplotlib.pyplot.close"),
            patch("matplotlib.pyplot.tight_layout"),
        ):
            visualizer.visualize_single_benchmark(test_file)

        # Check markdown report was created
        report_file = temp_output_dir / "telemetry_report_200_tokens.md"
        assert report_file.exists()

        # Check report content
        report_content = report_file.read_text()
        assert "# Telemetry Report - 200 Token Chunks" in report_content
        assert "## Summary Metrics" in report_content

    def test_visualize_benchmark_without_telemetry(
        self, visualizer: TelemetryVisualizer, temp_output_dir: Path, capsys
    ) -> None:
        """Test handling of benchmark file without telemetry data."""
        test_data = {"config": {"leaf_tokens": 200}, "metrics": {}}
        test_file = temp_output_dir / "no_telemetry.json"

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        visualizer.visualize_single_benchmark(test_file)

        # Check warning was printed
        captured = capsys.readouterr()
        assert "No telemetry data found" in captured.out

    @patch("matplotlib.pyplot.show")
    def test_plot_methods_with_empty_data(
        self, mock_show: MagicMock, visualizer: TelemetryVisualizer
    ) -> None:
        """Test that plot methods handle empty data gracefully."""
        import matplotlib.pyplot as plt

        # Test empty batch efficiency
        fig, ax = plt.subplots()
        empty_telemetry = {"format_version": "2.0", "documents": {}}
        visualizer._plot_batch_efficiency(empty_telemetry, ax)

        # Should display "No embedding batch data available"
        texts = [t.get_text() for t in ax.texts]
        assert any("No embedding batch data available" in text for text in texts)

        plt.close(fig)

    def test_retry_patterns_no_retries(self, visualizer: TelemetryVisualizer) -> None:
        """Test retry pattern visualization with no retries."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        telemetry = {
            "format_version": "2.0",
            "documents": {
                "test": {
                    "nodes": [
                        {
                            "height": 1,
                            "summary_attempts": [
                                {"status": "accepted", "is_retry": False}
                            ],
                        }
                    ]
                }
            },
        }

        visualizer._plot_retry_patterns(telemetry, ax)

        # Should show success message
        texts = [t.get_text() for t in ax.texts]
        assert any("No Retries Needed" in text for text in texts)

        plt.close(fig)

    def test_token_distributions_with_data(
        self, visualizer: TelemetryVisualizer, sample_telemetry_data: dict
    ) -> None:
        """Test token distribution plot with actual data."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Add more nodes for better distribution
        sample_telemetry_data["documents"]["test_doc"]["nodes"].extend(
            [
                {
                    "height": 1,
                    "summary_attempts": [
                        {
                            "status": "accepted",
                            "actual_tokens": 95,
                            "target_tokens": 100,
                        }
                    ],
                },
                {
                    "height": 2,
                    "summary_attempts": [
                        {
                            "status": "accepted",
                            "actual_tokens": 102,
                            "target_tokens": 100,
                        }
                    ],
                },
            ]
        )

        visualizer._plot_token_distributions(sample_telemetry_data, ax)

        # Check that violin plot was created
        assert len(ax.collections) > 0  # Violin plot creates collections
        assert ax.get_xlabel() == "Tree Level"
        assert ax.get_ylabel() == "Token Count"

        plt.close(fig)

    def test_batch_efficiency_histogram_creation(
        self, visualizer: TelemetryVisualizer
    ) -> None:
        """Test batch efficiency histogram is created correctly."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        telemetry = {
            "format_version": "2.0",
            "documents": {
                "test": {
                    "nodes": [
                        {
                            "embedding": {
                                "text_tokens": 100,
                                "batch_size": 1,
                                "start_time": 1.0,
                            }
                        },
                        {
                            "embedding": {
                                "text_tokens": 100,
                                "batch_size": 5,
                                "start_time": 2.0,
                            }
                        },
                        {
                            "embedding": {
                                "text_tokens": 100,
                                "batch_size": 1,
                                "start_time": 3.0,
                            }
                        },
                    ]
                }
            },
        }

        visualizer._plot_batch_efficiency(telemetry, ax)

        # Check histogram was created
        assert len(ax.patches) > 0  # Histogram creates patches
        assert ax.get_xlabel() == "Embedding Batch Size"
        assert ax.get_ylabel() == "Number of Batches"

        plt.close(fig)

    def test_constants_are_used(self, visualizer: TelemetryVisualizer) -> None:
        """Test that class constants are properly defined and accessible."""
        assert visualizer.SMALL_BIN_THRESHOLD == 20
        assert visualizer.MEDIUM_BIN_THRESHOLD == 100
        assert visualizer.SMALL_BIN_WIDTH == 5
        assert visualizer.MEDIUM_BIN_WIDTH == 10
        assert visualizer.LARGE_BIN_COUNT == 20
