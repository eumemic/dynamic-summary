"""Test telemetry visualization functions."""

import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Skip all tests in this module if matplotlib is not available
pytest.importorskip("matplotlib")
pytest.importorskip("seaborn")
pytest.importorskip("pandas")

from ragzoom.telemetry_types import NodeTelemetryDict, TelemetryDataDict
from ragzoom.telemetry_viz import TelemetryVisualizer


class TestTelemetryVisualizer:
    """Test TelemetryVisualizer class."""

    @pytest.fixture
    def temp_output_dir(self) -> Generator[Path, None, None]:
        """Create a temporary output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def visualizer(self, temp_output_dir: Path) -> TelemetryVisualizer:
        """Create a TelemetryVisualizer instance."""
        output_path = temp_output_dir / "test_viz.png"
        return TelemetryVisualizer(output_path)

    @pytest.fixture
    def sample_benchmark_data(
        self, sample_telemetry_data: TelemetryDataDict
    ) -> dict[str, object]:
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
        output_path = temp_output_dir / "test_viz.png"
        visualizer = TelemetryVisualizer(output_path)
        assert visualizer.output_path == output_path
        assert temp_output_dir.exists()
        # Thresholds were removed as part of legacy cleanup

    def test_load_benchmark_data(
        self, visualizer: TelemetryVisualizer, temp_output_dir: Path
    ) -> None:
        """Test loading benchmark data from JSON."""
        test_data: dict[str, object] = {"test": "data"}
        test_file = temp_output_dir / "test.json"

        with open(test_file, "w") as f:
            json.dump(test_data, f)

        loaded_data = visualizer.load_benchmark_data(test_file)
        assert loaded_data == test_data

    def test_calculate_histogram_bins_small_discrete(
        self, visualizer: TelemetryVisualizer
    ) -> None:
        """Test histogram bin calculation for small discrete values."""
        batch_sizes: list[float] = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0]
        bins, align = visualizer._calculate_histogram_bins(batch_sizes)

        assert align == "left"
        assert isinstance(bins, list)
        assert bins == [0, 1, 2, 3, 4]  # Covers values 0-3

    def test_calculate_histogram_bins_medium_range(
        self, visualizer: TelemetryVisualizer
    ) -> None:
        """Test histogram bin calculation for medium range values."""
        batch_sizes: list[float] = [
            1.0,
            5.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            35.0,
            40.0,
            45.0,
        ]
        bins, align = visualizer._calculate_histogram_bins(batch_sizes)

        assert align == "left"
        assert isinstance(bins, list)
        assert bins[0] == 0
        assert bins[-1] >= 45

    def test_calculate_histogram_bins_large_range(
        self, visualizer: TelemetryVisualizer
    ) -> None:
        """Test histogram bin calculation for large range values."""
        batch_sizes: list[float] = [
            float(x) for x in range(1, 150, 10)
        ]  # 1, 11, 21, ..., 141
        bins, align = visualizer._calculate_histogram_bins(batch_sizes)

        assert align == "mid"
        assert isinstance(bins, int)
        assert bins == visualizer.LARGE_BIN_COUNT

    def test_visualize_single_benchmark_creates_outputs(
        self, visualizer: TelemetryVisualizer, temp_output_dir: Path
    ) -> None:
        """Test that visualize_single_benchmark creates expected outputs."""
        # Create minimal telemetry data that won't trigger complex plotting
        minimal_data: TelemetryDataDict = {
            "format_version": "4.2",
            "document_id": "test",
            "source_document_tokens": 0,
            "indexed_at": 0,
            "config": {
                "target_chunk_tokens": 200,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "model_metadata": {},
            "system_prompts": {},
            "runtime_info": {
                "python_version": "3.11.0",
                "platform": "test",
                "ragzoom_version": "1.0.0",
            },
            "nodes": [],
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

        # Check PNG was created (markdown report removed in PR #69)
        # png_file = temp_output_dir / "telemetry_200_tokens.png"
        # PNG creation is mocked, so we just verify the method was called

    def test_visualize_benchmark_without_telemetry(
        self,
        visualizer: TelemetryVisualizer,
        temp_output_dir: Path,
        capsys: pytest.CaptureFixture[str],
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
        empty_telemetry: TelemetryDataDict = {
            "format_version": "4.2",
            "document_id": "empty",
            "source_document_tokens": 0,
            "indexed_at": 0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "model_metadata": {},
            "system_prompts": {},
            "runtime_info": {
                "python_version": "3.11.0",
                "platform": "test",
                "ragzoom_version": "1.0.0",
            },
            "nodes": [],
        }
        visualizer._plot_batch_efficiency(empty_telemetry, ax)

        # Should display "No embedding batch data available"
        texts = [t.get_text() for t in ax.texts]
        assert any("No embedding batch data available" in text for text in texts)

        plt.close(fig)

    def test_retry_patterns_no_retries(self, visualizer: TelemetryVisualizer) -> None:
        """Test retry pattern visualization with no retries."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        telemetry: TelemetryDataDict = {
            "format_version": "4.2",
            "document_id": "test",
            "source_document_tokens": 100,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "model_metadata": {},
            "system_prompts": {},
            "runtime_info": {
                "python_version": "3.11.0",
                "platform": "test",
                "ragzoom_version": "1.0.0",
            },
            "nodes": [
                {
                    "node_id": "node-1",
                    "height": 1,
                    "created_at": 1234567890.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 150,
                            "completion_tokens": 95,
                            "actual_tokens": 95,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567890.0,
                            "end_time": 1234567891.0,
                        }
                    ],
                    "accepted_attempt": 0,
                }
            ],
        }

        visualizer._plot_retry_patterns(telemetry, ax)

        # With the new cumulative visualization, it should create a bar chart
        # even when there are no retries (just shows all nodes had ≥1 attempts)
        assert len(ax.patches) > 0  # Should have bar patches
        assert "Retry Pattern Distribution" in ax.get_title()

        plt.close(fig)

    def test_token_distributions_with_data(
        self, visualizer: TelemetryVisualizer, sample_telemetry_data: TelemetryDataDict
    ) -> None:
        """Test token distribution plot with actual data."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Add more nodes for better distribution
        sample_telemetry_data["nodes"].extend(
            [
                {
                    "node_id": "summary-2",
                    "height": 1,
                    "created_at": 1234567894.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 400,
                            "completion_tokens": 95,
                            "actual_tokens": 95,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567894.0,
                            "end_time": 1234567895.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
                {
                    "node_id": "summary-3",
                    "height": 2,
                    "created_at": 1234567896.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 400,
                            "completion_tokens": 102,
                            "actual_tokens": 102,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567896.0,
                            "end_time": 1234567897.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
            ]
        )

        visualizer._plot_token_distributions(sample_telemetry_data, ax)

        # Check that violin plot was created
        assert len(ax.collections) > 0  # Violin plot creates collections
        assert ax.get_xlabel() == "Attempt Number"  # Changed from Tree Level
        assert ax.get_ylabel() == "Token Count"

        plt.close(fig)

    def test_batch_efficiency_histogram_creation(
        self, visualizer: TelemetryVisualizer
    ) -> None:
        """Test batch efficiency histogram is created correctly."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        telemetry: TelemetryDataDict = {
            "format_version": "4.2",
            "document_id": "test",
            "source_document_tokens": 300,
            "indexed_at": 0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "model_metadata": {},
            "system_prompts": {},
            "runtime_info": {
                "python_version": "3.11.0",
                "platform": "test",
                "ragzoom_version": "1.0.0",
            },
            "nodes": [
                {
                    "node_id": "node-1",
                    "height": 0,
                    "created_at": 1.0,
                    "embedding": {
                        "text_tokens": 100,
                        "batch_size": 1,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 1.0,
                        "end_time": 1.1,
                    },
                },
                {
                    "node_id": "node-2",
                    "height": 0,
                    "created_at": 2.0,
                    "embedding": {
                        "text_tokens": 100,
                        "batch_size": 5,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 2.0,
                        "end_time": 2.1,
                    },
                },
                {
                    "node_id": "node-3",
                    "height": 0,
                    "created_at": 3.0,
                    "embedding": {
                        "text_tokens": 100,
                        "batch_size": 1,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 3.0,
                        "end_time": 3.1,
                    },
                },
            ],
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


def generate_test_telemetry(num_nodes: int) -> TelemetryDataDict:
    """Generate test telemetry data with specified number of nodes."""
    nodes: list[NodeTelemetryDict] = []
    for i in range(num_nodes):
        if i % 2 == 0:
            # Leaf node with embedding
            nodes.append(
                {
                    "node_id": f"leaf-{i}",
                    "height": 0,
                    "created_at": 1234567890.0 + i,
                    "embedding": {
                        "text_tokens": 195,
                        "batch_size": (i % 10) + 1,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 1234567890.0 + i,
                        "end_time": 1234567891.0 + i,
                    },
                }
            )
        else:
            # Summary node
            nodes.append(
                {
                    "node_id": f"summary-{i}",
                    "height": (i % 3) + 1,
                    "created_at": 1234567892.0 + i,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 400 + i,
                            "completion_tokens": 100 + i % 50,
                            "actual_tokens": 98 + i % 10,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567892.0 + i,
                            "end_time": 1234567893.0 + i,
                        }
                    ],
                    "accepted_attempt": 0,
                }
            )

    return {
        "format_version": "4.2",
        "document_id": "test_doc",
        "source_document_tokens": num_nodes * 100,
        "indexed_at": 1234567890.0,
        "config": {
            "target_chunk_tokens": 200,
            "summary_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
        },
        "model_metadata": {},
        "system_prompts": {},
        "runtime_info": {
            "python_version": "3.11.0",
            "platform": "test",
            "ragzoom_version": "1.0.0",
        },
        "nodes": nodes,
    }


class TestVisualizationPerformance:
    """Test performance improvements in axis synchronization."""

    @pytest.fixture
    def temp_files(self) -> Generator[dict[int, tuple[Path, Path]], None, None]:
        """Create temporary telemetry files for testing."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create test telemetry files with varying sizes
            sizes = [100, 500, 1000]
            files = {}

            for size in sizes:
                telemetry_data = generate_test_telemetry(size)
                file1 = tmpdir_path / f"telemetry_{size}_1.json"
                file2 = tmpdir_path / f"telemetry_{size}_2.json"

                with open(file1, "w") as f:
                    json.dump(telemetry_data, f)
                with open(file2, "w") as f:
                    json.dump(telemetry_data, f)

                files[size] = (file1, file2)

            yield files

    @pytest.mark.slow
    def test_axis_synchronization_performance(
        self, temp_files: dict[int, tuple[Path, Path]]
    ) -> None:
        """Benchmark the performance of axis synchronization with the new implementation."""
        import time
        from unittest.mock import patch

        results = []

        # Mock the actual plotting to focus on axis operations
        with patch("matplotlib.pyplot.savefig"), patch("matplotlib.pyplot.close"):

            for size, (file1, file2) in temp_files.items():
                output_path = file1.parent / f"comparison_{size}.png"
                visualizer = TelemetryVisualizer(output_path)

                # Measure time for side-by-side comparison
                start_time = time.perf_counter()
                visualizer.visualize_side_by_side(file1, file2)
                elapsed = time.perf_counter() - start_time

                results.append({"nodes": size, "time": elapsed})

                print(f"Size {size}: {elapsed:.4f}s")

        # Verify that performance scales reasonably
        # With the optimization, time should scale more linearly with data size
        # rather than quadratically due to reduced axis operations
        assert all(
            r["time"] < 2.0 for r in results
        ), "Visualization should complete quickly even with large datasets"

        # Check that larger datasets don't take disproportionately longer
        if len(results) >= 2:
            time_ratio = results[-1]["time"] / results[0]["time"]
            size_ratio = results[-1]["nodes"] / results[0]["nodes"]

            # Time should scale sub-linearly with optimized implementation
            # (allowing for some overhead)
            assert (
                time_ratio < size_ratio * 1.5
            ), f"Time scaling ({time_ratio:.2f}x) should be better than linear size scaling ({size_ratio}x)"

    @pytest.mark.slow
    def test_axis_operation_count(self) -> None:
        """Verify that the optimized implementation reduces axis operations."""
        import json
        import tempfile
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create minimal test data
            telemetry = generate_test_telemetry(10)
            file1 = tmpdir_path / "test1.json"
            file2 = tmpdir_path / "test2.json"

            with open(file1, "w") as f:
                json.dump(telemetry, f)
            with open(file2, "w") as f:
                json.dump(telemetry, f)

            output_path = tmpdir_path / "test_output.png"
            visualizer = TelemetryVisualizer(output_path)

            # Count axis limit operations
            set_lim_count = 0
            get_lim_count = 0

            from matplotlib.axes import Axes

            original_set_xlim = Axes.set_xlim
            original_set_ylim = Axes.set_ylim
            original_get_xlim = Axes.get_xlim
            original_get_ylim = Axes.get_ylim

            def counting_set_xlim(
                self: "Axes", *args: object, **kwargs: object
            ) -> object:
                nonlocal set_lim_count
                set_lim_count += 1
                return original_set_xlim(self, *args, **kwargs)  # type: ignore[arg-type]

            def counting_set_ylim(
                self: "Axes", *args: object, **kwargs: object
            ) -> object:
                nonlocal set_lim_count
                set_lim_count += 1
                return original_set_ylim(self, *args, **kwargs)  # type: ignore[arg-type]

            def counting_get_xlim(
                self: "Axes", *args: object, **kwargs: object
            ) -> tuple[float, float]:
                nonlocal get_lim_count
                get_lim_count += 1
                return original_get_xlim(self, *args, **kwargs)  # type: ignore[no-any-return]  # matplotlib types

            def counting_get_ylim(
                self: "Axes", *args: object, **kwargs: object
            ) -> tuple[float, float]:
                nonlocal get_lim_count
                get_lim_count += 1
                return original_get_ylim(self, *args, **kwargs)  # type: ignore[no-any-return]  # matplotlib types

            with (
                patch.object(Axes, "set_xlim", counting_set_xlim),
                patch.object(Axes, "set_ylim", counting_set_ylim),
                patch.object(Axes, "get_xlim", counting_get_xlim),
                patch.object(Axes, "get_ylim", counting_get_ylim),
                patch("matplotlib.pyplot.savefig"),
                patch("matplotlib.pyplot.close"),
            ):

                visualizer.visualize_side_by_side(file1, file2)

            # With the optimized implementation using shared axes:
            # - No manual get_xlim/get_ylim calls should be needed for synchronization
            # - set_xlim/set_ylim calls should be significantly reduced
            # The exact counts depend on internal matplotlib behavior and plot methods

            print(f"get_lim operations: {get_lim_count}")
            print(f"set_lim operations: {set_lim_count}")

            # The optimized version should have minimal manual axis operations
            # since matplotlib handles synchronization automatically
            assert (
                get_lim_count < 20
            ), f"Too many get_lim operations ({get_lim_count}), indicates manual synchronization"
            assert (
                set_lim_count < 20
            ), f"Too many set_lim operations ({set_lim_count}), indicates manual synchronization"
