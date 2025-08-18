"""Performance benchmark for telemetry visualization axis synchronization optimization."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Skip all tests in this module if matplotlib is not available
plt = pytest.importorskip("matplotlib.pyplot")
pytest.importorskip("seaborn")

from ragzoom.telemetry_viz import TelemetryVisualizer  # noqa: E402


def generate_test_telemetry(num_nodes: int) -> dict:
    """Generate test telemetry data with specified number of nodes."""
    nodes = []
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
                            "status": "accepted",
                            "is_retry": False,
                            "prompt_tokens": 400 + i,
                            "completion_tokens": 100 + i % 50,
                            "input_text_tokens": 195,
                            "actual_tokens": 98 + i % 10,
                            "target_tokens": 100,
                        }
                    ],
                }
            )

    return {
        "format_version": "3.0",
        "document_id": "test_doc",
        "source_document_tokens": num_nodes * 100,
        "indexed_at": 1234567890.0,
        "config": {
            "target_chunk_tokens": 200,
            "summary_model": "gpt-4o-mini",
            "embedding_model": "text-embedding-3-small",
        },
        "nodes": nodes,
    }


class TestVisualizationPerformance:
    """Test performance improvements in axis synchronization."""

    @pytest.fixture
    def temp_files(self):
        """Create temporary telemetry files for testing."""
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

    def test_axis_synchronization_performance(self, temp_files):
        """Benchmark the performance of axis synchronization with the new implementation."""
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

    def test_axis_operation_count(self):
        """Verify that the optimized implementation reduces axis operations."""
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

            def counting_set_xlim(self, *args, **kwargs):
                nonlocal set_lim_count
                set_lim_count += 1
                return original_set_xlim(self, *args, **kwargs)

            def counting_set_ylim(self, *args, **kwargs):
                nonlocal set_lim_count
                set_lim_count += 1
                return original_set_ylim(self, *args, **kwargs)

            def counting_get_xlim(self, *args, **kwargs):
                nonlocal get_lim_count
                get_lim_count += 1
                return original_get_xlim(self, *args, **kwargs)

            def counting_get_ylim(self, *args, **kwargs):
                nonlocal get_lim_count
                get_lim_count += 1
                return original_get_ylim(self, *args, **kwargs)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
