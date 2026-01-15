"""Regression tests for Kernel Tuner measurement accuracy.

These tests verify that Kernel Tuner produces consistent results
over time by comparing against known-good baseline data.

Issue: https://github.com/KernelTuner/kernel_tuner/issues/99
"""
import json
from pathlib import Path

import numpy as np
import pytest

from kernel_tuner import tune_kernel

# Directory containing baseline JSON files
BASELINES_DIR = Path(__file__).parent / "baselines"


def load_baseline(filename: str) -> dict:
    """Load a baseline JSON file.

    Args:
        filename: Name of the baseline file in the baselines directory

    Returns:
        Dictionary containing the baseline data
    """
    filepath = BASELINES_DIR / filename
    with open(filepath, "r") as f:
        return json.load(f)


def compare_timing_results(actual: list, expected: dict, tolerance: float = 0.10):
    """Compare actual tuning results against expected baseline.

    Args:
        actual: List of result dictionaries from tune_kernel()
        expected: Baseline cache dictionary with 'cache' key
        tolerance: Allowed relative difference (default 10%)

    Returns:
        List of (config_key, expected_time, actual_time, diff_pct) for failures
    """
    failures = []

    for result in actual:
        # Build the cache key from result parameters
        config_key = str(result.get("block_size_x", ""))

        if config_key not in expected["cache"]:
            continue

        expected_time = expected["cache"][config_key]["time"]
        actual_time = result["time"]

        # Skip error results (non-numeric times indicate compilation/runtime failures)
        if not isinstance(actual_time, (int, float)):
            continue

        # Calculate relative difference
        if expected_time > 0:
            diff_pct = abs(actual_time - expected_time) / expected_time
            if diff_pct > tolerance:
                failures.append((config_key, expected_time, actual_time, diff_pct))

    return failures


@pytest.fixture
def vector_add_env():
    """Standard vector_add kernel environment for regression tests."""
    kernel_string = '''
    extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    '''
    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = {"block_size_x": [128 + 64 * i for i in range(15)]}

    return ["vector_add", kernel_string, size, args, tune_params]


class TestRegressionSimulation:
    """Regression tests using simulation mode (no GPU required).

    These tests verify the consistency of Kernel Tuner's internal
    processing by using cached baseline data in simulation mode.
    """

    def test_vector_add_rtx_a4000_simulation(self, vector_add_env):
        """Test vector_add results consistency using simulation mode.

        This test loads baseline data from an RTX A4000 and runs tuning
        in simulation mode to verify that Kernel Tuner processes and
        returns the cached data correctly.
        """
        baseline_file = "vector_add_NVIDIA_RTX_A4000.json"
        baseline = load_baseline(baseline_file)
        baseline_path = BASELINES_DIR / baseline_file

        # Run tuning in simulation mode using baseline as cache
        results, env = tune_kernel(
            *vector_add_env,
            cache=str(baseline_path),
            simulation_mode=True,
            verbose=False
        )

        # Verify we got results for all configurations
        assert len(results) == len(vector_add_env[-1]["block_size_x"])

        # Verify result structure
        for result in results:
            assert "time" in result
            assert "times" in result
            assert "block_size_x" in result
            assert isinstance(result["time"], (int, float))
            assert result["time"] > 0

        # In simulation mode, times should match exactly (very tight tolerance)
        failures = compare_timing_results(results, baseline, tolerance=0.001)
        assert len(failures) == 0, f"Timing mismatches in simulation: {failures}"

    def test_baseline_file_integrity(self):
        """Verify baseline files have required structure.

        This test checks that baseline JSON files contain all the
        required fields and have valid data types.
        """
        baseline_file = "vector_add_NVIDIA_RTX_A4000.json"
        baseline = load_baseline(baseline_file)

        # Check required top-level keys
        required_keys = ["device_name", "kernel_name", "tune_params_keys",
                         "tune_params", "cache"]
        for key in required_keys:
            assert key in baseline, f"Missing required key: {key}"

        # Verify metadata
        assert baseline["device_name"] == "NVIDIA RTX A4000"
        assert baseline["kernel_name"] == "vector_add"

        # Check cache entries have required fields
        for config_key, entry in baseline["cache"].items():
            assert "time" in entry, f"Missing 'time' in config {config_key}"
            assert "times" in entry, f"Missing 'times' in config {config_key}"
            assert isinstance(entry["time"], (int, float)), \
                f"Invalid time type in config {config_key}"
            assert isinstance(entry["times"], list), \
                f"Invalid times type in config {config_key}"
            assert entry["time"] > 0, f"Time must be positive in config {config_key}"

    def test_baseline_config_coverage(self, vector_add_env):
        """Verify baseline covers all expected configurations.

        This test ensures the baseline file contains data for all
        the block sizes we're testing.
        """
        baseline_file = "vector_add_NVIDIA_RTX_A4000.json"
        baseline = load_baseline(baseline_file)

        expected_block_sizes = vector_add_env[-1]["block_size_x"]

        for block_size in expected_block_sizes:
            config_key = str(block_size)
            assert config_key in baseline["cache"], \
                f"Missing baseline data for block_size_x={block_size}"

    def test_timing_values_reasonable(self):
        """Verify baseline timing values are in reasonable range.

        Sanity check that the baseline times are within expected
        bounds for a simple vector_add kernel.
        """
        baseline_file = "vector_add_NVIDIA_RTX_A4000.json"
        baseline = load_baseline(baseline_file)

        for config_key, entry in baseline["cache"].items():
            time_ms = entry["time"]
            # For a simple 100-element vector_add, times should be < 1 second
            assert time_ms < 1000, \
                f"Unreasonably high time {time_ms}ms for config {config_key}"
            # Times should be positive and not essentially zero
            assert time_ms > 0.001, \
                f"Suspiciously low time {time_ms}ms for config {config_key}"
