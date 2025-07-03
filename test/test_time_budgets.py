from itertools import product
from time import perf_counter

import numpy as np
import pytest
from pytest import raises

from kernel_tuner import tune_kernel

from .context import skip_if_no_gcc


@pytest.fixture
def env():
    kernel_name = "vector_add"
    kernel_string = """
        #include <time.h>

        float vector_add(float *c, float *a, float *b, int n) {
            struct timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            
            for (int i = 0; i < n; i++) {
                c[i] = a[i] + b[i];
            }
            
            clock_gettime(CLOCK_MONOTONIC, &end);
            double elapsed = (end.tv_sec - start.tv_sec) * 1e3 + (end.tv_nsec - start.tv_nsec) / 1e6;
            return (float) elapsed;
        }"""

    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = {"nthreads": [1, 2, 4]}

    return kernel_name, kernel_string, size, args, tune_params


@skip_if_no_gcc
def test_no_time_budget(env):
    """Ensure that a RuntimeError is raised if the startup takes longer than the time budget."""
    with raises(RuntimeError, match='startup time of the tuning process'):
        tune_kernel(*env, strategy="random_sample", strategy_options={"strategy": "random_sample", "time_limit": 0.0})

@skip_if_no_gcc
def test_some_time_budget(env):
    """Ensure that the time limit is respected."""
    time_limit = 1.0
    kernel_name, kernel_string, size, args, tune_params = env
    tune_params["bogus"] = list(range(1000))
    env = kernel_name, kernel_string, size, args, tune_params

    # Ensure that if the tuning takes longer than the time budget, the results are returned early.
    start_time = perf_counter()
    res, _ = tune_kernel(*env, strategy="random_sample", strategy_options={"time_limit": time_limit})

    # Ensure that there are at least some results, but not all.
    size_all = len(list(product(*tune_params.values())))
    assert 0 < len(res) < size_all

    # Ensure that the time limit was respected by some margin.
    assert perf_counter() - start_time < time_limit * 2

@skip_if_no_gcc
def test_full_time_budget(env):
    """Ensure that given ample time budget, the entire space is explored."""
    res, _ = tune_kernel(*env, strategy="brute_force", strategy_options={"time_limit": 10.0})

    # Ensure that the entire space is explored.
    tune_params = env[-1]
    size_all = len(list(product(*tune_params.values())))
    assert len(res) == size_all
