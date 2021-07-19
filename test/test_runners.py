from __future__ import print_function

import os
from collections import OrderedDict

import numpy as np
import pytest

import kernel_tuner
from kernel_tuner import util

from .context import skip_if_no_cuda

cache_filename = os.path.dirname(os.path.realpath(__file__)) + "/test_cache_file.json"


@pytest.fixture
def env():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [128 + 64 * i for i in range(15)]

    return ["vector_add", kernel_string, size, args, tune_params]


@skip_if_no_cuda
def test_sequential_runner_alt_block_size_names(env):

    kernel_string = """__global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_dim_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    tune_params = {
        "block_dim_x": [128 + 64 * i for i in range(5)],
        "block_size_y": [1],
        "block_size_z": [1]
    }

    env[1] = kernel_string
    env[-1] = tune_params

    ref = (env[3][1] + env[3][2]).astype(np.float32)
    answer = [ref, None, None, None]

    block_size_names = ["block_dim_x"]

    result, _ = kernel_tuner.tune_kernel(*env, grid_div_x=["block_dim_x"], answer=answer, block_size_names=block_size_names)

    assert len(result) == len(tune_params["block_dim_x"])


@skip_if_no_cuda
def test_smem_args(env):
    result, _ = kernel_tuner.tune_kernel(*env, smem_args=dict(size="block_size_x*4"), verbose=True)
    tune_params = env[-1]
    assert len(result) == len(tune_params["block_size_x"])
    result, _ = kernel_tuner.tune_kernel(*env, smem_args=dict(size=lambda p: p['block_size_x'] * 4), verbose=True)
    tune_params = env[-1]
    assert len(result) == len(tune_params["block_size_x"])


def test_simulation_runner(env):
    result, _ = kernel_tuner.tune_kernel(*env, cache=cache_filename, simulation_mode=True, verbose=True)
    tune_params = env[-1]
    assert len(result) == len(tune_params["block_size_x"])


def test_diff_evo(env):
    result, _ = kernel_tuner.tune_kernel(*env, strategy="diff_evo", verbose=True, cache=cache_filename, simulation_mode=True)
    assert len(result) > 0


def test_genetic_algorithm(env):
    options = dict(method="uniform", popsize=10, maxiter=2, mutation_change=1)
    result, _ = kernel_tuner.tune_kernel(*env, strategy="genetic_algorithm", strategy_options=options, verbose=True, cache=cache_filename, simulation_mode=True)
    assert len(result) > 0


def test_bayesian_optimization(env):
    for method in ["poi", "ei", "lcb", "lcb-srinivas", "multi", "multi-advanced", "multi-fast"]:
        print(method, flush=True)
        options = dict(popsize=5, max_fevals=10, method=method)
        result, _ = kernel_tuner.tune_kernel(*env, strategy="bayes_opt", strategy_options=options, verbose=True, cache=cache_filename, simulation_mode=True)
        assert len(result) > 0


def test_random_sample(env):
    result, _ = kernel_tuner.tune_kernel(*env, strategy="random_sample", strategy_options={ "fraction": 0.1 }, cache=cache_filename, simulation_mode=True)
    # check that number of benchmarked kernels is 10% (rounded up)
    assert len(result) == 2
    # check all returned results make sense
    for v in result:
        assert v['time'] > 0.0 and v['time'] < 1.0
