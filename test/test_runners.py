from __future__ import print_function

import numpy as np
import pytest

import kernel_tuner
from kernel_tuner import core
from kernel_tuner.interface import Options

from .context import skip_if_no_cuda

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
    tune_params = {"block_size_x": [128+64*i for i in range(15)]}

    return ["vector_add", kernel_string, size, args, tune_params]


def test_random_sample():

    kernel_string = "float test_kernel(float *a) { return 1.0f; }"
    a = np.arange(4, dtype=np.float32)

    tune_params = {"block_size_x": range(1, 25)}
    print(tune_params)

    result, _ = kernel_tuner.tune_kernel(
        "test_kernel", kernel_string, (1, 1), [a], tune_params,
        strategy="random_sample", strategy_options={"fraction": 0.1})

    print(result)

    # check that number of benchmarked kernels is 10% (rounded up)
    assert len(result) == 3

    # check all returned results make sense
    for v in result:
        assert v['time'] == 1.0


@skip_if_no_cuda
def test_diff_evo(env):
    result, _ = kernel_tuner.tune_kernel(*env, strategy="diff_evo", verbose=True)
    assert len(result) > 0

@skip_if_no_cuda
def test_genetic_algorithm(env):
    options = dict(method="uniform", popsize=10, maxiter=2, mutation_change=1)
    result, _ = kernel_tuner.tune_kernel(*env, strategy="genetic_algorithm", strategy_options=options, verbose=True)
    assert len(result) > 0


@skip_if_no_cuda
def test_sequential_runner_alt_block_size_names(env):

    kernel_string = """__global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_dim_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    tune_params = {"block_dim_x": [128 + 64 * i for i in range(5)],
                   "block_size_y": [1], "block_size_z": [1]}

    env[1] = kernel_string
    env[-1] = tune_params

    ref = (env[3][1]+env[3][2]).astype(np.float32)
    answer = [ref, None, None, None]

    block_size_names = ["block_dim_x"]

    result, _ = kernel_tuner.tune_kernel(*env,
                                         grid_div_x=["block_dim_x"], answer=answer,
                                         block_size_names=block_size_names)

    assert len(result) == len(tune_params["block_dim_x"])

