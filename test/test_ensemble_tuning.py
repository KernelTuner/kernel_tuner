import numpy as np
import pytest
import logging
import sys

from kernel_tuner import tune_kernel
from kernel_tuner.backends import nvcuda
from kernel_tuner.core import KernelInstance, KernelSource
from .context import skip_if_no_pycuda

try:
    import pycuda.driver
except Exception:
    pass


@pytest.fixture
def env():
    kernel_string = """
    extern "C" __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        int index = i + j * gridDim.x * blockDim.x;
        if (index < n) {
            c[index] = a[index] + b[index];
        }
    }
    """

    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = dict()

    # Extend the range of block sizes for a bigger search space
    tune_params["block_size_x"] = [128 + 64 * i for i in range(30)]
    tune_params["block_size_y"] = [1 + i for i in range(1, 16)]

    return ["vector_add", kernel_string, size, args, tune_params]


@skip_if_no_pycuda
def test_parallel_tune_kernel(env):
    strategy_options = {"ensemble": ["greedy_ils", "greedy_ils"]}
    result, _ = tune_kernel(
        *env, lang="CUDA", verbose=True, strategy="ensemble", parallel_mode=True, strategy_options=strategy_options
    )
    assert len(result) > 0
