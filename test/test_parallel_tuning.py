import numpy as np
import pytest

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
    tune_params = dict()
    tune_params["block_size_x"] = [128 + 64 * i for i in range(15)]

    return ["vector_add", kernel_string, size, args, tune_params]

@skip_if_no_pycuda
def test_parallel_tune_kernel(env):
    result, _ = tune_kernel(*env, lang="CUDA", verbose=True, remote_mode=True)
    assert len(result) > 0