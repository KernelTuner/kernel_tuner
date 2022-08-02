import numpy as np
from .context import skip_if_no_cuda
from .test_runners import env

import pytest
from kernel_tuner import nvcuda, tune_kernel
from kernel_tuner.core import KernelSource, KernelInstance

try:
    from cuda import cuda
except Exception:
    pass


@skip_if_no_cuda
def test_ready_argument_list():

    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)

    arguments = [c, a, b]

    dev = nvcuda.CudaFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    assert isinstance(gpu_args[0], cuda.CUdeviceptr)
    assert isinstance(gpu_args[1], np.int32)
    assert isinstance(gpu_args[2], cuda.CUdeviceptr)


@skip_if_no_cuda
def test_compile():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    kernel_name = "vector_add"
    kernel_sources = KernelSource(kernel_name, kernel_string, "cuda")
    kernel_instance = KernelInstance(kernel_name, kernel_sources, kernel_string, [], None, None, dict(), [])
    dev = nvcuda.CudaFunctions(0)
    try:
        dev.compile(kernel_instance)
    except Exception as e:
        pytest.fail("Did not expect any exception:" + str(e))


@skip_if_no_cuda
def test_run_kernel():

    threads = (1, 2, 3)
    grid = (4, 5, 1)

    def test_func(queue, global_size, local_size, arg):
        assert all(global_size == np.array([4, 10, 3]))
        return type('Event', (object,), {'wait': lambda self: 0})()
    dev = nvcuda.CudaFunctions(0)
    dev.run_kernel(dummy_func, [0], threads, grid)


@skip_if_no_cuda
def test_tune_kernel(env):
    result, _ = tune_kernel(*env, lang="nvcuda", verbose=True)
    assert len(result) > 0


def dummy_func(a, b, block=0, grid=0, stream=None, shared=0, texrefs=None):
    pass


