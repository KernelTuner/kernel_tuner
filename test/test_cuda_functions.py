import numpy as np
from .context import skip_if_no_cuda

import pytest
from kernel_tuner import cuda
from kernel_tuner.core import KernelSource, KernelInstance

try:
    import pycuda.driver
except Exception:
    pass


@skip_if_no_cuda
def test_ready_argument_list():

    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)

    arguments = [c, a, b]

    with cuda.CudaFunctions(0) as dev:
        gpu_args = dev.ready_argument_list(arguments)

        assert isinstance(gpu_args[0], pycuda.driver.DeviceAllocation)
        assert isinstance(gpu_args[1], np.int32)
        assert isinstance(gpu_args[2], pycuda.driver.DeviceAllocation)


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
    with cuda.CudaFunctions(0) as dev:
        try:
            dev.compile(kernel_instance)
        except Exception as e:
            pytest.fail("Did not expect any exception:" + str(e))


def dummy_func(a, b, block=0, grid=0, stream=None, shared=0, texrefs=None):
    pass


@skip_if_no_cuda
def test_benchmark():
    with cuda.CudaFunctions(0) as dev:
        args = [1, 2]
        res = dev.benchmark(dummy_func, args, (1, 2), (1, 2))
        assert res["time"] > 0
        assert len(res["times"]) == dev.iterations
