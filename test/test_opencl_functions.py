import numpy as np
import pytest

import kernel_tuner
from kernel_tuner.backends import opencl
from kernel_tuner.core import KernelInstance, KernelSource

from .context import skip_if_no_opencl

try:
    import pyopencl
except Exception:
    pass


@skip_if_no_opencl
def test_ready_argument_list():

    size = 1000
    a = np.int32(75)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)

    arguments = [c, a, b]

    dev = opencl.OpenCLFunctions(0)
    gpu_args = dev.ready_argument_list(arguments)

    assert isinstance(gpu_args[0], pyopencl.Buffer)
    assert isinstance(gpu_args[1], np.int32)
    assert isinstance(gpu_args[2], pyopencl.Buffer)

    gpu_args[0].release()
    gpu_args[2].release()


@skip_if_no_opencl
def test_compile():

    original_kernel = """
    __kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g) {
        int gid = get_global_id(0);
        __local float test[shared_size];
        test[0] = a_g[gid];
        res_g[gid] = test[0] + b_g[gid];
    }
    """

    kernel_sources = KernelSource("sum", original_kernel, "opencl")
    kernel_string = original_kernel.replace("shared_size", str(1024))
    kernel_instance = KernelInstance("sum", kernel_sources, kernel_string, [], None, None, dict(), [])

    dev = opencl.OpenCLFunctions(0)
    func = dev.compile(kernel_instance)

    assert isinstance(func, pyopencl.Kernel)


@skip_if_no_opencl
def test_run_kernel():

    threads = (1, 2, 3)
    grid = (4, 5, 1)

    def test_func(queue, global_size, local_size, arg):
        assert all(global_size == np.array([4, 10, 3]))
        return type('Event', (object,), {'wait': lambda self: 0})()
    dev = opencl.OpenCLFunctions(0)
    dev.run_kernel(test_func, [0], threads, grid)


@pytest.fixture
def env():
    kernel_string = """
        __kernel void vector_add(__global float *c, __global const float *a, __global const float *b, int n) {
            int i = get_global_id(0);
            if (i<n) {
                c[i] = a[i] + b[i];
            }
        }"""

    size = 100
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    c = np.zeros_like(b)
    n = np.int32(size)

    args = [c, a, b, n]
    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128]

    return ["vector_add", kernel_string, size, args, tune_params]


@skip_if_no_opencl
def test_benchmark(env):
    results, _ = kernel_tuner.tune_kernel(*env)
    assert len(results) == 3
    assert all(["block_size_x" in result for result in results])
    assert all(["time" in result for result in results])
    assert all([result["time"] > 0.0 for result in results])
