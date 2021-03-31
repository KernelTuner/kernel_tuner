import numpy as np
from .context import skip_if_no_opencl

from kernel_tuner import opencl
from kernel_tuner.core import KernelSource, KernelInstance

import pytest

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

    with opencl.OpenCLFunctions(0) as dev:
        func = dev.compile(kernel_instance)

        assert isinstance(func, pyopencl.Kernel)


def fun_test(queue, a, b, block=0, grid=0):
    profile = type('profile', (object,), {'end': 0.1, 'start': 0})
    return type(
        'Event', (object,), {'wait': lambda self: 0, 'profile': profile(), 'get_info': lambda x,y: 0})()


@pytest.fixture
def create_benchmark_args():
    with opencl.OpenCLFunctions(0) as dev:
        args = [1, 2]
        times = tuple(range(1, 4))

        yield dev, args, times


@skip_if_no_opencl
def test_benchmark(create_benchmark_args):
    dev, args, times = create_benchmark_args
    res = dev.benchmark(fun_test, args, times, times)
    assert res["time"] > 0
    assert len(res["times"]) == dev.iterations


@skip_if_no_opencl
def test_run_kernel():

    threads = (1, 2, 3)
    grid = (4, 5, 1)

    def test_func(queue, global_size, local_size, arg):
        assert all(global_size == np.array([4, 10, 3]))
        return type('Event', (object,), {'wait': lambda self: 0})()
    with opencl.OpenCLFunctions(0) as dev:
        dev.run_kernel(test_func, [0], threads, grid)
