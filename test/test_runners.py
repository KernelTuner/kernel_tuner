from __future__ import print_function

import numpy
import sys
from nose import SkipTest

import kernel_tuner
from .context import skip_if_no_cuda_device


def test_random_sample():

    kernel_string = "float test_kernel(float *a) { return 1.0f; }"
    a = numpy.array([1,2,3]).astype(numpy.float32)

    tune_params = {"block_size_x": range(1,25)}
    print(tune_params)

    result, _ = kernel_tuner.tune_kernel("test_kernel", kernel_string, (1,1), [a], tune_params, sample_fraction=0.1)

    print(result)

    #check that number of benchmarked kernels is 10% (rounded up)
    assert len(result) == 3

    #check all returned results make sense
    for v in result:
        assert v['time'] == 1.0


def test_noodles_runner():

    skip_if_no_cuda_device()

    if sys.version_info[0] < 3 or (sys.version_info[0] == 3 and sys.version_info[1] < 5):
        raise SkipTest("Noodles runner test requires Python 3.5 or newer")

    import importlib.util
    noodles_installed = importlib.util.find_spec("noodles") is not None

    if not noodles_installed:
        raise SkipTest("Noodles runner test requires Noodles")

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 100
    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]
    tune_params = {"block_size_x": [128+64*i for i in range(15)]}

    result, _ = kernel_tuner.tune_kernel("vector_add", kernel_string, size, args, tune_params,
                            use_noodles=True, num_threads=4)

    assert len(result) == len(tune_params["block_size_x"])




def test_sequential_runner_alt_block_size_names():

    skip_if_no_cuda_device()

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_dim_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 100
    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]
    tune_params = {"block_dim_x": [128+64*i for i in range(15)]}

    answer = [a+b, None, None, None]

    result, _ = kernel_tuner.tune_kernel("vector_add", kernel_string, size, args, tune_params,
                            answer=answer)

    assert len(result) == len(tune_params["block_dim_x"])
