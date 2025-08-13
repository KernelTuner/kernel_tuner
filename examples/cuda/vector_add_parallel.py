#!/usr/bin/env python

import numpy
from kernel_tuner import tune_kernel


def tune():
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = (blockIdx.x * block_size_x) + threadIdx.x;
        if ( i < n ) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 10000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [32 * i for i in range(1, 33)]

    results, env = tune_kernel("vector_add", kernel_string, size, args, tune_params, parallel_runner=4)

    return results


if __name__ == "__main__":
    tune()
