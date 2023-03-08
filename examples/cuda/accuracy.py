#!/usr/bin/env python

import numpy
from pprint import pprint
from kernel_tuner import tune_kernel
from kernel_tuner.accuracy import TunablePrecision

def tune():
    kernel_string = """
    #include <cuda_fp16.h>
    using half = __half;

    __global__ void vector_add(int n, float_type* left, float_type* right, float_type* output) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < n) {
            output[i] = left[i] + right[i];
        }
    }
    """

    size = 10000000

    n = numpy.int32(size)
    a = numpy.random.randn(size).astype(numpy.float64)
    b = numpy.random.randn(size).astype(numpy.float64)
    c = numpy.zeros_like(b)

    args = [
        n,
        TunablePrecision("float_type", a),
        TunablePrecision("float_type", b),
        TunablePrecision("float_type", c),
    ]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]
    tune_params["float_type"] = ["float", "double", "half"]

    results, env = tune_kernel("vector_add", kernel_string, size, args, tune_params)

    pprint(results)


if __name__ == "__main__":
    tune()
