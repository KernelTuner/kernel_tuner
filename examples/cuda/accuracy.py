#!/usr/bin/env python

import numpy
from pprint import pprint
from kernel_tuner import tune_kernel
from kernel_tuner.accuracy import TunablePrecision, AccuracyObserver


def tune():
    kernel_string = """
    #include <cuda_fp16.h>
    using half = __half;

    template <typename T>
    __global__ void vector_add(int n, const T* left, const T* right, T* output) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;

        if (i < n) {
            output[i] = left[i] + right[i];
        }
    }
    """

    size = 100000000

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

    answer = [None, None, None, a + b]

    tune_params = dict()
    tune_params["block_size_x"] = [32, 64, 128, 256, 512, 1024]
    tune_params["float_type"] = ["float", "double", "half"]

    observers = [
        AccuracyObserver("RMSE", "error_rmse"),
        AccuracyObserver("MRE", "error_relative"),
    ]

    results, env = tune_kernel(
        "vector_add<float_type>",
        kernel_string,
        size,
        args,
        tune_params,
        answer=answer,
        observers=observers,
        lang="CUDA",
    )

    pprint(results)


if __name__ == "__main__":
    tune()
