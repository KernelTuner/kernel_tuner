#!/usr/bin/env python
"""This is the minimal example from the README"""

import json

import numpy
import cupy as cp
from kernel_tuner import tune_kernel

def tune():

    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * block_size_x + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """

    size = 10000000

    a = cp.random.randn(size).astype(cp.float32)
    b = cp.random.randn(size).astype(cp.float32)
    c = cp.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    answer = [a+b, None, None, None]

    result = tune_kernel("vector_add", kernel_string, size, args, tune_params, answer=answer, verbose=True, lang="Cupy")

    with open("vector_add.json", 'w') as fp:
        json.dump(result, fp)

    return result


if __name__ == "__main__":
    tune()
