#!/usr/bin/env python
"""This is the minimal example of using a code generator with the Kernel Tuner"""

import json
import numpy
from kernel_tuner import tune_kernel

def my_fancy_generator(params):
    kernel_string = """
    __global__ void vector_add(float *c, float *a, float *b, int n) {
        int i = blockIdx.x * %s + threadIdx.x;
        if (i<n) {
            c[i] = a[i] + b[i];
        }
    }
    """ % params["block_size_x"]
    return kernel_string

def tune():

    size = 10000000

    a = numpy.random.randn(size).astype(numpy.float32)
    b = numpy.random.randn(size).astype(numpy.float32)
    c = numpy.zeros_like(b)
    n = numpy.int32(size)

    args = [c, a, b, n]

    tune_params = dict()
    tune_params["block_size_x"] = [128+64*i for i in range(15)]

    result = tune_kernel("vector_add", my_fancy_generator, size, args,
        tune_params, lang="CUDA")

    with open("vector_add.json", 'w') as fp:
        json.dump(result, fp)

    return result


if __name__ == "__main__":
    tune()
