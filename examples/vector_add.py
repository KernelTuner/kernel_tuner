#!/usr/bin/env python
"""This is the minimal example from the README"""

import numpy
from kernel_tuner import tune_kernel

kernel_string = """
__global__ void vector_add(float *c, float *a, float *b, int n) {
    int i = blockIdx.x * block_size_x + threadIdx.x;
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""

size = 10000000
problem_size = (size, 1)

a = numpy.random.randn(size).astype(numpy.float32)
b = numpy.random.randn(size).astype(numpy.float32)
c = numpy.zeros_like(b)

args = [c, a, b]

tune_params = dict()
tune_params["block_size_x"] = [128+64*i for i in range(15)]

tune_kernel("vector_add", kernel_string, problem_size, args, tune_params)
