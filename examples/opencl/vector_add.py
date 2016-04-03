#!/usr/bin/env python

import numpy
from kernel_tuner import tune_kernel

kernel_string = """
__kernel void vector_add(__global float *c, __global const float *a, __global const float *b) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

size = 10000000
problem_size = (size, 1)

a = numpy.random.rand(size).astype(numpy.float32)
b = numpy.random.rand(size).astype(numpy.float32)
c = numpy.zeros_like(a)

args = [c, a, b]

tune_params = dict()
tune_params["block_size_x"] = [128+64*i for i in range(15)]

tune_kernel("vector_add", kernel_string, problem_size, args, tune_params)

