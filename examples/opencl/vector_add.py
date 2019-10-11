#!/usr/bin/env python
import json
import numpy
from kernel_tuner import tune_kernel

kernel_string = """
__kernel void vector_add(__global float *c, __global const float *a, __global const float *b, int n) {
    int i = get_global_id(0);
    if (i<n) {
        c[i] = a[i] + b[i];
    }
}
"""

size = 10000000

a = numpy.random.rand(size).astype(numpy.float32)
b = numpy.random.rand(size).astype(numpy.float32)
c = numpy.zeros_like(a)
n = numpy.int32(size)

args = [c, a, b, n]

tune_params = dict()
tune_params["block_size_x"] = [128+64*i for i in range(15)]

result, env = tune_kernel("vector_add", kernel_string, size, args, tune_params)

with open("vector_add.json", 'w') as fp:
    json.dump(result, fp)

