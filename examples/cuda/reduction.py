#!/usr/bin/env python
import numpy
from kernel_tuner import tune_kernel

with open('reduction.cu', 'r') as f:
    kernel_string = f.read()

size = 80000000
blocks = 48
problem_size = (blocks, 1)

x = numpy.random.randn(size).astype(numpy.float32) + 1.0
sum = numpy.zeros(blocks).astype(numpy.int32)
n = numpy.int32(size)

args = [sum, x, n]

tune_params = dict()

tune_params["block_size_x"] = [2**i for i in range(4,11)]
tune_params["use_shuffle"] = [0, 1]
tune_params["vector"] = [2**i for i in range(3)]

tune_kernel("sum_floats", kernel_string,
    problem_size, args, tune_params,
    grid_div_x=[], verbose=True)





