#!/usr/bin/env python
import numpy
import kernel_tuner

with open('matmul.cu', 'r') as f:
    kernel_string = f.read()

problem_size = (512, 512)
size = numpy.prod(problem_size)

A = numpy.random.randn(*problem_size).astype(numpy.float32)
B = numpy.random.randn(*problem_size).astype(numpy.float32)
C = numpy.zeros_like(A)

args = [C, A, B]
tune_params = dict()
tune_params["block_size_x"] = [16*2**i for i in range(3)]
tune_params["block_size_y"] = [2**i for i in range(6)]

tune_params["tile_size_x"] = [2**i for i in range(4)]
tune_params["tile_size_y"] = [2**i for i in range(4)]

grid_div_x = ["block_size_x", "tile_size_x"]
grid_div_y = ["block_size_y", "tile_size_y"]

restrict = ["block_size_x==block_size_y*tile_size_y"]

kernel_tuner.tune_kernel("matmul_kernel", kernel_string,
    problem_size, args, tune_params,
    grid_div_y=grid_div_y, grid_div_x=grid_div_x,
    restrictions=restrict, verbose=True)

