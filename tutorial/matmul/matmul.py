#!/usr/bin/env python
import numpy
import kernel_tuner

problem_size = (4096, 4096)
size = numpy.prod(problem_size)

A = numpy.random.randn(size).astype(numpy.float32)
B = numpy.random.randn(size).astype(numpy.float32)
C = numpy.zeros_like(A)

args = [C, A, B]

params = {"block_size_x": 32, "block_size_y": 8, "tile_size_x": 4, "tile_size_y": 4}

grid_div_x = ["block_size_x", "tile_size_x"]
grid_div_y = ["block_size_y", "tile_size_y"]

results = kernel_tuner.run_kernel("matmul_kernel", "../examples/cuda/matmul.cu",
                                  problem_size, args, params,
                                  grid_div_x=grid_div_x, grid_div_y=grid_div_y)
