#!/usr/bin/env python
import numpy
import kernel_tuner

with open('zeromeantotalfilter.cu', 'r') as f:
    kernel_string = f.read()

kernel_name = "computeMeanVertically"

height = numpy.int32(4391)
width = numpy.int32(3539)

#only one row of thread-blocks is to be created
problem_size = (width, 1)

image = numpy.random.randn(height*width).astype(numpy.float32)

args = [height, width, image]

tune_params = dict()
tune_params["block_size_x"] = [32*i for i in range(1,9)]
tune_params["block_size_y"] = [2**i for i in range(6)]

grid_div_x = ["block_size_x"]
grid_div_y = None

kernel_tuner.tune_kernel(kernel_name, kernel_string,
    problem_size, args, tune_params,
    grid_div_y=grid_div_y, grid_div_x=grid_div_x)

