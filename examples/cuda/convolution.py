#!/usr/bin/env python
import numpy
import kernel_tuner
from collections import OrderedDict

def tune():
    with open('convolution.cu', 'r') as f:
        kernel_string = f.read()

    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    input_size = (problem_size[0]+16) * (problem_size[0]+16)

    output = numpy.zeros(size).astype(numpy.float32)
    input = numpy.random.randn(input_size).astype(numpy.float32)
    filter = numpy.random.randn(17*17).astype(numpy.float32)

    cmem_args= {'d_filter': filter }

    args = [output, input, filter]
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    return kernel_tuner.tune_kernel("convolution_kernel", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x, verbose=True, cmem_args=cmem_args)

if __name__ == "__main__":
    tune()
