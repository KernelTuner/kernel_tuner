#!/usr/bin/env python

from collections import OrderedDict
import numpy
from kernel_tuner import tune_kernel, run_kernel


def tune_zeromean():
    with open('zeromeanfilter.cu', 'r') as f:
        kernel_string = f.read()

    height = numpy.int32(4391)
    width = numpy.int32(3539)
    image = numpy.random.randn(height*width).astype(numpy.float32)

    tune_vertical(kernel_string, image, height, width)
    tune_horizontal(kernel_string, image, height, width)


def tune_vertical(kernel_string, image, height, width):
    args = [height, width, image]

    #only one row of thread-blocks is to be created
    problem_size = (width, 1)
    grid_div_x = ["block_size_x"]
    grid_div_y = []

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    return tune_kernel("computeMeanVertically", kernel_string, problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x)


def tune_horizontal(kernel_string, image, height, width):
    args = [height, width, image]

    #use only one column of thread blocks
    problem_size = (1, height)
    grid_div_x = []
    grid_div_y = ["block_size_y"]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    return tune_kernel("computeMeanHorizontally", kernel_string, problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x)



if __name__ == "__main__":
    tune_zeromean()


