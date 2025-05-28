#!/usr/bin/env python
import numpy
import kernel_tuner
from collections import OrderedDict

def tune():

    with open('stencil.cl', 'r') as f:
        kernel_string = f.read()

    problem_size = (4096, 2048)
    size = numpy.prod(problem_size)

    x_old = numpy.random.randn(size).astype(numpy.float32)
    x_new = numpy.copy(x_old)
    args = [x_new, x_old]

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    grid_div_x = ["block_size_x"]
    grid_div_y = ["block_size_y"]

    return kernel_tuner.tune_kernel("stencil_kernel", kernel_string, problem_size,
        args, tune_params, grid_div_x=grid_div_x, grid_div_y=grid_div_y,
        verbose = True)


if __name__ == "__main__":
    tune()
