#!/usr/bin/env python
import numpy
import logging
from kernel_tuner import tune_kernel
from collections import OrderedDict

def tune():
    with open('convolution.cu', 'r') as f:
        kernel_string = f.read()

    #setup tunable parameters
    tune_params = OrderedDict()
    tune_params["filter_height"] = [i for i in range(3,11,2)]
    tune_params["filter_width"] = [i for i in range(3,11,2)]
    tune_params["block_size_x"] = [16*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    #setup input and output dimensions
    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    largest_fh = max(tune_params["filter_height"])
    largest_fw = max(tune_params["filter_width"])
    input_size = ((problem_size[0]+largest_fw-1) * (problem_size[1]+largest_fh-1))

    #create input data
    output_image = numpy.zeros(size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(largest_fh * largest_fw).astype(numpy.float32)

    #setup kernel arguments
    cmem_args = {'d_filter': filter_weights}
    args = [output_image, input_image, filter_weights]

    #tell the Kernel Tuner how to compute grid dimensions
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    #start tuning separable convolution (row)
    tune_params["filter_height"] = [1]
    results_row = tune_kernel("convolution_kernel", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x, cmem_args=cmem_args, verbose=True, log=logging.DEBUG)

    #start tuning separable convolution (col)
    tune_params["filter_height"] = tune_params["filter_width"][:]
    tune_params["filter_width"] = [1]
    results_col = tune_kernel("convolution_kernel", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x, cmem_args=cmem_args, verbose=True, log=logging.DEBUG)

    return results_row, results_col


if __name__ == "__main__":
    results_row, results_col = tune()
    import json
    with open("separable_convolution_row.json", 'w') as fp:
        json.dump(results_row, fp)
    with open("separable_convolution_col.json", 'w') as fp:
        json.dump(results_col, fp)

