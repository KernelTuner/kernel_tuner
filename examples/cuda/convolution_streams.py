#!/usr/bin/env python
import json
import numpy
import kernel_tuner
from collections import OrderedDict
import logging

import pycuda.driver as drv

def allocate(n, dtype=numpy.float32):
    """ allocate context-portable pinned host memory """
    return drv.pagelocked_empty(int(n), dtype, order='C', mem_flags=drv.host_alloc_flags.PORTABLE)

def tune():

    #setup problem dimensions
    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    input_size = (problem_size[0]+16) * (problem_size[0]+16)

    #create input and output data using pinned memory
    output = allocate(size)
    numpy.copyto(output, numpy.zeros(size).astype(numpy.float32, order='C'))
    input = allocate(input_size)
    numpy.copyto(input, numpy.random.randn(input_size).astype(numpy.float32, order='C'))
    filter = allocate(17*17)
    numpy.copyto(filter, numpy.random.randn(17*17).astype(numpy.float32, order='C'))

    #kernel arguments
    args = [output, input, filter]

    #setup tuning parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16*i for i in range(1,17)]
    tune_params["block_size_y"] = [2**i for i in range(5)]
    tune_params["tile_size_x"] = [2**i for i in range(4)]
    tune_params["tile_size_y"] = [2**i for i in range(4)]
    tune_params["num_streams"] = [2**i for i in range(6)]

    #tell the Kernel Tuner how to compute grid dimensions
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y", "num_streams"]

    #filter kernels that use too much shared memory
    restrict = ["(block_size_x*tile_size_x+16)*(block_size_y*tile_size_y+16) < 12*1024"]

    #compute the answer using a naive kernel
    with open('convolution.cu', 'r') as f:
        kernel_string = f.read()
    params = { "block_size_x": 16, "block_size_y": 16 }
    results = kernel_tuner.run_kernel("convolution_naive", kernel_string,
        problem_size, args, params,
        grid_div_y=["block_size_y"], grid_div_x=["block_size_x"])

    #set non-output fields to None
    answer = [results[0], None, None]

    #start tuning the kernel
    result = kernel_tuner.tune_kernel("convolution_streams", ['convolution_streams.cu', 'convolution.cu'],
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x, restrictions=restrict, answer=answer, verbose=True, lang="C",
        compiler_options=["-arch=sm_52"])

    return result


if __name__ == "__main__":
    drv.init()
    context = drv.Device(0).make_context()
    try:
        results = tune()
    finally:
        context.pop()
    with open("convolution_streams.json", 'w') as fp:
        json.dump(results, fp)



