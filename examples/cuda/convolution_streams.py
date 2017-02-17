#!/usr/bin/env python
import numpy
import kernel_tuner
from collections import OrderedDict

def tune():

    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    input_size = (problem_size[0]+16) * (problem_size[0]+16)

    output = numpy.zeros(size).astype(numpy.float32, order='C')
    input = numpy.random.randn(input_size).astype(numpy.float32, order='C')
    filter = numpy.random.randn(17*17).astype(numpy.float32)

    args = [output, input, filter]
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [16*i for i in range(1,9)]
    tune_params["block_size_y"] = [2**i for i in range(6)]

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    tune_params["num_streams"] = [2**i for i in range(6)]

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
    return kernel_tuner.tune_kernel("convolution_streams", ['convolution_streams.cu', 'convolution.cu'],
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x, restrictions=restrict, answer=answer, verbose=True, lang="C", compiler_options=["-arch=sm_52"])


if __name__ == "__main__":
    tune()
