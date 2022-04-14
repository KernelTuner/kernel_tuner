#!/usr/bin/env python
""" convolution with correctness checks

This example is mostly the same as the Convolution example. The only
difference is that a naive kernel is used to compute a reference
output. This reference output is used to check the correctness of
every kernel before it is benchmarked.

This is done using the run_kernel() function of the Kernel Tuner and
the `answer` option of the tune_kernel function.

The run_kernel function simply runs a kernel using much of the same
interface as tune_kernel, however, for each tuning_parameter you pass
a single value instead of a list of options. The run_kernel function
returns a list of arguments that contains the output of the kernel.

When calling tune_kernel you specify the `answer` as a list, which
is similar to the arguments list of the kernel. To separate input
and output arguments you insert a `None` value in the answer list
for all arguments that are actually inputs to the kernel. The
values in the answers list that are not None are used to verify
the correctness of every kernel in the parameter space before it is
benchmarked.
"""
import numpy
import kernel_tuner
from collections import OrderedDict

def tune():
    with open('convolution.cu', 'r') as f:
        kernel_string = f.read()

    filter_size = (17, 17)
    problem_size = (4096, 4096)
    size = numpy.prod(problem_size)
    border_size = (filter_size[0]//2*2, filter_size[1]//2*2)
    input_size = ((problem_size[0]+border_size[0]) * (problem_size[1]+border_size[1]))

    output = numpy.zeros(size).astype(numpy.float32)
    input = numpy.random.randn(input_size).astype(numpy.float32)

    filter = numpy.random.randn(filter_size[0]*filter_size[1]).astype(numpy.float32)
    cmem_args= {'d_filter': filter }

    args = [output, input, filter]
    tune_params = OrderedDict()
    tune_params["filter_width"] = [filter_size[0]]
    tune_params["filter_height"] = [filter_size[1]]

    #tune_params["block_size_x"] = [16*i for i in range(1,3)]
    tune_params["block_size_x"] = [16*i for i in range(1,9)]
    #tune_params["block_size_y"] = [2**i for i in range(1,5)]
    tune_params["block_size_y"] = [2**i for i in range(1,6)]

    tune_params["tile_size_x"] = [2**i for i in range(3)]
    tune_params["tile_size_y"] = [2**i for i in range(3)]

    tune_params["use_padding"] = [0,1]  #toggle the insertion of padding in shared memory
    tune_params["read_only"] = [0,1]    #toggle using the read-only cache

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    #compute the answer using a naive kernel
    params = { "block_size_x": 16, "block_size_y": 16}
    tune_params["filter_width"] = [filter_size[0]]
    tune_params["filter_height"] = [filter_size[1]]
    results = kernel_tuner.run_kernel("convolution_naive", kernel_string,
        problem_size, args, params,
        grid_div_y=["block_size_y"], grid_div_x=["block_size_x"], lang='cupy')

    #set non-output fields to None
    answer = [results[0], None, None]

    #start kernel tuning with correctness verification
    return kernel_tuner.tune_kernel("convolution_kernel", kernel_string,
        problem_size, args, tune_params,
        grid_div_y=grid_div_y, grid_div_x=grid_div_x, verbose=True, cmem_args=cmem_args, answer=answer, lang='cupy')


if __name__ == "__main__":
    import time
    s1 = time.time()*1000
    results = tune()

    e1 = time.time()*1000
    print("\n Actualy time used:", e1-s1)
    import json
    with open("convolution_RTX_2070.json", 'w') as fp:
        json.dump(results, fp)

