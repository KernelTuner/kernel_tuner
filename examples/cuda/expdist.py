#!/usr/bin/env python
from collections import OrderedDict
import json
import numpy

from kernel_tuner import tune_kernel
from kernel_tuner import util

def tune_expdist():

    #setup tuning parameters
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(5,10)]
    tune_params["block_size_y"] = [2**i for i in range(6)]
    tune_params["tile_size_x"] = [2**i for i in range(4)]
    tune_params["tile_size_y"] = [2**i for i in range(4)]
    tune_params["use_shared_mem"] = [0, 1]

    #setup test input
    alloc_size = 3000
    size = numpy.int32(2000)
    max_blocks = numpy.int32( numpy.ceil(size / float(numpy.amin(tune_params["block_size_x"]))) *
                              numpy.ceil(size / float(numpy.amin(tune_params["block_size_y"]))) )
    ndim = numpy.int32(2)
    A = numpy.random.randn(alloc_size*ndim).astype(numpy.float64)
    B = A+0.00001*numpy.random.randn(alloc_size*ndim).astype(numpy.float64)
    scale_A = numpy.absolute(0.01*numpy.random.randn(alloc_size).astype(numpy.float64))
    scale_B = numpy.absolute(0.01*numpy.random.randn(alloc_size).astype(numpy.float64))
    cost = numpy.zeros((max_blocks)).astype(numpy.float64)

    #setup kernel
    with open('expdist.cu', 'r') as f:
        kernel_string = f.read()
    arguments = [A, B, size, size, scale_A, scale_B, cost]
    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    #tune using the Noodles runner for parallel tuning using 8 threads
    kernel1 = tune_kernel("ExpDist", kernel_string, (size, size), arguments, tune_params,
                          grid_div_x=grid_div_x, grid_div_y=grid_div_y, verbose=True)

    #dump the tuning results to a json file
    with open("expdist.json", 'w') as fp:
        json.dump(kernel1, fp)

    #get the number of blocks used by the best configuration in the first kernel
    best_config1 = util.get_best_config(kernel1[0], 'time')
    nblocks = numpy.int32( numpy.ceil(size / float(best_config1["block_size_x"]*best_config1["tile_size_x"])) *
                           numpy.ceil(size / float(best_config1["block_size_y"]*best_config1["tile_size_y"])) )

    #tunable parameters for the second kernel
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [32*i for i in range(1,33)]

    #tune the second kernel
    arguments = [numpy.zeros(1).astype(numpy.float64), cost, size, size, nblocks]
    kernel2 = tune_kernel("reduce_cross_term", kernel_string, 1, arguments, tune_params,
                grid_div_x=[], verbose=True)

    best_config2 = util.get_best_config(kernel2[0], 'time')
    print("best GPU configuration, total time=", best_config1['time'] + best_config2['time'])
    print(best_config1)
    print(best_config2)

if __name__ == "__main__":
    tune_expdist()
