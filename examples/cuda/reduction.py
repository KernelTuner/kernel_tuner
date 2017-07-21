#!/usr/bin/env python
import numpy
from kernel_tuner import tune_kernel
from kernel_tuner.util import get_config_string
from collections import OrderedDict
import json

def tune():
    with open('reduction.cu', 'r') as f:
        kernel_string = f.read()

    tune_params = OrderedDict()
    tune_params["block_size_x"] = [2**i for i in range(5,11)]
    tune_params["use_shuffle"] = [0, 1]
    tune_params["vector"] = [2**i for i in range(3)]
    tune_params["num_blocks"] = [2**i for i in range(5,11)]

    problem_size = "num_blocks"
    size = 80000000
    max_blocks = max(tune_params["num_blocks"])

    x = numpy.random.rand(size).astype(numpy.float32)
    sum_x = numpy.zeros(max_blocks).astype(numpy.float32)
    n = numpy.int32(size)

    args = [sum_x, x, n]

    #prepare output verification with custom function
    reference = [numpy.sum(x), None, None]
    def verify_partial_reduce(cpu_result, gpu_result, atol=None):
        return numpy.isclose(cpu_result, numpy.sum(gpu_result), atol=atol)

    #tune the first kernel
    first_kernel, _ = tune_kernel("sum_floats", kernel_string, problem_size,
        args, tune_params, grid_div_x=[], verbose=True, answer=reference, verify=verify_partial_reduce)

    #tune the second kernel for different input sizes
    #depending on the number of blocks used in the first kernel

    #store the parameter list used in the first kernel
    num_blocks = tune_params["num_blocks"]
    #fix num_blocks parameter to only 1 for the second kernel
    tune_params["num_blocks"] = [1]
    second_kernel = dict()
    for nblocks in num_blocks:
        #change the input size to nblocks
        args = [sum_x, x, numpy.int32(nblocks)]
        #tune the second kernel with n=nblocks
        result, _ = tune_kernel("sum_floats", kernel_string, problem_size,
        args, tune_params, grid_div_x=[], verbose=True)
        with open("reduce-kernel2-" + str(nblocks) + ".json", 'w') as fp:
            json.dump(result, fp)
        #only keep the best performing config
        second_kernel[nblocks] = min(result, key=lambda x:x['time'])

    #combine the results from the first kernel with best
    #second kernel that uses the same num_blocks
    for i, instance in enumerate(first_kernel):
        first_kernel[i]["total"] = instance["time"] + second_kernel[instance["num_blocks"]]["time"]

    best_config = min(first_kernel, key=lambda x:x['total'])

    print("Best performing config: \n" + get_config_string(best_config))
    print("uses the following config for the secondary kernel:")
    print(get_config_string(second_kernel[best_config["num_blocks"]]))

    with open("reduce.json", 'w') as fp:
        json.dump(first_kernel, fp)

    return first_kernel, second_kernel

if __name__ == "__main__":
    tune()


